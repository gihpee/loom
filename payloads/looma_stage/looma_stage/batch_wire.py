"""Батч на проводе.

Состав батча выбирает первая стадия. Остальные не выбирают ничего: они
получают список и обязаны прогнать ровно его, в том же порядке. Этот модуль —
про то, как этот список переезжает и как замечается, что он всё-таки разъехался.

Почему за этим надо следить отдельно. В склеенном батче тензор один на всех:
токены последовательностей лежат подряд, и границы между ними заданы только
списком участников. Ошибись стадия в составе или в порядке — тензор всё равно
разложится, просто не по тем границам. Модель посчитает, KV-кэш заполнится,
ответ придёт. Он будет бессмысленным, и ни одного исключения по дороге не
возникнет.

Поэтому здесь две проверки, и обе нарочно грубые: длины сходятся или нет.
Токенов в пришедшем тензоре ровно столько, сколько обещали участники, и
участников ровно столько, сколько строк вернул движок. Сойтись случайно эти
числа могут, но не тогда, когда состав действительно разъехался.
"""

from __future__ import annotations

from typing import Any, Dict, List

from looma_stage.scheduler import Sequence


class BatchMismatch(RuntimeError):
    """Состав батча на этой стадии не тот, что на предыдущей."""


def widths(sequences: List[Sequence], *, first_step: bool) -> List[int]:
    """Сколько токенов каждый участник кладёт в этот шаг.

    На prefill это весь промпт, на decode — ровно один токен: тот, что выдали
    в прошлом шаге. Отсюда же берутся границы нарезки общего тензора.
    """
    if first_step:
        return [len(sequence.prompt_ids) for sequence in sequences]
    return [1] * len(sequences)


def pack(sequences: List[Sequence], *, first_step: bool) -> List[Dict[str, Any]]:
    """Участники батча в том виде, в котором они поедут.

    Промпт едет целиком, а не одной длиной: vLLM строит запрос по токенам даже
    на тех стадиях, где эмбеддингов нет и сами значения не читаются. Платим за
    это один раз, на prefill — дальше едут только выданные токены.
    """
    members = []
    for sequence in sequences:
        members.append({
            "request_id": sequence.request_id,
            "prompt_ids": list(sequence.prompt_ids) if first_step else [],
            "prompt_len": len(sequence.prompt_ids),
            "output_ids": list(sequence.output_ids),
            "sampling": {
                "temperature": sequence.temperature,
                "top_p": sequence.top_p,
                "seed": sequence.seed,
            },
            "max_tokens": sequence.max_tokens,
        })
    return members


def unpack(members: List[Dict[str, Any]]) -> List[Sequence]:
    """Обратно — в том же порядке, в каком приехали.

    Порядок не сортируется и не нормализуется никак: он и есть соответствие
    между строкой логитов и запросом.
    """
    if not members:
        raise BatchMismatch("в сообщении нет ни одного участника батча")
    restored = []
    for member in members:
        request_id = member.get("request_id")
        if not request_id:
            raise BatchMismatch(f"участник батча без имени запроса: {member!r}")
        prompt_ids = list(member.get("prompt_ids") or [])
        if not prompt_ids:
            # Неголовная стадия на decode промпта не получает: значения ей не
            # нужны, а длина нужна — по ней vLLM считает позиции.
            prompt_ids = [0] * int(member.get("prompt_len") or 0)
        sampling = member.get("sampling") or {}
        restored.append(Sequence(
            request_id=request_id,
            prompt_ids=prompt_ids,
            output_ids=list(member.get("output_ids") or []),
            temperature=float(sampling.get("temperature") or 0.0),
            top_p=float(sampling.get("top_p") if sampling.get("top_p") is not None else 1.0),
            seed=sampling.get("seed"),
            max_tokens=int(member.get("max_tokens") or 128),
        ))
    return restored


def check_tokens(sequences: List[Sequence], rows: int, *, first_step: bool) -> None:
    """Столько ли токенов приехало, сколько обещали участники.

    Это единственный дешёвый способ поймать разъехавшийся состав до того, как
    он превратится в связную чушь: границы внутри тензора ничем не помечены,
    но их сумма — помечена.
    """
    expected = sum(widths(sequences, first_step=first_step))
    if rows != expected:
        raise BatchMismatch(
            f"пришло {rows} токенов на {len(sequences)} последовательностей, "
            f"а по составу батча их должно быть {expected}; состав разъехался "
            "между стадиями, и дальше считать нечего")


def check_rows(sequences: List[Sequence], rows: int) -> None:
    """Столько ли строк логитов, сколько последовательностей."""
    if rows != len(sequences):
        raise BatchMismatch(
            f"{rows} строк логитов на {len(sequences)} последовательностей; "
            "какому запросу принадлежит строка, сказать уже нельзя")


# ---------------------------------------------------------------- тензоры
def pack_tensors(torch, tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Карта тензоров в то, что уедет по проводу.

    Карта, а не один тензор, потому что vLLM передаёт между стадиями два:
    `hidden_states` и `residual` — сложение с остаточной связью разорвано ровно
    по границе среза слоёв, и потерянный `residual` не даёт ни ошибки, ни
    расхождения форм, только другой ответ. Собственный исполнитель отдаёт один
    тензор, и он едет такой же картой из одного имени: два формата на проводе
    означали бы, что стадия должна знать, каким движком считает соседняя.
    """
    from looma_stage import wire

    packed = {}
    for name, tensor in tensors.items():
        data, shape, dtype = wire.to_wire(torch, tensor)
        packed[name] = {"tensor_b64": _b64(data), "shape": shape, "dtype": dtype}
    return packed


def unpack_tensors(torch, payload: Dict[str, Any], *, device=None) -> Dict[str, Any]:
    """Обратно. `device` задаётся, когда тензоры нужны на карте."""
    import base64

    from looma_stage import wire

    if not payload:
        raise BatchMismatch("в сообщении нет тензоров, а стадия не первая")
    restored = {}
    for name, piece in payload.items():
        tensor = wire.from_wire(torch, base64.b64decode(piece["tensor_b64"]),
                                piece["shape"], piece["dtype"])
        restored[name] = tensor.to(device) if device is not None else tensor
    return restored


def token_rows(tensors: Dict[str, Any]) -> int:
    """Сколько токенов в карте тензоров.

    Токены лежат по первой оси у плоской формы `[токены, ширина]` и по второй
    у `[1, токены, ширина]` — обе встречаются, потому что их дают разные
    движки. Ошибиться осью тут дороже, чем кажется: число всё равно получится,
    и сверка состава пройдёт по неправильному.
    """
    for tensor in tensors.values():
        shape = list(getattr(tensor, "shape", []) or [])
        if len(shape) >= 3:
            return int(shape[1])
        if len(shape) == 2:
            return int(shape[0])
    raise BatchMismatch(
        "в карте тензоров нет ничего, по чему можно сосчитать токены")


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode()
