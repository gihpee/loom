"""Кого подвинуть, чтобы освободить узлы под аренду.

Главная механика платформы: пока прямого клиента нет, мощность занимают
собственные сервисы; пришёл клиент — они уступают. И главный источник тихих
поломок, потому что уступают тут живые работающие модели.

Четыре правила, и каждое написано после вопроса «а что будет, если наоборот».

**Сначала план, потом действие.** Считаем целиком: хватит ли вообще, кого
именно снять, какие узлы освободятся. Только если хватает — снимаем. Иначе
получилась бы худшая из возможных развязок: модели сняты, аренда не встала,
никто не работает.

**Отказ, а не частичное вытеснение.** Не хватает даже со снятием всего
доступного — отвечаем «столько нет» и не трогаем ничего.

**Защищённое не трогаем никогда.** Модель, на которой висит витрина, не должна
падать оттого, что кто-то арендовал кластер. Список решает администратор.

**Порядок вытеснения объясним.** Сначала те, что освобождают больше узлов за
одно снятие: меньше снятых моделей при том же результате. При равенстве —
позже развёрнутые: у давно стоящей больше шансов, что на неё кто-то полагается.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set


@dataclass(frozen=True)
class Standing:
    """Развёрнутая группа глазами вытеснения."""

    group_id: str
    label: str
    nodes: Sequence[str]
    submitted_at: float
    protected: bool = False


@dataclass
class Plan:
    """Что придётся сделать, чтобы дать аренде нужные узлы."""

    #: Узлы, которые получит аренда. Пусто, когда плана нет.
    nodes: List[str] = field(default_factory=list)
    #: Кого снять, по порядку.
    evict: List[Standing] = field(default_factory=list)
    #: Почему не вышло. Пусто — вышло.
    refusal: str = ""

    @property
    def possible(self) -> bool:
        return not self.refusal

    def explain(self) -> str:
        """Человеческое объяснение — уходит клиенту и в журнал."""
        if self.refusal:
            return self.refusal
        if not self.evict:
            return f"{len(self.nodes)} узлов взяты из свободных"
        names = ", ".join(f"{s.label or s.group_id}" for s in self.evict)
        return (f"{len(self.nodes)} узлов; ради этого сняты: {names}")


def plan(*, need: int, free: Sequence[str], standing: Sequence[Standing]) -> Plan:
    """Как дать аренде `need` узлов.

    `free` — узлы, никем не занятые. `standing` — что стоит сейчас и теоретически
    может уступить.
    """
    if need <= 0:
        return Plan(refusal="запрошено ноль узлов")

    free = list(dict.fromkeys(free))          # порядок сохраняем, дубли убираем
    if len(free) >= need:
        return Plan(nodes=free[:need])

    movable = [s for s in standing if not s.protected and s.nodes]
    taken: List[str] = list(free)
    evict: List[Standing] = []
    seen: Set[str] = set(free)

    # Больше узлов за одно снятие — раньше; при равенстве позже развёрнутые.
    # Оба ключа со знаком минус, чтобы сортировка шла по убыванию и оставалась
    # устойчивой: одинаковые входы обязаны давать одинаковый план, иначе
    # объяснить оператору, почему сняли именно это, будет нечем.
    for group in sorted(movable, key=lambda s: (-len(set(s.nodes) - seen),
                                                -s.submitted_at, s.group_id)):
        if len(taken) >= need:
            break
        fresh = [node for node in group.nodes if node not in seen]
        if not fresh:
            # Все её узлы уже посчитаны — снимать её незачем.
            continue
        evict.append(group)
        seen.update(fresh)
        taken.extend(fresh)

    if len(taken) < need:
        held = sum(1 for s in standing if s.protected)
        refusal = (f"нужно {need} узлов, свободно {len(free)}, "
                   f"освободить снятием можно ещё {len(taken) - len(free)}")
        if held:
            refusal += f"; {held} групп защищены от снятия"
        return Plan(refusal=refusal)

    return Plan(nodes=taken[:need], evict=evict)


def standing_from(groups: Dict[str, object], *, protected: Set[str],
                  resource_of) -> List[Standing]:
    """Перевести записи групп в то, чем оперирует план.

    `resource_of` отвечает, чем является группа: уступать могут только те, что
    держит сама платформа, а не чужая аренда. Арендованный кластер снимать
    ради другой аренды нельзя — за него уже платят.
    """
    from looma.usage.ledger import INFERENCE

    made = []
    for group_id, record in groups.items():
        if resource_of(group_id) != INFERENCE:
            continue
        made.append(Standing(
            group_id=group_id,
            label=getattr(record, "label", "") or "",
            nodes=sorted(set(getattr(record, "nodes", {}).values())),
            submitted_at=float(getattr(record, "submitted_at", 0.0)),
            protected=group_id in protected or
                      (getattr(record, "label", "") or "") in protected,
        ))
    return made
