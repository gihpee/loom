"""The orchestrator's HTTP face.

Three audiences and nothing else:

  clients    /v1/chat/completions — a model answers, and the machine answering
             it never opened a port
  operators  /admin/... — nodes, tasks, groups, join keys, agent rollout
  agents     /agent/release/... — the signed payload a node fetches

Every admin route is gated by one token; the release archive deliberately is
not, because it is signed and a node has no admin token to present.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse

from looma.accounts.store import SESSION_TTL, AccountError
from looma.api.auth import (
    ANONYMOUS,
    SESSION_COOKIE,
    Authenticator,
    refuse,
)
from looma.logging_config import get_logger
from looma.usage.ledger import COMPUTE, INFERENCE
from looma.orchestrator.agents import AgentError
from looma.orchestrator.models import (
    ModelError,
    describe,
    split_layers,
    stage_payload,
    stage_requirements,
    vllm_refusal,
)
from looma.orchestrator.connectivity import prefer_meshy, verdict
from looma.orchestrator.payloads import PayloadMissing, ray_payload
from looma.orchestrator.rendezvous import relay_addrs
from looma.orchestrator.releases import ReleaseError

logger = get_logger(__name__)


def _error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"message": message, "type": kind}})


# Причина закрытия WebSocket ограничена 123 байтами, и кириллица занимает по
# два на букву. Длинный текст не обрезается сам, а роняет закрытие — то есть
# пропадает ровно то объяснение, ради которого писался.
CLOSE_REASON_BYTES = 123


def _reason(text: str) -> str:
    """Причина закрытия, укладывающаяся в кадр."""
    raw = text.encode()
    if len(raw) <= CLOSE_REASON_BYTES:
        return text
    # Место под многоточие резервируется: само оно занимает три байта, и без
    # этого обрезка ровно по лимиту его же и переполняет.
    tail = "…".encode()
    return raw[:CLOSE_REASON_BYTES - len(tail)].decode(errors="ignore") + "…"


def _inputs_of(raw: dict) -> dict:
    """Файлы, которые едут вместе с задачей. Base64, потому что тело — JSON.

    Имя файла в ошибке: «неверный base64» ничего не стоит, когда файлов
    десяток, а испорчен один.
    """
    decoded: dict = {}
    for name, data in (raw.get("inputs") or {}).items():
        try:
            decoded[name] = base64.b64decode(data)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name!r}: {exc}") from None
    return decoded


def create_app(*, agents=None, releases=None, keystore=None, config=None,
               public_address=None, accounts=None, ledger=None,
               deployments=None) -> FastAPI:
    app = FastAPI(title="Looma", version="0.2.0")
    auth = Authenticator(accounts=accounts,
                         emergency_token=getattr(config, "admin_token", "") or "")

    # Проверка правами — ОДНИМ слоем, а не в каждом маршруте. Двадцать пять
    # одинаковых охранников, расставленных руками, — это гарантированно один
    # забытый маршрут, и заметен он будет не тем, кто его забыл.
    #
    # Правило читается целиком отсюда: /admin — администратору, /api и /v1 —
    # любому, кто представился. Остальное открыто: по корню отдаётся лендинг.
    # Вход и выход — единственное, что обязано работать без представления:
    # именно за тем на них и приходят. Без этой строчки войти невозможно в
    # принципе, и обнаруживается это не раньше первой попытки.
    # Демо на лендинге отвечает без представления — в том и смысл: человек
    # должен убедиться в скорости до того, как заводить учётную запись.
    OPEN = {"/api/session", "/api/demo"}

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        needs_admin = path.startswith("/admin")
        if path in OPEN or not (
                needs_admin or path.startswith("/api/") or path.startswith("/v1/")):
            return await call_next(request)
        caller = await auth.identify(
            session=request.cookies.get(SESSION_COOKIE),
            authorization=request.headers.get("authorization"),
            admin_token=request.headers.get("x-looma-admin-token"))
        why = refuse(caller, admin=needs_admin)
        if why:
            # 401 против 403 — разные действия для клиента: первое означает
            # «войди», второе «войдено, но не тебе сюда».
            return JSONResponse(status_code=401 if not caller.known else 403,
                                content={"error": {"message": why}})
        request.state.caller = caller
        return await call_next(request)

    def whoami(request: Request):
        return getattr(request.state, "caller", ANONYMOUS)

    async def _start_billing(account_id, record, resource: str, *,
                             nodes: int, gpus: int, label: str = "") -> int:
        """Начать считать потребление группы.

        Молча ничего не делает без базы или без опознанного вызывающего:
        оркестратор без учётных записей всё ещё должен уметь разворачивать
        модели, просто счёт вести не на кого.
        """
        if ledger is None or account_id is None:
            return 0
        try:
            return await ledger.open_lease(
                account_id=account_id, resource=resource,
                group_id=record.group_id, nodes=nodes, gpus=gpus, label=label)
        except Exception:
            # Счёт важен, но не важнее развёрнутой группы: она уже работает, и
            # ронять ответ из-за журнала значит оставить клиента без ответа при
            # успешном действии.
            logger.exception("не удалось открыть аренду для %s", record.group_id)
            return 0

    async def _count_tokens(request: Request, model: str, answer: bytes) -> None:
        """Записать токены ответа.

        Считаются те, что назвала сама стадия: пересчитывать их здесь значило
        бы держать второй токенизатор и расходиться с первым на краях.

        Потоковый ответ пока не учитывается — счётчики приходят в последнем
        куске, а он уезжает клиенту, минуя это место. Молчать об этом хуже, чем
        сказать: см. docs/BILLING.md.
        """
        who = whoami(request)
        if ledger is None or who.account_id is None:
            return
        try:
            usage = (json.loads(answer or b"{}") or {}).get("usage") or {}
            await ledger.record_tokens(
                account_id=who.account_id, model=model,
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0))
        except Exception:
            logger.exception("не удалось записать токены ответа")

    async def _plan_preemption(size: int, free: list):
        """Кого подвинуть. Считается целиком, ничего не трогая."""
        from looma.orchestrator.preemption import plan as make_plan
        from looma.orchestrator.preemption import standing_from

        protected = await deployments.protected_ids() if deployments else set()
        known = {d.group_id: d for d in (await deployments.list()
                                         if deployments else [])}

        def resource_of(group_id: str) -> str:
            # Уступает только то, что держит сама платформа. Арендованный
            # кластер снимать ради другой аренды нельзя — за него уже платят.
            return INFERENCE if group_id in known else ""

        return make_plan(need=size, free=free,
                         standing=standing_from(agents.groups, protected=protected,
                                                resource_of=resource_of))

    async def _evict(made) -> None:
        """Снять то, что назвал план. Со сливом, а не убийством: запрос,
        который сейчас отвечает, дописывает ответ."""
        for group in made.evict:
            try:
                agents.stop_group(group.group_id, reason="вытеснено арендой")
                await _stop_billing(group.group_id)
            except AgentError:
                logger.exception("не удалось снять %s", group.group_id)

    async def _restore(lease_id: int) -> None:
        """Вернуть снятое этой арендой.

        Вторая половина вытеснения. Без неё механика выглядит работающей ровно
        до конца первой аренды, а потом платформа тихо остаётся без инференса.
        """
        if deployments is None:
            return
        for gone in await deployments.evicted_by(lease_id):
            try:
                answer = await _deploy_model(gone.request,
                                             account_id=gone.account_id)
            except Exception:
                logger.exception("не удалось вернуть %s", gone.label or gone.group_id)
                continue
            if isinstance(answer, JSONResponse):
                logger.warning("вернуть %s пока некуда — узлы заняты",
                               gone.label or gone.group_id)
                continue
            await deployments.restored(gone.group_id)
            logger.info("вернули %s", gone.label or gone.group_id)

    async def _stop_billing(group_id: str) -> None:
        """Закрыть аренду и вернуть то, что она вытеснила.

        Возврат идёт здесь, а не отдельной кнопкой: аренда кончилась — базовая
        загрузка обязана вернуться сама, иначе платформа теряет инференс на
        каждой аренде, и виноватого не найти.
        """
        if ledger is None:
            return
        try:
            lease_id = await ledger.lease_for(group_id)
            await ledger.close_lease(group_id)
        except Exception:
            logger.exception("не удалось закрыть аренду %s", group_id)
            return
        if lease_id:
            await _restore(lease_id)

    async def _body(request: Request) -> dict:
        try:
            return await request.json()
        except Exception:
            return {}

    def need_agents():
        return _error(503, "this orchestrator is running without the agent gateway")

    # ------------------------------------------------------- вход и записи
    def need_accounts():
        return _error(503, "оркестратор поднят без базы: учётные записи "
                            "недоступны, работает только аварийный токен")

    @app.post("/api/session")
    async def sign_in(request: Request):
        """Вход по почте и паролю. Кладёт сессию в cookie."""
        if accounts is None:
            return need_accounts()
        body = await _body(request)
        account = await accounts.sign_in(email=body.get("email", ""),
                                         password=body.get("password", ""))
        if account is None:
            # Одна причина на оба случая: «нет такого адреса» и «неверный
            # пароль» вместе рассказывают, кто у нас зарегистрирован.
            return _error(401, "почта или пароль не подошли")
        token = await accounts.start_session(account.id)
        answer = JSONResponse(content=account.as_dict())
        answer.set_cookie(
            SESSION_COOKIE, token,
            httponly=True,      # чужой скрипт на странице не прочитает
            samesite="lax",     # и не отправит её с чужого сайта
            secure=request.url.scheme == "https",
            max_age=int(SESSION_TTL.total_seconds()),
            path="/")
        return answer

    @app.delete("/api/session")
    async def sign_out(request: Request):
        if accounts is not None:
            await accounts.end_session(request.cookies.get(SESSION_COOKIE) or "")
        answer = JSONResponse(content={"signed_out": True})
        answer.delete_cookie(SESSION_COOKIE, path="/")
        return answer

    @app.get("/api/me")
    async def me(request: Request):
        return whoami(request).as_dict()

    # ------------------------------------------------------------ ключи API
    @app.get("/api/keys")
    async def my_keys(request: Request):
        if accounts is None:
            return need_accounts()
        return {"keys": await accounts.keys_of(whoami(request).account_id)}

    @app.post("/api/keys")
    async def new_key(request: Request):
        """Создать ключ. Показывается один раз — в базе только хэш."""
        if accounts is None:
            return need_accounts()
        body = await _body(request)
        key, record = await accounts.issue_key(whoami(request).account_id,
                                               name=(body.get("name") or "").strip())
        return {**record, "key": key,
                "notice": "Сохраните ключ сейчас: он больше не будет показан"}

    @app.delete("/api/keys/{key_id}")
    async def drop_key(key_id: int, request: Request):
        if accounts is None:
            return need_accounts()
        if not await accounts.revoke_key(whoami(request).account_id, key_id):
            return _error(404, "такого ключа нет или он уже отозван")
        return {"revoked": key_id}

    # ------------------------------------------------------------ потребление
    def _period(request: Request):
        """Границы отчёта из строки запроса. Пусто — за всё время."""
        from datetime import datetime

        def when(name):
            raw = request.query_params.get(name)
            if not raw:
                return None
            try:
                return datetime.fromisoformat(raw)
            except ValueError:
                return None

        return when("since"), when("until")

    @app.get("/api/usage")
    async def my_usage(request: Request):
        """Что израсходовал я. Идущие аренды считаются по «сейчас»."""
        if ledger is None:
            return _error(503, "оркестратор поднят без базы: журнала нет")
        since, until = _period(request)
        return await ledger.report(account_id=whoami(request).account_id,
                                   since=since, until=until)

    @app.get("/admin/usage")
    async def all_usage(request: Request):
        """То же по всем. Без account_id — сводка, с ним — по одному."""
        if ledger is None:
            return _error(503, "оркестратор поднят без базы: журнала нет")
        since, until = _period(request)
        who = request.query_params.get("account_id")
        return await ledger.report(account_id=int(who) if who else None,
                                   since=since, until=until)

    @app.get("/admin/rates")
    async def read_rates():
        if ledger is None:
            return _error(503, "оркестратор поднят без базы: ставок нет")
        return {"rates": [r.as_dict() for r in await ledger.rates()]}

    @app.post("/admin/rates")
    async def write_rate(request: Request):
        """Ставка в КОПЕЙКАХ за GPU-час. Целыми: деньги в плавающей точке дают
        0.1 + 0.2 = 0.30000000000000004 в счёте."""
        if ledger is None:
            return _error(503, "оркестратор поднят без базы: ставок нет")
        body = await _body(request)
        try:
            rate = await ledger.set_rate(
                (body.get("resource") or "").strip(),
                int(body.get("per_hour") or 0),
                (body.get("currency") or "RUB").strip())
        except (ValueError, TypeError) as exc:
            return _error(400, str(exc))
        return rate.as_dict()

    # ---------------------------------------------------- защита от снятия
    @app.get("/admin/deployments")
    async def list_deployments():
        """Что развёрнуто и что из этого защищено от вытеснения."""
        if deployments is None:
            return _error(503, "оркестратор поднят без базы")
        return {"deployments": [d.as_dict() for d in await deployments.list()]}

    @app.post("/admin/deployments/{group_id}/protected")
    async def protect(group_id: str, request: Request):
        """Защитить от вытеснения или снять защиту.

        Модель, на которой висит витрина, не должна падать оттого, что кто-то
        арендовал кластер. Решает это администратор, а не арендатор.
        """
        if deployments is None:
            return _error(503, "оркестратор поднят без базы")
        body = await _body(request)
        want = bool(body.get("protected", True))
        if not await deployments.set_protected(group_id, want):
            return _error(404, f"нет развёртывания {group_id}")
        return {"group_id": group_id, "protected": want}

    # ------------------------------------------------------ записи (админ)
    @app.get("/admin/accounts")
    async def list_accounts():
        if accounts is None:
            return need_accounts()
        return {"accounts": [a.as_dict() for a in await accounts.list()]}

    @app.post("/admin/accounts")
    async def add_account(request: Request):
        if accounts is None:
            return need_accounts()
        body = await _body(request)
        try:
            made = await accounts.create(
                email=body.get("email", ""), password=body.get("password", ""),
                role=(body.get("role") or "client").strip(),
                display_name=(body.get("display_name") or "").strip())
        except AccountError as exc:
            return _error(400, str(exc))
        return made.as_dict()

    @app.post("/admin/accounts/{account_id}/password")
    async def reset_password(account_id: int, request: Request):
        if accounts is None:
            return need_accounts()
        body = await _body(request)
        try:
            await accounts.set_password(account_id, body.get("password", ""))
        except AccountError as exc:
            return _error(400, str(exc))
        return {"changed": account_id}

    @app.post("/admin/accounts/{account_id}/disabled")
    async def set_account_disabled(account_id: int, request: Request):
        """Отключить или вернуть. Отключение гасит все сессии и ключи разом."""
        if accounts is None:
            return need_accounts()
        body = await _body(request)
        await accounts.set_disabled(account_id, bool(body.get("disabled", True)))
        return {"account": account_id, "disabled": bool(body.get("disabled", True))}

    # ------------------------------------------------------------- clients
    @app.get("/v1/models")
    async def list_models():
        """What is up and answering, by the name a client would ask for."""
        if agents is None:
            return {"object": "list", "data": []}
        served = []
        for label in sorted({g.label for g in agents.groups.values() if g.label}):
            # Именно «отвечает», а не «процесс запущен»: список моделей — это
            # обещание клиенту, и оно не должно опережать загрузку весов.
            if await agents.serving(label) is not None:
                served.append({"id": label, "object": "model"})
        return {"object": "list", "data": served}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        """Отдать запрос группе, которая обслуживает эту модель.

        Оркестратор не смотрит в тело. Он знает, на каком узле ранг 0 и как до
        него дотянуться по соединению, которое узел открыл сам; что такое
        completion — дело стадии.
        """
        if agents is None:
            return need_agents()
        raw = await request.body()
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            return _error(400, "тело запроса не JSON")
        model = (body.get("model") or "").strip()
        if not model:
            return _error(400, "нужно поле 'model'")
        group = await agents.serving(model)
        if group is None:
            placed = agents.group_for(model)
            if placed is not None:
                return _error(503, f"{model} ещё поднимается: стадии загружают веса")
            return _error(404, f"{model!r} сейчас никто не обслуживает")

        head = group.tasks[0]
        if not body.get("stream"):
            try:
                status, headers, answer = await agents.request(
                    head, method="POST", path="/v1/chat/completions",
                    body=raw, headers={"Content-Type": "application/json"})
            except AgentError as exc:
                return _error(502, str(exc))
            await _count_tokens(request, model, answer)
            return Response(content=answer, status_code=status,
                            media_type=headers.get("Content-Type", "application/json"))

        async def pieces():
            """Токены наружу по мере появления.

            Ошибка после первого куска уже не может стать HTTP-статусом —
            заголовки ушли. Поэтому она уходит событием в тот же поток: клиент,
            который её проигнорирует, увидит оборванный ответ, а тот, кто
            читает, узнает причину.
            """
            try:
                async for piece in agents.request_stream(
                        head, method="POST", path="/v1/chat/completions",
                        body=raw, headers={"Content-Type": "application/json"}):
                    if not isinstance(piece, tuple):
                        yield piece
            except AgentError as exc:
                yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n".encode()
                yield b"data: [DONE]\n\n"

        return StreamingResponse(pieces(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # ---------------------------------------------------------------- демо
    # Открытый инференс на лендинге. Открытый — значит, за него платит парк, а
    # не клиент, поэтому ограничители тут не украшение, а условие того, что это
    # вообще можно включить: короткий запрос, короткий ответ, счётчик на адрес
    # и общий потолок. Без последнего одна вкладка с циклом займёт все карты.
    DEMO_PROMPT = 400          # символов
    DEMO_TOKENS = 220          # токенов в ответе
    DEMO_PER_HOUR = 20         # запросов с адреса
    DEMO_AT_ONCE = 4           # одновременных на всех
    demo_seen: dict[str, list[float]] = {}
    demo_now = 0

    def demo_who(request: Request) -> str:
        """Адрес обратившегося. За обратным прокси реальный адрес приходит
        заголовком: без этого все посетители выглядят одним клиентом, и общий
        счётчик закрывает демо для всех сразу."""
        sent = request.headers.get("x-forwarded-for", "")
        first = sent.split(",")[0].strip()
        return first or (request.client.host if request.client else "?")

    def demo_allow(who: str) -> str:
        """Пустая строка — можно; иначе причина отказа, как она уйдёт человеку."""
        now = time.time()
        seen = [t for t in demo_seen.get(who, []) if now - t < 3600]
        if len(seen) >= DEMO_PER_HOUR:
            return "на сегодня хватит: демо ограничено, ключ снимает предел"
        seen.append(now)
        demo_seen[who] = seen
        if len(demo_seen) > 4096:                 # не растим словарь без предела
            for key in [k for k, v in demo_seen.items() if not v or now - v[-1] > 3600]:
                demo_seen.pop(key, None)
        return ""

    async def demo_model() -> str:
        """Первая модель, которая действительно отвечает."""
        if agents is None:
            return ""
        for label in sorted({g.label for g in agents.groups.values() if g.label}):
            if await agents.serving(label) is not None:
                return label
        return ""

    @app.get("/api/demo")
    async def demo_ready():
        """Есть ли что показывать. Лендинг прячет блок, если сеть пуста: пустое
        поле ввода, которое ничего не отвечает, хуже отсутствующего блока."""
        label = await demo_model()
        nodes = 0
        if agents is not None and label:
            group = await agents.serving(label)
            nodes = len(group.tasks) if group else 0
        return {"model": label, "nodes": nodes, "limit": DEMO_TOKENS}

    @app.post("/api/demo")
    async def demo_ask(request: Request):
        nonlocal demo_now
        if agents is None:
            return need_agents()
        label = await demo_model()
        if not label:
            return _error(503, "сейчас ни одна модель не отвечает")

        try:
            body = json.loads(await request.body() or b"{}")
        except ValueError:
            return _error(400, "тело запроса не JSON")
        prompt = (body.get("prompt") or "").strip()[:DEMO_PROMPT]
        if not prompt:
            return _error(400, "пустой запрос")

        why = demo_allow(demo_who(request))
        if why:
            return _error(429, why)
        if demo_now >= DEMO_AT_ONCE:
            return _error(429, "демо сейчас занято, попробуйте через минуту")

        ask = json.dumps({
            "model": label,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": DEMO_TOKENS,
            "stream": True,
        }).encode()
        group = await agents.serving(label)
        head = group.tasks[0]

        async def pieces():
            nonlocal demo_now
            demo_now += 1
            try:
                async for piece in agents.request_stream(
                        head, method="POST", path="/v1/chat/completions",
                        body=ask, headers={"Content-Type": "application/json"}):
                    if not isinstance(piece, tuple):
                        yield piece
            except AgentError as exc:
                yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                # Именно finally: оборванное соединение — обычный исход, и без
                # этого счётчик занятости однажды упрётся в потолок навсегда.
                demo_now -= 1

        return StreamingResponse(pieces(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # --------------------------------------------------------------- nodes
    # HTML тут не раздаётся: панель — отдельный сервис (docker/web.Dockerfile),
    # который проксирует сюда /admin и /v1. Оркестратор отвечает только данными.

    @app.get("/admin/connect")
    async def admin_connect(x_looma_admin_token: str | None = Header(default=None)):
        """Everything needed to attach a machine: the address and one command."""
        return {
            "dial_address": getattr(public_address, "address", ""),
            "source": getattr(public_address, "source", "config"),
            "reachable_externally": getattr(public_address, "reachable_externally", None),
            "severity": getattr(public_address, "severity", "info"),
            "self_check": getattr(public_address, "self_check", None),
            "note": getattr(public_address, "note", None),
            "warning": getattr(public_address, "warning", None),
            "agent_image": "gihpee/looma-agent",
        }

    @app.get("/admin/agents")
    async def admin_agents(x_looma_admin_token: str | None = Header(default=None)):
        if agents is None:
            return need_agents()
        return {"nodes": agents.node_list()}

    # ---------------------------------------------------------------- keys
    @app.get("/admin/keys")
    async def admin_keys(x_looma_admin_token: str | None = Header(default=None)):
        if keystore is None:
            return {"keys": []}
        return {"keys": [k.public() for k in keystore.list()]}

    @app.post("/admin/keys")
    async def admin_issue_key(request: Request,
                              x_looma_admin_token: str | None = Header(default=None)):
        if keystore is None:
            return _error(503, "this orchestrator issues no keys")
        raw = await request.json()
        key = keystore.issue(label=(raw.get("label") or "").strip(),
                             max_nodes=int(raw.get("max_nodes") or 0))
        return {**key.public(), "key": key.encode(),
                "address": getattr(public_address, "address", ""),
                "agent_image": "gihpee/looma-agent"}

    @app.delete("/admin/keys/{key_id}")
    async def admin_revoke_key(key_id: str,
                               x_looma_admin_token: str | None = Header(default=None)):
        if keystore is None or not keystore.revoke(key_id):
            return _error(404, f"no key {key_id!r}")
        return {"revoked": key_id}

    # --------------------------------------------------------------- tasks
    @app.get("/admin/tasks")
    async def admin_tasks(x_looma_admin_token: str | None = Header(default=None)):
        if agents is None:
            return need_agents()
        ordered = sorted(agents.tasks.values(), key=lambda t: t.submitted_at, reverse=True)
        return {"tasks": [t.as_dict() for t in ordered]}

    @app.post("/admin/tasks")
    async def admin_submit_task(request: Request,
                                x_looma_admin_token: str | None = Header(default=None)):
        """Place one task and start it.

        Returns as soon as it is on its way. Provisioning an environment can
        take half an hour, so this must not wait for the task to start — follow
        it by its state instead. Input files come base64-encoded in `inputs`.
        """
        if agents is None:
            return need_agents()
        raw = await request.json()
        command = raw.get("command") or []
        if isinstance(command, str):
            command = command.split()
        try:
            inputs = _inputs_of(raw)
        except ValueError as exc:
            return _error(400, f"'inputs' must be base64 per file: {exc}")
        try:
            record = agents.submit(
                command=list(command),
                environment=raw.get("environment") or None,
                # Доля процессора названа явно: Ray пред-запускает воркер на
                # каждое «своё» ядро, и без этого два ранга на одной машине
                # заводят вдвое больше процессов, чем она стоит.
                resources=raw.get("resources") or {"cpus": raw.get("cpus") or 8},
                env=raw.get("env") or None,
                timeout_s=int(raw.get("timeout_s") or 3600),
                inputs=inputs,
                node_id=(raw.get("node_id") or "").strip(),
            )
        except AgentError as exc:
            # The reason is the useful part: "no node has 2 free GPUs" is
            # actionable, "unavailable" is not.
            return _error(409, str(exc))
        return record.as_dict()

    @app.get("/admin/tasks/{task_id}")
    async def admin_task(task_id: str,
                         x_looma_admin_token: str | None = Header(default=None)):
        if agents is None:
            return need_agents()
        record = agents.tasks.get(task_id)
        return record.as_dict() if record else _error(404, f"no task {task_id!r}")

    @app.get("/admin/tasks/{task_id}/logs")
    async def admin_task_logs(task_id: str, tail: int = 200,
                              x_looma_admin_token: str | None = Header(default=None)):
        if agents is None:
            return need_agents()
        try:
            return {"task_id": task_id, "text": await agents.logs(task_id, tail_lines=tail)}
        except AgentError as exc:
            return _error(409, str(exc))

    @app.get("/admin/tasks/{task_id}/results/{name:path}")
    async def admin_task_result(task_id: str, name: str,
                                x_looma_admin_token: str | None = Header(default=None)):
        if agents is None:
            return need_agents()
        try:
            payload = await agents.collect(task_id, name)
        except AgentError as exc:
            return _error(409, str(exc))
        return Response(content=payload, media_type="application/octet-stream",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{Path(name).name}"'})

    @app.post("/admin/tasks/{task_id}/stop")
    async def admin_stop_task(task_id: str,
                              x_looma_admin_token: str | None = Header(default=None)):
        if agents is None:
            return need_agents()
        try:
            return agents.stop(task_id, reason="stopped from the admin page").as_dict()
        except AgentError as exc:
            return _error(409, str(exc))

    @app.delete("/admin/tasks/{task_id}")
    async def admin_release_task(task_id: str,
                                 x_looma_admin_token: str | None = Header(default=None)):
        if agents is None:
            return need_agents()
        try:
            agents.release(task_id)
        except AgentError as exc:
            return _error(409, str(exc))
        return {"released": task_id}

    # -------------------------------------------------------------- groups
    @app.get("/admin/groups")
    async def admin_groups(x_looma_admin_token: str | None = Header(default=None)):
        if agents is None:
            return need_agents()
        ordered = sorted(agents.groups.values(), key=lambda g: g.submitted_at, reverse=True)
        # `finished` считает оркестратор: панели иначе пришлось бы сводить
        # список задач с составом группы и повторять это на каждом экране.
        return {"groups": [{**g.as_dict(), "finished": agents.group_finished(g)}
                           for g in ordered]}

    @app.post("/admin/groups")
    async def admin_submit_group(request: Request,
                                 x_looma_admin_token: str | None = Header(default=None)):
        """Place a job across several nodes — a model pipeline, usually.

        All-or-nothing: a pipeline missing a stage does not run slower, it does
        not run, so a group that cannot be placed whole is not placed at all.

        `inputs` едут каждому рангу — как и одиночной задаче, base64 на файл.
        Одна программа на всех, разное ей передаётся через `per_rank`: узел,
        впервые видящий эту работу, получает её код вместе с задачей, и никакой
        реестр пакетов посередине для этого не нужен.
        """
        if agents is None:
            return need_agents()
        raw = await request.json()
        try:
            inputs = _inputs_of(raw)
        except ValueError as exc:
            return _error(400, f"'inputs' must be base64 per file: {exc}")
        try:
            record = agents.submit_group(
                size=int(raw.get("size") or 1),
                command=list(raw.get("command") or []),
                environment=raw.get("environment") or None,
                resources=raw.get("resources") or None,
                env=raw.get("env") or None,
                timeout_s=int(raw.get("timeout_s") or 3600),
                serve_port=int(raw.get("serve_port") or 0),
                node_ids=list(raw.get("node_ids") or []) or None,
                per_rank=raw.get("per_rank") or None,
                label=(raw.get("label") or "").strip(),
                inputs=inputs,
            )
        except AgentError as exc:
            return _error(409, str(exc))
        return record.as_dict()

    @app.delete("/admin/groups/{group_id}")
    async def admin_forget_group(group_id: str,
                                 x_looma_admin_token: str | None = Header(default=None)):
        """Убрать группу из списка совсем: отпустить задачи и забыть запись.

        Отдельно от остановки, и намеренно: у остановленной задачи ещё лежит
        результат, за которым придут. Забытая не нужна никому.
        """
        if agents is None:
            return need_agents()
        try:
            await _stop_billing(group_id)
            if deployments is not None:
                await deployments.forget(group_id)
            record = agents.forget_group(group_id)
        except AgentError as exc:
            return _error(404, str(exc))
        return {"forgotten": record.group_id, "tasks": len(record.tasks)}

    @app.get("/admin/groups/{group_id}/health")
    async def admin_group_health(group_id: str,
                                 x_looma_admin_token: str | None = Header(default=None)):
        """Что каждая стадия делает прямо сейчас.

        «running» означает, что процесс запустился, а не что модель готова
        отвечать: веса грузятся минутами, и всё это время состояние задачи
        выглядит одинаково. Разницу знает только сама стадия, поэтому её и
        спрашивают.
        """
        if agents is None:
            return need_agents()
        record = agents.groups.get(group_id)
        if record is None:
            return _error(404, f"нет группы {group_id!r}")
        stages = []
        for rank in sorted(record.tasks):
            task_id = record.tasks[rank]
            task = agents.tasks.get(task_id)
            stage = {"rank": rank, "task_id": task_id,
                     "node_id": record.nodes.get(rank, ""),
                     "state": task.state if task else "gone",
                     "error": task.error if task else "",
                     "seconds": round(task.seconds, 1) if task else 0.0,
                     "stage": None}
            if task and task.state == "running":
                try:
                    status, _headers, answer = await agents.request(
                        task_id, path="/health", timeout_s=15)
                    stage["stage"] = json.loads(answer) if answer else None
                    stage["ready"] = status == 200
                except (AgentError, ValueError):
                    # Стадия ещё не слушает — обычное состояние на старте, а не
                    # повод показать группу сломанной.
                    stage["ready"] = False
            else:
                stage["ready"] = False
            stages.append(stage)
        return {"group_id": group_id, "label": record.label,
                "ready": all(s["ready"] for s in stages) if stages else False,
                "stages": stages}

    @app.post("/admin/groups/{group_id}/stop")
    async def admin_stop_group(group_id: str,
                               x_looma_admin_token: str | None = Header(default=None)):
        if agents is None:
            return need_agents()
        try:
            await _stop_billing(group_id)
            if deployments is not None:
                await deployments.forget(group_id)
            return agents.stop_group(group_id,
                                     reason="stopped from the admin page").as_dict()
        except AgentError as exc:
            return _error(409, str(exc))

    # -------------------------------------------------------------- models
    @app.post("/admin/models/describe")
    async def admin_describe_model(request: Request,
                                   x_looma_admin_token: str | None = Header(default=None)):
        """Сколько в модели слоёв — прежде чем решать, на сколько узлов её резать."""
        raw = await request.json()
        try:
            return describe((raw.get("repo") or "").strip()).as_dict()
        except ModelError as exc:
            return _error(400, str(exc))

    async def _deploy_model(raw: dict, *, account_id=None):
        """Развернуть модель конвейером по узлам.

        Одно действие вместо четырёх: узнать число слоёв, выбрать узлы,
        разрезать слои между ними, отправить стадии вместе с кодом стадии.
        Всё это можно сделать и руками через /admin/groups — здесь просто
        собрано то, что иначе набирают в четыре запроса и один раз ошибаются.
        """
        if agents is None:
            return need_agents()
        # Сочетания, которые не сойдутся ни при каком ответе HuggingFace, —
        # до обращения к нему. Опечатка в имени движка не стоит похода в сеть,
        # и оператор узнаёт о ней сразу, а не через задержку, которая выглядит
        # как работа.
        device = (raw.get("device") or "cuda").strip()
        engine = (raw.get("engine") or "torch").strip().lower()
        if engine not in ("torch", "vllm"):
            return _error(400, f"движок {engine!r} не поддерживается: torch или vllm")
        if engine == "vllm" and device != "cuda":
            # vLLM без карты не поднимается вовсе, и узнавать об этом из его
            # внутренней ошибки на узле — худший способ.
            return _error(400, "движок vllm работает только на cuda")
        try:
            model = describe((raw.get("repo") or "").strip())
        except ModelError as exc:
            return _error(400, str(exc))

        available = [n for n in agents.node_list() if n["accepts_tasks"]]
        if not available:
            return _error(409, "ни один подключённый узел не берёт задачи")
        named = list(raw.get("node_ids") or [])
        if named:
            by_id = {n["node_id"]: n for n in available}
            missing = [n for n in named if n not in by_id]
            if missing:
                return _error(409, f"эти узлы не подключены или не берут работу: {missing}")
            chosen = [by_id[n] for n in named]
        else:
            stages = int(raw.get("stages") or 1)
            if stages > len(available):
                return _error(409,
                              f"просят {stages} стадий, а работу берут {len(available)} узлов")
            # Самые свободные первыми: голове конвейера достаются ещё и
            # эмбеддинги, так что ей полезнее место.
            chosen = sorted(available, key=lambda n: -n["vram_free_bytes"])[:stages]

        if engine == "vllm":
            # Проверяется каждый выбранный узел, а не первый: конвейер встанет
            # ровно настолько, насколько встанет его худшая стадия, и узнать,
            # какая именно, лучше сейчас.
            refusals = [reason for reason in (vllm_refusal(n) for n in chosen) if reason]
            if refusals:
                return _error(409, "; ".join(refusals))

        weights = [n["vram_free_bytes"] for n in chosen] if raw.get("by_vram", True) else None
        try:
            ranges = split_layers(model.num_layers, len(chosen), weights)
            payload = stage_payload()
        except ModelError as exc:
            return _error(400, str(exc))

        dtype = (raw.get("dtype") or "bfloat16").strip()
        label = (raw.get("label") or model.repo.split("/")[-1]).strip()
        per_rank = [{
            "command": [
                "python", "-m", "looma_stage.server",
                "--model-id", label,
                "--weights-uri", model.repo,
                "--start-layer", str(start), "--end-layer", str(end),
                "--device", device, "--dtype", dtype,
                "--engine", engine,
                # Стадия знает только свой срез, а движку нужно знать, где
                # кончается модель: иначе он не определит, что эта стадия
                # последняя, и не соберёт lm_head.
                "--num-model-layers", str(model.num_layers),
            ],
        } for start, end in ranges]

        try:
            record = agents.submit_group(
                size=len(chosen),
                command=per_rank[0]["command"],
                per_rank=per_rank,
                node_ids=[n["node_id"] for n in chosen],
                label=label,
                serve_port=1,
                # Веса модели качает сама стадия — сюда едет только её код.
                inputs=payload,
                environment={"kind": "python",
                             "requirements": stage_requirements(engine)},
                # Без потолка: стадия живёт, пока модель развёрнута.
                timeout_s=raw.get("timeout_s") or 30 * 24 * 3600,
                env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")} if os.environ.get("HF_TOKEN") else None,
            )
        except AgentError as exc:
            return _error(409, str(exc))
        await _start_billing(account_id, record, INFERENCE,
                             nodes=len(chosen), label=label,
                             gpus=sum(int(n.get("gpus_total") or 1) for n in chosen))
        # Чем подняли — в базу. Без этого снятая ради аренды модель не
        # вернётся: тело запроса знает только он сам, а он вот-вот кончится.
        if deployments is not None:
            await deployments.remember(group_id=record.group_id, label=label,
                                       request=raw,
                                       account_id=account_id)
        return {
            **record.as_dict(),
            "model": model.as_dict(),
            "split": [{"rank": i, "node_id": chosen[i]["node_id"],
                       "start_layer": s, "end_layer": e}
                      for i, (s, e) in enumerate(ranges)],
        }

    @app.post("/admin/deploy")
    async def admin_deploy(request: Request,
                           x_looma_admin_token: str | None = Header(default=None)):
        """Развернуть модель. Тонкая обёртка: та же работа нужна и при возврате
        снятого, а звать обработчик из обработчика нельзя."""
        return await _deploy_model(await _body(request),
                                   account_id=whoami(request).account_id)

    async def _start_cluster(raw: dict, *, account_id=None, by_admin: bool):
        """Собрать Ray-кластер на нескольких узлах.

        То же самое, что модель, только жилец другой: обычная группа, обычное
        окружение, обычные входные файлы. Агент не отличает эту задачу от
        стадии инференса и про Ray ничего не знает — см. docs/RAY.md.

        `script` (base64) — точка входа клиента. Её запускает ранг 0, когда
        кластер уже собран; без неё кластер просто стоит и ждёт.
        """
        if agents is None:
            return need_agents()
        available = [n for n in agents.node_list() if n["accepts_tasks"]]
        if not available:
            return _error(409, "ни один подключённый узел не берёт задачи")
        named = list(raw.get("node_ids") or [])
        by_id = {n["node_id"]: n for n in available}
        evicted_ids: list = []
        if named and not by_admin:
            return _error(403, "выбирать узлы по именам может только администратор")
        if named:
            missing = [n for n in named if n not in by_id]
            if missing:
                return _error(409, f"эти узлы не подключены или не берут работу: {missing}")
            picked = [by_id[n] for n in named]
        else:
            size = int(raw.get("size") or 1)
            if size > len(available):
                return _error(409,
                              f"просят {size} узлов, а работу берут {len(available)}")
            # Сначала те, кто легче сходится с соседями: Ray гоняет через этот
            # путь весь свой обмен, а не восемь килобайт на токен, как стадия.
            free = [n for n in prefer_meshy(available)
                    if not int(n.get("tasks_running") or 0)]
            if len(free) >= size:
                picked = free[:size]
            else:
                # Свободных не хватает — прямой клиент вытесняет базовую
                # загрузку. Сначала план целиком, и только если он сходится,
                # что-то снимаем: снятые модели при не вставшей аренде — худшая
                # из возможных развязок.
                made = await _plan_preemption(size, [n["node_id"] for n in free])
                if not made.possible:
                    return _error(409, made.refusal)
                await _evict(made)
                evicted_ids = [g.group_id for g in made.evict]
                picked = [by_id[node] for node in made.nodes if node in by_id]
                if len(picked) < size:
                    return _error(409, "узлы, освобождённые снятием, не вернулись "
                                       "в список свободных — попробуйте ещё раз")
                logger.info("аренда: %s", made.explain())
        chosen = [n["node_id"] for n in picked]

        # Собраться группа обязана ДО того, как займёт карты. Единственный
        # безнадёжный случай — прямого линка нет и реле не развёрнуто: тогда
        # ранги не найдут друг друга, и снаружи это выглядит зависанием.
        relay_available = bool(relay_addrs())
        path = verdict(picked, relay_available=relay_available)
        if not path["ok"]:
            return _error(409, path["why"])

        try:
            payload = ray_payload()
        except PayloadMissing as exc:
            return _error(500, str(exc))

        # Точка входа клиента едет обычным входным файлом: задача стартует в
        # своём каталоге и видит его рядом с собой.
        script: dict = {}
        if raw.get("script"):
            try:
                script = _inputs_of({"inputs": {"job.py": raw["script"]}})
            except ValueError as exc:
                return _error(400, f"'script' должен быть base64: {exc}")
        entry = next(iter(script), "")

        version = (raw.get("ray_version") or "").strip()
        label = (raw.get("label") or "ray").strip()
        # Свой ранг задача узнаёт из окружения, которое ставит агент, поэтому
        # команда у всех одна. Отличается только нулевой: ему запускать код.
        command = ["python", "-m", "looma_ray.server", "--size", str(len(chosen))]
        per_rank = [
            {"command": command + (["--script", entry] if rank == 0 and entry else [])}
            for rank in range(len(chosen))
        ]

        try:
            record = agents.submit_group(
                size=len(chosen),
                command=command,
                per_rank=per_rank,
                node_ids=chosen,
                label=label,
                serve_port=1,
                inputs={**payload, **script},
                # ray[client], а не просто ray: серверная часть клиентского
                # входа лежит в этом extra, и без него `--ray-client-server-port`
                # не игнорируется, а роняет `ray start` целиком.
                environment={"kind": "python", "requirements": [
                    f"ray[client]=={version}" if version else "ray[client]",
                ]},
                resources=raw.get("resources") or None,
                # Скрипт кончается сам; стоящий кластер живёт, пока не снимут.
                timeout_s=int(raw.get("timeout_s") or
                              (6 * 3600 if entry else 30 * 24 * 3600)),
            )
        except AgentError as exc:
            return _error(409, str(exc))
        lease_id = await _start_billing(account_id, record, COMPUTE,
                                        nodes=len(chosen), label=label,
                                        gpus=len(chosen))
        # Снятое ради этой аренды помечается ЕЮ: по этой метке оно и вернётся,
        # когда аренду закроют.
        if deployments is not None and lease_id and evicted_ids:
            await deployments.mark_evicted(evicted_ids, lease_id)
        return {**record.as_dict(), "entry": entry, "nodes": chosen,
                # Каким путём соберётся кластер. Медленный кластер должен быть
                # объяснимым, а не загадочным.
                "path": path["path"], "relayed_pairs": path["relayed_pairs"],
                "warning": path["why"] if path["path"] == "relay" else ""}

    @app.post("/admin/ray")
    async def admin_ray(request: Request,
                        x_looma_admin_token: str | None = Header(default=None)):
        """Кластер от имени администратора: можно назвать узлы и не иметь
        потолка по времени."""
        return await _start_cluster(await _body(request),
                                   account_id=whoami(request).account_id,
                                   by_admin=True)

    # -------------------------------------------------- аренда для клиента
    #: Потолок на одну аренду. «До отмены» без потолка — это забытый кластер,
    #: который тикает, пока кто-нибудь не заметит счёт.
    MAX_RENT_HOURS = int(os.environ.get("LOOMA_MAX_RENT_HOURS", "24"))

    @app.post("/api/compute")
    async def rent_compute(request: Request):
        """Арендовать кластер. То же, что у администратора, но со потолком по
        времени и без выбора узлов: на чьей машине считать — не решение
        арендатора."""
        raw = dict(await _body(request))
        hours = min(max(1, int(raw.get("hours") or 1)), MAX_RENT_HOURS)
        raw.pop("node_ids", None)
        raw["timeout_s"] = hours * 3600
        return await _start_cluster(raw, account_id=whoami(request).account_id,
                                   by_admin=False)

    @app.get("/api/capacity")
    async def capacity(request: Request):
        """Занятость сети — столько, сколько клиенту положено знать.

        Без имён узлов: клиенту незачем знать, на чьей машине он считает, а имя
        узла — это чужой хост. Отдаём состояния и число карт, чтобы кабинет мог
        показать, что произойдёт при аренде: свободных не хватит — платформа
        подвинет свои модели.

        Считается из живого состояния, а не из отдельного счётчика: счётчик
        разошёлся бы с действительностью в первый же час.
        """
        if agents is None:
            return need_agents()
        mine_groups, rented_groups = set(), set()
        if ledger is not None:
            for row in await ledger.open_leases(resource=COMPUTE):
                rented_groups.add(row["group_id"])
                if row["account_id"] == whoami(request).account_id:
                    mine_groups.add(row["group_id"])
        inference_groups = set()
        if deployments is not None:
            inference_groups = {d.group_id for d in await deployments.list()}

        busy: dict = {}
        for group_id, record in agents.groups.items():
            state = ("mine" if group_id in mine_groups
                     else "rented" if group_id in rented_groups
                     else "inference" if group_id in inference_groups else "busy")
            for node_id in getattr(record, "nodes", {}).values():
                # Своё важнее чужого: узел, который держит клиент, показываем
                # ему как свой, даже если на нём же что-то ещё.
                if busy.get(node_id) != "mine":
                    busy[node_id] = state

        rows = [{"state": busy.get(node["node_id"], "free"),
                 "gpus": int(node.get("gpus_total") or 0)}
                for node in agents.node_list() if node.get("accepts_tasks")]
        return {"nodes": rows}

    @app.get("/api/compute")
    async def my_clusters(request: Request):
        """Мои идущие аренды. Только свои: чужие сюда не попадают."""
        if ledger is None:
            return _error(503, "оркестратор поднят без базы: журнала нет")
        mine = await ledger.open_leases(account_id=whoami(request).account_id,
                                        resource=COMPUTE)
        live = agents.groups if agents is not None else {}
        return {"clusters": [
            {**row, "alive": row["group_id"] in live} for row in mine]}

    @app.delete("/api/compute/{group_id}")
    async def drop_cluster(group_id: str, request: Request):
        """Снять свою аренду.

        Владение проверяется по журналу, а не по тому, что прислал клиент:
        иначе номер группы в адресе даёт власть над чужим кластером.
        """
        if ledger is None:
            return _error(503, "оркестратор поднят без базы: журнала нет")
        if not await ledger.lease_belongs_to(group_id,
                                             whoami(request).account_id):
            return _error(404, "такой аренды за вами не числится")
        await _stop_billing(group_id)
        try:
            agents.stop_group(group_id, reason="снято арендатором")
        except AgentError as exc:
            return _error(409, str(exc))
        return {"stopped": group_id}

    @app.websocket("/connect/{group_id}")
    async def connect_to_cluster(socket: WebSocket, group_id: str):
        """Байтовый канал до кластера — то, за чем приходит looma-connect.

        Публичного адреса у кластера нет и не будет: у узла нет входящих
        портов. Наружу смотрит только оркестратор, и он доносит байты по
        стриму, который узел открыл сам.

        Порт спрашивается у самой задачи, а не вычисляется здесь: раскладку
        портов определяет версия Ray, и знать её оркестратору значит обновлять
        оркестратор вместе с ней.
        """
        # Своя проверка, а не общий слой: middleware в Starlette обслуживает
        # только http, и веб-сокет прошёл бы мимо неё незамеченным.
        caller = await auth.identify(
            session=socket.cookies.get(SESSION_COOKIE),
            authorization=socket.headers.get("authorization"),
            admin_token=socket.headers.get("x-looma-admin-token"))
        why = refuse(caller)
        if why:
            # 1008 — «политика»: клиент увидит причину, а не молчаливый обрыв.
            await socket.close(code=1008, reason=_reason(why))
            return
        if agents is None:
            await socket.close(code=1011,
                               reason=_reason("this orchestrator runs no agents"))
            return
        record = agents.groups.get(group_id)
        if record is None:
            await socket.close(code=1008, reason=_reason(f"нет группы {group_id!r}"))
            return
        head = record.tasks.get(0, "")
        try:
            status, _headers, body = await agents.request(head, path="/health",
                                                          timeout_s=15)
            port = int(json.loads(body).get("client_port") or 0) if status == 200 else 0
        except (AgentError, ValueError, TypeError):
            port = 0
        if not port:
            await socket.close(code=1011, reason=_reason(
                "кластер не назвал порт: собирается или поднят без ray[client]"))
            return

        try:
            tunnel = agents.open_tunnel(head, port)
        except AgentError as exc:
            await socket.close(code=1011, reason=_reason(str(exc)))
            return
        await socket.accept()
        await _pipe(socket, tunnel)

    async def _pipe(socket: WebSocket, tunnel) -> None:
        """Возить байты, пока жива любая из сторон.

        Две задачи, а не одна: чтение веб-сокета и чтение канала блокируются
        независимо, и объединять их значит ставить одно в зависимость от
        другого.
        """
        async def outbound() -> None:
            try:
                while True:
                    tunnel.send(await socket.receive_bytes())
            except (WebSocketDisconnect, RuntimeError, KeyError):
                return

        async def inbound() -> None:
            try:
                while True:
                    piece = await tunnel.recv()
                    if not piece:
                        return
                    await socket.send_bytes(piece)
            except (WebSocketDisconnect, RuntimeError):
                return

        tasks = [asyncio.create_task(outbound()), asyncio.create_task(inbound())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Причина узла обязана дойти до человека. Без этого канал просто
            # закрывался, клиент видел таймаут, а объяснение оставалось в логе
            # агента — там, куда за ним никто не пойдёт.
            reason = getattr(tunnel, "error", "")
            tunnel.close()
            try:
                if reason:
                    await socket.close(code=1011, reason=_reason(reason))
                else:
                    await socket.close()
            except RuntimeError:
                pass

    # ------------------------------------------------------------ releases
    @app.get("/agent/release/{version}.tar.gz")
    async def agent_release_archive(version: str):
        """The payload itself. Deliberately unauthenticated.

        A node has a join key, not an admin token, and the payload is signed —
        so handing the bytes to anyone who asks costs nothing, while an
        unsigned payload is worthless to an attacker anyway.
        """
        if releases is None:
            return _error(404, "this orchestrator publishes no agent releases")
        payload = releases.archive_bytes(version)
        if payload is None:
            return _error(404, f"no agent release {version!r}")
        return Response(content=payload, media_type="application/gzip")

    @app.get("/admin/release")
    async def admin_release(x_looma_admin_token: str | None = Header(default=None)):
        """The version map: who is on what, before a wave is advanced."""
        if releases is None:
            return _error(503, "this orchestrator publishes no agent releases")
        return releases.version_map(agents.node_list() if agents else [])

    @app.post("/admin/release")
    async def admin_publish_release(request: Request,
                                    x_looma_admin_token: str | None = Header(default=None)):
        """Take a signed build. Publishing does not roll it out.

        The signature is made elsewhere, with a key this orchestrator does not
        have and must not: taking it over should let an attacker name a release
        we already signed, and nothing more.
        """
        if releases is None:
            return _error(503, "this orchestrator publishes no agent releases")
        raw = await request.json()
        try:
            release = releases.publish(
                version=(raw.get("version") or "").strip(),
                signature=bytes.fromhex((raw.get("signature") or "").strip()),
                archive=base64.b64decode(raw.get("archive") or ""),
            )
        except (ReleaseError, ValueError) as exc:
            return _error(400, str(exc))
        return release.as_dict()

    @app.post("/admin/release/wave")
    async def admin_release_wave(request: Request,
                                 x_looma_admin_token: str | None = Header(default=None)):
        """Advance the rollout. Small first: a bad build reaching the whole
        fleet at once can take with it the ability to ship the fix."""
        if releases is None:
            return _error(503, "this orchestrator publishes no agent releases")
        raw = await request.json()
        try:
            release = releases.set_wave(int(raw.get("percent", 0)))
        except (ReleaseError, TypeError, ValueError) as exc:
            return _error(400, str(exc))
        # Иначе выкатка дошла бы только до тех, кто переподключился: исправный
        # узел держит поток месяцами и о смене ступени не узнал бы никогда.
        told = agents.announce_release() if agents is not None else 0
        return {**release.as_dict(), "nodes_told": told}

    @app.post("/admin/release/withdraw")
    async def admin_release_withdraw(x_looma_admin_token: str | None = Header(default=None)):
        """Stop the spread. Nodes already updated stay updated — taking a
        version back means publishing an older one, which agents refuse."""
        if releases is None:
            return _error(503, "this orchestrator publishes no agent releases")
        releases.withdraw()
        return {"withdrawn": True}

    return app
