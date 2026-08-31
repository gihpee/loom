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

import base64
import json
import os
from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from loom.logging_config import get_logger
from loom.orchestrator.agents import AgentError
from loom.orchestrator.models import ModelError, describe, split_layers, stage_payload
from loom.orchestrator.connectivity import prefer_meshy, verdict
from loom.orchestrator.payloads import PayloadMissing, ray_payload
from loom.orchestrator.rendezvous import relay_addrs
from loom.orchestrator.releases import ReleaseError

logger = get_logger(__name__)


def _error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"message": message, "type": kind}})


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
               public_address=None) -> FastAPI:
    app = FastAPI(title="Loom", version="0.2.0")
    admin_token = getattr(config, "admin_token", "") or ""

    def forbidden(token: str | None) -> bool:
        return bool(admin_token) and token != admin_token

    def need_agents():
        return _error(503, "this orchestrator is running without the agent gateway")

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

    # --------------------------------------------------------------- nodes
    # HTML тут не раздаётся: панель — отдельный сервис (docker/web.Dockerfile),
    # который проксирует сюда /admin и /v1. Оркестратор отвечает только данными.

    @app.get("/admin/connect")
    async def admin_connect(x_loom_admin_token: str | None = Header(default=None)):
        """Everything needed to attach a machine: the address and one command."""
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        return {
            "dial_address": getattr(public_address, "address", ""),
            "source": getattr(public_address, "source", "config"),
            "reachable_externally": getattr(public_address, "reachable_externally", None),
            "severity": getattr(public_address, "severity", "info"),
            "self_check": getattr(public_address, "self_check", None),
            "note": getattr(public_address, "note", None),
            "warning": getattr(public_address, "warning", None),
            "agent_image": "gihpee/loomagent",
        }

    @app.get("/admin/agents")
    async def admin_agents(x_loom_admin_token: str | None = Header(default=None)):
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if agents is None:
            return need_agents()
        return {"nodes": agents.node_list()}

    # ---------------------------------------------------------------- keys
    @app.get("/admin/keys")
    async def admin_keys(x_loom_admin_token: str | None = Header(default=None)):
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if keystore is None:
            return {"keys": []}
        return {"keys": [k.public() for k in keystore.list()]}

    @app.post("/admin/keys")
    async def admin_issue_key(request: Request,
                              x_loom_admin_token: str | None = Header(default=None)):
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if keystore is None:
            return _error(503, "this orchestrator issues no keys")
        raw = await request.json()
        key = keystore.issue(label=(raw.get("label") or "").strip(),
                             max_nodes=int(raw.get("max_nodes") or 0))
        return {**key.public(), "key": key.encode(),
                "address": getattr(public_address, "address", ""),
                "agent_image": "gihpee/loomagent"}

    @app.delete("/admin/keys/{key_id}")
    async def admin_revoke_key(key_id: str,
                               x_loom_admin_token: str | None = Header(default=None)):
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if keystore is None or not keystore.revoke(key_id):
            return _error(404, f"no key {key_id!r}")
        return {"revoked": key_id}

    # --------------------------------------------------------------- tasks
    @app.get("/admin/tasks")
    async def admin_tasks(x_loom_admin_token: str | None = Header(default=None)):
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if agents is None:
            return need_agents()
        ordered = sorted(agents.tasks.values(), key=lambda t: t.submitted_at, reverse=True)
        return {"tasks": [t.as_dict() for t in ordered]}

    @app.post("/admin/tasks")
    async def admin_submit_task(request: Request,
                                x_loom_admin_token: str | None = Header(default=None)):
        """Place one task and start it.

        Returns as soon as it is on its way. Provisioning an environment can
        take half an hour, so this must not wait for the task to start — follow
        it by its state instead. Input files come base64-encoded in `inputs`.
        """
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
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
                         x_loom_admin_token: str | None = Header(default=None)):
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if agents is None:
            return need_agents()
        record = agents.tasks.get(task_id)
        return record.as_dict() if record else _error(404, f"no task {task_id!r}")

    @app.get("/admin/tasks/{task_id}/logs")
    async def admin_task_logs(task_id: str, tail: int = 200,
                              x_loom_admin_token: str | None = Header(default=None)):
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if agents is None:
            return need_agents()
        try:
            return {"task_id": task_id, "text": await agents.logs(task_id, tail_lines=tail)}
        except AgentError as exc:
            return _error(409, str(exc))

    @app.get("/admin/tasks/{task_id}/results/{name:path}")
    async def admin_task_result(task_id: str, name: str,
                                x_loom_admin_token: str | None = Header(default=None)):
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
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
                              x_loom_admin_token: str | None = Header(default=None)):
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if agents is None:
            return need_agents()
        try:
            return agents.stop(task_id, reason="stopped from the admin page").as_dict()
        except AgentError as exc:
            return _error(409, str(exc))

    @app.delete("/admin/tasks/{task_id}")
    async def admin_release_task(task_id: str,
                                 x_loom_admin_token: str | None = Header(default=None)):
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if agents is None:
            return need_agents()
        try:
            agents.release(task_id)
        except AgentError as exc:
            return _error(409, str(exc))
        return {"released": task_id}

    # -------------------------------------------------------------- groups
    @app.get("/admin/groups")
    async def admin_groups(x_loom_admin_token: str | None = Header(default=None)):
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if agents is None:
            return need_agents()
        ordered = sorted(agents.groups.values(), key=lambda g: g.submitted_at, reverse=True)
        return {"groups": [g.as_dict() for g in ordered]}

    @app.post("/admin/groups")
    async def admin_submit_group(request: Request,
                                 x_loom_admin_token: str | None = Header(default=None)):
        """Place a job across several nodes — a model pipeline, usually.

        All-or-nothing: a pipeline missing a stage does not run slower, it does
        not run, so a group that cannot be placed whole is not placed at all.

        `inputs` едут каждому рангу — как и одиночной задаче, base64 на файл.
        Одна программа на всех, разное ей передаётся через `per_rank`: узел,
        впервые видящий эту работу, получает её код вместе с задачей, и никакой
        реестр пакетов посередине для этого не нужен.
        """
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
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

    @app.get("/admin/groups/{group_id}/health")
    async def admin_group_health(group_id: str,
                                 x_loom_admin_token: str | None = Header(default=None)):
        """Что каждая стадия делает прямо сейчас.

        «running» означает, что процесс запустился, а не что модель готова
        отвечать: веса грузятся минутами, и всё это время состояние задачи
        выглядит одинаково. Разницу знает только сама стадия, поэтому её и
        спрашивают.
        """
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
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
                               x_loom_admin_token: str | None = Header(default=None)):
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if agents is None:
            return need_agents()
        try:
            return agents.stop_group(group_id,
                                     reason="stopped from the admin page").as_dict()
        except AgentError as exc:
            return _error(409, str(exc))

    # -------------------------------------------------------------- models
    @app.post("/admin/models/describe")
    async def admin_describe_model(request: Request,
                                   x_loom_admin_token: str | None = Header(default=None)):
        """Сколько в модели слоёв — прежде чем решать, на сколько узлов её резать."""
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        raw = await request.json()
        try:
            return describe((raw.get("repo") or "").strip()).as_dict()
        except ModelError as exc:
            return _error(400, str(exc))

    @app.post("/admin/deploy")
    async def admin_deploy(request: Request,
                           x_loom_admin_token: str | None = Header(default=None)):
        """Развернуть модель конвейером по узлам.

        Одно действие вместо четырёх: узнать число слоёв, выбрать узлы,
        разрезать слои между ними, отправить стадии вместе с кодом стадии.
        Всё это можно сделать и руками через /admin/groups — здесь просто
        собрано то, что иначе набирают в четыре запроса и один раз ошибаются.
        """
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if agents is None:
            return need_agents()
        raw = await request.json()
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

        weights = [n["vram_free_bytes"] for n in chosen] if raw.get("by_vram", True) else None
        try:
            ranges = split_layers(model.num_layers, len(chosen), weights)
            payload = stage_payload()
        except ModelError as exc:
            return _error(400, str(exc))

        device = (raw.get("device") or "cuda").strip()
        dtype = (raw.get("dtype") or "bfloat16").strip()
        label = (raw.get("label") or model.repo.split("/")[-1]).strip()
        per_rank = [{
            "command": [
                "python", "-m", "loom_stage.server",
                "--model-id", label,
                "--weights-uri", model.repo,
                "--start-layer", str(start), "--end-layer", str(end),
                "--device", device, "--dtype", dtype,
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
                environment={"kind": "python", "requirements": [
                    "torch", "transformers", "safetensors", "huggingface-hub",
                ]},
                # Без потолка: стадия живёт, пока модель развёрнута.
                timeout_s=raw.get("timeout_s") or 30 * 24 * 3600,
                env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")} if os.environ.get("HF_TOKEN") else None,
            )
        except AgentError as exc:
            return _error(409, str(exc))
        return {
            **record.as_dict(),
            "model": model.as_dict(),
            "split": [{"rank": i, "node_id": chosen[i]["node_id"],
                       "start_layer": s, "end_layer": e}
                      for i, (s, e) in enumerate(ranges)],
        }

    @app.post("/admin/ray")
    async def admin_ray(request: Request,
                        x_loom_admin_token: str | None = Header(default=None)):
        """Собрать Ray-кластер на нескольких узлах.

        То же самое, что модель, только жилец другой: обычная группа, обычное
        окружение, обычные входные файлы. Агент не отличает эту задачу от
        стадии инференса и про Ray ничего не знает — см. docs/RAY.md.

        `script` (base64) — точка входа клиента. Её запускает ранг 0, когда
        кластер уже собран; без неё кластер просто стоит и ждёт.
        """
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if agents is None:
            return need_agents()
        raw = await request.json()

        available = [n for n in agents.node_list() if n["accepts_tasks"]]
        if not available:
            return _error(409, "ни один подключённый узел не берёт задачи")
        named = list(raw.get("node_ids") or [])
        by_id = {n["node_id"]: n for n in available}
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
            picked = prefer_meshy(available)[:size]
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
        command = ["python", "-m", "loom_ray.server", "--size", str(len(chosen))]
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
                environment={"kind": "python", "requirements": [
                    f"ray=={version}" if version else "ray",
                ]},
                resources=raw.get("resources") or None,
                # Скрипт кончается сам; стоящий кластер живёт, пока не снимут.
                timeout_s=int(raw.get("timeout_s") or
                              (6 * 3600 if entry else 30 * 24 * 3600)),
            )
        except AgentError as exc:
            return _error(409, str(exc))
        return {**record.as_dict(), "entry": entry, "nodes": chosen,
                # Каким путём соберётся кластер. Медленный кластер должен быть
                # объяснимым, а не загадочным.
                "path": path["path"], "relayed_pairs": path["relayed_pairs"],
                "warning": path["why"] if path["path"] == "relay" else ""}

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
    async def admin_release(x_loom_admin_token: str | None = Header(default=None)):
        """The version map: who is on what, before a wave is advanced."""
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if releases is None:
            return _error(503, "this orchestrator publishes no agent releases")
        return releases.version_map(agents.node_list() if agents else [])

    @app.post("/admin/release")
    async def admin_publish_release(request: Request,
                                    x_loom_admin_token: str | None = Header(default=None)):
        """Take a signed build. Publishing does not roll it out.

        The signature is made elsewhere, with a key this orchestrator does not
        have and must not: taking it over should let an attacker name a release
        we already signed, and nothing more.
        """
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
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
                                 x_loom_admin_token: str | None = Header(default=None)):
        """Advance the rollout. Small first: a bad build reaching the whole
        fleet at once can take with it the ability to ship the fix."""
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
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
    async def admin_release_withdraw(x_loom_admin_token: str | None = Header(default=None)):
        """Stop the spread. Nodes already updated stay updated — taking a
        version back means publishing an older one, which agents refuse."""
        if forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        if releases is None:
            return _error(503, "this orchestrator publishes no agent releases")
        releases.withdraw()
        return {"withdrawn": True}

    return app
