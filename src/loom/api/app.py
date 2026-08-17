"""Client API (Phase 2): model-aware routing + catalog admin.

- /v1/models            — full catalog from the Model Registry
- /v1/chat/completions  — routed by body["model"] to endpoints of THAT model
                          only (model-aware EndpointRegistry + Phase-2 head)
- /admin/models         — catalog CRUD; triggers a Resource Broker pass
- /admin/status         — nodes, grants, endpoints, unscheduled models
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from loom.logging_config import get_logger
from loom.orchestrator.controller import MultiModelController
from loom.orchestrator.model_resolver import ModelResolveError, spec_from_hf
from loom.orchestrator.registry import ModelSpec
from loom.orchestrator.tunnel import TunnelError

logger = get_logger(__name__)


def _error(status: int, message: str, err_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": err_type}}
    )


def create_app(controller: MultiModelController) -> FastAPI:
    app = FastAPI(title="Loom API", version="0.0.2")
    client = httpx.AsyncClient(timeout=httpx.Timeout(20 * 60, connect=10))
    admin_token = controller.config.admin_token

    def _admin_forbidden(provided: str | None) -> bool:
        return bool(admin_token) and provided != admin_token

    @app.get("/healthz")
    async def healthz():
        status = controller.status()
        serving = [m for m, s in status["models"].items() if s["endpoints"]]
        return {"status": "ok" if serving else "no_capacity", "serving": serving}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": spec.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "loom",
                    "loom": {
                        "priority": spec.priority,
                        "serving": bool(controller.endpoints.candidates(spec.model_id)),
                    },
                }
                for spec in controller.registry.list()
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        model_id = body.get("model")
        if not model_id:
            ids = controller.registry.ids()
            if len(ids) == 1:
                model_id = ids[0]
            else:
                return _error(400, "'model' is required")
        if controller.registry.get(model_id) is None:
            return _error(404, f"model '{model_id}' not found")
        endpoint = controller.pick_endpoint(model_id)
        if endpoint is None:
            return _error(503, f"no serving capacity for model '{model_id}'", "server_error")

        controller.endpoints.mark_request_start(endpoint)
        started = time.monotonic()

        def finish(*, error: bool) -> None:
            ttft_ms = (time.monotonic() - started) * 1000
            controller.endpoints.mark_request_end(endpoint, error=error, ttft_ms=ttft_ms)
            controller.record_request(model_id, ttft_ms=ttft_ms, error=error)

        streaming = bool(body.get("stream"))
        payload = json.dumps(body).encode()
        try:
            if endpoint.base_url.startswith("tunnel://"):
                # Normal production path: relay over the worker's outbound
                # data-plane stream (worker exposes no address at all).
                head, chunks = await controller.tunnel.request(
                    endpoint.node_id,
                    model_id=model_id,
                    path="/v1/chat/completions",
                    body=payload,
                    stream=streaming,
                )
                finish(error=head.status >= 500)
                if streaming:
                    return StreamingResponse(
                        chunks,
                        status_code=head.status,
                        media_type=head.headers.get("Content-Type", "text/event-stream"),
                    )
                buf = bytearray()
                async for chunk in chunks:
                    buf.extend(chunk)
                return Response(
                    content=bytes(buf),
                    status_code=head.status,
                    media_type=head.headers.get("Content-Type", "application/json"),
                )

            # Directly dialable backend (local dev / tests).
            url = f"{endpoint.base_url}/v1/chat/completions"
            if streaming:
                upstream_req = client.build_request("POST", url, json=body)
                upstream_resp = await client.send(upstream_req, stream=True)
                finish(error=upstream_resp.status_code >= 500)

                async def relay():
                    try:
                        async for chunk in upstream_resp.aiter_bytes():
                            yield chunk
                    finally:
                        await upstream_resp.aclose()

                return StreamingResponse(
                    relay(),
                    status_code=upstream_resp.status_code,
                    media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
                )
            resp = await client.post(url, json=body)
            finish(error=resp.status_code >= 500)
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except TunnelError as exc:
            finish(error=True)
            logger.warning("tunnel to %s failed: %s", endpoint.node_id, exc)
            return _error(502, f"worker unreachable: {exc}", "server_error")
        except httpx.HTTPError as exc:
            finish(error=True)
            logger.warning("upstream %s failed: %s", endpoint.base_url, exc)
            return _error(502, f"upstream error: {exc}", "server_error")

    # ------------------------------------------------------------------ admin
    @app.get("/admin/ui")
    async def admin_ui():
        """Manual-testing dashboard (dev tool). Data calls carry the admin token."""
        page = Path(__file__).with_name("admin_ui.html").read_text()
        return HTMLResponse(page)

    @app.get("/admin/connect")
    async def admin_connect(x_loom_admin_token: str | None = Header(default=None)):
        """Everything needed to attach a GPU box: address + ready-made command."""
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        public = getattr(controller, "public_address", None)
        address = getattr(public, "address", controller.config.public_address)
        return {
            "dial_address": address,
            "source": getattr(public, "source", "config"),
            "reachable_externally": getattr(public, "reachable_externally", None),
            # "ok" | "info" | "warn" — a private LAN address is fine (info),
            # only loopback / a failed check is a real problem (warn).
            "severity": getattr(public, "severity", "info"),
            "self_check": getattr(public, "self_check", None),
            "note": getattr(public, "note", None),
            "warning": getattr(public, "warning", None),
            "worker_image": "gihpee/loomworker",
        }

    @app.post("/admin/models/from_hf")
    async def admin_add_model_from_hf(
        request: Request, x_loom_admin_token: str | None = Header(default=None)
    ):
        """Deploy a model by HuggingFace name — architecture is auto-detected."""
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        raw = await request.json()
        repo = (raw.get("repo") or raw.get("weights_uri") or "").strip()
        if not repo:
            return _error(400, "'repo' is required, e.g. 'Qwen/Qwen3-32B'")
        try:
            spec, config = await spec_from_hf(
                repo,
                model_id=raw.get("model_id") or None,
                backend_type=raw.get("backend_type") or "shard",
                priority=float(raw.get("priority", 1) or 1),
                demand_qps=float(raw.get("demand_qps", 1) or 1),
                price_willing=float(raw.get("price_willing", 1) or 1),
                target_pipelines=int(raw.get("target_pipelines", 1) or 1),
                slo_p95_ttft_ms=(
                    float(raw["slo_p95_ttft_ms"]) if raw.get("slo_p95_ttft_ms") else None
                ),
            )
        except ModelResolveError as exc:
            return _error(400, str(exc))
        except Exception as exc:  # network/JSON problems
            logger.exception("resolving %s failed", repo)
            return _error(502, f"could not resolve {repo}: {exc}", "server_error")

        await controller.add_model(spec)
        info = spec.model_info
        broker = controller.broker
        per_card = {
            f"{gb}GB": broker.layers_fitting(gb * 1024**3, info) for gb in (24, 48, 80)
        }
        return {
            "added": spec.model_id,
            "detected": {
                "num_layers": info.num_layers,
                "hidden_dim": info.hidden_dim,
                "num_kv_heads": info.num_kv_heads,
                "dtype_bytes": info.param_bytes_per_element,
                "architectures": config.get("architectures"),
            },
            "sizing": {
                "weights_gb": round(
                    (
                        info.num_layers * info.decoder_layer_io_bytes(roofline=False)
                        + (1 if info.tie_embedding else 2) * info.embedding_io_bytes
                    )
                    / 1024**3,
                    2,
                ),
                "layers_per_card": per_card,
                "param_mem_ratio": controller.config.param_mem_ratio,
            },
        }

    @app.post("/admin/keys")
    async def admin_issue_key(
        request: Request, x_loom_admin_token: str | None = Header(default=None)
    ):
        """Issue a join key. This is the ONLY thing a GPU owner needs."""
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        keystore = getattr(controller, "keystore", None)
        if keystore is None:
            return _error(503, "keystore not configured", "server_error")
        try:
            raw = await request.json()
        except Exception:
            raw = {}
        key = keystore.issue(
            label=str(raw.get("label", "")), max_nodes=int(raw.get("max_nodes", 0) or 0)
        )
        encoded = key.encode()
        return {
            "key": encoded,
            "key_id": key.key_id,
            "address": key.address,
            "label": key.label,
            "max_nodes": key.max_nodes,
            "run_command": f"docker run -d --gpus all gihpee/loomworker --key {encoded}",
        }

    @app.get("/admin/keys")
    async def admin_list_keys(x_loom_admin_token: str | None = Header(default=None)):
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        keystore = getattr(controller, "keystore", None)
        if keystore is None:
            return {"keys": []}
        return {"keys": [k.public() for k in keystore.list()]}

    @app.delete("/admin/keys/{key_id}")
    async def admin_revoke_key(
        key_id: str, x_loom_admin_token: str | None = Header(default=None)
    ):
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        keystore = getattr(controller, "keystore", None)
        if keystore is None or not keystore.revoke(key_id):
            return _error(404, f"key '{key_id}' not found")
        return {"revoked": key_id}

    @app.get("/admin/nodes")
    async def admin_nodes(x_loom_admin_token: str | None = Header(default=None)):
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        return controller.nodes_view()

    @app.get("/admin/models_view")
    async def admin_models_view(x_loom_admin_token: str | None = Header(default=None)):
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        return controller.models_view()

    @app.get("/admin/perfmap/{model_id}")
    async def admin_perfmap(
        model_id: str, x_loom_admin_token: str | None = Header(default=None)
    ):
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        view = controller.perfmap_view(model_id)
        if view is None:
            return _error(404, f"model '{model_id}' not found")
        return view

    @app.post("/admin/rebalance")
    async def admin_rebalance(x_loom_admin_token: str | None = Header(default=None)):
        """Force a Resource Broker pass now (same trigger the timer uses)."""
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        await controller.rebalance(reason="manual")
        return {
            "ok": True,
            "allocations": controller.last_plan.allocations if controller.last_plan else {},
            "unscheduled": controller.last_plan.unscheduled if controller.last_plan else [],
        }

    @app.get("/admin/status")
    async def admin_status(x_loom_admin_token: str | None = Header(default=None)):
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        return controller.status()

    @app.post("/admin/models")
    async def admin_add_model(
        request: Request, x_loom_admin_token: str | None = Header(default=None)
    ):
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        raw = await request.json()
        try:
            spec = ModelSpec.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            return _error(400, f"bad model spec: {exc}")
        await controller.add_model(spec)
        return {"added": spec.model_id, "score": spec.score()}

    @app.post("/admin/quota")
    async def admin_set_quota(
        request: Request, x_loom_admin_token: str | None = Header(default=None)
    ):
        """Ops tool: override a shard's VRAM quota on a worker (watchdog demo)."""
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        raw = await request.json()
        try:
            ack = await controller.set_quota(
                raw["model_id"], raw["node_id"], int(raw["vram_quota_bytes"])
            )
        except KeyError as exc:
            return _error(404, str(exc))
        return {"ok": ack.ok, "error": ack.error}

    @app.delete("/admin/models/{model_id}")
    async def admin_remove_model(
        model_id: str, x_loom_admin_token: str | None = Header(default=None)
    ):
        if _admin_forbidden(x_loom_admin_token):
            return _error(403, "invalid admin token")
        removed = await controller.remove_model(model_id)
        if not removed:
            return _error(404, f"model '{model_id}' not found")
        return {"removed": model_id}

    @app.on_event("shutdown")
    async def shutdown():
        await client.aclose()

    return app
