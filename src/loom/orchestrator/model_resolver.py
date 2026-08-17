"""Build a ModelSpec from a HuggingFace repo id.

Lets an operator deploy a model by name ("Qwen/Qwen3-32B") instead of
hand-writing the architectural `model_info` block: we fetch the model's
`config.json` and map its fields onto ModelInfo, which is what Phase-1 needs to
size shards.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import httpx

from loom.logging_config import get_logger
from loom.orchestrator.registry import ModelSpec
from loom.planning import ModelInfo

logger = get_logger(__name__)

HF_BASE = "https://huggingface.co"

_DTYPE_BYTES = {
    "float32": 4,
    "float": 4,
    "bfloat16": 2,
    "float16": 2,
    "half": 2,
    "int8": 1,
    "fp8": 1,
}


class ModelResolveError(RuntimeError):
    """Raised when a repo's config cannot be turned into a ModelInfo."""


async def fetch_hf_config(repo_id: str, *, token: Optional[str] = None) -> Dict[str, Any]:
    """Download `config.json` for a repo (public or with an HF token)."""
    repo_id = repo_id.strip().strip("/")
    for prefix in ("hf://", "huggingface://", f"{HF_BASE}/"):
        if repo_id.startswith(prefix):
            repo_id = repo_id[len(prefix) :]
    url = f"{HF_BASE}/{repo_id}/resolve/main/config.json"
    headers = {}
    token = token or os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
    if resp.status_code == 401 or resp.status_code == 403:
        raise ModelResolveError(
            f"{repo_id}: access denied — set HF_TOKEN on the orchestrator for gated models"
        )
    if resp.status_code == 404:
        raise ModelResolveError(f"{repo_id}: config.json not found (wrong repo id?)")
    if resp.status_code >= 400:
        raise ModelResolveError(f"{repo_id}: HuggingFace returned {resp.status_code}")
    return resp.json()


def model_info_from_hf_config(config: Dict[str, Any], *, model_name: str) -> ModelInfo:
    """Map a HF `config.json` onto ModelInfo (what the planner measures with)."""
    cfg = config.get("text_config", config)  # multimodal wrappers nest the LM

    required = ("num_hidden_layers", "hidden_size", "num_attention_heads", "vocab_size")
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        raise ModelResolveError(
            f"config.json lacks {', '.join(missing)} — unsupported architecture"
        )

    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg.get("num_key_value_heads") or heads)
    head_dim = int(cfg.get("head_dim") or (hidden // heads))
    dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16").lower()
    param_bytes = _DTYPE_BYTES.get(dtype, 2)

    # MoE (Mixtral/Qwen-MoE style) and MLA (DeepSeek style) extras.
    experts = cfg.get("num_local_experts") or cfg.get("num_experts")
    experts_per_tok = cfg.get("num_experts_per_tok")
    moe_intermediate = cfg.get("moe_intermediate_size")

    kwargs: Dict[str, Any] = dict(
        model_name=model_name,
        mlx_model_name=model_name,
        head_size=head_dim,
        hidden_dim=hidden,
        intermediate_dim=int(cfg.get("intermediate_size") or 4 * hidden),
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        vocab_size=int(cfg["vocab_size"]),
        num_layers=int(cfg["num_hidden_layers"]),
        ffn_num_projections=3,  # gate/up/down for all Llama-family MLPs
        tie_embedding=bool(cfg.get("tie_word_embeddings", False)),
        param_bytes_per_element=param_bytes,
        mlx_param_bytes_per_element=param_bytes,
        cache_bytes_per_element=param_bytes,
        embedding_bytes_per_element=param_bytes,
    )
    if experts and experts_per_tok:
        kwargs.update(
            num_local_experts=int(experts),
            num_experts_per_tok=int(experts_per_tok),
            moe_intermediate_dim=int(moe_intermediate) if moe_intermediate else None,
        )
    for mla_key in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
        if cfg.get(mla_key) is not None:
            kwargs[mla_key] = int(cfg[mla_key])
    return ModelInfo(**kwargs)


async def spec_from_hf(
    repo_id: str,
    *,
    model_id: Optional[str] = None,
    backend_type: str = "shard",
    priority: float = 1.0,
    demand_qps: float = 1.0,
    price_willing: float = 1.0,
    target_pipelines: int = 1,
    slo_p95_ttft_ms: Optional[float] = None,
    token: Optional[str] = None,
) -> Tuple[ModelSpec, Dict[str, Any]]:
    """Resolve a HF repo into a deployable ModelSpec.

    Returns the spec plus the raw config (handy for showing the operator what
    was detected).
    """
    config = await fetch_hf_config(repo_id, token=token)
    clean_repo = repo_id.strip().strip("/")
    info = model_info_from_hf_config(config, model_name=clean_repo)
    spec = ModelSpec(
        model_id=model_id or clean_repo.split("/")[-1].lower(),
        weights_uri=clean_repo,
        backend_type=backend_type,
        model_info=info,
        demand_qps=demand_qps,
        priority=priority,
        price_willing=price_willing,
        target_pipelines=target_pipelines,
        slo_p95_ttft_ms=slo_p95_ttft_ms,
    )
    logger.info(
        "resolved %s: %d layers, hidden %d, %s weights",
        clean_repo,
        info.num_layers,
        info.hidden_dim,
        f"{info.param_bytes_per_element}B/param",
    )
    return spec, config
