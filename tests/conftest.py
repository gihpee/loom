"""Shared fixtures for Looma planning tests."""

import os
import sys
import types
from pathlib import Path

import pytest

# Tests must never reach out to the internet: the public-IP lookup in
# orchestrator/public_addr.py is disabled for the whole session.
os.environ.setdefault("LOOMA_SKIP_IP_LOOKUP", "1")

# Path to the read-only Parallax checkout used for regression parity tests.
PARALLAX_SRC = Path(__file__).resolve().parent.parent.parent / "dllmi" / "parallax" / "src"

GIB = 1024**3


def make_model_info_kwargs(num_layers: int = 24):
    """Small dense-model architecture used across tests (Qwen3-0.6B-like)."""
    return dict(
        model_name="test-model",
        mlx_model_name="test-model-mlx",
        head_size=128,
        hidden_dim=1024,
        intermediate_dim=3072,
        num_attention_heads=16,
        num_kv_heads=8,
        vocab_size=151_936,
        num_layers=num_layers,
        ffn_num_projections=3,
        tie_embedding=True,
        param_bytes_per_element=2,
        mlx_param_bytes_per_element=2,
        cache_bytes_per_element=2,
        embedding_bytes_per_element=2,
    )


@pytest.fixture
def parallax_modules():
    """Import the ORIGINAL Parallax scheduling modules from the read-only tree.

    Heavy third-party deps (torch, mlx) are stubbed out: the scheduling code
    only touches dtype attributes and never executes device paths in these
    tests. Skips if the checkout is missing.
    """
    if not PARALLAX_SRC.exists():
        pytest.skip(f"Parallax source tree not found at {PARALLAX_SRC}")

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")

        class _Dtype:
            pass

        for name in ("float32", "bfloat16", "float16", "half", "int8"):
            setattr(torch_stub, name, _Dtype())
        torch_stub.cuda = types.SimpleNamespace(
            mem_get_info=lambda *a, **k: (0, 0), current_device=lambda: 0
        )
        sys.modules["torch"] = torch_stub

    # parallax_utils.utils imports HardwareInfo from parallax.server.server_info,
    # which pulls in mlx; the scheduling code under test never calls it.
    if "parallax.server.server_info" not in sys.modules:
        pkg = sys.modules.setdefault("parallax", types.ModuleType("parallax"))
        server_pkg = sys.modules.setdefault("parallax.server", types.ModuleType("parallax.server"))
        pkg.server = server_pkg
        server_info = types.ModuleType("parallax.server.server_info")

        class HardwareInfo:  # pragma: no cover - never used in tests
            @classmethod
            def detect(cls):
                raise RuntimeError("stub")

        server_info.HardwareInfo = HardwareInfo
        sys.modules["parallax.server.server_info"] = server_info
        server_pkg.server_info = server_info

    sys.path.insert(0, str(PARALLAX_SRC))
    try:
        from scheduling.layer_allocation import (  # noqa: F401
            DynamicProgrammingLayerAllocator as PxDPAllocator,
        )
        from scheduling.layer_allocation import GreedyLayerAllocator as PxGreedyAllocator
        from scheduling.model_info import ModelInfo as PxModelInfo
        from scheduling.node import Node as PxNode
        from scheduling.node import NodeHardwareInfo as PxHardware
        from scheduling.node_management import NodeManager as PxNodeManager

        yield types.SimpleNamespace(
            ModelInfo=PxModelInfo,
            Node=PxNode,
            NodeHardwareInfo=PxHardware,
            NodeManager=PxNodeManager,
            DPAllocator=PxDPAllocator,
            GreedyAllocator=PxGreedyAllocator,
        )
    finally:
        sys.path.remove(str(PARALLAX_SRC))
