# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax/server/shard_loader.py (MLXModelLoader — загрузка среза
# слоёв и ремап ключей веса на локальные индексы шарда, `_to_local_shard_model_key`)
# и src/parallax/vllm/model_runner.py (роли is_first_peer/is_last_peer, приём
# intermediate tensors на не-первой стадии).
# Изменения: реализация на torch/transformers вместо MLX и вместо наследования
# vLLM GPUModelRunner. Причина: нужен device-agnostic исполнитель (cpu/cuda),
# который можно проверить тестами без GPU-стенда и без привязки к внутренним
# API конкретной версии vLLM. Идея (грузить только веса своего диапазона,
# ремапить индексы слоёв в локальные 0..n-1, опускать embed/lm_head на
# ненужных стадиях) сохранена.
"""Partial model loading: materialise only this stage's layers."""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("loom_worker.shard.loader")

_DTYPES = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}


@dataclass
class ShardSpec:
    model_path: str
    start_layer: int
    end_layer: int
    is_first: bool
    is_last: bool
    device: str = "cpu"
    dtype: str = "float32"


class ShardModel:
    """This stage's slice of the model plus the pieces its role requires."""

    def __init__(self, spec: ShardSpec, config, torch_dtype) -> None:
        import torch

        self.spec = spec
        self.config = config
        self.torch = torch
        self.dtype = torch_dtype
        self.device = torch.device(spec.device)
        self.num_layers = spec.end_layer - spec.start_layer
        self.embed = None  # only on the first stage
        self.layers = None
        self.norm = None  # only on the last stage
        self.lm_head = None  # only on the last stage
        self.rotary = None

    # ------------------------------------------------------------------ build
    def build(self) -> "ShardModel":
        import torch
        from transformers import AutoModelForCausalLM

        cfg = self.config
        total_layers = int(getattr(cfg, "num_hidden_layers"))
        if not (0 <= self.spec.start_layer < self.spec.end_layer <= total_layers):
            raise ValueError(
                f"invalid layer range [{self.spec.start_layer}, {self.spec.end_layer}) "
                f"for a model with {total_layers} layers"
            )

        # Build the skeleton with ONLY this stage's layer count, on the meta
        # device so nothing is allocated yet.
        shard_cfg = type(cfg).from_dict(cfg.to_dict())
        shard_cfg.num_hidden_layers = self.num_layers
        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(shard_cfg)

        inner = skeleton.model if hasattr(skeleton, "model") else skeleton
        self.layers = inner.layers
        if self.spec.is_first:
            self.embed = inner.embed_tokens
        if self.spec.is_last:
            self.norm = inner.norm
            self.lm_head = skeleton.lm_head

        # Materialise only the modules this stage keeps (meta -> real memory).
        for module in (self.embed, self.layers, self.norm, self.lm_head):
            if module is not None:
                module.to_empty(device=self.device)

        # Rotary embeddings must NOT go through meta + to_empty(): `inv_freq` is
        # a non-persistent buffer computed in __init__ and absent from the
        # checkpoint, so to_empty() would leave uninitialised memory there. The
        # symptom is subtle — position 0 stays exact while the error grows with
        # position — so build it directly on the target device instead.
        rotary_module = getattr(inner, "rotary_emb", None)
        if rotary_module is not None:
            rotary_cls = type(rotary_module)
            try:
                self.rotary = rotary_cls(config=shard_cfg)
            except TypeError:  # older signatures want the device up front
                self.rotary = rotary_cls(config=shard_cfg, device=self.device)
            self.rotary.to(self.device)

        self._load_weights()
        self._assert_materialised()
        for module in (self.embed, self.layers, self.norm, self.lm_head):
            if module is not None:
                module.eval()
        logger.info(
            "shard built: layers [%d, %d) of %d on %s (first=%s last=%s dtype=%s)",
            self.spec.start_layer,
            self.spec.end_layer,
            total_layers,
            self.spec.device,
            self.spec.is_first,
            self.spec.is_last,
            self.spec.dtype,
        )
        return self

    def _assert_materialised(self) -> None:
        """Fail loudly if any tensor is still meta or holds garbage.

        Catches the class of bug that produced silently wrong outputs above:
        a module materialised with to_empty() whose values never got filled.
        """
        import torch

        for name, module in self._modules_by_prefix().items():
            for pname, tensor in list(module.named_parameters(recurse=True)) + list(
                module.named_buffers(recurse=True)
            ):
                if tensor.is_meta:
                    raise RuntimeError(f"{name}.{pname} is still on the meta device")
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"{name}.{pname} contains non-finite values")
        if self.rotary is not None:
            for pname, tensor in self.rotary.named_buffers(recurse=True):
                if tensor.is_meta or not torch.isfinite(tensor).all():
                    raise RuntimeError(f"rotary.{pname} was not initialised")

    # ------------------------------------------------------------- weights
    def _weight_files(self) -> List[str]:
        path = Path(self.spec.model_path)
        index = path / "model.safetensors.index.json"
        if index.exists():
            mapping = json.loads(index.read_text())["weight_map"]
            return sorted({str(path / f) for f in mapping.values()})
        files = sorted(glob.glob(str(path / "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no safetensors weights under {path}")
        return files

    def _target_name(self, key: str) -> Optional[str]:
        """Map a checkpoint key to this stage's local parameter name.

        `model.layers.{global}` -> `layers.{global - start_layer}`; embeddings
        and head are kept only where the role needs them.
        """
        spec = self.spec
        if key.startswith("model.layers."):
            rest = key[len("model.layers.") :]
            idx_str, _, tail = rest.partition(".")
            try:
                idx = int(idx_str)
            except ValueError:
                return None
            if not (spec.start_layer <= idx < spec.end_layer):
                return None
            return f"layers.{idx - spec.start_layer}.{tail}"
        if key in ("model.embed_tokens.weight", "embed_tokens.weight"):
            return "embed.weight" if spec.is_first else None
        if key in ("model.norm.weight", "norm.weight"):
            return "norm.weight" if spec.is_last else None
        if key in ("lm_head.weight",):
            return "lm_head.weight" if spec.is_last else None
        return None

    def _modules_by_prefix(self) -> Dict[str, object]:
        mods: Dict[str, object] = {"layers": self.layers}
        if self.embed is not None:
            mods["embed"] = self.embed
        if self.norm is not None:
            mods["norm"] = self.norm
        if self.lm_head is not None:
            mods["lm_head"] = self.lm_head
        return mods

    def _load_weights(self) -> None:
        from safetensors import safe_open

        mods = self._modules_by_prefix()
        targets: Dict[str, object] = {}
        for prefix, module in mods.items():
            for name, param in module.named_parameters(recurse=True):
                targets[f"{prefix}.{name}"] = param
            for name, buf in module.named_buffers(recurse=True):
                targets[f"{prefix}.{name}"] = buf

        loaded, skipped = 0, 0
        seen_lm_head = False
        for file in self._weight_files():
            with safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    target_name = self._target_name(key)
                    if target_name is None:
                        skipped += 1
                        continue
                    param = targets.get(target_name)
                    if param is None:
                        skipped += 1
                        continue
                    tensor = f.get_tensor(key).to(dtype=self.dtype)
                    with self.torch.no_grad():
                        param.copy_(tensor.to(self.device))
                    loaded += 1
                    if target_name == "lm_head.weight":
                        seen_lm_head = True

        # Tied embeddings: the checkpoint may omit lm_head entirely.
        if self.spec.is_last and self.lm_head is not None and not seen_lm_head:
            if getattr(self.config, "tie_word_embeddings", False):
                self._load_tied_lm_head(targets)
            else:
                raise RuntimeError("lm_head weights not found in checkpoint")
        logger.info("shard weights loaded: %d tensors (%d irrelevant keys skipped)", loaded, skipped)

    def _load_tied_lm_head(self, targets: Dict[str, object]) -> None:
        """Fill lm_head from the (tied) input embedding matrix."""
        from safetensors import safe_open

        param = targets["lm_head.weight"]
        for file in self._weight_files():
            with safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in ("model.embed_tokens.weight", "embed_tokens.weight"):
                        tensor = f.get_tensor(key).to(dtype=self.dtype)
                        with self.torch.no_grad():
                            param.copy_(tensor.to(self.device))
                        logger.info("lm_head filled from tied embeddings")
                        return
        raise RuntimeError("tie_word_embeddings=True but no embedding weights found")


def resolve_model_path(weights_uri: str, hf_token: Optional[str] = None) -> str:
    """Return a local directory for the weights, downloading from HF if needed."""
    uri = weights_uri
    for prefix in ("hf://", "huggingface://"):
        if uri.startswith(prefix):
            uri = uri[len(prefix) :]
    if os.path.isdir(uri):
        return uri
    from huggingface_hub import snapshot_download

    logger.info("downloading %s from HuggingFace…", uri)
    return snapshot_download(
        repo_id=uri,
        token=hf_token or os.environ.get("HF_TOKEN"),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
    )


def build_shard(spec: ShardSpec) -> Tuple[ShardModel, object]:
    """Load config + this stage's weights. Returns (shard, config)."""
    import torch
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(spec.model_path)
    if hasattr(config, "text_config"):  # multimodal wrappers
        config = config.text_config
    torch_dtype = getattr(torch, _DTYPES.get(spec.dtype, "float32"))
    shard = ShardModel(spec, config, torch_dtype).build()
    return shard, config
