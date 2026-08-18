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


def shard_target_key(
    key: str, *, start_layer: int, end_layer: int, is_first: bool, is_last: bool
) -> Optional[str]:
    """Map a checkpoint key to this stage's local parameter name, or None.

    `model.layers.{global}` -> `layers.{global - start_layer}`; embeddings and
    head belong only to the stages whose role needs them. Returning None means
    "this tensor is somebody else's" — which is also what tells the downloader
    that a whole safetensors file can be skipped.
    """
    if key.startswith("model.layers."):
        rest = key[len("model.layers.") :]
        idx_str, _, tail = rest.partition(".")
        try:
            idx = int(idx_str)
        except ValueError:
            return None
        if not (start_layer <= idx < end_layer):
            return None
        return f"layers.{idx - start_layer}.{tail}"
    if key in ("model.embed_tokens.weight", "embed_tokens.weight"):
        return "embed.weight" if is_first else None
    if key in ("model.norm.weight", "norm.weight"):
        return "norm.weight" if is_last else None
    if key == "lm_head.weight":
        return "lm_head.weight" if is_last else None
    return None


def needed_weight_files(
    weight_map: Dict[str, str],
    *,
    start_layer: int,
    end_layer: int,
    is_first: bool,
    is_last: bool,
    tie_word_embeddings: bool = False,
) -> List[str]:
    """Safetensors files this stage actually needs, from the checkpoint index.

    A 14B model split over two nodes ships ~8 shard files; a stage that owns
    layers [0,20) needs roughly half of them. Downloading the rest is pure
    waste of disk and time on every worker.

    An empty result means the checkpoint does not use the key naming we know,
    and the caller must fall back to all files rather than load nothing.
    """
    files = set()
    for key, file_name in weight_map.items():
        if shard_target_key(
            key,
            start_layer=start_layer,
            end_layer=end_layer,
            is_first=is_first,
            is_last=is_last,
        ) is not None:
            files.add(file_name)
    # Tied embeddings: the head is not in the checkpoint at all, so the last
    # stage has to read the input-embedding matrix instead.
    if is_last and tie_word_embeddings and "lm_head.weight" not in weight_map:
        for key in ("model.embed_tokens.weight", "embed_tokens.weight"):
            if key in weight_map:
                files.add(weight_map[key])
                break
    return sorted(files)


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
        """Only the files holding this stage's tensors.

        The others may not even be on disk: the downloader fetches this same
        subset (see `resolve_model_path`).
        """
        path = Path(self.spec.model_path)
        index = path / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            names = needed_weight_files(
                weight_map,
                start_layer=self.spec.start_layer,
                end_layer=self.spec.end_layer,
                is_first=self.spec.is_first,
                is_last=self.spec.is_last,
                tie_word_embeddings=bool(getattr(self.config, "tie_word_embeddings", False)),
            )
            if names:
                files = [str(path / name) for name in names]
                missing = [f for f in files if not os.path.exists(f)]
                if missing:
                    raise FileNotFoundError(
                        f"weight files missing for layers "
                        f"[{self.spec.start_layer}, {self.spec.end_layer}): {missing}"
                    )
                return files
            logger.warning(
                "checkpoint index uses unknown key naming; reading every shard file"
            )
            return sorted({str(path / f) for f in weight_map.values()})
        files = sorted(glob.glob(str(path / "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no safetensors weights under {path}")
        return files

    def _target_name(self, key: str) -> Optional[str]:
        return shard_target_key(
            key,
            start_layer=self.spec.start_layer,
            end_layer=self.spec.end_layer,
            is_first=self.spec.is_first,
            is_last=self.spec.is_last,
        )

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


_METADATA_PATTERNS = ["*.json", "*.model", "*.txt", "tokenizer*"]


def resolve_model_path(
    weights_uri: str,
    hf_token: Optional[str] = None,
    *,
    shard: Optional[ShardSpec] = None,
) -> str:
    """Return a local directory for the weights, downloading from HF if needed.

    With `shard`, only the safetensors files holding that stage's tensors are
    fetched: metadata first (config + index + tokenizer, a few hundred KB),
    then exactly the shard files named by `model.safetensors.index.json` for
    the stage's layer range. A 14B model on two nodes then costs each of them
    about half the checkpoint instead of all of it.
    """
    uri = weights_uri
    for prefix in ("hf://", "huggingface://"):
        if uri.startswith(prefix):
            uri = uri[len(prefix) :]
    if os.path.isdir(uri):
        return uri
    from huggingface_hub import snapshot_download

    token = hf_token or os.environ.get("HF_TOKEN")
    if shard is None:
        logger.info("downloading %s from HuggingFace (whole checkpoint)…", uri)
        return snapshot_download(
            repo_id=uri, token=token, allow_patterns=_METADATA_PATTERNS + ["*.safetensors"]
        )

    # Pass 1: metadata only — cheap, and it tells us what to fetch next.
    path = snapshot_download(repo_id=uri, token=token, allow_patterns=_METADATA_PATTERNS)
    patterns = _weight_patterns_for_shard(Path(path), shard)
    logger.info("downloading %s from HuggingFace: %d weight file(s)…", uri, len(patterns))
    return snapshot_download(
        repo_id=uri, token=token, allow_patterns=_METADATA_PATTERNS + patterns
    )


def build_stage_checkpoint_view(model_path: str, shard: ShardSpec) -> str:
    """Present a stage-sized checkpoint to an engine that reads files itself.

    Our own loader picks tensors out of whatever files are on disk, so it is
    happy with a partial download. vLLM is not: it enumerates every file listed
    in `model.safetensors.index.json` and opens it, so a missing file is an
    error rather than a saving.

    So we build a view directory: symlinks to the metadata and to the weight
    files this stage needs, plus an index rewritten to mention only those. vLLM
    then sees a complete, self-consistent checkpoint that happens to contain
    just this stage's layers, and downloads shrink from the whole model to the
    stage's share.

    Returns the original path unchanged when there is nothing to prune (a
    single-file checkpoint, or key naming we do not recognise).
    """
    source = Path(model_path)
    index_file = source / "model.safetensors.index.json"
    if not index_file.exists():
        return model_path

    index = json.loads(index_file.read_text())
    weight_map = index.get("weight_map") or {}
    tie = False
    config_file = source / "config.json"
    if config_file.exists():
        raw = json.loads(config_file.read_text())
        tie = bool(
            raw.get("tie_word_embeddings", raw.get("text_config", {}).get(
                "tie_word_embeddings", False
            ))
        )
    needed = needed_weight_files(
        weight_map,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        is_first=shard.is_first,
        is_last=shard.is_last,
        tie_word_embeddings=tie,
    )
    all_files = sorted(set(weight_map.values()))
    if not needed or set(needed) == set(all_files):
        return model_path

    view = _stage_view_dir(source, shard)
    view.mkdir(parents=True, exist_ok=True)
    keep = set(needed)
    for entry in source.iterdir():
        if entry.name == index_file.name:
            continue
        if entry.suffix == ".safetensors" and entry.name not in keep:
            continue  # another stage's weights
        _link(entry, view / entry.name)

    pruned = dict(index)
    pruned["weight_map"] = {k: v for k, v in weight_map.items() if v in keep}
    (view / index_file.name).write_text(json.dumps(pruned))
    logger.info(
        "stage checkpoint view: %d of %d weight files, %d of %d tensors -> %s",
        len(needed),
        len(all_files),
        len(pruned["weight_map"]),
        len(weight_map),
        view,
    )
    return str(view)


def _stage_view_dir(source: Path, shard: ShardSpec) -> Path:
    """Where a stage's view lives: beside the cache, keyed by its layer range."""
    root = os.environ.get("LOOM_STAGE_VIEW_DIR")
    if root:
        base = Path(root)
    else:
        base = Path(
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        ) / "loom-stage-views"
    tag = f"{source.name}-{shard.start_layer}-{shard.end_layer}"
    return base / tag


def _link(source: Path, target: Path) -> None:
    """Symlink into the cache; copy only if the filesystem refuses links."""
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        import shutil

        shutil.copy2(source, target)


def _weight_patterns_for_shard(path: Path, shard: ShardSpec) -> List[str]:
    """Which safetensors files to fetch for this stage (all, if unsure)."""
    index = path / "model.safetensors.index.json"
    if not index.exists():
        # Single-file checkpoint (or an unusual layout): nothing to select.
        return ["*.safetensors"]
    weight_map = json.loads(index.read_text())["weight_map"]
    tie = False
    config_file = path / "config.json"
    if config_file.exists():
        raw = json.loads(config_file.read_text())
        tie = bool(raw.get("tie_word_embeddings", raw.get("text_config", {}).get(
            "tie_word_embeddings", False
        )))
    names = needed_weight_files(
        weight_map,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        is_first=shard.is_first,
        is_last=shard.is_last,
        tie_word_embeddings=tie,
    )
    total = len(set(weight_map.values()))
    if not names:
        logger.warning(
            "checkpoint index uses unknown key naming; downloading all %d weight files",
            total,
        )
        return ["*.safetensors"]
    logger.info(
        "layers [%d, %d): %d of %d weight files needed (%s)",
        shard.start_layer,
        shard.end_layer,
        len(names),
        total,
        ", ".join(names[:4]) + ("…" if len(names) > 4 else ""),
    )
    return names


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
