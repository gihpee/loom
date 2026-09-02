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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("looma_stage.loader")

_DTYPES = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}


_LAYER_KEY = re.compile(r"^(?P<prefix>.*?)layers\.(?P<index>\d+)\.(?P<tail>.+)$")


def detect_key_prefix(keys, num_layers: int) -> Optional[str]:
    """The prefix a checkpoint puts before `layers.N.`, if it uses one.

    Plain text models write `model.layers.0....`. Others put the language
    model deeper — `model.language_model.layers.0....` is what Qwen's
    multimodal and MoE checkpoints use — and a loader that knows only the
    first form matches nothing at all.

    Which is exactly what happened, and quietly: every key was declared
    somebody else's business, zero tensors were loaded, and the stage came up
    "healthy" holding uninitialised memory.

    The prefix is chosen by counting, not guessing: a multimodal checkpoint
    also has `visual.blocks.N.` and similar, so the winner is the one whose
    layer indices actually reach the model's layer count.
    """
    spans: Dict[str, set] = {}
    for key in keys:
        match = _LAYER_KEY.match(key)
        if match:
            spans.setdefault(match.group("prefix"), set()).add(int(match.group("index")))
    if not spans:
        return None
    complete = [p for p, idx in spans.items() if max(idx) >= num_layers - 1]
    candidates = complete or list(spans)
    return max(candidates, key=lambda p: len(spans[p]))


def shard_target_key(
    key: str,
    *,
    start_layer: int,
    end_layer: int,
    is_first: bool,
    is_last: bool,
    prefix: str = "model.",
) -> Optional[str]:
    """Map a checkpoint key to this stage's local parameter name, or None.

    `{prefix}layers.{global}` -> `layers.{global - start_layer}`; embeddings
    and head belong only to the stages whose role needs them. Returning None
    means "this tensor is somebody else's" — which is also what tells the
    downloader that a whole safetensors file can be skipped.
    """
    match = _LAYER_KEY.match(key)
    if match and match.group("prefix") == prefix:
        idx = int(match.group("index"))
        if not (start_layer <= idx < end_layer):
            return None
        return f"layers.{idx - start_layer}.{match.group('tail')}"
    if key in (f"{prefix}embed_tokens.weight", "embed_tokens.weight"):
        return "embed.weight" if is_first else None
    if key in (f"{prefix}norm.weight", "norm.weight"):
        return "norm.weight" if is_last else None
    # The head is usually top level even when everything else is nested.
    if key in ("lm_head.weight", f"{prefix}lm_head.weight"):
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
    prefix = detect_key_prefix(weight_map, end_layer)
    if prefix is None:
        return []
    files = set()
    for key, file_name in weight_map.items():
        if shard_target_key(
            key,
            start_layer=start_layer,
            end_layer=end_layer,
            is_first=is_first,
            is_last=is_last,
            prefix=prefix,
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


def resolve_devices(device: str) -> list:
    """Every card this stage may use, in order.

    "cuda" means all of them — the whole point of this: a host with four cards
    should offer four, not the one torch picks by default. "cuda:1" pins to
    exactly that card, which is how an operator splits a machine between two
    workers, and how a test asks for one device on a multi-card box.

    LOOMA_SHARD_DEVICES overrides both, as a comma-separated list.
    """
    import torch

    override = os.environ.get("LOOMA_SHARD_DEVICES", "").strip()
    if override:
        return [torch.device(d.strip()) for d in override.split(",") if d.strip()]
    if device != "cuda":
        return [torch.device(device)]
    try:
        count = torch.cuda.device_count()
    except Exception:
        count = 0
    if count <= 1:
        return [torch.device("cuda")]
    return [torch.device(f"cuda:{i}") for i in range(count)]


def plan_layer_devices(num_layers: int, devices: list) -> list:
    """Which card each layer goes on, proportional to what is free on it.

    Contiguous runs, never interleaved: the hidden state crosses between cards
    once per boundary, so k cards cost k-1 crossings and nothing more. An even
    split would be simpler and wrong on a mixed host — a 4090 next to a 3090
    should carry more of the model, not half of it.

    A card with no room for even one layer is dropped rather than assigned a
    layer that will not fit.
    """
    if len(devices) == 1:
        return [devices[0]] * num_layers

    weights = [max(_free_bytes(d), 0) for d in devices]
    usable = [(d, w) for d, w in zip(devices, weights) if w > 0]
    if not usable:
        return [devices[0]] * num_layers
    devices = [d for d, _ in usable]
    weights = [w for _, w in usable]

    total = float(sum(weights))
    shares = [max(1, round(num_layers * w / total)) for w in weights]
    # Rounding can overshoot or undershoot; settle it on the largest card.
    biggest = weights.index(max(weights))
    shares[biggest] += num_layers - sum(shares)
    while shares[biggest] < 1:  # a pathological split; give the rest back
        donor = max(range(len(shares)), key=lambda i: shares[i])
        if donor == biggest:
            break
        shares[donor] -= 1
        shares[biggest] += 1

    plan = []
    for device, share in zip(devices, shares):
        plan.extend([device] * max(0, share))
    return plan[:num_layers] or [devices[0]] * num_layers


def _free_bytes(device) -> int:
    """Free memory on one card, or a neutral weight when it cannot be read."""
    import torch

    if getattr(device, "type", None) != "cuda":
        return 1  # CPU, or anything without a memory figure: weigh them alike
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except Exception:
        return 1


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
        # Every card this process may use, in order. One entry for a single
        # device or for CPU; several when the host has several GPUs and the
        # stage was not pinned to one of them.
        self.devices = resolve_devices(spec.device)
        self.device = self.devices[0]
        self.num_layers = spec.end_layer - spec.start_layer
        # Which card each of this stage's layers lives on. Filled at build.
        self.layer_devices: List = []
        self.embed = None  # only on the first stage
        self.layers = None
        self.norm = None  # only on the last stage
        self.lm_head = None  # only on the last stage
        self.rotary = None
        # How this checkpoint names its tensors; read from the file at load.
        self._key_prefix = "model."

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

        # Materialise only the modules this stage keeps (meta -> real memory),
        # spreading the layers over the cards this host actually has. The
        # embeddings sit with the first layer and the head with the last, so
        # the hidden state enters and leaves the stage without an extra move.
        self.layer_devices = plan_layer_devices(self.num_layers, self.devices)
        for layer, device in zip(self.layers, self.layer_devices):
            layer.to_empty(device=device)
        if self.embed is not None:
            self.embed.to_empty(device=self.devices[0])
        for module in (self.norm, self.lm_head):
            if module is not None:
                module.to_empty(device=self.devices[-1])

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
            self._device_summary(),
            self.spec.is_first,
            self.spec.is_last,
            self.spec.dtype,
        )
        return self

    def _device_summary(self) -> str:
        """How the layers ended up spread, for the one line that says so."""
        if len(self.devices) == 1:
            return str(self.devices[0])
        counts: dict = {}
        for device in self.layer_devices:
            counts[str(device)] = counts.get(str(device), 0) + 1
        return ", ".join(f"{d}: {n} layers" for d, n in counts.items())

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
            prefix=self._key_prefix,
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

        files = self._weight_files()
        # Resolve the checkpoint's naming once, from the keys themselves. A
        # model that nests its language model deeper than `model.layers.N`
        # would otherwise match nothing and load nothing.
        self._key_prefix = self._resolve_key_prefix(files)

        loaded, skipped = 0, 0
        seen_lm_head = False
        for file in files:
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
                        param.copy_(tensor.to(param.device))
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
        if loaded == 0:
            # Refusing here rather than serving. A stage that loaded nothing
            # holds uninitialised memory: it starts, reports healthy, answers
            # every request with noise, and the first sign of trouble is a
            # crash several stages away with no hint of the cause.
            raise RuntimeError(
                f"no weights matched this stage: {skipped} keys in "
                f"{len(files)} file(s), none of them for layers "
                f"[{self.spec.start_layer}, {self.spec.end_layer}). The "
                f"checkpoint names its tensors "
                f"'{self._key_prefix}layers.N....' — if that looks wrong, this "
                f"model's layout is one shard_target_key() does not know"
            )

    def _resolve_key_prefix(self, files: List[str]) -> str:
        """Read the naming out of the checkpoint rather than assuming it."""
        from safetensors import safe_open

        total = int(getattr(self.config, "num_hidden_layers", self.spec.end_layer))
        keys: List[str] = []
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                keys.extend(handle.keys())
        prefix = detect_key_prefix(keys, total)
        if prefix is None:
            return "model."  # nothing layer-shaped; the caller will refuse
        if prefix != "model.":
            logger.info("checkpoint nests its layers under %r", prefix)
        return prefix

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
                            param.copy_(tensor.to(param.device))
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
    root = os.environ.get("LOOMA_STAGE_VIEW_DIR")
    if root:
        base = Path(root)
    else:
        base = Path(
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        ) / "looma-stage-views"
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
