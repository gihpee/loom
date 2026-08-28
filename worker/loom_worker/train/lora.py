"""LoRA adapters on a stage's slice of a model.

Why adapters and not the whole model. Training the base weights needs, per
parameter: the weight, its gradient, and the optimiser's two moments. In fp32
that is sixteen bytes where inference needs two. A 27B model that just fits
across four 24 GB cards for inference needs sixteen such cards to fine-tune —
the fleet Loom is built for does not have them.

LoRA changes the arithmetic rather than the hardware. The base weights stay
frozen and shared with inference; each adapted projection gets two small
matrices, and only those carry gradients and optimiser state. For rank 16 on a
27B model that is well under one percent of the parameters, so a card that can
serve a stage can also train one.

It also decides what crosses the network, which matters more here than on a
single machine. Stages hold different layers, so there is no gradient to
all-reduce between them — nothing like the full-parameter exchange that makes
data-parallel training hard over a slow link. What crosses is the activations
forward and their gradients backward, and both are the size of one
micro-batch's hidden states.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger("loom_worker.train.lora")

# The projections adapters are usually attached to. Attention carries most of
# what fine-tuning changes, and adapting it alone keeps the parameter count —
# and so the memory and the checkpoint — small.
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


# Built on first use rather than at import: torch is not present in every
# image that imports this package, and a class statement needs it up front.
_LORA_CLASS = None


def _lora_class():
    global _LORA_CLASS
    if _LORA_CLASS is not None:
        return _LORA_CLASS
    import torch

    class LoraLinear(torch.nn.Module):
        """A frozen Linear with a trainable low-rank correction beside it.

        Wraps rather than replaces: the original module keeps its weights and
        its place, so the base model is untouched and the same process can
        still serve inference from it.
        """

        def __init__(self, base, rank: int, alpha: float, dropout: float = 0.0) -> None:
            super().__init__()
            self.base = base
            self.rank = rank
            # The published LoRA scaling. Keeping alpha apart from rank lets
            # the rank change without re-tuning the learning rate.
            self.scaling = alpha / rank
            weight = base.weight
            device, dtype = weight.device, weight.dtype

            # A random, B zero, so the adapter contributes exactly nothing at
            # step 0: training starts from the base model's own behaviour
            # rather than from a perturbed one.
            self.lora_a = torch.nn.Parameter(
                torch.empty(rank, weight.shape[1], device=device, dtype=dtype)
            )
            torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.lora_b = torch.nn.Parameter(
                torch.zeros(weight.shape[0], rank, device=device, dtype=dtype)
            )
            self.drop = torch.nn.Dropout(dropout) if dropout > 0 else None

        def forward(self, x):
            out = self.base(x)
            h = self.drop(x) if self.drop is not None else x
            correction = (h @ self.lora_a.T) @ self.lora_b.T
            return out + correction * self.scaling

        def trainable(self):
            return [self.lora_a, self.lora_b]

    _LORA_CLASS = LoraLinear
    return _LORA_CLASS


def train_everything(shard) -> List:
    """Classic fine-tuning: every weight of this stage is trainable.

    Offered because it is what "training" usually means, and refused by the
    memory before it is refused by anything else. Per parameter this needs the
    weight, its gradient and two optimiser moments — sixteen bytes in fp32
    where inference needs two. A stage that fits on a card for inference needs
    roughly eight such cards to train this way.

    It is exact where LoRA is an approximation, so it belongs in the tool for
    the small models where it fits. The caller is told the size (see
    TrainingStage) rather than left to discover it as an out-of-memory crash
    halfway through the first step.
    """
    params = []
    for module in _modules(shard):
        for param in module.parameters(recurse=True):
            param.requires_grad_(True)
            params.append(param)
    if not params:
        raise RuntimeError("this stage holds no parameters to train")
    logger.info("full fine-tuning: %d trainable parameters", sum(p.numel() for p in params))
    return params


def attach_lora(
    shard,
    *,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> Dict[str, LoraLinear]:
    """Freeze this stage's weights and hang adapters off its projections.

    Returns the adapters by name, keyed as `layers.{local}.{module path}` —
    local to the stage, because that is what this worker owns. The stage's
    layer offset is recorded separately, so a checkpoint can be reassembled
    into one adapter file no matter how the model was split.
    """
    import torch

    for module in _modules(shard):
        for param in module.parameters(recurse=True):
            param.requires_grad_(False)

    adapters: Dict[str, object] = {}
    wanted = tuple(targets)
    for index, layer in enumerate(shard.layers):
        for name, child in list(layer.named_modules()):
            if not name.endswith(wanted):
                continue
            if not isinstance(child, torch.nn.Linear):
                continue
            adapter = _lora_class()(child, rank=rank, alpha=alpha, dropout=dropout)
            _replace_child(layer, name, adapter)
            adapters[f"layers.{index}.{name}"] = adapter

    if not adapters:
        raise RuntimeError(
            f"no modules matched {list(wanted)} in this stage's layers; "
            f"nothing would be trained"
        )
    trainable = sum(p.numel() for a in adapters.values() for p in a.trainable())
    logger.info(
        "LoRA attached: %d adapters, %d trainable parameters (rank %d)",
        len(adapters),
        trainable,
        rank,
    )
    return adapters


def _modules(shard) -> List:
    return [m for m in (shard.embed, shard.layers, shard.norm, shard.lm_head) if m is not None]


def _replace_child(root, path: str, replacement) -> None:
    """Put `replacement` where `path` points, walking the dotted name."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = replacement
    else:
        setattr(parent, last, replacement)


def adapter_state(adapters: Dict[str, object], *, layer_offset: int = 0) -> Dict:
    """What this stage would save, in the names the whole model uses.

    The local layer index becomes the global one, so adapters trained on four
    machines concatenate into a single file that any inference runtime can
    load against the base model.
    """
    state = {}
    for name, adapter in adapters.items():
        state[f"{_globalise(name, layer_offset)}.lora_A.weight"] = adapter.lora_a.detach()
        state[f"{_globalise(name, layer_offset)}.lora_B.weight"] = adapter.lora_b.detach()
    return state


def _globalise(name: str, offset: int) -> str:
    match = re.match(r"^layers\.(\d+)\.(.+)$", name)
    if not match:
        return name
    return f"layers.{int(match.group(1)) + offset}.{match.group(2)}"
