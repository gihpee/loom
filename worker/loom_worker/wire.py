"""What activations look like on the wire between two stages.

Both engines — the transformers `shard` and the vLLM one — hand tensors to the
same transport, so the format lives here rather than twice.

The rule is one line long: **never widen a tensor to send it.** A stage running
a bfloat16 model produces bfloat16 hidden states; converting them to float32
for the trip doubles every byte and adds no information, because the receiving
stage immediately narrows them back to compute with. That upcast was there for
portability — a float32 CPU stage and a bfloat16 GPU stage had to understand
each other — but portability comes from the dtype travelling WITH the tensor,
which it already does, not from forcing one dtype on everybody.

Narrowing is the case worth being careful about, and it is why the default is
"whatever the tensor already is" rather than "always bfloat16": a stage that
genuinely computes in float32 (any CPU stage) would lose precision if its
output were squeezed on the way out.

What this is worth, per activation message of a 4096-wide model:

    float32   16 KB
    bfloat16   8 KB

On a three-stage pipeline that is 96 KB per token relayed through the
orchestrator, or 24 KB with bfloat16 and direct links — four times less traffic
for the same tokens.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

logger = logging.getLogger("loom_worker.wire")

# Names are exchanged as strings, so the mapping is the contract between two
# processes that may be running different builds. Adding a name here is a
# protocol change: a stage that does not know it must refuse the message
# (see from_wire) rather than guess and decode noise.
_WIRE_NAMES = ("float32", "float16", "bfloat16")

# "auto" keeps the tensor's own dtype. Anything else forces that dtype on every
# outgoing message — useful for measuring what the format costs, and for
# pinning a pipeline whose stages disagree.
WIRE_DTYPE = os.environ.get("LOOM_WIRE_DTYPE", "auto").strip().lower()


class WireFormatError(RuntimeError):
    """A message this stage cannot decode. Never decode it as something else."""


def torch_dtypes(torch):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }


def wire_dtype_for(torch, tensor):
    """The dtype to send `tensor` in.

    Its own, unless an operator pinned one. Widening is never chosen
    automatically: it costs bytes and buys nothing.
    """
    if WIRE_DTYPE in _WIRE_NAMES:
        return torch_dtypes(torch)[WIRE_DTYPE]
    dtype = tensor.dtype
    # Anything exotic (float64, an integer type) goes as float32: the wire
    # vocabulary is deliberately small, and these do not occur in hidden states.
    return dtype if _name_of(torch, dtype) else torch.float32


def _name_of(torch, dtype) -> str:
    for name, candidate in torch_dtypes(torch).items():
        if dtype == candidate:
            return name
    return ""


def to_wire(torch, tensor) -> Tuple[bytes, List[int], str]:
    """Tensor -> (bytes, shape, dtype name), ready for the transport."""
    target = wire_dtype_for(torch, tensor)
    flat = tensor.detach().to("cpu", dtype=target).contiguous()
    # Read through uint8 rather than calling .numpy(): numpy has no bfloat16
    # and would refuse. The bytes are the same bytes either way — the reader
    # puts them back with torch, which does know the type.
    raw = flat.view(torch.uint8).numpy().tobytes()
    return raw, list(flat.shape), _name_of(torch, target)


def from_wire(torch, data: bytes, shape: List[int], dtype: str):
    """(bytes, shape, dtype name) -> tensor, exactly as it was sent.

    An unknown dtype name is refused, not guessed. Guessing is how a mixed-
    version pipeline turns into an inference that runs happily on misread bytes
    — grammatical, confident and wrong, which is the worst failure this system
    can produce.
    """
    name = (dtype or "").strip()
    mapping = torch_dtypes(torch)
    if name not in mapping:
        raise WireFormatError(
            f"a peer sent activations as {name!r}, which this stage does not "
            f"know how to read (it understands {', '.join(sorted(mapping))}). "
            f"The stages of a pipeline must run the same worker image"
        )
    flat = torch.frombuffer(bytearray(data), dtype=mapping[name])
    return flat.reshape(tuple(shape))
