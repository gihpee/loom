"""One stage of a pipeline that trains instead of only answering.

Inference needs a stage to run its layers and forget. Training needs it to run
its layers, REMEMBER what it computed, and later — when the gradient of the
loss arrives from the stage after it — push that gradient back through the
same computation and hand what comes out to the stage before it.

    forward :  h_in  -> h_out        and keep the graph that connects them
    backward:  dh_out -> dh_in       by differentiating that kept graph

The chain of these is exactly backpropagation, cut into pieces that live on
different machines. Nothing about the mathematics changes; what changes is
that a piece of the graph is held on each node until its gradient arrives.

Why this is a better fit for a spread-out fleet than inference is. Decoding
one sequence uses one stage at a time — the others wait, and every hop is paid
in full at every token. Training runs several micro-batches at once: while
stage 3 is going backward over micro-batch 1, stage 0 is already going forward
over micro-batch 3. Every stage has work, and the hop is paid once per
micro-batch rather than once per token.

What is deliberately not here: training the base weights. Their gradients and
optimiser moments would need roughly eight times the memory the weights
themselves take, which the cards this runs on do not have. Adapters are the
whole point — see lora.py.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loom_worker.train.lora import (
    _globalise,
    adapter_state,
    attach_lora,
    train_everything,
)

logger = logging.getLogger("loom_worker.train.stage")


@dataclass
class MicroBatch:
    """What a stage keeps between its forward and the gradient coming back.

    Held per micro-batch, not per step: several are in flight at once, and
    their gradients arrive in an order the stage does not choose.
    """

    inputs: object          # what came in, with requires_grad where it matters
    outputs: object         # what went out — the graph hangs off this
    loss: Optional[object] = None   # last stage only
    tokens: int = 0


class TrainingStage:
    """Forward that remembers, backward that answers, and a step that applies.

    One object per model per worker. Thread safety is the same rule the
    inference executor follows: one caller inside the model at a time, because
    autograd's graph is process-wide state and two backward passes through the
    same modules at once corrupt it.
    """

    def __init__(
        self,
        shard,
        spec,
        *,
        mode: str = "lora",
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        lr: float = 1e-4,
        grad_clip: float = 1.0,
    ) -> None:
        import torch

        self.torch = torch
        self.shard = shard
        self.spec = spec
        self.mode = mode
        if mode == "lora":
            self.adapters: Dict[str, object] = attach_lora(
                shard, rank=rank, alpha=alpha, dropout=dropout
            )
            params = [p for a in self.adapters.values() for p in a.trainable()]
        elif mode == "full":
            self.adapters = {}
            params = train_everything(shard)
        else:
            raise ValueError(f"unknown training mode {mode!r}; use 'lora' or 'full'")
        # AdamW on the adapters only. Its moments are two tensors per trainable
        # parameter, which is affordable precisely because the base weights are
        # not among them.
        self.optimizer = torch.optim.AdamW(params, lr=lr)
        self.params = params
        self.grad_clip = grad_clip
        self._pending: Dict[str, MicroBatch] = {}
        self._lock = threading.RLock()
        self.steps = 0

    # ------------------------------------------------------------- forward
    def forward(self, *, micro_id: str, input_ids=None, hidden=None, labels=None):
        """Run this stage and keep the graph.

        Returns (hidden_out, loss) — the loss only on the last stage, where the
        labels are. Everything else returns hidden states for the next stage,
        detached from the graph on purpose: the graph must not cross the
        network, only the numbers do. Each stage differentiates its own piece
        and hands the result on, which is what makes the pieces independent.
        """
        torch = self.torch
        with self._lock:
            if self.spec.is_first:
                if input_ids is None:
                    raise ValueError("the first stage was given no token ids")
                ids = torch.as_tensor(input_ids, device=self.shard.device)
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                h = self.shard.embed(ids)
                entry = None
            else:
                if hidden is None:
                    raise ValueError("a later stage was given no hidden states")
                # The incoming activations are a leaf of THIS stage's graph,
                # and the gradient that collects on them is what the previous
                # stage is waiting for.
                entry = hidden.to(
                    device=self.shard.device, dtype=self.shard.dtype
                ).requires_grad_(True)
                h = entry

            h = self._run_layers(h)

            loss = None
            if self.spec.is_last:
                h = self.shard.norm(h)
                logits = self.shard.lm_head(h)
                if labels is not None:
                    loss = self._loss(logits, labels)

            self._pending[micro_id] = MicroBatch(
                inputs=entry,
                outputs=h if not self.spec.is_last else logits,
                loss=loss,
                tokens=int(getattr(labels, "numel", lambda: 0)()) if labels is not None else 0,
            )
            return (None if self.spec.is_last else h.detach()), loss

    def _run_layers(self, h):
        torch = self.torch
        seq = h.shape[1]
        positions = torch.arange(seq, device=h.device).unsqueeze(0)
        rotary = self.shard.rotary
        embeddings = rotary(h, positions) if rotary is not None else None
        kwargs = {
            "attention_mask": None,   # causal by default for a single sequence
            "position_ids": positions,
            "use_cache": False,       # no KV cache in training: the whole
                                      # sequence is present at once
            "position_embeddings": embeddings,
        }
        for layer in self.shard.layers:
            out = layer(h, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
        return h

    def _loss(self, logits, labels):
        """Next-token cross entropy, shifted the way causal models define it."""
        torch = self.torch
        labels = torch.as_tensor(labels, device=logits.device)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        predicted = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
        target = labels[:, 1:].reshape(-1)
        return torch.nn.functional.cross_entropy(predicted, target, ignore_index=-100)

    # ------------------------------------------------------------ backward
    def backward(self, *, micro_id: str, grad_output=None):
        """Push a gradient back through this stage's kept graph.

        Returns the gradient with respect to what came IN, which is what the
        stage before this one needs — or None on the first stage, where there
        is nobody left to tell.

        The kept graph is released here whether or not anything goes wrong: a
        micro-batch whose gradient never arrives would otherwise hold its
        activations for the rest of the run, and activations are the largest
        thing a training step touches.
        """
        torch = self.torch
        with self._lock:
            batch = self._pending.pop(micro_id, None)
            if batch is None:
                raise KeyError(f"no forward pass is waiting for gradient {micro_id!r}")
            try:
                if self.spec.is_last:
                    if batch.loss is None:
                        raise ValueError("the last stage has no loss to differentiate")
                    batch.loss.backward()
                else:
                    grad = grad_output.to(
                        device=batch.outputs.device, dtype=batch.outputs.dtype
                    )
                    batch.outputs.backward(gradient=grad)
                if batch.inputs is None:
                    return None
                return batch.inputs.grad.detach()
            finally:
                batch.inputs = batch.outputs = batch.loss = None

    # ---------------------------------------------------------------- step
    def step(self) -> dict:
        """Apply what the accumulated gradients say, and start over.

        Called once per optimiser step, after every micro-batch of it has been
        through backward. Each stage steps its own adapters: there is nothing
        to synchronise between stages, because no two of them hold the same
        parameter.
        """
        with self._lock:
            grad_norm = float(
                self.torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.steps += 1
            return {"step": self.steps, "grad_norm": grad_norm}

    def drop_all(self) -> int:
        """Let go of every micro-batch being held, and say how many.

        Called when a step is abandoned. Activations are the largest thing a
        step touches, and a stage that keeps them after the step they belong
        to has been given up will run out of memory long before the run ends.
        """
        with self._lock:
            count = len(self._pending)
            self._pending.clear()
            return count

    def zero_grad(self) -> None:
        """Discard gradients accumulated by a step that will not be applied.

        Half a step's gradients cannot be subtracted, so a retried step has to
        start from nothing. Skipping this would fold the abandoned attempt
        into the next one — silently, and with a weight nobody chose.
        """
        with self._lock:
            self.optimizer.zero_grad(set_to_none=True)

    def load_state(self, state: Dict) -> None:
        """Put a saved slice back into this stage."""
        torch = self.torch
        with self._lock, torch.no_grad():
            if self.mode == "lora":
                for name, adapter in self.adapters.items():
                    key = _globalise(name, self.spec.start_layer)
                    a, b = f"{key}.lora_A.weight", f"{key}.lora_B.weight"
                    if a in state:
                        adapter.lora_a.copy_(state[a].to(adapter.lora_a.device))
                    if b in state:
                        adapter.lora_b.copy_(state[b].to(adapter.lora_b.device))
                return
            offset = self.spec.start_layer
            for index, layer in enumerate(self.shard.layers):
                for name, param in layer.named_parameters(recurse=True):
                    key = f"model.layers.{index + offset}.{name}"
                    if key in state:
                        param.copy_(state[key].to(param.device))

    # ------------------------------------------------------------- the wire
    def serialize(self, tensor) -> Tuple[bytes, List[int], str]:
        """Hand a tensor to the transport, in the format inference already uses."""
        from loom_worker.wire import to_wire

        return to_wire(self.torch, tensor.detach())

    def in_flight(self) -> int:
        with self._lock:
            return len(self._pending)

    # ----------------------------------------------------------- the result
    def trainable_bytes(self) -> int:
        """What this stage's training state costs beyond the weights.

        Gradient plus two optimiser moments per trainable parameter. Reported
        so a node can refuse a job it cannot hold, instead of discovering the
        fact as an out-of-memory crash in the middle of a run.
        """
        per_param = sum(p.numel() * p.element_size() for p in self.params)
        return per_param * 3

    def state_dict(self) -> Dict:
        """What this stage would save, in the names the whole model uses.

        For LoRA that is the adapters. For full fine-tuning it is this
        stage's slice of the weights — the same names the checkpoint uses, so
        the pieces from every stage concatenate back into one model.
        """
        if self.mode == "lora":
            return adapter_state(self.adapters, layer_offset=self.spec.start_layer)
        state = {}
        offset = self.spec.start_layer
        for index, layer in enumerate(self.shard.layers):
            for name, param in layer.named_parameters(recurse=True):
                state[f"model.layers.{index + offset}.{name}"] = param.detach()
        if self.spec.is_first and self.shard.embed is not None:
            state["model.embed_tokens.weight"] = self.shard.embed.weight.detach()
        if self.spec.is_last:
            if self.shard.norm is not None:
                state["model.norm.weight"] = self.shard.norm.weight.detach()
            if self.shard.lm_head is not None:
                state["lm_head.weight"] = self.shard.lm_head.weight.detach()
        return state

    # kept under the old name for the code that already calls it
    def adapter_state(self) -> Dict:
        return self.state_dict()
