"""Driving a training run across stages, and surviving the parts that fail.

This is the half of training that is not mathematics. The stage knows how to
go forward and backward; the runner decides what to send, what to wait for,
what to do when it does not arrive, and what to write down so that a run
interrupted at hour three does not start again at hour zero.

Two objects:

  StageRunner   on every stage — receives protocol messages, calls the stage,
                sends the next message on. Knows nothing about steps or data.
  StepDriver    on stage 0 only — turns a batch of samples into micro-batches,
                pushes them into the chain, waits, retries, and checkpoints.

Splitting them this way keeps the failure handling in one place. Every stage
can report a failure, but only stage 0 decides what a failure means, because
only stage 0 knows how many micro-batches are outstanding and how many
attempts this step has already had.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from loom_worker.train import protocol

logger = logging.getLogger("loom_worker.train.runner")

# How long stage 0 waits for a micro-batch to come home before calling it lost.
# Generous on purpose: a slow stage is not a broken one, and abandoning a step
# costs every micro-batch already in it.
MICRO_TIMEOUT_S = float(os.environ.get("LOOM_TRAIN_MICRO_TIMEOUT_S", "300"))

# How many times one step is retried before the run gives up. A step that
# fails three times in a row is not unlucky — something is wrong that waiting
# will not fix.
MAX_STEP_ATTEMPTS = int(os.environ.get("LOOM_TRAIN_STEP_ATTEMPTS", "3"))


class StageRunner:
    """The protocol as it looks from one stage.

    `send` takes (target_stage, message) and is whatever the agent gave us —
    the same relay the inference stage uses, so training rides the transport
    that already exists rather than a second one.
    """

    def __init__(self, stage, *, send: Callable[[int, dict], None], stage_index: int,
                 num_stages: int,
                 on_micro_done: Optional[Callable[[str], None]] = None) -> None:
        self.stage = stage
        self.send = send
        self.index = stage_index
        self.num_stages = num_stages
        # Stage 0 only: what to call when a micro-batch has come all the way
        # home. Nobody else can know that it has.
        self.on_micro_done = on_micro_done
        self._lock = threading.RLock()

    # ------------------------------------------------------------ receiving
    def handle(self, msg: dict) -> None:
        kind = msg.get("kind")
        try:
            if kind == protocol.FWD:
                self._on_forward(msg)
            elif kind == protocol.BWD:
                self._on_backward(msg)
            elif kind == protocol.STEP:
                self.stage.step()
            elif kind == protocol.ABORT:
                self._on_abort(msg)
            elif kind == protocol.SAVE:
                self._on_save(msg)
            else:
                logger.warning("unknown training message %r", kind)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            logger.exception("training stage failed on %s", kind)
            self._report_failure(msg, exc)

    def _on_forward(self, msg: dict) -> None:
        micro = msg["micro_id"]
        hidden = _from_wire(self.stage.torch, msg) if not self.stage.spec.is_first else None
        out, loss = self.stage.forward(
            micro_id=micro,
            input_ids=msg.get("input_ids"),
            hidden=hidden,
            labels=msg.get("labels"),
        )
        if not self.stage.spec.is_last:
            self.send(self.index + 1, {
                "kind": protocol.FWD,
                "micro_id": micro,
                "step": msg.get("step"),
                "labels": msg.get("labels"),
                **_to_wire(self.stage, out),
            })
            return

        # The tail: it has the labels, so it has the loss. Report it home and
        # turn straight around into the backward pass — there is nobody after
        # it to wait for.
        self.send(0, {
            "kind": protocol.LOSS,
            "micro_id": micro,
            "step": msg.get("step"),
            "loss": float(loss.detach()),
        })
        grad = self.stage.backward(micro_id=micro)
        self._pass_back(micro, msg.get("step"), grad)

    def _on_backward(self, msg: dict) -> None:
        micro = msg["micro_id"]
        grad = _from_wire(self.stage.torch, msg)
        out = self.stage.backward(micro_id=micro, grad_output=grad)
        self._pass_back(micro, msg.get("step"), out)

    def _pass_back(self, micro: str, step, grad) -> None:
        if self.index == 0:
            # Nobody left to tell — which is precisely what "complete" means.
            if self.on_micro_done is not None:
                self.on_micro_done(micro)
            return
        self.send(self.index - 1, {
            "kind": protocol.BWD,
            "micro_id": micro,
            "step": step,
            **_to_wire(self.stage, grad),
        })

    def _on_abort(self, msg: dict) -> None:
        dropped = self.stage.drop_all()
        # And the gradients they already contributed. Half a step cannot be
        # subtracted, so a retried step has to start from nothing; keeping
        # them would fold the abandoned attempt into the next step with a
        # weight nobody chose.
        self.stage.zero_grad()
        if dropped:
            logger.info("abort: dropped %d micro-batches held for this step", dropped)

    def _on_save(self, msg: dict) -> None:
        path = msg.get("path") or ""
        written = save_shard(self.stage, path, step=int(msg.get("step") or 0))
        self.send(0, {
            "kind": protocol.LOSS,   # reuse the report channel; step 0 counts them
            "micro_id": f"saved:{self.index}",
            "step": msg.get("step"),
            "loss": 0.0,
            "saved": written,
        })

    def _report_failure(self, msg: dict, exc: BaseException) -> None:
        try:
            self.send(0, {
                "kind": protocol.ERROR,
                "micro_id": msg.get("micro_id", ""),
                "step": msg.get("step"),
                "stage": self.index,
                "error": f"stage {self.index}/{self.num_stages}: "
                         f"{type(exc).__name__}: {exc}",
            })
        except Exception:
            logger.exception("could not even report the failure home")


@dataclass
class StepResult:
    step: int
    loss: float
    micro_batches: int
    attempts: int
    seconds: float


class StepDriver:
    """Stage 0's view: micro-batches, retries and checkpoints.

    Holds no model state of its own. Everything it knows about progress is the
    step number and the losses reported back, which is exactly what has to
    survive a restart.
    """

    def __init__(
        self,
        runner: StageRunner,
        *,
        micro_batches: int = 4,
        checkpoint_dir: str = "",
        checkpoint_every: int = 50,
    ) -> None:
        self.runner = runner
        self.micro_batches = max(1, micro_batches)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = max(0, checkpoint_every)
        self.step_index = 0
        self._returned: "queue.Queue[dict]" = queue.Queue()
        self._losses: Dict[str, float] = {}
        self._failure: Optional[str] = None
        self.history: List[StepResult] = []

    # --------------------------------------------------- messages coming home
    def note(self, msg: dict) -> None:
        """Called by the stage server for LOSS / ERROR arriving at stage 0."""
        kind = msg.get("kind")
        if kind == protocol.LOSS:
            self._losses[msg.get("micro_id", "")] = float(msg.get("loss") or 0.0)
        elif kind == protocol.ERROR:
            self._failure = str(msg.get("error") or "a stage failed")
            self._returned.put({"failed": True})

    def micro_done(self, micro_id: str) -> None:
        """Stage 0 finished its own backward: this micro-batch is complete."""
        self._returned.put({"micro_id": micro_id})

    # ------------------------------------------------------------- the step
    def run_step(self, samples: List[dict]) -> StepResult:
        """One optimiser step over `samples`, retried if a stage lets go.

        Retries re-run the whole step rather than the lost micro-batch alone.
        Half a step's gradients are already accumulated when it fails, and
        there is no way to subtract them — so the honest recovery is to drop
        them and start the step again.
        """
        last_error = None
        for attempt in range(1, MAX_STEP_ATTEMPTS + 1):
            started = time.monotonic()
            try:
                loss = self._attempt(samples)
            except TrainingStepFailed as exc:
                last_error = str(exc)
                logger.warning(
                    "step %d attempt %d failed (%s); dropping what it accumulated "
                    "and retrying", self.step_index + 1, attempt, exc
                )
                self._abort_everywhere()
                continue
            self.step_index += 1
            # Broadcast only. A broadcast reaches every stage of the pipeline
            # INCLUDING this one, so stepping here as well applied the
            # optimiser twice on stage 0 — the head learning at double the
            # rate of every other stage, silently.
            self.runner.send(-1, {"kind": protocol.STEP, "step": self.step_index})
            result = StepResult(
                step=self.step_index,
                loss=loss,
                micro_batches=len(samples),
                attempts=attempt,
                seconds=time.monotonic() - started,
            )
            self.history.append(result)
            self._maybe_checkpoint()
            return result
        raise TrainingStepFailed(
            f"step {self.step_index + 1} failed {MAX_STEP_ATTEMPTS} times; "
            f"last error: {last_error}"
        )

    def _attempt(self, samples: List[dict]) -> float:
        self._losses.clear()
        self._failure = None
        while not self._returned.empty():
            self._returned.get_nowait()

        outstanding = []
        for i, sample in enumerate(samples):
            micro = f"s{self.step_index + 1}m{i}"
            outstanding.append(micro)
            self.runner._on_forward({
                "kind": protocol.FWD,
                "micro_id": micro,
                "step": self.step_index + 1,
                "input_ids": sample["input_ids"],
                "labels": sample.get("labels", sample["input_ids"]),
            })

        deadline = time.monotonic() + MICRO_TIMEOUT_S
        done = set()
        while len(done) < len(outstanding):
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise TrainingStepFailed(
                    f"{len(outstanding) - len(done)} of {len(outstanding)} "
                    f"micro-batches never came back within {MICRO_TIMEOUT_S:.0f}s"
                )
            try:
                event = self._returned.get(timeout=timeout)
            except queue.Empty:
                continue
            if event.get("failed"):
                raise TrainingStepFailed(self._failure or "a stage failed")
            done.add(event["micro_id"])

        measured = [v for k, v in self._losses.items() if not k.startswith("saved:")]
        return sum(measured) / len(measured) if measured else 0.0

    def _abort_everywhere(self) -> None:
        # Broadcast only, for the same reason as the step above.
        self.runner.send(-1, {"kind": protocol.ABORT, "step": self.step_index + 1})

    # ------------------------------------------------------- staying alive
    def _maybe_checkpoint(self) -> None:
        if not self.checkpoint_dir or not self.checkpoint_every:
            return
        if self.step_index % self.checkpoint_every:
            return
        self.checkpoint()

    def checkpoint(self) -> str:
        """Write every stage's slice, and stage 0's record of where we are.

        Each stage writes its own file: the pieces are on different machines
        and collecting them through the orchestrator on every checkpoint would
        move gigabytes across the fleet for no reason. They are gathered once,
        at the end, by whoever asked for the run.
        """
        path = os.path.join(self.checkpoint_dir, f"step-{self.step_index}")
        self.runner.send(-1, {
            "kind": protocol.SAVE, "step": self.step_index, "path": path
        })
        written = save_shard(self.runner.stage, path, step=self.step_index)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        with open(os.path.join(self.checkpoint_dir, "progress.json"), "w") as handle:
            json.dump({"step": self.step_index, "path": path}, handle)
        logger.info("checkpoint at step %d -> %s", self.step_index, path)
        return written

    def resume(self) -> int:
        """Pick up where a previous run stopped, if it left anything.

        Returns the step it resumed at. Stage 0 holds the record because it is
        the only stage that knows a step happened at all.
        """
        marker = os.path.join(self.checkpoint_dir or "", "progress.json")
        if not self.checkpoint_dir or not os.path.exists(marker):
            return 0
        with open(marker) as handle:
            state = json.load(handle)
        self.step_index = int(state.get("step") or 0)
        logger.info("resuming at step %d", self.step_index)
        return self.step_index


class TrainingStepFailed(RuntimeError):
    """A step could not be completed and its gradients were discarded."""


# ----------------------------------------------------------------- storage
def save_shard(stage, path: str, *, step: int) -> str:
    """This stage's slice of the trained weights, on this machine's disk."""
    import torch

    os.makedirs(path, exist_ok=True)
    name = f"stage-{stage.spec.start_layer}-{stage.spec.end_layer}.pt"
    target = os.path.join(path, name)
    torch.save(
        {
            "step": step,
            "mode": stage.mode,
            "layers": [stage.spec.start_layer, stage.spec.end_layer],
            "state": stage.state_dict(),
        },
        target,
    )
    return target


def load_shard(stage, path: str) -> int:
    """Put a saved slice back, and say which step it came from."""
    import torch

    import glob

    name = f"stage-{stage.spec.start_layer}-{stage.spec.end_layer}.pt"
    target = os.path.join(path, name)
    if not os.path.exists(target):
        others = sorted(os.path.basename(p) for p in glob.glob(os.path.join(path, "stage-*.pt")))
        if others:
            # Someone else's slices are here but not ours. Loading nothing and
            # carrying on would restart this stage from the base weights while
            # every other stage resumed — a run that looks resumed and is not.
            raise ValueError(
                f"checkpoint at {path} was written by a pipeline laid out "
                f"differently: it holds {others}, and this stage needs {name!r}"
            )
        return 0
    saved = torch.load(target, map_location="cpu", weights_only=False)
    if saved.get("layers") != [stage.spec.start_layer, stage.spec.end_layer]:
        raise ValueError(
            f"checkpoint holds layers {saved.get('layers')} but this stage has "
            f"[{stage.spec.start_layer}, {stage.spec.end_layer}); the pipeline "
            f"was laid out differently when it was written"
        )
    stage.load_state(saved["state"])
    return int(saved.get("step") or 0)


# -------------------------------------------------------------- the wire
def _to_wire(stage, tensor) -> dict:
    data, shape, dtype = stage.serialize(tensor)
    return {"tensor_b64": data, "shape": shape, "dtype": dtype}


def _from_wire(torch, msg: dict):
    from loom_worker.wire import from_wire

    return from_wire(torch, msg["tensor_b64"], msg["shape"], msg["dtype"])
