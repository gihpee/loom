"""What stages say to each other while training, and who says it when.

Deliberately a separate module from the inference stage server. The two share
a transport — the orchestrator routes by `pipeline_id` and `target_stage` and
never looks at the kind of message — but they share nothing else, and mixing
them would put a training branch inside the loop that answers user requests.

The exchange for one micro-batch, on a three-stage pipeline:

    stage 0   fwd ->
    stage 1              fwd ->
    stage 2                       forward, loss, backward
    stage 1              <- bwd
    stage 0   <- bwd
    (stage 0 runs its own backward; the micro-batch is done)

and once every micro-batch of a step has come home, stage 0 broadcasts `step`
and every stage applies its own optimiser. Nothing is synchronised between
stages because no two of them hold the same parameter.

Failure is part of the protocol, not an afterthought. A stage that raises
sends `train_error` home; a micro-batch whose gradient never returns is
abandoned by timeout. Either way stage 0 broadcasts `abort`, every stage drops
what it was holding for that step, and the step is retried from the last
checkpoint — which is the only reason a fleet of other people's machines can
finish a run at all.
"""

from __future__ import annotations

# Message kinds. Values are on the wire, so they are part of the contract.
FWD = "train_fwd"          # activations moving forward, plus the labels' owner
BWD = "train_bwd"          # gradient of the loss w.r.t. what the peer sent us
STEP = "train_step"        # broadcast: apply the optimiser
ABORT = "train_abort"      # broadcast: drop everything held for this step
ERROR = "train_error"      # a stage failed; carries which one and why
LOSS = "train_loss"        # the last stage tells stage 0 what it measured
SAVE = "train_save"        # broadcast: write a checkpoint of your own slice

ALL_KINDS = (FWD, BWD, STEP, ABORT, ERROR, LOSS, SAVE)


def is_training(kind: str) -> bool:
    """Does this message belong to training rather than inference?

    Used at the one place the two paths meet — the agent's inbox — so that a
    training message can never be handed to the inference stage, and the
    reverse. The prefix makes that decidable without a table.
    """
    return isinstance(kind, str) and kind.startswith("train_")
