"""Who runs a model, and which layers each of them runs.

Loom's default answer is "the broker decides": it weighs demand, price and free
VRAM and picks nodes on its own. That is what a marketplace wants and what
makes a measurement impossible — a benchmark comparing one node against two
against three needs the placement pinned by hand, identical between runs, and
changed without touching the workers.

So a model in the catalog is a model that *may* run, not one that is running.
Nothing is deployed until someone asks for it, in one of two ways:

  - `Placement.manual([...])` — exactly these nodes, exactly these layers, in
    this order. Stage 0 is the head (it owns the embeddings and accepts client
    requests), the last stage owns the LM head.
  - `Placement.auto()` — the broker's own choice, the behaviour Loom had
    before this existed.

Placements live in memory only, and an orchestrator restart is therefore
deliberately uneventful: it starts nothing. What it does do is adopt — a
pipeline the workers report as already running is taken back under management
as the placement it evidently is (see `MultiModelController._adopt_running_model`),
so a restart mid-experiment costs nothing and the operator can still take it
down. Nothing that was not already up comes up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# Backends that understand a layer range, i.e. can be ONE stage of a pipeline
# spread over several nodes. Everything else — `vllm`, `sglang`, `mlx` — loads
# a whole model and answers complete requests.
#
# The worker holds the same truth in loom_worker/backends (`serves_partial_shard`)
# and enforces it again on arrival; this copy exists so the orchestrator can
# refuse an impossible split immediately, instead of letting three nodes
# discover it separately after downloading a checkpoint each.
SHARDABLE_BACKENDS = frozenset({"shard", "vllm_shard", "mlx_shard", "echo"})

# Every backend a worker can be asked for. Validating the name here turns a
# typo into a message on the deploy form instead of a NACK from three nodes.
KNOWN_BACKENDS = frozenset(
    {"shard", "vllm_shard", "mlx_shard", "vllm", "sglang", "mlx", "echo"}
)


class PlacementError(ValueError):
    """A placement that cannot be honoured, with a message meant for a human."""


@dataclass(frozen=True)
class Stage:
    """One node's share of one model.

    `backend_type` is per stage, and empty means "whatever the deployment
    chose". Stages of one pipeline do not have to agree: an Apple node runs
    `mlx_shard` while a CUDA node next to it runs `vllm_shard`, because what a
    stage needs from its neighbours is the bytes of a hidden state, not their
    engine. The wire format carries the dtype with the tensor, so the two ends
    interoperate without knowing anything about each other.
    """

    node_id: str
    start_layer: int
    end_layer: int
    backend_type: str = ""

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "layers": self.num_layers,
            "backend_type": self.backend_type,
        }


@dataclass
class Placement:
    """Where one model runs. `stages` is empty exactly when mode is auto.

    `backend_type` overrides the catalog entry's own for this deployment. The
    catalog says what a model IS; how to run it is a property of the run — the
    same checkpoint is served whole by `vllm` and split by `shard`, and a
    measurement wants to switch between them without editing a catalog file
    and restarting the orchestrator.
    """

    model_id: str
    mode: str = "auto"  # "auto" | "manual"
    stages: List[Stage] = field(default_factory=list)
    backend_type: Optional[str] = None

    @classmethod
    def auto(cls, model_id: str, backend_type: Optional[str] = None) -> "Placement":
        return cls(model_id=model_id, mode="auto", backend_type=backend_type)

    @classmethod
    def manual(
        cls,
        model_id: str,
        stages: Sequence[Stage],
        backend_type: Optional[str] = None,
    ) -> "Placement":
        return cls(
            model_id=model_id,
            mode="manual",
            stages=list(stages),
            backend_type=backend_type,
        )

    @property
    def is_manual(self) -> bool:
        return self.mode == "manual"

    def node_ids(self) -> List[str]:
        return [stage.node_id for stage in self.stages]

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "mode": self.mode,
            "stages": [stage.as_dict() for stage in self.stages],
            "backend_type": self.backend_type,
        }


def stages_from_request(
    entries: Sequence[dict], *, num_model_layers: int
) -> List[Stage]:
    """Turn what the operator typed into a contiguous chain of stages.

    Two spellings are accepted, because both are natural to type:
      - `{"node_id": "n1", "layers": 20}` — counts, laid out in the given
        order. This is what the deploy form sends.
      - `{"node_id": "n1", "start_layer": 0, "end_layer": 20}` — explicit
        ranges, for a scripted run that wants no ambiguity at all.

    The order of `entries` is the order of the pipeline: the first entry is the
    head stage. That is a deliberate promise — on heterogeneous nodes it
    decides which card carries the embeddings and answers clients.
    """
    if not entries:
        raise PlacementError("pick at least one node")

    stages: List[Stage] = []
    cursor = 0
    for position, entry in enumerate(entries):
        node_id = str(entry.get("node_id") or "").strip()
        if not node_id:
            raise PlacementError(f"stage {position}: node_id is required")

        if entry.get("start_layer") is not None or entry.get("end_layer") is not None:
            try:
                start = int(entry["start_layer"])
                end = int(entry["end_layer"])
            except (KeyError, TypeError, ValueError):
                raise PlacementError(
                    f"stage {position} ({node_id}): give both start_layer and "
                    f"end_layer, or neither"
                ) from None
        else:
            try:
                count = int(entry.get("layers"))
            except (TypeError, ValueError):
                raise PlacementError(
                    f"stage {position} ({node_id}): 'layers' must be a number"
                ) from None
            start, end = cursor, cursor + count

        if end <= start:
            raise PlacementError(
                f"stage {position} ({node_id}): needs at least one layer, got {end - start}"
            )
        backend = str(entry.get("backend") or entry.get("backend_type") or "").strip()
        if backend and backend not in KNOWN_BACKENDS:
            raise PlacementError(
                f"stage {position} ({node_id}): unknown backend {backend!r}; "
                f"known: {', '.join(sorted(KNOWN_BACKENDS))}"
            )
        stages.append(
            Stage(
                node_id=node_id,
                start_layer=start,
                end_layer=end,
                backend_type=backend,
            )
        )
        cursor = end

    _validate_chain(stages, num_model_layers=num_model_layers)
    return stages


def _validate_chain(stages: Sequence[Stage], *, num_model_layers: int) -> None:
    """Every layer hosted exactly once, in order, on distinct nodes.

    These are not style rules. A gap means the pipeline computes a model that
    does not exist; an overlap means two nodes hold the same weights and the
    activations pass through them twice; a repeated node means one card is
    asked to be two stages of the same pipeline and to relay to itself.
    """
    seen: Dict[str, int] = {}
    for position, stage in enumerate(stages):
        if stage.node_id in seen:
            raise PlacementError(
                f"node {stage.node_id} appears twice (stages {seen[stage.node_id]} "
                f"and {position}); one node is one stage of a pipeline"
            )
        seen[stage.node_id] = position

    if stages[0].start_layer != 0:
        raise PlacementError(
            f"the first stage must start at layer 0, not {stages[0].start_layer}"
        )
    for previous, current in zip(stages, stages[1:]):
        if current.start_layer != previous.end_layer:
            gap = current.start_layer - previous.end_layer
            what = "a gap of" if gap > 0 else "an overlap of"
            raise PlacementError(
                f"{what} {abs(gap)} layers between {previous.node_id} "
                f"(ends at {previous.end_layer}) and {current.node_id} "
                f"(starts at {current.start_layer})"
            )
    total = stages[-1].end_layer
    if total != num_model_layers:
        short = num_model_layers - total
        raise PlacementError(
            f"the stages cover {total} of {num_model_layers} layers "
            f"({abs(short)} {'missing' if short > 0 else 'too many'}); "
            f"the numbers must add up to exactly {num_model_layers}"
        )


def check_stage_backends(stages: Sequence[Stage], default_backend: str) -> None:
    """Every stage's engine must be able to be one stage.

    Checked per stage rather than once for the deployment, because they can
    differ. A mixed pipeline is fine — an Apple node beside a CUDA one — but a
    whole-model engine anywhere in the chain is not: it would load the entire
    model and answer complete requests while pretending to be a link in it.
    """
    if len(stages) <= 1:
        return
    for position, stage in enumerate(stages):
        backend = stage.backend_type or default_backend
        if backend in SHARDABLE_BACKENDS:
            continue
        raise PlacementError(
            f"stage {position} ({stage.node_id}) would run backend "
            f"{backend!r}, which serves whole models only and cannot be one of "
            f"{len(stages)} pipeline stages. Choose 'shard', 'vllm_shard' or "
            f"'mlx_shard' for it"
        )


def check_backend_can_split(backend_type: str, num_stages: int) -> None:
    """Refuse a split the backend cannot perform, before anything downloads.

    This is the failure the stand hit: a catalog entry with `backend_type:
    "vllm"` was placed across three nodes. Two of them raised on arrival; the
    third was handed layers [0, 12) and would have been *fine* — `vllm serve`
    ignores the range and loads all 36 layers, so a third of a pipeline would
    have quietly answered complete requests. Catching it here means one clear
    message instead of three nodes each downloading a checkpoint to find out.
    """
    if num_stages <= 1 or backend_type in SHARDABLE_BACKENDS:
        return
    raise PlacementError(
        f"backend '{backend_type}' serves whole models only and cannot be "
        f"split across {num_stages} nodes. Choose 'shard' (transformers, runs "
        f"anywhere) or 'vllm_shard' (vLLM, CUDA only) for this deployment, or "
        f"place the model on a single node"
    )


def is_complete_pipeline(stages: Sequence[Stage], *, num_model_layers: int) -> bool:
    """Do these stages tile the whole model, in order, on distinct nodes?

    Used when adopting stages found already running on the workers: until every
    stage has reported in, what we have is a fragment, not a pipeline.
    """
    if not stages:
        return False
    try:
        _validate_chain(stages, num_model_layers=num_model_layers)
    except PlacementError:
        return False
    return True


def even_split(node_ids: Sequence[str], num_model_layers: int) -> List[Stage]:
    """The default the deploy form starts from: as equal as the layers allow.

    Not a recommendation — on nodes of different speed an even split is the
    wrong answer (see docs/VLLM_PIPELINE.md §5c) — but it is the neutral
    starting point for a measurement, and the operator edits from there.
    """
    if not node_ids:
        return []
    count = len(node_ids)
    base, remainder = divmod(num_model_layers, count)
    stages: List[Stage] = []
    cursor = 0
    for index, node_id in enumerate(node_ids):
        size = base + (1 if index < remainder else 0)
        stages.append(Stage(node_id=node_id, start_layer=cursor, end_layer=cursor + size))
        cursor += size
    return stages


def quota_for_layers(
    *,
    num_layers: int,
    is_first: bool,
    is_last: bool,
    per_layer_param_bytes: int,
    embedding_param_bytes: int,
    tie_embedding: bool,
    param_mem_ratio: float,
) -> int:
    """VRAM to grant a hand-placed stage, in the units capacity math uses.

    The broker normally sizes this. A manual placement skips the broker, but
    the worker still needs a quota: it is what the watchdog enforces and what
    the vLLM engine budgets its KV cache from. Inverting the capacity formula
    (`available = quota x param_mem_ratio`) keeps a hand-placed stage and a
    brokered one asking for memory on the same terms.
    """
    endpoints = 0
    if is_first:
        endpoints += embedding_param_bytes
    if is_last and not tie_embedding:
        endpoints += embedding_param_bytes
    params = num_layers * per_layer_param_bytes + endpoints
    ratio = param_mem_ratio if param_mem_ratio > 0 else 1.0
    return int(params / ratio)


def fit_report(
    stages: Sequence[Stage],
    *,
    node_vram: Dict[str, int],
    per_layer_param_bytes: int,
    embedding_param_bytes: int,
    tie_embedding: bool,
    param_mem_ratio: float,
) -> List[str]:
    """Complaints about stages whose weights will not fit on their node.

    Returned rather than raised so the caller decides: refusing outright is
    right by default, but an operator who knows the estimate is conservative
    can override it. Silence would be the wrong choice — the alternative is an
    out-of-memory crash minutes later, after a full checkpoint download.
    """
    problems: List[str] = []
    for index, stage in enumerate(stages):
        available = node_vram.get(stage.node_id)
        if not available:
            continue  # unknown node: the caller reports that separately
        need = quota_for_layers(
            num_layers=stage.num_layers,
            is_first=index == 0,
            is_last=index == len(stages) - 1,
            per_layer_param_bytes=per_layer_param_bytes,
            embedding_param_bytes=embedding_param_bytes,
            tie_embedding=tie_embedding,
            param_mem_ratio=param_mem_ratio,
        )
        if need > available:
            gib = 1024**3
            problems.append(
                f"{stage.node_id}: {stage.num_layers} layers need about "
                f"{need / gib:.1f} GB but the node reports {available / gib:.1f} GB free"
            )
    return problems


def max_layers_for(
    vram_bytes: int,
    *,
    per_layer_param_bytes: int,
    embedding_param_bytes: int,
    tie_embedding: bool,
    param_mem_ratio: float,
    with_endpoints: bool = False,
) -> int:
    """How many layers of this model a node could hold — the form's hint."""
    if per_layer_param_bytes <= 0:
        return 0
    budget = vram_bytes * param_mem_ratio
    if with_endpoints:
        budget -= embedding_param_bytes * (1 if tie_embedding else 2)
    return max(0, int(budget // per_layer_param_bytes))


def describe(placement: Optional[Placement]) -> str:
    """One line for logs: what is running where."""
    if placement is None:
        return "not deployed"
    if not placement.is_manual:
        return "auto (broker chooses)"
    return " -> ".join(
        f"{stage.node_id}[{stage.start_layer}:{stage.end_layer}]"
        for stage in placement.stages
    )


def stages_as_allocation(placement: Placement) -> List[Tuple[str, int, int]]:
    """The (node, start, end) triples the pipeline builder speaks."""
    return [(s.node_id, s.start_layer, s.end_layer) for s in placement.stages]
