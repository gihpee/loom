"""Renting the fleet for work that is not a language model.

The hard part is not running a container — it is deciding which machines run
it, and saying something useful when none can. Loom cannot look inside a
client's container the way it looks inside a model, so it cannot split the
work. What it can do is place tasks, and the two ways of doing that mean
different things:

  array   independent tasks spread over as many nodes as it takes
  gang    one computation on several machines at once, coordinating itself

Everything below pins down that distinction and the packing that serves it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loom.orchestrator.jobs import (  # noqa: E402
    GIB,
    JobError,
    Resources,
    gang_env,
    plan,
)


def fleet(**nodes):
    """{"a": 24, "b": 8} -> nodes with that many GB of free VRAM."""
    return {
        name: {
            "vram_free_bytes": int(gb * GIB),
            "ram_bytes": 64 * GIB,
            "cpus": 16.0,
            "num_gpus": 1,
        }
        for name, gb in nodes.items()
    }


def small(**overrides):
    raw = {"vram_gb": 8, "ram_gb": 4, "cpus": 2}
    raw.update(overrides)
    return Resources.from_request(raw)


def submit(nodes, **kwargs):
    kwargs.setdefault("image", "ghcr.io/acme/render:1")
    kwargs.setdefault("command", ["render", "--frame", "1"])
    kwargs.setdefault("resources", small())
    return plan(nodes=nodes, **kwargs)


# --------------------------------------------------------------- one machine
def test_a_task_runs_on_one_node():
    job = submit(fleet(a=24))
    assert [t.node_id for t in job.tasks] == ["a"]
    assert job.kind == "array" and job.state == "pending"


def test_tasks_spread_over_the_fleet_rather_than_filling_one_node():
    """Worst-fit on purpose.

    Best-fit would leave the fleet full of nodes with a little room each and
    nowhere to put the next large task — and these are not our machines to
    defragment afterwards.
    """
    job = submit(fleet(a=24, b=24, c=24), task_count=3)
    assert sorted(t.node_id for t in job.tasks) == ["a", "b", "c"]


def test_several_tasks_share_a_node_when_it_has_the_room():
    """Combining providers is for when one is not enough — not by default."""
    job = submit(fleet(big=24), task_count=3)
    assert [t.node_id for t in job.tasks] == ["big", "big", "big"]


def test_a_node_is_not_promised_more_than_it_has():
    job = submit(fleet(a=24), task_count=5)   # 5 x 8 GB against 24
    placed = [t.node_id for t in job.tasks if t.node_id]
    assert len(placed) == 3, "the node was oversubscribed"
    assert sum(1 for t in job.tasks if not t.node_id) == 2, (
        "an array job queues what does not fit; it does not fail"
    )


def test_what_other_jobs_already_hold_is_not_handed_out_again():
    job = submit(fleet(a=24), reserved={"a": small(vram_gb=20)})
    assert [t.node_id for t in job.tasks] == [""], "the same card went out twice"


# ------------------------------------------------------ several providers
def test_a_gang_job_takes_a_node_each():
    """One computation, several machines, started together."""
    job = submit(fleet(a=24, b=24, c=24), task_count=3, kind="gang")
    assert sorted(t.node_id for t in job.tasks) == ["a", "b", "c"]
    assert len(set(t.node_id for t in job.tasks)) == 3, "two ranks landed together"


def test_a_gang_job_is_refused_rather_than_half_placed():
    """Its tasks start together or not at all: half a gang is not a plan."""
    with pytest.raises(JobError, match="gang tasks start together"):
        submit(fleet(a=24, b=4), task_count=3, kind="gang")


def test_a_gang_task_is_told_where_its_peers_are():
    """Named the way torchrun and MPI already read, so a program written for
    them runs here unmodified. Loom allocates the group and gets out of the
    way."""
    job = submit(fleet(a=24, b=24), task_count=2, kind="gang")
    env = gang_env(job, job.tasks[1], {0: "10.0.0.1", 1: "10.0.0.2"})

    assert env["RANK"] == "1" and env["WORLD_SIZE"] == "2"
    assert env["MASTER_ADDR"] == "10.0.0.1"
    assert env["LOOM_PEERS"] == "10.0.0.1,10.0.0.2"


# ------------------------------------------------------- when it cannot run
def test_a_task_larger_than_any_machine_says_which_dimension_was_short():
    """"No node has room" sends someone to look at every machine.

    Naming what was short, and how much the roomiest node had, turns it into
    one decision: ask for less, or add a machine of that size.
    """
    with pytest.raises(JobError, match=r"VRAM \(asked 40.0 GB, the roomiest node has 24.0\)"):
        submit(fleet(a=24, b=16), resources=small(vram_gb=40))


def test_the_refusal_says_a_task_cannot_be_split():
    """The question everyone asks next, answered before they ask it."""
    with pytest.raises(JobError, match="A task runs on ONE machine"):
        submit(fleet(a=8), resources=small(vram_gb=40))


def test_an_empty_fleet_says_so_plainly():
    with pytest.raises(JobError, match="no nodes are connected"):
        submit({})


def test_a_job_without_an_image_is_refused():
    with pytest.raises(JobError, match="needs an image"):
        plan(image="", command=[], nodes=fleet(a=24), resources=small())


def test_an_unknown_kind_is_refused_rather_than_guessed():
    with pytest.raises(JobError, match="unknown job kind"):
        submit(fleet(a=24), kind="whatever")


# ------------------------------------------------------------- the accounting
def test_a_job_reports_itself_completely_enough_to_bill_from():
    job = submit(fleet(a=24), task_count=2)
    view = job.as_dict()
    assert view["image"] and view["command"]
    assert view["resources"]["vram_gb"] == 8.0
    assert len(view["tasks"]) == 2
    assert all("node_id" in t and "state" in t for t in view["tasks"])


# --------------------------------------------------- the node's own defences
def test_a_task_gets_no_route_to_the_owners_machine():
    """The node owner is the party that CAN be protected, so they are.

    Every flag here answers something a careless or hostile task would
    otherwise do to a machine it was only lent.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker"))
    from loom_worker.compute.runtime import TaskSpec, _docker_argv

    argv = _docker_argv(TaskSpec(task_id="t1", image="acme/x:1", ram_bytes=2 * GIB))
    joined = " ".join(argv)

    assert "--security-opt no-new-privileges" in joined
    assert "--cap-drop ALL" in joined
    assert "--network none" in joined, "a task gets the owner's network by default"
    assert "--memory" in joined and "--cpus" in joined
    assert "--pids-limit" in joined
    assert " -v " not in joined and "--volume" not in joined, (
        "a host path was mounted into a tenant's container"
    )


def test_a_task_does_not_inherit_the_workers_credentials():
    """The worker's environment holds the join key — a credential for the node."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker"))
    from loom_worker.compute.runtime import TaskSpec, _host_env

    import os as _os
    _os.environ["LOOM_KEY"] = "loom_secret"
    env = _host_env(TaskSpec(task_id="t1", image="x", env={"MY": "1"}))

    assert "LOOM_KEY" not in env, "the task could read the node's join key"
    assert env["MY"] == "1" and "PATH" in env


def test_a_node_that_can_run_nothing_says_why():
    """Refusing loudly beats accepting work that will never start."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker"))
    from loom_worker.compute import server
    from loom_worker.compute.runtime import TaskRefused

    server.STATE.update(runtime="", allowed_images=[])
    with pytest.raises(TaskRefused, match="docker.sock"):
        server.run_task({"task_id": "t", "image": "acme/x:1"})


def test_an_owner_can_limit_which_images_run_on_their_machine():
    """A marketplace where anyone may run anything on your hardware is one
    nobody sane joins."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker"))
    from loom_worker.compute import server
    from loom_worker.compute.runtime import TaskRefused

    server.STATE.update(runtime="docker", allowed_images=["ghcr.io/acme/*"])
    with pytest.raises(TaskRefused, match="only runs images matching"):
        server.run_task({"task_id": "t", "image": "evil/miner:latest"})
    assert server._permitted("ghcr.io/acme/render:1")
    server.STATE.update(allowed_images=[])
