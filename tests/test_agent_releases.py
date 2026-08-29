"""Rolling out a new agent: what reaches which node, and what never does."""

from __future__ import annotations

import base64
import hashlib

import pytest

from loom.orchestrator.releases import ReleaseError, ReleaseStore, in_wave


@pytest.fixture
def store(tmp_path):
    return ReleaseStore(tmp_path / "releases")


def publish(store, version="0.2.0", archive=b"a payload"):
    return store.publish(version=version, signature=b"\x01" * 64, archive=archive)


# ----------------------------------------------------------------- publishing
def test_publishing_does_not_roll_out(store):
    """Two acts, because a build reaching the fleet must be deliberate."""
    release = publish(store)
    assert release.wave_percent == 0
    assert store.offer_to("any-node") is None


def test_an_unsigned_release_is_refused_at_the_door(store):
    """Agents would refuse it anyway; failing here says so while a human is
    still looking."""
    with pytest.raises(ReleaseError) as exc:
        store.publish(version="0.2.0", signature=b"", archive=b"x")
    assert "signature" in str(exc.value)


def test_a_release_without_a_version_is_refused(store):
    with pytest.raises(ReleaseError):
        store.publish(version="  ", signature=b"\x01", archive=b"x")


def test_the_digest_is_of_what_was_stored(store):
    payload = b"the real archive bytes"
    release = publish(store, archive=payload)
    assert release.sha256 == hashlib.sha256(payload).hexdigest()
    assert store.archive_bytes("0.2.0") == payload


# -------------------------------------------------------------------- waves
def test_a_wave_reaches_roughly_its_share(store):
    publish(store)
    store.set_wave(25)
    reached = sum(1 for i in range(1000) if store.offer_to(f"node-{i}") is not None)
    assert 200 <= reached <= 300, f"{reached} of 1000 is not about a quarter"


def test_a_node_stays_in_the_same_wave_across_reconnects(store):
    """Rolling a die per registration would let a node update, be told to
    update again a minute later, and never settle."""
    publish(store)
    store.set_wave(30)
    answers = {store.offer_to("node-7") is not None for _ in range(20)}
    assert len(answers) == 1


def test_the_whole_fleet_is_the_whole_fleet(store):
    publish(store)
    store.set_wave(100)
    assert all(store.offer_to(f"node-{i}") is not None for i in range(50))


def test_withdrawing_stops_the_spread(store):
    publish(store)
    store.set_wave(100)
    store.withdraw()
    assert store.offer_to("node-1") is None


def test_a_wave_cannot_be_set_before_anything_is_published(store):
    with pytest.raises(ReleaseError):
        store.set_wave(50)


@pytest.mark.parametrize("given,expected", [(-10, 0), (0, 0), (150, 100)])
def test_a_wave_outside_the_range_is_clamped(store, given, expected):
    publish(store)
    assert store.set_wave(given).wave_percent == expected


# --------------------------------------------------------------- the map
def test_the_version_map_shows_who_is_where(store):
    publish(store, version="0.3.0")
    store.set_wave(100)
    nodes = [
        {"node_id": "a", "agent_version": "0.3.0"},
        {"node_id": "b", "agent_version": "0.2.0"},
        {"node_id": "c", "agent_version": "0.2.0"},
    ]
    view = store.version_map(nodes)
    assert view["versions"] == {"0.3.0": 1, "0.2.0": 2}
    assert view["nodes_total"] == 3
    assert view["nodes_on_target"] == 1
    assert view["nodes_in_wave"] == 3


def test_a_node_that_never_said_its_version_is_still_counted(store):
    view = store.version_map([{"node_id": "a", "agent_version": ""}])
    assert view["versions"] == {"unknown": 1}


def test_the_map_works_before_anything_is_published(store):
    view = store.version_map([{"node_id": "a", "agent_version": "0.1.0"}])
    assert view["release"] is None
    assert view["nodes_on_target"] == 0
