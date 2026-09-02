"""Rolling out a new agent: what reaches which node, and what never does."""

from __future__ import annotations

import base64
import hashlib

import pytest

from looma.orchestrator.releases import ReleaseError, ReleaseStore, in_wave


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


# --------------------------------------------------------------- подписание
def test_архив_нагрузки_воспроизводим(tmp_path):
    """Два прогона подряд должны дать один и тот же файл и одну подпись.

    Иначе повторный запуск ради потерянной подписи выдаёт подпись, которая не
    подходит к уже скачанному архиву — ловушка, которая не выглядит ловушкой:
    команда та же, ключ тот же, версия та же.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "sign_release.py"
    first = tmp_path / "a.tar.gz"
    second = tmp_path / "b.tar.gz"
    for out in (first, second):
        result = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, {str(script.parent)!r});"
             f"import sign_release; sign_release.build_archive('0.0.1', __import__('pathlib').Path({str(out)!r}))"],
            capture_output=True, text=True, cwd=root)
        assert result.returncode == 0, result.stderr
    assert first.read_bytes() == second.read_bytes(), \
        "два прогона дали разные байты — подпись не переживёт пересборку"


def test_манифест_лежит_рядом_с_архивом(tmp_path):
    """Подпись, отделённая от своего файла, бесполезна, а вывод команды
    теряется при первом же переключении окна."""
    import json as _json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    key = tmp_path / "release.key"
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "sign_release.py"),
         "keygen", "--out", str(key)],
        capture_output=True, text=True, cwd=root)
    # Код 3 = «публичный ключ в дереве не тронут»; пара при этом создана, и
    # это ровно то, что нужно тесту: подписывать своим, парк не трогать.
    if result.returncode not in (0, 3):
        pytest.skip(f"нет cryptography: {result.stderr[:80]}")
    out = tmp_path / "looma-agent-9.9.9.tar.gz"
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "sign_release.py"), "sign",
         "--key", str(key), "--version", "9.9.9", "--out", str(out)],
        capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr
    manifest = out.with_name("looma-agent-9.9.9.json")
    assert manifest.is_file(), "манифест не лёг рядом с архивом"
    body = _json.loads(manifest.read_text())
    assert body["version"] == "9.9.9"
    assert len(body["signature"]) == 128, "подпись ed25519 — 64 байта в hex"
    assert len(body["sha256"]) == 64


def test_keygen_не_подменяет_ключ_парка_молча(tmp_path):
    """Ключ в дереве — якорь доверия уже розданных образов.

    Подменить его значит лишить обновлений каждый узел с таким образом, а
    вернуть — только руками на каждой машине. Именно так этот тест однажды и
    сломал стенд, вызвав keygen мимоходом.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    anchor = root / "agent" / "looma_launcher" / "release_key.pub"
    if not anchor.is_file():
        pytest.skip("в этом дереве ключа парка нет")
    before = anchor.read_text()
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "sign_release.py"),
         "keygen", "--out", str(tmp_path / "other.key")],
        capture_output=True, text=True, cwd=root)
    assert anchor.read_text() == before, "keygen подменил ключ парка"
    assert result.returncode == 3
    assert "--install" in result.stderr
