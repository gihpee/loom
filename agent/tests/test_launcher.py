"""The launcher must survive a broken update, because nothing else can save it."""

from __future__ import annotations

from pathlib import Path

from looma_launcher import payload


def test_bundled_agent_is_the_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOMA_ROOT", str(tmp_path))
    chosen = payload.resolve()
    assert chosen.bundled
    assert chosen.version


def test_dangling_current_falls_back_instead_of_dying(tmp_path, monkeypatch):
    """A half-finished update leaves a symlink pointing nowhere.

    The node must come up on the agent we know is intact, not stay dead.
    """
    monkeypatch.setenv("LOOMA_ROOT", str(tmp_path))
    (tmp_path / "agent").mkdir()
    link = tmp_path / "agent" / "current"
    link.symlink_to(tmp_path / "agent" / "0.9.9-that-never-arrived")
    assert payload.resolve(link).bundled


def test_incomplete_payload_is_not_run(tmp_path, monkeypatch):
    """A directory exists but holds no agent: still not something to run."""
    monkeypatch.setenv("LOOMA_ROOT", str(tmp_path))
    version_dir = tmp_path / "agent" / "0.2.0"
    version_dir.mkdir(parents=True)
    link = tmp_path / "agent" / "current"
    link.symlink_to(version_dir)
    assert payload.resolve(link).bundled


def test_complete_payload_is_selected(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOMA_ROOT", str(tmp_path))
    version_dir = tmp_path / "agent" / "0.2.0" / "looma_agent"
    version_dir.mkdir(parents=True)
    (version_dir / "main.py").write_text("")
    link = tmp_path / "agent" / "current"
    link.symlink_to(version_dir.parent)
    chosen = payload.resolve(link)
    assert not chosen.bundled
    assert chosen.version == "0.2.0"
    assert chosen.path == version_dir.parent
