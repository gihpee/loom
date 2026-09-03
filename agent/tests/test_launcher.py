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


# ------------------------------------------------- отвергнутый релиз не крутится
def test_отказ_записывается_на_диск(tmp_path, monkeypatch):
    """Без этого узел качает отвергнутое заново при каждом подключении, отказ
    повторяется вместе с перезапуском — и так по кругу, раз в секунду."""
    import json

    from looma_launcher import payload as payload_mod

    monkeypatch.setenv("LOOMA_ROOT", str(tmp_path))
    incoming = payload_mod.incoming_dir()
    incoming.mkdir(parents=True, exist_ok=True)
    manifest = incoming / "0.0.1.json"
    manifest.write_text(json.dumps({"version": "0.0.1", "sha256": "нет",
                                    "signature": "00"}))
    (incoming / "0.0.1.tar.gz").write_bytes(b"not a real archive")

    assert payload_mod.install(manifest, installed_version="0.1.0") is None
    written = (payload_mod.refused_dir() / "0.0.1.txt").read_text()
    assert "not newer" in written


def test_отказ_переживает_перезапуск_агента(tmp_path, monkeypatch):
    """Состояние агента живёт ровно столько же, сколько его процесс."""
    from looma_agent.update import refusal_for

    incoming = tmp_path / "agent" / "incoming"
    incoming.mkdir(parents=True)
    refused = tmp_path / "agent" / "refused"
    refused.mkdir()
    (refused / "0.0.1.txt").write_text("не новее запущенной 0.1.0\n")
    monkeypatch.setenv("LOOMA_AGENT_INCOMING", str(incoming))

    assert "не новее" in refusal_for("0.0.1")
    assert refusal_for("9.9.9") == ""


def test_старый_лаунчер_без_отказов_не_ломает_агента(tmp_path, monkeypatch):
    """Он этих файлов не пишет — тогда защиты просто нет, ровно как раньше."""
    from looma_agent.update import refusal_for

    monkeypatch.setenv("LOOMA_AGENT_INCOMING", str(tmp_path / "incoming"))
    assert refusal_for("0.0.1") == ""


def test_отвергнутый_релиз_больше_не_качается(tmp_path, monkeypatch):
    """Скачать его снова — значит слить задачи, перезапуститься, получить тот
    же отказ и начать заново."""
    from looma_agent import update as update_mod
    from looma_agent.proto import agent_pb2

    incoming = tmp_path / "agent" / "incoming"
    incoming.mkdir(parents=True)
    (tmp_path / "agent" / "refused").mkdir()
    (tmp_path / "agent" / "refused" / "0.0.1.txt").write_text("не новее 0.1.0")
    monkeypatch.setenv("LOOMA_AGENT_INCOMING", str(incoming))

    fetched = []
    monkeypatch.setattr(update_mod.Updater, "_carry_out",
                        lambda self, release: fetched.append(release.version))
    updater = update_mod.Updater(current_version="0.1.0", drain=lambda _s: True,
                                 stop=lambda: None)
    updater.on_release(agent_pb2.AgentRelease(version="0.0.1", url="http://где-то"))

    assert fetched == []
    assert updater.status().state == "refused"
    assert "не новее" in updater.status().error
