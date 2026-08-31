"""Два ранга собираются в настоящий кластер Ray.

На одной машине это работает без всякого проброса: ранги разговаривают по
настоящему локалхосту, а порты каждый вычисляет сам. Между машинами те же
адреса начнёт обслуживать агент — и ни строчки в payload не изменится.

Здесь запускается настоящий Ray. Тест, проверяющий сборку кластера на моках,
не проверяет ничего.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ray = pytest.importorskip("ray", reason="без ray проверять нечего")

from loom_ray.ports import ports_for  # noqa: E402

# Своё основание, чтобы не столкнуться ни с чужим Ray на машине, ни с
# параллельным прогоном.
BASE = 31000
STRIDE = 60


JOB = """
import json, os, pathlib, ray
ray.init(log_to_driver=False)

@ray.remote
def square(n): return n * n

answers = ray.get([square.remote(i) for i in range(50)])
pathlib.Path(os.environ["LOOM_TASK_OUT"], "answer.json").write_text(json.dumps({
    "sum": sum(answers),
    "nodes": len([n for n in ray.nodes() if n["Alive"]]),
    "cpus": ray.cluster_resources().get("CPU", 0),
}))
"""


@pytest.fixture(autouse=True)
def чистый_ray():
    """Снести чужой и свой прошлый Ray до и после.

    Оставшийся от прошлого прогона кластер занимает те же порты, и следующий
    ранг подключается к НЕМУ — а голова потом падает на несовпадении имени
    сессии. Причина при этом называется так, что искать её будешь в своём коде.
    """
    _ray_stop()
    yield
    _ray_stop()


def _ray_stop() -> None:
    subprocess.run([sys.executable, "-m", "ray.scripts.scripts", "stop", "--force"],
                   capture_output=True, timeout=120)


def _end(proc: subprocess.Popen) -> None:
    """Снять ранг так, как это делает агент — всей группой процессов."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


def health(port: int) -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as a:
            return json.loads(a.read())
    except urllib.error.HTTPError as exc:      # 503, пока не готов — это ответ
        return json.loads(exc.read())
    except (urllib.error.URLError, OSError, ValueError):
        return {}


def rank_process(tmp_path: Path, rank: int, size: int, out: Path,
                 script: str = "") -> subprocess.Popen:
    """Ранг так, как его запустил бы агент: своим каталогом и своим коротким tmp."""
    scratch = Path("/tmp") / f"loom-ray-test-{os.getpid()}-{rank}"
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        LOOM_RANK=str(rank), LOOM_GROUP_SIZE=str(size),
        LOOM_TASK_ID=f"test-rank-{rank}", LOOM_TASK_OUT=str(out),
        LOOM_TASK_TMP=str(scratch),
        LOOM_RAY_PORT_BASE=str(BASE), LOOM_RAY_PORT_STRIDE=str(STRIDE),
        LOOM_RAY_HEAD_WAIT_S="180", LOOM_RAY_JOIN_WAIT_S="180",
        # Только ради разработки на макбуке: Ray запрещает многоузловые
        # кластеры на macOS и Windows. Узлы Loom — Linux, там этого нет, и в
        # сам payload такое ставить нечего.
        RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER="1",
    )
    argv = [sys.executable, "-m", "loom_ray.server",
            "--size", str(size), "--rank", str(rank),
            "--serve-port", str(9700 + rank), "--gpus", "0"]
    if script:
        argv += ["--script", script]
    # Своя группа процессов — как у настоящей задачи, и снимается так же.
    return subprocess.Popen(argv, env=env, cwd=str(tmp_path),
                            start_new_session=True)


@pytest.mark.slow
def test_два_ранга_собираются_в_кластер_и_считают(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    job = tmp_path / "job.py"
    job.write_text(JOB)

    # Ранг 1 стартует ПЕРВЫМ — он должен уметь ждать голову, а не падать на
    # том, что её ещё нет. На узле это норма: его окружение могло собраться
    # быстрее.
    second = rank_process(tmp_path, 1, 2, out)
    time.sleep(2)
    first = rank_process(tmp_path, 0, 2, out, script=str(job))

    try:
        deadline = time.time() + 240
        answer = out / "answer.json"
        while time.time() < deadline and not answer.exists():
            if first.poll() is not None and not answer.exists():
                pytest.fail(f"ранг 0 завершился с кодом {first.returncode}")
            time.sleep(1)
        assert answer.exists(), "кластер не собрался за отведённое время"

        got = json.loads(answer.read_text())
        assert got["sum"] == sum(i * i for i in range(50))
        assert got["nodes"] == 2, f"в кластере {got['nodes']} узлов вместо двух"

        # Ранг 1 всё это время докладывал о себе честно.
        assert health(9701).get("status") == "ok"
    finally:
        for proc in (first, second):
            _end(proc)


@pytest.mark.slow
def test_пока_кластер_не_собран_ранг_говорит_что_не_готов(tmp_path):
    """Тот же контракт, что у стадии: «running» — про процесс, а не про
    готовность. Запрос, пришедший в этом промежутке, должен получить отказ, а
    не упасть внутри Ray."""
    out = tmp_path / "out"
    out.mkdir()
    # Группа из двух, но поднимаем только один: собраться неоткуда.
    alone = rank_process(tmp_path, 0, 2, out)
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            state = health(9700)
            if alone.poll() is not None and not state:
                pytest.fail(f"ранг упал с кодом {alone.returncode}, не подняв /health")
            if state:
                assert state["status"] != "ok", "объявил готовность в одиночку"
                assert state["size"] == 2
                return
            time.sleep(1)
        pytest.fail("/health не отвечал вовсе")
    finally:
        _end(alone)
