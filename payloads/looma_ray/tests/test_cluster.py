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

from looma_ray.ports import ports_for  # noqa: E402

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
pathlib.Path(os.environ["LOOMA_TASK_OUT"], "answer.json").write_text(json.dumps({
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
    scratch = Path("/tmp") / f"looma-ray-test-{os.getpid()}-{rank}"
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        LOOMA_RANK=str(rank), LOOMA_GROUP_SIZE=str(size),
        LOOMA_TASK_ID=f"test-rank-{rank}", LOOMA_TASK_OUT=str(out),
        LOOMA_TASK_TMP=str(scratch),
        LOOMA_RAY_PORT_BASE=str(BASE), LOOMA_RAY_PORT_STRIDE=str(STRIDE),
        LOOMA_RAY_HEAD_WAIT_S="180", LOOMA_RAY_JOIN_WAIT_S="180",
        # Только ради разработки на макбуке: Ray запрещает многоузловые
        # кластеры на macOS и Windows. Узлы Looma — Linux, там этого нет, и в
        # сам payload такое ставить нечего.
        RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER="1",
    )
    argv = [sys.executable, "-m", "looma_ray.server",
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


@pytest.mark.slow
def test_ранг_переживает_негодный_номер_порта(tmp_path):
    """Со стенда: оркестратор шлёт serve_port=1 как «да, служи», и ранг падал
    на `Permission denied` через секунду после старта — порт ниже 1024 задаче
    не занять, она работает не под root.

    Предложенный номер — предложение: свой выбор ранг сообщает агенту сам.
    """
    out = tmp_path / "out"
    out.mkdir()
    scratch = Path("/tmp") / f"looma-ray-test-{os.getpid()}-priv"
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ, LOOMA_RANK="0", LOOMA_GROUP_SIZE="1",
        LOOMA_TASK_ID="privileged", LOOMA_TASK_OUT=str(out),
        LOOMA_TASK_TMP=str(scratch),
        LOOMA_RAY_PORT_BASE=str(BASE + 500), LOOMA_RAY_PORT_STRIDE=str(STRIDE),
        RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER="1",
    )
    alone = subprocess.Popen(
        [sys.executable, "-m", "looma_ray.server", "--size", "1", "--rank", "0",
         "--serve-port", "1", "--gpus", "0"],
        env=env, cwd=str(tmp_path), start_new_session=True)
    try:
        # Раньше умирал меньше чем за секунду. Живёт — значит выбрал другой порт.
        deadline = time.time() + 45
        while time.time() < deadline:
            if alone.poll() is not None:
                pytest.fail(f"упал с кодом {alone.returncode} вместо выбора порта")
            time.sleep(1)
    finally:
        _end(alone)


def test_упавший_ранг_не_трогает_соседей():
    """Со стенда: `ray stop --force` снимает ВСЕ процессы Ray этого
    пользователя на машине, а не только свои. Упавший ранг своей уборкой убивал
    голову живого соседа — тот потом стоял и ждал кластер, которого уже нет.

    Убирает за нами агент: задача работает в своей группе процессов.
    """
    import subprocess as sp

    from looma_ray import cluster

    ran = []
    original = sp.run
    try:
        sp.run = lambda *a, **k: ran.append(a) or original(
            [sys.executable, "-c", "pass"], capture_output=True)
        cluster.stop_node()
    finally:
        sp.run = original
    assert ran == [], "уборка запустила внешнюю команду — она снесёт и соседей"


def test_ранг_повторяет_попытку_пока_голова_не_готова(monkeypatch):
    """Открытый порт головы не значит, что голова готова: GCS занимает его в
    первую секунду, а собирается ещё десятки. Присоединение в этом промежутке
    падает по таймауту raylet'а, и отличить это снаружи нельзя — поэтому
    пробуют ещё раз, а не гадают."""
    import subprocess as sp

    from looma_ray import cluster

    monkeypatch.setattr(cluster, "JOIN_RETRY_S", 0.01)
    attempts = []

    class Result:
        def __init__(self, code):
            self.returncode, self.stdout, self.stderr = code, "", "raylet timed out"

    def flaky(*_a, **_k):
        attempts.append(1)
        return Result(1 if len(attempts) < 3 else 0)

    monkeypatch.setattr(sp, "run", flaky)
    cluster._run_start(["ray", "start"], rank=1, retries=4)
    assert len(attempts) == 3, "должен был повторять, пока голова не примет"


def test_голова_не_повторяет(monkeypatch):
    """Ей ждать некого: если она не поднялась, повтор ничего не меняет, а
    время до внятной ошибки растёт кратно."""
    import subprocess as sp

    from looma_ray import cluster

    attempts = []

    class Result:
        returncode, stdout, stderr = 1, "", "порт занят"

    monkeypatch.setattr(sp, "run", lambda *_a, **_k: attempts.append(1) or Result())
    with pytest.raises(cluster.ClusterRefused, match="ранга 0"):
        cluster._run_start(["ray", "start", "--head"], rank=0, retries=0)
    assert len(attempts) == 1

