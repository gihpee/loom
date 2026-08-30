"""Phase 2: environments are built once, reused, and evicted without casualties.

Without a cache this whole design is worse than what it replaces: provisioning
moves gigabytes out of `docker pull` into `pip install`, which is slower. These
tests are about the properties that make it pay off instead.
"""

from __future__ import annotations

import shutil
import sys
import threading
import time

import pytest

from loom_agent.tasks.env import EnvironmentCache
from loom_agent.tasks.env import cache as cache_mod
from loom_agent.tasks.env import python as python_env
from loom_agent.tasks.env.base import read_marker, write_marker
from loom_agent.tasks.spec import EnvSpec, TaskRefused


@pytest.fixture
def cache(tmp_path):
    return EnvironmentCache(tmp_path / "envs")


def plant(cache: EnvironmentCache, name: str, size_bytes: int, *, used_at: float = None):
    """A finished environment, without paying to build a real one."""
    path = cache.root / name
    path.mkdir(parents=True)
    write_marker(path, fingerprint=name, kind="python", size_bytes=size_bytes)
    if used_at is not None:
        import os

        os.utime(path / ".loom-env.json", (used_at, used_at))
    return path


# --------------------------------------------------------------- identity
def test_the_same_request_is_the_same_environment():
    """Order must not matter, or two identical asks would build twice."""
    one = EnvSpec(kind="python", requirements=("numpy", "torch"))
    other = EnvSpec(kind="python", requirements=("torch", "numpy"))
    assert one.fingerprint() == other.fingerprint()


def test_a_different_request_is_a_different_environment():
    assert EnvSpec(kind="python", requirements=("numpy",)).fingerprint() != \
           EnvSpec(kind="python", requirements=("numpy", "scipy")).fingerprint()


# ------------------------------------------------------------ building once
def test_an_environment_is_built_and_then_reused(cache):
    spec = EnvSpec(kind="python")
    first = cache.acquire(spec)
    assert first.path is not None and first.path.exists()
    assert read_marker(first.path) is not None
    built_at = read_marker(first.path)["built_at"]

    cache.release(first.fingerprint)
    second = cache.acquire(spec)
    assert second.path == first.path
    # Not rebuilt: the marker is the one the first build wrote.
    assert read_marker(second.path)["built_at"] == built_at


def test_two_tasks_wanting_one_environment_build_it_once(cache, monkeypatch):
    """The second waits for the first rather than racing it into the directory."""
    builds = []
    real = python_env.build

    def slow(target, requirements):
        builds.append(target)
        time.sleep(0.5)
        real(target, requirements)

    monkeypatch.setattr(cache_mod.python_env, "build", slow)
    spec = EnvSpec(kind="python", requirements=())
    results = []

    def take():
        results.append(cache.acquire(spec))

    threads = [threading.Thread(target=take) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert len(builds) == 1, f"built {len(builds)} times instead of once"
    assert len({r.path for r in results}) == 1


def test_a_half_built_environment_is_never_handed_out(cache):
    """A directory without a marker is not an environment, whatever it holds.

    This is the failure that costs an afternoon: the task starts, imports
    something that is only half there, and fails with an error about the wrong
    thing entirely.
    """
    fingerprint = EnvSpec(kind="python").fingerprint()
    unfinished = cache.root / fingerprint
    unfinished.mkdir(parents=True)
    (unfinished / "lib").mkdir()
    assert cache._ready(fingerprint) is None


def test_a_failed_build_leaves_nothing_behind(cache, monkeypatch):
    def explode(target, requirements):
        target.mkdir(parents=True, exist_ok=True)
        (target / "partial").write_text("half a venv")
        raise TaskRefused("could not install nosuchpackage: no matching distribution")

    monkeypatch.setattr(cache_mod.python_env, "build", explode)
    spec = EnvSpec(kind="python", requirements=("nosuchpackage",))
    with pytest.raises(TaskRefused) as exc:
        cache.acquire(spec)
    assert "nosuchpackage" in str(exc.value)
    assert not (cache.root / spec.fingerprint()).exists()
    assert list(cache.root.glob(".building-*")) == []


def test_what_a_crashed_agent_left_half_built_is_swept(tmp_path):
    """Брошенное убирается — но по возрасту, а не по одному лишь префиксу.

    Том может быть общим со вторым агентом на этой же машине, и подметание
    всего подряд снесло бы то, что он собирает прямо сейчас.
    """
    import os as _os

    root = tmp_path / "envs"
    root.mkdir()
    leftover = root / ".building-python-abc123-4242"
    leftover.mkdir()
    (leftover / "junk").write_text("x")
    old = time.time() - cache_mod.STALE_BUILD_S - 60
    _os.utime(leftover, (old, old))
    keep = root / "python-good"
    keep.mkdir()
    write_marker(keep, fingerprint="python-good", kind="python", size_bytes=1)

    EnvironmentCache(root)
    assert not leftover.exists()
    assert keep.exists()


# ---------------------------------------------------------------- eviction
def test_the_least_recently_used_environment_goes_first(cache):
    now = time.time()
    old = plant(cache, "python-old", 6 * 1024**3, used_at=now - 10_000)
    recent = plant(cache, "python-recent", 6 * 1024**3, used_at=now - 10)
    cache.quota_bytes = 8 * 1024**3

    cache.acquire(EnvSpec(kind="python"))
    assert not old.exists(), "the oldest environment should have been evicted"
    assert recent.exists()


def test_an_environment_in_use_is_never_evicted(cache):
    """Meeting a disk target by breaking a running task is the wrong trade."""
    spec = EnvSpec(kind="python")
    held = cache.acquire(spec)  # leased, not released
    plant(cache, "python-old", 20 * 1024**3, used_at=time.time() - 10_000)
    cache.quota_bytes = 1024  # everything is over quota now

    cache._evict_to_quota()
    assert held.path.exists(), "an environment a task is using was evicted"
    assert not (cache.root / "python-old").exists()


def test_a_released_environment_becomes_evictable(cache):
    spec = EnvSpec(kind="python")
    one = cache.acquire(spec)
    cache.release(one.fingerprint)
    cache.quota_bytes = 1
    cache._evict_to_quota()
    assert not one.path.exists()


def test_leases_are_counted_not_just_set(cache):
    """Two tasks on one environment: the first finishing must not free it."""
    spec = EnvSpec(kind="python")
    first = cache.acquire(spec)
    cache.acquire(spec)
    cache.release(first.fingerprint)
    cache.quota_bytes = 1
    cache._evict_to_quota()
    assert first.path.exists(), "still held by the second task"
    cache.release(first.fingerprint)
    cache._evict_to_quota()
    assert not first.path.exists()


def test_the_quota_is_reported_not_just_enforced(cache):
    plant(cache, "python-a", 3 * 1024**3)
    snapshot = cache.snapshot()
    assert snapshot["bytes"] == 3 * 1024**3
    assert snapshot["quota_bytes"] == cache.quota_bytes


# -------------------------------------------------------------- refusals
def test_an_environment_kind_we_cannot_build_says_which(cache):
    """A kind this agent does not know is named back, with the ones it does."""
    with pytest.raises(TaskRefused) as exc:
        cache.acquire(EnvSpec(kind="wasm", source="module.wasm"))
    message = str(exc.value)
    assert "wasm" in message
    assert "python" in message and "oci" in message


def test_no_environment_costs_nothing(cache):
    empty = cache.acquire(EnvSpec(kind="none"))
    assert empty.empty
    assert empty.overrides() == {}
    assert list(cache.root.iterdir()) == []


# ------------------------------------------------- the task actually uses it
@pytest.fixture
def registry(tmp_path, monkeypatch):
    from loom_agent.tasks.limits import resolve_isolation
    from loom_agent.tasks.registry import TaskRegistry

    monkeypatch.setenv("LOOM_ALLOW_UNPRIVILEGED_TASKS", "1")
    reg = TaskRegistry(
        root=tmp_path / "tasks",
        isolation=resolve_isolation(),
        environments=EnvironmentCache(tmp_path / "envs"),
        total_gpus=0,
        retention_s=60.0,
    )
    yield reg
    reg.stop_all()


def submit(registry, task_id, command, **kwargs):
    from loom_agent.tasks.spec import TaskSpec

    raw = {"task_id": task_id, "command": command}
    raw.update(kwargs)
    return registry.submit(TaskSpec.from_dict(raw))


def test_a_task_runs_against_the_environment_it_asked_for(registry):
    """`python` in the command means the environment's interpreter.

    That only works because the environment's bin directory is prepended to
    PATH rather than the task being told where the interpreter lives.
    """
    task = submit(registry, "p1", ["python", "-c", "import sys; print(sys.prefix)"],
                  environment={"kind": "python"})
    assert task.wait(timeout=120)
    assert task.state == "done", task.logs()
    assert str(task.environment.path) in task.logs()


def test_what_the_image_already_has_is_not_installed_again(registry):
    """`--system-site-packages` is the reason the cache is worth having.

    A node that has served one inference task carries torch. Reinstalling it
    per task, into a directory about to be deleted, would defeat the point —
    so the environment must be able to see what is already there.
    """
    task = submit(registry, "p2",
                  ["python", "-c", "import psutil; print('inherited', psutil.__name__)"],
                  environment={"kind": "python"})
    assert task.wait(timeout=120)
    assert task.state == "done", task.logs()
    assert "inherited" in task.logs()


def test_the_second_task_does_not_wait_for_a_build(registry):
    """The property the whole phase exists for."""
    env = {"kind": "python"}
    first = submit(registry, "p3", ["python", "-c", "pass"], environment=env)
    assert first.wait(timeout=120)

    started = time.time()
    second = submit(registry, "p4", ["python", "-c", "pass"], environment=env)
    elapsed = time.time() - started
    assert second.wait(timeout=60)
    assert second.environment.path == first.environment.path
    assert elapsed < 5, f"a cache hit took {elapsed:.1f}s; it was rebuilt"


def test_an_environment_outlives_the_task_that_built_it(registry):
    task = submit(registry, "p5", ["python", "-c", "pass"], environment={"kind": "python"})
    assert task.wait(timeout=120)
    path = task.environment.path
    registry.release("p5")
    assert not task.directory.root.exists()
    assert path.exists(), "the environment went away with the task that used it"


# ------------------------------------------- два агента на одной машине
def test_два_агента_на_общем_томе_не_ломают_сборку_друг_другу(tmp_path, monkeypatch):
    """Симптом со стенда: у одного упал ensurepip, у второго "File exists".

    Причина была одна: имя каталога сборки содержало pid, а в своих контейнерах
    оба агента — процесс номер 7. Пути совпадали, и один сносил недостроенное
    окружение другого. Ни одно из двух сообщений на настоящую причину не
    указывало.
    """
    shared = tmp_path / "envs"
    first, second = EnvironmentCache(shared), EnvironmentCache(shared)
    spec = EnvSpec(kind="python")

    started, hurt = threading.Barrier(2), []

    def build(cache):
        started.wait(timeout=10)
        try:
            cache.acquire(spec)
        except Exception as exc:      # noqa: BLE001 — интересен любой отказ
            hurt.append(exc)

    threads = [threading.Thread(target=build, args=(c,)) for c in (first, second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)

    assert not hurt, f"агенты помешали друг другу: {hurt}"
    assert cache_ready(first, spec) and cache_ready(second, spec)


def cache_ready(cache, spec):
    ready = cache._ready(spec.fingerprint())
    return ready is not None and (ready.path / ".loom-env.json").is_file()


def test_каталоги_сборки_не_совпадают_у_разных_процессов(tmp_path, monkeypatch):
    """Уникальность не должна держаться на pid: он повторяется между
    контейнерами."""
    cache = EnvironmentCache(tmp_path / "envs")
    seen = set()
    real = cache_mod.python_env.build

    def note(target, requirements):
        seen.add(target.name)
        real(target, requirements)

    monkeypatch.setattr(cache_mod.python_env, "build", note)
    for suffix in ("a", "b", "c"):
        cache.acquire(EnvSpec(kind="python", requirements=(suffix,) if False else ()))
        cache.release(EnvSpec(kind="python").fingerprint())
        # каждый раз с нуля, чтобы сборка действительно происходила
        shutil.rmtree(cache.root / EnvSpec(kind="python").fingerprint(), ignore_errors=True)
    assert len(seen) == 3, f"имена каталогов повторились: {seen}"


def test_свежую_чужую_сборку_не_подметают(tmp_path):
    """Том общий: подметание по одному лишь префиксу снесло бы то, что прямо
    сейчас собирает сосед."""
    root = tmp_path / "envs"
    root.mkdir()
    fresh = root / ".building-python-abc-собирается"
    fresh.mkdir()
    stale = root / ".building-python-abc-брошено"
    stale.mkdir()
    import os as _os

    old = time.time() - cache_mod.STALE_BUILD_S - 60
    _os.utime(stale, (old, old))

    EnvironmentCache(root)
    assert fresh.exists(), "снесли чужую живую сборку"
    assert not stale.exists(), "брошенное осталось лежать"


# ------------------------------------------------- колесо под драйвер узла
def test_индекс_torch_заменяет_а_не_дополняет():
    """`--extra-index-url` ДОБАВЛЯЕТ индекс, и pip дальше берёт версию повыше —
    с PyPI, то есть сборку под самую свежую CUDA.

    Симптом со стенда: каталог окружения назывался cu124, а внутри лежал torch,
    которому мало драйвера. Падало это при первом обращении к карте, через
    десять минут после скачивания весов. Проверено вживую: со старым флагом pip
    резолвил torch 2.13.0 с files.pythonhosted.org.
    """
    from unittest import mock

    from loom_agent.tasks.env.python import _torch_index

    with mock.patch("loom_agent.hwinfo.cuda_driver_version", return_value=(12, 4)):
        flags = _torch_index(["torch"])
    assert flags[0] == "--index-url", \
        "--extra-index-url не гарантирует сборку: pip выберет версию повыше"
    assert flags[1].endswith("/cu124")


def test_способ_сборки_входит_в_имя_окружения():
    """Иначе узел с уже собранным окружением переиспользовал бы собранное
    по-старому: имя обещало cu124, а внутри лежал torch с PyPI."""
    from unittest import mock

    from loom_agent.tasks.env.python import RECIPE, wheel_variant

    with mock.patch("loom_agent.hwinfo.cuda_driver_version", return_value=(12, 4)):
        assert wheel_variant(["torch"]) == f"cu124-r{RECIPE}"


def test_колесо_torch_выбирается_под_драйвер_узла():
    """Симптом со стенда: pip поставил torch под свежую CUDA, и он упал при
    первом обращении к карте — "The NVIDIA driver on your system is too old
    (found version 12040)". Сообщение не говорит ни что делать, ни кто виноват.

    Выбирает узел, а не оркестратор: тот говорит, ЧТО нужно, а какое колесо
    подходит этому железу, знает только само железо.
    """
    from unittest import mock

    from loom_agent.tasks.env.python import _torch_index

    with mock.patch("loom_agent.hwinfo.cuda_driver_version", return_value=(12, 4)):
        assert _torch_index(["torch"])[-1].endswith("/cu124")
    with mock.patch("loom_agent.hwinfo.cuda_driver_version", return_value=(12, 8)):
        assert _torch_index(["torch"])[-1].endswith("/cu128")


def test_без_карты_берутся_cpu_колёса():
    """Гигабайты CUDA на машине без NVIDIA — трафик владельца впустую."""
    from unittest import mock

    from loom_agent.tasks.env.python import _torch_index

    with mock.patch("loom_agent.hwinfo.cuda_driver_version", return_value=None):
        assert _torch_index(["torch"])[-1].endswith("/cpu")


def test_окружение_без_torch_не_трогает_индекс():
    from loom_agent.tasks.env.python import _torch_index

    assert _torch_index(["numpy", "pillow"]) == []
    assert _torch_index(["torchmetrics"]) == [], "torchmetrics — это не torch"


def test_смена_драйвера_даёт_новое_окружение(tmp_path):
    """Иначе узел переиспользовал бы кэш, собранный под старые условия, и падал
    бы ровно так же — включая случай, когда мы починили сам выбор колеса."""
    from unittest import mock

    cache = EnvironmentCache(tmp_path / "envs")
    spec = EnvSpec(kind="python", requirements=("torch",))
    with mock.patch("loom_agent.hwinfo.cuda_driver_version", return_value=(12, 4)):
        old = cache._key(spec)
    with mock.patch("loom_agent.hwinfo.cuda_driver_version", return_value=(12, 8)):
        new = cache._key(spec)
    assert old != new
    assert "cu124" in old and "cu128" in new
