"""Phase 1: a task gets a directory, limits, its own cards, and is cleaned up.

Everything here runs real processes. A task runner that is only tested against
fakes is a task runner nobody has watched kill anything.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from loom_agent.tasks.limits import Isolation, resolve_isolation
from loom_agent.tasks.env import EnvironmentCache
from loom_agent.tasks.registry import TaskRegistry
from loom_agent.tasks.spec import TaskRefused, TaskSpec


@pytest.fixture
def isolation(monkeypatch):
    """Tests do not run as root, so they take the arrangement the operator
    would have to opt into explicitly on a real node."""
    monkeypatch.setenv("LOOM_ALLOW_UNPRIVILEGED_TASKS", "1")
    return resolve_isolation()


@pytest.fixture
def registry(tmp_path, isolation):
    # Long retention so the reaper does not race the tests that release a task
    # by hand. Reclaiming on its own is exercised separately below.
    reg = TaskRegistry(root=tmp_path / "tasks", isolation=isolation,
                       environments=EnvironmentCache(tmp_path / "envs"),
                       total_gpus=4, retention_s=60.0)
    yield reg
    reg.stop_all()


def spec(task_id: str, command, **kwargs) -> TaskSpec:
    raw = {"task_id": task_id, "command": command}
    raw.update(kwargs)
    return TaskSpec.from_dict(raw)


# ------------------------------------------------------------------ the basics
def test_a_task_runs_and_reports_how_it_ended(registry):
    task = registry.submit(spec("t1", [sys.executable, "-c", "print('hello')"]))
    assert task.wait(timeout=30)
    assert task.state == "done"
    assert task.exit_code == 0
    assert "hello" in task.logs()


def test_a_failing_task_says_so(registry):
    task = registry.submit(spec("t2", [sys.executable, "-c", "raise SystemExit(3)"]))
    assert task.wait(timeout=30)
    assert task.state == "failed"
    assert task.exit_code == 3
    assert "3" in task.error


def test_a_task_runs_in_its_own_directory(registry):
    task = registry.submit(spec("t3", [sys.executable, "-c", "import os; print(os.getcwd())"]))
    assert task.wait(timeout=30)
    assert str(task.directory.work) in task.logs()


# ------------------------------------------------------------------- the limits
def test_the_timeout_takes_the_whole_process_tree(registry):
    """A task that started children must not leave them on the owner's machine.

    The parent is killed either way; the question is whether the child it
    spawned is still running afterwards, holding memory nobody is accounting
    for.
    """
    program = (
        "import subprocess, sys, time;"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        "print(child.pid, flush=True);"
        "time.sleep(60)"
    )
    task = registry.submit(spec("t4", [sys.executable, "-c", program], timeout_s=2))
    assert task.wait(timeout=40)
    assert task.state == "cancelled"
    assert "limit" in task.error
    child_pid = int(task.logs().strip().splitlines()[0])
    deadline = time.time() + 15
    while time.time() < deadline:
        if not _alive(child_pid):
            return
        time.sleep(0.2)
    pytest.fail(f"the child {child_pid} outlived its task and is still running")


def test_a_task_cannot_read_this_nodes_credentials(registry, monkeypatch):
    """The agent's environment holds the join key. The task's must not."""
    monkeypatch.setenv("LOOM_JOIN_KEY", "loom_secret-do-not-leak")
    monkeypatch.setenv("LOOM_SOMETHING_ELSE", "also-private")
    task = registry.submit(spec(
        "t5", [sys.executable, "-c", "import os; print(sorted(os.environ))"]
    ))
    assert task.wait(timeout=30)
    printed = task.logs()
    assert "LOOM_JOIN_KEY" not in printed
    assert "LOOM_SOMETHING_ELSE" not in printed
    assert "LOOM_TASK_ID" in printed


def test_a_task_is_told_where_to_put_its_result(registry):
    task = registry.submit(spec(
        "t6", [sys.executable, "-c", "import os; print(os.environ['LOOM_TASK_OUT'])"]
    ))
    assert task.wait(timeout=30)
    assert str(task.directory.out) in task.logs()


def test_a_task_over_its_disk_quota_is_stopped(registry):
    program = (
        "open('big', 'wb').write(b'x' * (6 * 1024 * 1024));"
        "import time; time.sleep(30)"
    )
    task = registry.submit(spec(
        "t7", [sys.executable, "-c", program],
        resources={"disk_bytes": 1024 * 1024}, timeout_s=30,
    ))
    assert task.wait(timeout=40)
    assert task.state == "cancelled"
    assert "disk quota" in task.error


# --------------------------------------------------------------------- the cards
def test_two_tasks_never_land_on_the_same_card(registry):
    """The bug this accounting exists to prevent.

    The old compute path gave every task devices 0..N-1, so a second task sat
    on card 0 next to the first while cards 1..3 idled.
    """
    first = registry.submit(spec("g1", [sys.executable, "-c", "import time; time.sleep(5)"],
                                 resources={"gpus": 2}))
    second = registry.submit(spec("g2", [sys.executable, "-c", "import time; time.sleep(5)"],
                                  resources={"gpus": 2}))
    assert set(first.devices).isdisjoint(second.devices)
    assert sorted(first.devices + second.devices) == [0, 1, 2, 3]
    assert registry.free_devices() == []


def test_a_task_is_told_which_cards_are_its_own(registry):
    task = registry.submit(spec(
        "g3", [sys.executable, "-c", "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))"],
        resources={"gpus": 2},
    ))
    assert task.wait(timeout=30)
    assert task.logs().strip() == ",".join(str(d) for d in task.devices)


def test_asking_for_more_cards_than_are_free_is_refused_with_the_count(registry):
    registry.submit(spec("g4", [sys.executable, "-c", "import time; time.sleep(5)"],
                         resources={"gpus": 3}))
    with pytest.raises(TaskRefused) as exc:
        registry.submit(spec("g5", ["true"], resources={"gpus": 2}))
    assert "1 of its 4" in str(exc.value)


def test_cards_come_back_when_the_task_ends(registry):
    task = registry.submit(spec("g6", [sys.executable, "-c", "pass"], resources={"gpus": 4}))
    assert task.wait(timeout=30)
    deadline = time.time() + 10
    while time.time() < deadline:
        if len(registry.free_devices()) == 4:
            return
        time.sleep(0.1)
    pytest.fail("the cards were still held after the task finished")


def test_a_refused_task_does_not_keep_the_cards_it_never_got(registry):
    with pytest.raises(TaskRefused):
        registry.submit(spec("g7", ["true"], resources={"gpus": 9}))
    assert len(registry.free_devices()) == 4


# ------------------------------------------------------------------- the cleanup
def test_releasing_a_task_takes_its_disk_back(registry):
    task = registry.submit(spec("c1", [sys.executable, "-c", "open('f','w').write('x'*1000)"]))
    assert task.wait(timeout=30)
    root = task.directory.root
    assert root.exists()
    registry.release("c1")
    assert not root.exists()
    assert registry.get("c1") is None


def test_releasing_a_running_task_stops_it_first(registry):
    task = registry.submit(spec("c2", [sys.executable, "-c", "import time; time.sleep(60)"]))
    registry.release("c2")
    assert task.finished
    assert not task.directory.root.exists()


def test_a_forgotten_task_has_its_disk_reclaimed(tmp_path, isolation):
    """Nobody collected the result. The owner still gets their disk back."""
    reg = TaskRegistry(root=tmp_path / "tasks", isolation=isolation,
                       environments=EnvironmentCache(tmp_path / "envs2"),
                       total_gpus=0, retention_s=0.0)
    task = reg.submit(spec("r1", [sys.executable, "-c", "open('f','w').write('x')"]))
    assert task.wait(timeout=30)
    deadline = time.time() + 15
    while time.time() < deadline:
        if not task.directory.root.exists() and reg.get("r1") is None:
            return
        time.sleep(0.1)
    pytest.fail("the directory of an uncollected task was never reclaimed")


def test_a_leftover_directory_is_not_handed_to_the_next_tenant(registry, tmp_path):
    """A crash can leave one task's files behind. They are not the next one's."""
    stale = tmp_path / "tasks" / "c3"
    (stale / "work").mkdir(parents=True)
    (stale / "work" / "secret").write_text("the previous tenant's data")
    task = registry.submit(spec("c3", [sys.executable, "-c",
                                       "import os; print(os.listdir('.'))"]))
    assert task.wait(timeout=30)
    assert "secret" not in task.logs()


# -------------------------------------------------------------------- refusals
def test_an_environment_kind_we_cannot_build_is_refused_not_ignored(registry):
    """Honest refusal beats a task that starts without what it asked for."""
    with pytest.raises(TaskRefused) as exc:
        registry.submit(spec("e1", ["true"], environment={"kind": "wasm"}))
    assert "wasm" in str(exc.value)


def test_an_unknown_environment_kind_is_refused(registry):
    with pytest.raises(TaskRefused):
        registry.submit(spec("e2", ["true"], environment={"kind": "wasm"}))


def test_a_task_without_a_command_is_refused(registry):
    with pytest.raises(TaskRefused):
        TaskSpec.from_dict({"task_id": "e3"})


def test_a_node_that_cannot_isolate_refuses_everything(tmp_path):
    """Running a stranger's code as the agent's own user is not a default."""
    blocked = Isolation(uid=None, gid=None, user="", unprivileged_fallback=False)
    reg = TaskRegistry(root=tmp_path / "t", isolation=blocked,
                       environments=EnvironmentCache(tmp_path / "e"), total_gpus=0)
    with pytest.raises(TaskRefused) as exc:
        reg.submit(spec("x", ["true"]))
    assert "separate user" in str(exc.value)


def test_the_same_task_is_not_taken_twice(registry):
    registry.submit(spec("d1", [sys.executable, "-c", "import time; time.sleep(5)"]))
    with pytest.raises(TaskRefused):
        registry.submit(spec("d1", ["true"]))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_a_node_whose_volume_is_missing_refuses_but_stays_up(tmp_path, isolation):
    """A crash loop tells the owner nothing and the orchestrator less.

    The node should register, report itself, and name the problem.
    """
    blocked = tmp_path / "not-writable"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        reg = TaskRegistry(root=blocked / "tasks", isolation=isolation,
                           environments=EnvironmentCache(tmp_path / "envs3"), total_gpus=0)
        assert reg.unusable
        assert "--root" in reg.unusable
        with pytest.raises(TaskRefused) as exc:
            reg.submit(spec("v1", ["true"]))
        assert "volume" in str(exc.value) or "Mount" in str(exc.value)
    finally:
        blocked.chmod(0o700)


def test_вывод_короткой_задачи_не_теряется(registry):
    """Лог полон к моменту, когда задача объявлена завершённой.

    На практике поток чтения уже сидит в read() к моменту выхода процесса, так
    что данные не терялись и до того, как завершение стало его дожидаться. Тест
    закрепляет гарантию, а не воспроизводит поломку: полагаться на то, что
    планировщик всегда успеет, — не то же самое, что дождаться.
    """
    for attempt in range(100):
        task = registry.submit(spec(f"fast-{attempt}",
                                    [sys.executable, "-c", "print('маркер')"]))
        assert task.wait(timeout=30)
        assert "маркер" in task.logs(), f"лог пуст на попытке {attempt}"
        registry.release(f"fast-{attempt}")


def test_многострочный_вывод_доезжает_целиком(registry):
    task = registry.submit(spec("many", [
        sys.executable, "-c", "[print('строка', i) for i in range(500)]"]))
    assert task.wait(timeout=30)
    lines = [l for l in task.logs().splitlines() if l.strip()]
    assert len(lines) == 500, f"доехало {len(lines)} строк из 500"


def test_аргумент_с_пробелами_доезжает_одним_куском(registry):
    """Команда — список, а не строка, и разрезать её должен тот, кто её вводит.

    Со стенда: `python -c "print(1)"`, разрезанное по пробелам, доехало тремя
    словами, третье из которых — строковый литерал в кавычках. Python честно
    его вычислил, ничего не напечатал и вышел с нулём: задача `done`, лог пуст.
    """
    task = registry.submit(spec("quoted", [
        sys.executable, "-c", "print('одна строка с пробелами')"]))
    assert task.wait(timeout=30)
    assert task.state == "done", task.logs()
    assert "одна строка с пробелами" in task.logs()


# ------------------------------------------------- короткий каталог для сокетов
# Предел ядра на путь unix-сокета. Не соглашение — bind() отказывает.
SOCKET_PATH_LIMIT = 103


def test_задаче_дают_каталог_под_короткий_путь(registry):
    task = registry.submit(spec(
        "t20", [sys.executable, "-c",
                "import os; print(os.environ['LOOM_TASK_TMP'])"]))
    assert task.wait(timeout=30)
    assert task.logs().strip() == task.directory.inner_scratch
    assert task.directory.scratch.is_dir(), "каталог обещан переменной, но не создан"


def test_в_нём_помещается_unix_сокет(registry):
    """Ровно то, из-за чего он существует.

    Каталог задачи с полным task_id уже съедает почти весь лимит: Ray кладёт
    плазма-сокет в `<tmp>/ray/session_<дата>_<время>_<мкс>_<pid>/sockets/`, и
    на реальном узле это не влезало на 5–7 байт. Диагностируется отвратительно:
    ошибка называет длину пути, а не то, что каталог задачи глубоко лежит.
    """
    program = (
        "import os, socket;"
        "p = os.path.join(os.environ['LOOM_TASK_TMP'],"
        " 'ray', 'session_2026-08-31_11-53-53_288216_84711', 'sockets');"
        "os.makedirs(p, exist_ok=True);"
        "s = socket.socket(socket.AF_UNIX);"
        "s.bind(os.path.join(p, 'plasma_store'));"
        "print('связался, длина пути', len(os.path.join(p, 'plasma_store')))"
    )
    task = registry.submit(spec("t21", [sys.executable, "-c", program]))
    assert task.wait(timeout=30)
    assert task.state == "done", task.logs()
    assert "связался" in task.logs()


def test_длинный_путь_задачи_в_лимит_НЕ_влезает(registry):
    """Обратная сторона: без короткого каталога это не работает.

    Тест на причину, а не на симптом — если каталоги задач когда-нибудь станут
    короче и проблема исчезнет, он об этом скажет."""
    task = registry.submit(spec("t22", [sys.executable, "-c", "pass"]))
    assert task.wait(timeout=30)
    tail = "/ray/session_2026-08-31_11-53-53_288216_84711/sockets/plasma_store"
    assert len(str(task.directory.work) + tail) > SOCKET_PATH_LIMIT
    assert len(task.directory.inner_scratch + tail) <= SOCKET_PATH_LIMIT


def test_короткий_каталог_считается_в_дисковую_квоту(registry):
    """Иначе квота обходится записью не туда."""
    program = (
        "import os;"
        "open(os.path.join(os.environ['LOOM_TASK_TMP'], 'big'), 'wb')"
        ".write(b'x' * (6 * 1024 * 1024));"
        "import time; time.sleep(30)"
    )
    task = registry.submit(spec(
        "t23", [sys.executable, "-c", program],
        resources={"disk_bytes": 1024 * 1024}, timeout_s=30,
    ))
    assert task.wait(timeout=40)
    assert task.state == "cancelled"
    assert "disk quota" in task.error


def test_короткий_каталог_убирается_вместе_с_задачей(registry):
    task = registry.submit(spec("t24", [sys.executable, "-c", "pass"]))
    assert task.wait(timeout=30)
    scratch = task.directory.scratch
    assert scratch.is_dir()
    registry.release("t24")
    assert not scratch.exists(), "остался бы копиться в /tmp навсегда"


def test_сокет_предшественника_не_мешает_следующей_задаче(registry):
    """Тот же id после падения агента: в коротком каталоге остаётся unix-сокет,
    а bind() по занятому пути отказывает. Задача падала бы на «адрес занят»
    из-за предшественника, которого давно нет."""
    bind = (
        "import os, socket;"
        "s = socket.socket(socket.AF_UNIX);"
        "s.bind(os.path.join(os.environ['LOOM_TASK_TMP'], 'sock'));"
        "print('связался')"
    )
    first = registry.submit(spec("t25", [sys.executable, "-c", bind]))
    assert first.wait(timeout=30)
    assert first.state == "done", first.logs()
    assert (first.directory.scratch / "sock").exists(), "сокет должен был остаться"

    # Агент забыл задачу, но каталог остался: так выглядит падение.
    registry._tasks.pop("t25", None)
    registry._release_devices("t25")

    second = registry.submit(spec("t25", [sys.executable, "-c", bind]))
    assert second.wait(timeout=30)
    assert second.state == "done", second.logs()


# ------------------------------------------------------ порт для служащей задачи
def test_задаче_не_выдают_привилегированный_порт(registry):
    """Со стенда: оркестратор шлёт serve_port=1 как «да, служи», агент —
    root, и `bind(1)` у него проходит. Номер уезжал задаче, а та работает
    под обычным пользователем и падала на

        PermissionError: [Errno 13] Permission denied

    — ошибке, которая не называет ни порт, ни того, кто его выбрал.
    """
    from loom_agent.tasks.registry import PRIVILEGED_PORTS, _free_port

    for hint in (1, 80, 443, 1023):
        assert _free_port(hint) >= PRIVILEGED_PORTS, f"выдал {hint}"


def test_свободный_обычный_порт_отдают_как_просили(registry):
    """Отбрасываются только привилегированные: в остальном подсказка узла —
    это способ рангам договориться, не спрашивая никого."""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        wanted = probe.getsockname()[1]
    from loom_agent.tasks.registry import _free_port

    assert _free_port(wanted) == wanted


def test_служащая_задача_получает_рабочий_порт(registry):
    """Сквозь весь путь: то, что попадёт в LOOM_SERVE_PORT, должно
    биндиться из-под задачи."""
    program = (
        "import os, socket;"
        "s = socket.socket();"
        "s.bind(('127.0.0.1', int(os.environ['LOOM_SERVE_PORT'])));"
        "print('занял', os.environ['LOOM_SERVE_PORT'])"
    )
    task = registry.submit(spec("t26", [sys.executable, "-c", program], serve_port=1))
    assert task.wait(timeout=30)
    assert task.state == "done", task.logs()
    assert "занял" in task.logs()


# ------------------------------------------------------------ потолок потоков
def test_задаче_хватает_потоков_на_настоящую_нагрузку(registry):
    """Со стенда: два ранга Ray на одной машине падали на

        RuntimeError: Resource temporarily unavailable

    — это EAGAIN от fork. RLIMIT_NPROC на Linux считает ПОТОКИ, а не процессы,
    и считает их на весь uid, под которым работают все задачи узла. Прежние
    512 выглядели просторно и таковыми не были: Ray поднимает воркер на ядро,
    у каждого свои потоки.

    Оговорка: на macOS этот лимит считает процессы, так что здесь тест ловит
    регрессию только на Linux — то есть там, где узлы и работают.
    """
    program = (
        "import threading, time;"
        "stop = threading.Event();"
        "ts = [threading.Thread(target=stop.wait, daemon=True) for _ in range(600)];"
        "[t.start() for t in ts];"
        "print('потоков', threading.active_count());"
        "stop.set()"
    )
    task = registry.submit(spec("t27", [sys.executable, "-c", program]))
    assert task.wait(timeout=60)
    assert task.state == "done", task.logs()
    assert "потоков" in task.logs()


def test_потолок_остаётся_конечным(monkeypatch):
    """Защита от форк-бомбы никуда не девается — она просто перестала мешать."""
    import importlib

    import loom_agent.tasks.limits as limits

    monkeypatch.setenv("LOOM_TASK_MAX_PROCESSES", "77")
    importlib.reload(limits)
    try:
        assert limits.MAX_PROCESSES == 77
    finally:
        monkeypatch.delenv("LOOM_TASK_MAX_PROCESSES", raising=False)
        importlib.reload(limits)
    assert 1024 <= limits.MAX_PROCESSES < 1_000_000, "потолок должен остаться конечным"


def test_потомки_не_переживают_задачу_вышедшую_самостоятельно(registry, tmp_path):
    """Со стенда: `ray start` разворачивает демонов и выходит сам. Агент
    сносил группу процессов только при СНЯТИИ, так что после самостоятельного
    выхода сотни процессов Ray оставались жить на чужой машине. Следующая
    задача упиралась в них лимитом потоков ещё до старта:

        RuntimeError: can't start new thread
    """
    marker = tmp_path / "потомок-жив"
    program = (
        "import subprocess, sys, os;"
        f"subprocess.Popen([sys.executable, '-c', "
        f"\"import time, pathlib;\\npathlib.Path({str(marker)!r}).touch()\\n"
        f"time.sleep(120)\"]);"
        "print('запустил потомка и выхожу')"
    )
    task = registry.submit(spec("t28", [sys.executable, "-c", program]))
    assert task.wait(timeout=30)
    assert task.state == "done", task.logs()

    deadline = time.time() + 15
    while time.time() < deadline and not marker.exists():
        time.sleep(0.2)
    assert marker.exists(), "потомок даже не стартовал — тест ничего не проверяет"

    # Группа снесена вместе с задачей: живых в ней не осталось.
    from loom_agent.tasks.runner import _group_alive

    assert task._group is not None
    deadline = time.time() + 15
    while time.time() < deadline and _group_alive(task._group):
        time.sleep(0.2)
    assert not _group_alive(task._group), "потомок пережил задачу"


def test_обычная_задача_завершается_как_прежде(registry):
    """Уборка не должна портить нормальный путь: код возврата и состояние
    остаются теми же, что и были."""
    task = registry.submit(spec("t29", [sys.executable, "-c", "print('готово')"]))
    assert task.wait(timeout=30)
    assert task.state == "done"
    assert task.exit_code == 0
    assert "готово" in task.logs()
