"""Повторы при сборке кластера — без настоящего Ray.

Отдельно от test_cluster.py: тот запускает настоящий кластер и целиком закрыт
`importorskip("ray")`. Здесь проверяется логика повторов, которой Ray не нужен,
и закрывать её тем же условием значило бы не проверять её нигде: на машине без
Ray файл пропускается молча.
"""

from __future__ import annotations

import pytest


def test_зависшая_попытка_считается_неудачной_и_повторяется(monkeypatch):
    """Так это и падало у живого кластера.

    Между машинами порт головы держит агент: он принимает соединение мгновенно,
    ещё до того, как на том конце появится GCS. «Соединение отвергнуто», на
    которое рассчитан быстрый провал, не приходит никогда — Ray просто ждёт.
    Раньше TimeoutExpired пролетал мимо повторов, и единственная попытка съедала
    весь срок.
    """
    import subprocess as sp

    from looma_ray import cluster

    monkeypatch.setattr(cluster, "JOIN_RETRY_S", 0.01)
    attempts = []

    class Result:
        returncode, stdout, stderr = 0, "", ""

    def hangs_then_works(*_a, **_k):
        attempts.append(1)
        if len(attempts) < 3:
            raise sp.TimeoutExpired(cmd=["ray", "start"], timeout=60)
        return Result()

    monkeypatch.setattr(sp, "run", hangs_then_works)
    cluster._run_start(["ray", "start"], rank=1, retries=4, timeout_s=60)
    assert len(attempts) == 3, "зависание должно быть попыткой, а не концом"


def test_все_попытки_зависли_дают_внятный_отказ(monkeypatch):
    """Не стек вызовов из subprocess.py: по нему не понять, что произошло, и
    человек идёт разбираться не туда."""
    import subprocess as sp

    from looma_ray import cluster

    monkeypatch.setattr(cluster, "JOIN_RETRY_S", 0.01)

    def always_hangs(*_a, **_k):
        raise sp.TimeoutExpired(cmd=["ray", "start"], timeout=60)

    monkeypatch.setattr(sp, "run", always_hangs)
    with pytest.raises(cluster.ClusterRefused) as отказ:
        cluster._run_start(["ray", "start"], rank=1, retries=2, timeout_s=60)
    assert "ранга 1" in str(отказ.value)
    assert "не ответил за 60с" in str(отказ.value)


def test_срок_одной_попытки_меньше_бюджета_всех(monkeypatch):
    """Числа обязаны быть согласованы. Раньше на попытку отводилось 600 секунд
    при пяти повторах по 15 — одна попытка перекрывала весь замысел, и повторов
    не случалось ни разу."""
    from looma_ray import cluster

    бюджет = cluster.JOIN_ATTEMPTS * cluster.JOIN_RETRY_S
    assert cluster.START_TIMEOUT_S <= бюджет, (
        "попытка не должна длиться дольше, чем все паузы между попытками")
    assert cluster.START_TIMEOUT_S < cluster.JOIN_WAIT_S


def test_срок_попытки_передаётся_в_subprocess(monkeypatch):
    """Иначе он остался бы украшением: значение есть, а на вызов не влияет."""
    import subprocess as sp

    from looma_ray import cluster

    видели = {}

    class Result:
        returncode, stdout, stderr = 0, "", ""

    def capture(*_a, **kw):
        видели.update(kw)
        return Result()

    monkeypatch.setattr(sp, "run", capture)
    cluster._run_start(["ray", "start"], rank=1, retries=0, timeout_s=42)
    assert видели.get("timeout") == 42


def test_зависший_ray_доносит_свои_слова(monkeypatch):
    """Единственный след того, чем он был занят.

    Вывод не льётся наружу, пока процесс жив, а живёт он до самого таймаута.
    Первая версия этой ветки выбрасывала вывод и подставляла догадку о причине
    («порт головы открыт») — догадка оказалась ещё и неверной: порт был закрыт,
    соединение отвергалось. Человек остался без единого следа.
    """
    import subprocess as sp

    from looma_ray import cluster

    monkeypatch.setattr(cluster, "JOIN_RETRY_S", 0.01)

    def hangs(*_a, **_k):
        raise sp.TimeoutExpired(
            cmd=["ray", "start"], timeout=60,
            output="подняли дашборд\n", stderr="RuntimeError: version mismatch\n")

    monkeypatch.setattr(sp, "run", hangs)
    with pytest.raises(cluster.ClusterRefused) as отказ:
        cluster._run_start(["ray", "start"], rank=0, retries=0, timeout_s=60)
    сказано = str(отказ.value)
    assert "version mismatch" in сказано, "слова ray обязаны дойти"
    assert "подняли дашборд" in сказано


def test_молчаливое_зависание_так_и_называется(monkeypatch):
    """Пустой вывод — тоже сведение, и врать про него нельзя."""
    import subprocess as sp

    from looma_ray import cluster

    def hangs(*_a, **_k):
        raise sp.TimeoutExpired(cmd=["ray", "start"], timeout=60)

    monkeypatch.setattr(sp, "run", hangs)
    with pytest.raises(cluster.ClusterRefused) as отказ:
        cluster._run_start(["ray", "start"], rank=0, retries=0, timeout_s=60)
    assert "и ничего не сказал" in str(отказ.value)


def test_байтовый_вывод_тоже_доносится(monkeypatch):
    """text=True не гарантирован для полей TimeoutExpired во всех версиях."""
    import subprocess as sp

    from looma_ray import cluster

    def hangs(*_a, **_k):
        raise sp.TimeoutExpired(cmd=["ray", "start"], timeout=60,
                                stderr=b"GCS \xd0\xbd\xd0\xb5 \xd0\xb2\xd1\x81\xd1\x82\xd0\xb0\xd0\xbb\n")

    monkeypatch.setattr(sp, "run", hangs)
    with pytest.raises(cluster.ClusterRefused) as отказ:
        cluster._run_start(["ray", "start"], rank=0, retries=0, timeout_s=60)
    assert "GCS не встал" in str(отказ.value)
