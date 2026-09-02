"""Веса, которые скачиваются один раз на узел.

Без кэша каждое развёртывание начинается с закачки: HOME у задачи указывает в
её собственный каталог, HuggingFace кладёт кэш туда же, и каталог живёт ровно
столько же, сколько задача.
"""

from __future__ import annotations

import os
import time

import pytest

from loom_agent.tasks.models import ModelCache


def repository(cache: ModelCache, name: str, *, size: int, age_s: float = 0.0):
    """Репозиторий в кэше, как его раскладывает huggingface_hub."""
    base = cache.root / "hub" / f"models--{name}"
    (base / "blobs").mkdir(parents=True, exist_ok=True)
    (base / "snapshots" / "abc").mkdir(parents=True, exist_ok=True)
    (base / "blobs" / "weights").write_bytes(b"x" * size)
    if age_s:
        when = time.time() - age_s
        for path in (base, base / "snapshots", base / "snapshots" / "abc"):
            os.utime(path, (when, when))
    return base


# ------------------------------------------------------------- для задачи
def test_задача_получает_путь_к_общему_кэшу(tmp_path):
    cache = ModelCache(tmp_path / "models")
    env = cache.env("/var/lib/loom/models")
    assert env["HF_HOME"] == "/var/lib/loom/models"


def test_задаются_все_имена_переменных(tmp_path):
    """Библиотеки читают то одно, то другое в зависимости от версии. Разойдись
    они — часть файлов уедет в кэш, а часть мимо, и заметно это станет по
    времени запуска, а не по ошибке."""
    env = ModelCache(tmp_path / "models").env("/кэш")
    assert set(env) == {"HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"}
    assert env["HF_HUB_CACHE"] == env["HUGGINGFACE_HUB_CACHE"] == "/кэш/hub"


def test_недоступный_кэш_не_ломает_задачу(tmp_path):
    """Без кэша задача просто скачает веса себе, как раньше. Ронять из-за
    этого работу узла хуже, чем медленно её делать."""
    занято = tmp_path / "файл"
    занято.write_text("не каталог")
    cache = ModelCache(занято)
    assert cache.unusable
    assert cache.env("/что-нибудь") == {}


# -------------------------------------------------------------- вытеснение
def test_в_пределах_квоты_ничего_не_трогаем(tmp_path):
    cache = ModelCache(tmp_path / "models", quota_bytes=1000)
    repository(cache, "a", size=100, age_s=99999)
    assert cache.sweep() == 0
    assert cache.total_bytes() == 100


def test_за_квотой_уходит_самое_давнее(tmp_path):
    cache = ModelCache(tmp_path / "models", quota_bytes=150, grace_s=1)
    старое = repository(cache, "old", size=100, age_s=99999)
    свежее = repository(cache, "new", size=100, age_s=10)
    cache.sweep()
    assert not старое.exists()
    assert свежее.exists()


def test_недавно_тронутое_не_трогаем_даже_за_квотой(tmp_path):
    """Возможно, оно прямо сейчас качается или читается стадией. Снести такое
    значит уронить загрузку."""
    cache = ModelCache(tmp_path / "models", quota_bytes=10, grace_s=3600)
    свежее = repository(cache, "new", size=100, age_s=5)
    assert cache.sweep() == 0
    assert свежее.exists()


def test_чужие_каталоги_не_считаются_и_не_сносятся(tmp_path):
    """В кэше HuggingFace лежит не только `models--*`."""
    cache = ModelCache(tmp_path / "models", quota_bytes=1, grace_s=0)
    (cache.root / "hub" / "datasets--что-то").mkdir(parents=True)
    (cache.root / "hub" / "datasets--что-то" / "файл").write_bytes(b"x" * 500)
    cache.sweep()
    assert (cache.root / "hub" / "datasets--что-то").exists()
    assert cache.total_bytes() == 0


def test_ссылки_не_считаются_дважды(tmp_path):
    """Снимки — это ссылки на blob'ы; сложить те и другие значило бы посчитать
    веса дважды и вытеснять вдвое раньше нужного."""
    cache = ModelCache(tmp_path / "models")
    base = repository(cache, "a", size=1000)
    link = base / "snapshots" / "abc" / "weights"
    link.symlink_to(base / "blobs" / "weights")
    assert cache.total_bytes() == 1000


def test_давность_считается_по_снимкам_а_не_по_каталогу(tmp_path):
    """Каталог репозитория меняется только при добавлении ревизии: по нему
    давно скачанная, но ежедневно используемая модель выглядела бы забытой."""
    cache = ModelCache(tmp_path / "models", quota_bytes=150, grace_s=1)
    давно_скачано = repository(cache, "used", size=100, age_s=99999)
    repository(cache, "other", size=100, age_s=99998)
    свежий = давно_скачано / "snapshots" / "abc"
    os.utime(свежий, None)                       # им только что пользовались
    cache.sweep()
    assert давно_скачано.exists()


def test_квота_ноль_отключает_вытеснение(tmp_path):
    cache = ModelCache(tmp_path / "models", quota_bytes=0)
    repository(cache, "a", size=1000, age_s=99999)
    assert cache.sweep() == 0


def test_отчёт_показывает_занятое_и_квоту(tmp_path):
    cache = ModelCache(tmp_path / "models", quota_bytes=500)
    repository(cache, "a", size=100)
    assert cache.snapshot() == {"unusable": "", "bytes": 100, "quota_bytes": 500}


# ------------------------------------------------------- что видит задача
class _Directory:
    def __init__(self, rootfs=None):
        self.rootfs = rootfs


def задача(cache, *, rootfs=None):
    from loom_agent.tasks.runner import Task

    made = Task.__new__(Task)
    made.models = cache
    made.directory = _Directory(rootfs)
    return made


def test_обычная_задача_видит_общий_кэш(tmp_path):
    cache = ModelCache(tmp_path / "models")
    env = задача(cache)._model_cache_env()
    assert env["HF_HOME"] == str(cache.root)


def test_задаче_в_образе_кэш_не_даётся(tmp_path):
    """Она видит только свой rootfs, а внешний путь ей ничего не говорит.
    Показать каталог снаружи можно было бы только монтированием — а это те
    самые привилегии, которых вся конструкция избегает."""
    cache = ModelCache(tmp_path / "models")
    assert задача(cache, rootfs=tmp_path / "rootfs")._model_cache_env() == {}


def test_без_кэша_задача_работает_как_раньше(tmp_path):
    assert задача(None)._model_cache_env() == {}
