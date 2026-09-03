"""Кого куда пускает приложение.

Проверяется не сам разбор токенов (это tests/test_auth.py), а то, что слой
защиты стоит на всех маршрутах разом. Двадцать пять охранников, расставленных
руками, — это гарантированно один забытый маршрут.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from looma.api.app import create_app


class Settings:
    admin_token = "s3cret-token"


class NoToken:
    admin_token = ""


def client(config=None):
    return TestClient(create_app(config=config or Settings()))


# --------------------------------------------------- дыра, которую закрыли
def test_без_настроенного_токена_никого_не_пускает():
    """Раньше проверка выглядела как `bool(token) and token != admin_token`:
    при пустом токене она возвращала False и пускала любого. Оркестратор,
    поднятый без переменной окружения и выставленный в интернет, не имел
    никакой защиты и ничем этого не выдавал."""
    answer = client(NoToken()).get("/admin/agents")
    assert answer.status_code == 401


# ----------------------------------------------------------- админские
def test_админский_маршрут_без_токена_отвергается():
    assert client().get("/admin/agents").status_code == 401


def test_админский_маршрут_с_токеном_проходит():
    answer = client().get("/admin/agents",
                          headers={"X-Looma-Admin-Token": "s3cret-token"})
    assert answer.status_code != 401


def test_неверный_токен_отвергается():
    assert client().get("/admin/agents",
                        headers={"X-Looma-Admin-Token": "wrong-token"}).status_code == 401


@pytest.mark.parametrize("path", [
    "/admin/agents", "/admin/tasks", "/admin/groups", "/admin/keys",
    "/admin/release", "/admin/connect", "/admin/accounts",
])
def test_защищены_все_админские_маршруты(path):
    """Смысл общего слоя: маршрут, добавленный завтра, защищён с рождения."""
    assert client().get(path).status_code == 401


# ------------------------------------------------------- клиентские и /v1
def test_инференс_требует_представиться():
    """До этого /v1 был открыт: кто дотянулся до порта, тот и пользовался."""
    answer = client().post("/v1/chat/completions", json={"model": "нет"})
    assert answer.status_code == 401


def test_кабинетные_маршруты_требуют_входа():
    assert client().get("/api/keys").status_code == 401
    assert client().get("/api/me").status_code == 401


def test_аварийного_токена_хватает_и_клиентским():
    answer = client().get("/api/me",
                          headers={"X-Looma-Admin-Token": "s3cret-token"})
    assert answer.status_code == 200
    assert answer.json()["how"] == "emergency"


def test_без_базы_вход_объясняет_причину():
    """«503 без слов» заставляет гадать, дело в пароле или в том, что база не
    поднялась."""
    answer = client().post("/api/session", json={"email": "a@b.ru", "password": "x"})
    assert answer.status_code == 503
    assert "базы" in answer.json()["error"]["message"]


# ---------------------------------------------------------------- открытое
def test_корень_остаётся_открытым():
    """По нему пойдёт лендинг: требовать вход на публичной странице незачем."""
    assert client().get("/").status_code != 401


# --------------------------------------------------- аренда клиентом
def test_клиентская_аренда_не_на_админском_маршруте():
    """Раньше кабинет звал /admin/ray и упирался в 403: админский префикс
    клиенту закрыт целиком."""
    app = create_app(config=Settings())
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/api/compute" in paths
    assert "/api/compute/{group_id}" in paths


def test_аренда_требует_входа():
    assert client().post("/api/compute", json={"size": 1}).status_code == 401


def test_список_кластеров_требует_входа():
    assert client().get("/api/compute").status_code == 401


def test_снятие_чужого_кластера_требует_входа():
    assert client().delete("/api/compute/чужой").status_code == 401
