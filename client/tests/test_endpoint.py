"""Адрес канала. Клиент почти наверняка скопирует его из панели, а там он
со схемой — значит принимать надо оба вида."""

from __future__ import annotations

from looma_connect.websocket import endpoint


def test_голый_адрес_идёт_под_wss():
    assert endpoint("1.2.3.4:8000", "group-a") == "wss://1.2.3.4:8000/connect/group-a"


def test_схема_из_панели_понимается():
    assert endpoint("http://1.2.3.4:8000", "group-a").startswith("ws://")
    assert endpoint("https://looma.example", "group-a").startswith("wss://")


def test_лишний_слеш_не_ломает_адрес():
    assert endpoint("https://looma.example/", "group-a") == \
        "wss://looma.example/connect/group-a"


def test_имя_кластера_экранируется():
    """Идентификатор приходит снаружи; в адрес он попадает как данные."""
    assert "/connect/group%2Fa" in endpoint("h", "group/a")


def test_insecure_переключает_схему():
    assert endpoint("1.2.3.4:8000", "g", insecure=True).startswith("ws://")


def test_неascii_токен_отвергается_внятно(capsys):
    """Иначе это всплывает из глубины библиотеки как «invalid
    X-Looma-Admin-Token header» — про заголовок, а не про то, что человек
    скопировал токен вместе с лишним."""
    from looma_connect.main import main

    assert main(["1.2.3.4:8000", "group-a", "--token", "токен"]) == 2
    assert "скопирован с лишним" in capsys.readouterr().err


def test_голый_адрес_с_insecure_идёт_по_ws():
    """Панель без TLS печатает команду с --insecure; без него клиент упирается
    в «WRONG_VERSION_NUMBER» — сообщение про TLS там, где TLS нет."""
    assert endpoint("1.2.3.4:8080", "g", insecure=True) == \
        "ws://1.2.3.4:8080/connect/g"
