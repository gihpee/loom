"""Адрес канала. Клиент почти наверняка скопирует его из панели, а там он
со схемой — значит принимать надо оба вида."""

from __future__ import annotations

from loom_connect.websocket import endpoint


def test_голый_адрес_идёт_под_wss():
    assert endpoint("1.2.3.4:8000", "group-a") == "wss://1.2.3.4:8000/connect/group-a"


def test_схема_из_панели_понимается():
    assert endpoint("http://1.2.3.4:8000", "group-a").startswith("ws://")
    assert endpoint("https://loom.example", "group-a").startswith("wss://")


def test_лишний_слеш_не_ломает_адрес():
    assert endpoint("https://loom.example/", "group-a") == \
        "wss://loom.example/connect/group-a"


def test_имя_кластера_экранируется():
    """Идентификатор приходит снаружи; в адрес он попадает как данные."""
    assert "/connect/group%2Fa" in endpoint("h", "group/a")


def test_insecure_переключает_схему():
    assert endpoint("1.2.3.4:8000", "g", insecure=True).startswith("ws://")


def test_неascii_токен_отвергается_внятно(capsys):
    """Иначе это всплывает из глубины библиотеки как «invalid
    X-Loom-Admin-Token header» — про заголовок, а не про то, что человек
    скопировал токен вместе с лишним."""
    from loom_connect.main import main

    assert main(["1.2.3.4:8000", "group-a", "--token", "токен"]) == 2
    assert "скопирован с лишним" in capsys.readouterr().err
