"""Пароли, токены сессий и ключи API.

Три разные вещи, и хранятся они по-разному не для красоты: пароль человек
выбирает сам и потому предсказуем, токен выбираем мы из системной случайности.
"""

from __future__ import annotations

import time

import pytest

from looma.accounts.secrets_ import (
    API_KEY_PREFIX,
    BadHash,
    hash_password,
    hash_token,
    key_hint,
    needs_rehash,
    new_api_key,
    new_session_token,
    token_matches,
    verify_password,
)


# ------------------------------------------------------------------ пароли
def test_пароль_проверяется():
    stored = hash_password("правильная лошадь батарейка скоба")
    assert verify_password("правильная лошадь батарейка скоба", stored)
    assert not verify_password("почти правильная лошадь", stored)


def test_одинаковые_пароли_дают_разные_хэши():
    """Иначе по базе видно, у кого пароли совпадают — а это половина работы
    по подбору."""
    assert hash_password("одно и то же") != hash_password("одно и то же")


def test_пустой_пароль_не_хэшируется():
    with pytest.raises(ValueError):
        hash_password("")


def test_параметры_лежат_внутри_строки():
    """Подняв их завтра, мы должны уметь проверить пароли, посчитанные вчера."""
    stored = hash_password("пароль")
    assert stored.startswith("scrypt$n=")
    assert stored.count("$") == 3


def test_старые_параметры_видно():
    свежий = hash_password("пароль")
    старый = свежий.replace("n=32768", "n=16384")
    assert needs_rehash(свежий) is False
    assert needs_rehash(старый) is True


def test_испорченная_запись_это_не_неподошедший_пароль():
    """Молча считать её неподошедшим паролем нельзя: пользователь будет
    доказывать, что пароль верный, и будет прав."""
    with pytest.raises(BadHash):
        verify_password("пароль", "мусор")
    with pytest.raises(BadHash, match="неизвестный алгоритм"):
        verify_password("пароль", "md5$n=1,r=1,p=1$c29sdA$aGFzaA")


def test_пароль_проверяется_не_мгновенно():
    """Дорогая проверка — единственная защита утёкшей базы от перебора по
    словарю. Порог намеренно низкий: тест ловит «параметры сбросили в единицу»,
    а не меряет скорость машины."""
    stored = hash_password("пароль")
    начало = time.perf_counter()
    verify_password("пароль", stored)
    assert time.perf_counter() - начало > 0.005


# ------------------------------------------------------- токены и ключи
def test_токены_не_повторяются():
    assert len({new_session_token() for _ in range(200)}) == 200


def test_ключ_узнаётся_по_началу():
    key, _stored = new_api_key()
    assert key.startswith(API_KEY_PREFIX)


def test_в_базу_едет_хэш_а_не_ключ():
    """Восстановить потерянный ключ нельзя — и интерфейс обязан сказать об
    этом до того, как пользователь закроет окно."""
    key, stored = new_api_key()
    assert key not in stored
    assert token_matches(key, stored)
    assert not token_matches(key + "x", stored)


def test_подсказка_показывает_начало_но_не_секрет():
    key, _stored = new_api_key()
    hint = key_hint(key)
    assert key.startswith(hint)
    assert len(hint) < len(key)


def test_сравнение_переживает_пустой_хэш():
    """Пользователь без ключей не должен давать исключение при проверке."""
    assert token_matches("looma_sk_что-то", "") is False


def test_хэш_токена_повторяем():
    assert hash_token("одно и то же") == hash_token("одно и то же")
