"""Пароли, токены сессий и ключи API — то, что нельзя хранить как есть.

Три разные вещи, и хранятся они по-разному не для красоты.

**Пароль** человек выбирает сам, а значит он предсказуем: словарь, дата, имя
кота. Утёкшую базу перебирают по словарю, и единственная защита — сделать одну
проверку дорогой. Отсюда scrypt с параметрами, при которых проверка занимает
десятки миллисекунд: пользователь этого не заметит, а перебор дорожает во
столько же раз.

**Токен сессии и ключ API** выбираем мы, из системного источника случайности.
Перебирать 256 бит энтропии бессмысленно при любой скорости, поэтому медленная
функция здесь не нужна — достаточно обычного sha256. Медленная была бы даже
вредна: её пришлось бы считать на каждый запрос.

Общее у всех трёх — сравнение за постоянное время. Обычное `==` сравнивает
побайтово и выходит на первом различии, а значит отвечает тем быстрее, чем
раньше ошибка. По времени ответа секрет подбирается посимвольно.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from typing import Tuple

# Параметры scrypt. Хранятся ВНУТРИ строки хэша, а не здесь: подняв их завтра,
# мы должны уметь проверить пароли, посчитанные вчера. Значение из RFC 7914 для
# интерактивной проверки, поднятое по памяти под нынешнее железо.
SCRYPT_N = 1 << 15
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
KEY_BYTES = 32

#: Ключ API узнаётся по началу — и в интерфейсе, и в чужом коде, куда его
#: случайно вставили. Секретная часть идёт после.
API_KEY_PREFIX = "looma_sk_"
#: Сколько символов ключа показывать после создания. Дальше он не показывается
#: никогда: в базе лежит только хэш.
API_KEY_HINT = 8


class BadHash(ValueError):
    """Строка хэша не разбирается. Это не «пароль не подошёл» — это испорченная
    запись, и молча считать её неподошедшим паролем нельзя."""


# ------------------------------------------------------------------ пароли
def hash_password(password: str) -> str:
    """Пароль в строку, которую не жалко положить в базу."""
    if not password:
        raise ValueError("пустой пароль не хэшируется")
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return "$".join([
        "scrypt",
        f"n={SCRYPT_N},r={SCRYPT_R},p={SCRYPT_P}",
        _b64(salt),
        _b64(digest),
    ])


def verify_password(password: str, stored: str) -> bool:
    """Подходит ли пароль. Параметры берутся из самой строки."""
    algorithm, params, salt, digest = _split(stored)
    if algorithm != "scrypt":
        raise BadHash(f"неизвестный алгоритм {algorithm!r}")
    n, r, p = _params(params)
    computed = _scrypt(password, salt, n, r, p)
    return hmac.compare_digest(computed, digest)


def needs_rehash(stored: str) -> bool:
    """Считан ли пароль по устаревшим параметрам. Пересчитывать его можно
    только в момент удачного входа — другого раза, когда пароль есть в
    открытом виде, не будет."""
    try:
        _algorithm, params, _salt, _digest = _split(stored)
        return _params(params) != (SCRYPT_N, SCRYPT_R, SCRYPT_P)
    except BadHash:
        return True


# ------------------------------------------------------- токены и ключи
def new_session_token() -> str:
    """Токен сессии. Живёт в cookie и больше нигде."""
    return secrets.token_urlsafe(32)


def new_api_key() -> Tuple[str, str]:
    """Ключ API: (что показать один раз, что положить в базу).

    Показывается ровно один раз, при создании. В базе — только хэш, поэтому
    восстановить потерянный ключ нельзя, и интерфейс обязан об этом сказать
    ДО того, как пользователь закроет окно.
    """
    key = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return key, hash_token(key)


def hash_token(token: str) -> str:
    """Хэш токена или ключа. Быстрый намеренно — см. шапку модуля."""
    return hashlib.sha256(token.encode()).hexdigest()


def token_matches(token: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_token(token), stored_hash or "")


def key_hint(key: str) -> str:
    """Начало ключа — чтобы владелец узнал свой среди нескольких."""
    body = key[len(API_KEY_PREFIX):] if key.startswith(API_KEY_PREFIX) else key
    return API_KEY_PREFIX + body[:API_KEY_HINT]


# ------------------------------------------------------------------ внутри
def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p,
                          dklen=KEY_BYTES, maxmem=2 * n * r * 128 + (1 << 20))


def _split(stored: str) -> Tuple[str, str, bytes, bytes]:
    parts = (stored or "").split("$")
    if len(parts) != 4:
        raise BadHash("строка хэша должна состоять из четырёх частей")
    try:
        return parts[0], parts[1], _unb64(parts[2]), _unb64(parts[3])
    except Exception as exc:
        raise BadHash(f"не разобрать соль или хэш: {exc}") from None


def _params(text: str) -> Tuple[int, int, int]:
    try:
        found = dict(piece.split("=", 1) for piece in text.split(","))
        return int(found["n"]), int(found["r"]), int(found["p"])
    except Exception as exc:
        raise BadHash(f"не разобрать параметры {text!r}: {exc}") from None


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
