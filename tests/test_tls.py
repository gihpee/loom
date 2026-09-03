"""Шифрование канала, по которому узлы получают команды.

По нему едут секрет ключа подключения и команды запуска задач — то есть право
исполнить код на чужой машине. Всё, что здесь проверяется, отвечает на два
вопроса: включилось ли шифрование на самом деле и переживёт ли оно перевыпуск
сертификата.
"""

from __future__ import annotations

import os
import time

import pytest

from looma.orchestrator.tls import (
    CertPaths,
    RotatingCertificate,
    TlsUnusable,
    read_pair,
)


def pair(tmp_path, chain=b"---chain---", key=b"---key---") -> CertPaths:
    (tmp_path / "fullchain.pem").write_bytes(chain)
    (tmp_path / "privkey.pem").write_bytes(key)
    return CertPaths(certificate=str(tmp_path / "fullchain.pem"),
                     private_key=str(tmp_path / "privkey.pem"))


# ------------------------------------------------------------ включение
def test_без_путей_шифрования_нет():
    assert CertPaths().configured is False


def test_нужны_оба_файла():
    """Один без другого — это не «наполовину включено», это ничего."""
    assert CertPaths(certificate="/есть.pem").configured is False
    assert CertPaths(private_key="/есть.pem").configured is False


def test_пути_берутся_из_окружения(monkeypatch):
    monkeypatch.setenv("LOOMA_TLS_CERT", "/c.pem")
    monkeypatch.setenv("LOOMA_TLS_KEY", "/k.pem")
    assert CertPaths.from_env().configured


# --------------------------------------------------------------- чтение
def test_порядок_ключ_потом_цепочка(tmp_path):
    """Перепутанные местами файлы дают ошибку из глубины OpenSSL, в которой про
    сертификаты ни слова."""
    key, chain = read_pair(pair(tmp_path))
    assert key == b"---key---" and chain == b"---chain---"


def test_отсутствующий_файл_называется_поимённо(tmp_path):
    paths = CertPaths(certificate=str(tmp_path / "нет.pem"),
                      private_key=str(tmp_path / "тоже-нет.pem"))
    with pytest.raises(TlsUnusable, match="нет.pem"):
        read_pair(paths)


def test_названный_но_негодный_сертификат_это_отказ(tmp_path):
    """Канал остаётся открытым только когда сертификат не назван вовсе. Здесь
    он назван — значит оператор просил шифрование."""
    with pytest.raises(TlsUnusable, match="ещё не выпущен"):
        read_pair(pair(tmp_path, chain=b""))


# ------------------------------------------------------------ перевыпуск
def test_пара_читается_один_раз(tmp_path):
    rotating = RotatingCertificate(pair(tmp_path))
    assert rotating.current() == rotating.current()
    assert rotating.changed() is False


def test_новый_файл_подхватывается(tmp_path):
    """Let's Encrypt меняет файл каждые 60 дней, а gRPC берёт учётные данные
    один раз при создании сервера: без этого узлы отвалились бы все разом на
    90-й день."""
    paths = pair(tmp_path)
    rotating = RotatingCertificate(paths)
    assert rotating.current()[1] == b"---chain---"

    later = time.time() + 10
    (tmp_path / "fullchain.pem").write_bytes(b"---renewed-chain---")
    os.utime(tmp_path / "fullchain.pem", (later, later))

    assert rotating.changed() is True
    assert rotating.current()[1] == b"---renewed-chain---"
    assert rotating.changed() is False


def test_исчезнувший_файл_не_считается_сменой(tmp_path):
    """certbot мог застать нас в середине замены; старая пара ещё годна."""
    paths = pair(tmp_path)
    rotating = RotatingCertificate(paths)
    rotating.current()
    os.remove(paths.certificate)
    assert rotating.changed() is False


# ------------------------------------------------- ключ несёт способ связи
def test_ключ_с_шифрованием_отличается_от_обычного(tmp_path):
    from looma.orchestrator.keys import KeyStore, decode_key

    plain = KeyStore(public_address="looma.example:9000",
                     path=tmp_path / "plain.json").issue()
    secure = KeyStore(public_address="looma.example:9000",
                      path=tmp_path / "secure.json", tls=True).issue()

    assert decode_key(plain.encode())["tls"] is False
    assert decode_key(secure.encode())["tls"] is True


def test_агент_читает_способ_из_ключа(tmp_path):
    """Угадывание по виду адреса означало бы, что узел молча уходит в открытый
    канал там, где оркестратор ждал шифрованный."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
    from looma_agent.identity import parse_join_key

    from looma.orchestrator.keys import KeyStore

    secure = KeyStore(public_address="loomafloat.ru:9000",
                      path=tmp_path / "k.json", tls=True).issue()
    parsed = parse_join_key(secure.encode())
    assert parsed.tls is True and parsed.address == "loomafloat.ru:9000"


def test_ключи_прежнего_образца_продолжают_работать(tmp_path):
    """Строка ключа у владельца узла на руках, и переписать её мы не можем."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
    from looma_agent.identity import parse_join_key

    from looma.orchestrator.keys import KeyStore

    old = KeyStore(public_address="203.0.113.7:9000", path=tmp_path / "k.json").issue()
    assert parse_join_key(old.encode()).tls is False
