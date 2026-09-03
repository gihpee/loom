"""Сертификат для канала, по которому узлы получают команды.

По этому каналу едут две вещи, и обе стоят шифрования больше, чем что-либо
ещё в системе: секрет ключа подключения при регистрации и команды запуска
задач. Перехвативший канал может выдать себя за узел или, что хуже, отправить
узлу свою задачу — то есть исполнить произвольный код на чужой машине.

Включается наличием файлов, а не флагом. Флаг умеет разойтись с
действительностью: «TLS включён», а сертификата нет — и процесс либо падает на
старте, либо, что хуже, молча слушает открытым текстом. Есть пути к
сертификату — значит TLS; нет — открытый канал для разработки и тестов.

Перевыпуск переживается без перезапуска. Let's Encrypt меняет файл каждые 60
дней, а gRPC берёт учётные данные один раз при создании сервера: без
обновления на лету узлы отвалились бы все разом на 90-й день, и до этого дня
ничто бы об этом не сказало.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("looma.orchestrator.tls")


class TlsUnusable(RuntimeError):
    """Сертификат назван, но взять его нельзя. Поднимать канал открытым в этом
    случае нельзя тем более: оператор просил шифрование."""


@dataclass(frozen=True)
class CertPaths:
    """Где лежит пара. Пустые пути означают «шифрования нет»."""

    certificate: str = ""
    private_key: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.certificate and self.private_key)

    @classmethod
    def from_env(cls) -> "CertPaths":
        return cls(
            certificate=os.environ.get("LOOMA_TLS_CERT", "").strip(),
            private_key=os.environ.get("LOOMA_TLS_KEY", "").strip(),
        )


def read_pair(paths: CertPaths) -> Tuple[bytes, bytes]:
    """Прочитать (ключ, цепочку). Отказ называет файл, а не «permission denied».

    Порядок именно такой, потому что его требует gRPC, и перепутанные местами
    файлы дают ошибку из глубины OpenSSL, в которой про сертификаты ни слова.
    """
    try:
        chain = _read(paths.certificate)
        key = _read(paths.private_key)
    except OSError as exc:
        raise TlsUnusable(
            f"не прочитать {exc.filename}: {exc.strerror}. Канал узлов остаётся "
            "без шифрования только когда сертификат не назван вовсе — здесь он "
            "назван, поэтому это отказ, а не предупреждение") from None
    if not chain or not key:
        raise TlsUnusable(
            f"{paths.certificate} или {paths.private_key} пуст; сертификат ещё "
            "не выпущен?")
    return key, chain


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


class RotatingCertificate:
    """Пара, перечитываемая когда файл на диске изменился.

    gRPC спрашивает её на новых соединениях. Читать файлы на каждое соединение
    незачем — сравниваем время изменения, и это же делает перевыпуск заметным
    в логе: строчка появится ровно тогда, когда certbot дописал новый файл.
    """

    def __init__(self, paths: CertPaths) -> None:
        self.paths = paths
        self._lock = threading.Lock()
        self._stamp: Optional[Tuple[float, float]] = None
        self._pair: Optional[Tuple[bytes, bytes]] = None

    def current(self) -> Tuple[bytes, bytes]:
        """Пара, годная прямо сейчас."""
        with self._lock:
            stamp = self._mtimes()
            if self._pair is None or stamp != self._stamp:
                if self._pair is not None:
                    logger.info("сертификат перевыпущен, беру новый: %s",
                                self.paths.certificate)
                self._pair = read_pair(self.paths)
                self._stamp = stamp
            return self._pair

    def changed(self) -> bool:
        """Сменился ли файл с прошлого чтения. Без побочных действий."""
        with self._lock:
            return self._pair is not None and self._mtimes() != self._stamp

    def _mtimes(self) -> Optional[Tuple[float, float]]:
        try:
            return (os.stat(self.paths.certificate).st_mtime,
                    os.stat(self.paths.private_key).st_mtime)
        except OSError:
            # Файла нет — не роняем: старая пара ещё годна, а certbot мог
            # застать нас в середине замены.
            return self._stamp


def server_credentials(paths: CertPaths):
    """Учётные данные gRPC-сервера, переживающие перевыпуск.

    Возвращает None, когда сертификат не назван: вызывающий поднимет открытый
    канал и скажет об этом вслух.
    """
    if not paths.configured:
        return None
    import grpc

    rotating = RotatingCertificate(paths)

    def configuration():
        key, chain = rotating.current()
        return grpc.ssl_server_certificate_configuration([(key, chain)])

    # Первое чтение — сразу: негодный сертификат должен обнаружиться на старте,
    # а не на первом подключении узла, когда смотреть в лог уже некому.
    initial = configuration()

    def fetch():
        try:
            return configuration() if rotating.changed() else None
        except TlsUnusable as exc:
            # None означает «оставь прежнее». Это лучше падения: старый
            # сертификат ещё несколько дней действителен, а узлы важнее.
            logger.error("не удалось перечитать сертификат (%s); работаю на "
                         "прежнем", exc)
            return None

    return grpc.dynamic_ssl_server_credentials(initial, fetch)
