"""Minimal logging helpers for Looma (own code, no borrowed logic)."""

import logging

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger with a default stream handler installed once."""
    root = logging.getLogger("looma")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    if name.startswith("looma"):
        return logging.getLogger(name)
    return logging.getLogger(f"looma.{name}")
