import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_join_key(address: str) -> str:
    body = json.dumps({"i": "k1", "s": "secret", "a": address}).encode()
    return "looma_" + base64.urlsafe_b64encode(body).decode().rstrip("=")
