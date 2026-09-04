"""Демо на лендинге: открытый инференс и его ограничители.

Маршрут отвечает без представления — в этом его смысл. Ровно поэтому здесь
проверяются не только ответы, но и пределы: открытый инференс без них означает
бесплатный API для всех, оплаченный чужими домашними картами.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from looma.api.app import create_app


class Settings:
    admin_token = "s3cret-token"


class Group:
    def __init__(self, label: str, nodes: int) -> None:
        self.label = label
        self.tasks = [f"task-{i}" for i in range(nodes)]


class Hub:
    """Минимальная замена парку: одна группа, которая отвечает двумя токенами."""

    def __init__(self, label: str = "qwen3-4b", nodes: int = 2, up: bool = True) -> None:
        self.groups = {"g": Group(label, nodes)}
        self.up = up
        self.sent: list[dict] = []

    async def serving(self, label: str):
        group = self.groups["g"]
        return group if self.up and label == group.label else None

    def group_for(self, label: str):
        return None

    async def request_stream(self, task, *, method, path, body, headers):
        self.sent.append(json.loads(body))
        for piece in ("при", "вет"):
            yield ("data: " + json.dumps({"choices": [{"delta": {"content": piece}}]})
                   + "\n\n").encode()
        yield b"data: [DONE]\n\n"


def client(hub=None):
    return TestClient(create_app(agents=hub or Hub(), config=Settings()))


# ------------------------------------------------------------------ открытость
def test_демо_отвечает_без_представления():
    """Иначе блок бессмыслен: человек приходит убедиться в скорости ДО того,
    как заводить учётную запись."""
    answer = client().get("/api/demo")
    assert answer.status_code == 200
    assert answer.json()["model"] == "qwen3-4b"
    assert answer.json()["nodes"] == 2


def test_остальной_инференс_по_прежнему_закрыт():
    """Открыт ровно один адрес, а не раздел целиком."""
    assert client().post("/v1/chat/completions", json={"model": "x"}).status_code == 401


def test_пустая_сеть_не_показывает_демо():
    """Лендинг прячет блок по пустой модели: поле ввода, которое ничего не
    отвечает, хуже отсутствующего раздела."""
    assert client(Hub(up=False)).get("/api/demo").json()["model"] == ""


# ---------------------------------------------------------------------- ответ
def собрать(text: str) -> str:
    """Склеить токены из потока — так же, как это делает лендинг."""
    out = ""
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        out += json.loads(payload)["choices"][0]["delta"]["content"]
    return out


def test_ответ_приходит_потоком():
    answer = client().post("/api/demo", json={"prompt": "привет"})
    assert answer.status_code == 200
    assert собрать(answer.text) == "привет"
    assert answer.text.rstrip().endswith("[DONE]")


def test_модель_выбирается_сама():
    """Снаружи её имя не принимается: иначе демо стало бы способом занять
    любую группу в парке."""
    hub = Hub()
    client(hub).post("/api/demo", json={"prompt": "привет", "model": "другая"})
    assert hub.sent[0]["model"] == "qwen3-4b"


def test_ответ_ограничен_по_длине():
    hub = Hub()
    client(hub).post("/api/demo", json={"prompt": "привет"})
    assert hub.sent[0]["max_tokens"] == 220
    assert hub.sent[0]["stream"] is True


def test_запрос_обрезается():
    hub = Hub()
    client(hub).post("/api/demo", json={"prompt": "я" * 5000})
    assert len(hub.sent[0]["messages"][0]["content"]) == 400


def test_пустой_запрос_отвергается():
    assert client().post("/api/demo", json={"prompt": "  "}).status_code == 400


def test_без_модели_честный_отказ():
    answer = client(Hub(up=False)).post("/api/demo", json={"prompt": "привет"})
    assert answer.status_code == 503


# ------------------------------------------------------------------- пределы
def test_счётчик_на_адрес_закрывается():
    """Двадцать первый запрос с того же адреса отвергается. Без этого одна
    вкладка с циклом занимает парк целиком."""
    one = client()
    for _ in range(20):
        assert one.post("/api/demo", json={"prompt": "привет"}).status_code == 200
    refused = one.post("/api/demo", json={"prompt": "привет"})
    assert refused.status_code == 429
    assert "ключ" in refused.json()["error"]["message"]


def test_предел_у_каждого_адреса_свой():
    """Иначе первый же посетитель закрывает демо для всех остальных."""
    one = client()
    for _ in range(20):
        one.post("/api/demo", json={"prompt": "привет"},
                 headers={"X-Forwarded-For": "10.0.0.1"})
    другой = one.post("/api/demo", json={"prompt": "привет"},
                      headers={"X-Forwarded-For": "10.0.0.2"})
    assert другой.status_code == 200
