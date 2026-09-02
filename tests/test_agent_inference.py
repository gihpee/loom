"""A real model, split across nodes, answering through the agent transport.

Everything below is real: a real (tiny) Llama, real weights sliced by layer, a
real forward pass per stage, real activations crossing the orchestrator between
machines. The stage payload is the one that ships; the only thing standing in
for production is the size of the model.

The point is the thing phase 7 exists for — that inference is now a task like
any other, and the pipeline it forms uses the same transport a client's own
code would.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from make_tiny_model import ensure_tiny_model
from test_agent_gateway import Orchestrator

PAYLOAD = str(Path(__file__).resolve().parent.parent / "payloads" / "looma_stage")


@pytest.fixture(scope="module")
def tiny_model():
    return str(ensure_tiny_model())


def stage_command(model: str, start: int, end: int) -> list:
    return [
        sys.executable, "-m", "looma_stage.server",
        "--model-id", "tiny", "--weights-uri", model,
        "--start-layer", str(start), "--end-layer", str(end),
        "--device", "cpu", "--dtype", "float32",
    ]


def start_nodes(orchestrator, tmp_path, count: int):
    from conftest_agent import start_agent

    running = []
    for index in range(count):
        running.append(start_agent(orchestrator.port, tmp_path / f"n{index}",
                                   node_id=f"node-{index}"))
    deadline = time.time() + 30
    while time.time() < deadline and len(orchestrator.hub.sessions) < count:
        time.sleep(0.05)
    assert len(orchestrator.hub.sessions) == count
    return running


def wait_ready(orchestrator, group, timeout: float = 300.0) -> None:
    """Weights take a while. Silence for a while is not the same as failure."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        states = [orchestrator.hub.tasks[t].state for t in group.tasks.values()]
        if any(s in ("failed", "cancelled") for s in states):
            logs = orchestrator.call(orchestrator.hub.logs(list(group.tasks.values())[0]))
            pytest.fail(f"a stage died: {states}\n{logs}")
        if all(s == "running" for s in states):
            healthy = 0
            for task_id in group.tasks.values():
                try:
                    status, _headers, body = orchestrator.call(
                        orchestrator.hub.request(task_id, path="/health", timeout_s=20))
                    if status == 200 and json.loads(body).get("status") == "ok":
                        healthy += 1
                except Exception:
                    break
            if healthy == len(group.tasks):
                return
        time.sleep(1.0)
    pytest.fail("the pipeline never came up")


def ask(orchestrator, group, prompt: str, *, max_tokens: int = 4) -> dict:
    status, _headers, body = orchestrator.call(orchestrator.hub.request(
        group.tasks[0], method="POST", path="/v1/chat/completions",
        body=json.dumps({
            "model": "tiny",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }).encode(),
        headers={"Content-Type": "application/json"},
        timeout_s=300,
    ))
    assert status == 200, body[:400]
    return json.loads(body)


@pytest.mark.slow
def test_a_model_split_over_two_nodes_answers(tmp_path, monkeypatch, tiny_model):
    """Layers 0-2 on one machine, 3-6 on another, one answer out of the pair."""
    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOMA_P2P", "0")
    orchestrator = Orchestrator().start()
    running = start_nodes(orchestrator, tmp_path, 2)
    try:
        group = orchestrator.hub.submit_group(
            size=2,
            command=stage_command(tiny_model, 0, 3),
            env={"PYTHONPATH": PAYLOAD},
            serve_port=1,          # non-zero: "serve, and tell me on what"
            timeout_s=600,
            node_ids=["node-0", "node-1"],
            # The split is the orchestrator's decision, because only it knows
            # what each node can hold. Here it is fixed because the model is.
            per_rank=[
                {"command": stage_command(tiny_model, 0, 3)},
                {"command": stage_command(tiny_model, 3, 6)},
            ],
        )
        wait_ready(orchestrator, group)

        answer = ask(orchestrator, group, "hello")
        assert answer["choices"][0]["message"]["content"] is not None
        assert answer["usage"]["completion_tokens"] >= 1
    finally:
        for agent, thread in running:
            agent.stop()
            thread.join(timeout=15)
        orchestrator.stop()


@pytest.mark.slow
def test_a_model_on_one_node_answers_without_touching_the_network(tmp_path, monkeypatch,
                                                                  tiny_model):
    """Both stages on one machine: the fast arrangement, and the same code."""
    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOMA_P2P", "0")
    orchestrator = Orchestrator().start()
    running = start_nodes(orchestrator, tmp_path, 1)
    try:
        group = orchestrator.hub.submit_group(
            size=2,
            command=stage_command(tiny_model, 0, 3),
            env={"PYTHONPATH": PAYLOAD},
            serve_port=1,
            timeout_s=600,
            node_ids=["node-0", "node-0"],
            per_rank=[
                {"command": stage_command(tiny_model, 0, 3)},
                {"command": stage_command(tiny_model, 3, 6)},
            ],
        )
        wait_ready(orchestrator, group)
        answer = ask(orchestrator, group, "hello")
        assert answer["choices"][0]["message"]["content"] is not None
    finally:
        for agent, thread in running:
            agent.stop()
            thread.join(timeout=15)
        orchestrator.stop()


@pytest.mark.slow
def test_splitting_the_model_does_not_change_the_answer(tmp_path, monkeypatch, tiny_model):
    """The test that matters more than "it runs".

    A pipeline that answers is not the same as a pipeline that answers
    CORRECTLY: an off-by-one in the layer split, a lost KV cache entry or a
    dropped position id all still produce fluent-looking output. Greedy
    decoding is deterministic, so one stage and two must emit the same tokens
    from the same prompt — and if they do not, the split is wrong.
    """
    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOMA_P2P", "0")
    orchestrator = Orchestrator().start()
    running = start_nodes(orchestrator, tmp_path, 2)
    try:
        whole = orchestrator.hub.submit_group(
            size=1, command=stage_command(tiny_model, 0, 6),
            env={"PYTHONPATH": PAYLOAD}, serve_port=1, timeout_s=600,
            node_ids=["node-0"],
        )
        wait_ready(orchestrator, whole)
        one_stage = ask(orchestrator, whole, "hello", max_tokens=8)
        orchestrator.hub.stop_group(whole.group_id)

        split = orchestrator.hub.submit_group(
            size=2, command=stage_command(tiny_model, 0, 3),
            env={"PYTHONPATH": PAYLOAD}, serve_port=1, timeout_s=600,
            node_ids=["node-0", "node-1"],
            per_rank=[
                {"command": stage_command(tiny_model, 0, 3)},
                {"command": stage_command(tiny_model, 3, 6)},
            ],
        )
        wait_ready(orchestrator, split)
        two_stages = ask(orchestrator, split, "hello", max_tokens=8)

        assert two_stages["choices"][0]["message"]["content"] == \
            one_stage["choices"][0]["message"]["content"], \
            "the same prompt gave different answers whole and split"
    finally:
        for agent, thread in running:
            agent.stop()
            thread.join(timeout=15)
        orchestrator.stop()


@pytest.mark.slow
def test_ответ_приходит_по_частям_а_не_целиком(tmp_path, monkeypatch, tiny_model):
    """Генерация занимает минуты, и ждать её целиком ради первого слова —
    значит выглядеть зависшим ровно столько же.

    Проверяется именно инкрементальность: одна часть на весь ответ означала бы,
    что где-то по пути его собрали и придержали.
    """
    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOMA_P2P", "0")
    orchestrator = Orchestrator().start()
    running = start_nodes(orchestrator, tmp_path, 1)
    try:
        group = orchestrator.hub.submit_group(
            size=1, command=stage_command(tiny_model, 0, 6),
            env={"PYTHONPATH": PAYLOAD}, serve_port=1, timeout_s=600,
            node_ids=["node-0"], label="tiny")
        wait_ready(orchestrator, group)

        async def collect():
            head, pieces = None, []
            body = json.dumps({"model": "tiny", "stream": True, "max_tokens": 6,
                               "messages": [{"role": "user", "content": "привет"}]}).encode()
            async for piece in orchestrator.hub.request_stream(
                    group.tasks[0], method="POST", path="/v1/chat/completions",
                    body=body, headers={"Content-Type": "application/json"},
                    timeout_s=180):
                if isinstance(piece, tuple):
                    head = piece
                else:
                    pieces.append(piece)
            return head, pieces

        (status, headers), pieces = orchestrator.call(collect(), timeout=240)
        assert status == 200
        assert "text/event-stream" in headers.get("Content-Type", "")
        assert len(pieces) > 1, "весь ответ пришёл одним куском — это не стрим"

        text = b"".join(pieces).decode()
        events = [line for line in text.splitlines() if line.startswith("data: ")]
        assert len(events) > 1
        assert events[-1].strip() == "data: [DONE]"
    finally:
        for agent, thread in running:
            agent.stop()
            thread.join(timeout=15)
        orchestrator.stop()


@pytest.mark.slow
def test_состояние_стадий_видно_пока_модель_поднимается(tmp_path, monkeypatch, tiny_model):
    """«running» — это про процесс, а не про готовность отвечать."""
    from fastapi.testclient import TestClient

    from looma.api.app import create_app
    from test_agent_gateway import _Settings

    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOMA_P2P", "0")
    orchestrator = Orchestrator().start()
    running = start_nodes(orchestrator, tmp_path, 1)
    try:
        group = orchestrator.hub.submit_group(
            size=1, command=stage_command(tiny_model, 0, 6),
            env={"PYTHONPATH": PAYLOAD}, serve_port=1, timeout_s=600,
            node_ids=["node-0"], label="tiny")
        wait_ready(orchestrator, group)
        client = TestClient(create_app(agents=orchestrator.hub, config=_Settings()))
        view = client.get(f"/admin/groups/{group.group_id}/health").json()
        assert view["ready"] is True
        stage = view["stages"][0]
        assert stage["rank"] == 0
        assert stage["stage"]["layers"] == [0, 6], "стадия не сказала, какие слои держит"
    finally:
        for agent, thread in running:
            agent.stop()
            thread.join(timeout=15)
        orchestrator.stop()


@pytest.mark.slow
def test_два_клиента_обслуживаются_одновременно(tmp_path, monkeypatch, tiny_model):
    """То, ради чего появился общий цикл.

    Раньше каждый запрос вёл конвейер сам и стоял в очереди за замком вокруг
    модели: параллельные клиенты делили пропускную способность, а не
    складывали её. Теперь запросы попадают в один шаг движка, и проверяется
    здесь именно это — оба получают СВОЙ ответ, а не один чужой на двоих.

    Ответы сравниваются с одиночными, снятыми тем же жадным декодированием.
    Батч, склеившийся в одну последовательность, ответ бы дал — просто
    другой, и без этой сверки такая поломка выглядела бы как успех.
    """
    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOMA_P2P", "0")
    orchestrator = Orchestrator().start()
    running = start_nodes(orchestrator, tmp_path, 1)
    try:
        group = orchestrator.hub.submit_group(
            size=1, command=stage_command(tiny_model, 0, 6),
            env={"PYTHONPATH": PAYLOAD}, serve_port=1, timeout_s=600,
            node_ids=["node-0"],
        )
        wait_ready(orchestrator, group)

        alone = {prompt: ask(orchestrator, group, prompt, max_tokens=4)
                 ["choices"][0]["message"]["content"]
                 for prompt in ("hello", "goodbye")}

        import concurrent.futures as futures

        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            both = list(pool.map(
                lambda prompt: (prompt, ask(orchestrator, group, prompt,
                                            max_tokens=4)),
                ("hello", "goodbye")))

        for prompt, answer in both:
            assert answer["usage"]["completion_tokens"] >= 1
            assert answer["choices"][0]["message"]["content"] == alone[prompt], \
                f"вместе и по одному ответы разошлись на {prompt!r}"

        # Попали ли они в один шаг — здесь не утверждается: это зависит от
        # того, успел ли второй прийти, пока считался первый, и на разной
        # машине выходит по-разному. Число видно в ответе (`batch_max`), и
        # проверяется оно там, где это можно сделать без гонки, — на самом
        # цикле в tests/test_pipeline.py.
        assert all(answer["timings"]["batch_max"] >= 1 for _p, answer in both)
    finally:
        for agent, thread in running:
            agent.stop()
            thread.join(timeout=15)
        orchestrator.stop()
