"""The training stage as a process: transport in, transport out, run driven.

Deliberately a sibling of the inference stage server rather than a branch
inside it. They share a transport and nothing else; putting a training path
into the loop that answers user requests would risk the thing that already
works for the sake of the thing being built.

What it speaks:

    POST /stage/forward   inter-stage training messages, from the agent
    POST /train/start     stage 0 only: begin a run over a dataset
    POST /train/stop      stage 0 only: stop after the current step
    GET  /train/status    where the run is
    POST /train/save      write this stage's slice now
    GET  /health          liveness, and what this stage holds

Messages leave the same way inference's do: POSTed to the agent's relay, which
routes them by pipeline and target stage. Routing never looks at what a
message says, so training rides the transport that already exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from loom_worker.shard.loader import ShardSpec, build_shard, resolve_model_path
from loom_worker.train import protocol
from loom_worker.train.runner import (
    StageRunner,
    StepDriver,
    TrainingStepFailed,
    load_shard,
    save_shard,
)
from loom_worker.train.stage import TrainingStage

logger = logging.getLogger("loom_worker.train.server")

STATE: dict = {}
INBOX: "queue.Queue[dict]" = queue.Queue()


def inbox_loop() -> None:
    """One message at a time, on one thread.

    The same rule the inference stage follows and for the same reason:
    autograd's graph is process-wide, and two backward passes through the same
    modules at once corrupt it. Micro-batches overlap ACROSS stages, which is
    where the parallelism comes from — not inside one.
    """
    while True:
        msg = INBOX.get()
        try:
            STATE["runner"].handle(msg)
        except Exception:
            logger.exception("training message failed")
        finally:
            INBOX.task_done()


def relay(target_stage: int, msg: dict) -> None:
    """Hand a message to the agent, which knows how to reach that stage."""
    payload = dict(msg)
    payload["target_stage"] = target_stage
    payload["pipeline_id"] = STATE["topology"]["pipeline_id"]
    payload["model_id"] = STATE["model_id"]
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        STATE["relay_url"], data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        response.read()


# ------------------------------------------------------------------ the data
def load_dataset(path: str, tokenizer, *, max_length: int) -> List[dict]:
    """A JSONL file of {"text": ...} turned into token ids.

    Held in memory on purpose for now: the runs this is built for are
    fine-tuning runs on curated data, not pre-training corpora, and a file
    small enough to fine-tune on is small enough to hold.
    """
    samples = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text") or record.get("content") or ""
            if not text:
                continue
            ids = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
            if len(ids) > 1:  # a single token has no next token to predict
                samples.append({"input_ids": ids})
    if not samples:
        raise ValueError(f"{path} yielded no usable samples")
    logger.info("dataset: %d samples from %s", len(samples), path)
    return samples


# ------------------------------------------------------------------- the run
def run_loop() -> None:
    """Stage 0's loop: steps until told to stop, or until the data runs out."""
    driver: StepDriver = STATE["driver"]
    samples = STATE["samples"]
    micro = STATE["micro_batches"]
    epochs = STATE["epochs"]
    cursor = 0
    STATE["run"]["status"] = "running"
    try:
        for epoch in range(epochs):
            while cursor + micro <= len(samples):
                if STATE["run"]["stopping"]:
                    STATE["run"]["status"] = "stopped"
                    return
                batch = samples[cursor:cursor + micro]
                cursor += micro
                result = driver.run_step(batch)
                STATE["run"].update(
                    step=result.step, loss=result.loss,
                    attempts=result.attempts, seconds=round(result.seconds, 3),
                )
            cursor = 0
            STATE["run"]["epoch"] = epoch + 1
        STATE["run"]["status"] = "finished"
    except TrainingStepFailed as exc:
        # The run stops rather than looping on a failure that retries did not
        # clear. What it wrote so far stands: the last checkpoint is valid.
        logger.error("training stopped: %s", exc)
        STATE["run"].update(status="failed", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("training stopped unexpectedly")
        STATE["run"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        driver.checkpoint()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the access log out of the way
        return

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            stage = STATE.get("stage")
            self._json(200, {
                "ok": stage is not None,
                "mode": STATE.get("mode"),
                "layers": [STATE["spec"].start_layer, STATE["spec"].end_layer],
                "stage_index": STATE["topology"]["stage_index"],
                "in_flight": stage.in_flight() if stage else 0,
                "steps": stage.steps if stage else 0,
                "trainable_bytes": stage.trainable_bytes() if stage else 0,
            })
        elif self.path == "/train/status":
            self._json(200, dict(STATE.get("run") or {}))
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")

        if self.path == "/stage/forward":
            INBOX.put(payload)
            self._json(200, {"ok": True})
        elif self.path == "/train/start":
            self._json(*_start_run(payload))
        elif self.path == "/train/status":
            # Also on POST: the orchestrator reaches a worker through the data
            # plane, and that channel carries POSTs.
            self._json(200, dict(STATE.get("run") or {}))
        elif self.path == "/train/stop":
            STATE["run"]["stopping"] = True
            self._json(200, {"ok": True, "status": STATE["run"]["status"]})
        elif self.path == "/train/save":
            path = payload.get("path") or STATE["checkpoint_dir"]
            relay(-1, {"kind": protocol.SAVE, "step": STATE["run"].get("step", 0),
                       "path": path})
            self._json(200, {"ok": True, "path": path})
        else:
            self._json(404, {"error": "not found"})


def _start_run(payload: dict):
    if not STATE["topology"]["is_first"]:
        return 400, {"error": "only stage 0 drives a run"}
    if STATE["run"]["status"] == "running":
        return 409, {"error": "a run is already in progress"}

    dataset = payload.get("dataset") or STATE.get("dataset")
    if not dataset:
        return 400, {"error": "no dataset given"}
    STATE["samples"] = load_dataset(
        dataset, STATE["tokenizer"], max_length=int(payload.get("max_length", 512))
    )
    STATE["micro_batches"] = int(payload.get("micro_batches", 4))
    STATE["epochs"] = int(payload.get("epochs", 1))
    STATE["run"].update(status="starting", stopping=False, error=None,
                        step=STATE["driver"].step_index)
    threading.Thread(target=run_loop, name="loom-train-run", daemon=True).start()
    return 200, {"ok": True, "samples": len(STATE["samples"])}


def main(argv=None) -> None:
    logging.basicConfig(
        level=os.environ.get("LOOM_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    parser = argparse.ArgumentParser(prog="loom-train-stage")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--weights-uri", required=True)
    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--end-layer", type=int, required=True)
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument("--pipeline-id", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--relay-url", default="")
    parser.add_argument("--device", default=os.environ.get("LOOM_SHARD_DEVICE", "cpu"))
    parser.add_argument("--dtype", default=os.environ.get("LOOM_SHARD_DTYPE", "float32"))
    parser.add_argument("--mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--micro-batches", type=int, default=4)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args(argv)

    is_first = args.stage_index == 0
    is_last = args.stage_index == args.num_stages - 1
    spec = ShardSpec(
        model_path="", start_layer=args.start_layer, end_layer=args.end_layer,
        is_first=is_first, is_last=is_last, device=args.device, dtype=args.dtype,
    )
    # Only this stage's safetensors files are fetched — the same saving the
    # inference stage makes, and it matters more here because a training run
    # holds the download for hours rather than minutes.
    spec.model_path = resolve_model_path(
        args.weights_uri, os.environ.get("HF_TOKEN") or None, shard=spec
    )
    shard, config = build_shard(spec)
    stage = TrainingStage(shard, spec, mode=args.mode, rank=args.rank,
                          alpha=args.alpha, lr=args.lr)

    STATE.update(
        stage=stage, spec=spec, model_id=args.model_id, mode=args.mode,
        relay_url=args.relay_url, checkpoint_dir=args.checkpoint_dir,
        dataset=args.dataset, micro_batches=args.micro_batches, epochs=1,
        topology={
            "pipeline_id": args.pipeline_id, "stage_index": args.stage_index,
            "num_stages": args.num_stages, "is_first": is_first, "is_last": is_last,
        },
        run={"status": "idle", "step": 0, "loss": None, "stopping": False},
    )

    runner = StageRunner(stage, send=relay, stage_index=args.stage_index,
                         num_stages=args.num_stages)
    STATE["runner"] = runner
    if is_first:
        driver = StepDriver(runner, micro_batches=args.micro_batches,
                            checkpoint_dir=args.checkpoint_dir,
                            checkpoint_every=args.checkpoint_every)
        runner.on_micro_done = driver.micro_done
        STATE["driver"] = driver
        resumed = driver.resume()
        if resumed:
            load_shard(stage, os.path.join(args.checkpoint_dir, f"step-{resumed}"))
        from transformers import AutoTokenizer

        STATE["tokenizer"] = AutoTokenizer.from_pretrained(spec.model_path)
    else:
        # Later stages restore their own slice; only stage 0 knows the step.
        if args.checkpoint_dir and os.path.exists(
            os.path.join(args.checkpoint_dir, "progress.json")
        ):
            with open(os.path.join(args.checkpoint_dir, "progress.json")) as handle:
                step = int(json.load(handle).get("step") or 0)
            if step:
                load_shard(stage, os.path.join(args.checkpoint_dir, f"step-{step}"))

    # Reports coming home to stage 0 go to the driver, not the model.
    if is_first:
        original = runner.handle

        def handle(msg: dict) -> None:
            if msg.get("kind") in (protocol.LOSS, protocol.ERROR):
                STATE["driver"].note(msg)
                return
            original(msg)

        runner.handle = handle

    threading.Thread(target=inbox_loop, name="loom-train-inbox", daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    logger.info(
        "training stage %d/%d ready on port %d (layers [%d, %d), mode=%s)",
        args.stage_index, args.num_stages, server.server_port,
        args.start_layer, args.end_layer, args.mode,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
