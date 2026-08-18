#!/usr/bin/env python3
"""Load-test a Loom deployment and report what actually matters.

    python3 scripts/bench.py --url http://localhost:8000 --model qwen3-14b \\
        --tokens 200 --requests 8 --concurrency 4

Reports per-request TTFT and decode speed (from the head stage's own `timings`,
so prefill is excluded), plus the AGGREGATE token rate — the number that shows
whether the pipeline bubble is being filled. A two-stage pipeline serving one
request at a time leaves both GPUs idle half the time; the aggregate rate
should climb with concurrency while per-request speed stays flat.

Only the standard library: this runs on the GPU box, which has no dev deps.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def one_request(url: str, token: str, body: dict) -> dict:
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            answer = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read()[:200].decode('utf-8', 'replace')}"}
    except Exception as exc:  # network, timeout
        return {"error": str(exc)}
    wall_ms = (time.perf_counter() - started) * 1000
    usage = answer.get("usage") or {}
    timings = answer.get("timings") or {}
    return {
        "wall_ms": wall_ms,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "finish_reason": (answer.get("choices") or [{}])[0].get("finish_reason"),
        # Present only on the shard backend; vLLM answers have no timings.
        "ttft_ms": timings.get("ttft_ms"),
        "decode_tps": timings.get("decode_tokens_per_s"),
        "inter_p50": timings.get("inter_token_ms_p50"),
        "inter_p95": timings.get("inter_token_ms_p95"),
        "stages": timings.get("stages"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000", help="orchestrator base URL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Explain distributed inference in three sentences.")
    ap.add_argument("--tokens", type=int, default=200, help="max_tokens per request")
    ap.add_argument("--requests", type=int, default=8, help="total requests")
    ap.add_argument("--concurrency", type=int, default=1, help="requests in flight")
    ap.add_argument("--no-think", action="store_true", help="disable <think> on reasoning models")
    ap.add_argument("--token", default="", help="API token, if the endpoint needs one")
    ap.add_argument("--warmup", type=int, default=1, help="requests to discard (compile/caches)")
    args = ap.parse_args(argv)

    url = args.url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.tokens,
    }
    if args.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    if args.warmup:
        print(f"warm-up: {args.warmup} request(s)…", flush=True)
        for _ in range(args.warmup):
            warm = one_request(url, args.token, body)
            if "error" in warm:
                print(f"  warm-up failed: {warm['error']}", file=sys.stderr)
                return 1

    print(
        f"running {args.requests} requests, {args.concurrency} in flight, "
        f"{args.tokens} tokens each…",
        flush=True,
    )
    done = 0
    lock = threading.Lock()

    def run(_):
        nonlocal done
        result = one_request(url, args.token, body)
        with lock:
            done += 1
            print(f"  {done}/{args.requests}", end="\r", file=sys.stderr, flush=True)
        return result

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run, range(args.requests)))
    wall_s = time.perf_counter() - started
    print(" " * 30, end="\r", file=sys.stderr)

    failures = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    if not ok:
        print("all requests failed:", failures[0]["error"], file=sys.stderr)
        return 1

    total_tokens = sum(r["completion_tokens"] for r in ok)
    walls = [r["wall_ms"] for r in ok]
    decode = [r["decode_tps"] for r in ok if r["decode_tps"]]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    truncated = sum(1 for r in ok if r["finish_reason"] == "length")

    print()
    print(f"  запросов        {len(ok)} ok" + (f", {len(failures)} ошибок" if failures else ""))
    print(f"  токенов всего   {total_tokens}" + (f" ({truncated} обрезано лимитом)" if truncated else ""))
    print(f"  время всего     {wall_s:.1f} s")
    print(f"  СУММАРНО        {total_tokens / wall_s:.1f} tok/s   <- растёт с concurrency")
    print(f"  на запрос       {statistics.median(walls) / 1000:.2f} s (медиана),"
          f" p95 {percentile(walls, 0.95) / 1000:.2f} s")
    if decode:
        print(f"  decode          {statistics.median(decode):.2f} tok/s (медиана на запрос)"
              "   <- должен почти не падать")
    if ttfts:
        print(f"  TTFT            {statistics.median(ttfts):.0f} ms (медиана),"
              f" p95 {percentile(ttfts, 0.95):.0f} ms")
    stages = {r["stages"] for r in ok if r["stages"]}
    if stages:
        p50 = [r["inter_p50"] for r in ok if r["inter_p50"]]
        p95 = [r["inter_p95"] for r in ok if r["inter_p95"]]
        print(f"  стадий          {sorted(stages)}, между токенами"
              f" p50 {statistics.median(p50):.1f} / p95 {statistics.median(p95):.1f} ms")
    else:
        print("  (бэкенд не прислал timings — vLLM или старый образ воркера)")
    if failures:
        print(f"\n  первая ошибка: {failures[0]['error']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
