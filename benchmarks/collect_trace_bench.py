#!/usr/bin/env python3
"""
DualState 30-min trace-driven serving benchmark.

Uses real arrival timestamps from Kimi/Qwen traces + real content from morspec data.
Records per-minute throughput and TTFT for plotting.

Usage:
  python collect_trace_bench.py --config baseline --trace kimi --duration 1800
  python collect_trace_bench.py --config dualstate --trace qwen --duration 1800
"""

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests as req_lib

# ─── Config ────────────────────────────────────────────────────────────────

TRACES = {
    "kimi": "/mnt/models/kimik25/kimi-k25-trace/kimi_k25_conv_1day.csv",
    "qwen": "/tmp/qwen_traces/qwen_traceB_blksz_16.jsonl",
}

CONTENT_FILES = {
    "gsm8k": "/home/zhujianian/morspec/data/gsm8k.jsonl",
    "mt_bench": "/home/zhujianian/morspec/data/mt_bench.jsonl",
    "humaneval": "/home/zhujianian/morspec/data/humaneval.jsonl",
}

# ─── HTTP ──────────────────────────────────────────────────────────────────

_session = None

def get_session():
    global _session
    if _session is None:
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        for k in ("http_proxy", "https_proxy", "all_proxy", "ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(k, None)
        _session = req_lib.Session()
        _session.trust_env = False
    return _session


def send_request(base_url, prompt, max_tokens=32, timeout=120):
    s = get_session()
    start = time.perf_counter()
    try:
        r = s.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        if r.status_code == 200:
            data = r.json()
            usage = data.get("usage", {})
            return {
                "success": True,
                "ttft": elapsed,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "finish_time": time.time(),
            }
        return {"success": False, "ttft": elapsed, "error": f"HTTP {r.status_code}", "finish_time": time.time()}
    except Exception as e:
        return {"success": False, "ttft": time.perf_counter() - start, "error": str(e), "finish_time": time.time()}


# ─── Load content ─────────────────────────────────────────────────────────

def load_content_pool():
    """Load real text content from morspec datasets."""
    pool = []
    # GSM8K questions
    with open(CONTENT_FILES["gsm8k"]) as f:
        for line in f:
            d = json.loads(line)
            pool.append(d["question"])
    # MT-Bench multi-turn
    with open(CONTENT_FILES["mt_bench"]) as f:
        for line in f:
            d = json.loads(line)
            for turn in d.get("turns", []):
                pool.append(turn)
    # HumanEval prompts
    with open(CONTENT_FILES["humaneval"]) as f:
        for line in f:
            d = json.loads(line)
            pool.append(d["prompt"])
    random.shuffle(pool)
    return pool


# ─── Load traces ──────────────────────────────────────────────────────────

def load_kimi_trace(path, duration_sec, max_requests):
    arrivals = []
    first_ts = None
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            from datetime import datetime
            try:
                ts = datetime.fromisoformat(row["TIMESTAMP"].replace("+00:00", "+00:00"))
            except ValueError:
                continue
            if first_ts is None:
                first_ts = ts
            elapsed = (ts - first_ts).total_seconds()
            if elapsed > duration_sec:
                break
            ctx = min(int(row["ContextTokens"]), 4096)  # Cap for our model
            gen = min(int(row["GeneratedTokens"]), 64)
            arrivals.append({"time": elapsed, "ctx": ctx, "gen": gen})
            if len(arrivals) >= max_requests:
                break
    return arrivals


def load_qwen_trace(path, duration_sec, max_requests):
    arrivals = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            t = d.get("timestamp", 0)
            if t > duration_sec:
                break
            ctx = min(d.get("input_length", 500), 4096)
            gen = min(d.get("output_length", 32), 64)
            arrivals.append({"time": t, "ctx": ctx, "gen": gen})
            if len(arrivals) >= max_requests:
                break
    return arrivals


# ─── Replay ───────────────────────────────────────────────────────────────

def replay_trace(base_url, arrivals, content_pool, concurrency=16):
    """Replay trace with real timestamps and content. Record per-minute metrics."""
    results = []
    minute_buckets = defaultdict(list)  # minute -> list of results

    start_time = time.perf_counter()
    start_wall = time.time()

    # Build shared prefix groups (simulate agentic: same system prompt for nearby requests)
    system_prompts = content_pool[:10]  # 10 system prompt groups

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for i, arrival in enumerate(arrivals):
            # Wait until arrival time
            target = start_time + arrival["time"]
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)

            # Build prompt: shared system prompt + unique question
            sys_prompt = system_prompts[i % len(system_prompts)]
            question = content_pool[(i + 10) % len(content_pool)]
            # Truncate to match trace's context tokens (rough: 1 word ≈ 1.3 tokens)
            target_words = max(10, arrival["ctx"] // 2)  # Use half to keep fast
            prompt = sys_prompt + "\n\n" + " ".join(question.split()[:target_words])

            max_tokens = min(arrival["gen"], 32)
            future = executor.submit(send_request, base_url, prompt, max_tokens)
            futures[future] = {"idx": i, "arrival_time": arrival["time"]}

        for future in as_completed(futures):
            info = futures[future]
            result = future.result()
            result.update(info)
            results.append(result)

            # Bucket by minute
            minute = int(info["arrival_time"] // 60)
            minute_buckets[minute].append(result)

    # Compute per-minute metrics
    per_minute = {}
    for minute, reqs in sorted(minute_buckets.items()):
        successful = [r for r in reqs if r.get("success")]
        n_total = len(reqs)
        n_success = len(successful)
        ttfts = sorted(r["ttft"] for r in successful) if successful else []
        n = len(ttfts)

        per_minute[minute] = {
            "total": n_total,
            "success": n_success,
            "throughput": n_success / 60.0 if n_success > 0 else 0,
            "ttft_mean": statistics.mean(ttfts) if ttfts else 0,
            "ttft_p50": ttfts[n // 2] if n > 0 else 0,
            "ttft_p95": ttfts[int(n * 0.95)] if n >= 20 else (ttfts[-1] if ttfts else 0),
            "ttft_p99": ttfts[int(n * 0.99)] if n >= 100 else (ttfts[-1] if ttfts else 0),
        }

    return results, per_minute


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Tag: baseline or dualstate")
    parser.add_argument("--trace", required=True, choices=["kimi", "qwen"])
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--duration", type=int, default=1800, help="Duration in seconds")
    parser.add_argument("--max-requests", type=int, default=5000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Config: {args.config}, Trace: {args.trace}, Duration: {args.duration}s")

    # Load content
    content_pool = load_content_pool()
    print(f"Loaded {len(content_pool)} content items")

    # Load trace
    if args.trace == "kimi":
        arrivals = load_kimi_trace(TRACES["kimi"], args.duration, args.max_requests)
    else:
        arrivals = load_qwen_trace(TRACES["qwen"], args.duration, args.max_requests)
    print(f"Loaded {len(arrivals)} arrivals over {args.duration}s")

    if not arrivals:
        print("ERROR: No arrivals loaded")
        return

    # Warmup
    print("Warmup...")
    for _ in range(3):
        send_request(args.base_url, "Hello world", max_tokens=4, timeout=180)

    # Replay
    print(f"Replaying {len(arrivals)} requests...")
    results, per_minute = replay_trace(
        args.base_url, arrivals, content_pool, args.concurrency
    )

    # Summary
    successful = [r for r in results if r.get("success")]
    print(f"\nTotal: {len(results)} requests, {len(successful)} successful")
    if successful:
        ttfts = [r["ttft"] for r in successful]
        print(f"TTFT: mean={statistics.mean(ttfts)*1000:.0f}ms, "
              f"p50={sorted(ttfts)[len(ttfts)//2]*1000:.0f}ms, "
              f"p95={sorted(ttfts)[int(len(ttfts)*0.95)]*1000:.0f}ms")

    # Save
    output = {
        "config": args.config,
        "trace": args.trace,
        "duration": args.duration,
        "n_requests": len(results),
        "n_success": len(successful),
        "per_minute": per_minute,
        "summary": {
            "ttft_mean": statistics.mean(ttfts) if successful else 0,
            "ttft_p50": sorted(ttfts)[len(ttfts)//2] if successful else 0,
            "ttft_p95": sorted(ttfts)[int(len(ttfts)*0.95)] if len(successful) >= 20 else 0,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
