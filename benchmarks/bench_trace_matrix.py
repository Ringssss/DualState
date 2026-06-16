#!/usr/bin/env python3
"""
DualState Full Experiment Matrix — Trace-Driven Benchmark.

Tests DualState vs SGLang-PD-Disaggregate Baseline using real production traces:
- Kimi K25 trace (1 day, 245K requests)
- Azure LLM Inference trace (1 week, 27M requests)

Matrix: 2 models × 2 traces × multiple arrival rate scales × (baseline, dualstate)

Usage:
  python bench_trace_matrix.py --mode baseline --model qwen --trace kimi --duration 120
  python bench_trace_matrix.py --mode dualstate --model kimi --trace azure --duration 120
"""

import argparse
import csv
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

# ─── Configuration ──────────────────────────────────────────────────────────

TRACES = {
    "kimi": "/mnt/models/kimik25/kimi-k25-trace/kimi_k25_conv_1day.csv",
    "azure": "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv",
}

# Content dataset for generating request text with specific token counts
CONTENT_DATASETS = {
    "mt_bench": "/home/zhujianian/morspec/data/mt_bench.jsonl",
    "mixed": "/home/zhujianian/morspec/mixed_content.jsonl",
}

# Shared prefixes that simulate real prefix-sharing patterns
SYSTEM_PROMPTS = {
    "short": "You are a helpful assistant. Answer concisely. ",
    "medium": (
        "You are a helpful AI assistant specialized in mathematics, science, and code. "
        "Please analyze problems carefully, show your work step by step, and provide "
        "rigorous solutions. Consider edge cases and verify your answers. "
    ) * 5,
    "long": (
        "You are an expert AI assistant deployed in a production environment. "
        "Your role is to assist users with complex technical queries spanning "
        "software engineering, data science, mathematics, and system design. "
        "Always provide accurate, well-structured responses with examples. "
        "When uncertain, explicitly state your confidence level. "
    ) * 15,
}


def _no_proxy_session():
    import os
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    for k in ("http_proxy", "https_proxy", "all_proxy", "ALL_PROXY"):
        os.environ.pop(k, None)
    s = requests.Session()
    s.trust_env = False
    return s


_session = None


def send_request(base_url, prompt, max_tokens=16, timeout=90):
    global _session
    if _session is None:
        _session = _no_proxy_session()
    start = time.perf_counter()
    try:
        r = _session.post(
            f"{base_url}/generate",
            json={"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens}},
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        if r.status_code == 200:
            return {"success": True, "ttft": elapsed}
        return {"success": False, "error": f"HTTP {r.status_code}", "ttft": elapsed}
    except Exception as e:
        return {"success": False, "error": str(e), "ttft": time.perf_counter() - start}


# ─── Trace Loading ──────────────────────────────────────────────────────────

def load_trace(trace_path, duration_sec=120, scale=1.0, max_requests=500, start_offset_sec=0):
    """
    Load a trace CSV and extract requests within a time window.
    Scale adjusts inter-arrival times (scale=0.5 → 2x faster).
    """
    requests_data = []
    first_ts = None

    with open(trace_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row["TIMESTAMP"]
            ctx_tokens = int(row["ContextTokens"])
            gen_tokens = int(row["GeneratedTokens"])

            # Parse timestamp
            try:
                ts = datetime.fromisoformat(ts_str.replace("+00:00", "+00:00"))
            except ValueError:
                continue

            if first_ts is None:
                first_ts = ts

            elapsed = (ts - first_ts).total_seconds()

            # Skip to start offset
            if elapsed < start_offset_sec:
                continue

            # Relative time from start_offset
            rel_time = (elapsed - start_offset_sec) * scale

            if rel_time > duration_sec:
                break

            # Cap token counts to reasonable limits for our models
            ctx_tokens = min(ctx_tokens, 2048)
            gen_tokens = min(gen_tokens, 64)

            requests_data.append({
                "arrival_time": rel_time,
                "context_tokens": ctx_tokens,
                "gen_tokens": gen_tokens,
            })

            if len(requests_data) >= max_requests:
                break

    return requests_data


def generate_prompt(context_tokens, prefix_type="medium"):
    """Generate a prompt with approximately context_tokens tokens using a shared prefix."""
    prefix = SYSTEM_PROMPTS.get(prefix_type, SYSTEM_PROMPTS["medium"])
    # Rough: 1 word ≈ 1.3 tokens for English
    words_needed = max(1, int(context_tokens / 1.3))
    prefix_words = prefix.split()

    if len(prefix_words) >= words_needed:
        return " ".join(prefix_words[:words_needed])

    # Pad with unique suffix to reach target length
    suffix_words = words_needed - len(prefix_words)
    unique_suffix = " ".join(
        f"token_{i}_{random.randint(0, 9999)}" for i in range(suffix_words)
    )
    return prefix + " " + unique_suffix


# ─── Trace Replay ──────────────────────────────────────────────────────────

def replay_trace(base_url, trace_requests, prefix_type="medium", concurrency=8):
    """
    Replay trace requests respecting inter-arrival times.
    Returns list of result dicts with timing info.
    """
    results = []
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}

        for i, req_info in enumerate(trace_requests):
            # Wait until arrival time
            target_time = start_time + req_info["arrival_time"]
            now = time.perf_counter()
            if target_time > now:
                time.sleep(target_time - now)

            # Generate prompt with shared prefix
            prompt = generate_prompt(req_info["context_tokens"], prefix_type)
            max_tokens = req_info["gen_tokens"]

            # Submit request
            future = executor.submit(send_request, base_url, prompt, max_tokens)
            futures[future] = {
                "idx": i,
                "arrival_time": req_info["arrival_time"],
                "context_tokens": req_info["context_tokens"],
                "gen_tokens": req_info["gen_tokens"],
            }

        # Collect results
        for future in as_completed(futures):
            info = futures[future]
            result = future.result()
            result.update(info)
            results.append(result)

    results.sort(key=lambda x: x["idx"])
    return results


# ─── Stats ──────────────────────────��───────────────────────────────────────

def compute_stats(results):
    successful = [r for r in results if r.get("success")]
    n = len(successful)
    if n == 0:
        return {"n_total": len(results), "n_success": 0, "success_rate": 0}

    ttfts = sorted([r["ttft"] for r in successful])
    return {
        "n_total": len(results),
        "n_success": n,
        "success_rate": n / len(results),
        "ttft_mean": sum(ttfts) / n,
        "ttft_p50": ttfts[n // 2],
        "ttft_p90": ttfts[int(n * 0.9)],
        "ttft_p99": ttfts[int(n * 0.99)] if n >= 100 else ttfts[-1],
        "ttft_min": ttfts[0],
        "ttft_max": ttfts[-1],
        "ttft_first_10_mean": sum(ttfts[:10]) / min(10, n),
        "ttft_last_10_mean": sum(ttfts[-10:]) / min(10, n),
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DualState Trace-Driven Experiment Matrix")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", required=True, help="e.g. baseline_qwen_kimi_trace")
    parser.add_argument("--model-name", default="unknown")
    parser.add_argument("--trace", choices=["kimi", "azure"], required=True)
    parser.add_argument("--duration", type=int, default=120, help="Trace replay duration in seconds")
    parser.add_argument("--scale", type=float, default=1.0, help="Time scale (0.5=2x faster)")
    parser.add_argument("--max-requests", type=int, default=200, help="Max requests to replay")
    parser.add_argument("--prefix-type", choices=["short", "medium", "long"], default="medium")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--start-offset", type=float, default=0, help="Skip first N seconds of trace")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    trace_path = TRACES[args.trace]
    print(f"\n{'='*60}")
    print(f"DualState Trace-Driven Benchmark")
    print(f"  Tag: {args.tag}")
    print(f"  Model: {args.model_name}")
    print(f"  Trace: {args.trace} ({trace_path})")
    print(f"  Duration: {args.duration}s, Scale: {args.scale}x")
    print(f"  Max requests: {args.max_requests}, Concurrency: {args.concurrency}")
    print(f"  Prefix: {args.prefix_type}")
    print(f"{'='*60}")

    # Warmup
    print(f"\n[warmup] {args.warmup} requests...")
    for i in range(args.warmup):
        r = send_request(args.base_url, f"Hello warmup {i}", max_tokens=8)
        if not r["success"]:
            print(f"  WARNING: warmup {i} failed: {r.get('error')}")

    # Load trace
    print(f"\n[load] Loading trace from {trace_path}...")
    trace_requests = load_trace(
        trace_path,
        duration_sec=args.duration,
        scale=args.scale,
        max_requests=args.max_requests,
        start_offset_sec=args.start_offset,
    )
    print(f"  Loaded {len(trace_requests)} requests, "
          f"span: {trace_requests[-1]['arrival_time']:.1f}s" if trace_requests else "  No requests!")

    if not trace_requests:
        print("ERROR: No requests loaded!")
        return

    # Replay
    print(f"\n[replay] Starting trace replay ({len(trace_requests)} requests)...")
    t0 = time.perf_counter()
    results = replay_trace(
        args.base_url, trace_requests,
        prefix_type=args.prefix_type,
        concurrency=args.concurrency,
    )
    wall_time = time.perf_counter() - t0
    print(f"  Replay completed in {wall_time:.1f}s")

    # Stats
    stats = compute_stats(results)
    print(f"\n[stats] Results:")
    print(f"  Success: {stats['n_success']}/{stats['n_total']} ({stats['success_rate']*100:.0f}%)")
    if stats['n_success'] > 0:
        print(f"  TTFT mean: {stats['ttft_mean']*1000:.0f}ms")
        print(f"  TTFT P50:  {stats['ttft_p50']*1000:.0f}ms")
        print(f"  TTFT P90:  {stats['ttft_p90']*1000:.0f}ms")
        print(f"  TTFT P99:  {stats['ttft_p99']*1000:.0f}ms")

    # Save
    output = {
        "tag": args.tag,
        "model_name": args.model_name,
        "trace": args.trace,
        "trace_path": trace_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "duration": args.duration,
            "scale": args.scale,
            "max_requests": args.max_requests,
            "prefix_type": args.prefix_type,
            "concurrency": args.concurrency,
            "start_offset": args.start_offset,
        },
        "stats": stats,
        "wall_time": wall_time,
        "n_requests_loaded": len(trace_requests),
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
