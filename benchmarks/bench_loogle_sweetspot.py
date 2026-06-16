#!/usr/bin/env python3
"""
DualState Long-Context Sweet-Spot Benchmark.

Uses LooGLE real documents (14k-50k words each) to find the peak benefit
of DualState under long shared-prefix + high fan-out conditions.

Modes:
  sweep   — systematic prefix_len × fan_out × delay matrix
  trace   — trace-driven replay with LooGLE documents as content

Usage:
  # Sweep mode (core)
  python bench_loogle_sweetspot.py sweep \
    --base-url http://127.0.0.1:30000 \
    --output results.json --tag dualstate_qwen36 \
    --prefix-lens 2048,4096,8192,16384 \
    --fan-outs 8,16,32 --delays 0,200

  # Trace replay mode
  python bench_loogle_sweetspot.py trace \
    --base-url http://127.0.0.1:30000 \
    --output results.json --tag baseline_kimi_trace \
    --trace kimi --prefix-len 8192 --fan-out 16 --duration 120

  # Quick smoke test
  python bench_loogle_sweetspot.py sweep \
    --base-url http://127.0.0.1:30000 \
    --output smoke.json --tag smoke \
    --prefix-lens 4096 --fan-outs 8 --delays 0
"""

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests as req_lib

# ─── Constants ─────────────────────────────────────────────────────────────

LOOGLE_PATH = "/mnt/models/LooGLE/data/shortdep_qa.jsonl"

TRACES = {
    "kimi": "/mnt/models/kimik25/kimi-k25-trace/kimi_k25_conv_1day.csv",
    "azure": "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv",
}

# ─── HTTP session ──────────────────────────────────────────────────────────

_session = None


def _get_session():
    global _session
    if _session is None:
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        for k in ("http_proxy", "https_proxy", "all_proxy", "ALL_PROXY",
                   "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(k, None)
        _session = req_lib.Session()
        _session.trust_env = False
    return _session


def send_request(base_url, prompt, max_tokens=16, timeout=180):
    """Send a /generate request and measure TTFT."""
    s = _get_session()
    start = time.perf_counter()
    try:
        r = s.post(
            f"{base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
            },
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        if r.status_code == 200:
            data = r.json()
            meta = data.get("meta_info", {})
            return {
                "success": True,
                "ttft": elapsed,
                "prompt_tokens": meta.get("prompt_tokens", len(prompt.split())),
                "completion_tokens": meta.get("completion_tokens", 0),
            }
        return {"success": False, "error": f"HTTP {r.status_code}", "ttft": elapsed}
    except Exception as e:
        return {"success": False, "error": str(e), "ttft": time.perf_counter() - start}


# ─── LooGLE Data Loading ──────────────────────────────────────────────────

def load_loogle_documents(path=LOOGLE_PATH, min_questions=8):
    """
    Load LooGLE shortdep_qa, group by doc_id, keep docs with ≥min_questions.
    Returns list of {doc_id, title, context, questions: [str]}.
    """
    by_doc = defaultdict(lambda: {"questions": [], "context": "", "title": ""})

    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            doc_id = entry["doc_id"]
            by_doc[doc_id]["context"] = entry["context"]
            by_doc[doc_id]["title"] = entry.get("title", "")
            by_doc[doc_id]["questions"].append(entry["question"])

    docs = []
    for doc_id, info in by_doc.items():
        if len(info["questions"]) >= min_questions:
            docs.append({
                "doc_id": doc_id,
                "title": info["title"],
                "context": info["context"],
                "questions": info["questions"],
                "word_count": len(info["context"].split()),
            })

    docs.sort(key=lambda d: len(d["questions"]), reverse=True)
    return docs


def truncate_to_tokens(text, target_tokens):
    """
    Truncate text to approximately target_tokens.
    Rough estimate: 1 word ≈ 1.3 tokens for English text.
    """
    target_words = int(target_tokens / 1.3)
    words = text.split()
    if len(words) <= target_words:
        return text
    return " ".join(words[:target_words])


def build_prompts(doc, prefix_token_len, fan_out):
    """
    Build fan_out prompts from a single document:
    - Shared prefix = truncated document context
    - Unique suffix = different question for each request
    """
    prefix = truncate_to_tokens(doc["context"], prefix_token_len)
    questions = doc["questions"][:fan_out]

    # If not enough questions, cycle through them
    while len(questions) < fan_out:
        extra_idx = len(questions) % len(doc["questions"])
        q = doc["questions"][extra_idx]
        questions.append(f"{q} (variant {len(questions)})")

    prompts = []
    for i, q in enumerate(questions):
        full_prompt = f"Document:\n{prefix}\n\nQuestion: {q}\nAnswer concisely:"
        prompts.append({
            "prompt": full_prompt,
            "question_idx": i,
            "question": q[:100],
            "prefix_words": len(prefix.split()),
        })
    return prompts


# ─── Stats ─────────────────────────────────────────────────────────────────

def compute_stats(results):
    successful = [r for r in results if r.get("success")]
    n = len(successful)
    if n == 0:
        return {"n_total": len(results), "n_success": 0, "success_rate": 0}

    ttfts = sorted(r["ttft"] for r in successful)
    warm = ttfts[1:] if n > 1 else ttfts  # skip first (cold cache)

    stats = {
        "n_total": len(results),
        "n_success": n,
        "success_rate": n / len(results),
        "ttft_mean": statistics.mean(ttfts),
        "ttft_p50": ttfts[n // 2],
        "ttft_p90": ttfts[int(n * 0.9)] if n >= 10 else ttfts[-1],
        "ttft_p95": ttfts[int(n * 0.95)] if n >= 20 else ttfts[-1],
        "ttft_p99": ttfts[int(n * 0.99)] if n >= 100 else ttfts[-1],
        "ttft_min": ttfts[0],
        "ttft_max": ttfts[-1],
        "ttft_first": ttfts[0],
    }
    if warm:
        stats["ttft_subsequent_mean"] = statistics.mean(warm)
        stats["ttft_subsequent_p50"] = sorted(warm)[len(warm) // 2]
    return stats


# ─── Sweep Mode ────────────────────────────────────────────────────────────

def run_sweep_point(base_url, doc, prefix_len, fan_out, delay_ms, concurrency,
                    max_tokens=16):
    """
    Run one sweep point:
    1. Send a warm-up request (seed the cache)
    2. Send fan_out requests sharing the same prefix, different suffixes
    """
    prompts = build_prompts(doc, prefix_len, fan_out)

    # Warm-up: send first prompt to seed prefix cache
    warmup = send_request(base_url, prompts[0]["prompt"], max_tokens=8, timeout=300)

    # Run fan-out requests
    results = []
    if delay_ms == 0 and concurrency > 1:
        # Burst mode: concurrent
        with ThreadPoolExecutor(max_workers=min(concurrency, fan_out)) as ex:
            futures = {}
            for i, p in enumerate(prompts):
                f = ex.submit(send_request, base_url, p["prompt"], max_tokens)
                futures[f] = i
            for f in as_completed(futures):
                idx = futures[f]
                result = f.result()
                result["request_idx"] = idx
                result["question"] = prompts[idx]["question"]
                results.append(result)
        results.sort(key=lambda r: r["request_idx"])
    else:
        # Sequential with delay
        for i, p in enumerate(prompts):
            result = send_request(base_url, p["prompt"], max_tokens)
            result["request_idx"] = i
            result["question"] = p["question"]
            results.append(result)
            if delay_ms > 0 and i < len(prompts) - 1:
                time.sleep(delay_ms / 1000.0)

    return {
        "results": results,
        "stats": compute_stats(results),
        "warmup_ttft": warmup.get("ttft"),
        "warmup_success": warmup.get("success", False),
        "doc_id": doc["doc_id"],
        "doc_title": doc["title"][:80],
        "prefix_words": prompts[0]["prefix_words"],
    }


def run_sweep(args):
    """Run full sweep matrix."""
    docs = load_loogle_documents()
    print(f"Loaded {len(docs)} documents with ≥8 questions")

    prefix_lens = [int(x) for x in args.prefix_lens.split(",")]
    fan_outs = [int(x) for x in args.fan_outs.split(",")]
    delays = [int(x) for x in args.delays.split(",")]
    concurrency = args.concurrency

    # Select documents: pick ones with enough questions for max fan_out
    max_fan = max(fan_outs)
    usable_docs = [d for d in docs if len(d["questions"]) >= max_fan]
    if not usable_docs:
        usable_docs = docs[:5]
    # Use top N docs (most questions)
    selected_docs = usable_docs[:args.num_docs]
    print(f"Selected {len(selected_docs)} documents (max fan_out={max_fan})")
    for d in selected_docs:
        print(f"  {d['doc_id'][:12]}: {d['word_count']} words, {len(d['questions'])} questions")

    total_points = len(prefix_lens) * len(fan_outs) * len(delays) * len(selected_docs)
    print(f"\nSweep matrix: {len(prefix_lens)} prefix × {len(fan_outs)} fan_out × "
          f"{len(delays)} delays × {len(selected_docs)} docs = {total_points} points")

    # Warmup server
    print(f"\n[warmup] {args.warmup} requests...")
    for i in range(args.warmup):
        r = send_request(args.base_url, f"Hello warmup {i}", max_tokens=8, timeout=300)
        if r.get("success"):
            print(f"  warmup {i}: {r['ttft']:.3f}s")
        else:
            print(f"  warmup {i}: FAILED - {r.get('error', '?')}")

    sweep_results = []
    point_idx = 0

    for prefix_len in prefix_lens:
        for fan_out in fan_outs:
            for delay_ms in delays:
                for doc in selected_docs:
                    point_idx += 1
                    label = f"p{prefix_len}_f{fan_out}_d{delay_ms}_{doc['doc_id'][:8]}"
                    print(f"\n[{point_idx}/{total_points}] {label}")

                    result = run_sweep_point(
                        args.base_url, doc, prefix_len, fan_out,
                        delay_ms, concurrency, max_tokens=args.max_tokens,
                    )
                    result["label"] = label
                    result["prefix_len"] = prefix_len
                    result["fan_out"] = fan_out
                    result["delay_ms"] = delay_ms
                    result["concurrency"] = concurrency

                    s = result["stats"]
                    sub = s.get("ttft_subsequent_mean")
                    print(f"  → {s['n_success']}/{s['n_total']} success, "
                          f"mean={s.get('ttft_mean', 0):.4f}s, "
                          f"sub_mean={sub:.4f}s" if sub else "")

                    sweep_results.append(result)

    return sweep_results


# ─── Trace Mode ────────────────────────────────────────────────────────────

def load_trace_arrivals(trace_path, duration_sec=120, scale=1.0,
                        max_requests=500, start_offset=60):
    """Load arrival times from production trace."""
    arrivals = []
    first_ts = None

    with open(trace_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row["TIMESTAMP"]
            ctx_tokens = int(row["ContextTokens"])

            try:
                ts = datetime.fromisoformat(ts_str.replace("+00:00", "+00:00"))
            except ValueError:
                continue

            if first_ts is None:
                first_ts = ts

            elapsed = (ts - first_ts).total_seconds()
            if elapsed < start_offset:
                continue

            rel_time = (elapsed - start_offset) * scale
            if rel_time > duration_sec:
                break

            arrivals.append({
                "arrival_time": rel_time,
                "original_ctx_tokens": ctx_tokens,
            })
            if len(arrivals) >= max_requests:
                break

    return arrivals


def run_trace(args):
    """
    Trace-driven replay: use trace arrival times but replace content
    with LooGLE long documents for realistic prefix sharing.
    """
    docs = load_loogle_documents()
    print(f"Loaded {len(docs)} documents")

    # Select documents for prefix sharing groups
    selected = [d for d in docs if len(d["questions"]) >= args.fan_out][:args.num_docs]
    if not selected:
        selected = docs[:3]
    print(f"Selected {len(selected)} documents for trace replay")

    trace_path = TRACES[args.trace]
    arrivals = load_trace_arrivals(
        trace_path, args.duration, args.scale,
        args.max_requests, args.start_offset,
    )
    print(f"Loaded {len(arrivals)} arrivals from {args.trace} trace")
    if not arrivals:
        return []

    # Assign arrivals to document groups (round-robin)
    # Each group of fan_out consecutive requests shares a document prefix
    fan_out = args.fan_out
    prefix_len = args.prefix_len

    # Build prompts for each document
    doc_prompts = {}
    for doc in selected:
        doc_prompts[doc["doc_id"]] = build_prompts(doc, prefix_len, fan_out * 2)

    # Warmup
    print(f"\n[warmup] {args.warmup} requests...")
    for i in range(args.warmup):
        r = send_request(args.base_url, f"Hello warmup {i}", max_tokens=8, timeout=300)
        status = "ok" if r.get("success") else f"FAIL: {r.get('error', '?')}"
        print(f"  warmup {i}: {status}")

    # Replay
    results = []
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for i, arrival in enumerate(arrivals):
            # Pick document (cycle through selected docs)
            doc = selected[i // fan_out % len(selected)]
            prompts = doc_prompts[doc["doc_id"]]
            prompt_info = prompts[i % len(prompts)]

            # Wait until arrival time
            target_time = start_time + arrival["arrival_time"]
            now = time.perf_counter()
            if target_time > now:
                time.sleep(target_time - now)

            future = executor.submit(
                send_request, args.base_url, prompt_info["prompt"],
                args.max_tokens, 180,
            )
            futures[future] = {
                "idx": i,
                "arrival_time": arrival["arrival_time"],
                "doc_id": doc["doc_id"][:12],
                "question_idx": prompt_info["question_idx"],
            }

        for future in as_completed(futures):
            info = futures[future]
            result = future.result()
            result.update(info)
            results.append(result)

    results.sort(key=lambda r: r["idx"])
    wall_time = time.perf_counter() - start_time

    return [{
        "label": f"trace_{args.trace}_p{prefix_len}_f{fan_out}",
        "trace": args.trace,
        "prefix_len": prefix_len,
        "fan_out": fan_out,
        "concurrency": args.concurrency,
        "wall_time": wall_time,
        "n_arrivals": len(arrivals),
        "results": results,
        "stats": compute_stats(results),
    }]


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DualState Long-Context Sweet-Spot Benchmark"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # Sweep mode
    sp_sweep = sub.add_parser("sweep", help="Systematic sweep matrix")
    sp_sweep.add_argument("--base-url", default="http://127.0.0.1:30000")
    sp_sweep.add_argument("--output", required=True)
    sp_sweep.add_argument("--tag", required=True)
    sp_sweep.add_argument("--model-name", default="unknown")
    sp_sweep.add_argument("--prefix-lens", default="2048,4096,8192,16384",
                          help="Comma-separated prefix token lengths")
    sp_sweep.add_argument("--fan-outs", default="8,16,32",
                          help="Comma-separated fan-out counts")
    sp_sweep.add_argument("--delays", default="0,200",
                          help="Comma-separated inter-request delays in ms")
    sp_sweep.add_argument("--concurrency", type=int, default=16)
    sp_sweep.add_argument("--max-tokens", type=int, default=16)
    sp_sweep.add_argument("--num-docs", type=int, default=3,
                          help="Number of documents to use")
    sp_sweep.add_argument("--warmup", type=int, default=3)

    # Trace mode
    sp_trace = sub.add_parser("trace", help="Trace-driven replay")
    sp_trace.add_argument("--base-url", default="http://127.0.0.1:30000")
    sp_trace.add_argument("--output", required=True)
    sp_trace.add_argument("--tag", required=True)
    sp_trace.add_argument("--model-name", default="unknown")
    sp_trace.add_argument("--trace", choices=["kimi", "azure"], required=True)
    sp_trace.add_argument("--prefix-len", type=int, default=8192)
    sp_trace.add_argument("--fan-out", type=int, default=16)
    sp_trace.add_argument("--duration", type=int, default=120)
    sp_trace.add_argument("--scale", type=float, default=1.0)
    sp_trace.add_argument("--max-requests", type=int, default=200)
    sp_trace.add_argument("--start-offset", type=float, default=60)
    sp_trace.add_argument("--concurrency", type=int, default=16)
    sp_trace.add_argument("--max-tokens", type=int, default=16)
    sp_trace.add_argument("--num-docs", type=int, default=5)
    sp_trace.add_argument("--warmup", type=int, default=3)

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"DualState Long-Context Sweet-Spot Benchmark")
    print(f"  Mode: {args.mode}")
    print(f"  Tag: {args.tag}")
    print(f"  URL: {args.base_url}")
    print(f"{'='*70}")

    if args.mode == "sweep":
        sweep_results = run_sweep(args)
    else:
        sweep_results = run_trace(args)

    # Save results
    output = {
        "tag": args.tag,
        "model_name": args.model_name,
        "mode": args.mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sweep_results": sweep_results,
    }
    # Add mode-specific config
    if args.mode == "sweep":
        output["config"] = {
            "prefix_lens": args.prefix_lens,
            "fan_outs": args.fan_outs,
            "delays": args.delays,
            "concurrency": args.concurrency,
            "num_docs": args.num_docs,
        }
    else:
        output["config"] = {
            "trace": args.trace,
            "prefix_len": args.prefix_len,
            "fan_out": args.fan_out,
            "duration": args.duration,
            "scale": args.scale,
            "concurrency": args.concurrency,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — {args.tag}")
    print(f"{'='*70}")
    if args.mode == "sweep":
        print(f"{'Label':<45} {'OK':>5} {'Mean':>8} {'SubMean':>8} {'P50':>8}")
        print("-" * 78)
        for sr in sweep_results:
            s = sr["stats"]
            sub = s.get("ttft_subsequent_mean")
            if s["n_success"] > 0:
                print(f"  {sr['label']:<43} {s['n_success']:>3}/{s['n_total']:<3}"
                      f" {s['ttft_mean']:.4f}s"
                      f" {sub:.4f}s" if sub else ""
                      f" {s['ttft_p50']:.4f}s")
    else:
        for sr in sweep_results:
            s = sr["stats"]
            if s["n_success"] > 0:
                print(f"  {sr['label']}")
                print(f"    Success: {s['n_success']}/{s['n_total']}")
                print(f"    TTFT mean={s['ttft_mean']:.4f}s  p50={s['ttft_p50']:.4f}s  "
                      f"p90={s.get('ttft_p90', 0):.4f}s")

    print(f"\nResults saved: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
