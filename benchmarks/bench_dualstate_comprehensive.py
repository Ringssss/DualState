#!/usr/bin/env python3
"""
DualState Comprehensive Benchmark.

Tests DualState vs Baseline across:
- Multiple workloads: shared_prefix, multi_turn, burst_identical
- Multiple arrival rates: back-to-back, 100ms, 500ms, 1000ms inter-request delays
- Reports TTFT breakdown per workload/rate
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def _no_proxy_session():
    import os
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    for k in ("http_proxy", "https_proxy", "all_proxy", "ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    s = requests.Session()
    s.trust_env = False
    return s


_session = None


def send_request(base_url, prompt, max_tokens=16, temperature=0, timeout=90):
    global _session
    if _session is None:
        _session = _no_proxy_session()
    start = time.perf_counter()
    try:
        r = _session.post(
            f"{base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                },
            },
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        if r.status_code == 200:
            data = r.json()
            return {
                "success": True,
                "ttft": elapsed,
                "text": data.get("text", "")[:100],
                "prompt_tokens": data.get("meta_info", {}).get("prompt_tokens", 0),
            }
        else:
            return {"success": False, "error": f"HTTP {r.status_code}", "ttft": elapsed}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"success": False, "error": str(e), "ttft": elapsed}


# ─── Workloads ──────────────────────────────────────────────────────────────

def workload_shared_prefix(base_url, n_requests=10, delay_ms=0):
    """Sequential requests with same long prefix, different suffix."""
    shared_prefix = (
        "You are a helpful AI assistant specialized in mathematics and science. "
        "Please analyze the following problem carefully, showing all steps of your work. "
        "Consider edge cases and provide a rigorous solution. "
    ) * 15  # ~450 tokens

    results = []
    for i in range(n_requests):
        prompt = shared_prefix + f"\n\nQuestion {i+1}: What is the square root of {(i+1)*17}?"
        result = send_request(base_url, prompt, max_tokens=32)
        result["request_idx"] = i
        result["workload"] = "shared_prefix"
        results.append(result)
        if delay_ms > 0 and i < n_requests - 1:
            time.sleep(delay_ms / 1000.0)
    return results


def workload_multi_turn(base_url, n_turns=5, delay_ms=0):
    """Multi-turn conversation with growing context."""
    system = "You are a concise tutor. Answer in one sentence."
    conversation = system
    results = []

    questions = [
        "What is differentiation?",
        "Give an example with x^3.",
        "What about integration?",
        "Integrate x^2 from 0 to 1.",
        "What is the fundamental theorem of calculus?",
    ]

    for i, q in enumerate(questions[:n_turns]):
        conversation += f" User: {q} Assistant:"
        result = send_request(base_url, conversation, max_tokens=48)
        result["request_idx"] = i
        result["workload"] = "multi_turn"
        result["turn"] = i
        result["conversation_len"] = len(conversation)
        results.append(result)
        answer = result.get("text", "I don't know.")[:80]
        conversation += f" {answer}"
        if delay_ms > 0 and i < n_turns - 1:
            time.sleep(delay_ms / 1000.0)
    return results


def workload_burst_identical(base_url, n_requests=8, concurrency=4):
    """Burst of identical requests (max prefix sharing opportunity)."""
    prompt = (
        "Explain the concept of attention mechanism in transformers. "
        "Cover multi-head attention, scaled dot-product attention, "
        "and how it enables parallel processing of sequences. "
    ) * 10  # ~300 tokens

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(send_request, base_url, prompt, 32): i
            for i in range(n_requests)
        }
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            result["request_idx"] = idx
            result["workload"] = "burst_identical"
            results.append(result)

    results.sort(key=lambda x: x["request_idx"])
    return results


# ─── Stats ──────────────────────────────────────────────────────────────────

def compute_stats(results):
    successful = [r for r in results if r.get("success")]
    if not successful:
        return {"n_total": len(results), "n_success": 0}

    ttfts = [r["ttft"] for r in successful]
    ttfts_sorted = sorted(ttfts)
    n = len(ttfts_sorted)

    return {
        "n_total": len(results),
        "n_success": n,
        "ttft_mean": sum(ttfts) / n,
        "ttft_p50": ttfts_sorted[n // 2],
        "ttft_p90": ttfts_sorted[int(n * 0.9)] if n >= 10 else ttfts_sorted[-1],
        "ttft_min": ttfts_sorted[0],
        "ttft_max": ttfts_sorted[-1],
        "ttft_first": ttfts[0],
        "ttft_subsequent_mean": sum(ttfts[1:]) / (n - 1) if n > 1 else None,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def run_benchmark(base_url, delays_ms, n_warmup=3):
    """Run all workloads at all arrival rates."""
    # Warmup
    print(f"  [warmup] {n_warmup} requests...")
    for i in range(n_warmup):
        r = send_request(base_url, f"Hello warmup {i}", max_tokens=8)
        if not r["success"]:
            print(f"    WARNING: warmup {i} failed: {r.get('error')}")

    all_results = {}

    for delay_ms in delays_ms:
        delay_label = f"{delay_ms}ms" if delay_ms > 0 else "back2back"
        print(f"\n  === Arrival rate: {delay_label} ===")

        # Shared prefix
        print(f"    [shared_prefix] 10 requests, delay={delay_label}...")
        results = workload_shared_prefix(base_url, n_requests=10, delay_ms=delay_ms)
        stats = compute_stats(results)
        all_results[f"shared_prefix_{delay_label}"] = {
            "results": results,
            "stats": stats,
            "delay_ms": delay_ms,
            "workload": "shared_prefix",
        }
        succ = stats["n_success"]
        sub_mean = stats.get("ttft_subsequent_mean")
        print(f"      → {succ}/10 success, sub_mean={sub_mean:.3f}s" if sub_mean else f"      → {succ}/10 success")

        # Multi-turn
        print(f"    [multi_turn] 5 turns, delay={delay_label}...")
        results = workload_multi_turn(base_url, n_turns=5, delay_ms=delay_ms)
        stats = compute_stats(results)
        all_results[f"multi_turn_{delay_label}"] = {
            "results": results,
            "stats": stats,
            "delay_ms": delay_ms,
            "workload": "multi_turn",
        }
        succ = stats["n_success"]
        print(f"      → {succ}/5 success, mean={stats.get('ttft_mean', 0):.3f}s")

        # Burst identical (only for back-to-back)
        if delay_ms == 0:
            print(f"    [burst_identical] 8 requests, concurrency=4...")
            results = workload_burst_identical(base_url, n_requests=8, concurrency=4)
            stats = compute_stats(results)
            all_results[f"burst_identical_{delay_label}"] = {
                "results": results,
                "stats": stats,
                "delay_ms": delay_ms,
                "workload": "burst_identical",
            }
            succ = stats["n_success"]
            print(f"      → {succ}/8 success, mean={stats.get('ttft_mean', 0):.3f}s")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="DualState Comprehensive Benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--tag", required=True, help="Tag (e.g. baseline_qwen36, dualstate_qwen36)")
    parser.add_argument("--model-name", default="unknown", help="Model name for metadata")
    parser.add_argument(
        "--delays", default="0,100,500",
        help="Comma-separated inter-request delays in ms (default: 0,100,500)"
    )
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    delays_ms = [int(d) for d in args.delays.split(",")]

    print(f"\n{'='*60}")
    print(f"DualState Comprehensive Benchmark")
    print(f"  Tag: {args.tag}")
    print(f"  Model: {args.model_name}")
    print(f"  URL: {args.base_url}")
    print(f"  Delays: {delays_ms} ms")
    print(f"{'='*60}")

    all_results = run_benchmark(args.base_url, delays_ms, n_warmup=args.warmup)

    output = {
        "tag": args.tag,
        "model_name": args.model_name,
        "base_url": args.base_url,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "delays_ms": delays_ms,
        "workloads": all_results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: {args.tag} ({args.model_name})")
    print(f"{'='*70}")
    print(f"{'Workload':<30} {'Success':>8} {'1st TTFT':>10} {'Sub Mean':>10} {'P50':>8}")
    print(f"{'-'*70}")
    for name, data in all_results.items():
        s = data["stats"]
        if s["n_success"] > 0:
            sub = f"{s['ttft_subsequent_mean']:.3f}s" if s.get("ttft_subsequent_mean") else "N/A"
            print(
                f"{name:<30} {s['n_success']}/{s['n_total']:>4}  "
                f"{s['ttft_first']:.3f}s  "
                f"{sub:>10}  "
                f"{s['ttft_p50']:.3f}s"
            )
        else:
            print(f"{name:<30} {s['n_success']}/{s['n_total']:>4}  FAILED")
    print(f"{'='*70}\n")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
