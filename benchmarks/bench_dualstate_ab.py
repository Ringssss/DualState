#!/usr/bin/env python3
"""
DualState A/B benchmark client.
Sends workloads designed to exercise prefix sharing and measures TTFT.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def _no_proxy_session():
    """Create a session that bypasses proxy."""
    import os
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("all_proxy", None)
    s = requests.Session()
    s.trust_env = False
    return s

_session = None

def send_request(base_url, prompt, max_tokens=16, temperature=0, timeout=60):
    """Send a generate request and measure TTFT."""
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
                "prompt_tokens": len(prompt.split()),
            }
        else:
            return {"success": False, "error": f"HTTP {r.status_code}", "ttft": elapsed}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"success": False, "error": str(e), "ttft": elapsed}


def workload_shared_prefix(base_url, n_requests=10):
    """Send requests with identical long prefix + different suffix."""
    shared_prefix = (
        "You are a helpful AI assistant specialized in mathematics and science. "
        "Please analyze the following problem carefully, showing all steps of your work. "
        "Consider edge cases and provide a rigorous solution. "
    ) * 15  # ~450 tokens shared prefix

    results = []
    for i in range(n_requests):
        prompt = shared_prefix + f"\n\nQuestion {i+1}: What is the square root of {(i+1)*17}?"
        result = send_request(base_url, prompt, max_tokens=32)
        result["request_idx"] = i
        result["workload"] = "shared_prefix"
        results.append(result)
        print(f"  req {i}: TTFT={result['ttft']:.3f}s success={result['success']}")
    return results


def workload_multi_turn(base_url, n_turns=5):
    """Simulate multi-turn conversation (growing prefix)."""
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
        print(f"  turn {i}: TTFT={result['ttft']:.3f}s len={len(conversation)}")
    return results


def workload_burst_identical(base_url, n_requests=8, concurrency=4):
    """Burst of identical requests (maximum prefix sharing opportunity)."""
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
    for r in results:
        print(f"  req {r['request_idx']}: TTFT={r['ttft']:.3f}s success={r['success']}")
    return results


def compute_stats(results):
    """Compute summary statistics."""
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
        "ttft_p99": ttfts_sorted[int(n * 0.99)] if n >= 100 else ttfts_sorted[-1],
        "ttft_min": ttfts_sorted[0],
        "ttft_max": ttfts_sorted[-1],
        "ttft_first": ttfts[0] if ttfts else None,
        "ttft_subsequent_mean": sum(ttfts[1:]) / (n - 1) if n > 1 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--tag", default="test")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"Benchmark: {args.tag}")
    print(f"Base URL: {args.base_url}")
    print(f"{'='*50}\n")

    # Warmup
    print("[warmup] Sending warmup requests...")
    for i in range(args.warmup):
        r = send_request(args.base_url, f"Hello, warmup request {i}.", max_tokens=8)
        print(f"  warmup {i}: TTFT={r['ttft']:.3f}s success={r['success']}")

    all_results = {}

    # Workload 1: Shared prefix
    print("\n[workload] shared_prefix (10 requests, same long prefix)")
    results = workload_shared_prefix(args.base_url, n_requests=10)
    all_results["shared_prefix"] = {
        "results": results,
        "stats": compute_stats(results),
    }

    # Workload 2: Multi-turn
    print("\n[workload] multi_turn (5 turns, growing conversation)")
    results = workload_multi_turn(args.base_url, n_turns=5)
    all_results["multi_turn"] = {
        "results": results,
        "stats": compute_stats(results),
    }

    # Workload 3: Burst identical
    print("\n[workload] burst_identical (8 requests, concurrency=4)")
    results = workload_burst_identical(args.base_url, n_requests=8, concurrency=4)
    all_results["burst_identical"] = {
        "results": results,
        "stats": compute_stats(results),
    }

    # Summary
    output = {
        "tag": args.tag,
        "base_url": args.base_url,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workloads": all_results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.tag}")
    print(f"{'='*60}")
    print(f"{'Workload':<20} {'Mean TTFT':>10} {'P50':>8} {'1st':>8} {'Sub-Mean':>10}")
    print(f"{'-'*60}")
    for name, data in all_results.items():
        s = data["stats"]
        if s["n_success"] > 0:
            sub = f"{s['ttft_subsequent_mean']:.3f}s" if s["ttft_subsequent_mean"] else "N/A"
            print(
                f"{name:<20} {s['ttft_mean']:.3f}s "
                f"{s['ttft_p50']:.3f}s "
                f"{s['ttft_first']:.3f}s "
                f"{sub:>10}"
            )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
