#!/usr/bin/env python3
"""
Benchmark workload harness for hybrid model prefix cache state-gating analysis.

Generates OpenAI-compatible requests against a running SGLang server to exercise
different prefix sharing patterns and measure state-gated prefix reuse behavior.

Usage:
    # Run all workloads
    python bench_hybrid_prefix_state_gate.py --server-url http://localhost:30000

    # Run specific workload
    python bench_hybrid_prefix_state_gate.py --server-url http://localhost:30000 --workload shared_document_qa

    # Custom output dir
    python bench_hybrid_prefix_state_gate.py --server-url http://localhost:30000 --output-dir results/run1
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Bypass proxy for localhost connections
for _pvar in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "all_proxy", "ALL_PROXY", "SOCKS_PROXY", "socks_proxy"]:
    os.environ.pop(_pvar, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def get_model_name(server_url: str) -> str:
    resp = requests.get(f"{server_url}/v1/models", timeout=10)
    resp.raise_for_status()
    models = resp.json()["data"]
    return models[0]["id"] if models else ""


def send_chat_request(
    server_url: str,
    model: str,
    messages: List[dict],
    max_tokens: int = 64,
    temperature: float = 0.0,
    timeout: int = 120,
) -> dict:
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        end = time.perf_counter()
        usage = data.get("usage", {})
        return {
            "status": "ok",
            "latency": end - start,
            "ttft": end - start,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "response_id": data.get("id", ""),
        }
    except Exception as e:
        end = time.perf_counter()
        return {
            "status": "error",
            "error": str(e),
            "latency": end - start,
            "ttft": end - start,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "response_id": "",
        }


def generate_long_text(base: str, target_words: int) -> str:
    words = base.split()
    if len(words) == 0:
        words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog."]
    result = []
    while len(result) < target_words:
        result.extend(words)
    return " ".join(result[:target_words])


SYSTEM_PROMPT_BASE = (
    "You are a helpful AI assistant. You have been trained on a wide variety of data "
    "and can assist with many tasks including writing, analysis, coding, math, and more. "
    "Please provide clear, accurate, and helpful responses. "
)

DOCUMENT_BASE = (
    "The following is a technical document about distributed systems and machine learning "
    "infrastructure. Modern large language models require sophisticated serving infrastructure "
    "that can handle high throughput while maintaining low latency. Key challenges include "
    "memory management, batching strategies, and prefix caching. The KV cache stores "
    "key-value pairs from the attention mechanism, enabling efficient autoregressive generation. "
    "For hybrid models that combine full attention with linear attention or state-space models, "
    "additional state management is required. The recurrent state from Mamba or linear attention "
    "layers must be checkpointed and managed alongside the KV cache. "
)

TOOL_SCHEMA = """You have access to the following tools:
1. search(query: str) -> List[str]: Search the knowledge base for relevant information.
2. calculate(expression: str) -> float: Evaluate a mathematical expression.
3. get_weather(city: str) -> dict: Get current weather for a city.
4. translate(text: str, target_lang: str) -> str: Translate text to target language.
5. summarize(text: str, max_words: int) -> str: Summarize text in given word limit.
When you need to use a tool, respond with the tool call in JSON format."""

QUESTIONS = [
    "What is the main topic of this document?",
    "Summarize the key challenges mentioned.",
    "What role does the KV cache play?",
    "How do hybrid models differ from pure transformers?",
    "What is prefix caching and why is it important?",
    "Explain the relationship between Mamba state and KV cache.",
    "What are the memory management challenges?",
    "How does batching strategy affect throughput?",
    "What is the difference between full attention and linear attention?",
    "Describe the checkpointing requirements for hybrid models.",
    "What is autoregressive generation?",
    "How does the attention mechanism work?",
    "What are state-space models?",
    "Explain the concept of recurrent state.",
    "What is the significance of low latency in LLM serving?",
    "How does distributed serving work?",
    "What is the relationship between throughput and latency?",
    "Describe the key-value pair storage mechanism.",
    "What optimizations can improve serving performance?",
    "How do modern LLMs handle long sequences?",
]


def workload_identical_prompt(n: int = 10, prompt_words: int = 500) -> List[List[dict]]:
    long_prompt = generate_long_text(SYSTEM_PROMPT_BASE + DOCUMENT_BASE, prompt_words)
    messages = [
        {"role": "system", "content": long_prompt},
        {"role": "user", "content": "Summarize the above in one sentence."},
    ]
    return [messages for _ in range(n)]


def workload_shared_system_prompt(
    n: int = 10, system_words: int = 500
) -> List[List[dict]]:
    system_prompt = generate_long_text(SYSTEM_PROMPT_BASE, system_words)
    requests_list = []
    for i in range(n):
        q = QUESTIONS[i % len(QUESTIONS)]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        requests_list.append(messages)
    return requests_list


def workload_shared_document_qa(
    n: int = 10, doc_words: int = 1000
) -> List[List[dict]]:
    document = generate_long_text(DOCUMENT_BASE, doc_words)
    requests_list = []
    for i in range(n):
        q = QUESTIONS[i % len(QUESTIONS)]
        messages = [
            {"role": "system", "content": "You are a document QA assistant."},
            {
                "role": "user",
                "content": f"Document:\n{document}\n\nQuestion: {q}",
            },
        ]
        requests_list.append(messages)
    return requests_list


def workload_multi_turn_agent(n: int = 10) -> List[List[dict]]:
    requests_list = []
    base_history = [
        {"role": "system", "content": TOOL_SCHEMA},
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {
            "role": "assistant",
            "content": '{"tool": "get_weather", "args": {"city": "Tokyo"}}',
        },
        {
            "role": "user",
            "content": "The weather is 22C and sunny. Now translate 'hello world' to Japanese.",
        },
        {
            "role": "assistant",
            "content": '{"tool": "translate", "args": {"text": "hello world", "target_lang": "ja"}}',
        },
    ]
    follow_ups = [
        "Now search for information about distributed systems.",
        "Calculate 3.14 * 2.718 * 1.618.",
        "What's the weather in London?",
        "Summarize our conversation so far in 50 words.",
        "Translate 'machine learning' to French.",
        "Search for recent papers on prefix caching.",
        "Calculate the factorial of 10.",
        "What's the weather in New York?",
        "Summarize the tool outputs we've seen.",
        "Translate 'artificial intelligence' to Spanish.",
    ]
    for i in range(n):
        msgs = list(base_history) + [
            {"role": "user", "content": follow_ups[i % len(follow_ups)]}
        ]
        requests_list.append(msgs)
    return requests_list


def workload_adversarial_near_prefix(
    n: int = 10, prefix_words: int = 500
) -> List[List[dict]]:
    base_text = generate_long_text(DOCUMENT_BASE, prefix_words)
    words = base_text.split()
    requests_list = []
    for i in range(n):
        branch_point = max(1, len(words) - 10 - i * 3)
        modified = list(words)
        modified[branch_point] = f"VARIANT_{i}"
        text = " ".join(modified)
        messages = [
            {"role": "user", "content": text + "\n\nSummarize this."},
        ]
        requests_list.append(messages)
    return requests_list


def workload_prefix_ladder(n_levels: int = 6, base_words: int = 200) -> List[List[dict]]:
    base = generate_long_text(DOCUMENT_BASE, base_words)
    words = base.split()
    requests_list = []
    lengths = [base_words * (2**i) // (2 ** (n_levels - 1)) for i in range(n_levels)]
    lengths = [max(50, min(l, len(words))) for l in lengths]

    for length in lengths:
        prefix = " ".join(words[:length])
        messages = [
            {"role": "user", "content": prefix + "\n\nWhat is this about?"},
        ]
        requests_list.append(messages)

    for length in lengths[2:]:
        for q_idx in range(2):
            prefix = " ".join(words[:length])
            q = QUESTIONS[q_idx]
            messages = [
                {"role": "user", "content": prefix + f"\n\nQuestion: {q}"},
            ]
            requests_list.append(messages)

    return requests_list


def workload_branch_after_cached_leaf(
    n_branches: int = 5, doc_words: int = 800
) -> List[List[dict]]:
    document = generate_long_text(DOCUMENT_BASE, doc_words)
    requests_list = []
    first_msg = [
        {
            "role": "user",
            "content": f"Document:\n{document}\n\nQuestion: {QUESTIONS[0]}",
        }
    ]
    requests_list.append(first_msg)

    for i in range(1, n_branches + 1):
        messages = [
            {
                "role": "user",
                "content": f"Document:\n{document}\n\nQuestion: {QUESTIONS[i % len(QUESTIONS)]}",
            }
        ]
        requests_list.append(messages)

    return requests_list


def workload_random_no_share(n: int = 10, prompt_words: int = 200) -> List[List[dict]]:
    """Negative control: completely unrelated prompts with no shared prefix."""
    import hashlib
    topics = [
        "quantum computing and qubit error correction",
        "19th century impressionist painting techniques",
        "deep sea bioluminescent organisms and their ecology",
        "ancient Roman aqueduct engineering and hydraulics",
        "the mathematics of fractals and Mandelbrot sets",
        "traditional Japanese tea ceremony Chanoyu rituals",
        "CRISPR gene editing in agricultural applications",
        "the history of jazz music in New Orleans",
        "exoplanet atmospheric composition detection methods",
        "Mesoamerican pyramid construction and astronomy",
        "blockchain consensus mechanisms proof of stake",
        "Renaissance sculpture techniques by Donatello",
        "neural network pruning and knowledge distillation",
        "Polynesian navigation using ocean currents and stars",
        "superconducting materials at room temperature research",
    ]
    requests_list = []
    for i in range(n):
        topic = topics[i % len(topics)]
        seed = hashlib.md5(f"random_{i}_{topic}".encode()).hexdigest()[:8]
        filler = generate_long_text(
            f"Topic {seed}: {topic}. This is a unique discussion about {topic} "
            f"that has no overlap with any other prompt in this batch. ",
            prompt_words,
        )
        messages = [
            {"role": "user", "content": f"{filler}\n\nExplain the main concept in one sentence."},
        ]
        requests_list.append(messages)
    return requests_list


WORKLOADS = {
    "random_no_share": workload_random_no_share,
    "identical_prompt": workload_identical_prompt,
    "shared_system_prompt": workload_shared_system_prompt,
    "shared_document_qa": workload_shared_document_qa,
    "multi_turn_agent": workload_multi_turn_agent,
    "adversarial_near_prefix": workload_adversarial_near_prefix,
    "prefix_ladder": workload_prefix_ladder,
    "branch_after_cached_leaf": workload_branch_after_cached_leaf,
}


def run_workload(
    server_url: str,
    model: str,
    workload_name: str,
    messages_list: List[List[dict]],
    max_tokens: int = 64,
    concurrency: int = 1,
    warmup: int = 0,
) -> List[dict]:
    results = []

    if warmup > 0:
        print(f"  Warmup: sending {warmup} requests...")
        for i in range(min(warmup, len(messages_list))):
            send_chat_request(server_url, model, messages_list[i], max_tokens=16)

    print(f"  Running {len(messages_list)} requests (concurrency={concurrency})...")

    if concurrency == 1:
        for idx, messages in enumerate(messages_list):
            result = send_chat_request(server_url, model, messages, max_tokens)
            result["workload"] = workload_name
            result["request_idx"] = idx
            results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for idx, messages in enumerate(messages_list):
                f = executor.submit(
                    send_chat_request, server_url, model, messages, max_tokens
                )
                futures[f] = idx
            for f in as_completed(futures):
                idx = futures[f]
                result = f.result()
                result["workload"] = workload_name
                result["request_idx"] = idx
                results.append(result)

    results.sort(key=lambda r: r["request_idx"])
    return results


def compute_summary(results: List[dict]) -> dict:
    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        return {"total": len(results), "success": 0, "error": len(results)}

    latencies = [r["latency"] for r in ok_results]
    ttfts = [r["ttft"] for r in ok_results]
    prompt_tokens = [r["prompt_tokens"] for r in ok_results]
    completion_tokens = [r["completion_tokens"] for r in ok_results]

    tps = [
        c / l if l > 0 else 0 for c, l in zip(completion_tokens, latencies)
    ]

    return {
        "total": len(results),
        "success": len(ok_results),
        "error": len(results) - len(ok_results),
        "latency_avg": sum(latencies) / len(latencies),
        "latency_p50": _percentile(latencies, 50),
        "latency_p95": _percentile(latencies, 95),
        "latency_p99": _percentile(latencies, 99),
        "ttft_avg": sum(ttfts) / len(ttfts),
        "ttft_p50": _percentile(ttfts, 50),
        "ttft_p95": _percentile(ttfts, 95),
        "ttft_p99": _percentile(ttfts, 99),
        "output_tps_avg": sum(tps) / len(tps),
        "output_tps_p50": _percentile(tps, 50),
        "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
        "avg_completion_tokens": sum(completion_tokens) / len(completion_tokens),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hybrid model prefix cache state-gating"
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:30000",
        help="SGLang server URL",
    )
    parser.add_argument(
        "--workload",
        choices=list(WORKLOADS.keys()) + ["all"],
        default="all",
        help="Workload to run",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max output tokens")
    parser.add_argument("--concurrency", type=int, default=1, help="Request concurrency")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests per workload")
    parser.add_argument("--num-requests", type=int, default=10, help="Requests per workload")
    parser.add_argument(
        "--prompt-words", type=int, default=500, help="Prompt length in words"
    )
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/{ts}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {args.server_url}...")
    try:
        model = get_model_name(args.server_url)
        print(f"Model: {model}")
    except Exception as e:
        print(f"Failed to connect: {e}", file=sys.stderr)
        sys.exit(1)

    workloads_to_run = list(WORKLOADS.keys()) if args.workload == "all" else [args.workload]

    all_results = {}
    all_summaries = {}

    for wl_name in workloads_to_run:
        print(f"\n{'='*60}")
        print(f"Workload: {wl_name}")
        print(f"{'='*60}")

        wl_func = WORKLOADS[wl_name]

        if wl_name == "identical_prompt":
            messages_list = wl_func(n=args.num_requests, prompt_words=args.prompt_words)
        elif wl_name == "random_no_share":
            messages_list = wl_func(n=args.num_requests, prompt_words=args.prompt_words)
        elif wl_name == "shared_system_prompt":
            messages_list = wl_func(n=args.num_requests, system_words=args.prompt_words)
        elif wl_name == "shared_document_qa":
            messages_list = wl_func(n=args.num_requests, doc_words=args.prompt_words)
        elif wl_name == "multi_turn_agent":
            messages_list = wl_func(n=args.num_requests)
        elif wl_name == "adversarial_near_prefix":
            messages_list = wl_func(n=args.num_requests, prefix_words=args.prompt_words)
        elif wl_name == "prefix_ladder":
            messages_list = wl_func()
        elif wl_name == "branch_after_cached_leaf":
            messages_list = wl_func(n_branches=args.num_requests, doc_words=args.prompt_words)
        else:
            messages_list = wl_func()

        results = run_workload(
            args.server_url,
            model,
            wl_name,
            messages_list,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            warmup=args.warmup,
        )

        summary = compute_summary(results)
        all_results[wl_name] = results
        all_summaries[wl_name] = summary

        print(f"  Results: {summary['success']}/{summary['total']} OK")
        if summary["success"] > 0:
            print(f"  Latency p50={summary['latency_p50']:.3f}s p95={summary['latency_p95']:.3f}s")
            print(f"  TTFT    p50={summary['ttft_p50']:.3f}s p95={summary['ttft_p95']:.3f}s")
            print(f"  Output  TPS avg={summary['output_tps_avg']:.1f}")
            print(f"  Avg prompt tokens: {summary['avg_prompt_tokens']:.0f}")

    with open(output_dir / "requests.jsonl", "w") as f:
        for wl_name, results in all_results.items():
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_dir}/")
    print(f"  requests.jsonl  - per-request results")
    print(f"  summary.json    - aggregate statistics")
    print(f"\nTo analyze trace:")
    print(f"  python benchmark/analyze_mamba_radix_trace.py $SGLANG_MAMBA_RADIX_TRACE_FILE")


if __name__ == "__main__":
    main()
