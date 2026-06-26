#!/usr/bin/env python3
"""
Prefix Family Analyzer — Quantify avoidable recurrent state transitions.

Analyzes production traces (Kimi K25, Qwen-Bailian, Azure) to measure:
1. Prefix family structure (shared prefix groups)
2. State-gating amplification (recurrent state blocks KV cache hits)
3. Avoidable transitions (recompute + transfer that DualState eliminates)
4. Exact vs boundary vs sparse checkpoint opportunity

This is a pure offline analysis — no GPU or server needed.

Usage:
  python prefix_family_analyzer.py --trace kimi --output results/
  python prefix_family_analyzer.py --trace azure --output results/
  python prefix_family_analyzer.py --trace qwen_b --output results/
"""

import argparse
import csv
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# ─── Model Parameters (Kimi-Linear-48B-A3B) ───────────────────────────────
MODEL_PARAMS = {
    "name": "Kimi-Linear-48B-A3B",
    "mamba_state_mb": 30.8,
    "mamba_recompute_ms": 40,
    "mamba_transfer_ms_mooncake": 21,  # at 1.5 GB/s
    "mamba_transfer_ms_ib": 1.2,      # at 25 GB/s IB HDR
    "kv_per_token_bytes": 20480,      # bf16, all full-attn layers
    "cow_ms": 0.03,
    "prefill_ms_per_token": 0.02,     # calibrated from experiments
}


# ─── Trace Loaders ────────────────────────────────────────────────────────

def load_kimi_trace(path, max_requests=100000):
    """Load Kimi K25 CSV trace."""
    reqs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_requests:
                break
            reqs.append({
                "idx": i,
                "timestamp": None,  # parsed below
                "ctx_tokens": int(row["ContextTokens"]),
                "gen_tokens": int(row["GeneratedTokens"]),
                "trace": "kimi",
            })
    return reqs


def load_kimi_detailed(log_dir, max_requests=50000):
    """Load Kimi K25 detailed redacted logs with tool info."""
    reqs = []
    for log_file in sorted(os.listdir(log_dir)):
        if not log_file.endswith(".redacted.log"):
            continue
        with open(os.path.join(log_dir, log_file)) as f:
            for line in f:
                if not line.startswith("Finish:"):
                    continue
                m_input = re.search(r"input_ids_len=(\d+)", line)
                m_rid = re.search(r"rid='([^']+)'", line)
                m_tools = re.findall(r"functions\.(\w+):", line)
                if m_input:
                    reqs.append({
                        "idx": len(reqs),
                        "ctx_tokens": int(m_input.group(1)),
                        "rid": m_rid.group(1) if m_rid else None,
                        "tools": tuple(sorted(set(m_tools))) if m_tools else (),
                        "n_tools": len(set(m_tools)) if m_tools else 0,
                        "trace": "kimi_detailed",
                    })
                if len(reqs) >= max_requests:
                    break
        if len(reqs) >= max_requests:
            break
    return reqs


def load_azure_trace(path, max_requests=100000):
    """Load Azure LLM Inference trace."""
    reqs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_requests:
                break
            reqs.append({
                "idx": i,
                "ctx_tokens": int(row["ContextTokens"]),
                "gen_tokens": int(row["GeneratedTokens"]),
                "trace": "azure",
            })
    return reqs


def load_qwen_trace(path, max_requests=100000):
    """Load Qwen-Bailian JSONL trace."""
    reqs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_requests:
                break
            try:
                d = json.loads(line)
                reqs.append({
                    "idx": i,
                    "ctx_tokens": d.get("input_length", 0),
                    "gen_tokens": d.get("output_length", 0),
                    "chat_id": d.get("chat_id"),
                    "parent_chat_id": d.get("parent_chat_id"),
                    "turn": d.get("turn", 1),
                    "hash_ids": d.get("hash_ids", []),
                    "req_type": d.get("type", "text"),
                    "trace": "qwen",
                })
            except json.JSONDecodeError:
                continue
    return reqs


# ─── Prefix Family Analysis ───────────────────────────────────────────────

def analyze_prefix_families(reqs, trace_type="generic"):
    """
    Group requests into prefix families based on available info.

    For Kimi detailed: group by tool combination (shared system prompt)
    For Qwen: group by chat_id (conversation) and hash_ids (content)
    For generic: group by context length bucket (approximation)
    """
    families = defaultdict(list)

    if trace_type == "kimi_detailed":
        for r in reqs:
            key = r.get("tools", ())
            families[key].append(r)
    elif trace_type == "qwen":
        # Group by shared prefix via hash_ids
        for r in reqs:
            hids = tuple(r.get("hash_ids", [])[:4])  # first 4 blocks = ~64 tokens
            if hids:
                families[hids].append(r)
            else:
                families[("singleton", r["idx"])].append(r)
    else:
        # Generic: bucket by context length (1k granularity)
        for r in reqs:
            bucket = r["ctx_tokens"] // 1024
            families[bucket].append(r)

    return dict(families)


def compute_avoidable_transitions(reqs, families, params=MODEL_PARAMS):
    """
    Compute avoidable recurrent state transitions.

    For each prefix family:
    - First request: must pay full recurrent cost (compute + transfer)
    - Subsequent requests: could COW from checkpoint (0.03ms)
    - Without DualState: ALL pay full cost
    - Avoidable = (family_size - 1) × per_request_recurrent_cost
    """
    n = len(reqs)
    mamba_recompute = params["mamba_recompute_ms"]
    mamba_transfer = params["mamba_transfer_ms_mooncake"]
    cow = params["cow_ms"]

    total_baseline_ms = n * (mamba_recompute + mamba_transfer)

    # With DualState: first of each family pays, rest COW
    n_families = len(families)
    n_reusable = n - n_families

    total_dualstate_ms = (
        n_families * (mamba_recompute + mamba_transfer)  # first of each family
        + n_reusable * cow                                 # rest COW
    )

    avoidable_ms = total_baseline_ms - total_dualstate_ms

    return {
        "n_requests": n,
        "n_families": n_families,
        "n_reusable": n_reusable,
        "reuse_ratio": n_reusable / n if n > 0 else 0,
        "total_baseline_recurrent_ms": total_baseline_ms,
        "total_dualstate_recurrent_ms": total_dualstate_ms,
        "avoidable_recurrent_ms": avoidable_ms,
        "avoidable_recompute_ms": n_reusable * mamba_recompute,
        "avoidable_transfer_mb": n_reusable * params["mamba_state_mb"],
        "family_size_distribution": {
            "min": min(len(v) for v in families.values()),
            "max": max(len(v) for v in families.values()),
            "avg": statistics.mean(len(v) for v in families.values()),
            "p50": sorted(len(v) for v in families.values())[len(families) // 2],
        },
    }


def compute_stategating_amplification(reqs, params=MODEL_PARAMS):
    """
    Compute the state-gating amplification effect.

    When recurrent state is missing, the ENTIRE prefix cache hit is wasted:
    - KV cache structural match → effective match = 0
    - P must re-prefill entire prefix
    - Wasted = KV_transfer + prefill_compute (not just mamba_transfer)
    """
    results = []
    for prefix_len in [512, 1024, 2048, 4096, 8192, 16384]:
        kv_bytes = prefix_len * params["kv_per_token_bytes"]
        kv_transfer_ms = kv_bytes / (25e9) * 1000  # IB HDR
        prefill_ms = prefix_len * params["prefill_ms_per_token"]
        mamba_mb = params["mamba_state_mb"]

        # Total waste when recurrent state is missing
        total_waste_ms = prefill_ms + kv_transfer_ms + params["mamba_transfer_ms_ib"]

        # Amplification: waste / mamba_size
        amplification = total_waste_ms / (params["mamba_transfer_ms_ib"] + params["cow_ms"])

        results.append({
            "prefix_len": prefix_len,
            "kv_size_mb": kv_bytes / 1e6,
            "mamba_size_mb": mamba_mb,
            "mamba_fraction": mamba_mb / (kv_bytes / 1e6 + mamba_mb),
            "prefill_waste_ms": prefill_ms,
            "kv_transfer_waste_ms": kv_transfer_ms,
            "total_waste_ms": total_waste_ms,
            "amplification_factor": amplification,
        })

    return results


def compute_checkpoint_opportunity(reqs, families, trace_type="generic"):
    """
    Decompose reuse opportunity into:
    - Exact: full prefix match (current DualState)
    - Boundary: system prompt / tool schema / turn boundary
    - Sparse: any partial prefix overlap
    """
    n = len(reqs)

    # Exact: current DualState covers this
    exact_reusable = sum(len(v) - 1 for v in families.values() if len(v) > 1)

    # Boundary: for multi-turn, check if turns share earlier boundaries
    boundary_additional = 0
    if trace_type == "qwen":
        # Qwen has chat_id and turn — multi-turn same chat shares prefix
        by_chat = defaultdict(list)
        for r in reqs:
            cid = r.get("chat_id")
            if cid is not None and cid >= 0:
                by_chat[cid].append(r)
        for chat_reqs in by_chat.values():
            if len(chat_reqs) > 1:
                # Turn 2+ can reuse turn 1's checkpoint at system prompt boundary
                boundary_additional += len(chat_reqs) - 1
    elif trace_type == "kimi_detailed":
        # Kimi: requests with same tool set but different context lengths
        # could reuse checkpoint at tool schema boundary
        by_tools = defaultdict(list)
        for r in reqs:
            by_tools[r.get("tools", ())].append(r)
        for tool_reqs in by_tools.values():
            if len(tool_reqs) > 1:
                # Different context lengths but same tools = boundary reuse
                unique_lens = len(set(r["ctx_tokens"] for r in tool_reqs))
                if unique_lens > 1:
                    boundary_additional += len(tool_reqs) - unique_lens

    return {
        "exact_reusable": exact_reusable,
        "exact_coverage": exact_reusable / n if n > 0 else 0,
        "boundary_additional": boundary_additional,
        "boundary_coverage": (exact_reusable + boundary_additional) / n if n > 0 else 0,
        "total_opportunity": exact_reusable + boundary_additional,
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prefix Family Analyzer")
    parser.add_argument("--trace", choices=["kimi", "kimi_detailed", "azure", "qwen_b"], required=True)
    parser.add_argument("--max-requests", type=int, default=100000)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trace
    print(f"Loading {args.trace} trace...")
    if args.trace == "kimi":
        reqs = load_kimi_trace(
            "/mnt/models/kimik25/kimi-k25-trace/kimi_k25_conv_1day.csv",
            args.max_requests,
        )
        trace_type = "generic"
    elif args.trace == "kimi_detailed":
        reqs = load_kimi_detailed(
            "/mnt/models/kimik25/kimi-k25-workers-5p3d-all-decodes_finish_fullids_redacted_2026-04-07",
            args.max_requests,
        )
        trace_type = "kimi_detailed"
    elif args.trace == "azure":
        reqs = load_azure_trace(
            "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv",
            args.max_requests,
        )
        trace_type = "generic"
    elif args.trace == "qwen_b":
        reqs = load_qwen_trace(
            "/tmp/qwen_traces/qwen_traceB_blksz_16.jsonl",
            args.max_requests,
        )
        trace_type = "qwen"

    print(f"  Loaded {len(reqs)} requests")

    # Context distribution
    ctxs = sorted(r["ctx_tokens"] for r in reqs)
    n = len(ctxs)
    print(f"\n  Context distribution:")
    print(f"    min={ctxs[0]}, p50={ctxs[n//2]}, p90={ctxs[int(n*0.9)]}, max={ctxs[-1]}")

    # Prefix families
    print(f"\nAnalyzing prefix families...")
    families = analyze_prefix_families(reqs, trace_type)

    # Avoidable transitions
    transitions = compute_avoidable_transitions(reqs, families)
    print(f"\n{'='*70}")
    print(f"  AVOIDABLE RECURRENT STATE TRANSITIONS ({args.trace})")
    print(f"{'='*70}")
    print(f"  Requests: {transitions['n_requests']}")
    print(f"  Prefix families: {transitions['n_families']}")
    print(f"  Reusable requests: {transitions['n_reusable']} ({transitions['reuse_ratio']*100:.1f}%)")
    print(f"  Family size: avg={transitions['family_size_distribution']['avg']:.0f}, max={transitions['family_size_distribution']['max']}")
    print(f"  Avoidable recompute: {transitions['avoidable_recompute_ms']/1000:.0f}s ({transitions['avoidable_recompute_ms']/transitions['total_baseline_recurrent_ms']*100:.1f}%)")
    print(f"  Avoidable transfer: {transitions['avoidable_transfer_mb']/1024:.1f} GB")
    print(f"  Total avoidable: {transitions['avoidable_recurrent_ms']/1000:.0f}s / {transitions['total_baseline_recurrent_ms']/1000:.0f}s ({transitions['avoidable_recurrent_ms']/transitions['total_baseline_recurrent_ms']*100:.1f}%)")

    # State-gating amplification
    amplification = compute_stategating_amplification(reqs)
    print(f"\n{'='*70}")
    print(f"  STATE-GATING AMPLIFICATION EFFECT")
    print(f"{'='*70}")
    print(f"  {'Prefix':>8} {'KV(MB)':>8} {'Mamba%':>8} {'Prefill waste':>14} {'Amplification':>14}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*14} {'-'*14}")
    for a in amplification:
        print(f"  {a['prefix_len']:>8} {a['kv_size_mb']:>6.0f}MB {a['mamba_fraction']*100:>6.0f}% {a['prefill_waste_ms']:>12.1f}ms {a['amplification_factor']:>12.1f}x")

    # Checkpoint opportunity
    opportunity = compute_checkpoint_opportunity(reqs, families, trace_type)
    print(f"\n{'='*70}")
    print(f"  CHECKPOINT OPPORTUNITY DECOMPOSITION")
    print(f"{'='*70}")
    print(f"  Exact (current DualState): {opportunity['exact_reusable']} ({opportunity['exact_coverage']*100:.1f}%)")
    print(f"  Boundary (turn/tool): +{opportunity['boundary_additional']} ({opportunity['boundary_coverage']*100:.1f}% total)")

    # Save results
    results = {
        "trace": args.trace,
        "n_requests": len(reqs),
        "context_distribution": {
            "min": ctxs[0], "p50": ctxs[n//2], "p90": ctxs[int(n*0.9)], "max": ctxs[-1],
        },
        "transitions": transitions,
        "amplification": amplification,
        "opportunity": opportunity,
    }

    out_path = output_dir / f"prefix_family_{args.trace}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
