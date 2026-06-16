#!/usr/bin/env python3
"""
Analyze JSONL trace files produced by MambaRadixCache instrumentation.

Produces three summary views:
  - event_level: all trace events (may double-count across TP ranks)
  - rank0_only: only events from tp_rank=0
  - request_level: deduplicated by (key_hash, timestamp bucket)

Usage:
    python analyze_mamba_radix_trace.py /tmp/sglang_mamba_radix_trace.jsonl
    python analyze_mamba_radix_trace.py trace.jsonl --output results/trace_summary.csv
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def load_trace(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def compute_coverage_stats(records: List[dict]) -> dict:
    match_events = [r for r in records if r.get("event") == "match"]
    ranks = set()
    key_hashes = set()
    rank_key_pairs = set()
    for r in match_events:
        rank = r.get("tp_rank", 0)
        kh = r.get("key_hash", "")
        ranks.add(rank)
        if kh:
            key_hashes.add(kh)
            rank_key_pairs.add((rank, kh))
    return {
        "num_match_events": len(match_events),
        "num_unique_key_hashes": len(key_hashes),
        "num_unique_rank_key_pairs": len(rank_key_pairs),
        "num_ranks_observed": len(ranks),
        "ranks_list": sorted(ranks),
    }


def _analyze_match_list(match_events: List[dict], label: str) -> dict:
    if not match_events:
        return {f"{label}_match_count": 0}

    structural = [r["structural_match_len"] for r in match_events]
    state_cp = [r["state_checkpoint_match_len"] for r in match_events]
    effective = [r["effective_match_len"] for r in match_events]
    gated_loss = [r["gated_match_loss"] for r in match_events]
    input_lens = [r.get("input_len", 0) for r in match_events]

    has_structural = sum(1 for s in structural if s > 0)
    has_state_cp = sum(1 for s in state_cp if s > 0)
    gated = sum(1 for s, sc in zip(structural, state_cp) if s > sc)
    zero_eff = sum(1 for s, e in zip(structural, effective) if s > 0 and e == 0)
    splits = sum(1 for r in match_events if r.get("split_happened"))
    tombstones = [r.get("num_tombstone_nodes_on_path", 0) for r in match_events]

    n = len(match_events)
    pfx = f"{label}_" if label else ""
    return {
        f"{pfx}match_count": n,
        f"{pfx}input_len_avg": sum(input_lens) / n,
        f"{pfx}structural_match_len_avg": sum(structural) / n,
        f"{pfx}structural_match_len_p50": percentile(structural, 50),
        f"{pfx}structural_match_len_p95": percentile(structural, 95),
        f"{pfx}state_checkpoint_match_len_avg": sum(state_cp) / n,
        f"{pfx}state_checkpoint_match_len_p50": percentile(state_cp, 50),
        f"{pfx}state_checkpoint_match_len_p95": percentile(state_cp, 95),
        f"{pfx}effective_match_len_avg": sum(effective) / n,
        f"{pfx}effective_match_len_p50": percentile(effective, 50),
        f"{pfx}effective_match_len_p95": percentile(effective, 95),
        f"{pfx}gated_match_loss_avg": sum(gated_loss) / n,
        f"{pfx}gated_match_loss_p50": percentile(gated_loss, 50),
        f"{pfx}gated_match_loss_p95": percentile(gated_loss, 95),
        f"{pfx}frac_structural_hit": has_structural / n,
        f"{pfx}frac_state_cp_hit": has_state_cp / n,
        f"{pfx}frac_structural_gt_state_cp": gated / n,
        f"{pfx}frac_zero_eff_with_structural": zero_eff / n,
        f"{pfx}split_happened_count": splits,
        f"{pfx}tombstone_on_path_avg": sum(tombstones) / n,
        f"{pfx}tombstone_on_path_p95": percentile(tombstones, 95),
    }


def dedup_by_request(match_events: List[dict]) -> List[dict]:
    """Keep one event per (key_hash, timestamp_bucket) — approximate request dedup."""
    seen = set()
    result = []
    for r in match_events:
        kh = r.get("key_hash", "")
        ts_bucket = round(r.get("timestamp", 0), 1)
        dedup_key = (kh, ts_bucket)
        if dedup_key not in seen:
            seen.add(dedup_key)
            result.append(r)
    return result


def analyze_all_views(records: List[dict]) -> dict:
    match_all = [r for r in records if r.get("event") == "match"]
    match_rank0 = [r for r in match_all if r.get("tp_rank", 0) == 0]
    match_dedup = dedup_by_request(match_all)

    results = {}
    results.update(_analyze_match_list(match_all, "event"))
    results.update(_analyze_match_list(match_rank0, "rank0"))
    results.update(_analyze_match_list(match_dedup, "request"))
    return results


def analyze_split_events(records: List[dict]) -> dict:
    split_events = [r for r in records if r.get("event") == "split"]
    if not split_events:
        return {"split_count": 0}
    tombstone_created = sum(
        1 for r in split_events if r.get("split_created_mamba_tombstone")
    )
    return {
        "split_count": len(split_events),
        "split_tombstone_created_count": tombstone_created,
        "split_tombstone_frac": tombstone_created / len(split_events),
    }


def analyze_insert_events(records: List[dict]) -> dict:
    insert_events = [r for r in records if r.get("event") == "insert"]
    if not insert_events:
        return {"insert_count": 0}
    n = len(insert_events)
    return {
        "insert_count": n,
        "mamba_checkpoint_inserted_count": sum(
            1 for r in insert_events if r.get("mamba_checkpoint_inserted")
        ),
        "mamba_tombstone_restored_count": sum(
            1 for r in insert_events if r.get("mamba_checkpoint_restored_tombstone")
        ),
        "mamba_already_existed_count": sum(
            1 for r in insert_events if r.get("mamba_value_already_existed")
        ),
    }


def analyze_eviction_events(records: List[dict]) -> dict:
    mamba_evicts = [r for r in records if r.get("event") == "evict_mamba"]
    kv_evicts = [r for r in records if r.get("event") == "evict_kv"]
    return {
        "mamba_eviction_events": len(mamba_evicts),
        "mamba_eviction_internal_tombstone": sum(
            1 for r in mamba_evicts if r.get("was_internal_tombstone")
        ),
        "mamba_total_slots_evicted": sum(
            r.get("mamba_slots_evicted", 0) for r in mamba_evicts
        ),
        "kv_eviction_events": len(kv_evicts),
        "kv_total_tokens_evicted": sum(
            r.get("kv_tokens_evicted", 0) for r in kv_evicts
        ),
    }


def analyze_cache_pressure(records: List[dict]) -> dict:
    match_events = [r for r in records if r.get("event") == "match"]
    if not match_events:
        return {}
    full_ev = [r.get("full_evictable_size", 0) for r in match_events]
    mamba_ev = [r.get("mamba_evictable_size", 0) for r in match_events]
    return {
        "full_evictable_size_avg": sum(full_ev) / len(full_ev),
        "mamba_evictable_size_avg": sum(mamba_ev) / len(mamba_ev),
        "full_evictable_size_min": min(full_ev),
        "mamba_evictable_size_min": min(mamba_ev),
    }


def format_report(results: dict, coverage: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("MambaRadixCache Trace Analysis Report")
    lines.append("=" * 70)

    lines.append("\n--- Coverage ---")
    lines.append(f"  num_match_events: {coverage['num_match_events']}")
    lines.append(f"  num_unique_key_hashes: {coverage['num_unique_key_hashes']}")
    lines.append(f"  num_unique_rank_key_pairs: {coverage['num_unique_rank_key_pairs']}")
    lines.append(f"  num_ranks_observed: {coverage['num_ranks_observed']}")
    lines.append(f"  ranks: {coverage['ranks_list']}")

    for view in ["event", "rank0", "request"]:
        lines.append(f"\n--- Match Events ({view}_level) ---")
        pfx = f"{view}_"
        keys = [
            "match_count", "input_len_avg",
            "structural_match_len_avg", "structural_match_len_p50", "structural_match_len_p95",
            "state_checkpoint_match_len_avg", "state_checkpoint_match_len_p50", "state_checkpoint_match_len_p95",
            "effective_match_len_avg", "effective_match_len_p50", "effective_match_len_p95",
            "gated_match_loss_avg", "gated_match_loss_p50", "gated_match_loss_p95",
            "frac_structural_hit", "frac_state_cp_hit",
            "frac_structural_gt_state_cp", "frac_zero_eff_with_structural",
            "split_happened_count", "tombstone_on_path_avg", "tombstone_on_path_p95",
        ]
        for key in keys:
            full_key = f"{pfx}{key}"
            if full_key in results:
                val = results[full_key]
                if isinstance(val, float):
                    lines.append(f"  {key}: {val:.4f}")
                else:
                    lines.append(f"  {key}: {val}")

    other_sections = {
        "Split Events": [
            "split_count", "split_tombstone_created_count", "split_tombstone_frac",
        ],
        "Insert Events": [
            "insert_count", "mamba_checkpoint_inserted_count",
            "mamba_tombstone_restored_count", "mamba_already_existed_count",
        ],
        "Eviction Events": [
            "mamba_eviction_events", "mamba_eviction_internal_tombstone",
            "mamba_total_slots_evicted", "kv_eviction_events", "kv_total_tokens_evicted",
        ],
        "Cache Pressure": [
            "full_evictable_size_avg", "mamba_evictable_size_avg",
            "full_evictable_size_min", "mamba_evictable_size_min",
        ],
    }
    for section_name, keys in other_sections.items():
        lines.append(f"\n--- {section_name} ---")
        for key in keys:
            if key in results:
                val = results[key]
                if isinstance(val, float):
                    lines.append(f"  {key}: {val:.4f}")
                else:
                    lines.append(f"  {key}: {val}")

    return "\n".join(lines)


def write_csv(results: dict, path: str) -> None:
    with open(path, "w") as f:
        f.write("metric,value\n")
        for k, v in sorted(results.items()):
            if isinstance(v, list):
                v = str(v)
            f.write(f"{k},{v}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MambaRadixCache JSONL trace"
    )
    parser.add_argument("trace_file", help="Path to JSONL trace file")
    parser.add_argument("--output", "-o", help="Output CSV path", default=None)
    parser.add_argument("--json-output", help="Output JSON summary path", default=None)
    args = parser.parse_args()

    records = load_trace(args.trace_file)
    if not records:
        print(f"No records found in {args.trace_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} trace records from {args.trace_file}")

    event_counts = Counter(r.get("event") for r in records)
    print(f"Event breakdown: {dict(event_counts)}")

    coverage = compute_coverage_stats(records)

    results = {}
    results.update(coverage)
    results.update(analyze_all_views(records))
    results.update(analyze_split_events(records))
    results.update(analyze_insert_events(records))
    results.update(analyze_eviction_events(records))
    results.update(analyze_cache_pressure(records))

    print(format_report(results, coverage))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_csv(results, args.output)
        print(f"\nCSV written to {args.output}")

    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"JSON written to {args.json_output}")


if __name__ == "__main__":
    main()
