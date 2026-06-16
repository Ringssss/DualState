#!/bin/bash
# Phase 3-mini: Hybrid prefix cache state-gating parameter sweep
# 5 configs × 4 workloads on Qwen3.6-35B-A3B (TP=2)
#
# Usage:
#   bash scripts/run_hybrid_prefix_sweep.sh
#   MODEL_PATH=/mnt/models/Kimi-Linear-48B-A3B-Instruct bash scripts/run_hybrid_prefix_sweep.sh

set -euo pipefail

# Clear ALL proxy settings to prevent localhost connections from going through proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY all_proxy ALL_PROXY SOCKS_PROXY socks_proxy 2>/dev/null || true
export NO_PROXY="localhost,127.0.0.1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/zhujianian/miniconda3/envs/sglang-bench/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/models/Qwen3.6-35B-A3B}"
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
PORT="${PORT:-30100}"
NUM_REQUESTS="${NUM_REQUESTS:-10}"
PROMPT_WORDS="${PROMPT_WORDS:-500}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/results/sweep_$(date +%Y%m%d_%H%M%S)}"

WORKLOADS="random_no_share identical_prompt branch_after_cached_leaf prefix_ladder"

declare -A CONFIG_NAMES
declare -A CONFIG_ARGS
CONFIG_NAMES[A]="default_no_buffer"
CONFIG_ARGS[A]=""
CONFIG_NAMES[B]="extra_buffer"
CONFIG_ARGS[B]="--mamba-scheduler-strategy extra_buffer"
CONFIG_NAMES[C]="ratio_0.5"
CONFIG_ARGS[C]="--mamba-full-memory-ratio 0.5"
CONFIG_NAMES[D]="ratio_0.9"
CONFIG_ARGS[D]="--mamba-full-memory-ratio 0.9"
CONFIG_NAMES[E]="disable_cache"
CONFIG_ARGS[E]="--disable-radix-cache"

CONFIG_ORDER="A B C D E"

kill_server() {
    pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
    sleep 3
    # force kill any remaining
    pkill -9 -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
    sleep 2
}

wait_server_ready() {
    local url="http://localhost:${PORT}/v1/models"
    local max_wait=600
    local waited=0
    echo "  Waiting for server on port ${PORT}..."
    while ! curl -s --noproxy localhost --max-time 5 "$url" 2>/dev/null | grep -q "id"; do
        sleep 10
        waited=$((waited + 10))
        if [ $waited -ge $max_wait ]; then
            echo "  ERROR: Server did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "  Server ready after ~${waited}s"
    # warmup: first request triggers JIT compile
    curl -s --noproxy localhost --max-time 180 "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL_PATH}\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":8}" \
        > /dev/null 2>&1 || true
    echo "  JIT warmup done"
}

mkdir -p "$RESULTS_ROOT"
echo "=========================================="
echo "Phase 3-mini Sweep"
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${GPUS} (TP=${TP})"
echo "Workloads: ${WORKLOADS}"
echo "Results: ${RESULTS_ROOT}"
echo "=========================================="

for cfg_key in $CONFIG_ORDER; do
    cfg_name="${CONFIG_NAMES[$cfg_key]}"
    cfg_args="${CONFIG_ARGS[$cfg_key]}"
    cfg_dir="${RESULTS_ROOT}/${cfg_key}_${cfg_name}"
    mkdir -p "$cfg_dir"

    echo ""
    echo "=========================================="
    echo "Config ${cfg_key}: ${cfg_name}"
    echo "  Extra args: ${cfg_args:-'(none)'}"
    echo "=========================================="

    # Kill any previous server
    kill_server

    # Set trace file for this config
    TRACE_FILE="${cfg_dir}/mamba_radix_trace.jsonl"
    rm -f "$TRACE_FILE"

    # Capture server log
    SERVER_LOG="${cfg_dir}/server.log"

    # Start server
    echo "  Starting server..."
    source "$HOME/.cargo/env" 2>/dev/null || true
    export PATH="/home/zhujianian/miniconda3/envs/sglang-bench/bin:$PATH"

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY all_proxy ALL_PROXY SOCKS_PROXY socks_proxy 2>/dev/null || true

    SGLANG_MAMBA_RADIX_TRACE=1 \
    SGLANG_MAMBA_RADIX_TRACE_FILE="$TRACE_FILE" \
    CUDA_VISIBLE_DEVICES="$GPUS" \
    nohup "$PYTHON" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --tp "$TP" \
        $cfg_args \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "  Server PID: $SERVER_PID"

    if ! wait_server_ready; then
        echo "  SKIPPING config ${cfg_name} - server failed to start"
        kill_server
        echo "FAILED" > "${cfg_dir}/status.txt"
        continue
    fi

    # Capture GPU memory snapshot
    nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free \
        --format=csv,noheader > "${cfg_dir}/gpu_memory.csv" 2>/dev/null || true

    # Run workloads
    for wl in $WORKLOADS; do
        echo "  Running workload: ${wl}..."
        wl_dir="${cfg_dir}/${wl}"
        mkdir -p "$wl_dir"

        # Clear trace for this workload
        > "$TRACE_FILE"

        "$PYTHON" "${SCRIPT_DIR}/benchmark/bench_hybrid_prefix_state_gate.py" \
            --server-url "http://localhost:${PORT}" \
            --workload "$wl" \
            --num-requests "$NUM_REQUESTS" \
            --prompt-words "$PROMPT_WORDS" \
            --warmup 1 \
            --output-dir "$wl_dir" \
            2>&1 | tee "${wl_dir}/benchmark.log"

        # Copy and analyze trace
        if [ -s "$TRACE_FILE" ]; then
            cp "$TRACE_FILE" "${wl_dir}/trace.jsonl"
            "$PYTHON" "${SCRIPT_DIR}/benchmark/analyze_mamba_radix_trace.py" \
                "${wl_dir}/trace.jsonl" \
                --output "${wl_dir}/trace_summary.csv" \
                --json-output "${wl_dir}/trace_summary.json" \
                2>&1 | tee "${wl_dir}/trace_analysis.log"
        else
            echo "  WARNING: No trace data for ${wl}"
        fi
    done

    echo "OK" > "${cfg_dir}/status.txt"
    kill_server
    echo "  Config ${cfg_name} done."
done

echo ""
echo "=========================================="
echo "Sweep complete. Generating summary..."
echo "=========================================="

# Generate combined summary table
"$PYTHON" -c "
import json, os, sys
from pathlib import Path

root = Path('${RESULTS_ROOT}')
rows = []
header = ['config', 'workload',
          'rank0_gated_match_loss_avg', 'rank0_frac_structural_gt_state_cp',
          'rank0_frac_zero_eff_with_structural', 'split_tombstone_frac',
          'rank0_tombstone_on_path_avg',
          'rank0_structural_match_len_avg', 'rank0_state_checkpoint_match_len_avg',
          'rank0_effective_match_len_avg',
          'mamba_eviction_events', 'kv_eviction_events',
          'latency_p50', 'latency_p95', 'ttft_p50', 'ttft_p95']

for cfg_dir in sorted(root.iterdir()):
    if not cfg_dir.is_dir():
        continue
    cfg_name = cfg_dir.name
    for wl_dir in sorted(cfg_dir.iterdir()):
        if not wl_dir.is_dir():
            continue
        wl_name = wl_dir.name
        trace_json = wl_dir / 'trace_summary.json'
        bench_json = wl_dir / 'summary.json'
        row = {'config': cfg_name, 'workload': wl_name}
        if trace_json.exists():
            t = json.load(open(trace_json))
            for k in header[2:11]:
                row[k] = t.get(k, '')
        if bench_json.exists():
            b = json.load(open(bench_json))
            wl_summary = b.get(wl_name, {})
            row['latency_p50'] = wl_summary.get('latency_p50', '')
            row['latency_p95'] = wl_summary.get('latency_p95', '')
            row['ttft_p50'] = wl_summary.get('ttft_p50', '')
            row['ttft_p95'] = wl_summary.get('ttft_p95', '')
        rows.append(row)

# Write CSV
csv_path = root / 'sweep_summary.csv'
with open(csv_path, 'w') as f:
    f.write(','.join(header) + '\n')
    for r in rows:
        f.write(','.join(str(r.get(h, '')) for h in header) + '\n')
print(f'Summary written to {csv_path}')

# Print table
print()
fmt = '{:<25} {:<25} {:>12} {:>12} {:>12} {:>12} {:>12}'
print(fmt.format('config', 'workload', 'gated_loss', 'frac_gt', 'frac_zero', 'tomb_frac', 'tomb_path'))
print('-' * 115)
for r in rows:
    def fv(k):
        v = r.get(k, '')
        if isinstance(v, float): return f'{v:.2f}'
        return str(v)[:12]
    print(fmt.format(
        r['config'][:25], r['workload'][:25],
        fv('rank0_gated_match_loss_avg'), fv('rank0_frac_structural_gt_state_cp'),
        fv('rank0_frac_zero_eff_with_structural'), fv('split_tombstone_frac'),
        fv('rank0_tombstone_on_path_avg')))
" 2>&1 | tee "${RESULTS_ROOT}/sweep_summary.log"

echo ""
echo "All results in: ${RESULTS_ROOT}/"
echo "Summary CSV: ${RESULTS_ROOT}/sweep_summary.csv"
