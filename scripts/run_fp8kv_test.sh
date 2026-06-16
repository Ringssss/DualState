#!/bin/bash
# FP8 KV Cache Transfer Test
# Compare: baseline bf16 vs fp8_e4m3 KV cache on P/D disagg
# Both with and without DualState
set -euo pipefail

PYTHON="/home/zhujianian/miniconda3/envs/sglang-bench/bin/python"
MODEL_QWEN="/mnt/models/Qwen3.6-35B-A3B"
BENCH="/home/zhujianian/sglang/codex_coding/src/dualstate/bench_loogle_sweetspot.py"
HOST="127.0.0.1"
P_PORT=30100; D_PORT=30200; LB_PORT=30000; BOOTSTRAP_PORT=30500
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="/home/zhujianian/sglang/codex_coding/results/dualstate/fp8kv_${TIMESTAMP}"

export MC_TCP_ENABLE_CONNECTION_POOL=true
export PATH="/home/zhujianian/miniconda3/envs/sglang-bench/bin:$PATH"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export NO_PROXY="localhost,127.0.0.1"

mkdir -p "$RESULTS/logs"

cleanup() {
    pkill -f "sglang" 2>/dev/null || true; sleep 3
    pkill -9 -f "sglang" 2>/dev/null || true; sleep 2
    fuser -k $P_PORT/tcp $D_PORT/tcp $LB_PORT/tcp $BOOTSTRAP_PORT/tcp 2>/dev/null || true; sleep 1
}

wait_health() {
    local url=$1 label=$2 max_wait=${3:-600} waited=0
    while true; do
        curl -s --noproxy localhost --max-time 3 "${url}/v1/models" 2>/dev/null | grep -q "id" && break
        sleep 10; waited=$((waited + 10))
        [ $waited -ge $max_wait ] && echo "TIMEOUT: $label" && return 1
    done
    echo "  $label ready (${waited}s)"
}

launch_pd() {
    local extra_args="${1:-}" log_prefix="$2"
    echo "[launch] P+D: $log_prefix ($extra_args)"

    SGLANG_DISAGG_TRANSFER_TIMING=1 \
    $PYTHON -m sglang.launch_server \
        --model-path "$MODEL_QWEN" --host $HOST --port $P_PORT \
        --tp 2 --base-gpu-id 0 --trust-remote-code \
        --disaggregation-mode prefill \
        --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
        --disaggregation-transfer-backend mooncake_tcp \
        $extra_args \
        > "$RESULTS/logs/${log_prefix}_prefill.log" 2>&1 &

    SGLANG_DISAGG_TRANSFER_TIMING=1 \
    $PYTHON -m sglang.launch_server \
        --model-path "$MODEL_QWEN" --host $HOST --port $D_PORT \
        --tp 2 --base-gpu-id 2 --trust-remote-code \
        --disaggregation-mode decode \
        --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
        --disaggregation-transfer-backend mooncake_tcp \
        $extra_args \
        > "$RESULTS/logs/${log_prefix}_decode.log" 2>&1 &

    wait_health "http://$HOST:$P_PORT" "Prefill" || return 1
    wait_health "http://$HOST:$D_PORT" "Decode" || return 1

    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://$HOST:$P_PORT" --decode "http://$HOST:$D_PORT" \
        --host $HOST --port $LB_PORT \
        > "$RESULTS/logs/${log_prefix}_lb.log" 2>&1 &
    sleep 5

    # Warmup with retries
    for attempt in 1 2 3; do
        curl -s --noproxy localhost --max-time 180 \
            "http://$HOST:$LB_PORT/generate" \
            -H "Content-Type: application/json" \
            -d '{"text":"Hello","sampling_params":{"max_new_tokens":4,"temperature":0}}' \
            2>/dev/null | grep -q "text" && break
        sleep 10
    done
    echo "  Ready."
}

run_bench() {
    local tag="$1" output="$2"
    echo "[bench] $tag"
    $PYTHON "$BENCH" sweep \
        --base-url "http://$HOST:$LB_PORT" \
        --output "$output" --tag "$tag" \
        --model-name "Qwen3.6-35B-A3B" \
        --prefix-lens "2048,4096" --fan-outs "8,16" --delays "0,200" \
        --concurrency 16 --num-docs 2 --max-tokens 16 --warmup 3
}

extract_timing() {
    local log="$1" label="$2"
    echo "  [$label] Transfer timing:"
    grep "TRANSFER_TIMING" "$log" 2>/dev/null | \
        awk '{print $NF}' | sort | head -5
    # Summary
    grep "TRANSFER_TIMING" "$log" 2>/dev/null | \
        grep -oP 'total=\K[0-9.]+' | \
        awk '{s+=$1; n++} END {if(n>0) printf "    avg=%.1fMB over %d transfers\n", s/n, n}'
    grep "TRANSFER_TIMING" "$log" 2>/dev/null | \
        grep -oP 'elapsed=\K[0-9.]+' | \
        awk '{s+=$1; n++} END {if(n>0) printf "    avg_time=%.1fms\n", s/n}'
    grep "TRANSFER_TIMING" "$log" 2>/dev/null | \
        grep -oP 'bw=\K[0-9.]+' | \
        awk '{s+=$1; n++} END {if(n>0) printf "    avg_bw=%.1fGB/s\n", s/n}'
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FP8 KV Cache Transfer Experiment (Qwen3.6-35B-A3B)       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ═══ Config 1: Baseline (bf16 KV, no DualState) ═══
echo ""; echo "═══ Config 1: baseline_bf16 ═══"
cleanup
if launch_pd "" "baseline_bf16"; then
    run_bench "baseline_bf16" "$RESULTS/baseline_bf16.json"
    extract_timing "$RESULTS/logs/baseline_bf16_prefill.log" "P-side"
fi
cleanup

# ═══ Config 2: DualState (bf16 KV) ═══
echo ""; echo "═══ Config 2: dualstate_bf16 ═══"
if launch_pd "--enable-dualstate --dualstate-cache-ratio 0.3" "dualstate_bf16"; then
    run_bench "dualstate_bf16" "$RESULTS/dualstate_bf16.json"
    extract_timing "$RESULTS/logs/dualstate_bf16_prefill.log" "P-side"
fi
cleanup

# ═══ Config 3: Baseline (FP8 KV, no DualState) ═══
echo ""; echo "═══ Config 3: baseline_fp8 ═══"
if launch_pd "--kv-cache-dtype fp8_e4m3" "baseline_fp8"; then
    run_bench "baseline_fp8" "$RESULTS/baseline_fp8.json"
    extract_timing "$RESULTS/logs/baseline_fp8_prefill.log" "P-side"
fi
cleanup

# ═══ Config 4: DualState + FP8 KV ═══
echo ""; echo "═══ Config 4: dualstate_fp8 ═══"
if launch_pd "--enable-dualstate --dualstate-cache-ratio 0.3 --kv-cache-dtype fp8_e4m3" "dualstate_fp8"; then
    run_bench "dualstate_fp8" "$RESULTS/dualstate_fp8.json"
    extract_timing "$RESULTS/logs/dualstate_fp8_prefill.log" "P-side"
fi
cleanup

# ═══ Analysis ═══
echo ""
echo "═══ RESULTS COMPARISON ═══"
$PYTHON << 'PYEOF'
import json, glob, os, statistics

RESULTS = os.environ.get("RESULTS_DIR", "PLACEHOLDER")
# Find results dir from script
import sys
for d in sorted(glob.glob("/home/zhujianian/sglang/codex_coding/results/dualstate/fp8kv_*"), reverse=True):
    RESULTS = d
    break

configs = {}
for f in sorted(glob.glob(f"{RESULTS}/*.json")):
    d = json.load(open(f))
    tag = d["tag"]
    points = []
    for sr in d.get("sweep_results", []):
        s = sr.get("stats", {})
        sub = s.get("ttft_subsequent_mean", s.get("ttft_mean", 0))
        if sub and sub > 0 and s.get("n_success", 0) > 0:
            points.append({"label": sr["label"], "sub": sub, "pl": sr["prefix_len"], "fo": sr["fan_out"], "delay": sr["delay_ms"]})
    configs[tag] = points

if not configs:
    print("No results found")
    sys.exit(0)

print(f"\n{'Config':<25} {'p=2k,f=8,d=200':>15} {'p=2k,f=16,d=0':>15} {'p=4k,f=8,d=200':>15} {'p=4k,f=16,d=0':>15}")
print("-" * 90)

for tag in ["baseline_bf16", "dualstate_bf16", "baseline_fp8", "dualstate_fp8"]:
    if tag not in configs:
        continue
    points = configs[tag]
    row = f"  {tag:<23}"
    for target in [(2048,8,200), (2048,16,0), (4096,8,200), (4096,16,0)]:
        matches = [p for p in points if p["pl"]==target[0] and p["fo"]==target[1] and p["delay"]==target[2]]
        if matches:
            avg_ms = statistics.mean(p["sub"] for p in matches) * 1000
            row += f" {avg_ms:>13.1f}ms"
        else:
            row += f" {'—':>15}"
    print(row)

# Compute improvements
print()
base_bf16 = {(p["pl"],p["fo"],p["delay"]): p["sub"] for p in configs.get("baseline_bf16", [])}
for tag in ["dualstate_bf16", "baseline_fp8", "dualstate_fp8"]:
    if tag not in configs:
        continue
    print(f"  {tag} vs baseline_bf16:")
    for p in configs[tag]:
        key = (p["pl"], p["fo"], p["delay"])
        if key in base_bf16 and base_bf16[key] > 0:
            pct = (p["sub"] - base_bf16[key]) / base_bf16[key] * 100
            if abs(pct) > 2:
                print(f"    {p['label']}: {pct:+.1f}%")
PYEOF

echo ""
echo "Results in: $RESULTS"
echo "Done."
