#!/usr/bin/env bash
# DualState Long-Context Sweet-Spot Experiment Matrix
# Finds peak DualState TTFT reduction under long prefix + high fan-out
#
# Hardware: 8×H100 80GB
#   P: GPU 0,1 (TP=2)
#   D: GPU 2,3 (TP=2)
#
# Usage:
#   # Full matrix (~2-3 hours)
#   bash run_sweetspot_matrix.sh
#
#   # Quick smoke test (~5 min)
#   SMOKE=1 bash run_sweetspot_matrix.sh
set -euo pipefail

# ─── Configuration ─────────────────────────────────────────────────────────
PYTHON="/home/zhujianian/miniconda3/envs/sglang-bench/bin/python"
MODEL="/mnt/models/Qwen3.6-35B-A3B"
BENCH_SCRIPT="/home/zhujianian/sglang/codex_coding/src/dualstate/bench_loogle_sweetspot.py"
HOST="127.0.0.1"
P_PORT=30100
D_PORT=30200
LB_PORT=30000
BOOTSTRAP_PORT=30500

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_ROOT="/home/zhujianian/sglang/codex_coding/results/dualstate/sweetspot_${TIMESTAMP}"

# Env setup
export MC_TCP_ENABLE_CONNECTION_POOL=true
export PATH="/home/zhujianian/miniconda3/envs/sglang-bench/bin:$PATH"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export NO_PROXY="localhost,127.0.0.1"

mkdir -p "$RESULTS_ROOT"
echo "Results: $RESULTS_ROOT"

# ─── Smoke vs Full Matrix ──────────────────────────────────────────────────
if [ "${SMOKE:-0}" = "1" ]; then
    PREFIX_LENS="4096"
    FAN_OUTS="8"
    DELAYS="0"
    NUM_DOCS=1
    CONCURRENCY=8
    echo "[MODE] SMOKE TEST"
else
    PREFIX_LENS="2048,4096,8192,16384"
    FAN_OUTS="8,16,32"
    DELAYS="0,200"
    NUM_DOCS=3
    CONCURRENCY=16
    echo "[MODE] FULL MATRIX"
fi

# ─── Helper Functions ──────────────────────────────────────────────────────
cleanup() {
    echo "[cleanup] Killing sglang processes..."
    # Kill everything sglang-related to avoid port conflicts
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang_router" 2>/dev/null || true
    pkill -f "sglang.launch_load_balancer" 2>/dev/null || true
    sleep 3
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang_router" 2>/dev/null || true
    pkill -9 -f "sglang.launch_load_balancer" 2>/dev/null || true
    sleep 3
    # Ensure ports are free
    for port in $P_PORT $D_PORT $LB_PORT $BOOTSTRAP_PORT; do
        fuser -k ${port}/tcp 2>/dev/null || true
    done
    sleep 2
}

wait_health() {
    local url=$1
    local label=$2
    local max_wait=${3:-600}
    local waited=0
    echo "  Waiting for ${label} (${url})..."
    while ! curl -s --noproxy localhost --max-time 5 "${url}/health" 2>/dev/null | grep -qi "ok\|healthy\|true"; do
        if curl -s --noproxy localhost --max-time 5 "${url}/v1/models" 2>/dev/null | grep -q "id"; then
            break
        fi
        sleep 10
        waited=$((waited + 10))
        if [ $waited -ge $max_wait ]; then
            echo "  ERROR: ${label} not ready after ${max_wait}s"
            return 1
        fi
    done
    echo "  ${label} ready (${waited}s)"
}

launch_pd() {
    local dualstate_args="${1:-}"
    local log_prefix="$2"
    local log_dir="${RESULTS_ROOT}/logs"
    mkdir -p "$log_dir"

    echo "[launch] P server (GPU 0,1, TP=2)..."
    $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --host $HOST --port $P_PORT \
        --tp 2 --base-gpu-id 0 \
        --trust-remote-code \
        --disaggregation-mode prefill \
        --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
        --disaggregation-transfer-backend mooncake_tcp \
        $dualstate_args \
        > "${log_dir}/${log_prefix}_prefill.log" 2>&1 &

    echo "[launch] D server (GPU 2,3, TP=2)..."
    $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --host $HOST --port $D_PORT \
        --tp 2 --base-gpu-id 2 \
        --trust-remote-code \
        --disaggregation-mode decode \
        --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
        --disaggregation-transfer-backend mooncake_tcp \
        $dualstate_args \
        > "${log_dir}/${log_prefix}_decode.log" 2>&1 &

    wait_health "http://$HOST:$P_PORT" "Prefill" 600 || return 1
    wait_health "http://$HOST:$D_PORT" "Decode" 600 || return 1

    echo "[launch] Load Balancer..."
    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://$HOST:$P_PORT" \
        --decode "http://$HOST:$D_PORT" \
        --host $HOST --port $LB_PORT \
        > "${log_dir}/${log_prefix}_lb.log" 2>&1 &
    sleep 5

    # Verify with a warmup request
    echo "[verify] Sending warmup..."
    local warmup_ok=0
    for attempt in 1 2 3; do
        if curl -s --noproxy localhost --max-time 180 \
            "http://$HOST:$LB_PORT/generate" \
            -H "Content-Type: application/json" \
            -d '{"text":"Hello","sampling_params":{"max_new_tokens":8,"temperature":0}}' \
            2>/dev/null | grep -q "text"; then
            warmup_ok=1
            break
        fi
        echo "  Warmup attempt $attempt failed, retrying..."
        sleep 10
    done

    if [ $warmup_ok -eq 0 ]; then
        echo "  ERROR: Warmup failed after 3 attempts"
        return 1
    fi
    echo "[ready] All services up"
}

run_sweep_benchmark() {
    local tag="$1"
    local output="$2"

    echo "[bench] Running: $tag"
    $PYTHON "$BENCH_SCRIPT" sweep \
        --base-url "http://$HOST:$LB_PORT" \
        --output "$output" \
        --tag "$tag" \
        --model-name "Qwen3.6-35B-A3B" \
        --prefix-lens "$PREFIX_LENS" \
        --fan-outs "$FAN_OUTS" \
        --delays "$DELAYS" \
        --concurrency "$CONCURRENCY" \
        --num-docs "$NUM_DOCS" \
        --max-tokens 16 \
        --warmup 3
    echo "[bench] Done: $output"
}

run_trace_benchmark() {
    local tag="$1"
    local output="$2"
    local trace="$3"
    local prefix_len="$4"
    local fan_out="$5"

    echo "[bench] Running trace: $tag"
    $PYTHON "$BENCH_SCRIPT" trace \
        --base-url "http://$HOST:$LB_PORT" \
        --output "$output" \
        --tag "$tag" \
        --model-name "Qwen3.6-35B-A3B" \
        --trace "$trace" \
        --prefix-len "$prefix_len" \
        --fan-out "$fan_out" \
        --duration 120 \
        --scale 0.5 \
        --max-requests 100 \
        --concurrency "$CONCURRENCY" \
        --num-docs 5 \
        --warmup 3
    echo "[bench] Done: $output"
}

# ─── Main ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  DualState Long-Context Sweet-Spot Experiment Matrix    ║"
echo "║  Model: Qwen3.6-35B-A3B (TP=2, P/D disagg)            ║"
echo "║  Data: LooGLE (14k-50k words/doc)                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════
# Part 1: BASELINE sweep
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Part 1/4: BASELINE Sweep"
echo "═══════════════════════════════════════════════════════════"
cleanup
if launch_pd "" "baseline"; then
    run_sweep_benchmark "baseline" "${RESULTS_ROOT}/baseline_sweep.json"
else
    echo "FAILED to launch baseline servers"
fi
cleanup

# ═══════════════════════════════════════════════════════════════
# Part 2: DUALSTATE sweep
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Part 2/4: DUALSTATE Sweep"
echo "═══════════════════════════════════════════════════════════"
if launch_pd "--enable-dualstate --dualstate-cache-ratio 0.3" "dualstate"; then
    run_sweep_benchmark "dualstate" "${RESULTS_ROOT}/dualstate_sweep.json"
else
    echo "FAILED to launch dualstate servers"
fi
cleanup

# ═══════════════════════════════════════════════════════════════
# Part 3: Trace Replay (only if not smoke test)
# ═══════════════════════════════════════════════════════════════
if [ "${SMOKE:-0}" != "1" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Part 3/4: Trace Replay (Kimi + Azure)"
    echo "═══════════════════════════════════════════════════════════"

    for trace in kimi azure; do
        for prefix_len in 4096 8192; do
            # Baseline
            cleanup
            if launch_pd "" "baseline_trace_${trace}_p${prefix_len}"; then
                run_trace_benchmark \
                    "baseline_${trace}_p${prefix_len}" \
                    "${RESULTS_ROOT}/trace_baseline_${trace}_p${prefix_len}.json" \
                    "$trace" "$prefix_len" 16
            fi
            cleanup

            # DualState
            if launch_pd "--enable-dualstate --dualstate-cache-ratio 0.3" "dualstate_trace_${trace}_p${prefix_len}"; then
                run_trace_benchmark \
                    "dualstate_${trace}_p${prefix_len}" \
                    "${RESULTS_ROOT}/trace_dualstate_${trace}_p${prefix_len}.json" \
                    "$trace" "$prefix_len" 16
            fi
            cleanup
        done
    done
fi

# ═══════════════════════════════════════════════════════════════
# Part 4: Analysis
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Part 4/4: Analysis"
echo "═══════════════════════════════════════════════════════════"

ANALYSIS_SCRIPT="/home/zhujianian/sglang/codex_coding/src/dualstate/analyze_sweetspot_results.py"
if [ -f "$ANALYSIS_SCRIPT" ]; then
    $PYTHON "$ANALYSIS_SCRIPT" \
        --results-dir "$RESULTS_ROOT" \
        --output-dir "${RESULTS_ROOT}/analysis"
else
    echo "  [skip] Analysis script not found: $ANALYSIS_SCRIPT"
fi

# ═══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  EXPERIMENT COMPLETE                                    ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Results: ${RESULTS_ROOT}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
find "$RESULTS_ROOT" -name "*.json" | sort
