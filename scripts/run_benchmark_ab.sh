#!/usr/bin/env bash
# DualState A/B Benchmark: baseline vs DualState
# P(TP=2, GPU 0,1) + D(TP=2, GPU 2,3) via mooncake_tcp
set -e

PYTHON="/home/zhujianian/miniconda3/envs/sglang-bench/bin/python"
MODEL="/mnt/models/Qwen3.6-35B-A3B"
HOST="127.0.0.1"
P_PORT=30100
D_PORT=30200
LB_PORT=30000
BOOTSTRAP_PORT=30500
RESULTS_DIR="/home/zhujianian/sglang/codex_coding/results/dualstate/benchmark"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MC_TCP_ENABLE_CONNECTION_POOL=true
export PATH="/home/zhujianian/miniconda3/envs/sglang-bench/bin:$PATH"

mkdir -p "$RESULTS_DIR"

cleanup() {
    echo "[cleanup] Killing all sglang processes..."
    pkill -f "sglang.launch_server.*$P_PORT" 2>/dev/null || true
    pkill -f "sglang.launch_server.*$D_PORT" 2>/dev/null || true
    pkill -f "sglang_router.*$LB_PORT" 2>/dev/null || true
    sleep 3
}

wait_health() {
    local url=$1
    local timeout=${2:-300}
    local start=$(date +%s)
    while true; do
        if curl -s --max-time 5 "$url/health" > /dev/null 2>&1; then
            echo "[ready] $url"
            return 0
        fi
        local elapsed=$(( $(date +%s) - start ))
        if [ $elapsed -ge $timeout ]; then
            echo "[TIMEOUT] $url not ready after ${timeout}s"
            return 1
        fi
        sleep 3
    done
}

launch_pd() {
    local dualstate_args="$1"
    local log_prefix="$2"

    echo "[launch] Prefill server (GPU 0,1, TP=2)..."
    $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --host $HOST --port $P_PORT \
        --tp 2 --base-gpu-id 0 \
        --trust-remote-code \
        --disaggregation-mode prefill \
        --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
        --disaggregation-transfer-backend mooncake_tcp \
        $dualstate_args \
        > "${RESULTS_DIR}/${log_prefix}_prefill.log" 2>&1 &

    echo "[launch] Decode server (GPU 2,3, TP=2)..."
    $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --host $HOST --port $D_PORT \
        --tp 2 --base-gpu-id 2 \
        --trust-remote-code \
        --disaggregation-mode decode \
        --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
        --disaggregation-transfer-backend mooncake_tcp \
        $dualstate_args \
        > "${RESULTS_DIR}/${log_prefix}_decode.log" 2>&1 &

    echo "[wait] Waiting for servers..."
    wait_health "http://$HOST:$P_PORT" 300 || return 1
    wait_health "http://$HOST:$D_PORT" 300 || return 1

    echo "[launch] Load balancer..."
    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://$HOST:$P_PORT" \
        --decode "http://$HOST:$D_PORT" \
        --host $HOST --port $LB_PORT \
        > "${RESULTS_DIR}/${log_prefix}_lb.log" 2>&1 &
    sleep 3
    echo "[ready] All services up"
}

run_benchmark() {
    local tag="$1"
    local output="${RESULTS_DIR}/${tag}_results.json"

    echo "[bench] Running benchmark: $tag"
    $PYTHON /home/zhujianian/sglang/codex_coding/src/dualstate/bench_dualstate_ab.py \
        --base-url "http://$HOST:$LB_PORT" \
        --output "$output" \
        --tag "$tag"
    echo "[bench] Results saved to: $output"
}

# ============ Main ============
echo "=============================="
echo "DualState A/B Benchmark"
echo "Model: $MODEL"
echo "=============================="

# --- Baseline ---
echo ""
echo ">>> Phase A: BASELINE (no DualState) <<<"
cleanup
launch_pd "" "baseline"
run_benchmark "baseline"
cleanup

# --- DualState ---
echo ""
echo ">>> Phase B: DUALSTATE <<<"
launch_pd "--enable-dualstate --dualstate-cache-ratio 0.3" "dualstate"
run_benchmark "dualstate"
cleanup

echo ""
echo "=============================="
echo "Benchmark complete! Results in: $RESULTS_DIR"
echo "=============================="
