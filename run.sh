#!/bin/bash
# DualState One-Click Run
# Usage: bash run.sh <model> <config> [extra_args]
#   model:  qwen | kimi
#   config: baseline | dualstate | ds_fp8
#
# Examples:
#   bash run.sh qwen baseline          # SGLang P/D baseline (Qwen3.6)
#   bash run.sh qwen dualstate         # DualState only
#   bash run.sh qwen ds_fp8            # DualState + FP8 KV
#   bash run.sh kimi baseline          # SGLang P/D baseline (Kimi-Linear)
#   bash run.sh kimi dualstate         # DualState on Kimi-Linear
#   bash run.sh all                    # Run all 5 configs sequentially

SGLANG_DIR="${SGLANG_DIR:-/home/zhujianian/sglang}"
PYTHON="${PYTHON:-/home/zhujianian/miniconda3/envs/sglang-bench/bin/python}"
BENCH="$SGLANG_DIR/codex_coding/src/dualstate/bench_loogle_sweetspot.py"

# Model paths (adjust to your setup)
QWEN_MODEL="${QWEN_MODEL:-/mnt/models/Qwen3.6-35B-A3B}"
KIMI_MODEL="${KIMI_MODEL:-/mnt/models/Kimi-Linear-48B-A3B-Instruct}"

HOST="127.0.0.1"
P_PORT=30100; D_PORT=30200; LB_PORT=30000; BOOTSTRAP_PORT=30500
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="$SGLANG_DIR/codex_coding/results/dualstate/run_${TIMESTAMP}"

export MC_TCP_ENABLE_CONNECTION_POOL=true
export NO_PROXY="localhost,127.0.0.1"
export PATH="$(dirname $PYTHON):$PATH"

set +e
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do
    unset $v 2>/dev/null
done

do_cleanup() {
    pkill -f "sglang" 2>/dev/null || true
    sleep 3
    pkill -9 -f "sglang" 2>/dev/null || true
    sleep 3
}

run_one() {
    local MODEL_PATH="$1"
    local MODEL_NAME="$2"
    local TAG="$3"
    local EXTRA="$4"
    local RESULTS="$RESULTS_BASE/$TAG"

    mkdir -p "$RESULTS/logs"

    echo ""
    echo "================================================================"
    echo "  $TAG | $MODEL_NAME"
    echo "  Extra args: ${EXTRA:-none}"
    echo "  Results: $RESULTS"
    echo "================================================================"
    do_cleanup

    # Launch P
    $PYTHON -m sglang.launch_server \
        --model-path "$MODEL_PATH" --host $HOST --port $P_PORT \
        --tp 2 --base-gpu-id 0 --trust-remote-code \
        --disaggregation-mode prefill --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
        --disaggregation-transfer-backend mooncake_tcp \
        $EXTRA \
        > "$RESULTS/logs/prefill.log" 2>&1 &

    # Launch D
    $PYTHON -m sglang.launch_server \
        --model-path "$MODEL_PATH" --host $HOST --port $D_PORT \
        --tp 2 --base-gpu-id 2 --trust-remote-code \
        --disaggregation-mode decode --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
        --disaggregation-transfer-backend mooncake_tcp \
        $EXTRA \
        > "$RESULTS/logs/decode.log" 2>&1 &

    # Wait for ready
    echo "  Waiting for servers..."
    for i in $(seq 1 60); do
        sleep 10
        local p=0 d=0
        curl -s --noproxy localhost --max-time 3 "http://$HOST:$P_PORT/v1/models" 2>/dev/null | grep -q "id" && p=1
        curl -s --noproxy localhost --max-time 3 "http://$HOST:$D_PORT/v1/models" 2>/dev/null | grep -q "id" && d=1
        if [ $p -eq 1 ] && [ $d -eq 1 ]; then
            echo "  Servers ready ($((i*10))s)"
            break
        fi
        if [ $i -eq 60 ]; then echo "  TIMEOUT"; do_cleanup; return 1; fi
    done

    # Launch LB
    $PYTHON -m sglang_router.launch_router --pd-disaggregation --mini-lb \
        --prefill "http://$HOST:$P_PORT" --decode "http://$HOST:$D_PORT" \
        --host $HOST --port $LB_PORT > "$RESULTS/logs/lb.log" 2>&1 &
    sleep 5

    # Warmup
    for a in 1 2 3 4 5; do
        curl -s --noproxy localhost --max-time 180 "http://$HOST:$LB_PORT/generate" \
            -H "Content-Type: application/json" \
            -d '{"text":"Hello world test","sampling_params":{"max_new_tokens":4,"temperature":0}}' \
            2>/dev/null | grep -q "text" && break
        sleep 10
    done

    # Sweep benchmark
    echo "  [sweep] Running (prefix=2k/4k × fan_out=8/16 × delay=0/200ms)..."
    $PYTHON "$BENCH" sweep \
        --base-url "http://$HOST:$LB_PORT" \
        --output "$RESULTS/sweep.json" --tag "$TAG" \
        --model-name "$MODEL_NAME" \
        --prefix-lens "2048,4096" --fan-outs "8,16" --delays "0,200" \
        --concurrency 16 --num-docs 2 --max-tokens 16 --warmup 3

    # Trace replay
    echo "  [trace] Running (Kimi K25 trace, p=4k, f=16)..."
    $PYTHON "$BENCH" trace \
        --base-url "http://$HOST:$LB_PORT" \
        --output "$RESULTS/trace.json" --tag "$TAG" \
        --model-name "$MODEL_NAME" \
        --trace kimi --prefix-len 4096 --fan-out 16 --duration 120 --scale 0.5 \
        --max-requests 100 --concurrency 16 --num-docs 5 --warmup 3

    echo "  ✓ Done: $TAG"
    do_cleanup
}

# ═══ Parse arguments ═══

MODEL="${1:-all}"
CONFIG="${2:-all}"

if [ "$MODEL" = "all" ]; then
    run_one "$QWEN_MODEL" "Qwen3.6-35B-A3B" "qwen_baseline" ""
    run_one "$QWEN_MODEL" "Qwen3.6-35B-A3B" "qwen_dualstate" "--enable-dualstate --dualstate-cache-ratio 0.3"
    run_one "$QWEN_MODEL" "Qwen3.6-35B-A3B" "qwen_ds_fp8" "--enable-dualstate --dualstate-cache-ratio 0.3 --kv-cache-dtype fp8_e4m3"
    run_one "$KIMI_MODEL" "Kimi-Linear-48B" "kimi_baseline" ""
    run_one "$KIMI_MODEL" "Kimi-Linear-48B" "kimi_dualstate" "--enable-dualstate --dualstate-cache-ratio 0.3"
else
    case "$MODEL" in
        qwen) MODEL_PATH="$QWEN_MODEL"; MODEL_NAME="Qwen3.6-35B-A3B" ;;
        kimi) MODEL_PATH="$KIMI_MODEL"; MODEL_NAME="Kimi-Linear-48B" ;;
        *) echo "Unknown model: $MODEL (use qwen|kimi|all)"; exit 1 ;;
    esac

    case "$CONFIG" in
        baseline)   EXTRA=""; TAG="${MODEL}_baseline" ;;
        dualstate)  EXTRA="--enable-dualstate --dualstate-cache-ratio 0.3"; TAG="${MODEL}_dualstate" ;;
        ds_fp8)     EXTRA="--enable-dualstate --dualstate-cache-ratio 0.3 --kv-cache-dtype fp8_e4m3"; TAG="${MODEL}_ds_fp8" ;;
        *)          echo "Unknown config: $CONFIG (use baseline|dualstate|ds_fp8|all)"; exit 1 ;;
    esac

    run_one "$MODEL_PATH" "$MODEL_NAME" "$TAG" "$EXTRA"
fi

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  All runs complete!                               ║"
echo "║  Results: $RESULTS_BASE"
echo "╚════════════════════════════════════════════════════╝"
ls "$RESULTS_BASE"/*/sweep.json "$RESULTS_BASE"/*/trace.json 2>/dev/null
