#!/bin/bash
# PD Disaggregation Experiment: Phase 1-3 for Hybrid Model (Qwen3.6-35B-A3B)
# P(GPU 0,1 TP=2) + D(GPU 2,3 TP=2) with mooncake_tcp
set -euo pipefail

# === Environment ===
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY SOCKS_PROXY socks_proxy 2>/dev/null || true
export NO_PROXY="localhost,127.0.0.1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/zhujianian/miniconda3/envs/sglang-bench/bin/python"
MODEL_PATH="/mnt/models/Qwen3.6-35B-A3B"
P_PORT=30100
D_PORT=30200
LB_PORT=30000
BOOTSTRAP_PORT=8998
TP=2
RESULTS_ROOT="${SCRIPT_DIR}/results/pd_experiment_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_ROOT"
echo "Results: $RESULTS_ROOT"

# === Helper Functions ===
kill_all_servers() {
    pkill -f "sglang.launch_server.*--port ${P_PORT}" 2>/dev/null || true
    pkill -f "sglang.launch_server.*--port ${D_PORT}" 2>/dev/null || true
    pkill -f "sglang.launch_load_balancer" 2>/dev/null || true
    sleep 3
    pkill -9 -f "sglang.launch_server.*--port ${P_PORT}" 2>/dev/null || true
    pkill -9 -f "sglang.launch_server.*--port ${D_PORT}" 2>/dev/null || true
    pkill -9 -f "sglang.launch_load_balancer" 2>/dev/null || true
    sleep 2
}

wait_server_ready() {
    local url="$1"
    local label="$2"
    local max_wait=600
    local waited=0
    echo "  Waiting for $label ($url)..."
    while ! curl -s --noproxy localhost --max-time 5 "${url}/health" 2>/dev/null | grep -q -i "ok\|healthy\|true"; do
        # Also try /v1/models as fallback
        if curl -s --noproxy localhost --max-time 5 "${url}/v1/models" 2>/dev/null | grep -q "id"; then
            break
        fi
        sleep 10
        waited=$((waited + 10))
        if [ $waited -ge $max_wait ]; then
            echo "  ERROR: $label did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "  $label ready after ~${waited}s"
}

warmup_request() {
    local url="$1"
    echo "  Sending warmup request to $url..."
    curl -s --noproxy localhost --max-time 180 "${url}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL_PATH}\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":8}" \
        > /dev/null 2>&1 || true
    echo "  Warmup done."
}

# === PHASE 1.1: P2D2 Baseline Launch + Correctness Verification ===
echo ""
echo "=========================================="
echo "PHASE 1.1: P2D2 Baseline Launch"
echo "=========================================="

kill_all_servers

P_LOG="${RESULTS_ROOT}/phase1_prefill.log"
D_LOG="${RESULTS_ROOT}/phase1_decode.log"

# Start Prefill server (GPU 0,1)
echo "Starting Prefill server (GPU 0,1, TP=$TP, port=$P_PORT)..."
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_MAMBA_RADIX_TRACE=1 \
SGLANG_MAMBA_RADIX_TRACE_FILE="${RESULTS_ROOT}/phase1_prefill_trace.jsonl" \
nohup "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$P_PORT" \
    --host 0.0.0.0 \
    --tp "$TP" \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend mooncake_tcp \
    --trust-remote-code \
    > "$P_LOG" 2>&1 &
P_PID=$!
echo "  Prefill PID: $P_PID"

# Start Decode server (GPU 2,3)
echo "Starting Decode server (GPU 2,3, TP=$TP, port=$D_PORT)..."
CUDA_VISIBLE_DEVICES=2,3 \
SGLANG_MAMBA_RADIX_TRACE=1 \
SGLANG_MAMBA_RADIX_TRACE_FILE="${RESULTS_ROOT}/phase1_decode_trace.jsonl" \
nohup "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$D_PORT" \
    --host 0.0.0.0 \
    --tp "$TP" \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend mooncake_tcp \
    --trust-remote-code \
    > "$D_LOG" 2>&1 &
D_PID=$!
echo "  Decode PID: $D_PID"

# Wait for both to be ready
if ! wait_server_ready "http://localhost:${P_PORT}" "Prefill"; then
    echo "FAILED: Prefill server did not start"
    cat "$P_LOG" | tail -50
    kill_all_servers
    exit 1
fi

if ! wait_server_ready "http://localhost:${D_PORT}" "Decode"; then
    echo "FAILED: Decode server did not start"
    cat "$D_LOG" | tail -50
    kill_all_servers
    exit 1
fi

# Start load balancer
echo "Starting Load Balancer (port=$LB_PORT)..."
nohup "$PYTHON" -m sglang.launch_load_balancer \
    --host 0.0.0.0 \
    --port "$LB_PORT" \
    --worker-urls "http://localhost:${P_PORT}" "http://localhost:${D_PORT}" \
    > "${RESULTS_ROOT}/phase1_lb.log" 2>&1 &
LB_PID=$!
sleep 5
echo "  LB PID: $LB_PID"

# Correctness check
echo ""
echo "--- Correctness Verification ---"
warmup_request "http://localhost:${LB_PORT}"

CORRECTNESS_OUTPUT=$("$PYTHON" -c "
import requests, json, os
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

url = 'http://localhost:${LB_PORT}/v1/chat/completions'
payload = {
    'model': '${MODEL_PATH}',
    'messages': [{'role': 'user', 'content': 'What is 2+3? Answer with just the number.'}],
    'max_tokens': 16,
    'temperature': 0.0,
}
resp = requests.post(url, json=payload, timeout=60)
data = resp.json()
content = data['choices'][0]['message']['content']
print(f'Response: {content}')
print(f'Usage: {data.get(\"usage\", {})}')
print(f'Status: OK' if '5' in content else 'Status: WRONG')
" 2>&1)
echo "$CORRECTNESS_OUTPUT"
echo "$CORRECTNESS_OUTPUT" > "${RESULTS_ROOT}/phase1_correctness.txt"

# GPU memory snapshot
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu \
    --format=csv > "${RESULTS_ROOT}/phase1_gpu_memory.csv" 2>/dev/null || true

kill_all_servers
echo "Phase 1.1 complete."

# === PHASE 1.2: Latency Breakdown Across Configs ===
echo ""
echo "=========================================="
echo "PHASE 1.2: Latency Breakdown (3 configs)"
echo "=========================================="

# Config A: P radix ON + D chunk (default)
# Config B: P radix ON + D radix ON (--disaggregation-decode-enable-radix-cache)
# Config C: P disable radix + D chunk

declare -A CFG_NAMES
declare -A CFG_P_ARGS
declare -A CFG_D_ARGS
CFG_NAMES[A]="P_radix_D_chunk"
CFG_P_ARGS[A]=""
CFG_D_ARGS[A]=""
CFG_NAMES[B]="P_radix_D_radix"
CFG_P_ARGS[B]=""
CFG_D_ARGS[B]="--disaggregation-decode-enable-radix-cache"
CFG_NAMES[C]="P_disabled_D_chunk"
CFG_P_ARGS[C]="--disable-radix-cache"
CFG_D_ARGS[C]=""

WORKLOADS="identical_prompt shared_document_qa random_no_share"
NUM_REQUESTS=8

for cfg_key in A B C; do
    cfg_name="${CFG_NAMES[$cfg_key]}"
    p_extra="${CFG_P_ARGS[$cfg_key]}"
    d_extra="${CFG_D_ARGS[$cfg_key]}"
    cfg_dir="${RESULTS_ROOT}/phase1_2/${cfg_key}_${cfg_name}"
    mkdir -p "$cfg_dir"

    echo ""
    echo "--- Config ${cfg_key}: ${cfg_name} ---"
    echo "  P extra: ${p_extra:-'(none)'}"
    echo "  D extra: ${d_extra:-'(none)'}"

    kill_all_servers

    TRACE_P="${cfg_dir}/prefill_trace.jsonl"
    TRACE_D="${cfg_dir}/decode_trace.jsonl"
    rm -f "$TRACE_P" "$TRACE_D"

    # Start P
    CUDA_VISIBLE_DEVICES=0,1 \
    SGLANG_MAMBA_RADIX_TRACE=1 \
    SGLANG_MAMBA_RADIX_TRACE_FILE="$TRACE_P" \
    nohup "$PYTHON" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$P_PORT" \
        --host 0.0.0.0 \
        --tp "$TP" \
        --disaggregation-mode prefill \
        --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
        --disaggregation-transfer-backend mooncake_tcp \
        --trust-remote-code \
        $p_extra \
        > "${cfg_dir}/prefill.log" 2>&1 &

    # Start D
    CUDA_VISIBLE_DEVICES=2,3 \
    SGLANG_MAMBA_RADIX_TRACE=1 \
    SGLANG_MAMBA_RADIX_TRACE_FILE="$TRACE_D" \
    nohup "$PYTHON" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$D_PORT" \
        --host 0.0.0.0 \
        --tp "$TP" \
        --disaggregation-mode decode \
        --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
        --disaggregation-transfer-backend mooncake_tcp \
        --trust-remote-code \
        $d_extra \
        > "${cfg_dir}/decode.log" 2>&1 &

    if ! wait_server_ready "http://localhost:${P_PORT}" "Prefill"; then
        echo "  FAILED: Prefill server"
        tail -30 "${cfg_dir}/prefill.log"
        kill_all_servers
        echo "FAILED" > "${cfg_dir}/status.txt"
        continue
    fi
    if ! wait_server_ready "http://localhost:${D_PORT}" "Decode"; then
        echo "  FAILED: Decode server"
        tail -30 "${cfg_dir}/decode.log"
        kill_all_servers
        echo "FAILED" > "${cfg_dir}/status.txt"
        continue
    fi

    # Start LB
    nohup "$PYTHON" -m sglang.launch_load_balancer \
        --host 0.0.0.0 --port "$LB_PORT" \
        --worker-urls "http://localhost:${P_PORT}" "http://localhost:${D_PORT}" \
        > "${cfg_dir}/lb.log" 2>&1 &
    sleep 5

    warmup_request "http://localhost:${LB_PORT}"

    # GPU memory
    nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free \
        --format=csv,noheader > "${cfg_dir}/gpu_memory.csv" 2>/dev/null || true

    # Run workloads
    for wl in $WORKLOADS; do
        echo "  Running workload: ${wl}..."
        wl_dir="${cfg_dir}/${wl}"
        mkdir -p "$wl_dir"

        > "$TRACE_P"
        > "$TRACE_D"

        "$PYTHON" "${SCRIPT_DIR}/benchmark/bench_hybrid_prefix_state_gate.py" \
            --server-url "http://localhost:${LB_PORT}" \
            --workload "$wl" \
            --num-requests "$NUM_REQUESTS" \
            --prompt-words 500 \
            --warmup 1 \
            --max-tokens 64 \
            --output-dir "$wl_dir" \
            2>&1 | tee "${wl_dir}/benchmark.log"

        # Copy traces
        [ -s "$TRACE_P" ] && cp "$TRACE_P" "${wl_dir}/prefill_trace.jsonl"
        [ -s "$TRACE_D" ] && cp "$TRACE_D" "${wl_dir}/decode_trace.jsonl"

        # Analyze P-side trace
        if [ -s "${wl_dir}/prefill_trace.jsonl" ]; then
            "$PYTHON" "${SCRIPT_DIR}/benchmark/analyze_mamba_radix_trace.py" \
                "${wl_dir}/prefill_trace.jsonl" \
                --json-output "${wl_dir}/prefill_trace_summary.json" \
                > "${wl_dir}/prefill_trace_analysis.log" 2>&1 || true
        fi
    done

    echo "OK" > "${cfg_dir}/status.txt"
    kill_all_servers
    echo "  Config ${cfg_name} done."
done

echo "Phase 1.2 complete."

# === PHASE 2: Transfer Time + State Size Profiling ===
echo ""
echo "=========================================="
echo "PHASE 2: Transfer & State Size Analysis"
echo "=========================================="

PH2_DIR="${RESULTS_ROOT}/phase2"
mkdir -p "$PH2_DIR"

kill_all_servers

# Launch with verbose logging for transfer profiling
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_MAMBA_RADIX_TRACE=1 \
SGLANG_MAMBA_RADIX_TRACE_FILE="${PH2_DIR}/prefill_trace.jsonl" \
SGLANG_LOG_LEVEL=DEBUG \
nohup "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$P_PORT" \
    --host 0.0.0.0 \
    --tp "$TP" \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend mooncake_tcp \
    --trust-remote-code \
    > "${PH2_DIR}/prefill.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 \
SGLANG_MAMBA_RADIX_TRACE=1 \
SGLANG_MAMBA_RADIX_TRACE_FILE="${PH2_DIR}/decode_trace.jsonl" \
SGLANG_LOG_LEVEL=DEBUG \
nohup "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$D_PORT" \
    --host 0.0.0.0 \
    --tp "$TP" \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend mooncake_tcp \
    --trust-remote-code \
    > "${PH2_DIR}/decode.log" 2>&1 &

if ! wait_server_ready "http://localhost:${P_PORT}" "Prefill"; then
    echo "FAILED: Phase 2 Prefill"
    tail -50 "${PH2_DIR}/prefill.log"
    kill_all_servers
    echo "Phase 2 FAILED" > "${PH2_DIR}/status.txt"
else
    if ! wait_server_ready "http://localhost:${D_PORT}" "Decode"; then
        echo "FAILED: Phase 2 Decode"
        tail -50 "${PH2_DIR}/decode.log"
        kill_all_servers
        echo "Phase 2 FAILED" > "${PH2_DIR}/status.txt"
    else
        nohup "$PYTHON" -m sglang.launch_load_balancer \
            --host 0.0.0.0 --port "$LB_PORT" \
            --worker-urls "http://localhost:${P_PORT}" "http://localhost:${D_PORT}" \
            > "${PH2_DIR}/lb.log" 2>&1 &
        sleep 5
        warmup_request "http://localhost:${LB_PORT}"

        # Run varied-length requests for state size profiling
        "$PYTHON" -c "
import requests, json, time, os
os.environ.pop('http_proxy', None); os.environ.pop('https_proxy', None)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

url = 'http://localhost:${LB_PORT}/v1/chat/completions'
model = '${MODEL_PATH}'
results = []

# Test different prompt lengths
prompt_lengths = [100, 200, 500, 1000, 2000]
base_text = 'The quick brown fox jumps over the lazy dog. ' * 100

for target_words in prompt_lengths:
    prompt = ' '.join(base_text.split()[:target_words])
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt + ' Summarize in one sentence.'}],
        'max_tokens': 64,
        'temperature': 0.0,
    }
    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=180)
        elapsed = time.perf_counter() - start
        data = resp.json()
        usage = data.get('usage', {})
        results.append({
            'target_words': target_words,
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'ttft': elapsed,
            'status': 'ok',
        })
        print(f'  prompt_words={target_words}: prompt_tokens={usage.get(\"prompt_tokens\",0)}, TTFT={elapsed:.3f}s')
    except Exception as e:
        results.append({'target_words': target_words, 'status': 'error', 'error': str(e)})
        print(f'  prompt_words={target_words}: ERROR {e}')

# Run identical prefix sharing (5 requests with same prefix)
print()
print('--- Prefix sharing test (5 identical requests) ---')
shared_prompt = ' '.join(base_text.split()[:500])
for i in range(5):
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': shared_prompt + f' Question {i}: What is this about?'}],
        'max_tokens': 32,
        'temperature': 0.0,
    }
    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=180)
        elapsed = time.perf_counter() - start
        data = resp.json()
        usage = data.get('usage', {})
        results.append({
            'test': 'prefix_sharing',
            'request_idx': i,
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'ttft': elapsed,
            'status': 'ok',
        })
        print(f'  req {i}: prompt_tokens={usage.get(\"prompt_tokens\",0)}, TTFT={elapsed:.3f}s')
    except Exception as e:
        print(f'  req {i}: ERROR {e}')

with open('${PH2_DIR}/varied_length_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to ${PH2_DIR}/varied_length_results.json')
" 2>&1 | tee "${PH2_DIR}/varied_length.log"

        # Extract mamba state info from server logs
        echo ""
        echo "--- Extracting state transfer info from logs ---"
        grep -i "mamba\|state.*transfer\|kv.*transfer\|transfer.*complete\|item_len\|state_item" "${PH2_DIR}/prefill.log" | head -50 > "${PH2_DIR}/prefill_transfer_log_excerpts.txt" 2>/dev/null || true
        grep -i "mamba\|state.*transfer\|kv.*transfer\|transfer.*complete\|item_len\|state_item" "${PH2_DIR}/decode.log" | head -50 > "${PH2_DIR}/decode_transfer_log_excerpts.txt" 2>/dev/null || true

        # Extract KV args info (state sizes)
        grep -i "state_item_lens\|kv_item_len\|state_types\|state_dim" "${PH2_DIR}/prefill.log" | head -20 > "${PH2_DIR}/kv_args_info.txt" 2>/dev/null || true
        grep -i "state_item_lens\|kv_item_len\|state_types\|state_dim" "${PH2_DIR}/decode.log" | head -20 >> "${PH2_DIR}/kv_args_info.txt" 2>/dev/null || true

        echo "OK" > "${PH2_DIR}/status.txt"
        kill_all_servers
    fi
fi

echo "Phase 2 complete."

# === PHASE 3: Mamba State Size Calculation (offline) ===
echo ""
echo "=========================================="
echo "PHASE 3: Mamba State Size Analysis"
echo "=========================================="

PH3_DIR="${RESULTS_ROOT}/phase3"
mkdir -p "$PH3_DIR"

"$PYTHON" -c "
import json, sys
sys.path.insert(0, '${SCRIPT_DIR}/python')

# Load model config
with open('${MODEL_PATH}/config.json') as f:
    cfg = json.load(f)

text_cfg = cfg.get('text_config', cfg)
layer_types = text_cfg.get('layer_types', [])
hidden_size = text_cfg.get('hidden_size', 0)
num_layers = text_cfg.get('num_hidden_layers', len(layer_types))
num_kv_heads = text_cfg.get('num_key_value_heads', text_cfg.get('num_attention_heads', 0))
head_dim = text_cfg.get('head_dim', hidden_size // text_cfg.get('num_attention_heads', 1))
linear_key_head_dim = text_cfg.get('linear_key_head_dim', head_dim)
linear_value_head_dim = text_cfg.get('linear_value_head_dim', head_dim)
linear_num_key_heads = text_cfg.get('linear_num_key_heads', num_kv_heads)
linear_num_value_heads = text_cfg.get('linear_num_value_heads', num_kv_heads)
conv_kernel_dim = text_cfg.get('linear_conv_kernel_dim', 4)
mamba_ssm_dtype = text_cfg.get('mamba_ssm_dtype', 'float32')

# Count layer types
full_attn_layers = [i for i, t in enumerate(layer_types) if 'full' in t]
linear_attn_layers = [i for i, t in enumerate(layer_types) if 'linear' in t or 'mamba' in t.lower()]

print('=' * 60)
print('MODEL ARCHITECTURE ANALYSIS')
print('=' * 60)
print(f'Model: Qwen3.6-35B-A3B')
print(f'Total layers: {num_layers}')
print(f'Full attention layers: {len(full_attn_layers)} (indices: {full_attn_layers[:5]}...)')
print(f'Linear/Mamba layers: {len(linear_attn_layers)} (indices: {linear_attn_layers[:5]}...)')
print(f'Hidden size: {hidden_size}')
print(f'Num KV heads (full attn): {num_kv_heads}')
print(f'Head dim (full attn): {head_dim}')
print(f'Linear key heads: {linear_num_key_heads}, dim: {linear_key_head_dim}')
print(f'Linear value heads: {linear_num_value_heads}, dim: {linear_value_head_dim}')
print(f'Conv kernel dim: {conv_kernel_dim}')
print(f'Mamba SSM dtype: {mamba_ssm_dtype}')

# Calculate sizes
bytes_per_elem_kv = 2  # bfloat16
bytes_per_elem_state = 4 if mamba_ssm_dtype == 'float32' else 2

# KV cache per token per layer (full attention only)
kv_per_token_per_layer = num_kv_heads * head_dim * 2 * bytes_per_elem_kv  # K + V
kv_per_token_total = kv_per_token_per_layer * len(full_attn_layers)

# Mamba/Linear attention state per request (NOT per token - it's a fixed state)
# For linear attention: conv state + ssm state
# Conv state: [num_heads, conv_kernel_dim, head_dim] per layer
# SSM state (recurrent): typically [num_heads, state_size, head_dim] per layer
# For Qwen3.5 linear attention, the state is the conv buffer
conv_state_per_layer = linear_num_key_heads * conv_kernel_dim * linear_key_head_dim * bytes_per_elem_state
# Plus the recurrent KV state for linear attention
linear_kv_state_per_layer = (linear_num_key_heads * linear_key_head_dim + linear_num_value_heads * linear_value_head_dim) * bytes_per_elem_state

mamba_state_per_layer = conv_state_per_layer + linear_kv_state_per_layer
mamba_state_total = mamba_state_per_layer * len(linear_attn_layers)

print()
print('=' * 60)
print('SIZE ANALYSIS')
print('=' * 60)
print(f'KV cache per token per full-attn layer: {kv_per_token_per_layer} bytes ({kv_per_token_per_layer/1024:.1f} KB)')
print(f'KV cache per token (all full-attn layers): {kv_per_token_total} bytes ({kv_per_token_total/1024:.1f} KB)')
print(f'')
print(f'Mamba conv state per linear layer: {conv_state_per_layer} bytes ({conv_state_per_layer/1024:.1f} KB)')
print(f'Mamba linear KV state per linear layer: {linear_kv_state_per_layer} bytes ({linear_kv_state_per_layer/1024:.1f} KB)')
print(f'Mamba total state per linear layer: {mamba_state_per_layer} bytes ({mamba_state_per_layer/1024:.1f} KB)')
print(f'Mamba total state (all linear layers): {mamba_state_total} bytes ({mamba_state_total/1024/1024:.2f} MB)')
print()

# Comparison at different seq lengths
print('=' * 60)
print('TRANSFER SIZE COMPARISON: KV vs Mamba State')
print('=' * 60)
print(f'{\"seq_len\":>10} | {\"KV_size\":>12} | {\"Mamba_size\":>12} | {\"Mamba/KV ratio\":>14} | {\"Mamba/Total\":>12}')
print('-' * 70)
for seq_len in [128, 256, 512, 1024, 2048, 4096, 8192]:
    kv_size = kv_per_token_total * seq_len
    ratio = mamba_state_total / kv_size if kv_size > 0 else float('inf')
    mamba_frac = mamba_state_total / (kv_size + mamba_state_total)
    print(f'{seq_len:>10} | {kv_size/1024/1024:>9.2f} MB | {mamba_state_total/1024/1024:>9.2f} MB | {ratio:>14.4f} | {mamba_frac:>10.1%}')

# Recompute breakeven analysis
print()
print('=' * 60)
print('RECOMPUTE vs TRANSFER BREAKEVEN')
print('=' * 60)
# Assumptions for H100:
# - PCIe bandwidth (cross-GPU via mooncake_tcp): ~10-25 GB/s effective
# - NVLink bandwidth: ~450 GB/s
# - Mamba forward per token: ~0.05-0.2 ms (depends on batch size)
# We'll estimate transfer time and forward time

transfer_bw_low = 5e9   # 5 GB/s (TCP conservative)
transfer_bw_high = 20e9  # 20 GB/s (TCP optimistic)

# Forward time per token for mamba layers only (rough estimate: ~0.1ms per token for 30 layers)
forward_per_token_ms = 0.1  # ms per token (mamba layers only, single request)

transfer_time_low = mamba_state_total / transfer_bw_low * 1000  # ms
transfer_time_high = mamba_state_total / transfer_bw_high * 1000  # ms

breakeven_tokens_low = transfer_time_low / forward_per_token_ms
breakeven_tokens_high = transfer_time_high / forward_per_token_ms

print(f'Mamba state total: {mamba_state_total/1024/1024:.2f} MB')
print(f'')
print(f'Transfer time (5 GB/s TCP conservative): {transfer_time_low:.2f} ms')
print(f'Transfer time (20 GB/s TCP optimistic): {transfer_time_high:.2f} ms')
print(f'')
print(f'Forward time per token (mamba layers): ~{forward_per_token_ms} ms')
print(f'')
print(f'Breakeven tokens (conservative BW): {breakeven_tokens_low:.0f} tokens')
print(f'Breakeven tokens (optimistic BW): {breakeven_tokens_high:.0f} tokens')
print(f'')
print(f'Interpretation:')
print(f'  If a request shares > {breakeven_tokens_high:.0f} prefix tokens with a cached state,')
print(f'  it is cheaper to TRANSFER the mamba state than to RECOMPUTE it.')
print(f'  If < {breakeven_tokens_low:.0f} prefix tokens: recomputation is cheaper than transfer.')

results = {
    'num_full_attn_layers': len(full_attn_layers),
    'num_linear_layers': len(linear_attn_layers),
    'kv_per_token_total_bytes': kv_per_token_total,
    'mamba_state_total_bytes': mamba_state_total,
    'mamba_state_total_MB': mamba_state_total / 1024 / 1024,
    'transfer_time_conservative_ms': transfer_time_low,
    'transfer_time_optimistic_ms': transfer_time_high,
    'breakeven_tokens_conservative': breakeven_tokens_low,
    'breakeven_tokens_optimistic': breakeven_tokens_high,
    'layer_types': layer_types,
    'full_attn_layer_indices': full_attn_layers,
    'linear_layer_indices': linear_attn_layers,
}
with open('${PH3_DIR}/mamba_state_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nResults saved to ${PH3_DIR}/mamba_state_analysis.json')
" 2>&1 | tee "${PH3_DIR}/analysis.log"

echo "Phase 3 complete."

# === Final Summary ===
echo ""
echo "=========================================="
echo "ALL PHASES COMPLETE"
echo "=========================================="
echo "Results directory: ${RESULTS_ROOT}"
echo ""
find "$RESULTS_ROOT" -name "*.json" -o -name "*.csv" -o -name "status.txt" | sort
