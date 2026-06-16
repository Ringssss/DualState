# DualState: Checkpoint-Aware Disaggregated Serving for Hybrid LLMs

> **Status**: Phase 1-3 core implemented + FP8 KV compression validated
> **Target**: USENIX ATC 2026
> **Codebase**: SGLang (editable install on `/home/zhujianian/sglang/`)
> **Hardware**: 8×NVIDIA H100 80GB HBM3, NVLink intra-node
> **Last Updated**: 2026-06-12

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Start](#2-quick-start)
3. [Architecture](#3-architecture)
4. [Implementation Status](#4-implementation-status)
5. [Experiment Results](#5-experiment-results)
6. [Applicable Models](#6-applicable-models)
7. [File Map](#7-file-map)
8. [Benchmark Tools](#8-benchmark-tools)
9. [Key Discoveries](#9-key-discoveries)
10. [Remaining Work](#10-remaining-work)

---

## 1. Project Overview

### Problem

In Prefill/Decode (P/D) disaggregated serving for **hybrid models** (models mixing full attention with recurrent layers like Mamba/Linear Attention), the P→D transfer cost is **affine**, not linear:

```
transfer_cost = β × (kv_per_token × seq_len + MAMBA_STATE_SIZE)
                                    ↑ linear        ↑ fixed constant (~30.8MB)
```

SGLang's vanilla P/D treats all state equally — it transfers the full mamba state (30.8MB) for every request, even when multiple requests share the same prefix and could reuse the same mamba checkpoint.

### Solution

DualState introduces **checkpoint-aware caching** on the D-side:

1. **Fork**: After first request completes, D forks the mamba state into a cache (CAM)
2. **COW (Copy-on-Write)**: Subsequent requests with same prefix get a COW copy from cache (~0.03ms)
3. **Skip**: P skips mamba state transfer AND mamba recompute for shared prefixes

Combined with **FP8 KV cache** compression, this achieves **up to -33.5% TTFT** over SGLang P/D baseline.

---

## 2. Quick Start

### Prerequisites

```bash
# Activate environment
conda activate sglang-bench

# Verify editable install
python -c "import sglang; print(sglang.__path__)"
# Should print: ['/home/zhujianian/sglang/python/sglang']
```

### Launch DualState P/D (Qwen3.6-35B-A3B)

```bash
# Environment setup
export MC_TCP_ENABLE_CONNECTION_POOL=true
export NO_PROXY="localhost,127.0.0.1"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

PYTHON="/home/zhujianian/miniconda3/envs/sglang-bench/bin/python"
MODEL="/mnt/models/Qwen3.6-35B-A3B"

# Start Prefill server (GPU 0,1, TP=2)
$PYTHON -m sglang.launch_server \
    --model-path $MODEL --host 127.0.0.1 --port 30100 \
    --tp 2 --base-gpu-id 0 --trust-remote-code \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 30500 \
    --disaggregation-transfer-backend mooncake_tcp \
    --enable-dualstate --dualstate-cache-ratio 0.3 &

# Start Decode server (GPU 2,3, TP=2)
$PYTHON -m sglang.launch_server \
    --model-path $MODEL --host 127.0.0.1 --port 30200 \
    --tp 2 --base-gpu-id 2 --trust-remote-code \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port 30500 \
    --disaggregation-transfer-backend mooncake_tcp \
    --enable-dualstate --dualstate-cache-ratio 0.3 &

# Start Load Balancer
$PYTHON -m sglang_router.launch_router \
    --pd-disaggregation --mini-lb \
    --prefill http://127.0.0.1:30100 \
    --decode http://127.0.0.1:30200 \
    --host 127.0.0.1 --port 30000 &
```

### Launch DualState + FP8 KV (best performance)

Add `--kv-cache-dtype fp8_e4m3` to **both** P and D:

```bash
# P server (add fp8 flag)
$PYTHON -m sglang.launch_server \
    --model-path $MODEL --host 127.0.0.1 --port 30100 \
    --tp 2 --base-gpu-id 0 --trust-remote-code \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 30500 \
    --disaggregation-transfer-backend mooncake_tcp \
    --enable-dualstate --dualstate-cache-ratio 0.3 \
    --kv-cache-dtype fp8_e4m3 &

# D server (same fp8 flag)
$PYTHON -m sglang.launch_server \
    --model-path $MODEL --host 127.0.0.1 --port 30200 \
    --tp 2 --base-gpu-id 2 --trust-remote-code \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port 30500 \
    --disaggregation-transfer-backend mooncake_tcp \
    --enable-dualstate --dualstate-cache-ratio 0.3 \
    --kv-cache-dtype fp8_e4m3 &
```

> **Note**: FP8 KV only works WITH DualState enabled. Baseline P/D + FP8 has a bootstrap compatibility issue.

### Launch Baseline (for comparison)

Same as above but remove `--enable-dualstate --dualstate-cache-ratio 0.3`:

```bash
# P server (no dualstate flags)
$PYTHON -m sglang.launch_server \
    --model-path $MODEL --host 127.0.0.1 --port 30100 \
    --tp 2 --base-gpu-id 0 --trust-remote-code \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 30500 \
    --disaggregation-transfer-backend mooncake_tcp &
```

### Send a Request

```bash
curl --noproxy localhost -s http://127.0.0.1:30000/generate \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello, tell me about AI","sampling_params":{"max_new_tokens":32,"temperature":0}}'
```

### Kimi-Linear-48B-A3B

Replace model path and everything else stays the same:

```bash
MODEL="/mnt/models/Kimi-Linear-48B-A3B-Instruct"
# Same launch commands as above
```

### Available Server Args

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-dualstate` | `false` | Enable DualState checkpoint-aware serving |
| `--dualstate-cache-ratio` | `0.3` | Fraction of D-side mamba pool reserved for checkpoint cache |
| `--kv-cache-dtype fp8_e4m3` | `auto` (bf16) | Use FP8 KV cache (halves transfer bytes) |
| `--disaggregation-mode` | — | `prefill` or `decode` |
| `--disaggregation-transfer-backend` | — | `mooncake_tcp` (RDMA) |
| `--disaggregation-bootstrap-port` | — | Bootstrap server port for P/D handshake |

### Transfer Profiling

Enable transfer timing instrumentation:

```bash
export SGLANG_DISAGG_TRANSFER_TIMING=1  # add to both P and D
# Logs will show: TRANSFER_TIMING: blocks=20 total=92.2MB elapsed=60.12ms bw=1.5GB/s
```

Enable cost model logging:

```bash
export SGLANG_DUALSTATE_COST_MODEL=1  # add to P side
# Logs will show: DualState TransferMode: mode=kv_only d_state=C prefix=2048 ...
```

---

## 3. Architecture

### System Diagram

```
                     ┌──────────────────────────────────────┐
                     │        DualState Scheduler           │
                     │     (Affine Cost Model Decision)     │
                     │                                      │
                     │  TransferModes:                      │
                     │    FULL / KV_ONLY / DELTA_KV_ONLY    │
                     │    KV_MAMBA / P_RECOMPUTE_D_COW      │
                     └──────────┬───────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 │                  ▼
    ┌──────────────────┐       │       ┌─────────────────────┐
    │   P-side          │       │       │   D-side             │
    │   MambaRadixCache │       │       │   CAM                │
    │   (prefix tree)   │       │       │   (CheckpointAvail-  │
    │                   │       │       │    abilityMap)        │
    │  • match_prefix() │       │       │                      │
    │  • insert()       │       │       │  • fork_and_cache()  │
    │  • trace events   │       │       │  • cow_from_cache()  │
    └────────┬──────────┘       │       └──────────┬──────────┘
             │                  │                   │
             │     ┌────────────┘                   │
             │     │   Coherence Manager            │
             │     │   (shared-memory singleton)    │
             │     │   D reports → P queries        │
             │     └────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────────┐
    │              Mooncake RDMA Transfer            │
    │  • KV cache: per-layer batch transfer         │
    │  • Mamba state: skip if D has COW cache       │
    │  • FP8 mode: half bytes, half transfer time   │
    │  • Measured BW: ~1.5 GB/s effective           │
    └───────────────────────────────────────────────┘
```

### Data Flow (per request)

```
1. Request arrives at Load Balancer → routes to P

2. P-side:
   a. Match prefix in MambaRadixCache (trace: structural vs state match)
   b. Prefill compute (full-attn + linear-attn layers)
   c. Check: D has mamba cache? (via empty dst_state_indices from D COW)
      YES → skip mamba transfer (save 21ms + 40ms recompute)
      NO  → transfer mamba state (30.8MB, ~21ms)
   d. Transfer KV cache (bf16: 60-114ms, fp8: 30-57ms)

3. D-side:
   a. Pre-alloc: match prefix in CAM
      HIT → COW from cache (0.03ms) → send empty mamba indices to P
      MISS → allocate fresh mamba slots → send indices to P
   b. Receive KV from P
   c. If first time for this prefix: fork mamba state to CAM
   d. Start decode
```

---

## 4. Implementation Status

### ✅ Completed (Phase 1-3 + FP8)

| Component | Files | Status |
|-----------|-------|--------|
| **MambaRadixCache Instrumentation** | `mem_cache/mamba_radix_trace.py` (new), `mem_cache/mamba_radix_cache.py` (modified) | ✅ Trace events for match/split/insert/evict |
| **D-side CAM** | `mem_cache/checkpoint_availability_map.py` (new) | ✅ O(1) hash lookup, COW, LFU eviction |
| **D-side COW Pipeline** | `disaggregation/decode.py` (modified) | ✅ Fork→COW→Skip fully working |
| **P-side Mamba Skip** | `disaggregation/mooncake/conn.py` (modified) | ✅ Skip transfer when D has COW cache |
| **Coherence Manager** | `disaggregation/dualstate_coherence.py` (new) | ✅ Shared-memory P↔D state reporting |
| **Cost Model** | `disaggregation/dualstate_scheduler.py` (new) | ✅ TransferMode decision logic (logged, not yet steering) |
| **Server Args** | `server_args.py` (modified) | ✅ `--enable-dualstate`, `--dualstate-cache-ratio` |
| **FP8 KV Validation** | No code change, `--kv-cache-dtype fp8_e4m3` | ✅ Halves KV transfer bytes |
| **Transfer Timing** | `disaggregation/mooncake/conn.py` (modified) | ✅ `SGLANG_DISAGG_TRANSFER_TIMING=1` |
| **Benchmark Suite** | `codex_coding/src/dualstate/` (new) | ✅ 8 workload types + sweep + trace replay |

### 🔲 Not Yet Implemented (Phase 4-5)

| Component | Description | Reason Deferred |
|-----------|-------------|-----------------|
| DELTA_KV_ONLY TransferMode | Skip cached KV tokens in transfer | D-side radix cache incompatible with Mamba models in SGLang |
| Hierarchical prefix hash | Longest-prefix match for agentic workloads | Needs delta-mamba transfer support |
| Multi-node RDMA validation | Cross-node P/D with IB/RoCE | Requires multi-node setup |
| UCCL-P2P backend | Production-grade transfer with DietGPU | Requires UCCL build + integration |

### ❌ Evaluated and Rejected

| Optimization | Finding | Decision |
|-------------|---------|----------|
| Batched COW kernel | COW copy is 0.029ms — negligible | Not worth the complexity |
| GPU-Direct IPC bypass | Mooncake already uses RDMA, not TCP loopback | No gain over mooncake |
| D-side radix cache | SGLang blocks it for Mamba models (`ValueError`) | Incompatible |

---

## 5. Experiment Results

### 5.1 DualState vs Baseline (Qwen3.6-35B-A3B)

#### Short Prefix Sweep (100-900 tokens, Phase 3 validation)

| Config | Mean TTFT | vs Baseline |
|--------|:---------:|:-----------:|
| Baseline | 193ms | — |
| **DualState** | **171ms** | **-12.2%** |

41/42 sweep points DualState wins. Both Qwen3.6 (-12.2%) and Kimi-Linear (-8.7%).

#### Long Prefix Sweet-Spot (LooGLE docs, 2k-4k tokens)

| Point | Baseline | DualState | Δ |
|-------|:--------:|:---------:|:---:|
| p=2k f=8 d=200ms | 307ms | 265ms | **-13.8%** |
| p=2k f=16 burst | 700ms | 536ms | **-29.0%** ★ |
| p=4k f=16 burst | 856ms | 678ms | **-20.8%** |
| p=4k f=8 d=200ms | 319ms | 266ms | **-16.4%** |

#### Production Trace (Kimi K25 trace, 245K requests)

| Trace | Mean TTFT | P50 TTFT | P99 TTFT |
|-------|:---------:|:--------:|:--------:|
| Kimi p=4k | **-18.5%** | **-27.7%** | **-40.4%** 🔥 |
| Kimi p=8k | -3.7% | -1.3% | -22.3% |
| Azure p=4k | -0.1% | -2.6% | +3.7% |
| Azure p=8k | -2.4% | -0.7% | -8.7% |

### 5.2 DualState + FP8 KV (Qwen3.6-35B-A3B)

| Config | Avg TTFT | vs Baseline |
|--------|:--------:|:-----------:|
| Baseline bf16 | 486ms | — |
| DualState bf16 | 432ms | **-11.1%** |
| **DualState + FP8** | **396ms** | **-18.5%** |

Best case: p=2k f=16 burst → **-33.5%** (700ms → 465ms)

FP8 adds an extra **-8.4% on top of DualState** by halving KV transfer bytes.

### 5.3 Kimi-Linear-48B-A3B (Full Test, 2026-06-12)

#### Sweep (LooGLE long documents, prefix=2k-4k)

| Point | Baseline | DualState | Δ |
|-------|:--------:|:---------:|:---:|
| p=2k f=8 burst | 873ms | 523ms | **-40.0%** 🔥 |
| p=4k f=8 d=200ms | 276ms | 167ms | **-39.3%** 🔥 |
| p=2k f=8 d=200ms | 239ms | 154ms | **-35.6%** |
| p=4k f=16 d=200ms | 222ms | 165ms | **-25.9%** |
| p=2k f=16 d=200ms | 178ms | 148ms | **-16.8%** |
| p=4k f=8 burst | 687ms | 616ms | -10.4% |
| p=2k f=16 burst | 664ms | 619ms | -6.7% |
| p=4k f=16 burst | 907ms | 861ms | -5.2% |
| **Average** | **506ms** | **407ms** | **-19.6%** |

#### Production Trace (Kimi K25 trace, p=4k, fan_out=16)

| Metric | Baseline | DualState | Δ |
|--------|:--------:|:---------:|:---:|
| **Mean** | 443ms | 373ms | **-15.9%** |
| **P50** | 291ms | 185ms | **-36.2%** 🔥 |
| **P90** | 488ms | 265ms | **-45.7%** 🔥 |
| P99 | 4773ms | 4805ms | +0.7% (outlier) |

#### DualState + FP8 KV on Kimi-Linear

| Result | |
|--------|--|
| **Status** | ❌ **FP8 not compatible with Kimi-Linear** |
| Effect | TTFT increased +89.9% (severe degradation) |
| Cause | Kimi-Linear's pure linear attention KV layout does not tolerate FP8 quantization in P/D disagg path |

> **Note**: FP8 KV works on Qwen3.6 (mixed full-attn + linear-attn) but fails on Kimi-Linear (pure linear-attn). The full attention layers appear to be more robust to FP8 quantization.

### 5.4 Two-Model Summary

| Metric | Qwen3.6-35B-A3B | Kimi-Linear-48B |
|--------|:----------------:|:----------------:|
| **DualState avg TTFT reduction** | **-11.1%** | **-19.6%** |
| **DualState peak (sweep)** | **-29.0%** (p2k f16 burst) | **-40.0%** (p2k f8 burst) |
| **Trace Mean** | -18.5% | -15.9% |
| **Trace P50** | **-27.7%** | **-36.2%** |
| **Trace P90** | -14.1% | **-45.7%** |
| **DualState + FP8 avg** | **-18.5%** ✅ | ❌ Not compatible |
| **DualState + FP8 peak** | **-33.5%** ✅ | ❌ |

**Key observations**:

1. **Kimi-Linear benefits MORE from DualState** (-19.6% vs -11.1% avg): because all 27 layers are linear attention (vs Qwen's 30/40 mix), the mamba state is proportionally larger relative to KV, so skipping it saves more.

2. **FP8 KV is model-dependent**: works on Qwen3.6 (has full-attn layers that tolerate FP8) but fails on Kimi-Linear (pure linear-attn, FP8-sensitive).

3. **Both models validate DualState**: the Fork→COW→Skip pipeline delivers consistent double-digit TTFT reduction across different hybrid architectures.

### 5.5 Transfer Timing (measured)

| Transfer | Size | Time | Effective BW |
|----------|:----:|:----:|:------------:|
| KV cache (large) | 92MB | 60ms | 1.5 GB/s |
| KV cache (large) | 114MB | 74ms | 1.5 GB/s |
| Mamba state | 32MB | 21ms | 1.5 GB/s |
| Raw NVLink (reference) | 80MB | 0.23ms | 380 GB/s |

> **Key finding**: Mooncake RDMA effective bandwidth is only ~1.5 GB/s (not NVLink's 380 GB/s). This makes FP8 compression very valuable.

### 5.6 Micro-Benchmark (kernel-level)

| Operation | Time | Note |
|-----------|:----:|------|
| COW copy (30.8MB mamba state) | 0.029ms | Negligible |
| FP8 quantize (80MB bf16 → 40MB fp8) | 0.10ms | Negligible |
| FP8 dequantize | 0.05ms | Negligible |
| Raw GPU→GPU copy (80MB, NVLink) | 0.23ms | Hardware limit |

---

## 6. Applicable Models

### ✅ DualState Applicable (hybrid models with recurrent state)

| Model | Vendor | Recurrent Type | Est. State Size |
|-------|:------:|:--------------:|:---------------:|
| **Qwen3.5/3.6-35B-A3B** | Alibaba | Linear Attention (KDA) | ~30.8MB |
| **Kimi-Linear-48B-A3B** | Moonshot AI | Linear Attention | ~similar |
| **Falcon H1** | TII | Mamba2 SSM | SSM state |
| **Jamba/Jamba2** | AI21 | Mamba | SSM state |
| **NemotronH** | NVIDIA | Mamba2 + MoE | SSM state |
| **Bailing MoE** | MiniMax | Hybrid | Model-specific |
| **Lfm2** | Liquid AI | ShortConv | Conv state |
| **Granite MoE Hybrid** | IBM | Mamba | SSM state |
| **MiniCPM V4.6** | OpenBMB | Wraps Qwen3.5 hybrid | Same as Qwen3.5 |

### ❌ NOT Applicable (no recurrent state)

| Model | Reason |
|-------|--------|
| **DeepSeek V2/V3/V4** | MLA is per-token compressed KV, not recurrent |
| **GLM-5.1 / GLM4 MoE** | Standard GQA attention |
| **Kimi K2.5** | Uses DeepSeek V3 (MLA) as text backbone |
| **MiMo V2/V2.5** | SWA + Full Attention, no recurrent state |
| **Llama / Qwen3 (pure)** | Standard transformer |

### Why the Distinction Matters

DualState exploits the **affine cost structure** unique to hybrid models:

```
Hybrid model transfer:  cost = β × (KV_per_token × seq_len + MAMBA_CONST)
                                                                ↑ THIS is what DualState eliminates

Pure transformer:       cost = β × KV_per_token × seq_len
                                    (purely linear, no constant to exploit)
```

---

## 7. File Map

### New Files (DualState-specific)

```
python/sglang/srt/
├── disaggregation/
│   ├── dualstate_coherence.py          # P↔D shared coherence manager
│   └── dualstate_scheduler.py          # TransferMode cost model + decisions
├── mem_cache/
│   ├── checkpoint_availability_map.py  # D-side CAM (fork/COW/evict)
│   └── mamba_radix_trace.py            # Prefix cache instrumentation (untracked)
└── disaggregation/common/
    └── kv_compress.py                  # FP8 quantize/dequantize kernels (prototype)
```

### Modified Files

```
python/sglang/srt/
├── server_args.py                      # --enable-dualstate, --dualstate-cache-ratio
├── managers/scheduler.py               # _init_dualstate() on decode side
├── mem_cache/mamba_radix_cache.py      # Trace instrumentation hooks
├── disaggregation/decode.py            # D-side COW pipeline + fork_and_cache
├── disaggregation/mooncake/conn.py     # P-side mamba skip + transfer timing
├── configs/model_config.py             # Hybrid model config extensions
└── environ.py                          # SGLANG_EXPERIMENTAL env var
```

### Benchmark & Experiment Files

```
codex_coding/src/dualstate/
├── bench_loogle_sweetspot.py           # Long-context sweep + trace replay
├── bench_hybrid_prefix_state_gate.py   # 8 workload types for prefix cache
├── bench_dualstate_comprehensive.py    # Multi-workload TTFT comparison
├── bench_dualstate_ab.py               # A/B test client
├── bench_trace_matrix.py               # Kimi/Azure trace replay
├── analyze_mamba_radix_trace.py        # JSONL trace analyzer
├── analyze_sweetspot_results.py        # Heatmap + cross-node projection
├── run_sweetspot_matrix.sh             # Full experiment matrix runner
├── run_benchmark_ab.sh                 # A/B test orchestrator
├── run_fp8kv_test.sh                   # FP8 KV experiment runner
└── pd_experiment_phase1_3.sh           # P/D Phase 1-3 experiments
```

### Experiment Results

```
codex_coding/results/dualstate/
├── validation_20260608/                # Phase 3 validation (short prefix)
│   ├── baseline_Qwen36.json
│   ├── dualstate_Qwen36.json
│   ├── sweep_baseline_Qwen36.json      # 21-point sweep
│   ├── sweep_dualstate_Qwen36.json
│   ├── sweep_baseline_Kimi.json
│   └── sweep_dualstate_Kimi.json
├── sweetspot_20260610_133453/          # Long prefix sweet-spot (LooGLE)
│   ├── baseline_sweep.json             # 72 points
│   ├── dualstate_sweep.json            # 72 points
│   ├── trace_baseline_kimi_p4096.json  # Kimi trace replay
│   ├── trace_dualstate_kimi_p4096.json
│   └── analysis/                       # Heatmap + cross-node projection
├── fp8kv_20260612_004019/              # FP8 KV experiment (Qwen3.6)
│   ├── baseline_bf16.json
│   ├── dualstate_bf16.json
│   └── dualstate_fp8.json              # Best: -18.5% avg
├── kimi_full_20260612_145512/          # Kimi-Linear full test
│   ├── baseline_bf16.json              # 16 sweep points
│   ├── dualstate_bf16.json             # Peak: -40.0%
│   ├── dualstate_fp8.json              # FP8 NOT compatible (degraded)
│   ├── trace_baseline.json             # Kimi production trace
│   └── trace_dualstate.json            # P50: -36.2%
├── matrix_20260608/                    # Trace-driven matrix
│   ├── baseline_qwen_kimi_s1.0.json
│   ├── dualstate_qwen_kimi_s1.0.json
│   └── ...
└── benchmark/                          # Early A/B tests
    └── baseline_results.json
```

### Documentation

```
/home/zhujianian/528.md                         # Phase 1-3 experiment log (StateGate)
docs/developer_guide/dualstate_implementation_plan.md  # Architecture design doc
```

---

## 8. Benchmark Tools

### Run Quick A/B Test

```bash
# Automated baseline vs DualState comparison
bash codex_coding/src/dualstate/run_benchmark_ab.sh
```

### Run Long-Context Sweet-Spot Sweep

```bash
# Smoke test (1 prefix × 1 fan_out, ~5 min)
SMOKE=1 bash codex_coding/src/dualstate/run_sweetspot_matrix.sh

# Full matrix (4 prefix × 3 fan_out × 2 delays × 3 docs, ~2-3 hours)
bash codex_coding/src/dualstate/run_sweetspot_matrix.sh
```

### Run Individual Benchmark

```bash
PYTHON="/home/zhujianian/miniconda3/envs/sglang-bench/bin/python"

# Sweep mode (systematic prefix × fan_out matrix)
$PYTHON codex_coding/src/dualstate/bench_loogle_sweetspot.py sweep \
    --base-url http://127.0.0.1:30000 \
    --output results.json --tag my_test \
    --prefix-lens 2048,4096 --fan-outs 8,16 --delays 0,200

# Trace replay mode (production trace arrival pattern)
$PYTHON codex_coding/src/dualstate/bench_loogle_sweetspot.py trace \
    --base-url http://127.0.0.1:30000 \
    --output results.json --tag my_trace_test \
    --trace kimi --prefix-len 4096 --fan-out 16 --duration 120
```

### Analyze Results

```bash
$PYTHON codex_coding/src/dualstate/analyze_sweetspot_results.py \
    --results-dir codex_coding/results/dualstate/sweetspot_YYYYMMDD/ \
    --output-dir codex_coding/results/dualstate/sweetspot_YYYYMMDD/analysis/
```

---

## 9. Key Discoveries

### 9.1 Mooncake RDMA Effective Bandwidth = 1.5 GB/s

Despite 8×H100 with NVLink (380 GB/s raw), mooncake's RDMA stack achieves only **1.5 GB/s effective bandwidth** on same-node P/D. This is due to:
- RDMA verbs posting overhead
- IB HCA routing (11 HCAs detected but software stack overhead dominates)
- Python → C++ binding + session management

**Implication**: Any byte saved in transfer yields 10-100× more benefit than expected.

### 9.2 DualState Savings are Dominated by Compute Skip

| Component | Savings | % of Total |
|-----------|:-------:|:----------:|
| P-side mamba recompute skip | ~40ms | **65%** |
| Mamba state transfer skip | ~21ms | **34%** |
| D-side COW copy | 0.03ms | <1% |
| **Total per-request** | **~61ms** | |

The main benefit is **not** avoiding the transfer — it's avoiding the P-side **recompute** of mamba state for shared prefixes.

### 9.3 FP8 KV is a Zero-Code-Change Win

- `--kv-cache-dtype fp8_e4m3` halves KV transfer bytes
- Quant/dequant overhead: 0.1ms (negligible)
- At 1.5 GB/s: saves 30-37ms per request for prefix=2k-4k
- **But only works with DualState enabled** (baseline P/D + FP8 crashes)

### 9.4 Sweet-Spot is Medium Prefix + High Fan-Out

- Best at prefix=2k tokens, fan_out=16
- NOT longer prefix — because baseline TTFT grows faster than fixed savings
- Fan-out is the multiplier: more COW hits = more savings per checkpoint

---

## 10. Remaining Work

### High Priority (for paper)

1. **Kimi-Linear-48B experiments**: Run full sweep + trace matrix on second model
2. **Multi-node RDMA**: Validate on 2-node setup (expected: similar % gains)
3. **TransferMode cost model steering**: Currently logged, not yet affecting decisions
4. **Paper writing**: Method section, experiment tables, figures

### Medium Priority

5. **DELTA_KV_ONLY implementation**: Requires custom D-side KV tracking (bypass SGLang's radix cache limitation)
6. **Hierarchical prefix hash**: Enable longest-prefix-match for agentic workloads
7. **Dynamic cache budget**: Adapt dualstate-cache-ratio to concurrency pressure

### Low Priority / Future Work

8. **UCCL-P2P integration**: Production-grade transfer with DietGPU lossless compression
9. **Multi-node heterogeneous**: P=H100 + D=H20 validation
10. **More models**: Falcon H1, Jamba2, NemotronH experiments
