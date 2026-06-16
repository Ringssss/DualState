# StateGate 实验记录 — 2026-05-28

## 1. 项目概述

**研究题目**：State-Gated Prefix Reuse in Hybrid Model Serving

**核心假设**：在混合模型（交替 full-attention 和 linear/Mamba 层）的 prefix cache 中，KV cache 的结构性 prefix match 被 Mamba state checkpoint 的缺失所门控，导致 effective reuse 显著低于 structural reuse。

**研究目标**：量化这个 gap，理解其来源，设计更高效的 checkpoint placement 策略。

---

## 2. 实验环境搭建

### 2.1 硬件环境

| 项 | 配置 |
|---|------|
| GPU | 8 × NVIDIA H100 80GB HBM3 |
| 互联 | NVLink (intra-node) |
| 机器 | 单节点 |

### 2.2 SGLang Editable 安装

```bash
# 1. 激活 conda 环境
conda activate sglang-bench

# 2. 安装 Rust（sglang gRPC 扩展需要）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 3. 设置 protoc 路径
export PROTOC=$(find $CONDA_PREFIX -name protoc -type f | head -1)

# 4. Editable 安装
pip install -e /home/zhujianian/sglang/python

# 5. 清除残留的 site-packages/sglang/ 目录（旧安装遗留，会覆盖 editable）
rm -rf /home/zhujianian/miniconda3/envs/sglang-bench/lib/python3.11/site-packages/sglang/

# 6. 验证
python -c "import sglang; print(sglang.__path__)"
# 应输出：['/home/zhujianian/sglang/python/sglang']
```

安装结果：
- SGLang 版本：0.5.12.post2 (editable)
- Python 3.11
- 源码路径：`/home/zhujianian/sglang/python/sglang/`

### 2.3 可用模型

| 模型 | 路径 | 类型 | TP |
|------|------|------|-----|
| Qwen3.6-35B-A3B | `/mnt/models/Qwen3.6-35B-A3B` | qwen3_5_moe, 40层, 256 experts, full_attention_interval=4 | 2 |
| Kimi-Linear-48B-A3B-Instruct | `/mnt/models/Kimi-Linear-48B-A3B-Instruct` | kimi_linear, 27层, 256 experts | 2 |
| Qwen3-8B | `/mnt/models/Qwen3-8B` | full-attention | 1 |

### 2.4 启动 Server 命令（基础版）

```bash
conda activate sglang-bench

# 基础启动
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path /mnt/models/Qwen3.6-35B-A3B \
  --port 30000 --host 0.0.0.0 --tp 2

# 带 trace 启动
SGLANG_MAMBA_RADIX_TRACE=1 \
SGLANG_MAMBA_RADIX_TRACE_FILE=/home/zhujianian/mamba_radix_trace.jsonl \
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path /mnt/models/Qwen3.6-35B-A3B \
  --port 30000 --host 0.0.0.0 --tp 2
```

**注意事项**：
- 首次启动 JIT 编译 kernel 需要 1-2 分钟，第一个请求会超时，需 `--max-time 120`
- 环境有 http_proxy 设置，启动前需 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`
- 单卡 80G 不够装 Qwen3.6（hybrid mamba state cache 需额外显存），必须 TP=2

---

## 3. Phase 1：Instrumentation（MambaRadixCache 插桩）

### 3.1 实现概述

在 SGLang 的 `MambaRadixCache` 中添加零开销 JSONL tracer，记录 prefix cache 的 match/split/insert/evict 行为。

**新建文件**：
- `python/sglang/srt/mem_cache/mamba_radix_trace.py` — tracer 模块

**修改文件**：
- `python/sglang/srt/mem_cache/mamba_radix_cache.py` — 在关键路径插入 trace 调用

### 3.2 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SGLANG_MAMBA_RADIX_TRACE` | `0` | 设为 `1` 启用 trace |
| `SGLANG_MAMBA_RADIX_TRACE_FILE` | `/tmp/sglang_mamba_radix_trace.jsonl` | trace 输出路径 |
| `SGLANG_MAMBA_RADIX_TRACE_SAMPLE_RATE` | `1.0` | 采样率 0.0-1.0 |

### 3.3 插桩点

| 方法 | 记录内容 |
|------|---------|
| `_match_prefix_helper` | structural_match_len, state_checkpoint_match_len, effective_match_len, gated_match_loss, num_tombstone_nodes_on_path |
| `_match_post_processor` | cow_mamba_triggered, mamba_branching_seqlen, cache pressure metrics |
| `_split_node` | split_created_mamba_tombstone, new_node_id, child_had_mamba_value |
| `_insert_helper` | mamba_checkpoint_inserted, mamba_checkpoint_restored_tombstone, mamba_value_already_existed |
| `evict_mamba` | mamba_slots_evicted, was_internal_tombstone |
| `_evict_leaf_node` | kv_tokens_evicted |

### 3.4 JSONL 记录格式（match 事件示例）

```json
{
  "event": "match",
  "timestamp": 1779907390.966,
  "tp_rank": 0,
  "dp_rank": 0,
  "pid": 1695345,
  "server_role": "colocated",
  "model": "/mnt/models/Qwen3.6-35B-A3B",
  "input_len": 619,
  "structural_match_len": 619,
  "state_checkpoint_match_len": 619,
  "effective_match_len": 619,
  "gated_match_loss": 0,
  "num_traversed_nodes": 2,
  "num_traversed_nodes_with_mamba": 1,
  "num_tombstone_nodes_on_path": 1,
  "last_node_id": 37,
  "best_mamba_node_id": 37,
  "split_happened": false,
  "page_size": 1,
  "enable_mamba_extra_buffer": false,
  "mamba_cache_chunk_size": 64,
  "full_evictable_size": 634,
  "mamba_evictable_size": 3,
  "key_hash": "217c869f10c8"
}
```

### 3.5 核心指标定义

| 指标 | 定义 | 含义 |
|------|------|------|
| `structural_match_len` | radix tree 上 KV 结构可匹配的最长 prefix 长度 | KV cache 侧能做到的最大复用 |
| `state_checkpoint_match_len` | 有 mamba_value（state checkpoint）的最长 prefix 长度 | Mamba state 侧能做到的最大复用 |
| `effective_match_len` | 实际返回的可复用 prefix 长度 | 系统最终使用的复用长度 |
| `gated_match_loss` | structural - effective | 被 state 门控浪费的 token 数 |
| `split_tombstone_frac` | split 创建 mamba_value=None 的比例 | split 操作产生 tombstone 的频率 |

---

## 4. Phase 2：Benchmark Workload Harness

### 4.1 新建文件

- `benchmark/bench_hybrid_prefix_state_gate.py` — 8 种 workload 的 benchmark 生成器
- `benchmark/analyze_mamba_radix_trace.py` — JSONL trace 分析器（3 级视图）

### 4.2 Workload 设计

| Workload | 设计意图 | prefix 共享模式 |
|----------|---------|---------------|
| `random_no_share` | 负对照：完全不同 prompt | 无共享 |
| `identical_prompt` | 正对照：完全相同 prompt | 100% 共享 |
| `shared_system_prompt` | 共享 system prompt，不同 user query | system prompt 共享 |
| `shared_document_qa` | 共享长文档，不同问题 | document 共享 |
| `multi_turn_agent` | 共享 tool schema + 部分对话历史 | 历史前缀共享 |
| `adversarial_near_prefix` | 几乎相同但在尾部 branch | 深层 branch |
| `prefix_ladder` | 不同深度 prefix 阶梯式共享 | 阶梯式 |
| `branch_after_cached_leaf` | 先缓存完整请求，再发同 document 不同 question | document 共享，question branch |

### 4.3 运行命令

```bash
# 运行所有 workload
python benchmark/bench_hybrid_prefix_state_gate.py \
  --server-url http://localhost:30000 \
  --workload all \
  --num-requests 10 \
  --prompt-words 500 \
  --output-dir results/run1

# 运行单个 workload
python benchmark/bench_hybrid_prefix_state_gate.py \
  --server-url http://localhost:30000 \
  --workload branch_after_cached_leaf \
  --num-requests 10 \
  --prompt-words 500

# 分析 trace
python benchmark/analyze_mamba_radix_trace.py \
  /home/zhujianian/mamba_radix_trace.jsonl \
  --output results/run1/trace_summary.csv \
  --json-output results/run1/trace_summary.json
```

### 4.4 Analyzer 三级视图

| 视图 | 含义 | 用途 |
|------|------|------|
| `event_level` | 所有 trace 事件（TP 多 rank 会重复） | 原始数据全貌 |
| `rank0_only` | 仅 tp_rank=0 的事件 | 消除 TP 重复计数 |
| `request_level` | 按 (key_hash, timestamp_bucket) 去重 | 近似 per-request 统计 |

---

## 5. Smoke Test 验证

### 5.1 运行命令

```bash
# 启动 server (带 trace)
SGLANG_MAMBA_RADIX_TRACE=1 \
SGLANG_MAMBA_RADIX_TRACE_FILE=/home/zhujianian/mamba_radix_trace.jsonl \
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path /mnt/models/Qwen3.6-35B-A3B \
  --port 30001 --host 0.0.0.0 --tp 2

# 发送测试请求
curl --noproxy localhost -s --max-time 120 \
  http://localhost:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/models/Qwen3.6-35B-A3B","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'

# 运行 benchmark
python benchmark/bench_hybrid_prefix_state_gate.py \
  --server-url http://localhost:30001 \
  --workload branch_after_cached_leaf \
  --num-requests 5 --prompt-words 500 \
  --output-dir results/smoke_test_v2

# 分析 trace
python benchmark/analyze_mamba_radix_trace.py \
  /home/zhujianian/mamba_radix_trace.jsonl \
  --output results/smoke_test_v2/trace_summary.csv
```

### 5.2 Smoke Test 结果 (branch_after_cached_leaf, default no_buffer)

| 指标 | 值 |
|------|-----|
| match_count | 24 |
| structural_match_len_avg | 458.0 tokens |
| state_checkpoint_match_len_avg | 259.4 tokens |
| effective_match_len_avg | 259.4 tokens |
| **gated_match_loss_avg** | **198.6 tokens** |
| frac_structural_gt_state_cp | 41.7% |
| frac_zero_effective_with_structural_hit | 41.7% |
| split_tombstone_frac | 100% |
| tombstone_on_path_avg | 1.5 |
| split_count | 10 |
| mamba_eviction_events | 0 |
| kv_eviction_events | 0 |

**结论**：核心假设初步成立。平均 198.6 tokens 的 KV 结构命中被 state checkpoint 门控浪费。

---

## 6. Phase 3-mini：参数扫描

### 6.1 运行命令

```bash
# 完整 sweep（5 configs × 4 workloads）
bash scripts/run_hybrid_prefix_sweep.sh

# 自定义参数
MODEL_PATH=/mnt/models/Kimi-Linear-48B-A3B-Instruct \
NUM_REQUESTS=20 \
PROMPT_WORDS=1000 \
bash scripts/run_hybrid_prefix_sweep.sh
```

### 6.2 Sweep 配置

| 配置 ID | 名称 | Server 参数 |
|---------|------|------------|
| A | default_no_buffer | (默认) |
| B | extra_buffer | `--mamba-scheduler-strategy extra_buffer` |
| C | ratio_0.5 | `--mamba-full-memory-ratio 0.5` |
| D | ratio_0.9 | `--mamba-full-memory-ratio 0.9` |
| E | disable_cache | `--disable-radix-cache` |

每个配置的完整启动命令（以 Config B 为例）：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
SGLANG_MAMBA_RADIX_TRACE=1 \
SGLANG_MAMBA_RADIX_TRACE_FILE=trace.jsonl \
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path /mnt/models/Qwen3.6-35B-A3B \
  --port 30100 --host 0.0.0.0 --tp 2 \
  --mamba-scheduler-strategy extra_buffer
```

### 6.3 GPU 显存使用

| 配置 | GPU 0 (MiB) | GPU 1 (MiB) | 总占用 |
|------|-------------|-------------|--------|
| A. default | 65,292 | 64,652 | 129,944 |
| B. extra_buffer | 64,970 | 64,330 | 129,300 |
| C. ratio=0.5 | 65,088 | 64,448 | 129,536 |
| D. ratio=0.9 | 65,292 | 64,652 | 129,944 |
| E. disable cache | 66,050 | 65,410 | 131,460 |

### 6.4 Sweep 完整结果表

#### 表 1：gated_match_loss_avg（被门控浪费的 token 数）

| Config | branch_leaf | identical | prefix_ladder | random |
|--------|:-----------:|:---------:|:-------------:|:------:|
| A. default | **271.6** | 2.4 | **150.2** | 90.6 |
| B. extra_buffer | **0.19** | 2.1 | **0.21** | — |
| C. ratio=0.5 | 257.7 | — | 153.3 | 33.4 |
| D. ratio=0.9 | 257.7 | — | 155.0 | 32.0 |
| E. disable cache | — | — | — | — |

#### 表 2：frac_structural_gt_state_cp（结构命中 > state checkpoint 的请求比例）

| Config | branch_leaf | identical | prefix_ladder | random |
|--------|:-----------:|:---------:|:-------------:|:------:|
| A. default | 48% | 53% | 50% | 38% |
| B. extra_buffer | **6.3%** | 48% | **7.1%** | — |
| C. ratio=0.5 | 50% | — | 49% | 52% |
| D. ratio=0.9 | 50% | — | 50% | 50% |

#### 表 3：structural_match_len_avg / effective_match_len_avg

| Config | Workload | structural | effective | loss_ratio |
|--------|----------|:----------:|:---------:|:----------:|
| A. default | branch_leaf | 592.5 | 320.9 | 45.8% |
| A. default | prefix_ladder | 311.0 | 160.8 | 48.3% |
| B. extra_buffer | branch_leaf | 540.2 | 540.0 | 0.03% |
| B. extra_buffer | prefix_ladder | 352.2 | 352.0 | 0.06% |
| C. ratio=0.5 | branch_leaf | 572.7 | 315.0 | 45.0% |
| D. ratio=0.9 | branch_leaf | 566.7 | 309.0 | 45.5% |

#### 表 4：TTFT p50 (秒)

| Config | branch_leaf | identical | prefix_ladder | random |
|--------|:-----------:|:---------:|:-------------:|:------:|
| A. default | 0.412 | 0.411 | 0.515 | 0.521 |
| B. extra_buffer | 0.471 | 0.349 | 0.368 | 0.369 |
| C. ratio=0.5 | 0.417 | 0.415 | 0.414 | 0.426 |
| D. ratio=0.9 | 0.415 | 0.424 | 0.409 | 0.419 |
| E. disable cache | 0.351 | 0.345 | 0.343 | 0.350 |

#### 表 5：其他关键指标

| Config | Workload | split_tombstone_frac | tombstone_on_path_avg | mamba_evictions | kv_evictions |
|--------|----------|:--------------------:|:---------------------:|:--------------:|:------------:|
| A. default | branch_leaf | 100% | 1.96 | 0 | 0 |
| A. default | prefix_ladder | 100% | 3.87 | 0 | 0 |
| B. extra_buffer | branch_leaf | — | 1.31 | 0 | 0 |
| B. extra_buffer | prefix_ladder | — | 2.00 | 0 | 0 |
| C. ratio=0.5 | branch_leaf | 100% | 1.88 | 0 | 0 |
| D. ratio=0.9 | branch_leaf | 100% | 1.88 | 0 | 0 |

---

## 7. 关键发现

### Finding 1：State-gated prefix reuse 现象确实存在且严重

- Default 配置下，branch workload 平均浪费 **271.6 tokens**（占 structural match 的 **46%**）
- **48%** 的请求有结构命中但 effective 更低
- 100% 的 split 操作创建了 mamba tombstone

### Finding 2：extra_buffer 几乎完全消除了 branch/ladder 的 gating

- gated_match_loss：271.6 → **0.19**（降低 **99.9%**）
- frac_gated：48% → **6.3%**
- 代价：每请求 2 个 mamba slot（ping-pong buffer）

### Finding 3：memory ratio 变化对 gated_match_loss 几乎无影响

- ratio=0.5 vs ratio=0.9：branch loss 257.7 vs 257.7（几乎相同）
- **结论**：问题本质是 **placement problem，不是 capacity problem**

### Finding 4：disable cache 的 TTFT 反而最低

- disable cache TTFT ~0.35s，default ~0.41s，extra_buffer ~0.37-0.47s
- 说明当前 cache lookup + state management 本身有开销
- prefix cache 的收益需要在 longer prefix / higher concurrency 下才能体现

### Finding 5：identical prompt 在所有配置下 loss 都很小（~2 tokens）

- 这是 split/edge 边界效应，不是 tombstone gating
- 说明 tombstone gating 主要发生在 branch 场景

---

## 8. 结果文件路径

```
/home/zhujianian/sglang/
├── results/
│   ├── smoke_test_v2/                    # Smoke test 结果
│   │   ├── requests.jsonl
│   │   ├── summary.json
│   │   ├── trace_summary.csv
│   │   └── trace_summary.json
│   └── sweep_20260528_034117/            # Phase 3-mini sweep 结果
│       ├── A_default_no_buffer/
│       │   ├── server.log
│       │   ├── gpu_memory.csv
│       │   ├── branch_after_cached_leaf/
│       │   │   ├── requests.jsonl
│       │   │   ├── summary.json
│       │   │   ├── trace.jsonl
│       │   │   ├── trace_summary.csv
│       │   │   └── trace_summary.json
│       │   ├── identical_prompt/
│       │   ├── prefix_ladder/
│       │   └── random_no_share/
│       ├── B_extra_buffer/
│       ├── C_ratio_0.5/
│       ├── D_ratio_0.9/
│       ├── E_disable_cache/
│       ├── sweep_summary.csv             # 汇总 CSV
│       └── sweep_summary.log             # 汇总表格
├── benchmark/
│   ├── bench_hybrid_prefix_state_gate.py # Benchmark workload 生成器
│   └── analyze_mamba_radix_trace.py      # Trace 分析器
├── scripts/
│   └── run_hybrid_prefix_sweep.sh        # Sweep 自动化脚本
└── python/sglang/srt/mem_cache/
    ├── mamba_radix_trace.py              # Tracer 模块（新建）
    └── mamba_radix_cache.py              # 已插桩（修改）
```

---

## 9. 方法设计方向

基于实验数据，论文方法设计如下：

### 核心问题

> Hybrid prefix caching is a joint checkpoint placement and transfer scheduling problem: structural KV hits are gated by atomic recurrent state availability, and the current solutions (no checkpoint vs always-double-buffer) represent two extremes of a cost-benefit spectrum.

### 三个挑战与三个方法

| Challenge | Method | Key Idea |
|-----------|--------|----------|
| C1: 无法量化 state-gating | M1: StateGate Profiler | per-request 诊断框架，定义 gated_match_loss |
| C2: checkpoint placement 是 all-or-nothing | M2: Demand-Driven Branch Checkpointing | split 时 opportunistic checkpoint + branch-protect eviction |
| C3: P/D 下 state transfer 是 atomic critical path | M3: Cost-Based State Routing | fetch vs recompute vs route 的 cost model |

### 预期收益

| 场景 | Default | Extra_buffer (2x mem) | Method 2 (预期) |
|------|---------|----------------------|-----------------|
| branch loss | 271.6 tokens | 0.2 tokens | 10-50 tokens |
| 内存开销 | 1x | 2x | 1.1-1.3x |
| loss 降低 | baseline | 99.9% | 60-90% |

---

## 10. 下一步计划

| 优先级 | 任务 | 状态 |
|--------|------|------|
| 1 | 实现 Method 2 (demand-driven branch checkpoint) | 待开始 |
| 2 | 用 sweep 框架验证 Method 2 | 待开始 |
| 3 | 单机 P/D 实验 + state transfer 测量 | 待开始 |
| 4 | Network cost model (Method 3) | 待开始 |
| 5 | Kimi-Linear-48B 模型复现 | 待开始 |
| 6 | 论文写作 | 待开始 |
