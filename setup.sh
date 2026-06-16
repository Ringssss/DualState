#!/bin/bash
# DualState One-Click Setup & Run
# Usage:
#   git clone git@github.com:Ringssss/DualState.git
#   cd DualState
#   bash setup.sh                    # Apply patches to SGLang
#   bash run.sh qwen baseline       # Run Qwen3.6 baseline
#   bash run.sh qwen dualstate      # Run Qwen3.6 DualState
#   bash run.sh qwen ds_fp8         # Run Qwen3.6 DualState+FP8
#   bash run.sh kimi baseline       # Run Kimi-Linear baseline
#   bash run.sh kimi dualstate      # Run Kimi-Linear DualState

set -e

SGLANG_DIR="${SGLANG_DIR:-/home/zhujianian/sglang}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "╔════════════════════════════════════════════════════╗"
echo "║  DualState Setup — Applying patches to SGLang     ║"
echo "║  SGLang dir: $SGLANG_DIR"
echo "╚════════════════════════════════════════════════════╝"

# Verify SGLang exists
if [ ! -f "$SGLANG_DIR/python/sglang/srt/server_args.py" ]; then
    echo "ERROR: SGLang not found at $SGLANG_DIR"
    echo "Set SGLANG_DIR env var to your SGLang editable install path."
    echo "Example: SGLANG_DIR=/path/to/sglang bash setup.sh"
    exit 1
fi

# 1. Copy new files
echo "[1/3] Copying new DualState files..."

cp "$REPO_DIR/sglang_patches/disaggregation/dualstate_scheduler.py" \
   "$SGLANG_DIR/python/sglang/srt/disaggregation/"

cp "$REPO_DIR/sglang_patches/disaggregation/dualstate_coherence.py" \
   "$SGLANG_DIR/python/sglang/srt/disaggregation/"

cp "$REPO_DIR/sglang_patches/disaggregation/kv_compress.py" \
   "$SGLANG_DIR/python/sglang/srt/disaggregation/common/" 2>/dev/null || true

cp "$REPO_DIR/sglang_patches/mem_cache/checkpoint_availability_map.py" \
   "$SGLANG_DIR/python/sglang/srt/mem_cache/"

cp "$REPO_DIR/sglang_patches/mem_cache/mamba_radix_trace.py" \
   "$SGLANG_DIR/python/sglang/srt/mem_cache/"

echo "  ✓ 5 new files copied"

# 2. Apply patches
echo "[2/3] Applying patches to existing SGLang files..."

cd "$SGLANG_DIR"
PATCHED=0
FAILED=0

for patch in server_args scheduler decode mooncake_conn mamba_radix_cache; do
    PATCH_FILE="$REPO_DIR/sglang_patches/${patch}.patch"
    if [ -s "$PATCH_FILE" ]; then
        if git apply --check "$PATCH_FILE" 2>/dev/null; then
            git apply "$PATCH_FILE"
            echo "  ✓ ${patch}.patch applied"
            PATCHED=$((PATCHED + 1))
        else
            echo "  ⚠ ${patch}.patch already applied or conflicts — skipping"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  - ${patch}.patch is empty — skipping"
    fi
done

echo "  Applied: $PATCHED, Skipped: $FAILED"

# 3. Copy benchmark tools
echo "[3/3] Installing benchmark tools..."

mkdir -p "$SGLANG_DIR/codex_coding/src/dualstate"
cp "$REPO_DIR/benchmarks/"*.py "$SGLANG_DIR/codex_coding/src/dualstate/"
cp "$REPO_DIR/scripts/"*.sh "$SGLANG_DIR/codex_coding/src/dualstate/"
chmod +x "$SGLANG_DIR/codex_coding/src/dualstate/"*.sh

echo "  ✓ Benchmarks and scripts installed"

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  Setup complete!                                  ║"
echo "║                                                   ║"
echo "║  Next: bash run.sh qwen baseline                  ║"
echo "║        bash run.sh qwen dualstate                 ║"
echo "║        bash run.sh qwen ds_fp8                    ║"
echo "║        bash run.sh kimi baseline                  ║"
echo "║        bash run.sh kimi dualstate                 ║"
echo "╚════════════════════════════════════════════════════╝"
