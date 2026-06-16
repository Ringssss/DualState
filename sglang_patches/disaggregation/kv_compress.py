"""
KV Cache Transfer Compression: BF16 → FP8 online quantization.

Simple approach: compress KV data in-place in the source pool BEFORE transfer,
transfer the compressed (half-size) data, then decompress on D-side AFTER receive.

This works by:
1. P-side: quantize bf16 → fp8 into a staging buffer, transfer fp8 + scale
2. D-side: receive fp8, dequantize back to bf16 into the KV pool

Since P and D KV pools are pre-registered with mooncake, we use a separate
staging buffer approach: P copies bf16→fp8 into staging, transfers staging→D_staging,
D copies fp8→bf16 from D_staging into pool.

Enable: SGLANG_DISAGG_KV_COMPRESS=1 on both P and D sides.
"""

import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

FP8_MAX = 448.0
MIN_COMPRESS_ELEMENTS = 1024 * 1024  # ~2MB in bf16

_enabled: Optional[bool] = None


def is_kv_compress_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = os.environ.get("SGLANG_DISAGG_KV_COMPRESS", "0") == "1"
        if _enabled:
            logger.info("KV FP8 transfer compression enabled")
    return _enabled


@torch.no_grad()
def quantize_kv_inplace(kv_bf16: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize bf16 KV tensor to fp8 with per-tensor scale. Returns (fp8, scale)."""
    amax = kv_bf16.abs().amax()
    scale = float((FP8_MAX / amax.clamp(min=1e-12)).clamp(max=1e4).item())
    fp8 = (kv_bf16 * scale).to(torch.float8_e4m3fn)
    return fp8, scale


@torch.no_grad()
def dequantize_kv_inplace(fp8: torch.Tensor, scale: float, dst_bf16: torch.Tensor):
    """Dequantize fp8 back to bf16 into dst tensor."""
    dst_bf16.copy_(fp8.to(dst_bf16.dtype) * (1.0 / scale))
