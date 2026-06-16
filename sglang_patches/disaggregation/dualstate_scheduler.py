"""
DualState Scheduler: Affine-Cost Cross-Node Cache Coordination.

Key insight: In PDH (PD + Hybrid Attention), transfer cost is:
    cost = beta * (kv_per_token * seq_len + MAMBA_STATE_SIZE)
where MAMBA_STATE_SIZE is a large constant (30.8MB for Qwen3.6-35B-A3B),
making cost affine rather than linear in seq_len.

This changes optimal caching: D-side eviction degenerates to pure LFU
because per-hit benefit is nearly constant across prefix lengths.
"""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.checkpoint_availability_map import (
        CheckpointAvailabilityMap,
    )


class TransferMode(Enum):
    FULL = "full"
    KV_MAMBA = "kv_mamba"
    KV_ONLY = "kv_only"
    DELTA_KV_ONLY = "delta_kv_only"
    P_RECOMPUTE_D_COW = "p_recompute_d_cow"


@dataclass
class CostModelParams:
    alpha_linear: float = 0.003
    alpha_full: float = 0.005
    beta: float = 1.0 / (10 * 1e9) * 1000  # ms per byte at 10 GB/s
    mamba_state_bytes: int = int(30.8 * 1024 * 1024)
    kv_per_token_bytes: int = 20480
    n_linear_layers: int = 30
    n_full_layers: int = 10


class DualStateCostModel:
    def __init__(self, params: CostModelParams):
        self.p = params

    def compute_cost(
        self,
        prefix_len: int,
        new_tokens: int,
        p_has_checkpoint: bool,
        d_state: str,
    ) -> Dict[TransferMode, float]:
        costs = {}
        total_tokens = prefix_len + new_tokens

        compute_full = (
            self.p.alpha_linear * total_tokens * self.p.n_linear_layers
            + self.p.alpha_full * total_tokens * self.p.n_full_layers
        )
        transfer_full = self.p.beta * (
            self.p.kv_per_token_bytes * total_tokens + self.p.mamba_state_bytes
        )
        costs[TransferMode.FULL] = compute_full + transfer_full

        if p_has_checkpoint:
            compute_delta = (
                self.p.alpha_linear * new_tokens * self.p.n_linear_layers
                + self.p.alpha_full * new_tokens * self.p.n_full_layers
            )
            transfer_kv_mamba = self.p.beta * (
                self.p.kv_per_token_bytes * new_tokens + self.p.mamba_state_bytes
            )
            costs[TransferMode.KV_MAMBA] = compute_delta + transfer_kv_mamba

            if d_state == "C":
                transfer_delta_kv = self.p.beta * (
                    self.p.kv_per_token_bytes * new_tokens
                )
                costs[TransferMode.DELTA_KV_ONLY] = compute_delta + transfer_delta_kv

        if not p_has_checkpoint and d_state == "C":
            compute_recompute = (
                self.p.alpha_linear * total_tokens * self.p.n_linear_layers
                + self.p.alpha_full * new_tokens * self.p.n_full_layers
            )
            transfer_kv_only = self.p.beta * (
                self.p.kv_per_token_bytes * total_tokens
            )
            costs[TransferMode.P_RECOMPUTE_D_COW] = (
                compute_recompute + transfer_kv_only
            )

        return costs

    def select_optimal_action(
        self,
        prefix_len: int,
        new_tokens: int,
        p_has_checkpoint: bool,
        d_state: str,
    ) -> TransferMode:
        costs = self.compute_cost(prefix_len, new_tokens, p_has_checkpoint, d_state)
        return min(costs, key=costs.get)

    def p_eviction_score(self, prefix_len: int, frequency: float) -> float:
        compute_saved = (
            self.p.alpha_linear * prefix_len * self.p.n_linear_layers
            + self.p.alpha_full * prefix_len * self.p.n_full_layers
        )
        return frequency * compute_saved

    def d_eviction_score(self, prefix_len: int, frequency: float) -> float:
        transfer_saved = self.p.beta * (
            self.p.kv_per_token_bytes * prefix_len + self.p.mamba_state_bytes
        )
        return frequency * transfer_saved


class DualStateScheduler:
    THRESHOLD_LOW = 0.0
    THRESHOLD_MEDIUM = 1.0
    THRESHOLD_HIGH = 3.0

    def __init__(self, cost_model: DualStateCostModel):
        self.cost_model = cost_model
        self._global_access_tracker: Dict[str, float] = {}

    def decide_action(
        self,
        prefix_len: int,
        new_tokens: int,
        p_has_checkpoint: bool,
        d_state: str,
    ) -> TransferMode:
        return self.cost_model.select_optimal_action(
            prefix_len, new_tokens, p_has_checkpoint, d_state
        )

    def should_cache(
        self,
        prefix_hash: str,
        prefix_len: int,
        cam: "CheckpointAvailabilityMap",
    ) -> bool:
        if cam.contains(prefix_hash):
            return False

        # Record access FIRST, then check frequency
        self._record_global_access(prefix_hash)
        freq = self._global_access_tracker.get(prefix_hash, 0)

        threshold = self._get_admission_threshold(cam)
        if freq < threshold:
            return False

        if cam.available_cache_slots > 0:
            return True

        all_entries = cam.get_all_entries()
        if not all_entries:
            return True
        min_cached_freq = min(e.ema_frequency for e in all_entries.values())
        return freq > min_cached_freq

    def coordinate_p_eviction(
        self,
        prefix_hash: str,
        cam: "CheckpointAvailabilityMap",
    ) -> bool:
        from sglang.srt.mem_cache.checkpoint_availability_map import CoherenceState

        entry = cam.lookup(prefix_hash)
        if entry is not None and entry.coherence_state == CoherenceState.CLEAN:
            return True
        return False

    def _get_admission_threshold(self, cam: "CheckpointAvailabilityMap") -> float:
        if cam._max_cache_slots <= 0:
            return self.THRESHOLD_HIGH
        ratio = cam.available_cache_slots / cam._max_cache_slots
        if ratio > 0.7:
            return self.THRESHOLD_LOW
        elif ratio > 0.3:
            return self.THRESHOLD_MEDIUM
        else:
            return self.THRESHOLD_HIGH

    def _record_global_access(self, prefix_hash: str):
        current = self._global_access_tracker.get(prefix_hash, 0)
        self._global_access_tracker[prefix_hash] = current + 1

    @staticmethod
    def from_model_config(model_config: dict) -> "DualStateScheduler":
        params = CostModelParams()
        text_cfg = model_config.get("text_config", model_config)
        layer_types = text_cfg.get("layer_types", [])
        if layer_types:
            params.n_linear_layers = sum(
                1 for t in layer_types if "linear" in t.lower()
            )
            params.n_full_layers = sum(
                1 for t in layer_types if "full" in t.lower()
            )

        num_kv_heads = text_cfg.get("num_key_value_heads", 2)
        head_dim = text_cfg.get("head_dim", 256)
        if params.n_full_layers > 0:
            params.kv_per_token_bytes = (
                num_kv_heads * head_dim * 2 * 2 * params.n_full_layers
            )

        cost_model = DualStateCostModel(params)
        return DualStateScheduler(cost_model)
