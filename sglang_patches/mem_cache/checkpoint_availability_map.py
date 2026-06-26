"""
D-side Checkpoint Availability Map (CAM) for DualState serving.

Unlike the P-side radix tree:
- No tree structure → no split → no tombstone problem
- O(1) hash lookup instead of O(prefix_len) tree traversal
- Tracks coherence state and access frequency per entry
"""

import hashlib
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch


class CoherenceState(Enum):
    INVALID = "I"
    CLEAN = "C"
    DIRTY = "D"


@dataclass
class CheckpointEntry:
    prefix_hash: str
    prefix_len: int
    mamba_slot_indices: torch.Tensor
    coherence_state: CoherenceState = CoherenceState.CLEAN
    ema_frequency: float = 1.0
    last_access_time: float = 0.0
    ref_count: int = 0
    locked: bool = False
    creation_time: float = 0.0

    @property
    def is_evictable(self) -> bool:
        return (
            self.ref_count == 0
            and not self.locked
            and self.coherence_state == CoherenceState.CLEAN
        )


class CheckpointAvailabilityMap:
    """
    D-side checkpoint registry. Maps prefix hash → mamba checkpoint entry.

    Interacts with MambaPool for slot allocation/deallocation.
    """

    EMA_DECAY_LAMBDA = 0.1

    def __init__(self, mamba_pool, max_cache_slots: int, total_pool_size: int):
        self._registry: Dict[str, CheckpointEntry] = {}
        self._mamba_pool = mamba_pool
        self._max_cache_slots = max_cache_slots
        self._total_pool_size = total_pool_size
        self._cached_prefix_lens: set = set()  # Track all cached prefix lengths for boundary lookup

    def lookup(self, prefix_hash: str) -> Optional[CheckpointEntry]:
        entry = self._registry.get(prefix_hash)
        if entry is not None and entry.coherence_state == CoherenceState.CLEAN:
            return entry
        return None

    def fork_and_cache(
        self,
        prefix_hash: str,
        prefix_len: int,
        src_mamba_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Fork: copy mamba state from src indices to new cached slots.

        Called BEFORE decode starts overwriting src slots.

        Returns:
            Cached slot indices tensor, or None if allocation failed.
        """
        if len(self._registry) >= self._max_cache_slots:
            evicted = self._evict_one()
            if evicted is None:
                return None

        need_size = src_mamba_indices.numel()
        new_slots = self._mamba_pool.alloc(need_size)
        if new_slots is None:
            return None

        # Ensure both src and dst are the same tensor shape for copy_from
        src = src_mamba_indices.reshape(-1)
        dst = new_slots.reshape(-1)
        self._mamba_pool.copy_from(
            src_indices=src,
            dst_indices=dst,
        )

        now = time.monotonic()
        entry = CheckpointEntry(
            prefix_hash=prefix_hash,
            prefix_len=prefix_len,
            mamba_slot_indices=new_slots.clone(),
            coherence_state=CoherenceState.CLEAN,
            ema_frequency=1.0,
            last_access_time=now,
            creation_time=now,
        )
        self._registry[prefix_hash] = entry
        self._cached_prefix_lens.add(prefix_len)
        return new_slots

    def cow_from_cache(self, prefix_hash: str) -> Optional[torch.Tensor]:
        """
        Copy-on-Write: create working copy from cached checkpoint.

        Returns new mamba slot indices with copied state, or None.
        """
        entry = self._registry.get(prefix_hash)
        if entry is None or entry.coherence_state != CoherenceState.CLEAN:
            return None

        need_size = entry.mamba_slot_indices.numel()
        work_slots = self._mamba_pool.alloc(need_size)
        if work_slots is None:
            return None

        entry.ref_count += 1
        self._mamba_pool.copy_from(
            src_indices=entry.mamba_slot_indices,
            dst_indices=work_slots,
        )
        entry.ref_count -= 1
        self._record_access(entry)

        return work_slots

    def release_cow_ref(self, prefix_hash: str):
        entry = self._registry.get(prefix_hash)
        if entry and entry.ref_count > 0:
            entry.ref_count -= 1

    def lock_entry(self, prefix_hash: str) -> bool:
        entry = self._registry.get(prefix_hash)
        if entry and entry.coherence_state == CoherenceState.CLEAN:
            entry.locked = True
            return True
        return False

    def unlock_entry(self, prefix_hash: str):
        entry = self._registry.get(prefix_hash)
        if entry:
            entry.locked = False

    def get_cache_status_report(self, prefix_hash: str) -> dict:
        entry = self._registry.get(prefix_hash)
        if entry and entry.coherence_state == CoherenceState.CLEAN:
            return {
                "cached_prefix_hash": prefix_hash,
                "cached_prefix_len": entry.prefix_len,
                "d_checkpoint_state": "C",
            }
        return {
            "cached_prefix_hash": None,
            "cached_prefix_len": 0,
            "d_checkpoint_state": "I",
        }

    def _evict_one(self) -> Optional[str]:
        """Evict lowest-frequency evictable entry (LFU)."""
        candidates = [
            (ph, e) for ph, e in self._registry.items() if e.is_evictable
        ]
        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1].ema_frequency)
        victim_hash, victim_entry = candidates[0]

        self._mamba_pool.free(victim_entry.mamba_slot_indices)
        # Remove prefix_len from tracked set if no other entry has same length
        if not any(e.prefix_len == victim_entry.prefix_len for e in self._registry.values() if e is not victim_entry):
            self._cached_prefix_lens.discard(victim_entry.prefix_len)
        del self._registry[victim_hash]
        return victim_hash

    def evict_by_hash(self, prefix_hash: str) -> bool:
        entry = self._registry.get(prefix_hash)
        if entry and entry.is_evictable:
            self._mamba_pool.free(entry.mamba_slot_indices)
            del self._registry[prefix_hash]
            return True
        return False

    def _record_access(self, entry: CheckpointEntry):
        now = time.monotonic()
        dt = now - entry.last_access_time
        decay = math.exp(-self.EMA_DECAY_LAMBDA * dt)
        entry.ema_frequency = entry.ema_frequency * decay + 1.0
        entry.last_access_time = now

    def record_access(self, prefix_hash: str):
        entry = self._registry.get(prefix_hash)
        if entry:
            self._record_access(entry)

    @property
    def size(self) -> int:
        return len(self._registry)

    @property
    def available_cache_slots(self) -> int:
        return max(0, self._max_cache_slots - len(self._registry))

    def contains(self, prefix_hash: str) -> bool:
        entry = self._registry.get(prefix_hash)
        return entry is not None and entry.coherence_state == CoherenceState.CLEAN

    def lookup_longest_prefix(
        self, token_ids, granularity: int = 64
    ) -> Optional[CheckpointEntry]:
        """
        Hierarchical prefix match: find the longest cached prefix that is
        a TRUE prefix of the given token_ids.

        This enables agentic/multi-turn workloads where requests share a
        growing prefix. E.g., if we cached mamba state for tokens[:200],
        and a new request has tokens[:300], we can COW from the 200-token
        checkpoint (the mamba state is valid for the first 200 tokens).

        The COW'd state is valid for the first `entry.prefix_len` tokens.
        The caller (decode.py) must inform P-side to only compute mamba
        for tokens[entry.prefix_len:] (delta mamba computation).

        Args:
            token_ids: Full token sequence of the new request
            granularity: Step size for prefix length probing
        Returns:
            Best matching CheckpointEntry, or None
        """
        from sglang.srt.mem_cache.checkpoint_availability_map import (
            compute_prefix_hash_at_length,
        )

        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        total_len = len(token_ids)

        # Exact match first (fastest, most common case)
        exact_hash = compute_prefix_hash_at_length(token_ids, total_len)
        exact = self.lookup(exact_hash)
        if exact is not None:
            return exact

        # Probe shorter prefixes at CACHED lengths (longest first)
        # Only check lengths that actually exist in CAM
        candidate_lens = sorted(
            (pl for pl in self._cached_prefix_lens if pl < total_len),
            reverse=True,
        )
        best_entry = None
        for probe_len in candidate_lens:
            probe_hash = compute_prefix_hash_at_length(token_ids, probe_len)
            entry = self.lookup(probe_hash)
            if entry is not None:
                best_entry = entry
                break  # longest match wins

        return best_entry

    def get_all_entries(self) -> Dict[str, CheckpointEntry]:
        return dict(self._registry)

    def clear(self):
        for entry in self._registry.values():
            self._mamba_pool.free(entry.mamba_slot_indices)
        self._registry.clear()


def compute_prefix_hash(token_ids, max_tokens: int = 128) -> str:
    """
    Compute a content-and-length hash for a token prefix.

    IMPORTANT: The hash must uniquely identify the COMPLETE token sequence
    that produced the mamba state. Two sequences with the same prefix but
    different lengths produce different mamba states (recurrent computation
    depends on ALL tokens). We include the total length in the hash to
    prevent false matches between different-length sequences.
    """
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    total_len = len(token_ids)
    # Use first+last tokens + total length for efficient but safe hashing
    prefix = token_ids[:max_tokens]
    suffix = token_ids[-min(32, total_len):] if total_len > max_tokens else []
    raw = bytes(f"{total_len}:{prefix}:{suffix}", "utf-8")
    return hashlib.md5(raw).hexdigest()[:16]


def compute_prefix_hash_at_length(token_ids, length: int) -> str:
    """
    Compute hash for the first `length` tokens of a sequence.
    Used for hierarchical prefix matching: find the longest cached
    prefix that is a true prefix of this request.
    """
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    sub = token_ids[:length]
    raw = bytes(f"{length}:{sub}", "utf-8")
    return hashlib.md5(raw).hexdigest()[:16]
