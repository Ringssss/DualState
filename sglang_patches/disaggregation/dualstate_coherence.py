"""
DualState Coherence Manager: coordinates P and D sides for mamba transfer skip.

For single-node PD (same machine, different GPUs), this module provides
a shared-memory mechanism for D to report its CAM status and P to query it
before deciding to transfer mamba state.

For multi-node PD (future), this would be replaced by ZMQ protocol extension.
"""

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DualStateCoherenceManager:
    """
    Shared coherence state between P-side and D-side.
    Thread-safe for concurrent P/D access on single node.
    """

    _instance: Optional["DualStateCoherenceManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._d_cache_status: Dict[str, dict] = {}
        self._status_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "DualStateCoherenceManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def d_report_cache_status(self, prefix_hash: str, prefix_len: int, state: str):
        """D-side reports it has (or lost) a cached checkpoint."""
        with self._status_lock:
            if state == "C":
                self._d_cache_status[prefix_hash] = {
                    "prefix_len": prefix_len,
                    "state": state,
                }
            else:
                self._d_cache_status.pop(prefix_hash, None)

    def p_query_d_status(self, prefix_hash: str) -> dict:
        """P-side queries D's cache status for a prefix."""
        with self._status_lock:
            entry = self._d_cache_status.get(prefix_hash)
            if entry and entry["state"] == "C":
                return {
                    "cached_prefix_hash": prefix_hash,
                    "cached_prefix_len": entry["prefix_len"],
                    "d_checkpoint_state": "C",
                }
        return {
            "cached_prefix_hash": None,
            "cached_prefix_len": 0,
            "d_checkpoint_state": "I",
        }

    def d_remove_entry(self, prefix_hash: str):
        """D-side removes entry (eviction)."""
        with self._status_lock:
            self._d_cache_status.pop(prefix_hash, None)

    def clear(self):
        with self._status_lock:
            self._d_cache_status.clear()

    @property
    def size(self) -> int:
        with self._status_lock:
            return len(self._d_cache_status)
