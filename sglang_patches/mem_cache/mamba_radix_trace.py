"""
Lightweight JSONL tracer for MambaRadixCache instrumentation.

Gated by environment variables:
  SGLANG_MAMBA_RADIX_TRACE=1           Enable tracing
  SGLANG_MAMBA_RADIX_TRACE_FILE=path   Output file (default: /tmp/sglang_mamba_radix_trace.jsonl)
  SGLANG_MAMBA_RADIX_TRACE_SAMPLE_RATE Sample rate 0.0-1.0 (default: 1.0)
"""

from __future__ import annotations

import atexit
import json
import os
import random
import threading
import time
from typing import Any, Dict, Optional


_ENABLED = os.environ.get("SGLANG_MAMBA_RADIX_TRACE", "0") == "1"
_TRACE_FILE = os.environ.get(
    "SGLANG_MAMBA_RADIX_TRACE_FILE", "/tmp/sglang_mamba_radix_trace.jsonl"
)
_SAMPLE_RATE = float(os.environ.get("SGLANG_MAMBA_RADIX_TRACE_SAMPLE_RATE", "1.0"))

_FLUSH_INTERVAL = 5.0
_FLUSH_THRESHOLD = 100


class _TraceWriter:
    def __init__(self, path: str):
        self._path = path
        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._file = open(path, "a", buffering=1)
        atexit.register(self.flush)

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            self._buffer.append(line)
            now = time.monotonic()
            if (
                len(self._buffer) >= _FLUSH_THRESHOLD
                or now - self._last_flush >= _FLUSH_INTERVAL
            ):
                self._do_flush()

    def flush(self) -> None:
        with self._lock:
            self._do_flush()

    def _do_flush(self) -> None:
        if not self._buffer:
            return
        try:
            self._file.write("\n".join(self._buffer) + "\n")
            self._file.flush()
        except Exception:
            pass
        self._buffer.clear()
        self._last_flush = time.monotonic()

    def close(self) -> None:
        self.flush()
        try:
            self._file.close()
        except Exception:
            pass


_writer: Optional[_TraceWriter] = None
_pid: int = 0
_tp_rank: int = 0
_dp_rank: int = 0
_server_role: str = "colocated"
_model_name: str = ""


def is_trace_enabled() -> bool:
    return _ENABLED


def init_tracer(
    tp_rank: int = 0,
    dp_rank: int = 0,
    server_role: str = "colocated",
    model_name: str = "",
) -> None:
    global _writer, _pid, _tp_rank, _dp_rank, _server_role, _model_name
    if not _ENABLED:
        return
    _pid = os.getpid()
    _tp_rank = tp_rank
    _dp_rank = dp_rank
    _server_role = server_role
    _model_name = model_name
    _writer = _TraceWriter(_TRACE_FILE)


def _should_sample() -> bool:
    if _SAMPLE_RATE >= 1.0:
        return True
    return random.random() < _SAMPLE_RATE


def _base_record(event: str) -> Dict[str, Any]:
    return {
        "event": event,
        "timestamp": time.time(),
        "tp_rank": _tp_rank,
        "dp_rank": _dp_rank,
        "pid": _pid,
        "server_role": _server_role,
        "model": _model_name,
    }


def trace_match(
    input_len: int,
    structural_match_len: int,
    state_checkpoint_match_len: int,
    effective_match_len: int,
    num_traversed_nodes: int,
    num_traversed_nodes_with_mamba: int,
    num_tombstone_nodes_on_path: int,
    last_node_id: int,
    best_mamba_node_id: int,
    split_happened: bool,
    split_prefix_len: Optional[int],
    split_node_has_mamba_value: Optional[bool],
    cow_mamba_triggered: bool,
    mamba_branching_seqlen: Optional[int],
    page_size: int,
    enable_mamba_extra_buffer: bool,
    mamba_cache_chunk_size: int,
    full_evictable_size: int,
    mamba_evictable_size: int,
    full_protected_size: int,
    mamba_protected_size: int,
    key_hash: Optional[str] = None,
) -> None:
    if not _ENABLED or _writer is None or not _should_sample():
        return
    r = _base_record("match")
    r.update(
        {
            "input_len": input_len,
            "structural_match_len": structural_match_len,
            "state_checkpoint_match_len": state_checkpoint_match_len,
            "effective_match_len": effective_match_len,
            "gated_match_loss": structural_match_len - effective_match_len,
            "num_traversed_nodes": num_traversed_nodes,
            "num_traversed_nodes_with_mamba": num_traversed_nodes_with_mamba,
            "num_tombstone_nodes_on_path": num_tombstone_nodes_on_path,
            "last_node_id": last_node_id,
            "best_mamba_node_id": best_mamba_node_id,
            "split_happened": split_happened,
            "split_prefix_len": split_prefix_len,
            "split_node_has_mamba_value": split_node_has_mamba_value,
            "cow_mamba_triggered": cow_mamba_triggered,
            "mamba_branching_seqlen": mamba_branching_seqlen,
            "page_size": page_size,
            "enable_mamba_extra_buffer": enable_mamba_extra_buffer,
            "mamba_cache_chunk_size": mamba_cache_chunk_size,
            "full_evictable_size": full_evictable_size,
            "mamba_evictable_size": mamba_evictable_size,
            "full_protected_size": full_protected_size,
            "mamba_protected_size": mamba_protected_size,
            "key_hash": key_hash,
        }
    )
    _writer.write(r)


def trace_split(
    split_prefix_len: int,
    new_node_id: int,
    child_node_id: int,
    child_had_mamba_value: bool,
    new_node_has_mamba_value: bool,
) -> None:
    if not _ENABLED or _writer is None or not _should_sample():
        return
    r = _base_record("split")
    r.update(
        {
            "split_prefix_len": split_prefix_len,
            "new_node_id": new_node_id,
            "child_node_id": child_node_id,
            "child_had_mamba_value": child_had_mamba_value,
            "split_created_mamba_tombstone": not new_node_has_mamba_value,
        }
    )
    _writer.write(r)


def trace_insert(
    total_prefix_length: int,
    new_key_len: int,
    mamba_checkpoint_inserted: bool,
    mamba_checkpoint_restored_tombstone: bool,
    mamba_value_already_existed: bool,
    node_id: int,
) -> None:
    if not _ENABLED or _writer is None or not _should_sample():
        return
    r = _base_record("insert")
    r.update(
        {
            "total_prefix_length": total_prefix_length,
            "new_key_len": new_key_len,
            "mamba_checkpoint_inserted": mamba_checkpoint_inserted,
            "mamba_checkpoint_restored_tombstone": mamba_checkpoint_restored_tombstone,
            "mamba_value_already_existed": mamba_value_already_existed,
            "node_id": node_id,
        }
    )
    _writer.write(r)


def trace_evict_mamba(
    mamba_slots_evicted: int,
    was_internal_tombstone: bool,
    node_id: int,
    node_had_children: bool,
) -> None:
    if not _ENABLED or _writer is None or not _should_sample():
        return
    r = _base_record("evict_mamba")
    r.update(
        {
            "mamba_slots_evicted": mamba_slots_evicted,
            "was_internal_tombstone": was_internal_tombstone,
            "node_id": node_id,
            "node_had_children": node_had_children,
        }
    )
    _writer.write(r)


def trace_evict_kv(
    kv_tokens_evicted: int,
    node_id: int,
    node_had_mamba_value: bool,
) -> None:
    if not _ENABLED or _writer is None or not _should_sample():
        return
    r = _base_record("evict_kv")
    r.update(
        {
            "kv_tokens_evicted": kv_tokens_evicted,
            "node_id": node_id,
            "node_had_mamba_value": node_had_mamba_value,
        }
    )
    _writer.write(r)
