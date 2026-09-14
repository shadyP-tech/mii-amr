"""Bounded diagnostics from the passive observer's actual executing TF buffer.

Receipt means entry into that buffer's listener ingestion call, not DDS arrival
or broadcaster publication. TF header stamps describe transform validity and
may be future-dated; their difference from local ROS time is not network
latency. No diagnostic value supplies a transform or admission authority.
"""

from __future__ import annotations

from collections import OrderedDict, deque
import math
import threading
import time
from typing import Callable


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite TF diagnostic timestamp")
    return value


def _frame(value):
    if not isinstance(value, str) or not 0 < len(value) <= 128:
        raise ValueError("invalid or oversized TF diagnostic frame")
    return value


def _stamp(transform):
    stamp = transform.header.stamp
    return _finite(float(stamp.sec) + float(stamp.nanosec) * 1e-9)


def _difference(later, earlier, *, nonnegative=False):
    if later is None or earlier is None:
        return None
    difference = later - earlier
    if not math.isfinite(difference):
        return None
    return max(0., difference) if nonnegative else difference


class ObserverTfDeliveryTrace:
    """Fixed-capacity, thread-safe histories; no additional TF subscription."""

    def __init__(self, *, ros_now: Callable[[], float],
                 monotonic: Callable[[], float] = time.monotonic,
                 max_edges: int = 16, history_limit: int = 4,
                 lookup_history_limit: int = 8):
        for value, maximum in ((max_edges, 32), (history_limit, 8), (lookup_history_limit, 16)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("TF diagnostic limits must be bounded positive integers")
        self._ros_now, self._monotonic = ros_now, monotonic
        self._max_edges, self._history_limit = max_edges, history_limit
        self._edges = OrderedDict()
        self._lookups = deque(maxlen=lookup_history_limit)
        self._lock = threading.Lock()
        self._counts = dict(receipts=0, ingestion_returned=0, ingestion_raised=0,
                            ignored_receipts=0, edge_evictions=0,
                            lookup_returned=0, lookup_raised=0)

    def _timestamps(self):
        try:
            return _finite(self._ros_now()), _finite(self._monotonic())
        except Exception:
            return None, None

    def _begin_receipt(self, transform, static):
        # Capture before invoking the core. No diagnostic lock spans core work.
        ros_sec, monotonic_sec = self._timestamps()
        try:
            parent, child = _frame(transform.header.frame_id), _frame(transform.child_frame_id)
            stamp = _stamp(transform)
        except Exception:
            with self._lock:
                self._counts["ignored_receipts"] += 1
            return None
        record = {
            "source_stamp_sec": stamp, "received_ros_sec": ros_sec,
            "received_monotonic_sec": monotonic_sec,
            "source_to_receipt_sec": None if static else _difference(ros_sec, stamp),
            "ingestion_call_state": "entered",
        }
        key = parent, child, static
        with self._lock:
            self._counts["receipts"] += 1
            if key not in self._edges:
                if len(self._edges) >= self._max_edges:
                    self._edges.popitem(last=False)
                    self._counts["edge_evictions"] += 1
                self._edges[key] = {
                    "parent_frame": parent, "child_frame": child, "static": static,
                    "timeless": static, "received_count": 0,
                    "ingestion_returned_count": 0, "ingestion_raised_count": 0,
                    "last_receipts": deque(maxlen=self._history_limit),
                }
            self._edges.move_to_end(key)
            edge = self._edges[key]
            edge["received_count"] += 1
            record["receipt_index"] = self._counts["receipts"]
            edge["last_receipts"].append(record)
        return edge, record

    def _finish_receipt(self, token, exception):
        if token is None:
            return
        ros_sec, monotonic_sec = self._timestamps()
        edge, record = token
        outcome = "returned" if exception is None else "raised"
        with self._lock:
            self._counts[f"ingestion_{outcome}"] += 1
            edge[f"ingestion_{outcome}_count"] += 1
            record.update(
                ingestion_call_state=outcome, finished_ros_sec=ros_sec,
                finished_monotonic_sec=monotonic_sec,
                exception_type=None if exception is None else type(exception).__name__[:128],
            )

    def ingest(self, operation, transform, static, *args, **kwargs):
        """Preserve the original insertion call, return value and exception."""
        try:
            token = self._begin_receipt(transform, static)
        except Exception:
            token = None
        try:
            result = operation(transform, *args, **kwargs)
        except BaseException as exc:
            try:
                self._finish_receipt(token, exc)
            except Exception:
                pass
            raise
        try:
            self._finish_receipt(token, None)
        except Exception:
            pass
        return result

    def _record_lookup(self, request, started_at, result, exception):
        ended_ros, ended_monotonic = self._timestamps()
        started_ros, started_monotonic = started_at
        try:
            returned_stamp = None if result is None else _stamp(result)
        except Exception:
            returned_stamp = None
        stamp = request.get("query_stamp_sec")
        query = {
            "target_frame": _frame(request["target_frame"]),
            "source_frame": _frame(request["source_frame"]),
            "query_kind": str(request["query_kind"])[:64],
            "query_stamp_sec": None if stamp is None else _finite(stamp),
            "requested_ros_sec": started_ros, "requested_monotonic_sec": started_monotonic,
            "finished_ros_sec": ended_ros, "finished_monotonic_sec": ended_monotonic,
            "elapsed_sec": _difference(ended_monotonic, started_monotonic, nonnegative=True),
            "outcome": "returned" if exception is None else "raised",
            "returned_stamp_sec": returned_stamp,
            "exception_type": None if exception is None else type(exception).__name__[:128],
        }
        with self._lock:
            self._counts[f"lookup_{query['outcome']}"] += 1
            query["receipt_count_at_completion"] = self._counts["receipts"]
            self._lookups.append(query)

    def lookup(self, request, operation):
        started_at = self._timestamps()
        try:
            result = operation()
        except BaseException as exc:
            try:
                self._record_lookup(request, started_at, None, exc)
            except Exception:
                pass
            raise
        try:
            self._record_lookup(request, started_at, result, None)
        except Exception:
            pass
        return result

    def snapshot(self):
        ros_sec, monotonic_sec = self._timestamps()
        with self._lock:
            edges = []
            for edge in self._edges.values():
                receipts = [dict(record) for record in edge["last_receipts"]]
                last_receipt = receipts[-1]["received_monotonic_sec"]
                edges.append({
                    **edge, "last_receipts": receipts,
                    "last_receipt_age_sec": _difference(monotonic_sec, last_receipt, nonnegative=True),
                })
            return {
                "schema_version": 1, "source": "observer_execution_buffer",
                "receipt_semantics": "set_transform_entry_not_dds_receive_or_publication",
                "header_stamp_semantics": "transform_validity_time_not_publication_time",
                "diagnostic_only": True, "motion_authorized": False,
                "insertion_acceptance_proven": False,
                "snapshot_ros_sec": ros_sec, "snapshot_monotonic_sec": monotonic_sec,
                "limits": {"edges": self._max_edges, "receipts_per_edge": self._history_limit,
                           "lookups": self._lookups.maxlen},
                "counts": dict(self._counts), "edges": edges,
                "recent_lookups": [dict(record) for record in self._lookups],
            }


def create_observer_traced_buffer(buffer_type, *, trace, **buffer_kwargs):
    """Instrument the existing listener's actual Buffer, with no extra queries."""
    class TracedBuffer(buffer_type):
        def set_transform(self, transform, *args, **kwargs):
            return trace.ingest(super().set_transform, transform, False, *args, **kwargs)

        def set_transform_static(self, transform, *args, **kwargs):
            return trace.ingest(super().set_transform_static, transform, True, *args, **kwargs)

    return TracedBuffer(**buffer_kwargs)


def traced_observer_lookup(trace, request, operation):
    """Allow ROS-free legacy adapters without instrumentation to stay unchanged."""
    return operation() if trace is None else trace.lookup(request, operation)
