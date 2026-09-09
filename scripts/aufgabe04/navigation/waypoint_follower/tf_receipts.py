"""Bounded diagnostics at the executing listener's dynamic-buffer ingestion.

Humble TransformListener.callback calls Buffer.set_transform(transform,
authority); its static callback calls set_transform_static instead. Wrapping
the former traces the actual execution buffer without another subscription.
A returned call does not prove TF core accepted the sample. These records are
never localization readiness or motion authority.
"""

from __future__ import annotations

from collections import deque
import math
import threading
import time
from typing import Callable, Mapping


TF_RECEIPT_HISTORY_LIMIT = 8


def _stamp_sec(transform) -> float:
    stamp = transform.header.stamp
    return _finite_sec(float(stamp.sec) + float(stamp.nanosec) * 1e-9)


def _finite_sec(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite TF diagnostic timestamp")
    return result


class TfBufferReceipts:
    """Thread-safe, fixed-size receipt histories for the two required edges."""

    def __init__(self, edges: Mapping[str, tuple[str, str]], *, ros_now: Callable[[], float],
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self._clock, self._monotonic = ros_now, monotonic
        self._lock = threading.Lock()
        self._created_monotonic, self._created_ros = self.timestamps()
        self._edges = {
            role: {"target_frame": target, "source_frame": source,
                   "received_count": 0, "ingestion_returned_count": 0,
                   "ingestion_exception_count": 0, "last_receipts": deque(maxlen=TF_RECEIPT_HISTORY_LIMIT)}
            for role, (target, source) in edges.items()
        }

    def timestamps(self):
        try:
            return _finite_sec(self._monotonic()), _finite_sec(self._clock())
        except Exception:
            return None, None

    def begin(self, transform):
        for role, edge in self._edges.items():
            if (transform.header.frame_id, transform.child_frame_id) != (
                edge["target_frame"], edge["source_frame"],
            ):
                continue
            received_monotonic, received_ros = self.timestamps()
            if received_monotonic is None or received_ros is None:
                return None
            stamp = _stamp_sec(transform)
            record = {"received_monotonic_sec": received_monotonic,
                      "received_ros_sec": received_ros, "header_stamp_sec": stamp,
                      "age_at_receipt_sec": _finite_sec(received_ros - stamp),
                      "ingestion_call_state": "entered"}
            with self._lock:
                edge["received_count"] += 1
                record["receipt_index"] = edge["received_count"]
                edge["last_receipts"].append(record)
            return role, record
        return None

    def finish(self, token, *, finished_at, exception=None, newest_buffer_stamp=None,
               newest_buffer_error=None) -> None:
        role, record = token
        finished_monotonic, finished_ros = finished_at
        with self._lock:
            self._edges[role]["ingestion_returned_count" if exception is None
                              else "ingestion_exception_count"] += 1
            record.update(
                ingestion_call_state="returned" if exception is None else "raised",
                ingestion_finished_monotonic_sec=finished_monotonic,
                ingestion_finished_ros_sec=finished_ros,
                ingestion_exception_type=None if exception is None else type(exception).__name__,
                newest_buffer_stamp_sec=newest_buffer_stamp,
                newest_buffer_lookup_error=newest_buffer_error,
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            edges = {role: {**edge, "last_receipts": [dict(item) for item in edge["last_receipts"]]}
                     for role, edge in self._edges.items()}
        return {"schema_version": 1, "source": "execution_buffer_set_transform",
                "receipt_semantics": "dynamic_set_transform_entry_not_dds_receive",
                "receipt_ros_clock_owner": "tf_listener_node",
                "created_monotonic_sec": self._created_monotonic,
                "created_ros_sec": self._created_ros,
                "diagnostic_only": True, "insertion_acceptance_proven": False,
                "history_limit_per_edge": TF_RECEIPT_HISTORY_LIMIT, "edges": edges}


def create_receipt_traced_buffer(buffer_type, *, node,
                                edges: Mapping[str, tuple[str, str]],
                                latest_time_factory: Callable,
                                monotonic: Callable[[], float] = time.monotonic):
    """Preserve the real buffer's return/exception and both listener APIs.

    lookup_transform_core uses an immediate core query: no timeout, sleep,
    spin, or diagnostic lock is held during either TF operation. Static
    ingestion stays inherited and does not count as a dynamic receipt.
    """

    receipts = TfBufferReceipts(
        edges, ros_now=lambda: node.get_clock().now().nanoseconds * 1e-9,
        monotonic=monotonic,
    )

    class ReceiptTracedBuffer(buffer_type):
        def tf_receipt_snapshot(self):
            return receipts.snapshot()

        def set_transform(self, transform, authority):
            try:
                token = receipts.begin(transform)
            except Exception:
                token = None  # Diagnostics must never prevent actual ingestion.
            try:
                result = super().set_transform(transform, authority)
            except BaseException as exc:
                if token is not None:
                    try:
                        receipts.finish(token, finished_at=receipts.timestamps(), exception=exc)
                    except Exception:
                        pass
                raise
            if token is not None:
                finished_at = receipts.timestamps()
                newest, error = None, None
                try:
                    newest = _stamp_sec(self.lookup_transform_core(
                        transform.header.frame_id, transform.child_frame_id,
                        latest_time_factory(),
                    ))
                except Exception as exc:
                    error = type(exc).__name__
                try:
                    receipts.finish(token, finished_at=finished_at,
                                    newest_buffer_stamp=newest, newest_buffer_error=error)
                except Exception:
                    pass
            return result

    return ReceiptTracedBuffer(node=node)
