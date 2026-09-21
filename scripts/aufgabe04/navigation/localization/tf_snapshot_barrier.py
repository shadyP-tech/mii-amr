"""Deadline-bound admission of a TF snapshot after a fixed evidence window."""

from dataclasses import dataclass
import math
import time


DEFAULT_TF_SNAPSHOT_WAIT_SEC = 0.5


@dataclass(frozen=True)
class TfSnapshotResult:
    transform: object | None
    retries: int
    elapsed_sec: float
    timed_out: bool
    last_error: str = ""


def acquire_tf_snapshot(*, lookup, stamp_sec, service_callbacks,
                        minimum_stamp_sec, lookup_errors,
                        timeout_sec=DEFAULT_TF_SNAPSHOT_WAIT_SEC,
                        monotonic=time.monotonic):
    """Use zero-wait lookups; only service callbacks while the predicate fails.

    The caller freezes ``minimum_stamp_sec`` before entering this function and
    validates the returned transform's identity, age and geometry afterwards.
    No old transform is returned if a later lookup reports that it disappeared.
    """
    if not math.isfinite(timeout_sec) or timeout_sec < 0:
        raise ValueError("TF snapshot timeout must be finite and non-negative")
    if not math.isfinite(minimum_stamp_sec):
        raise ValueError("TF snapshot minimum stamp must be finite")
    start = monotonic()
    deadline = start + timeout_sec
    retries = 0
    while True:
        transform = None
        error = ""
        try:
            transform = lookup()
        except lookup_errors as exc:
            error = str(exc)
        now = monotonic()
        if (transform is not None and stamp_sec(transform) >= minimum_stamp_sec
                and (retries == 0 or now <= deadline)):
            return TfSnapshotResult(transform, retries, now - start, False)
        remaining = deadline - now
        if remaining <= 0:
            return TfSnapshotResult(transform, retries, now - start, True, error)
        service_callbacks(min(0.01, remaining))
        retries += 1
