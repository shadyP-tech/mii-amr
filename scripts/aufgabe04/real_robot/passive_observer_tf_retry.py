"""Pure exact-timestamp TF retry state for the passive camera observer.

The ROS adapter owns callback scheduling and transform lookup.  This module
only preserves one stamped sensor tuple until that exact tuple is either
consumed or explicitly discarded.  It performs no I/O and has no motion
capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Generic, Literal, TypeVar


FrameT = TypeVar("FrameT")
RetryState = Literal["idle", "pending", "transform_ready"]


class PassiveObserverTfRetryError(ValueError):
    """Raised when a caller violates the stamped-frame lifecycle."""


@dataclass(frozen=True)
class StampedObserverFrame(Generic[FrameT]):
    """One generic sensor tuple bound to its original image timestamp."""

    stamp_sec: float
    frame: FrameT


@dataclass(frozen=True)
class PassiveObserverTfRetryEvidence:
    """Immutable audit snapshot for the current retry lifecycle."""

    state: RetryState
    pending_stamp_sec: float | None
    retry_count: int
    first_failure_time_sec: float | None
    first_failure_reason: str | None
    last_failure_time_sec: float | None
    last_failure_reason: str | None
    last_consumed_stamp_sec: float | None
    last_discarded_stamp_sec: float | None
    last_discard_reason: str | None


def _require_stamp(stamp_sec: float, *, field: str) -> float:
    try:
        value = float(stamp_sec)
    except (TypeError, ValueError) as exc:
        raise PassiveObserverTfRetryError(
            f"{field} must be finite and positive"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise PassiveObserverTfRetryError(
            f"{field} must be finite and positive"
        )
    return value


def _require_event_time(observed_sec: float) -> float:
    try:
        value = float(observed_sec)
    except (TypeError, ValueError) as exc:
        raise PassiveObserverTfRetryError(
            "observed_sec must be finite and non-negative"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise PassiveObserverTfRetryError(
            "observed_sec must be finite and non-negative"
        )
    return value


def _require_reason(reason: str, *, field: str) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise PassiveObserverTfRetryError(f"{field} must be non-empty")
    return reason.strip()


class PassiveObserverTfRetryScheduler(Generic[FrameT]):
    """Hold one exact sensor tuple across transient TF readiness failures.

    ``offer`` returns ``False`` for a strictly newer frame while another frame
    is pending.  The caller may offer that newer frame again after the pending
    frame reaches a terminal state.  Duplicate, stale, or out-of-order stamps
    are rejected instead of being silently reused.
    """

    def __init__(self) -> None:
        self._state: RetryState = "idle"
        self._pending: StampedObserverFrame[FrameT] | None = None
        self._retry_count = 0
        self._first_failure_time_sec: float | None = None
        self._first_failure_reason: str | None = None
        self._last_failure_time_sec: float | None = None
        self._last_failure_reason: str | None = None
        self._last_consumed_stamp_sec: float | None = None
        self._last_discarded_stamp_sec: float | None = None
        self._last_discard_reason: str | None = None

    @property
    def evidence(self) -> PassiveObserverTfRetryEvidence:
        """Return an immutable snapshot without exposing the held frame."""

        return PassiveObserverTfRetryEvidence(
            state=self._state,
            pending_stamp_sec=(
                self._pending.stamp_sec if self._pending is not None else None
            ),
            retry_count=self._retry_count,
            first_failure_time_sec=self._first_failure_time_sec,
            first_failure_reason=self._first_failure_reason,
            last_failure_time_sec=self._last_failure_time_sec,
            last_failure_reason=self._last_failure_reason,
            last_consumed_stamp_sec=self._last_consumed_stamp_sec,
            last_discarded_stamp_sec=self._last_discarded_stamp_sec,
            last_discard_reason=self._last_discard_reason,
        )

    @property
    def pending_frame(self) -> StampedObserverFrame[FrameT] | None:
        """Return the exact tuple to query, or ``None`` once it is ready."""

        return self._pending if self._state == "pending" else None

    def offer(self, frame: FrameT, *, stamp_sec: float) -> bool:
        """Offer a new tuple without replacing an existing pending tuple."""

        stamp = _require_stamp(stamp_sec, field="stamp_sec")
        terminal_stamp = self._latest_terminal_stamp()
        if terminal_stamp is not None and stamp <= terminal_stamp:
            raise PassiveObserverTfRetryError(
                "offered frame stamp must be strictly newer than the last "
                "terminal frame stamp"
            )
        if self._pending is not None:
            if stamp <= self._pending.stamp_sec:
                raise PassiveObserverTfRetryError(
                    "offered frame stamp must be strictly newer than the "
                    "pending frame stamp"
                )
            return False

        self._pending = StampedObserverFrame(stamp_sec=stamp, frame=frame)
        self._state = "pending"
        self._reset_pending_failure_evidence()
        return True

    def mark_transform_unavailable(
        self,
        *,
        stamp_sec: float,
        observed_sec: float,
        reason: str,
    ) -> PassiveObserverTfRetryEvidence:
        """Record a transient failure while retaining the same exact tuple."""

        self._require_pending_stamp(stamp_sec, required_state="pending")
        observed = _require_event_time(observed_sec)
        failure_reason = _require_reason(reason, field="reason")
        if (
            self._last_failure_time_sec is not None
            and observed < self._last_failure_time_sec
        ):
            raise PassiveObserverTfRetryError(
                "observed_sec must not precede the prior TF failure"
            )
        self._retry_count += 1
        if self._first_failure_time_sec is None:
            self._first_failure_time_sec = observed
            self._first_failure_reason = failure_reason
        self._last_failure_time_sec = observed
        self._last_failure_reason = failure_reason
        return self.evidence

    def mark_transform_ready(self, *, stamp_sec: float) -> None:
        """Seal successful exact-time TF readiness for the pending tuple."""

        self._require_pending_stamp(stamp_sec, required_state="pending")
        self._state = "transform_ready"

    def consume(self, *, stamp_sec: float) -> StampedObserverFrame[FrameT]:
        """Consume a transform-ready tuple exactly once."""

        pending = self._require_pending_stamp(
            stamp_sec,
            required_state="transform_ready",
        )
        self._last_consumed_stamp_sec = pending.stamp_sec
        self._pending = None
        self._state = "idle"
        return pending

    def discard(self, *, stamp_sec: float, reason: str) -> None:
        """Discard a pending or ready tuple without emitting its frame."""

        discard_reason = _require_reason(reason, field="reason")
        pending = self._require_pending_stamp(stamp_sec)
        self._last_discarded_stamp_sec = pending.stamp_sec
        self._last_discard_reason = discard_reason
        self._pending = None
        self._state = "idle"

    def _latest_terminal_stamp(self) -> float | None:
        stamps = tuple(
            stamp
            for stamp in (
                self._last_consumed_stamp_sec,
                self._last_discarded_stamp_sec,
            )
            if stamp is not None
        )
        return max(stamps) if stamps else None

    def _require_pending_stamp(
        self,
        stamp_sec: float,
        *,
        required_state: RetryState | None = None,
    ) -> StampedObserverFrame[FrameT]:
        stamp = _require_stamp(stamp_sec, field="stamp_sec")
        pending = self._pending
        if pending is None:
            raise PassiveObserverTfRetryError("no stamped frame is pending")
        if stamp != pending.stamp_sec:
            raise PassiveObserverTfRetryError(
                "operation stamp does not match the pending frame stamp"
            )
        if required_state is not None and self._state != required_state:
            raise PassiveObserverTfRetryError(
                f"operation requires state {required_state!r}, "
                f"found {self._state!r}"
            )
        return pending

    def _reset_pending_failure_evidence(self) -> None:
        self._retry_count = 0
        self._first_failure_time_sec = None
        self._first_failure_reason = None
        self._last_failure_time_sec = None
        self._last_failure_reason = None


__all__ = [
    "PassiveObserverTfRetryError",
    "PassiveObserverTfRetryEvidence",
    "PassiveObserverTfRetryScheduler",
    "StampedObserverFrame",
]
