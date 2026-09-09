"""Observation ages in one clock domain, independent of frame arrival order."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObservationFreshness:
    accepted: bool
    reason: str
    age_sec: float | None


def observation_freshness(
    *,
    observed_at_sec: float | None,
    now_sec: float,
    max_age_sec: float,
    max_future_sec: float = 0.0,
) -> ObservationFreshness:
    """Check an absolute age budget; zero explicitly disables that budget."""

    if not math.isfinite(max_age_sec) or max_age_sec < 0.0:
        raise ValueError("max_age_sec must be finite and non-negative")
    if not math.isfinite(max_future_sec) or max_future_sec < 0.0:
        raise ValueError("max_future_sec must be finite and non-negative")
    age = (
        None
        if observed_at_sec is None
        or not math.isfinite(observed_at_sec)
        or not math.isfinite(now_sec)
        else now_sec - observed_at_sec
    )
    if max_age_sec == 0.0:
        return ObservationFreshness(True, "age_budget_disabled", age)
    if age is None:
        return ObservationFreshness(False, "observation_timestamp_unavailable", None)
    if age < -max_future_sec:
        return ObservationFreshness(False, "observation_timestamp_in_future", age)
    if age > max_age_sec:
        return ObservationFreshness(False, "observation_too_old", age)
    return ObservationFreshness(True, "observation_fresh", age)


def source_stamp_in_monotonic_domain(
    *,
    source_stamp_sec: float | None,
    received_wall_sec: float | None,
    received_monotonic_sec: float | None,
) -> float | None:
    """Anchor source age at receipt so processing uses a monotonic clock.

    The source stamp must use the receiver's wall-clock domain. Simulation
    timestamps must instead be compared with the simulation clock directly.
    """

    values = (source_stamp_sec, received_wall_sec, received_monotonic_sec)
    if any(value is None or not math.isfinite(value) for value in values):
        return None
    return received_monotonic_sec - (received_wall_sec - source_stamp_sec)
