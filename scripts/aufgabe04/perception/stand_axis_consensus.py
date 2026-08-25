"""Temporal consensus for camera-relative stand-head yaw evidence."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class AxisConsensus:
    yaw_rad: float
    sample_count: int
    max_deviation_rad: float
    source: str


@dataclass(frozen=True)
class AxisConditioning:
    accepted: bool
    reason: str
    obliqueness_rad: float
    max_obliqueness_rad: float


def axis_conditioning(
    yaw_rad: float,
    *,
    max_obliqueness_rad: float = math.radians(30.0),
) -> AxisConditioning:
    """Reject face estimates that are too oblique for reliable silhouette geometry."""

    if not math.isfinite(yaw_rad):
        return AxisConditioning(False, "non_finite_axis", math.inf, max_obliqueness_rad)
    if not math.isfinite(max_obliqueness_rad) or max_obliqueness_rad <= 0.0:
        raise ValueError("max obliqueness must be finite and positive")
    obliqueness = abs(math.atan2(math.sin(yaw_rad), math.cos(yaw_rad)))
    if obliqueness > max_obliqueness_rad:
        return AxisConditioning(
            False, "oblique_silhouette", obliqueness, max_obliqueness_rad
        )
    return AxisConditioning(True, "well_conditioned", obliqueness, max_obliqueness_rad)


class AxisConsensusAccumulator:
    def __init__(self, *, required_samples: int = 5, max_deviation_rad: float = math.radians(8.0)):
        if required_samples < 2 or max_deviation_rad <= 0.0:
            raise ValueError("invalid axis consensus configuration")
        self.required_samples = required_samples
        self.max_deviation_rad = max_deviation_rad
        self._values: deque[float] = deque(maxlen=required_samples)
        self._key: tuple[str, str, object] | None = None

    @property
    def sample_count(self) -> int:
        """Number of currently retained, mutually compatible axis samples."""

        return len(self._values)

    def add(
        self,
        *,
        yaw_rad: float,
        source: str,
        side: str,
        qr_texts: tuple[str, ...],
        target_key: str | None = None,
    ) -> AxisConsensus | None:
        if (
            not math.isfinite(yaw_rad)
            or not source
            or (side == "unknown_side" and not target_key)
        ):
            self.reset()
            return None
        key = (
            (source, "metric_target", target_key)
            if target_key
            else (source, side, tuple(qr_texts))
        )
        if key != self._key:
            self._key = key
            self._values.clear()
        self._values.append(yaw_rad)
        if len(self._values) < self.required_samples:
            return None
        sin_mean = sum(math.sin(value) for value in self._values) / len(self._values)
        cos_mean = sum(math.cos(value) for value in self._values) / len(self._values)
        mean = math.atan2(sin_mean, cos_mean)
        deviations = [abs(math.atan2(math.sin(value - mean), math.cos(value - mean))) for value in self._values]
        maximum = max(deviations)
        if maximum > self.max_deviation_rad:
            return None
        return AxisConsensus(mean, len(self._values), maximum, source)

    def reset(self) -> None:
        self._values.clear()
        self._key = None
