"""Temporal consensus for 180-degree-symmetric stand axes."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from scripts.aufgabe04.perception.stand_axis_handoff.geometry import (
    axial_difference_rad,
    axial_normalize_rad,
)


@dataclass(frozen=True)
class AxialConsensus:
    angle_rad: float
    sample_count: int
    max_deviation_rad: float
    source: str


class AxialConsensusAccumulator:
    def __init__(
        self,
        *,
        required_samples: int = 5,
        max_deviation_rad: float = math.radians(8.0),
    ) -> None:
        if required_samples < 2 or max_deviation_rad <= 0.0:
            raise ValueError("invalid axial consensus configuration")
        self.required_samples = required_samples
        self.max_deviation_rad = max_deviation_rad
        self._values: deque[float] = deque(maxlen=required_samples)
        self._key: tuple[str, str, object] | None = None

    def add(
        self,
        *,
        angle_rad: float,
        source: str,
        side: str,
        qr_texts: tuple[str, ...],
        target_key: str | None = None,
    ) -> AxialConsensus | None:
        if (
            not math.isfinite(angle_rad)
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
        self._values.append(axial_normalize_rad(angle_rad))
        if len(self._values) < self.required_samples:
            return None
        doubled_sine = sum(math.sin(2.0 * value) for value in self._values)
        doubled_cosine = sum(math.cos(2.0 * value) for value in self._values)
        mean = axial_normalize_rad(
            0.5 * math.atan2(doubled_sine, doubled_cosine)
        )
        maximum = max(
            axial_difference_rad(value, mean) for value in self._values
        )
        if maximum > self.max_deviation_rad:
            return None
        return AxialConsensus(mean, len(self._values), maximum, source)

    def reset(self) -> None:
        self._values.clear()
        self._key = None
