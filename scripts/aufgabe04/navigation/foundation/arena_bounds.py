"""Rectangular arena approximation for Aufgabe 04 placement and filtering.

The physical long walls are 3.70 m; 3.90 m is the measured span from one
short wall to the spot corridor at the opposite short wall. This rectangle
does not encode that corridor geometry. See the physical arena measurements
in docs/setups/aufgabe04_real_pipeline.md (clarified 2026-10-02).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scripts.aufgabe04.navigation.foundation.models import Pose2D


DEFAULT_ARENA_LENGTH_M = 3.90
DEFAULT_ARENA_WIDTH_M = 1.898
DEFAULT_ARENA_CENTER_X_M = 0.0
DEFAULT_ARENA_CENTER_Y_M = 0.0
DEFAULT_ARENA_YAW_DEG = 0.0
DEFAULT_ARENA_MARGIN_M = 0.0


@dataclass(frozen=True)
class ArenaBounds:
    length_m: float = DEFAULT_ARENA_LENGTH_M
    width_m: float = DEFAULT_ARENA_WIDTH_M
    center_x_m: float = DEFAULT_ARENA_CENTER_X_M
    center_y_m: float = DEFAULT_ARENA_CENTER_Y_M
    yaw_deg: float = DEFAULT_ARENA_YAW_DEG
    margin_m: float = DEFAULT_ARENA_MARGIN_M

    def validate(self) -> None:
        if self.length_m <= 0.0:
            raise ValueError("arena length must be positive")
        if self.width_m <= 0.0:
            raise ValueError("arena width must be positive")
        if self.margin_m < 0.0:
            raise ValueError("arena margin must be non-negative")
        if self.margin_m * 2.0 >= self.length_m:
            raise ValueError("arena margin leaves no usable arena length")
        if self.margin_m * 2.0 >= self.width_m:
            raise ValueError("arena margin leaves no usable arena width")

    def contains(self, pose: Pose2D) -> bool:
        return self.boundary_clearance_m(pose) >= self.margin_m

    def boundary_clearance_m(self, pose: Pose2D) -> float:
        """Return signed clearance from a pose to the configured rectangle.

        Positive values are inside the configured rectangle, zero is on its
        boundary, and negative values are outside.  Keeping this independent
        of ``margin_m`` lets perception reject wall returns while route
        planning can continue to apply its own placement margin.
        """

        yaw = math.radians(self.yaw_deg)
        dx = pose.x_m - self.center_x_m
        dy = pose.y_m - self.center_y_m
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        x_clearance = self.length_m / 2.0 - abs(local_x)
        y_clearance = self.width_m / 2.0 - abs(local_y)
        return min(x_clearance, y_clearance)

    def to_metadata(self) -> dict[str, float]:
        return {
            "length_m": self.length_m,
            "width_m": self.width_m,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "yaw_deg": self.yaw_deg,
            "margin_m": self.margin_m,
        }
