"""Measured physical arena bounds for Aufgabe 04 dry-run placement."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scripts.aufgabe04.navigation.models import Pose2D


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
        yaw = math.radians(self.yaw_deg)
        dx = pose.x_m - self.center_x_m
        dy = pose.y_m - self.center_y_m
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        half_length = self.length_m / 2.0 - self.margin_m
        half_width = self.width_m / 2.0 - self.margin_m
        return abs(local_x) <= half_length and abs(local_y) <= half_width

    def to_metadata(self) -> dict[str, float]:
        return {
            "length_m": self.length_m,
            "width_m": self.width_m,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "yaw_deg": self.yaw_deg,
            "margin_m": self.margin_m,
        }
