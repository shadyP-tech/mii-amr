"""Adapter-neutral follower result models for Aufgabe 04 navigation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FollowerResult:
    status: str
    stop_reason: str
    duration_sec: float
    distance_estimate_m: float
    motion_published: bool
    target_index: int = -1
    remaining_distance_m: float = 0.0
    final_x_m: float | None = None
    final_y_m: float | None = None
    final_yaw_rad: float | None = None
