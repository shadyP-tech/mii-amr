"""ROS-free waypoint, pose and controller values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Waypoint:
    index: int
    x: float
    y: float


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw_deg: float
    stamp_sec: float | None = None
    frame_id: str = ""


@dataclass(frozen=True)
class TargetState:
    distance_m: float
    heading_deg: float
    yaw_error_deg: float


@dataclass(frozen=True)
class ScanSafety:
    safe: bool
    reason: str
    valid_count: int
    min_range_m: float | None
    percentile_5_m: float | None


@dataclass(frozen=True)
class AmclHealth:
    ok: bool
    warnings: list[str]
    cov_x: float | None
    cov_y: float | None
    cov_yaw: float | None
    age_sec: float | None


@dataclass(frozen=True)
class StartSelection:
    waypoints: list[Waypoint]
    selected_segment_index: int | None
    selected_waypoint_index: int | None
    distance_to_path_m: float | None


@dataclass(frozen=True)
class TwistCommand:
    linear_x: float
    angular_z: float



@dataclass(frozen=True)
class ControllerStep:
    command: TwistCommand
    mode: str
    target: Waypoint | None
    distance_m: float
    yaw_error_deg: float
    reached: bool


class RouteState:
    def __init__(self, waypoints):
        self.waypoints = list(waypoints)
        self.current_waypoint_index = 0

    @property
    def complete(self):
        return self.current_waypoint_index >= len(self.waypoints)

    def current_waypoint(self):
        return None if self.complete else self.waypoints[self.current_waypoint_index]

    def is_final(self):
        return self.current_waypoint_index == len(self.waypoints) - 1

    def remaining(self):
        return self.waypoints[self.current_waypoint_index:]

    def advance(self):
        self.current_waypoint_index += 1
