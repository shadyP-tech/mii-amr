from __future__ import annotations

import math
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
class LookaheadGuardResult:
    safe: bool
    status: str
    reason: str
    candidate_count: int
    blocked_cell_count: int
    selected_target_index: int | None = None
    selected_target_distance_m: float | None = None
    selected_target_x_m: float | None = None
    selected_target_y_m: float | None = None


@dataclass(frozen=True)
class ControllerStep:
    command: TwistCommand
    mode: str
    target: Waypoint | tuple[float, float] | None
    distance_m: float
    yaw_error_deg: float
    reached: bool
    guard_result: LookaheadGuardResult | None = None


@dataclass(frozen=True)
class TrackingPathValidation:
    source: str
    point_count: int
    endpoint_error_m: float | None = None
    start_error_m: float | None = None
    start_projection_error_m: float | None = None
    validation_status: str = "not_applicable"
    warnings: tuple[str, ...] = ()


@dataclass
class RouteState:
    waypoints: list[Waypoint]
    tracking_points: list[Waypoint] | None = None
    current_waypoint_index: int = 0
    tracking_progress_index: int = 0
    tracking_source: str = "waypoints"
    tracking_validation: TrackingPathValidation | None = None

    def __init__(
        self,
        waypoints,
        tracking_points=None,
        current_waypoint_index=0,
        tracking_progress_index=0,
        tracking_source="waypoints",
        tracking_validation=None,
        current_index=None,
    ):
        if current_index is not None:
            current_waypoint_index = current_index
        self.waypoints = list(waypoints)
        self.tracking_points = (
            None if tracking_points is None else list(tracking_points)
        )
        self.current_waypoint_index = self._clamp_waypoint_index(
            current_waypoint_index,
        )
        self.tracking_progress_index = self._clamp_tracking_index(
            tracking_progress_index,
        )
        self.tracking_source = str(tracking_source)
        self.tracking_validation = tracking_validation

    def _clamp_waypoint_index(self, index):
        return max(0, min(int(index), len(self.waypoints)))

    def _clamp_tracking_index(self, index):
        points = self.effective_tracking_points()
        if not points:
            return 0
        return max(0, min(int(index), len(points) - 1))

    @property
    def current_index(self):
        return self.current_waypoint_index

    @current_index.setter
    def current_index(self, value):
        self.current_waypoint_index = self._clamp_waypoint_index(value)

    @property
    def complete(self):
        return self.current_waypoint_index >= len(self.waypoints)

    def remaining(self):
        return list(self.waypoints[self.current_waypoint_index :])

    def current_waypoint(self):
        if self.complete:
            return None
        return self.waypoints[self.current_waypoint_index]

    def is_final(self):
        return (
            bool(self.waypoints)
            and self.current_waypoint_index == len(self.waypoints) - 1
        )

    def final_goal(self):
        if not self.waypoints:
            return None
        return self.waypoints[-1]

    def advance(self):
        if not self.complete:
            self.current_waypoint_index += 1
        return self.complete

    def mark_complete(self):
        self.current_waypoint_index = len(self.waypoints)
        return True

    def advance_if_reached(self, pose, waypoint_tolerance_m, goal_tolerance_m):
        waypoint = self.current_waypoint()
        if waypoint is None:
            return False
        distance_m = math.hypot(waypoint.x - pose.x, waypoint.y - pose.y)
        tolerance = goal_tolerance_m if self.is_final() else waypoint_tolerance_m
        if distance_m > tolerance:
            return False
        self.advance()
        return True

    def effective_tracking_points(self):
        if self.tracking_points is not None:
            return self.tracking_points
        return self.waypoints

    def remaining_tracking_points(self):
        points = self.effective_tracking_points()
        if not points:
            return []
        index = self._clamp_tracking_index(self.tracking_progress_index)
        return list(points[index:])

    def advance_tracking_progress(self, pose, progress_tolerance_m):
        points = self.effective_tracking_points()
        if len(points) < 2:
            self.tracking_progress_index = 0
            return self.tracking_progress_index

        index = self._clamp_tracking_index(self.tracking_progress_index)
        while index < len(points) - 1:
            point = points[index]
            distance_m = math.hypot(point.x - pose.x, point.y - pose.y)
            if distance_m > progress_tolerance_m:
                break
            index += 1

        best = None
        for segment_index in range(index, len(points) - 1):
            distance_m, projection = self._distance_to_tracking_segment(
                pose,
                points[segment_index],
                points[segment_index + 1],
            )
            candidate = (distance_m, segment_index, projection)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            distance_m, segment_index, projection = best
            if distance_m <= progress_tolerance_m and projection > 0.0:
                index = max(index, min(segment_index + 1, len(points) - 1))

        self.tracking_progress_index = index
        return self.tracking_progress_index

    @staticmethod
    def _distance_to_tracking_segment(point, segment_start, segment_end):
        dx = segment_end.x - segment_start.x
        dy = segment_end.y - segment_start.y
        length_sq = dx * dx + dy * dy
        if length_sq == 0.0:
            return (
                math.hypot(point.x - segment_start.x, point.y - segment_start.y),
                0.0,
            )
        projection = (
            (point.x - segment_start.x) * dx + (point.y - segment_start.y) * dy
        ) / length_sq
        projection = max(0.0, min(1.0, projection))
        closest_x = segment_start.x + projection * dx
        closest_y = segment_start.y + projection * dy
        return math.hypot(point.x - closest_x, point.y - closest_y), projection

    def replace_route(
        self,
        waypoints,
        current_index=0,
        tracking_points=None,
        tracking_progress_index=0,
        tracking_source="waypoints",
        tracking_validation=None,
    ):
        self.waypoints = list(waypoints)
        self.tracking_points = (
            None if tracking_points is None else list(tracking_points)
        )
        self.current_waypoint_index = self._clamp_waypoint_index(current_index)
        self.tracking_progress_index = self._clamp_tracking_index(
            tracking_progress_index,
        )
        self.tracking_source = str(tracking_source)
        self.tracking_validation = tracking_validation
