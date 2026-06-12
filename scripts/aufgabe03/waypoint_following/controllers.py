from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

from .math_utils import clamp, normalize_angle_rad, shortest_angle_delta_deg
from .models import ControllerStep, RouteState, TwistCommand
from .path_curves import (
    ROUTE_HEADING_LOOKAHEAD_M,
    RouteHeading,
    RouteProjection,
    branch_compatible_path_from_projection,
    lookahead_target_from_route_anchor,
    project_point_to_route,
    project_point_to_route_branch_window,
    project_point_to_route_progress_window,
    pure_pursuit_curve_command,
    route_cumulative_distances,
    route_heading_at_progress,
    route_heading_from_projection,
    route_point_at_progress,
    route_points_from_projection,
)
from .path_progress import target_state, waypoint_reached
from .velocity_scheduler import (
    SPEED_PROFILE_CURVATURE_AWARE,
    SPEED_PROFILE_FIXED,
    PurePursuitVelocityScheduler,
    pure_pursuit_geometry,
)


PROJECTION_LOCK_REQUIRED_SAMPLES = 3
PROJECTION_LOCK_PROGRESS_TOLERANCE_M = 0.03
ROTATE_ANCHOR_LOCAL_WINDOW_BACK_M = 0.08
ROTATE_ANCHOR_LOCAL_WINDOW_FORWARD_M = 0.20
ROTATE_ANCHOR_ROUTE_HEADING_EXIT_SAMPLES = 3
POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG = 60.0
POST_ROTATE_BRANCH_RELEASE_STABLE_SAMPLES = 2
POST_ROTATE_BRANCH_MIN_RELEASE_PROGRESS_M = 0.18
POST_ROTATE_BRANCH_END_TOLERANCE_M = 0.03
POST_ROTATE_BRANCH_END_LATERAL_TOLERANCE_M = 0.08
POST_ROTATE_ZERO_LINEAR_EPS_MPS = 0.003
ANGULAR_FEASIBILITY_STOP_EPS_MPS = 0.003
FORWARD_CONTROL_TARGET_BEARING = "target-bearing"
FORWARD_CONTROL_ROUTE_DAMPED = "route-damped"
FORWARD_CONTROL_MODES = (
    FORWARD_CONTROL_TARGET_BEARING,
    FORWARD_CONTROL_ROUTE_DAMPED,
)
PATH_PROFILE_SCHEDULING_OFF = "off"
PATH_PROFILE_SCHEDULING_ON = "on"
PATH_PROFILE_SCHEDULING_MODES = (
    PATH_PROFILE_SCHEDULING_OFF,
    PATH_PROFILE_SCHEDULING_ON,
)
PATH_PROFILE_STATUS_OFF = "off"
PATH_PROFILE_STATUS_BASE = "base"
PATH_PROFILE_STATUS_STRAIGHT_FAST = "straight_fast"
PATH_PROFILE_STATUS_APPROACH_BEND = "approach_bend"
PATH_PROFILE_STATUS_SHORT_SEGMENT = "short_segment"
PATH_PROFILE_STATUS_FORCE_ROTATE_PENDING = "force_rotate_pending"
PATH_PROFILE_STATUS_FORCE_ROTATE_HANDOFF = "force_rotate_handoff"
PATH_PROFILE_HEADING_BREAK_DEG = 20.0
PATH_PROFILE_FORCE_ROTATE_DEG = 120.0
PATH_PROFILE_SLOWDOWN_WINDOW_M = 0.30
PATH_PROFILE_SHORT_SEGMENT_M = 0.25
PATH_PROFILE_BEND_SPEED_CAP_MPS = 0.035
PATH_PROFILE_SHORT_SPEED_CAP_MPS = 0.040
PATH_PROFILE_STRAIGHT_LOOKAHEAD_M = 0.24
PATH_PROFILE_BEND_LOOKAHEAD_M = 0.16
PATH_PROFILE_FORCE_ROTATE_HANDOFF_M = 0.06
PATH_PROFILE_STRAIGHT_ENTER_SAMPLES = 3
PATH_PROFILE_STRAIGHT_ENTER_HEADING_ERROR_DEG = 6.0
PATH_PROFILE_STRAIGHT_ENTER_CROSS_TRACK_M = 0.025
PATH_PROFILE_STRAIGHT_ENTER_BREAK_DISTANCE_M = 0.50
PATH_PROFILE_STRAIGHT_EXIT_HEADING_ERROR_DEG = 10.0
PATH_PROFILE_STRAIGHT_EXIT_CROSS_TRACK_M = 0.04
PATH_PROFILE_STRAIGHT_EXIT_BREAK_DISTANCE_M = 0.35
PATH_PROFILE_MIN_BREAK_RUN_M = 0.075


@dataclass
class RotateProjectionAnchor:
    progress_m: float
    segment_index: int
    segment_ratio: float
    projected_point: tuple[float, float]
    route_heading_rad: float
    cross_track_m: float
    max_backward_delta_m: float = 0.0
    max_forward_delta_m: float = 0.0
    route_heading_aligned_sample_count: int = 0
    max_route_heading_aligned_sample_count: int = 0
    handoff_reason: str = ""


@dataclass
class PostRotateBranchLock:
    preferred_heading_rad: float
    last_progress_m: float
    last_segment_index: int
    start_progress_m: float
    stable_count: int = 0
    rejected_wrong_heading_count: int = 0
    max_heading_error_deg: float = 0.0
    activations: int = 0
    ambiguity_failures: int = 0
    rotate_anchor_aligned_sample_count: int = 0
    rotate_anchor_handoff_reason: str = ""
    suppress_alpha_rotate: bool = False


@dataclass(frozen=True)
class BranchFrameEndCheck:
    along_past_m: float
    lateral_error_m: float
    lateral_tolerance_m: float
    yaw_error_deg: float
    hard_cross_track_exceeded: bool
    handoff_allowed: bool


@dataclass(frozen=True)
class ForwardControlResult:
    mode: str
    fallback_reason: str
    alpha_deg: float
    route_heading_error_deg: float | None
    signed_cross_track_error_m: float | None
    cte_correction_deg: float
    blended_forward_error_deg: float
    speed_taper_error_deg: float
    raw_angular_z: float
    command_angular_z: float
    angular_feasibility_limited: bool = False
    angular_feasibility_scale: float = 1.0
    linear_before_feasibility_mps: float = 0.0
    linear_after_feasibility_mps: float = 0.0


@dataclass(frozen=True)
class PathProfileScheduleResult:
    status: str
    speed_cap_mps: float
    lookahead_m: float
    distance_to_heading_break_m: float | None = None
    heading_break_delta_deg: float | None = None
    branch_end_progress_m: float | None = None
    branch_end_point: tuple[float, float] | None = None
    next_heading_deg: float | None = None
    straight_stable_count: int = 0


@dataclass(frozen=True)
class PathProfileHeadingRun:
    start_progress_m: float
    end_progress_m: float
    heading_deg: float

    @property
    def length_m(self):
        return max(0.0, self.end_progress_m - self.start_progress_m)


@dataclass(frozen=True)
class PathProfileRouteAhead:
    distance_to_heading_break_m: float | None
    heading_break_delta_deg: float | None
    branch_end_progress_m: float | None
    branch_end_point: tuple[float, float] | None
    next_heading_deg: float | None
    compatible_length_m: float


class PathController(Protocol):
    def compute(self, pose, route_state: RouteState) -> ControllerStep:
        ...


def should_rotate(current_mode, yaw_error_deg, start_threshold_deg, stop_threshold_deg):
    abs_error = abs(yaw_error_deg)
    if current_mode == "rotate":
        return abs_error > stop_threshold_deg
    return abs_error > start_threshold_deg


def velocity_command(
    distance_m,
    yaw_error_deg,
    rotate_mode,
    linear_speed_mps,
    min_linear_speed_mps,
    linear_gain,
    max_angular_speed_radps,
    yaw_gain,
    forward_yaw_deadband_deg=0.0,
    forward_stop_heading_error_deg=180.0,
):
    angular_z = clamp(
        math.radians(yaw_error_deg) * yaw_gain,
        -max_angular_speed_radps,
        max_angular_speed_radps,
    )
    abs_yaw_error = abs(yaw_error_deg)
    if rotate_mode or abs_yaw_error >= forward_stop_heading_error_deg:
        return 0.0, angular_z

    linear_x = clamp(
        distance_m * linear_gain,
        min_linear_speed_mps,
        linear_speed_mps,
    )
    if abs_yaw_error <= forward_yaw_deadband_deg:
        return linear_x, 0.0

    scale_span = forward_stop_heading_error_deg - forward_yaw_deadband_deg
    heading_scale = 1.0
    if scale_span > 0.0:
        heading_scale = 1.0 - (abs_yaw_error - forward_yaw_deadband_deg) / scale_span
    heading_scale = clamp(heading_scale, 0.0, 1.0)
    linear_x *= heading_scale
    if linear_x > 0.0:
        linear_x = max(min_linear_speed_mps, linear_x)
    return linear_x, angular_z


class StopGoController:
    def __init__(self, args):
        self.args = args
        self.mode = "forward"
        self._last_route_position = None

    def _reset_if_target_changed(self, route_state):
        route_position = (
            route_state.current_waypoint_index,
            None
            if route_state.current_waypoint() is None
            else route_state.current_waypoint().index,
        )
        if self._last_route_position != route_position:
            self.mode = "forward"
            self._last_route_position = route_position

    def compute(self, pose, route_state):
        self._reset_if_target_changed(route_state)
        waypoint = route_state.current_waypoint()
        if waypoint is None:
            return ControllerStep(
                TwistCommand(0.0, 0.0),
                "forward",
                None,
                0.0,
                0.0,
                True,
            )
        state = target_state(pose, waypoint)
        reached = waypoint_reached(
            state.distance_m,
            route_state.is_final(),
            self.args.waypoint_tolerance_m,
            self.args.goal_tolerance_m,
        )
        if reached:
            return ControllerStep(
                TwistCommand(0.0, 0.0),
                self.mode,
                waypoint,
                state.distance_m,
                state.yaw_error_deg,
                True,
            )
        rotate_mode = should_rotate(
            self.mode,
            state.yaw_error_deg,
            self.args.rotate_start_heading_error_deg,
            self.args.rotate_stop_heading_error_deg,
        )
        self.mode = "rotate" if rotate_mode else "forward"
        linear_x, angular_z = velocity_command(
            state.distance_m,
            state.yaw_error_deg,
            rotate_mode,
            self.args.linear_speed,
            self.args.min_linear_speed,
            self.args.linear_gain,
            self.args.max_angular_speed,
            self.args.yaw_gain,
            self.args.forward_yaw_deadband_deg,
            self.args.forward_stop_heading_error_deg,
        )
        return ControllerStep(
            TwistCommand(linear_x, angular_z),
            self.mode,
            waypoint,
            state.distance_m,
            state.yaw_error_deg,
            False,
        )


class PurePursuitController:
    def __init__(self, args, lookahead_guard=None):
        self.args = args
        self.lookahead_guard = lookahead_guard
        self.velocity_scheduler = PurePursuitVelocityScheduler.from_args(args)
        self.mode = "forward"
        self.last_projection_segment_index = None
        self.last_route_progress_m = None
        self.projection_locked = False
        self.projection_lock_sample_count = 0
        self.max_projection_backward_delta_m = 0.0
        self.max_rotate_anchor_backward_delta_m = 0.0
        self.max_rotate_anchor_forward_delta_m = 0.0
        self.max_rotate_anchor_aligned_samples = 0
        self.last_rotate_anchor_aligned_samples = 0
        self.rotate_anchor_activations = 0
        self.last_accepted_projection = None
        self.rotate_projection_anchor = None
        self.post_rotate_branch_lock = None
        self.post_rotate_branch_lock_activations = 0
        self.post_rotate_branch_ambiguity_failures = 0
        self.post_rotate_branch_rejected_wrong_heading_count = 0
        self.post_rotate_branch_max_heading_error_deg = 0.0
        self.post_rotate_branch_target_clip_count = 0
        self.post_rotate_branch_heading_break_handoff_count = 0
        self.post_rotate_branch_physical_handoff_count = 0
        self.last_rotate_sign = 1.0
        self.rotate_gate_entries = 0
        self.path_profile_straight_stable_count = 0
        self.path_profile_last_status = PATH_PROFILE_STATUS_OFF

    def reset_route_projection_state(self):
        self.last_projection_segment_index = None
        self.last_route_progress_m = None
        self.projection_locked = False
        self.projection_lock_sample_count = 0
        self.max_projection_backward_delta_m = 0.0
        self.max_rotate_anchor_backward_delta_m = 0.0
        self.max_rotate_anchor_forward_delta_m = 0.0
        self.max_rotate_anchor_aligned_samples = 0
        self.last_rotate_anchor_aligned_samples = 0
        self.rotate_anchor_activations = 0
        self.last_accepted_projection = None
        self.rotate_projection_anchor = None
        self.post_rotate_branch_lock = None
        self.post_rotate_branch_lock_activations = 0
        self.post_rotate_branch_ambiguity_failures = 0
        self.post_rotate_branch_rejected_wrong_heading_count = 0
        self.post_rotate_branch_max_heading_error_deg = 0.0
        self.post_rotate_branch_target_clip_count = 0
        self.post_rotate_branch_heading_break_handoff_count = 0
        self.post_rotate_branch_physical_handoff_count = 0
        self._reset_path_profile_state()

    def _reset_path_profile_state(self):
        self.path_profile_straight_stable_count = 0
        self.path_profile_last_status = (
            PATH_PROFILE_STATUS_OFF
            if not self._path_profile_enabled()
            else PATH_PROFILE_STATUS_BASE
        )

    def _path_profile_enabled(self):
        return (
            getattr(
                self.args,
                "pure_pursuit_path_profile_scheduling",
                PATH_PROFILE_SCHEDULING_OFF,
            )
            == PATH_PROFILE_SCHEDULING_ON
        )

    def _path_profile_base_result(self, lookahead_m):
        return PathProfileScheduleResult(
            PATH_PROFILE_STATUS_OFF
            if not self._path_profile_enabled()
            else PATH_PROFILE_STATUS_BASE,
            abs(float(self.args.linear_speed)),
            max(0.0, float(lookahead_m)),
            straight_stable_count=int(self.path_profile_straight_stable_count),
        )

    @staticmethod
    def _path_profile_heading_runs(route_points, projection):
        route = [(float(x), float(y)) for x, y in route_points]
        if len(route) < 2:
            return []
        cumulative = route_cumulative_distances(route)
        if not cumulative or cumulative[-1] <= 1e-9:
            return []
        start_progress_m = clamp(
            float(projection.route_progress_m),
            0.0,
            cumulative[-1],
        )
        runs = []
        for index in range(len(route) - 1):
            segment_start_m = cumulative[index]
            segment_end_m = cumulative[index + 1]
            local_start_m = max(segment_start_m, start_progress_m)
            if segment_end_m <= local_start_m + 1e-9:
                continue
            start = route[index]
            end = route[index + 1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            if math.hypot(dx, dy) <= 1e-9:
                continue
            heading_deg = math.degrees(math.atan2(dy, dx))
            if (
                runs
                and abs(
                    shortest_angle_delta_deg(
                        runs[-1].heading_deg,
                        heading_deg,
                    )
                )
                <= 1e-6
            ):
                previous = runs[-1]
                runs[-1] = PathProfileHeadingRun(
                    previous.start_progress_m,
                    segment_end_m,
                    previous.heading_deg,
                )
            else:
                runs.append(
                    PathProfileHeadingRun(
                        local_start_m,
                        segment_end_m,
                        heading_deg,
                    )
                )
        return runs

    def _path_profile_route_ahead(self, route_points, projection):
        runs = self._path_profile_heading_runs(route_points, projection)
        if not runs:
            return PathProfileRouteAhead(
                None,
                None,
                None,
                None,
                None,
                max(0.0, float(projection.remaining_route_m)),
            )
        current_heading = runs[0].heading_deg
        compatible_end_m = runs[0].end_progress_m
        for run in runs[1:]:
            heading_delta = abs(
                shortest_angle_delta_deg(current_heading, run.heading_deg)
            )
            if (
                heading_delta >= PATH_PROFILE_HEADING_BREAK_DEG
                and run.length_m >= PATH_PROFILE_MIN_BREAK_RUN_M
            ):
                break_progress_m = run.start_progress_m
                route = [(float(x), float(y)) for x, y in route_points]
                cumulative = route_cumulative_distances(route)
                break_x, break_y, _index, _ratio = route_point_at_progress(
                    route,
                    cumulative,
                    break_progress_m,
                )
                distance_to_break_m = max(
                    0.0,
                    break_progress_m - float(projection.route_progress_m),
                )
                return PathProfileRouteAhead(
                    distance_to_break_m,
                    heading_delta,
                    break_progress_m,
                    (break_x, break_y),
                    run.heading_deg,
                    max(0.0, distance_to_break_m),
                )
            compatible_end_m = run.end_progress_m
        return PathProfileRouteAhead(
            None,
            None,
            None,
            None,
            None,
            max(0.0, compatible_end_m - float(projection.route_progress_m)),
        )

    @staticmethod
    def _finite_or_inf(value):
        return math.inf if value is None else float(value)

    def _path_profile_effective_speed_cap(self, configured_cap_mps):
        return min(
            abs(float(self.args.linear_speed)),
            abs(float(configured_cap_mps)),
        )

    def _path_profile_schedule(
        self,
        route_points,
        projection,
        base_lookahead_m,
    ):
        if not self._path_profile_enabled():
            return self._path_profile_base_result(base_lookahead_m)

        ahead = self._path_profile_route_ahead(route_points, projection)
        base_speed = abs(float(self.args.linear_speed))
        base_lookahead = max(0.0, float(base_lookahead_m))
        min_profile_lookahead = max(
            0.01,
            PATH_PROFILE_BEND_LOOKAHEAD_M,
            float(getattr(self.args, "pure_pursuit_min_guarded_lookahead_m", 0.0)),
        )
        distance_to_break = ahead.distance_to_heading_break_m
        break_delta = ahead.heading_break_delta_deg
        break_distance_for_tests = self._finite_or_inf(distance_to_break)
        route_heading_error = abs(float(projection.heading_error_to_route_deg))
        cross_track_error = abs(float(projection.cross_track_error_m))
        previous_status = self.path_profile_last_status
        status = PATH_PROFILE_STATUS_BASE
        speed_cap = base_speed
        lookahead_cap = base_lookahead

        if (
            break_delta is not None
            and break_delta >= PATH_PROFILE_FORCE_ROTATE_DEG
            and distance_to_break is not None
        ):
            if distance_to_break <= PATH_PROFILE_FORCE_ROTATE_HANDOFF_M:
                status = PATH_PROFILE_STATUS_FORCE_ROTATE_HANDOFF
            else:
                status = PATH_PROFILE_STATUS_FORCE_ROTATE_PENDING
            self.path_profile_straight_stable_count = 0
            speed_cap = self._path_profile_effective_speed_cap(
                getattr(
                    self.args,
                    "pure_pursuit_path_profile_short_speed_cap_mps",
                    PATH_PROFILE_SHORT_SPEED_CAP_MPS,
                )
            )
            lookahead_cap = min(min_profile_lookahead, distance_to_break)
        elif (
            distance_to_break is not None
            and distance_to_break < PATH_PROFILE_SHORT_SEGMENT_M
        ):
            status = PATH_PROFILE_STATUS_SHORT_SEGMENT
            self.path_profile_straight_stable_count = 0
            speed_cap = self._path_profile_effective_speed_cap(
                getattr(
                    self.args,
                    "pure_pursuit_path_profile_short_speed_cap_mps",
                    PATH_PROFILE_SHORT_SPEED_CAP_MPS,
                )
            )
            lookahead_cap = min(min_profile_lookahead, distance_to_break)
        elif (
            break_delta is not None
            and break_delta >= PATH_PROFILE_HEADING_BREAK_DEG
            and distance_to_break is not None
            and distance_to_break <= PATH_PROFILE_SLOWDOWN_WINDOW_M
        ):
            status = PATH_PROFILE_STATUS_APPROACH_BEND
            self.path_profile_straight_stable_count = 0
            speed_cap = self._path_profile_effective_speed_cap(
                getattr(
                    self.args,
                    "pure_pursuit_path_profile_bend_speed_cap_mps",
                    PATH_PROFILE_BEND_SPEED_CAP_MPS,
                )
            )
            lookahead_cap = min(min_profile_lookahead, distance_to_break)
        else:
            exit_straight = (
                route_heading_error > PATH_PROFILE_STRAIGHT_EXIT_HEADING_ERROR_DEG
                or cross_track_error > PATH_PROFILE_STRAIGHT_EXIT_CROSS_TRACK_M
                or break_distance_for_tests
                < PATH_PROFILE_STRAIGHT_EXIT_BREAK_DISTANCE_M
            )
            enter_straight = (
                route_heading_error
                <= PATH_PROFILE_STRAIGHT_ENTER_HEADING_ERROR_DEG
                and cross_track_error <= PATH_PROFILE_STRAIGHT_ENTER_CROSS_TRACK_M
                and break_distance_for_tests
                >= PATH_PROFILE_STRAIGHT_ENTER_BREAK_DISTANCE_M
            )
            if previous_status == PATH_PROFILE_STATUS_STRAIGHT_FAST and not exit_straight:
                status = PATH_PROFILE_STATUS_STRAIGHT_FAST
                self.path_profile_straight_stable_count = max(
                    PATH_PROFILE_STRAIGHT_ENTER_SAMPLES,
                    self.path_profile_straight_stable_count,
                )
            elif enter_straight:
                self.path_profile_straight_stable_count += 1
                if (
                    self.path_profile_straight_stable_count
                    >= PATH_PROFILE_STRAIGHT_ENTER_SAMPLES
                ):
                    status = PATH_PROFILE_STATUS_STRAIGHT_FAST
            else:
                self.path_profile_straight_stable_count = 0

            if status == PATH_PROFILE_STATUS_STRAIGHT_FAST:
                speed_cap = abs(
                    float(self.args.pure_pursuit_path_profile_straight_speed_mps)
                )
                lookahead_cap = min(
                    PATH_PROFILE_STRAIGHT_LOOKAHEAD_M,
                    max(base_lookahead, 0.0),
                )
            else:
                speed_cap = base_speed
                lookahead_cap = base_lookahead

        if distance_to_break is not None:
            lookahead_cap = min(lookahead_cap, max(0.0, distance_to_break))
        lookahead_cap = min(lookahead_cap, max(0.0, float(projection.remaining_route_m)))
        self.path_profile_last_status = status
        return PathProfileScheduleResult(
            status,
            max(0.0, float(speed_cap)),
            max(0.0, float(lookahead_cap)),
            distance_to_break,
            break_delta,
            ahead.branch_end_progress_m,
            ahead.branch_end_point,
            ahead.next_heading_deg,
            straight_stable_count=int(self.path_profile_straight_stable_count),
        )

    def _target_from_path_profile(
        self,
        route_points,
        projection,
        path_profile_result,
    ):
        if (
            path_profile_result.branch_end_point is not None
            and path_profile_result.distance_to_heading_break_m is not None
            and path_profile_result.lookahead_m
            >= path_profile_result.distance_to_heading_break_m - 1e-9
        ):
            return path_profile_result.branch_end_point
        path_points = route_points_from_projection(route_points, projection)
        return lookahead_target_from_route_anchor(
            path_points,
            min(
                max(0.0, path_profile_result.lookahead_m),
                max(0.0, projection.remaining_route_m),
            ),
        )

    @staticmethod
    def _lookahead_for_target(projection, target_point, fallback_m):
        projected = projection.projected_point
        distance_to_target = math.hypot(
            float(target_point[0]) - float(projected[0]),
            float(target_point[1]) - float(projected[1]),
        )
        if distance_to_target > 1e-9:
            return max(0.01, distance_to_target)
        return max(0.01, min(float(fallback_m), float(projection.remaining_route_m)))

    def compute(self, pose, route_state):
        final_goal = route_state.final_goal()
        if final_goal is None:
            return ControllerStep(
                TwistCommand(0.0, 0.0),
                "forward",
                None,
                0.0,
                0.0,
                True,
            )

        distance_to_goal = math.hypot(final_goal.x - pose.x, final_goal.y - pose.y)
        goal_tolerance_m = getattr(
            self.args,
            "pure_pursuit_goal_tolerance_m",
            self.args.goal_tolerance_m,
        )
        reached = waypoint_reached(
            distance_to_goal,
            True,
            goal_tolerance_m,
            goal_tolerance_m,
        )
        heading_to_goal = math.degrees(
            math.atan2(final_goal.y - pose.y, final_goal.x - pose.x)
        )
        yaw_error_to_goal = shortest_angle_delta_deg(
            pose.yaw_deg,
            heading_to_goal,
        )
        if reached:
            return ControllerStep(
                TwistCommand(0.0, 0.0),
                "forward",
                final_goal,
                distance_to_goal,
                yaw_error_to_goal,
                True,
            )

        current_point = (float(pose.x), float(pose.y))
        lookahead_m = getattr(self.args, "path_lookahead_m", 0.18)
        tracking_points = route_state.effective_tracking_points()
        route_points = [(float(wp.x), float(wp.y)) for wp in tracking_points]
        if len(route_points) < 2:
            route_points = [current_point, (float(final_goal.x), float(final_goal.y))]

        if self.rotate_projection_anchor is not None:
            return self._compute_with_rotate_anchor(
                pose,
                route_state,
                route_points,
                final_goal,
                distance_to_goal,
                goal_tolerance_m,
                lookahead_m,
                current_point,
            )

        if self.post_rotate_branch_lock is not None:
            return self._compute_with_post_rotate_branch_lock(
                pose,
                route_state,
                route_points,
                final_goal,
                distance_to_goal,
                goal_tolerance_m,
                lookahead_m,
                current_point,
            )

        route_state.advance_tracking_progress(
            pose,
            float(
                getattr(
                    self.args,
                    "pure_pursuit_tracking_progress_tolerance_m",
                    0.06,
                )
            ),
        )
        start_segment = (
            max(0, self.last_projection_segment_index - 1)
            if self.last_projection_segment_index is not None
            else max(0, route_state.tracking_progress_index - 1)
        )
        try:
            projection = project_point_to_route(
                route_points,
                pose,
                start_segment_index=start_segment,
                previous_progress_m=self.last_route_progress_m,
                max_forward_jump_m=max(2.0 * lookahead_m, 0.35),
                backward_tolerance_m=0.03,
                allow_backward=not self.projection_locked,
                projection_status="locked" if self.projection_locked else "acquiring",
            )
        except RuntimeError as error:
            if (
                str(error) != "route_projection_moved_backward"
                or self.last_accepted_projection is None
            ):
                raise
            anchor_heading = route_heading_from_projection(
                route_points,
                self.last_accepted_projection,
                pose.yaw_deg,
                ROUTE_HEADING_LOOKAHEAD_M,
            )
            self._activate_rotate_anchor(
                self.last_accepted_projection,
                anchor_heading,
            )
            return self._compute_with_rotate_anchor(
                pose,
                route_state,
                route_points,
                final_goal,
                distance_to_goal,
                goal_tolerance_m,
                lookahead_m,
                current_point,
                require_rotate=True,
            )
        self.max_projection_backward_delta_m = max(
            self.max_projection_backward_delta_m,
            projection.route_progress_backward_delta_m,
        )
        if (
            projection.cross_track_error_m
            > getattr(self.args, "pure_pursuit_max_cross_track_error_m", 0.25)
        ):
            return ControllerStep(
                TwistCommand(0.0, 0.0),
                "off_route",
                projection.projected_point,
                distance_to_goal,
                projection.heading_error_to_route_deg,
                False,
                route_projection_result=projection,
                pure_pursuit_status="off_route",
            )
        self._accept_projection(projection, route_state)

        path_profile_result = self._path_profile_schedule(
            route_points,
            projection,
            lookahead_m,
        )
        target_point = self._target_from_path_profile(
            route_points,
            projection,
            path_profile_result,
        )
        command_lookahead_m = self._lookahead_for_target(
            projection,
            target_point,
            path_profile_result.lookahead_m,
        )
        path_points = route_points_from_projection(route_points, projection)
        geometry = pure_pursuit_geometry(pose, target_point, command_lookahead_m)
        heading_lookahead_m = ROUTE_HEADING_LOOKAHEAD_M
        if path_profile_result.distance_to_heading_break_m is not None:
            heading_lookahead_m = min(
                heading_lookahead_m,
                max(0.0, path_profile_result.distance_to_heading_break_m),
            )
        route_heading = route_heading_from_projection(
            route_points,
            projection,
            pose.yaw_deg,
            heading_lookahead_m,
        )
        if path_profile_result.status == PATH_PROFILE_STATUS_FORCE_ROTATE_HANDOFF:
            return self._path_profile_force_rotate_step(
                pose,
                projection,
                path_profile_result,
                distance_to_goal,
            )
        alpha_rad = self._stable_alpha_for_rotate(geometry.alpha_rad)
        alpha_deg = math.degrees(alpha_rad)
        rotate_mode, rotate_reason, rotate_source, rotate_error_deg = (
            self._route_heading_rotate_gate(
                alpha_deg,
                route_heading,
                distance_to_goal,
                goal_tolerance_m,
            )
        )
        rotate_error_rad = (
            alpha_rad
            if rotate_source == "alpha"
            else math.radians(rotate_error_deg)
        )
        if rotate_mode:
            if self.mode != "rotate":
                self.rotate_gate_entries += 1
            self._activate_rotate_anchor(projection, route_heading)
            projection = self._projection_from_rotate_anchor(
                route_points,
                pose,
                projection,
            )
            route_heading = self._route_heading_from_rotate_anchor(pose)
            if rotate_source == "route_heading":
                rotate_error_deg = route_heading.heading_error_deg or rotate_error_deg
                rotate_error_rad = math.radians(rotate_error_deg)
            self.mode = "rotate"
            angular_z = clamp(
                rotate_error_rad * self.args.yaw_gain,
                -abs(self.args.pure_pursuit_max_rotate_angular_speed_radps),
                abs(self.args.pure_pursuit_max_rotate_angular_speed_radps),
            )
            return ControllerStep(
                TwistCommand(0.0, angular_z),
                "rotate",
                target_point,
                distance_to_goal,
                rotate_error_deg,
                False,
                route_projection_result=projection,
                pure_pursuit_status="rotate_gate",
                route_heading_result=route_heading,
                pure_pursuit_rotate_reason=rotate_reason,
                pure_pursuit_rotate_source=rotate_source,
                path_profile_result=path_profile_result,
            )
        self.mode = "forward"
        guard_result = None
        if self.lookahead_guard is not None:
            target_point, guard_result = self.lookahead_guard.select_target(
                pose,
                path_points,
                current_point,
                target_point,
                max(0.0, path_profile_result.lookahead_m),
                getattr(self.args, "pure_pursuit_min_guarded_lookahead_m", 0.12),
                final_goal=final_goal,
                distance_to_goal_m=distance_to_goal,
            )
            if not guard_result.safe:
                self.reset_route_projection_state()
                target_heading_deg = math.degrees(
                    math.atan2(
                        target_point[1] - pose.y,
                        target_point[0] - pose.x,
                    )
                )
                return ControllerStep(
                    TwistCommand(0.0, 0.0),
                    "blocked",
                    target_point,
                    distance_to_goal,
                    shortest_angle_delta_deg(pose.yaw_deg, target_heading_deg),
                    False,
                    guard_result,
                    route_heading_result=route_heading,
                    path_profile_result=path_profile_result,
                )
            if guard_result.selected_target_distance_m is not None:
                command_lookahead_m = guard_result.selected_target_distance_m
            geometry = pure_pursuit_geometry(pose, target_point, command_lookahead_m)
            alpha_rad = geometry.alpha_rad
        speed_profile = getattr(
            self.args,
            "pure_pursuit_speed_profile",
            SPEED_PROFILE_FIXED,
        )
        velocity_schedule_result = None
        if speed_profile == SPEED_PROFILE_FIXED:
            (
                linear_x,
                angular_z,
                alpha_rad,
                forward_control_result,
            ) = self._fixed_forward_command(
                pose,
                target_point,
                command_lookahead_m,
                route_heading,
                projection,
                geometry,
                allow_route_damped=True,
                speed_cap_mps=path_profile_result.speed_cap_mps,
            )
            mode = "forward"
        else:
            velocity_schedule_result = self.velocity_scheduler.schedule(
                geometry,
                allow_rotate=False,
                linear_speed_cap_mps=path_profile_result.speed_cap_mps,
            )
            linear_x = velocity_schedule_result.command.linear_x
            angular_z = velocity_schedule_result.command.angular_z
            alpha_rad = geometry.alpha_rad
            mode = velocity_schedule_result.mode
            forward_control_result = None
        if abs(linear_x) <= 1e-12 and abs(angular_z) <= 1e-12:
            self._reset_path_profile_state()
        return ControllerStep(
            TwistCommand(linear_x, angular_z),
            mode,
            target_point,
            distance_to_goal,
            math.degrees(alpha_rad),
            False,
            guard_result,
            velocity_schedule_result,
            projection,
            "forward",
            route_heading,
            forward_control_result=forward_control_result,
            path_profile_result=path_profile_result,
        )

    def _compute_with_rotate_anchor(
        self,
        pose,
        route_state,
        route_points,
        final_goal,
        distance_to_goal,
        goal_tolerance_m,
        lookahead_m,
        current_point,
        require_rotate=False,
    ):
        anchor = self.rotate_projection_anchor
        if anchor is None:
            raise RuntimeError("rotate_projection_anchor_missing")

        try:
            raw_projection = project_point_to_route_progress_window(
                route_points,
                pose,
                anchor.progress_m - ROTATE_ANCHOR_LOCAL_WINDOW_BACK_M,
                anchor.progress_m + ROTATE_ANCHOR_LOCAL_WINDOW_FORWARD_M,
                previous_progress_m=anchor.progress_m,
                projection_status="rotate_anchor_raw",
            )
        except RuntimeError:
            self.last_rotate_anchor_aligned_samples = 0
            self.rotate_projection_anchor = None
            return ControllerStep(
                TwistCommand(0.0, 0.0),
                "off_route",
                anchor.projected_point,
                distance_to_goal,
                0.0,
                False,
                pure_pursuit_status="off_route",
            )

        if (
            raw_projection.cross_track_error_m
            > getattr(self.args, "pure_pursuit_max_cross_track_error_m", 0.25)
        ):
            projection = self._projection_from_rotate_anchor(
                route_points,
                pose,
                raw_projection,
            )
            self.last_rotate_anchor_aligned_samples = 0
            self.rotate_projection_anchor = None
            return ControllerStep(
                TwistCommand(0.0, 0.0),
                "off_route",
                projection.projected_point,
                distance_to_goal,
                projection.heading_error_to_route_deg,
                False,
                route_projection_result=projection,
                pure_pursuit_status="off_route",
            )

        route_heading = self._route_heading_from_rotate_anchor(pose)
        self._update_rotate_anchor_alignment(anchor, route_heading)
        projection = self._projection_from_rotate_anchor(
            route_points,
            pose,
            raw_projection,
        )
        path_points = route_points_from_projection(route_points, projection)
        target_point = lookahead_target_from_route_anchor(
            path_points,
            min(lookahead_m, max(0.0, projection.remaining_route_m)),
        )
        command_lookahead_m = max(0.01, min(lookahead_m, projection.remaining_route_m))
        geometry = pure_pursuit_geometry(pose, target_point, command_lookahead_m)
        alpha_rad = self._stable_alpha_for_rotate(geometry.alpha_rad)
        alpha_deg = math.degrees(alpha_rad)
        rotate_mode, rotate_reason, rotate_source, rotate_error_deg = (
            self._route_heading_rotate_gate(
                alpha_deg,
                route_heading,
                distance_to_goal,
                goal_tolerance_m,
            )
        )
        if require_rotate and not rotate_mode:
            self.last_rotate_anchor_aligned_samples = 0
            self.rotate_projection_anchor = None
            raise RuntimeError("route_projection_moved_backward")

        rotate_error_rad = (
            alpha_rad
            if rotate_source == "alpha"
            else math.radians(rotate_error_deg)
        )
        if rotate_mode and self._rotate_anchor_stable_handoff_ready(anchor):
            self.mode = "forward"
            anchor.handoff_reason = "route_heading_stable"
            self._activate_post_rotate_branch_lock(anchor)
            self.rotate_projection_anchor = None
            return self._compute_with_post_rotate_branch_lock(
                pose,
                route_state,
                route_points,
                final_goal,
                distance_to_goal,
                goal_tolerance_m,
                lookahead_m,
                current_point,
                suppress_alpha_rotate=True,
            )

        if rotate_mode:
            self.mode = "rotate"
            angular_z = clamp(
                rotate_error_rad * self.args.yaw_gain,
                -abs(self.args.pure_pursuit_max_rotate_angular_speed_radps),
                abs(self.args.pure_pursuit_max_rotate_angular_speed_radps),
            )
            return ControllerStep(
                TwistCommand(0.0, angular_z),
                "rotate",
                target_point,
                distance_to_goal,
                rotate_error_deg,
                False,
                route_projection_result=projection,
                pure_pursuit_status="rotate_gate",
                route_heading_result=route_heading,
                pure_pursuit_rotate_reason=rotate_reason,
                pure_pursuit_rotate_source=rotate_source,
            )

        self.mode = "forward"
        anchor.handoff_reason = "rotate_gate_clear"
        self._activate_post_rotate_branch_lock(anchor)
        self.rotate_projection_anchor = None
        return self._compute_with_post_rotate_branch_lock(
            pose,
            route_state,
            route_points,
            final_goal,
            distance_to_goal,
            goal_tolerance_m,
            lookahead_m,
            current_point,
        )

    def _compute_with_post_rotate_branch_lock(
        self,
        pose,
        route_state,
        route_points,
        final_goal,
        distance_to_goal,
        goal_tolerance_m,
        lookahead_m,
        current_point,
        suppress_alpha_rotate=False,
    ):
        lock = self.post_rotate_branch_lock
        if lock is None:
            raise RuntimeError("post_rotate_branch_lock_missing")
        release_span_m = self._post_rotate_branch_release_span_m(lookahead_m)
        max_span_m = self._post_rotate_branch_max_span_m(lookahead_m)
        try:
            projection = project_point_to_route_branch_window(
                route_points,
                pose,
                lock.last_progress_m - PROJECTION_LOCK_PROGRESS_TOLERANCE_M,
                lock.last_progress_m + max(2.0 * lookahead_m, 0.35),
                math.degrees(lock.preferred_heading_rad),
                POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG,
                previous_progress_m=lock.last_progress_m,
                heading_lookahead_m=ROUTE_HEADING_LOOKAHEAD_M,
                stable_count=lock.stable_count,
                branch_lock_start_progress_m=lock.start_progress_m,
                branch_lock_release_required_span_m=release_span_m,
            )
        except RuntimeError as error:
            if str(error) == "pure_pursuit_branch_ambiguous":
                handoff = self._branch_heading_break_handoff_from_lock(
                    pose,
                    route_points,
                    lock,
                    distance_to_goal,
                    release_span_m,
                )
                if handoff is not None:
                    return handoff
                lock.ambiguity_failures += 1
                self.post_rotate_branch_ambiguity_failures += 1
            raise

        lock.rejected_wrong_heading_count += projection.rejected_wrong_heading_segment_count
        self.post_rotate_branch_rejected_wrong_heading_count += (
            projection.rejected_wrong_heading_segment_count
        )
        if projection.selected_branch_heading_error_deg is not None:
            lock.max_heading_error_deg = max(
                lock.max_heading_error_deg,
                abs(float(projection.selected_branch_heading_error_deg)),
            )
            self.post_rotate_branch_max_heading_error_deg = max(
                self.post_rotate_branch_max_heading_error_deg,
                lock.max_heading_error_deg,
            )

        if (
            projection.cross_track_error_m
            > getattr(self.args, "pure_pursuit_max_cross_track_error_m", 0.25)
        ):
            self.post_rotate_branch_lock = None
            return ControllerStep(
                TwistCommand(0.0, 0.0),
                "off_route",
                projection.projected_point,
                distance_to_goal,
                projection.heading_error_to_route_deg,
                False,
                route_projection_result=projection,
                pure_pursuit_status="off_route",
            )

        if projection.branch_lock_progress_span_m > max_span_m + 1e-9:
            lock.ambiguity_failures += 1
            self.post_rotate_branch_ambiguity_failures += 1
            self.post_rotate_branch_lock = None
            raise RuntimeError("pure_pursuit_branch_ambiguous")

        lock.last_progress_m = projection.route_progress_m
        lock.last_segment_index = projection.segment_index
        effective_span_m = max(0.0, lock.last_progress_m - lock.start_progress_m)
        if effective_span_m + 1e-9 < release_span_m:
            lock.stable_count = 0
        else:
            if self._branch_release_probe_safe(
                route_points,
                pose,
                lock,
                lookahead_m,
            ):
                lock.stable_count += 1
            else:
                lock.stable_count = 0
        projection = self._with_branch_lock_metadata(
            projection,
            lock.stable_count,
            release_span_m,
            lock.rotate_anchor_aligned_sample_count,
            lock.rotate_anchor_handoff_reason,
        )
        branch_path = branch_compatible_path_from_projection(
            route_points,
            projection,
            math.degrees(lock.preferred_heading_rad),
            POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG,
            lookahead_m,
            current_yaw_deg=pose.yaw_deg,
        )
        if branch_path.branch_target_clipped_to_heading_break:
            self.post_rotate_branch_target_clip_count += 1
        projection = replace(
            projection,
            branch_compatible_length_m=branch_path.compatible_length_m,
            branch_target_clipped_to_heading_break=(
                branch_path.branch_target_clipped_to_heading_break
            ),
            branch_heading_break=branch_path.heading_break,
            branch_end_progress_m=branch_path.branch_end_progress_m,
            branch_compatible_target_progress_m=(
                branch_path.branch_compatible_target_progress_m
            ),
            heading_break_delta_deg=branch_path.heading_break_delta_deg,
            next_heading_error_deg=branch_path.next_heading_error_deg,
        )
        branch_end_check = None
        if branch_path.heading_break and branch_path.next_heading_deg is not None:
            branch_end_check = self._branch_frame_end_check(
                pose,
                branch_path,
                lock,
            )
            projection = self._with_branch_frame_end_metadata(
                projection,
                branch_end_check,
            )
            if branch_end_check.hard_cross_track_exceeded:
                self.post_rotate_branch_lock = None
                return ControllerStep(
                    TwistCommand(0.0, 0.0),
                    "off_route",
                    projection.projected_point,
                    distance_to_goal,
                    projection.heading_error_to_route_deg,
                    False,
                    route_projection_result=projection,
                    pure_pursuit_status="off_route",
                )
            if branch_end_check.handoff_allowed:
                self.post_rotate_branch_physical_handoff_count += 1
                projection = self._with_branch_frame_end_metadata(
                    projection,
                    branch_end_check,
                    "physical_branch_end",
                )
                return self._branch_heading_break_rotate_step(
                    pose,
                    projection,
                    branch_path,
                    distance_to_goal,
                )
        branch_end_delta_m = branch_path.branch_end_progress_m - projection.route_progress_m
        selected_branch_error = getattr(
            projection,
            "selected_branch_heading_error_deg",
            None,
        )
        still_on_locked_branch = (
            selected_branch_error is None
            or abs(float(selected_branch_error))
            <= POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG
        )
        at_branch_end = branch_path.heading_break and (
            0.0 <= branch_end_delta_m <= POST_ROTATE_BRANCH_END_TOLERANCE_M
            or (
                abs(branch_end_delta_m) <= POST_ROTATE_BRANCH_END_TOLERANCE_M
                and still_on_locked_branch
            )
        )
        if at_branch_end and branch_path.next_heading_deg is not None:
            return self._branch_heading_break_rotate_step(
                pose,
                projection,
                branch_path,
                distance_to_goal,
            )

        path_points = branch_path.path_points
        path_profile_result = self._path_profile_schedule(
            route_points,
            projection,
            min(lookahead_m, max(0.0, branch_path.compatible_length_m)),
        )
        target_point = lookahead_target_from_route_anchor(
            path_points,
            min(
                path_profile_result.lookahead_m,
                max(0.0, branch_path.compatible_length_m),
            ),
        )
        command_lookahead_m = max(
            0.01,
            min(path_profile_result.lookahead_m, branch_path.compatible_length_m),
        )
        geometry = pure_pursuit_geometry(pose, target_point, command_lookahead_m)
        heading_lookahead_m = min(
            ROUTE_HEADING_LOOKAHEAD_M,
            max(0.0, branch_path.compatible_length_m),
        )
        route_heading = route_heading_from_projection(
            route_points,
            projection,
            pose.yaw_deg,
            heading_lookahead_m,
        )
        alpha_rad = self._stable_alpha_for_rotate(geometry.alpha_rad)
        alpha_deg = math.degrees(alpha_rad)
        rotate_mode, rotate_reason, rotate_source, rotate_error_deg = (
            self._route_heading_rotate_gate(
                alpha_deg,
                route_heading,
                distance_to_goal,
                goal_tolerance_m,
            )
        )
        rotate_error_rad = (
            alpha_rad
            if rotate_source == "alpha"
            else math.radians(rotate_error_deg)
        )
        alpha_rotate_suppressed = (
            (suppress_alpha_rotate or lock.suppress_alpha_rotate)
            and rotate_source == "alpha"
        )
        if rotate_mode and not alpha_rotate_suppressed:
            self.mode = "rotate"
            self.post_rotate_branch_lock = None
            self._activate_rotate_anchor(projection, route_heading)
            angular_z = clamp(
                rotate_error_rad * self.args.yaw_gain,
                -abs(self.args.pure_pursuit_max_rotate_angular_speed_radps),
                abs(self.args.pure_pursuit_max_rotate_angular_speed_radps),
            )
            return ControllerStep(
                TwistCommand(0.0, angular_z),
                "rotate",
                target_point,
                distance_to_goal,
                rotate_error_deg,
                False,
                route_projection_result=projection,
                pure_pursuit_status="rotate_gate",
                route_heading_result=route_heading,
                pure_pursuit_rotate_reason=rotate_reason,
                pure_pursuit_rotate_source=rotate_source,
                path_profile_result=path_profile_result,
            )

        self.mode = "forward"
        guard_result = None
        if self.lookahead_guard is not None:
            target_point, guard_result = self.lookahead_guard.select_target(
                pose,
                path_points,
                current_point,
                target_point,
                max(0.0, path_profile_result.lookahead_m),
                getattr(self.args, "pure_pursuit_min_guarded_lookahead_m", 0.12),
                final_goal=final_goal,
                distance_to_goal_m=distance_to_goal,
            )
            if not guard_result.safe:
                self.reset_route_projection_state()
                target_heading_deg = math.degrees(
                    math.atan2(
                        target_point[1] - pose.y,
                        target_point[0] - pose.x,
                    )
                )
                return ControllerStep(
                    TwistCommand(0.0, 0.0),
                    "blocked",
                    target_point,
                    distance_to_goal,
                    shortest_angle_delta_deg(pose.yaw_deg, target_heading_deg),
                    False,
                    guard_result,
                    route_heading_result=route_heading,
                    path_profile_result=path_profile_result,
                )
            if guard_result.selected_target_distance_m is not None:
                command_lookahead_m = guard_result.selected_target_distance_m
            geometry = pure_pursuit_geometry(pose, target_point, command_lookahead_m)
            alpha_rad = geometry.alpha_rad

        linear_x, angular_z, alpha_rad, mode, velocity_schedule_result, forward_control_result = (
            self._forward_command_from_geometry(
                pose,
                target_point,
                command_lookahead_m,
                geometry,
                route_heading=route_heading,
                projection=projection,
                allow_route_damped=True,
                speed_cap_mps=path_profile_result.speed_cap_mps,
            )
        )
        if abs(linear_x) <= 1e-12 and abs(angular_z) <= 1e-12:
            self._reset_path_profile_state()
        if (
            branch_path.branch_target_clipped_to_heading_break
            and alpha_rotate_suppressed
            and forward_control_result is not None
            and abs(float(forward_control_result.linear_after_feasibility_mps))
            <= POST_ROTATE_ZERO_LINEAR_EPS_MPS
        ):
            branch_end_check = self._branch_frame_end_check(
                pose,
                branch_path,
                lock,
            )
            projection = self._with_branch_frame_end_metadata(
                projection,
                branch_end_check,
            )
            if branch_end_check.hard_cross_track_exceeded:
                self.post_rotate_branch_lock = None
                return ControllerStep(
                    TwistCommand(0.0, 0.0),
                    "off_route",
                    projection.projected_point,
                    distance_to_goal,
                    projection.heading_error_to_route_deg,
                    False,
                    route_projection_result=projection,
                    pure_pursuit_status="off_route",
                )
            if branch_end_check.handoff_allowed:
                self.post_rotate_branch_physical_handoff_count += 1
                projection = self._with_branch_frame_end_metadata(
                    projection,
                    branch_end_check,
                    "physical_branch_end_zero_linear",
                )
                return self._branch_heading_break_rotate_step(
                    pose,
                    projection,
                    branch_path,
                    distance_to_goal,
                )
        self._accept_projection(projection, route_state)
        if lock.stable_count >= POST_ROTATE_BRANCH_RELEASE_STABLE_SAMPLES:
            self.post_rotate_branch_lock = None
        return ControllerStep(
            TwistCommand(linear_x, angular_z),
            mode,
            target_point,
            distance_to_goal,
            math.degrees(alpha_rad),
            False,
            guard_result,
            velocity_schedule_result,
            projection,
            "forward",
            route_heading,
            forward_control_result=forward_control_result,
            path_profile_result=path_profile_result,
        )

    def _branch_frame_end_check(self, pose, branch_path, lock):
        heading_rad = float(lock.preferred_heading_rad)
        branch_dir_x = math.cos(heading_rad)
        branch_dir_y = math.sin(heading_rad)
        delta_x = float(pose.x) - float(branch_path.branch_end_point[0])
        delta_y = float(pose.y) - float(branch_path.branch_end_point[1])
        along_past_m = delta_x * branch_dir_x + delta_y * branch_dir_y
        lateral_error_m = abs(branch_dir_x * delta_y - branch_dir_y * delta_x)
        hard_cross_track_m = abs(
            float(getattr(self.args, "pure_pursuit_max_cross_track_error_m", 0.25))
        )
        lateral_tolerance_m = min(
            POST_ROTATE_BRANCH_END_LATERAL_TOLERANCE_M,
            hard_cross_track_m,
        )
        preferred_heading_deg = math.degrees(heading_rad)
        yaw_error_deg = shortest_angle_delta_deg(
            float(pose.yaw_deg),
            preferred_heading_deg,
        )
        hard_cross_track_exceeded = lateral_error_m > hard_cross_track_m
        handoff_allowed = (
            branch_path.heading_break
            and branch_path.next_heading_deg is not None
            and along_past_m >= -POST_ROTATE_BRANCH_END_TOLERANCE_M
            and lateral_error_m <= lateral_tolerance_m
            and abs(yaw_error_deg) <= POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG
            and not hard_cross_track_exceeded
        )
        return BranchFrameEndCheck(
            along_past_m,
            lateral_error_m,
            lateral_tolerance_m,
            yaw_error_deg,
            hard_cross_track_exceeded,
            handoff_allowed,
        )

    @staticmethod
    def _with_branch_frame_end_metadata(
        projection,
        branch_end_check,
        handoff_reason="",
    ):
        return replace(
            projection,
            branch_end_along_past_m=branch_end_check.along_past_m,
            branch_end_lateral_error_m=branch_end_check.lateral_error_m,
            branch_end_handoff_reason=handoff_reason,
            branch_end_handoff_lateral_tolerance_m=(
                branch_end_check.lateral_tolerance_m
            ),
        )

    def _branch_heading_break_handoff_from_lock(
        self,
        pose,
        route_points,
        lock,
        distance_to_goal,
        release_span_m,
    ):
        route = [(float(x), float(y)) for x, y in route_points]
        if len(route) < 2:
            return None
        cumulative = route_cumulative_distances(route)
        if not cumulative or cumulative[-1] <= 1e-9:
            return None
        progress_m = clamp(float(lock.last_progress_m), 0.0, cumulative[-1])
        x, y, segment_index, segment_ratio = route_point_at_progress(
            route,
            cumulative,
            progress_m,
        )
        projected_point = (x, y)
        cross_track_m = math.hypot(float(pose.x) - x, float(pose.y) - y)
        preferred_heading_deg = math.degrees(lock.preferred_heading_rad)
        has_preceding_compatible_segment = False
        for index in range(min(segment_index, len(route) - 2), -1, -1):
            segment_start_progress = cumulative[index]
            segment_end_progress = cumulative[index + 1]
            if segment_start_progress > progress_m + 1e-9:
                continue
            segment_length = segment_end_progress - segment_start_progress
            if segment_length <= 1e-9:
                continue
            start = route[index]
            end = route[index + 1]
            preceding_heading_deg = math.degrees(
                math.atan2(end[1] - start[1], end[0] - start[0])
            )
            if (
                abs(shortest_angle_delta_deg(preferred_heading_deg, preceding_heading_deg))
                <= POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG
            ):
                has_preceding_compatible_segment = True
                break
        if not has_preceding_compatible_segment:
            return None
        projection = RouteProjection(
            projected_point,
            segment_index,
            segment_ratio,
            progress_m,
            preferred_heading_deg,
            shortest_angle_delta_deg(float(pose.yaw_deg), preferred_heading_deg),
            cross_track_m,
            0.0,
            max(0.0, cumulative[-1] - progress_m),
            projection_status="post_rotate_branch_lock",
            route_progress_delta_m=0.0,
            raw_projection_progress_m=progress_m,
            raw_projection_segment_index=segment_index,
            effective_projection_progress_m=progress_m,
            preferred_branch_heading_deg=preferred_heading_deg,
            selected_segment_heading_deg=preferred_heading_deg,
            selected_branch_heading_error_deg=0.0,
            branch_lock_stable_count=int(lock.stable_count),
            branch_lock_progress_span_m=max(0.0, progress_m - lock.start_progress_m),
            branch_lock_release_required_span_m=float(release_span_m),
            local_cross_track_m=cross_track_m,
            rotate_anchor_route_heading_aligned_samples=int(
                lock.rotate_anchor_aligned_sample_count
            ),
            rotate_anchor_handoff_reason=str(lock.rotate_anchor_handoff_reason),
        )
        if (
            projection.cross_track_error_m
            > getattr(self.args, "pure_pursuit_max_cross_track_error_m", 0.25)
        ):
            self.post_rotate_branch_lock = None
            return ControllerStep(
                TwistCommand(0.0, 0.0),
                "off_route",
                projection.projected_point,
                distance_to_goal,
                projection.heading_error_to_route_deg,
                False,
                route_projection_result=projection,
                pure_pursuit_status="off_route",
            )
        branch_path = branch_compatible_path_from_projection(
            route,
            projection,
            preferred_heading_deg,
            POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG,
            0.0,
            current_yaw_deg=pose.yaw_deg,
        )
        if (
            not branch_path.heading_break
            or branch_path.next_heading_deg is None
            or branch_path.compatible_length_m > POST_ROTATE_BRANCH_END_TOLERANCE_M
        ):
            return None
        projection = replace(
            projection,
            branch_compatible_length_m=branch_path.compatible_length_m,
            branch_target_clipped_to_heading_break=(
                branch_path.branch_target_clipped_to_heading_break
            ),
            branch_heading_break=branch_path.heading_break,
            branch_end_progress_m=branch_path.branch_end_progress_m,
            branch_compatible_target_progress_m=(
                branch_path.branch_compatible_target_progress_m
            ),
            heading_break_delta_deg=branch_path.heading_break_delta_deg,
            next_heading_error_deg=branch_path.next_heading_error_deg,
        )
        return self._branch_heading_break_rotate_step(
            pose,
            projection,
            branch_path,
            distance_to_goal,
        )

    def _branch_heading_break_rotate_step(
        self,
        pose,
        projection,
        branch_path,
        distance_to_goal,
    ):
        self.mode = "rotate"
        self.post_rotate_branch_lock = None
        self.post_rotate_branch_heading_break_handoff_count += 1
        next_heading_error_deg = shortest_angle_delta_deg(
            float(pose.yaw_deg),
            float(branch_path.next_heading_deg),
        )
        route_heading = RouteHeading(
            float(branch_path.next_heading_deg),
            next_heading_error_deg,
            "branch_heading_break",
            branch_path.branch_end_point,
            0.0,
        )
        self._activate_rotate_anchor(projection, route_heading)
        angular_z = clamp(
            math.radians(next_heading_error_deg) * self.args.yaw_gain,
            -abs(self.args.pure_pursuit_max_rotate_angular_speed_radps),
            abs(self.args.pure_pursuit_max_rotate_angular_speed_radps),
        )
        return ControllerStep(
            TwistCommand(0.0, angular_z),
            "rotate",
            branch_path.branch_end_point,
            distance_to_goal,
            next_heading_error_deg,
            False,
            route_projection_result=projection,
            pure_pursuit_status="rotate_gate",
            route_heading_result=route_heading,
            pure_pursuit_rotate_reason="branch_heading_break",
            pure_pursuit_rotate_source="route_heading",
        )

    def _path_profile_force_rotate_step(
        self,
        pose,
        projection,
        path_profile_result,
        distance_to_goal,
    ):
        if (
            path_profile_result.branch_end_point is None
            or path_profile_result.next_heading_deg is None
        ):
            return ControllerStep(
                TwistCommand(0.0, 0.0),
                "blocked",
                projection.projected_point,
                distance_to_goal,
                projection.heading_error_to_route_deg,
                False,
                route_projection_result=projection,
                pure_pursuit_status="path_profile_force_rotate_unavailable",
                path_profile_result=path_profile_result,
            )
        self.mode = "rotate"
        self._reset_path_profile_state()
        next_heading_error_deg = shortest_angle_delta_deg(
            float(pose.yaw_deg),
            float(path_profile_result.next_heading_deg),
        )
        route_heading = RouteHeading(
            float(path_profile_result.next_heading_deg),
            next_heading_error_deg,
            "path_profile_heading_break",
            path_profile_result.branch_end_point,
            0.0,
        )
        self._activate_rotate_anchor(projection, route_heading)
        angular_z = clamp(
            math.radians(next_heading_error_deg) * self.args.yaw_gain,
            -abs(self.args.pure_pursuit_max_rotate_angular_speed_radps),
            abs(self.args.pure_pursuit_max_rotate_angular_speed_radps),
        )
        return ControllerStep(
            TwistCommand(0.0, angular_z),
            "rotate",
            path_profile_result.branch_end_point,
            distance_to_goal,
            next_heading_error_deg,
            False,
            route_projection_result=projection,
            pure_pursuit_status="rotate_gate",
            route_heading_result=route_heading,
            pure_pursuit_rotate_reason="branch_heading_break",
            pure_pursuit_rotate_source="route_heading",
            path_profile_result=path_profile_result,
        )

    def _forward_command_from_geometry(
        self,
        pose,
        target_point,
        command_lookahead_m,
        geometry,
        route_heading=None,
        projection=None,
        allow_route_damped=False,
        speed_cap_mps=None,
    ):
        speed_profile = getattr(
            self.args,
            "pure_pursuit_speed_profile",
            SPEED_PROFILE_FIXED,
        )
        velocity_schedule_result = None
        forward_control_result = None
        if speed_profile == SPEED_PROFILE_FIXED:
            linear_x, angular_z, alpha_rad, forward_control_result = (
                self._fixed_forward_command(
                    pose,
                    target_point,
                    command_lookahead_m,
                    route_heading,
                    projection,
                    geometry,
                    allow_route_damped=allow_route_damped,
                    speed_cap_mps=speed_cap_mps,
                )
            )
            mode = "forward"
        else:
            velocity_schedule_result = self.velocity_scheduler.schedule(
                geometry,
                allow_rotate=False,
                linear_speed_cap_mps=speed_cap_mps,
            )
            linear_x = velocity_schedule_result.command.linear_x
            angular_z = velocity_schedule_result.command.angular_z
            alpha_rad = geometry.alpha_rad
            mode = velocity_schedule_result.mode
        return linear_x, angular_z, alpha_rad, mode, velocity_schedule_result, forward_control_result

    def _fixed_forward_command(
        self,
        pose,
        target_point,
        command_lookahead_m,
        route_heading,
        projection,
        geometry,
        allow_route_damped,
        speed_cap_mps=None,
    ):
        speed_cap = abs(
            float(
                self.args.linear_speed
                if speed_cap_mps is None
                else speed_cap_mps
            )
        )
        forward_control = getattr(
            self.args,
            "pure_pursuit_forward_control",
            FORWARD_CONTROL_TARGET_BEARING,
        )
        if not allow_route_damped or forward_control != FORWARD_CONTROL_ROUTE_DAMPED:
            linear_x, angular_z, alpha_rad = pure_pursuit_curve_command(
                pose,
                target_point,
                command_lookahead_m,
                speed_cap,
                self.args.pure_pursuit_max_track_angular_speed_radps,
                self.args.pure_pursuit_rotate_start_heading_error_deg,
            )
            return linear_x, angular_z, alpha_rad, ForwardControlResult(
                FORWARD_CONTROL_TARGET_BEARING,
                "",
                math.degrees(alpha_rad),
                None,
                None,
                0.0,
                math.degrees(alpha_rad),
                math.degrees(abs(alpha_rad)),
                angular_z,
                angular_z,
                False,
                1.0,
                linear_x,
                linear_x,
            )

        if (
            route_heading is None
            or route_heading.heading_error_deg is None
            or projection is None
        ):
            linear_x, angular_z, alpha_rad = pure_pursuit_curve_command(
                pose,
                target_point,
                command_lookahead_m,
                speed_cap,
                self.args.pure_pursuit_max_track_angular_speed_radps,
                self.args.pure_pursuit_rotate_start_heading_error_deg,
            )
            return linear_x, angular_z, alpha_rad, ForwardControlResult(
                FORWARD_CONTROL_TARGET_BEARING,
                "route_heading_unavailable",
                math.degrees(alpha_rad),
                None,
                None,
                0.0,
                math.degrees(alpha_rad),
                math.degrees(abs(alpha_rad)),
                angular_z,
                angular_z,
                False,
                1.0,
                linear_x,
                linear_x,
            )

        return self._route_damped_fixed_forward_command(
            geometry,
            route_heading,
            projection,
            speed_cap_mps=speed_cap,
        )

    def _route_damped_fixed_forward_command(
        self,
        geometry,
        route_heading,
        projection,
        speed_cap_mps=None,
    ):
        alpha_rad = geometry.alpha_rad
        alpha_deg = math.degrees(alpha_rad)
        route_heading_error_deg = float(route_heading.heading_error_deg)
        signed_cross_track_error_m = float(projection.signed_cross_track_error_m)
        speed_cap = abs(
            float(
                self.args.linear_speed
                if speed_cap_mps is None
                else speed_cap_mps
            )
        )
        cte_speed_ref = max(
            speed_cap,
            float(self.args.pure_pursuit_cross_track_speed_floor_mps),
        )
        cte_correction_rad = -math.atan2(
            float(self.args.pure_pursuit_cross_track_gain)
            * signed_cross_track_error_m,
            cte_speed_ref,
        )
        cte_correction_deg = clamp(
            math.degrees(cte_correction_rad),
            -abs(float(self.args.pure_pursuit_max_cross_track_correction_deg)),
            abs(float(self.args.pure_pursuit_max_cross_track_correction_deg)),
        )
        route_error_rad = normalize_angle_rad(
            math.radians(route_heading_error_deg + cte_correction_deg)
        )
        blended_error_rad = self._circular_blend_error(
            alpha_rad,
            route_error_rad,
            float(self.args.pure_pursuit_route_heading_blend),
        )
        speed_taper_error_rad = max(
            abs(blended_error_rad),
            abs(math.radians(route_heading_error_deg)),
        )
        linear_x = self._fixed_linear_speed_for_error(
            speed_taper_error_rad,
            speed_cap,
        )
        raw_angular_z = blended_error_rad * self.args.yaw_gain
        linear_before_feasibility_mps = linear_x
        angular_z = clamp(
            raw_angular_z,
            -abs(self.args.pure_pursuit_max_track_angular_speed_radps),
            abs(self.args.pure_pursuit_max_track_angular_speed_radps),
        )
        (
            linear_x,
            angular_feasibility_limited,
            angular_feasibility_scale,
        ) = self._apply_angular_feasibility_limit(linear_x, raw_angular_z)
        return linear_x, angular_z, blended_error_rad, ForwardControlResult(
            FORWARD_CONTROL_ROUTE_DAMPED,
            "",
            alpha_deg,
            route_heading_error_deg,
            signed_cross_track_error_m,
            cte_correction_deg,
            math.degrees(blended_error_rad),
            math.degrees(speed_taper_error_rad),
            raw_angular_z,
            angular_z,
            angular_feasibility_limited,
            angular_feasibility_scale,
            linear_before_feasibility_mps,
            linear_x,
        )

    def _apply_angular_feasibility_limit(self, linear_x, raw_angular_z):
        if (
            getattr(self.args, "pure_pursuit_angular_feasibility_speed_limit", "on")
            != "on"
        ):
            return linear_x, False, 1.0
        track_cap = abs(float(self.args.pure_pursuit_max_track_angular_speed_radps))
        raw_abs = abs(float(raw_angular_z))
        if raw_abs <= track_cap or raw_abs <= 1e-12:
            return linear_x, False, 1.0
        margin = float(self.args.pure_pursuit_angular_feasibility_margin)
        scale = clamp(track_cap * margin / raw_abs, 0.0, 1.0)
        limited_linear = min(float(linear_x), float(linear_x) * scale)
        limited_linear = max(0.0, limited_linear)
        if limited_linear < ANGULAR_FEASIBILITY_STOP_EPS_MPS:
            limited_linear = 0.0
        return limited_linear, True, scale

    @staticmethod
    def _circular_blend_error(alpha_rad, route_error_rad, blend):
        weight = clamp(float(blend), 0.0, 1.0)
        x = (1.0 - weight) * math.cos(alpha_rad) + weight * math.cos(route_error_rad)
        y = (1.0 - weight) * math.sin(alpha_rad) + weight * math.sin(route_error_rad)
        if math.hypot(x, y) < 1e-6:
            return normalize_angle_rad(route_error_rad)
        return normalize_angle_rad(math.atan2(y, x))

    def _fixed_linear_speed_for_error(self, error_rad, speed_cap_mps=None):
        speed_cap = abs(
            float(
                self.args.linear_speed
                if speed_cap_mps is None
                else speed_cap_mps
            )
        )
        rotate_start_rad = max(
            1e-6,
            math.radians(abs(self.args.pure_pursuit_rotate_start_heading_error_deg)),
        )
        abs_error = abs(float(error_rad))
        if abs_error >= rotate_start_rad:
            return 0.0
        start_cos = math.cos(rotate_start_rad)
        denominator = max(1e-6, 1.0 - start_cos)
        linear_scale = clamp(
            (math.cos(abs_error) - start_cos) / denominator,
            0.0,
            1.0,
        )
        return speed_cap * linear_scale

    def _accept_projection(self, projection, route_state):
        self.last_projection_segment_index = projection.segment_index
        route_state.tracking_progress_index = max(
            route_state.tracking_progress_index,
            projection.segment_index,
        )
        self.last_route_progress_m = projection.route_progress_m
        self.last_accepted_projection = projection
        self._update_projection_lock(projection)

    def _activate_rotate_anchor(self, projection, route_heading):
        if self.rotate_projection_anchor is not None:
            return
        self._reset_path_profile_state()
        heading_deg = (
            route_heading.heading_deg
            if route_heading is not None and route_heading.heading_deg is not None
            else projection.route_heading_deg
        )
        self.rotate_projection_anchor = RotateProjectionAnchor(
            progress_m=float(projection.route_progress_m),
            segment_index=int(projection.segment_index),
            segment_ratio=float(projection.segment_ratio),
            projected_point=(
                float(projection.projected_point[0]),
                float(projection.projected_point[1]),
            ),
            route_heading_rad=math.radians(float(heading_deg)),
            cross_track_m=float(projection.cross_track_error_m),
        )
        self.rotate_anchor_activations += 1

    def _activate_post_rotate_branch_lock(self, anchor):
        if self.post_rotate_branch_lock is not None:
            return
        heading_rad = float(anchor.route_heading_rad)
        self.post_rotate_branch_lock = PostRotateBranchLock(
            preferred_heading_rad=heading_rad,
            last_progress_m=float(anchor.progress_m),
            last_segment_index=int(anchor.segment_index),
            start_progress_m=float(anchor.progress_m),
            rotate_anchor_aligned_sample_count=(
                int(getattr(anchor, "route_heading_aligned_sample_count", 0))
            ),
            rotate_anchor_handoff_reason=str(
                getattr(anchor, "handoff_reason", "")
            ),
            suppress_alpha_rotate=(
                getattr(anchor, "handoff_reason", "") == "route_heading_stable"
            ),
        )
        self.post_rotate_branch_lock_activations += 1

    def _update_rotate_anchor_alignment(self, anchor, route_heading):
        route_error = (
            route_heading.heading_error_deg if route_heading is not None else None
        )
        if (
            route_error is not None
            and abs(float(route_error))
            <= self.args.pure_pursuit_route_heading_rotate_stop_deg
        ):
            anchor.route_heading_aligned_sample_count += 1
        else:
            anchor.route_heading_aligned_sample_count = 0
        anchor.max_route_heading_aligned_sample_count = max(
            anchor.max_route_heading_aligned_sample_count,
            anchor.route_heading_aligned_sample_count,
        )
        self.last_rotate_anchor_aligned_samples = (
            anchor.route_heading_aligned_sample_count
        )
        self.max_rotate_anchor_aligned_samples = max(
            self.max_rotate_anchor_aligned_samples,
            anchor.max_route_heading_aligned_sample_count,
        )

    @staticmethod
    def _rotate_anchor_stable_handoff_ready(anchor):
        return (
            anchor.route_heading_aligned_sample_count
            >= ROTATE_ANCHOR_ROUTE_HEADING_EXIT_SAMPLES
        )

    def _branch_release_probe_safe(self, route_points, pose, lock, lookahead_m):
        try:
            probe = project_point_to_route(
                route_points,
                pose,
                start_segment_index=max(0, int(lock.last_segment_index) - 1),
                previous_progress_m=lock.last_progress_m,
                max_forward_jump_m=max(2.0 * lookahead_m, 0.35),
                backward_tolerance_m=PROJECTION_LOCK_PROGRESS_TOLERANCE_M,
                allow_backward=False,
                projection_status="branch_release_probe",
            )
        except RuntimeError:
            return False
        if probe.route_progress_backward_delta_m > 1e-6:
            return False
        if probe.route_progress_m + 1e-9 < lock.last_progress_m:
            return False
        if (
            probe.cross_track_error_m
            > getattr(self.args, "pure_pursuit_max_cross_track_error_m", 0.25)
        ):
            return False
        preferred_heading = math.degrees(lock.preferred_heading_rad)
        smoothed_heading = route_heading_at_progress(
            route_points,
            probe.route_progress_m,
            ROUTE_HEADING_LOOKAHEAD_M,
        )
        if smoothed_heading is None:
            return False
        smoothed_error = abs(shortest_angle_delta_deg(preferred_heading, smoothed_heading))
        raw_error = abs(shortest_angle_delta_deg(preferred_heading, probe.route_heading_deg))
        return (
            smoothed_error <= POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG
            and raw_error <= POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG
        )

    @staticmethod
    def _post_rotate_branch_release_span_m(lookahead_m):
        return max(float(lookahead_m), POST_ROTATE_BRANCH_MIN_RELEASE_PROGRESS_M)

    @staticmethod
    def _post_rotate_branch_max_span_m(lookahead_m):
        release_span_m = PurePursuitController._post_rotate_branch_release_span_m(
            lookahead_m
        )
        return max(2.0 * float(lookahead_m), 0.35, release_span_m + 0.10)

    @staticmethod
    def _with_branch_lock_metadata(
        projection,
        stable_count,
        release_span_m,
        rotate_anchor_aligned_samples=0,
        rotate_anchor_handoff_reason="",
    ):
        return replace(
            projection,
            branch_lock_stable_count=int(stable_count),
            branch_lock_release_required_span_m=float(release_span_m),
            rotate_anchor_route_heading_aligned_samples=int(
                rotate_anchor_aligned_samples
            ),
            rotate_anchor_handoff_reason=str(rotate_anchor_handoff_reason),
        )

    def _route_heading_from_rotate_anchor(self, pose):
        anchor = self.rotate_projection_anchor
        if anchor is None:
            return RouteHeading(None, None, "unavailable")
        heading_deg = math.degrees(anchor.route_heading_rad)
        return RouteHeading(
            heading_deg,
            shortest_angle_delta_deg(float(pose.yaw_deg), heading_deg),
            "rotate_anchor",
            anchor.projected_point,
            0.0,
        )

    def _projection_from_rotate_anchor(self, route_points, pose, raw_projection):
        anchor = self.rotate_projection_anchor
        if anchor is None:
            raise RuntimeError("rotate_projection_anchor_missing")
        cumulative = route_cumulative_distances(route_points)
        total_length = cumulative[-1] if cumulative else anchor.progress_m
        heading_deg = math.degrees(anchor.route_heading_rad)
        heading_error = shortest_angle_delta_deg(float(pose.yaw_deg), heading_deg)
        raw_progress = float(raw_projection.route_progress_m)
        progress_delta = raw_progress - anchor.progress_m
        backward_delta = max(0.0, -progress_delta)
        forward_delta = max(0.0, progress_delta)
        anchor.max_backward_delta_m = max(anchor.max_backward_delta_m, backward_delta)
        anchor.max_forward_delta_m = max(anchor.max_forward_delta_m, forward_delta)
        self.max_rotate_anchor_backward_delta_m = max(
            self.max_rotate_anchor_backward_delta_m,
            anchor.max_backward_delta_m,
        )
        self.max_rotate_anchor_forward_delta_m = max(
            self.max_rotate_anchor_forward_delta_m,
            anchor.max_forward_delta_m,
        )
        return RouteProjection(
            projected_point=anchor.projected_point,
            segment_index=anchor.segment_index,
            segment_ratio=anchor.segment_ratio,
            route_progress_m=anchor.progress_m,
            route_heading_deg=heading_deg,
            heading_error_to_route_deg=heading_error,
            cross_track_error_m=raw_projection.cross_track_error_m,
            signed_cross_track_error_m=raw_projection.signed_cross_track_error_m,
            remaining_route_m=max(0.0, total_length - anchor.progress_m),
            projection_status="rotate_anchor",
            route_progress_delta_m=0.0,
            route_progress_backward_delta_m=0.0,
            route_progress_forward_delta_m=0.0,
            raw_projection_progress_m=raw_progress,
            raw_projection_segment_index=raw_projection.segment_index,
            effective_projection_progress_m=anchor.progress_m,
            anchor_progress_m=anchor.progress_m,
            anchor_segment_index=anchor.segment_index,
            rotate_anchor_backward_delta_m=backward_delta,
            rotate_anchor_forward_delta_m=forward_delta,
            rotate_anchor_route_heading_aligned_samples=(
                anchor.route_heading_aligned_sample_count
            ),
            rotate_anchor_handoff_reason=anchor.handoff_reason,
            local_cross_track_m=raw_projection.cross_track_error_m,
        )

    def _stable_alpha_for_rotate(self, alpha_rad):
        if abs(abs(alpha_rad) - math.pi) <= math.radians(0.5):
            return math.copysign(math.pi, self.last_rotate_sign)
        if abs(alpha_rad) > 1e-9:
            self.last_rotate_sign = 1.0 if alpha_rad > 0.0 else -1.0
        return alpha_rad

    def _update_projection_lock(self, projection):
        if self.projection_locked:
            self.projection_lock_sample_count = max(
                self.projection_lock_sample_count,
                PROJECTION_LOCK_REQUIRED_SAMPLES,
            )
            return
        progress_delta = projection.route_progress_delta_m
        stable_progress = (
            progress_delta is None
            or abs(progress_delta) <= PROJECTION_LOCK_PROGRESS_TOLERANCE_M
        )
        below_warning = (
            projection.cross_track_error_m
            <= getattr(self.args, "pure_pursuit_cross_track_warning_m", 0.15)
        )
        if below_warning or stable_progress:
            self.projection_lock_sample_count += 1
        else:
            self.projection_lock_sample_count = 0
        if self.projection_lock_sample_count >= PROJECTION_LOCK_REQUIRED_SAMPLES:
            self.projection_locked = True

    def _route_heading_rotate_gate(
        self,
        alpha_deg,
        route_heading,
        distance_to_goal_m,
        goal_tolerance_m,
    ):
        route_error = route_heading.heading_error_deg
        route_gate_available = (
            route_error is not None
            and distance_to_goal_m > goal_tolerance_m
        )
        alpha_limit = (
            self.args.pure_pursuit_rotate_stop_heading_error_deg
            if self.mode == "rotate"
            else self.args.pure_pursuit_rotate_start_heading_error_deg
        )
        route_limit = (
            self.args.pure_pursuit_route_heading_rotate_stop_deg
            if self.mode == "rotate"
            else self.args.pure_pursuit_route_heading_rotate_start_deg
        )
        alpha_active = abs(alpha_deg) > alpha_limit
        route_active = route_gate_available and abs(route_error) > route_limit
        if not (alpha_active or route_active):
            return False, "", "alpha", alpha_deg
        if alpha_active and route_active:
            reason = "both"
        elif route_active:
            reason = "route_heading"
        else:
            reason = "alpha"
        if (
            route_gate_available
            and abs(route_error) > self.args.pure_pursuit_route_heading_rotate_stop_deg
        ):
            return True, reason, "route_heading", route_error
        return True, reason, "alpha", alpha_deg


def build_path_controller(args, lookahead_guard=None) -> PathController:
    controller = getattr(args, "controller", "stop-go")
    if controller == "pure-pursuit":
        return PurePursuitController(args, lookahead_guard=lookahead_guard)
    if controller == "stop-go":
        return StopGoController(args)
    raise ValueError(f"unsupported controller: {controller!r}")
