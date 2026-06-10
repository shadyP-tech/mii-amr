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
    lookahead_target_from_route_anchor,
    project_point_to_route,
    project_point_to_route_branch_window,
    project_point_to_route_progress_window,
    pure_pursuit_curve_command,
    route_cumulative_distances,
    route_heading_at_progress,
    route_heading_from_projection,
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
POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG = 60.0
POST_ROTATE_BRANCH_RELEASE_STABLE_SAMPLES = 2
POST_ROTATE_BRANCH_MIN_RELEASE_PROGRESS_M = 0.18
FORWARD_CONTROL_TARGET_BEARING = "target-bearing"
FORWARD_CONTROL_ROUTE_DAMPED = "route-damped"
FORWARD_CONTROL_MODES = (
    FORWARD_CONTROL_TARGET_BEARING,
    FORWARD_CONTROL_ROUTE_DAMPED,
)


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
        self.rotate_anchor_activations = 0
        self.last_accepted_projection = None
        self.rotate_projection_anchor = None
        self.post_rotate_branch_lock = None
        self.post_rotate_branch_lock_activations = 0
        self.post_rotate_branch_ambiguity_failures = 0
        self.post_rotate_branch_rejected_wrong_heading_count = 0
        self.post_rotate_branch_max_heading_error_deg = 0.0
        self.last_rotate_sign = 1.0
        self.rotate_gate_entries = 0

    def reset_route_projection_state(self):
        self.last_projection_segment_index = None
        self.last_route_progress_m = None
        self.projection_locked = False
        self.projection_lock_sample_count = 0
        self.max_projection_backward_delta_m = 0.0
        self.max_rotate_anchor_backward_delta_m = 0.0
        self.max_rotate_anchor_forward_delta_m = 0.0
        self.rotate_anchor_activations = 0
        self.last_accepted_projection = None
        self.rotate_projection_anchor = None
        self.post_rotate_branch_lock = None
        self.post_rotate_branch_lock_activations = 0
        self.post_rotate_branch_ambiguity_failures = 0
        self.post_rotate_branch_rejected_wrong_heading_count = 0
        self.post_rotate_branch_max_heading_error_deg = 0.0

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
            max(
                min(self.args.waypoint_tolerance_m, lookahead_m),
                getattr(self.args, "pure_pursuit_max_cross_track_error_m", 0.25),
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

        path_points = route_points_from_projection(route_points, projection)
        target_point = lookahead_target_from_route_anchor(
            path_points,
            min(lookahead_m, max(0.0, projection.remaining_route_m)),
        )
        command_lookahead_m = max(0.01, min(lookahead_m, projection.remaining_route_m))
        geometry = pure_pursuit_geometry(pose, target_point, command_lookahead_m)
        route_heading = route_heading_from_projection(
            route_points,
            projection,
            pose.yaw_deg,
            ROUTE_HEADING_LOOKAHEAD_M,
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
            )
        self.mode = "forward"
        guard_result = None
        if self.lookahead_guard is not None:
            target_point, guard_result = self.lookahead_guard.select_target(
                pose,
                path_points,
                current_point,
                target_point,
                lookahead_m,
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
            )
            mode = "forward"
        else:
            velocity_schedule_result = self.velocity_scheduler.schedule(
                geometry,
                allow_rotate=False,
            )
            linear_x = velocity_schedule_result.command.linear_x
            angular_z = velocity_schedule_result.command.angular_z
            alpha_rad = geometry.alpha_rad
            mode = velocity_schedule_result.mode
            forward_control_result = None
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

        projection = self._projection_from_rotate_anchor(
            route_points,
            pose,
            raw_projection,
        )
        route_heading = self._route_heading_from_rotate_anchor(pose)
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
            self.rotate_projection_anchor = None
            raise RuntimeError("route_projection_moved_backward")

        rotate_error_rad = (
            alpha_rad
            if rotate_source == "alpha"
            else math.radians(rotate_error_deg)
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
        self._activate_post_rotate_branch_lock(anchor)
        guard_result = None
        if self.lookahead_guard is not None:
            target_point, guard_result = self.lookahead_guard.select_target(
                pose,
                path_points,
                current_point,
                target_point,
                lookahead_m,
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
            )
        )
        self._accept_projection(projection, route_state)
        self.rotate_projection_anchor = None
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
        )

        path_points = route_points_from_projection(route_points, projection)
        target_point = lookahead_target_from_route_anchor(
            path_points,
            min(lookahead_m, max(0.0, projection.remaining_route_m)),
        )
        command_lookahead_m = max(0.01, min(lookahead_m, projection.remaining_route_m))
        geometry = pure_pursuit_geometry(pose, target_point, command_lookahead_m)
        route_heading = route_heading_from_projection(
            route_points,
            projection,
            pose.yaw_deg,
            ROUTE_HEADING_LOOKAHEAD_M,
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
            )

        self.mode = "forward"
        guard_result = None
        if self.lookahead_guard is not None:
            target_point, guard_result = self.lookahead_guard.select_target(
                pose,
                path_points,
                current_point,
                target_point,
                lookahead_m,
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
            )
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
                )
            )
            mode = "forward"
        else:
            velocity_schedule_result = self.velocity_scheduler.schedule(
                geometry,
                allow_rotate=False,
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
    ):
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
                self.args.linear_speed,
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
                self.args.linear_speed,
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
            )

        return self._route_damped_fixed_forward_command(
            geometry,
            route_heading,
            projection,
        )

    def _route_damped_fixed_forward_command(self, geometry, route_heading, projection):
        alpha_rad = geometry.alpha_rad
        alpha_deg = math.degrees(alpha_rad)
        route_heading_error_deg = float(route_heading.heading_error_deg)
        signed_cross_track_error_m = float(projection.signed_cross_track_error_m)
        cte_speed_ref = max(
            abs(float(self.args.linear_speed)),
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
        linear_x = self._fixed_linear_speed_for_error(speed_taper_error_rad)
        raw_angular_z = blended_error_rad * self.args.yaw_gain
        angular_z = clamp(
            raw_angular_z,
            -abs(self.args.pure_pursuit_max_track_angular_speed_radps),
            abs(self.args.pure_pursuit_max_track_angular_speed_radps),
        )
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
        )

    @staticmethod
    def _circular_blend_error(alpha_rad, route_error_rad, blend):
        weight = clamp(float(blend), 0.0, 1.0)
        x = (1.0 - weight) * math.cos(alpha_rad) + weight * math.cos(route_error_rad)
        y = (1.0 - weight) * math.sin(alpha_rad) + weight * math.sin(route_error_rad)
        if math.hypot(x, y) < 1e-6:
            return normalize_angle_rad(route_error_rad)
        return normalize_angle_rad(math.atan2(y, x))

    def _fixed_linear_speed_for_error(self, error_rad):
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
        return abs(float(self.args.linear_speed)) * linear_scale

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
        )
        self.post_rotate_branch_lock_activations += 1

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
    def _with_branch_lock_metadata(projection, stable_count, release_span_m):
        return replace(
            projection,
            branch_lock_stable_count=int(stable_count),
            branch_lock_release_required_span_m=float(release_span_m),
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
