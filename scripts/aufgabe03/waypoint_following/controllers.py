from __future__ import annotations

import math
from typing import Protocol

from .math_utils import clamp, shortest_angle_delta_deg
from .models import ControllerStep, RouteState, TwistCommand
from .path_curves import (
    lookahead_target_from_route_anchor,
    project_point_to_route,
    pure_pursuit_curve_command,
    route_points_from_projection,
)
from .path_progress import target_state, waypoint_reached
from .velocity_scheduler import (
    SPEED_PROFILE_CURVATURE_AWARE,
    SPEED_PROFILE_FIXED,
    PurePursuitVelocityScheduler,
    pure_pursuit_geometry,
)


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
        self.last_rotate_sign = 1.0
        self.rotate_gate_entries = 0

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
        route_state.advance_tracking_progress(
            pose,
            max(
                min(self.args.waypoint_tolerance_m, lookahead_m),
                getattr(self.args, "pure_pursuit_max_cross_track_error_m", 0.25),
            ),
        )
        tracking_points = route_state.effective_tracking_points()
        route_points = [(float(wp.x), float(wp.y)) for wp in tracking_points]
        if len(route_points) < 2:
            route_points = [current_point, (float(final_goal.x), float(final_goal.y))]
        start_segment = (
            max(0, self.last_projection_segment_index - 1)
            if self.last_projection_segment_index is not None
            else max(0, route_state.tracking_progress_index - 1)
        )
        projection = project_point_to_route(
            route_points,
            pose,
            start_segment_index=start_segment,
            previous_progress_m=self.last_route_progress_m,
            max_forward_jump_m=max(2.0 * lookahead_m, 0.35),
            backward_tolerance_m=0.03,
        )
        self.last_projection_segment_index = projection.segment_index
        self.last_route_progress_m = projection.route_progress_m
        route_state.tracking_progress_index = max(
            route_state.tracking_progress_index,
            projection.segment_index,
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

        path_points = route_points_from_projection(route_points, projection)
        target_point = lookahead_target_from_route_anchor(
            path_points,
            min(lookahead_m, max(0.0, projection.remaining_route_m)),
        )
        command_lookahead_m = max(0.01, min(lookahead_m, projection.remaining_route_m))
        geometry = pure_pursuit_geometry(pose, target_point, command_lookahead_m)
        alpha_rad = self._stable_alpha_for_rotate(geometry.alpha_rad)
        alpha_deg = math.degrees(alpha_rad)
        rotate_mode = should_rotate(
            self.mode,
            alpha_deg,
            self.args.pure_pursuit_rotate_start_heading_error_deg,
            self.args.pure_pursuit_rotate_stop_heading_error_deg,
        )
        if rotate_mode:
            if self.mode != "rotate":
                self.rotate_gate_entries += 1
            self.mode = "rotate"
            angular_z = clamp(
                alpha_rad * self.args.yaw_gain,
                -abs(self.args.pure_pursuit_max_rotate_angular_speed_radps),
                abs(self.args.pure_pursuit_max_rotate_angular_speed_radps),
            )
            return ControllerStep(
                TwistCommand(0.0, angular_z),
                "rotate",
                target_point,
                distance_to_goal,
                alpha_deg,
                False,
                route_projection_result=projection,
                pure_pursuit_status="rotate_gate",
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
            linear_x, angular_z, alpha_rad = pure_pursuit_curve_command(
                pose,
                target_point,
                command_lookahead_m,
                self.args.linear_speed,
                self.args.pure_pursuit_max_track_angular_speed_radps,
                self.args.pure_pursuit_rotate_start_heading_error_deg,
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
        )

    def _stable_alpha_for_rotate(self, alpha_rad):
        if abs(abs(alpha_rad) - math.pi) <= math.radians(0.5):
            return math.copysign(math.pi, self.last_rotate_sign)
        if abs(alpha_rad) > 1e-9:
            self.last_rotate_sign = 1.0 if alpha_rad > 0.0 else -1.0
        return alpha_rad


def build_path_controller(args, lookahead_guard=None) -> PathController:
    controller = getattr(args, "controller", "stop-go")
    if controller == "pure-pursuit":
        return PurePursuitController(args, lookahead_guard=lookahead_guard)
    if controller == "stop-go":
        return StopGoController(args)
    raise ValueError(f"unsupported controller: {controller!r}")
