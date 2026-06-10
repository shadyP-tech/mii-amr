from __future__ import annotations

import math
from typing import Protocol

from .math_utils import clamp, shortest_angle_delta_deg
from .models import ControllerStep, RouteState, TwistCommand
from .path_curves import polyline_lookahead_target, pure_pursuit_curve_command
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
            min(self.args.waypoint_tolerance_m, lookahead_m),
        )
        path_points = [current_point]
        path_points.extend(
            (float(wp.x), float(wp.y))
            for wp in route_state.remaining_tracking_points()
        )
        if len(path_points) == 1:
            path_points.append((float(final_goal.x), float(final_goal.y)))
        target_point = polyline_lookahead_target(path_points, current_point, lookahead_m)
        guard_result = None
        command_lookahead_m = lookahead_m
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
        speed_profile = getattr(
            self.args,
            "pure_pursuit_speed_profile",
            SPEED_PROFILE_CURVATURE_AWARE,
        )
        velocity_schedule_result = None
        if speed_profile == SPEED_PROFILE_FIXED:
            linear_x, angular_z, alpha_rad = pure_pursuit_curve_command(
                pose,
                target_point,
                command_lookahead_m,
                self.args.linear_speed,
                self.args.max_angular_speed,
            )
            mode = "forward"
        else:
            geometry = pure_pursuit_geometry(pose, target_point, command_lookahead_m)
            velocity_schedule_result = self.velocity_scheduler.schedule(geometry)
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
        )


def build_path_controller(args, lookahead_guard=None) -> PathController:
    controller = getattr(args, "controller", "stop-go")
    if controller == "pure-pursuit":
        return PurePursuitController(args, lookahead_guard=lookahead_guard)
    if controller == "stop-go":
        return StopGoController(args)
    raise ValueError(f"unsupported controller: {controller!r}")
