"""Conservative stop-go waypoint control with heading hysteresis."""

import math

from .math_utils import clamp
from .models import ControllerStep, TwistCommand
from .path_progress import target_state, waypoint_reached


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
