"""Pre-motion admission of finite deadlines for physical stand approaches."""

from __future__ import annotations

from dataclasses import replace

from scripts.aufgabe04.navigation.control.driving_behavior import (
    CommandSmoothingConfig,
    HEADING_CORRIDOR_ROUTE_KINDS,
    PHYSICAL_ROUTE_KINDS,
)
from scripts.aufgabe04.navigation.control.segment_time_budget import (
    MAX_WAYPOINT_TIMEOUT_SEC,
    MIN_WAYPOINT_TIMEOUT_SEC,
    SegmentTimeBudgetError,
    route_time_budgets,
    waypoint_time_budget,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import ControllerConfig
from .localization_admission import _preflight_pose


def station_motion_configs(args, route_kind):
    """Resolve the same immutable command limits for admission and execution."""

    return ControllerConfig(
        max_linear_mps=args.max_linear_mps,
        max_angular_radps=args.max_angular_radps,
        goal_tolerance_m=args.goal_tolerance_m,
        heading_tolerance_rad=args.heading_tolerance_rad,
        lookahead_distance_m=args.lookahead_distance_m,
        slow_heading_error_rad=args.slow_heading_error_rad,
        stop_heading_error_rad=args.stop_heading_error_rad,
        min_linear_speed_scale=args.min_linear_speed_scale,
        max_progress_advance_m=args.max_progress_advance_m,
        enforce_heading_corridor=route_kind in HEADING_CORRIDOR_ROUTE_KINDS,
        exact_vertex_pursuit=route_kind in PHYSICAL_ROUTE_KINDS,
    ), CommandSmoothingConfig(
        enabled=not args.disable_command_smoothing,
        max_linear_accel_mps2=args.max_linear_accel_mps2,
        max_angular_accel_radps2=args.max_angular_accel_radps2,
    )


def admit_route_time_budget(
    *, args, resolved, route_kind, waypoints, preflight, controller, smoothing,
    egress_certificate=None,
):
    """Budget the actual admitted route in the execution pose's frame.

    Preflight only exposes route/odom poses after freshness admission. Its
    successful result plus exact frame identity is required; a route anchor
    cannot substitute for the robot's initial heading.
    """

    if resolved.use_sim_time or route_kind != "detected_stand_preapproach":
        return {}
    if not preflight.ok:
        raise SegmentTimeBudgetError("route timing requires a passing preflight")
    odom = args.execution_pose_frame == "odom"
    frame_id = resolved.odom_frame if odom else resolved.map_frame
    start = _preflight_pose(
        preflight.odom_pose if odom else preflight.route_pose,
        frame_id=frame_id,
        child_frame_id=resolved.base_frame,
        name="odom pose" if odom else "route pose",
    )
    budgets = list(route_time_budgets(
        waypoints, start_pose=start, controller=controller, smoothing=smoothing,
        timeout_limit_sec=args.waypoint_timeout_sec,
    ))
    if egress_certificate is not None and egress_certificate.required:
        # A certified initial escape has a lower translation ceiling.
        egress_index = egress_certificate.waypoint_index
        for index, budget in enumerate(budgets):
            if budget.target_index == egress_index:
                budgets[index] = waypoint_time_budget(
                    waypoints, egress_index,
                    start_pose=start if index == 0 else None,
                    controller=replace(controller, max_linear_mps=min(
                        controller.max_linear_mps, args.start_egress_max_linear_mps,
                    )),
                    smoothing=smoothing,
                    timeout_limit_sec=args.waypoint_timeout_sec,
                )
    return {
        "policy": "route_derived_segment_timeout",
        "execution_frame": frame_id,
        "minimum_timeout_sec": MIN_WAYPOINT_TIMEOUT_SEC,
        "maximum_timeout_sec": MAX_WAYPOINT_TIMEOUT_SEC,
        "explicit_timeout_sec": args.waypoint_timeout_sec,
        "start_pose": {"x_m": start.x_m, "y_m": start.y_m, "yaw_rad": start.yaw_rad},
        "targets": [budget.to_dict() for budget in budgets],
        "motion_authorized": False,
    }
