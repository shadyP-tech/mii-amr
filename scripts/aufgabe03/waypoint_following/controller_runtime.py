from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ControllerRuntimeContext:
    TrackingPathValidation: Any
    TwistCommand: Any
    CommandSmoother: Any
    CommandSmoothingConfig: Any
    LookaheadGuard: Any
    load_tracking_path_csv: Callable[..., Any]
    validate_tracking_path_geometry: Callable[..., Any]
    clamp: Callable[[float, float, float], float]
    default_controller: str
    default_pure_pursuit_lookahead_guard: str
    default_pure_pursuit_command_smoothing: str
    lookahead_guard_off: str
    command_smoothing_off: str
    command_smoothing_rate_limit: str
    speed_profile_curvature_aware: str
    scheduler_status_deadband: str
    projection_lock_required_samples: int
    projection_lock_progress_tolerance_m: float
    route_heading_lookahead_m: float
    rotate_anchor_route_heading_exit_samples: int
    post_rotate_branch_heading_tolerance_deg: float
    post_rotate_branch_release_stable_samples: int
    post_rotate_branch_min_release_progress_m: float
    post_rotate_branch_end_lateral_tolerance_m: float
    post_rotate_zero_linear_eps_mps: float


def warn_logger(logger, message, context):
    if logger is None:
        return
    warn = getattr(logger, "warn", None)
    if warn is not None:
        warn(message)


def build_sparse_tracking_validation(source, point_count, status, context):
    return context.TrackingPathValidation(
        source=source,
        point_count=point_count,
        validation_status=status,
    )


def prepare_tracking_setup(
    args,
    route_waypoints,
    context,
    current_pose=None,
    logger=None,
    structural_only=False,
):
    route_waypoints = list(route_waypoints)
    if getattr(args, "controller", context.default_controller) != "pure-pursuit":
        return None, build_sparse_tracking_validation(
            source="ignored_stop_go",
            point_count=0,
            status="ignored",
            context=context,
        )

    if not getattr(args, "tracking_path_csv", None):
        message = (
            "Pure-pursuit has no --tracking-path-csv; "
            "falling back to sparse waypoint geometry."
        )
        warn_logger(logger, message, context)
        return None, build_sparse_tracking_validation(
            source="waypoints",
            point_count=len(route_waypoints),
            status="fallback_sparse_waypoints",
            context=context,
        )

    tracking_points, warnings = context.load_tracking_path_csv(
        args.tracking_path_csv,
        max_segment_m=args.tracking_max_segment_m,
    )
    for warning in warnings:
        warn_logger(logger, warning, context)
    if structural_only:
        return tracking_points, context.TrackingPathValidation(
            source="csv",
            point_count=len(tracking_points),
            validation_status="structural_ok",
            warnings=tuple(warnings),
        )
    validation = context.validate_tracking_path_geometry(
        route_waypoints,
        tracking_points,
        endpoint_tolerance_m=args.tracking_endpoint_tolerance_m,
        start_tolerance_m=args.tracking_start_tolerance_m,
        allow_mismatch=args.allow_tracking_path_mismatch,
        current_pose=current_pose,
        source="csv",
        structural_warnings=warnings,
    )
    for warning in validation.warnings:
        warn_logger(logger, warning, context)
    return tracking_points, validation


def _format_optional_m(value):
    return "n/a" if value is None else f"{value:.3f}"


def format_optional_m(value, context):
    return _format_optional_m(value)


def notes_with_tracking_metadata(notes, args, tracking_validation, context):
    if (
        getattr(args, "controller", context.default_controller) != "pure-pursuit"
        or tracking_validation is None
    ):
        return notes
    return (
        f"{notes};controller={args.controller};"
        f"tracking_source={tracking_validation.source};"
        f"tracking_point_count={tracking_validation.point_count};"
        f"tracking_validation_status={tracking_validation.validation_status}"
    )


def build_lookahead_guard(args, context, run_local_map_fn=None):
    if getattr(args, "controller", context.default_controller) != "pure-pursuit":
        return None
    guard_mode = getattr(
        args,
        "pure_pursuit_lookahead_guard",
        context.default_pure_pursuit_lookahead_guard,
    )
    if guard_mode == context.lookahead_guard_off:
        return None
    return context.LookaheadGuard.from_static_map(
        args.static_map,
        args.pure_pursuit_lookahead_guard_static_inflation_radius_m,
        mode=guard_mode,
        run_local_map_fn=run_local_map_fn,
    )


def command_smoothing_active(args, context):
    return (
        getattr(args, "controller", context.default_controller) == "pure-pursuit"
        and getattr(
            args,
            "pure_pursuit_command_smoothing",
            context.default_pure_pursuit_command_smoothing,
        )
        == context.command_smoothing_rate_limit
    )


def build_command_smoother(args, context):
    if not command_smoothing_active(args, context):
        return None
    return context.CommandSmoother(
        context.CommandSmoothingConfig(
            max_linear_accel_mps2=args.pure_pursuit_max_linear_accel_mps2,
            max_linear_decel_mps2=args.pure_pursuit_max_linear_decel_mps2,
            max_angular_accel_radps2=args.pure_pursuit_max_angular_accel_radps2,
            max_angular_decel_radps2=args.pure_pursuit_max_angular_decel_radps2,
            final_decel_distance_m=args.pure_pursuit_final_decel_distance_m,
            min_smoothed_linear_speed_mps=(
                args.pure_pursuit_min_smoothed_linear_speed_mps
            ),
        )
    )


def reset_command_smoother(node, context):
    smoother = getattr(node, "command_smoother", None)
    if smoother is not None:
        smoother.reset()
    if hasattr(node, "last_smoothed_command_time_sec"):
        node.last_smoothed_command_time_sec = None
    if hasattr(node, "last_velocity_scheduler_status"):
        node.last_velocity_scheduler_status = None
    if hasattr(node, "last_velocity_scheduler_log_sec"):
        node.last_velocity_scheduler_log_sec = None
    if hasattr(node, "last_smoothed_motion_mode"):
        node.last_smoothed_motion_mode = None


def reset_route_projection_controller(controller, context):
    reset = getattr(controller, "reset_route_projection_state", None)
    if reset is not None:
        reset()


def smoothing_dt_sec(node, now_sec, context):
    args = node.args
    default_dt = 1.0 / args.control_rate_hz
    max_dt = 2.0 / args.control_rate_hz
    previous_sec = getattr(node, "last_smoothed_command_time_sec", None)
    if previous_sec is None:
        return default_dt
    dt_sec = now_sec - previous_sec
    if not math.isfinite(dt_sec):
        return default_dt
    return context.clamp(dt_sec, 0.0, max_dt)


def smoothed_step_command(node, step, now_sec, context):
    smoother = getattr(node, "command_smoother", None)
    if smoother is None:
        return step.command
    if step.command.linear_x == 0.0 and step.command.angular_z == 0.0:
        reset_command_smoother(node, context)
        return step.command
    previous_mode = getattr(node, "last_smoothed_motion_mode", None)
    if step.mode == "rotate" and previous_mode != "rotate":
        smoother.reset()
        if hasattr(node, "last_smoothed_command_time_sec"):
            node.last_smoothed_command_time_sec = None
    dt_sec = smoothing_dt_sec(node, now_sec, context)
    raw_command = (
        context.TwistCommand(0.0, step.command.angular_z)
        if step.mode == "rotate"
        else step.command
    )
    command = smoother.apply(
        raw_command,
        dt_sec,
        step.distance_m,
        node.args.pure_pursuit_goal_tolerance_m,
    )
    if step.mode == "rotate":
        command = context.TwistCommand(0.0, command.angular_z)
    node.last_smoothed_command_time_sec = now_sec
    node.last_smoothed_motion_mode = step.mode
    return command


def notes_with_smoothing_metadata(notes, args, context):
    if not command_smoothing_active(args, context):
        return notes
    return (
        f"{notes};pure_pursuit_command_smoothing="
        f"{args.pure_pursuit_command_smoothing};"
        "pure_pursuit_max_linear_accel_mps2="
        f"{args.pure_pursuit_max_linear_accel_mps2:.3f};"
        "pure_pursuit_max_linear_decel_mps2="
        f"{args.pure_pursuit_max_linear_decel_mps2:.3f};"
        "pure_pursuit_max_angular_accel_radps2="
        f"{args.pure_pursuit_max_angular_accel_radps2:.3f};"
        "pure_pursuit_max_angular_decel_radps2="
        f"{args.pure_pursuit_max_angular_decel_radps2:.3f};"
        "pure_pursuit_final_decel_distance_m="
        f"{args.pure_pursuit_final_decel_distance_m:.3f};"
        "pure_pursuit_min_smoothed_linear_speed_mps="
        f"{args.pure_pursuit_min_smoothed_linear_speed_mps:.3f}"
    )


def notes_with_velocity_scheduler_metadata(notes, args, context):
    if getattr(args, "controller", context.default_controller) != "pure-pursuit":
        return notes
    ROUTE_HEADING_LOOKAHEAD_M = context.route_heading_lookahead_m
    return (
        f"{notes};pure_pursuit_speed_profile="
        f"{args.pure_pursuit_speed_profile};"
        "pure_pursuit_forward_control="
        f"{args.pure_pursuit_forward_control};"
        "pure_pursuit_route_heading_blend="
        f"{args.pure_pursuit_route_heading_blend:.3f};"
        "pure_pursuit_cross_track_gain="
        f"{args.pure_pursuit_cross_track_gain:.3f};"
        "pure_pursuit_cross_track_speed_floor_mps="
        f"{args.pure_pursuit_cross_track_speed_floor_mps:.3f};"
        "pure_pursuit_max_cross_track_correction_deg="
        f"{args.pure_pursuit_max_cross_track_correction_deg:.3f};"
        "pure_pursuit_angular_feasibility_speed_limit="
        f"{args.pure_pursuit_angular_feasibility_speed_limit};"
        "pure_pursuit_angular_feasibility_margin="
        f"{args.pure_pursuit_angular_feasibility_margin:.3f};"
        "pure_pursuit_default_linear_speed_resolved_mps="
        f"{args.linear_speed:.3f};"
        "pure_pursuit_default_max_angular_speed_resolved_radps="
        f"{args.max_angular_speed:.3f};"
        "pure_pursuit_target_source=route_projection;"
        "pure_pursuit_max_track_angular_speed_radps="
        f"{args.pure_pursuit_max_track_angular_speed_radps:.3f};"
        "pure_pursuit_max_rotate_angular_speed_radps="
        f"{args.pure_pursuit_max_rotate_angular_speed_radps:.3f};"
        "pure_pursuit_cross_track_warning_m="
        f"{args.pure_pursuit_cross_track_warning_m:.3f};"
        "pure_pursuit_max_cross_track_error_m="
        f"{args.pure_pursuit_max_cross_track_error_m:.3f};"
        "pure_pursuit_max_lateral_accel_mps2="
        f"{args.pure_pursuit_max_lateral_accel_mps2:.3f};"
        "pure_pursuit_turn_speed_margin="
        f"{args.pure_pursuit_turn_speed_margin:.3f};"
        "pure_pursuit_heading_deadband_deg="
        f"{args.pure_pursuit_heading_deadband_deg:.3f};"
        "pure_pursuit_lateral_deadband_m="
        f"{args.pure_pursuit_lateral_deadband_m:.3f};"
        "pure_pursuit_curvature_limit_start_heading_error_deg="
        f"{args.pure_pursuit_curvature_limit_start_heading_error_deg:.3f};"
        "pure_pursuit_curvature_limit_full_heading_error_deg="
        f"{args.pure_pursuit_curvature_limit_full_heading_error_deg:.3f};"
        "pure_pursuit_rotate_start_heading_error_deg="
        f"{args.pure_pursuit_rotate_start_heading_error_deg:.3f};"
        "pure_pursuit_rotate_stop_heading_error_deg="
        f"{args.pure_pursuit_rotate_stop_heading_error_deg:.3f};"
        "pure_pursuit_route_heading_lookahead_m="
        f"{ROUTE_HEADING_LOOKAHEAD_M:.3f};"
        "pure_pursuit_route_heading_rotate_start_deg="
        f"{args.pure_pursuit_route_heading_rotate_start_deg:.3f};"
        "pure_pursuit_route_heading_rotate_stop_deg="
        f"{args.pure_pursuit_route_heading_rotate_stop_deg:.3f}"
    )


def notes_with_route_projection_metadata(notes, args, node, context):
    if getattr(args, "controller", context.default_controller) != "pure-pursuit":
        return notes
    format_optional_m = _format_optional_m
    PROJECTION_LOCK_REQUIRED_SAMPLES = context.projection_lock_required_samples
    PROJECTION_LOCK_PROGRESS_TOLERANCE_M = (
        context.projection_lock_progress_tolerance_m
    )
    ROTATE_ANCHOR_ROUTE_HEADING_EXIT_SAMPLES = (
        context.rotate_anchor_route_heading_exit_samples
    )
    POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG = (
        context.post_rotate_branch_heading_tolerance_deg
    )
    POST_ROTATE_BRANCH_RELEASE_STABLE_SAMPLES = (
        context.post_rotate_branch_release_stable_samples
    )
    POST_ROTATE_BRANCH_MIN_RELEASE_PROGRESS_M = (
        context.post_rotate_branch_min_release_progress_m
    )
    POST_ROTATE_BRANCH_END_LATERAL_TOLERANCE_M = (
        context.post_rotate_branch_end_lateral_tolerance_m
    )
    POST_ROTATE_ZERO_LINEAR_EPS_MPS = context.post_rotate_zero_linear_eps_mps
    count = getattr(node, "cross_track_error_count", 0)
    mean_error = (
        0.0
        if count <= 0
        else getattr(node, "cross_track_error_sum_m", 0.0) / count
    )
    return (
        f"{notes};pure_pursuit_target_source=route_projection;"
        "pure_pursuit_max_cross_track_error_observed_m="
        f"{getattr(node, 'max_cross_track_error_m', 0.0):.3f};"
        "pure_pursuit_mean_abs_cross_track_error_m="
        f"{mean_error:.3f};"
        "pure_pursuit_max_route_heading_error_deg="
        f"{getattr(node, 'max_route_heading_error_deg', 0.0):.3f};"
        "pure_pursuit_rotate_gate_entries="
        f"{getattr(node, 'pure_pursuit_rotate_gate_entries', 0)};"
        "pure_pursuit_projection_status="
        f"{getattr(node, 'last_projection_acquisition_status', '')};"
        "pure_pursuit_projection_lock_samples="
        f"{getattr(node, 'last_projection_lock_sample_count', 0)};"
        "pure_pursuit_max_projection_backward_delta_m="
        f"{getattr(node, 'max_projection_backward_delta_m', 0.0):.3f};"
        "pure_pursuit_projection_lock_required_samples="
        f"{PROJECTION_LOCK_REQUIRED_SAMPLES};"
        "pure_pursuit_projection_lock_progress_tolerance_m="
        f"{PROJECTION_LOCK_PROGRESS_TOLERANCE_M:.3f};"
        "pure_pursuit_route_heading_source="
        f"{getattr(node, 'last_route_heading_source', '')};"
        "pure_pursuit_last_route_heading_error_deg="
        f"{format_optional_m(getattr(node, 'last_route_heading_error_deg', None))};"
        "pure_pursuit_last_rotate_reason="
        f"{getattr(node, 'last_pure_pursuit_rotate_reason', '')};"
        "pure_pursuit_last_rotate_source="
        f"{getattr(node, 'last_pure_pursuit_rotate_source', '')};"
        "pure_pursuit_rotate_anchor_activations="
        f"{getattr(node, 'pure_pursuit_rotate_anchor_activations', 0)};"
        "pure_pursuit_max_rotate_anchor_backward_delta_m="
        f"{getattr(node, 'max_rotate_anchor_backward_delta_m', 0.0):.3f};"
        "pure_pursuit_max_rotate_anchor_forward_delta_m="
        f"{getattr(node, 'max_rotate_anchor_forward_delta_m', 0.0):.3f};"
        "pure_pursuit_last_rotate_anchor_aligned_samples="
        f"{getattr(node, 'last_rotate_anchor_aligned_samples', 0)};"
        "pure_pursuit_max_rotate_anchor_aligned_samples="
        f"{getattr(node, 'max_rotate_anchor_aligned_samples', 0)};"
        "pure_pursuit_rotate_anchor_route_heading_exit_samples="
        f"{ROTATE_ANCHOR_ROUTE_HEADING_EXIT_SAMPLES};"
        "pure_pursuit_post_rotate_branch_lock_activations="
        f"{getattr(node, 'post_rotate_branch_lock_activations', 0)};"
        "pure_pursuit_post_rotate_branch_max_heading_error_deg="
        f"{getattr(node, 'post_rotate_branch_max_heading_error_deg', 0.0):.3f};"
        "pure_pursuit_post_rotate_branch_rejected_wrong_heading_count="
        f"{getattr(node, 'post_rotate_branch_rejected_wrong_heading_count', 0)};"
        "pure_pursuit_post_rotate_branch_ambiguity_failures="
        f"{getattr(node, 'post_rotate_branch_ambiguity_failures', 0)};"
        "pure_pursuit_post_rotate_branch_heading_tolerance_deg="
        f"{POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG:.3f};"
        "pure_pursuit_post_rotate_branch_release_samples="
        f"{POST_ROTATE_BRANCH_RELEASE_STABLE_SAMPLES};"
        "pure_pursuit_post_rotate_branch_min_release_progress_m="
        f"{POST_ROTATE_BRANCH_MIN_RELEASE_PROGRESS_M:.3f};"
        "pure_pursuit_post_rotate_branch_target_clip_count="
        f"{getattr(node, 'post_rotate_branch_target_clip_count', 0)};"
        "pure_pursuit_post_rotate_branch_heading_break_handoff_count="
        f"{getattr(node, 'post_rotate_branch_heading_break_handoff_count', 0)};"
        "pure_pursuit_post_rotate_branch_physical_handoff_count="
        f"{getattr(node, 'post_rotate_branch_physical_handoff_count', 0)};"
        "pure_pursuit_post_rotate_branch_end_lateral_tolerance_m="
        f"{POST_ROTATE_BRANCH_END_LATERAL_TOLERANCE_M:.3f};"
        "pure_pursuit_post_rotate_zero_linear_eps_mps="
        f"{POST_ROTATE_ZERO_LINEAR_EPS_MPS:.3f}"
    )


def notes_with_guard_metadata(notes, args, guard_result, context):
    format_optional_m = _format_optional_m
    if (
        getattr(args, "controller", context.default_controller) != "pure-pursuit"
        or getattr(args, "pure_pursuit_lookahead_guard", context.lookahead_guard_off)
        == context.lookahead_guard_off
        or guard_result is None
    ):
        return notes
    return (
        f"{notes};lookahead_guard={args.pure_pursuit_lookahead_guard};"
        f"lookahead_guard_status={guard_result.status};"
        "lookahead_guard_selected_distance_m="
        f"{format_optional_m(guard_result.selected_target_distance_m)};"
        f"lookahead_guard_blocked_cell_count={guard_result.blocked_cell_count}"
    )


def maybe_log_velocity_scheduler_result(self, result, now_sec, context):
    SPEED_PROFILE_CURVATURE_AWARE = context.speed_profile_curvature_aware
    SCHEDULER_STATUS_DEADBAND = context.scheduler_status_deadband
    if (
        result is None
        or not self.args.verbose
        or self.args.controller != "pure-pursuit"
        or self.args.pure_pursuit_speed_profile != SPEED_PROFILE_CURVATURE_AWARE
        or result.status == SCHEDULER_STATUS_DEADBAND
    ):
        return
    status_changed = result.status != self.last_velocity_scheduler_status
    log_due = (
        self.last_velocity_scheduler_log_sec is None
        or now_sec - self.last_velocity_scheduler_log_sec >= 2.0
    )
    if not status_changed and not log_due:
        return
    self.last_velocity_scheduler_status = result.status
    self.last_velocity_scheduler_log_sec = now_sec
    self.get_logger().info(
        "Pure-pursuit scheduler: "
        f"status={result.status}, "
        f"alpha_deg={result.alpha_deg:.2f}, "
        f"lateral_error_m={result.lateral_error_m:.3f}, "
        f"angular_scale={result.angular_scale:.3f}, "
        f"speed_limit_blend={result.speed_limit_blend:.3f}, "
        f"raw_v={result.raw_linear_x:.3f}, "
        f"scheduled_v={result.scheduled_linear_x:.3f}, "
        f"raw_omega={result.raw_angular_z:.3f}, "
        f"scheduled_omega={result.scheduled_angular_z:.3f}"
    )


def record_route_projection_result(self, step, context):
    projection = getattr(step, "route_projection_result", None)
    if projection is None:
        return
    error_m = abs(float(projection.cross_track_error_m))
    self.max_cross_track_error_m = max(self.max_cross_track_error_m, error_m)
    self.cross_track_error_sum_m += error_m
    self.cross_track_error_count += 1
    route_heading = getattr(step, "route_heading_result", None)
    route_heading_error = (
        getattr(route_heading, "heading_error_deg", None)
        if route_heading is not None
        else None
    )
    self.max_route_heading_error_deg = max(
        self.max_route_heading_error_deg,
        abs(
            float(
                route_heading_error
                if route_heading_error is not None
                else projection.heading_error_to_route_deg
            )
        ),
    )
    if route_heading is not None:
        self.last_route_heading_source = getattr(route_heading, "source", "")
        self.last_route_heading_error_deg = route_heading_error
    self.last_pure_pursuit_rotate_reason = getattr(
        step,
        "pure_pursuit_rotate_reason",
        "",
    )
    self.last_pure_pursuit_rotate_source = getattr(
        step,
        "pure_pursuit_rotate_source",
        "",
    )
    self.max_projection_backward_delta_m = max(
        self.max_projection_backward_delta_m,
        float(getattr(projection, "route_progress_backward_delta_m", 0.0)),
    )
    self.max_rotate_anchor_backward_delta_m = max(
        self.max_rotate_anchor_backward_delta_m,
        float(getattr(projection, "rotate_anchor_backward_delta_m", 0.0)),
    )
    self.max_rotate_anchor_forward_delta_m = max(
        self.max_rotate_anchor_forward_delta_m,
        float(getattr(projection, "rotate_anchor_forward_delta_m", 0.0)),
    )
    aligned_samples = int(
        getattr(projection, "rotate_anchor_route_heading_aligned_samples", 0)
    )
    self.last_rotate_anchor_aligned_samples = aligned_samples
    self.max_rotate_anchor_aligned_samples = max(
        self.max_rotate_anchor_aligned_samples,
        aligned_samples,
    )
    controller = getattr(self, "_current_path_controller", None)
    controller_anchor_activations = getattr(
        controller,
        "rotate_anchor_activations",
        None,
    )
    if controller_anchor_activations is not None:
        self.pure_pursuit_rotate_anchor_activations = controller_anchor_activations
    self.post_rotate_branch_rejected_wrong_heading_count += int(
        getattr(projection, "rejected_wrong_heading_segment_count", 0),
    )
    branch_heading_error = getattr(
        projection,
        "selected_branch_heading_error_deg",
        None,
    )
    if branch_heading_error is not None:
        self.post_rotate_branch_max_heading_error_deg = max(
            self.post_rotate_branch_max_heading_error_deg,
            abs(float(branch_heading_error)),
        )
    controller_branch_activations = getattr(
        controller,
        "post_rotate_branch_lock_activations",
        None,
    )
    if controller_branch_activations is not None:
        self.post_rotate_branch_lock_activations = controller_branch_activations
    controller_branch_failures = getattr(
        controller,
        "post_rotate_branch_ambiguity_failures",
        None,
    )
    if controller_branch_failures is not None:
        self.post_rotate_branch_ambiguity_failures = controller_branch_failures
    controller_branch_max_error = getattr(
        controller,
        "post_rotate_branch_max_heading_error_deg",
        None,
    )
    if controller_branch_max_error is not None:
        self.post_rotate_branch_max_heading_error_deg = max(
            self.post_rotate_branch_max_heading_error_deg,
            controller_branch_max_error,
        )
    controller_branch_clip_count = getattr(
        controller,
        "post_rotate_branch_target_clip_count",
        None,
    )
    if controller_branch_clip_count is not None:
        self.post_rotate_branch_target_clip_count = controller_branch_clip_count
    controller_branch_handoff_count = getattr(
        controller,
        "post_rotate_branch_heading_break_handoff_count",
        None,
    )
    if controller_branch_handoff_count is not None:
        self.post_rotate_branch_heading_break_handoff_count = (
            controller_branch_handoff_count
        )
    controller_branch_physical_handoff_count = getattr(
        controller,
        "post_rotate_branch_physical_handoff_count",
        None,
    )
    if controller_branch_physical_handoff_count is not None:
        self.post_rotate_branch_physical_handoff_count = (
            controller_branch_physical_handoff_count
        )
    self.last_projection_acquisition_status = getattr(
        projection,
        "projection_status",
        "",
    )
    controller_lock_samples = getattr(
        getattr(self, "_current_path_controller", None),
        "projection_lock_sample_count",
        None,
    )
    self.last_projection_lock_sample_count = (
        controller_lock_samples
        if controller_lock_samples is not None
        else self.last_projection_lock_sample_count
    )
    status = getattr(step, "pure_pursuit_status", "") or step.mode
    if status == "rotate_gate" and self.last_recorded_pure_pursuit_status != status:
        self.pure_pursuit_rotate_gate_entries += 1
    self.last_recorded_pure_pursuit_status = status


def maybe_log_route_projection_result(self, step, now_sec, context):
    format_optional_m = _format_optional_m
    projection = getattr(step, "route_projection_result", None)
    if (
        projection is None
        or not self.args.verbose
        or self.args.controller != "pure-pursuit"
    ):
        return
    status = getattr(step, "pure_pursuit_status", "") or step.mode
    warning = (
        abs(float(projection.cross_track_error_m))
        >= self.args.pure_pursuit_cross_track_warning_m
    )
    projection_status = getattr(projection, "projection_status", "locked")
    status_key = f"{status}:{warning}:{projection_status}"
    status_changed = status_key != self.last_route_projection_status
    log_due = (
        self.last_route_projection_log_sec is None
        or now_sec - self.last_route_projection_log_sec >= 2.0
    )
    if not status_changed and not log_due:
        return
    self.last_route_projection_status = status_key
    self.last_route_projection_log_sec = now_sec
    route_heading = getattr(step, "route_heading_result", None)
    forward_control = getattr(step, "forward_control_result", None)
    message = (
        "Pure-pursuit route projection: "
        f"status={status}, "
        "projection_status="
        f"{getattr(projection, 'projection_status', 'locked')}, "
        "projection_lock_samples="
        f"{getattr(getattr(self, '_current_path_controller', None), 'projection_lock_sample_count', 0)}, "
        f"route_progress_m={projection.route_progress_m:.3f}, "
        "route_progress_delta_m="
        f"{format_optional_m(getattr(projection, 'route_progress_delta_m', None))}, "
        "route_progress_backward_delta_m="
        f"{getattr(projection, 'route_progress_backward_delta_m', 0.0):.3f}, "
        "route_progress_forward_delta_m="
        f"{getattr(projection, 'route_progress_forward_delta_m', 0.0):.3f}, "
        "raw_projection_progress_m="
        f"{format_optional_m(getattr(projection, 'raw_projection_progress_m', None))}, "
        "raw_projection_segment_index="
        f"{getattr(projection, 'raw_projection_segment_index', None)}, "
        "effective_projection_progress_m="
        f"{format_optional_m(getattr(projection, 'effective_projection_progress_m', None))}, "
        "anchor_progress_m="
        f"{format_optional_m(getattr(projection, 'anchor_progress_m', None))}, "
        "anchor_segment_index="
        f"{getattr(projection, 'anchor_segment_index', None)}, "
        "rotate_anchor_backward_delta_m="
        f"{getattr(projection, 'rotate_anchor_backward_delta_m', 0.0):.3f}, "
        "rotate_anchor_forward_delta_m="
        f"{getattr(projection, 'rotate_anchor_forward_delta_m', 0.0):.3f}, "
        "rotate_anchor_route_heading_aligned_samples="
        f"{getattr(projection, 'rotate_anchor_route_heading_aligned_samples', 0)}, "
        "rotate_anchor_handoff_reason="
        f"{getattr(projection, 'rotate_anchor_handoff_reason', '')}, "
        "local_cross_track_m="
        f"{format_optional_m(getattr(projection, 'local_cross_track_m', None))}, "
        "preferred_branch_heading_deg="
        f"{format_optional_m(getattr(projection, 'preferred_branch_heading_deg', None))}, "
        "selected_segment_heading_deg="
        f"{format_optional_m(getattr(projection, 'selected_segment_heading_deg', None))}, "
        "selected_branch_heading_error_deg="
        f"{format_optional_m(getattr(projection, 'selected_branch_heading_error_deg', None))}, "
        "rejected_wrong_heading_segment_count="
        f"{getattr(projection, 'rejected_wrong_heading_segment_count', 0)}, "
        "branch_lock_stable_count="
        f"{getattr(projection, 'branch_lock_stable_count', 0)}, "
        "branch_lock_progress_span_m="
        f"{getattr(projection, 'branch_lock_progress_span_m', 0.0):.3f}, "
        "branch_lock_release_required_span_m="
        f"{getattr(projection, 'branch_lock_release_required_span_m', 0.0):.3f}, "
        "branch_compatible_length_m="
        f"{getattr(projection, 'branch_compatible_length_m', 0.0):.3f}, "
        "branch_target_clipped_to_heading_break="
        f"{getattr(projection, 'branch_target_clipped_to_heading_break', False)}, "
        "branch_heading_break="
        f"{getattr(projection, 'branch_heading_break', False)}, "
        "branch_end_progress_m="
        f"{format_optional_m(getattr(projection, 'branch_end_progress_m', None))}, "
        "branch_compatible_target_progress_m="
        f"{format_optional_m(getattr(projection, 'branch_compatible_target_progress_m', None))}, "
        "heading_break_delta_deg="
        f"{format_optional_m(getattr(projection, 'heading_break_delta_deg', None))}, "
        "next_heading_error_deg="
        f"{format_optional_m(getattr(projection, 'next_heading_error_deg', None))}, "
        "branch_end_along_past_m="
        f"{format_optional_m(getattr(projection, 'branch_end_along_past_m', None))}, "
        "branch_end_lateral_error_m="
        f"{format_optional_m(getattr(projection, 'branch_end_lateral_error_m', None))}, "
        "branch_end_handoff_reason="
        f"{getattr(projection, 'branch_end_handoff_reason', '')}, "
        "branch_end_handoff_lateral_tolerance_m="
        f"{format_optional_m(getattr(projection, 'branch_end_handoff_lateral_tolerance_m', None))}, "
        f"cross_track_error_m={projection.cross_track_error_m:.3f}, "
        f"signed_cross_track_error_m={projection.signed_cross_track_error_m:.3f}, "
        f"route_heading_deg={projection.route_heading_deg:.1f}, "
        f"heading_error_to_route_deg={projection.heading_error_to_route_deg:.1f}, "
        "route_heading_source="
        f"{getattr(route_heading, 'source', 'unavailable') if route_heading is not None else 'unavailable'}, "
        "smoothed_route_heading_deg="
        f"{format_optional_m(getattr(route_heading, 'heading_deg', None) if route_heading is not None else None)}, "
        "smoothed_route_heading_error_deg="
        f"{format_optional_m(getattr(route_heading, 'heading_error_deg', None) if route_heading is not None else None)}, "
        "forward_control="
        f"{getattr(forward_control, 'mode', '') if forward_control is not None else ''}, "
        "forward_control_fallback="
        f"{getattr(forward_control, 'fallback_reason', '') if forward_control is not None else ''}, "
        "alpha_deg="
        f"{format_optional_m(getattr(forward_control, 'alpha_deg', None) if forward_control is not None else None)}, "
        "forward_route_heading_error_deg="
        f"{format_optional_m(getattr(forward_control, 'route_heading_error_deg', None) if forward_control is not None else None)}, "
        "forward_signed_cross_track_error_m="
        f"{format_optional_m(getattr(forward_control, 'signed_cross_track_error_m', None) if forward_control is not None else None)}, "
        "cte_correction_deg="
        f"{format_optional_m(getattr(forward_control, 'cte_correction_deg', None) if forward_control is not None else None)}, "
        "blended_forward_error_deg="
        f"{format_optional_m(getattr(forward_control, 'blended_forward_error_deg', None) if forward_control is not None else None)}, "
        "speed_taper_error_deg="
        f"{format_optional_m(getattr(forward_control, 'speed_taper_error_deg', None) if forward_control is not None else None)}, "
        "raw_angular_z="
        f"{format_optional_m(getattr(forward_control, 'raw_angular_z', None) if forward_control is not None else None)}, "
        "command_angular_z="
        f"{format_optional_m(getattr(forward_control, 'command_angular_z', None) if forward_control is not None else None)}, "
        "angular_feasibility_limited="
        f"{getattr(forward_control, 'angular_feasibility_limited', False) if forward_control is not None else False}, "
        "angular_feasibility_scale="
        f"{format_optional_m(getattr(forward_control, 'angular_feasibility_scale', None) if forward_control is not None else None)}, "
        "linear_before_feasibility_mps="
        f"{format_optional_m(getattr(forward_control, 'linear_before_feasibility_mps', None) if forward_control is not None else None)}, "
        "linear_after_feasibility_mps="
        f"{format_optional_m(getattr(forward_control, 'linear_after_feasibility_mps', None) if forward_control is not None else None)}, "
        "rotate_reason="
        f"{getattr(step, 'pure_pursuit_rotate_reason', '')}, "
        "rotate_source="
        f"{getattr(step, 'pure_pursuit_rotate_source', '')}, "
        f"target={step.target}, "
        f"track_angular_cap={self.args.pure_pursuit_max_track_angular_speed_radps:.3f}, "
        f"rotate_angular_cap={self.args.pure_pursuit_max_rotate_angular_speed_radps:.3f}"
    )
    if warning:
        self.get_logger().warn(message)
    else:
        self.get_logger().info(message)
