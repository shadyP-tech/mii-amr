from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class NodeSetupContext:
    RuntimeDiagnostics: Any
    ReplanManager: Any
    build_command_smoother: Callable[..., Any]
    build_lookahead_guard: Callable[..., Any]
    projection_lock_required_samples: int
    projection_lock_progress_tolerance_m: float
    route_heading_lookahead_m: float
    post_rotate_branch_heading_tolerance_deg: float
    post_rotate_branch_release_stable_samples: int
    post_replan_route_clearance_preview_distance_m: Callable[..., float]
    resolve_post_replan_escape_steering_mode: Callable[..., str]


def initialize_runtime_state(node, args, context):
    node.last_scan = None
    node.last_scan_received_sec = None
    node.last_amcl = None
    node.last_amcl_received_sec = None
    node.last_odom = None
    node.last_odom_pose = None
    node.last_odom_received_sec = None
    node.last_odom_frame_id = ""
    node.last_odom_child_frame_id = ""
    node.base_frame_used = ""
    node.reached_count = 0
    node.start_pose = None
    node.final_pose = None
    node.last_scan_safety = None
    node.last_amcl_health = None
    node.diagnostics = context.RuntimeDiagnostics(
        max_tf_update_gap_sec=args.max_tf_update_gap_sec,
    )
    node.last_tf_stamp_sec = None
    node.last_tf_stamp_change_local_sec = None
    node.run_local_map = None
    node.live_replan_attempt_count = 0
    node.known_corridor_repair_count = 0
    node.lookahead_guard_budget_repair_count = 0
    node.last_known_corridor_repair_signature = None
    node.suppressed_known_corridor_signature = None
    node.last_scan_block_budget_repair_signature = None
    node.last_lookahead_guard_block_signature = None
    node.last_lookahead_guard_budget_repair_signature = None
    node.last_lookahead_guard_budget_repair_reason = ""
    node.last_lookahead_guard_result = None
    node.active_route_generation_id = 0
    node.post_replan_recovery = None
    node.post_replan_recovery_activations = 0
    node.last_post_replan_recovery_status = ""
    node.last_post_replan_recovery_phase = ""
    node.last_post_replan_recovery_clear_count = 0
    node.max_post_replan_recovery_clear_count = 0
    node.last_post_replan_recovery_escape_distance_m = 0.0
    node.last_post_replan_recovery_best_escape_distance_m = 0.0
    node.last_post_replan_recovery_escape_distance_source = ""
    node.last_post_replan_recovery_escape_no_motion_elapsed_sec = None
    node.last_post_replan_recovery_escape_straight_active = False
    node.last_post_replan_recovery_escape_elapsed_sec = None
    node.last_post_replan_recovery_escape_timeout_sec = None
    node.last_post_replan_recovery_heading_error_deg = None
    node.last_post_replan_recovery_alignment_heading_deg = None
    node.last_post_replan_recovery_alignment_heading_source = ""
    node.last_post_replan_recovery_alignment_segment_index = None
    node.last_post_replan_recovery_alignment_segment_ratio = None
    node.last_post_replan_recovery_escape_command_linear_mps = 0.0
    node.last_post_replan_recovery_escape_command_angular_radps = 0.0
    node.last_post_replan_recovery_escape_angular_hint_source = ""
    node.last_post_replan_recovery_escape_steering_mode_resolved = ""
    node.last_post_replan_recovery_escape_odom_distance_m = None
    node.last_post_replan_recovery_escape_map_distance_m = None
    node.last_post_replan_recovery_escape_odom_stamp_delta_sec = None
    node.last_post_replan_recovery_escape_progress_source = ""
    node.last_post_replan_recovery_escape_no_motion_reason = ""
    node.last_post_replan_recovery_escape_odom_source = ""
    node.last_post_replan_recovery_escape_odom_source_fallback_reason = ""
    node.last_post_replan_recovery_escape_direct_odom_distance_m = None
    node.last_post_replan_recovery_escape_tf_odom_distance_m = None
    node.last_post_replan_recovery_escape_direct_odom_age_sec = None
    node.last_post_replan_recovery_escape_direct_odom_stamp_delta_sec = None
    node.last_post_replan_recovery_escape_tf_odom_stamp_delta_sec = None
    node.last_post_replan_recovery_escape_direct_odom_frame_id = ""
    node.last_post_replan_recovery_escape_direct_odom_child_frame_id = ""
    node.last_post_replan_recovery_escape_odom_disagreement = ""
    node.last_post_replan_clearance_search_attempted = False
    node.last_post_replan_clearance_search_direction = 0.0
    node.last_post_replan_clearance_search_yaw_delta_deg = 0.0
    node.last_post_replan_clearance_search_baseline_p05_m = None
    node.last_post_replan_clearance_search_best_p05_m = None
    node.last_post_replan_clearance_search_baseline_min_m = None
    node.last_post_replan_clearance_search_best_min_m = None
    node.last_post_replan_clearance_search_result = ""
    node.last_post_replan_clearance_search_direction_source = ""
    node.last_post_replan_route_clearance_reason = ""
    node.last_post_replan_route_corridor_min_distance_m = None
    node.last_post_replan_route_corridor_blocked_count = 0
    node.last_post_replan_route_clear_side_obstacle_count = 0
    node.last_post_replan_route_corridor_preview_distance_m = 0.0
    node.last_post_replan_route_corridor_nearest_blocked_segment_index = None
    node.last_post_replan_route_corridor_nearest_blocked_progress_m = None
    node.last_post_replan_route_corridor_nearest_blocked_penetration_m = None
    node.last_post_replan_route_corridor_nearest_blocked_x_m = None
    node.last_post_replan_route_corridor_nearest_blocked_y_m = None
    node.last_post_replan_route_corridor_nearest_blocked_range_m = None
    node.last_post_replan_route_corridor_nearest_blocked_angle_deg = None
    node.last_post_replan_escape_route_blocked_streak = 0
    node.last_post_replan_escape_route_blocked_tolerated_count = 0
    node.last_post_replan_escape_route_block_decision = ""
    node.post_replan_route_block_repair_count = 0
    node.post_replan_route_block_repair_status = ""
    node.post_replan_route_block_repair_signature = ""
    node.post_replan_route_block_repair_extra_update_used = False
    node.post_replan_route_block_repair_failure_reason = ""
    node.post_replan_route_block_extra_update_count = 0
    node.last_post_replan_route_block_repair_signature = None
    node.last_post_replan_activation_min_target_distance_m = 0.0
    node.last_post_replan_activation_pruned_sparse_count = 0
    node.last_post_replan_activation_pruned_dense_count = 0
    node.last_post_replan_activation_projection_progress_m = None
    node.last_post_replan_activation_first_target_distance_m = None
    node.last_post_replan_activation_status = ""
    node.last_post_replan_recovery_log_sec = None
    node.command_smoother = context.build_command_smoother(args)
    node.last_smoothed_command_time_sec = None
    node.last_smoothed_motion_mode = None
    node.last_velocity_scheduler_status = None
    node.last_velocity_scheduler_log_sec = None
    node.last_route_projection_status = None
    node.last_route_projection_log_sec = None
    node.pure_pursuit_rotate_gate_entries = 0
    node.last_recorded_pure_pursuit_status = None
    node.max_cross_track_error_m = 0.0
    node.cross_track_error_sum_m = 0.0
    node.cross_track_error_count = 0
    node.cross_track_error_samples_m = []
    node.max_route_heading_error_deg = 0.0
    node.angular_feasibility_sample_count = 0
    node.angular_feasibility_limited_count = 0
    node.angular_feasibility_min_scale = 1.0
    node.angular_feasibility_last_scale = None
    node.angular_feasibility_max_raw_angular_z_radps = 0.0
    node.last_route_heading_source = ""
    node.last_route_heading_error_deg = None
    node.last_pure_pursuit_rotate_reason = ""
    node.last_pure_pursuit_rotate_source = ""
    node.max_projection_backward_delta_m = 0.0
    node.max_rotate_anchor_backward_delta_m = 0.0
    node.max_rotate_anchor_forward_delta_m = 0.0
    node.last_rotate_anchor_aligned_samples = 0
    node.max_rotate_anchor_aligned_samples = 0
    node.pure_pursuit_rotate_anchor_activations = 0
    node.post_rotate_branch_lock_activations = 0
    node.post_rotate_branch_ambiguity_failures = 0
    node.post_rotate_branch_rejected_wrong_heading_count = 0
    node.post_rotate_branch_max_heading_error_deg = 0.0
    node.post_rotate_branch_target_clip_count = 0
    node.post_rotate_branch_heading_break_handoff_count = 0
    node.post_rotate_branch_physical_handoff_count = 0
    node.last_projection_acquisition_status = ""
    node.last_projection_lock_sample_count = 0
    node._current_path_controller = None
    node.last_replan_tracking_points = None
    node.last_replan_tracking_source = "waypoints"
    node.last_replan_tracking_validation = None
    node.rviz_last_blocked_cells = set()
    node.replan_manager = context.ReplanManager(node)
    node.lookahead_guard = context.build_lookahead_guard(
        args,
        run_local_map_fn=lambda: node.run_local_map,
    )


def log_startup_configuration(node, args, context):
    if args.verbose:
        node.get_logger().info(
            "Waypoint follower ROS topics: "
            f"cmd_vel={args.cmd_vel_topic}, "
            f"scan={args.scan_topic}, "
            f"amcl={args.amcl_topic}, "
            f"odom={args.odom_topic}, "
            f"max_odom_age_sec={args.max_odom_age_sec:.3f}"
        )
    if node.lookahead_guard is not None and args.verbose:
        node.get_logger().info(
            "Pure-pursuit lookahead guard enabled: "
            f"mode={args.pure_pursuit_lookahead_guard}, "
            "unknown_cells=blocked, "
            "static_inflation_radius_m="
            f"{args.pure_pursuit_lookahead_guard_static_inflation_radius_m:.3f}, "
            f"static_blocked_cells={len(node.lookahead_guard.static_blocked_cells)}"
        )
    if args.controller == "pure-pursuit" and args.verbose:
        short_effective_cap_mps = min(
            abs(float(args.linear_speed)),
            abs(float(args.pure_pursuit_path_profile_short_speed_cap_mps)),
        )
        bend_effective_cap_mps = min(
            abs(float(args.linear_speed)),
            abs(float(args.pure_pursuit_path_profile_bend_speed_cap_mps)),
        )
        node.get_logger().info(
            "Pure-pursuit speed profile: "
            f"profile={args.pure_pursuit_speed_profile}, "
            f"forward_control={args.pure_pursuit_forward_control}, "
            "path_profile_scheduling="
            f"{args.pure_pursuit_path_profile_scheduling}, "
            "path_profile_straight_speed="
            f"{args.pure_pursuit_path_profile_straight_speed_mps:.3f}, "
            "path_profile_short_speed_cap="
            f"{args.pure_pursuit_path_profile_short_speed_cap_mps:.3f}, "
            "path_profile_short_effective_speed_cap="
            f"{short_effective_cap_mps:.3f}, "
            "path_profile_bend_speed_cap="
            f"{args.pure_pursuit_path_profile_bend_speed_cap_mps:.3f}, "
            "path_profile_bend_effective_speed_cap="
            f"{bend_effective_cap_mps:.3f}, "
            f"route_heading_blend={args.pure_pursuit_route_heading_blend:.3f}, "
            f"cross_track_gain={args.pure_pursuit_cross_track_gain:.3f}, "
            "cross_track_speed_floor="
            f"{args.pure_pursuit_cross_track_speed_floor_mps:.3f}, "
            "max_cross_track_correction="
            f"{args.pure_pursuit_max_cross_track_correction_deg:.1f} deg, "
            "angular_feasibility_speed_limit="
            f"{args.pure_pursuit_angular_feasibility_speed_limit}, "
            "angular_feasibility_margin="
            f"{args.pure_pursuit_angular_feasibility_margin:.3f}, "
            f"resolved_linear_speed={args.linear_speed:.3f}, "
            f"resolved_max_angular_speed={args.max_angular_speed:.3f}, "
            f"track_angular_cap={args.pure_pursuit_max_track_angular_speed_radps:.3f}, "
            f"rotate_angular_cap={args.pure_pursuit_max_rotate_angular_speed_radps:.3f}, "
            f"cross_track_warning={args.pure_pursuit_cross_track_warning_m:.3f}, "
            f"cross_track_max={args.pure_pursuit_max_cross_track_error_m:.3f}, "
            "tracking_progress_tolerance="
            f"{args.pure_pursuit_tracking_progress_tolerance_m:.3f}, "
            "projection_lock_samples="
            f"{context.projection_lock_required_samples}, "
            "projection_lock_progress_tolerance="
            f"{context.projection_lock_progress_tolerance_m:.3f}, "
            "route_heading_lookahead="
            f"{context.route_heading_lookahead_m:.3f}, "
            "route_heading_rotate_start="
            f"{args.pure_pursuit_route_heading_rotate_start_deg:.1f} deg, "
            "route_heading_rotate_stop="
            f"{args.pure_pursuit_route_heading_rotate_stop_deg:.1f} deg, "
            "post_rotate_branch_heading_tolerance="
            f"{context.post_rotate_branch_heading_tolerance_deg:.1f} deg, "
            "post_rotate_branch_release_samples="
            f"{context.post_rotate_branch_release_stable_samples}, "
            f"max_lateral_accel={args.pure_pursuit_max_lateral_accel_mps2:.3f}, "
            f"turn_speed_margin={args.pure_pursuit_turn_speed_margin:.3f}, "
            f"heading_deadband={args.pure_pursuit_heading_deadband_deg:.1f} deg, "
            f"lateral_deadband={args.pure_pursuit_lateral_deadband_m:.3f} m, "
            "curvature_limit_start="
            f"{args.pure_pursuit_curvature_limit_start_heading_error_deg:.1f} deg, "
            "curvature_limit_full="
            f"{args.pure_pursuit_curvature_limit_full_heading_error_deg:.1f} deg, "
            "rotate_start="
            f"{args.pure_pursuit_rotate_start_heading_error_deg:.1f} deg, "
            "rotate_stop="
            f"{args.pure_pursuit_rotate_stop_heading_error_deg:.1f} deg"
        )
    if node.command_smoother is not None and args.verbose:
        node.get_logger().info(
            "Pure-pursuit command smoothing enabled: "
            f"mode={args.pure_pursuit_command_smoothing}, "
            f"linear_accel={args.pure_pursuit_max_linear_accel_mps2:.3f}, "
            f"linear_decel={args.pure_pursuit_max_linear_decel_mps2:.3f}, "
            f"angular_accel={args.pure_pursuit_max_angular_accel_radps2:.3f}, "
            f"angular_decel={args.pure_pursuit_max_angular_decel_radps2:.3f}, "
            f"final_decel_distance={args.pure_pursuit_final_decel_distance_m:.3f}, "
            "dt_clamp=[0, 2/control_rate_hz]"
        )
    if args.enable_lidar_map_replan and args.verbose:
        route_clearance_preview = (
            context.post_replan_route_clearance_preview_distance_m(args)
        )
        escape_steering_mode = context.resolve_post_replan_escape_steering_mode(
            args,
        )
        node.get_logger().info(
            "Post-replan recovery: "
            f"mode={args.post_replan_recovery}, "
            f"clear_scan_samples={args.post_replan_clear_scan_samples}, "
            f"clearance_mode={args.post_replan_clearance_mode}, "
            "escape_steering_mode_configured="
            f"{args.post_replan_escape_steering_mode}, "
            "escape_steering_mode_resolved="
            f"{escape_steering_mode}, "
            "route_clearance_preview_distance_configured="
            f"{args.post_replan_route_clearance_preview_distance_m:.3f}, "
            "route_clearance_preview_distance_effective="
            f"{route_clearance_preview:.3f}, "
            f"timeout={args.post_replan_timeout_sec:.3f}, "
            f"escape_distance={args.post_replan_escape_distance_m:.3f}, "
            f"escape_linear_speed={args.post_replan_escape_linear_speed_mps:.3f}, "
            f"align_heading_error={args.post_replan_align_heading_error_deg:.1f} deg"
        )
