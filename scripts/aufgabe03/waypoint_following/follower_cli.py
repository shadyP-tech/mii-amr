from __future__ import annotations

from typing import Any, Mapping


CliContext = Mapping[str, Any]

def require_motion_confirmation(args, waypoints):
    if args.yes:
        return True

    print(f"\nThis command will publish {args.cmd_vel_topic} to the physical TurtleBot.")
    print("Safety requirements:")
    print("  - Nav2 localization is running with the saved map")
    print(f"  - RViz pose estimate is set and {args.scan_topic} aligns with the map")
    print(f"  - no active Nav2 goal/controller is publishing {args.cmd_vel_topic}")
    print("  - clear the path and keep an operator near the robot")
    print("  - keep Ctrl+C and physical stop available")
    print(f"Run ID: {args.run_id}")
    print(f"Waypoints: {len(waypoints)} from {args.waypoints}")
    response = input("Type RUN to start waypoint following: ").strip()
    return response == "RUN"

def wait_before_follow_confirmation(args, current_pose, executable_waypoints, input_fn=input):
    if not args.wait_before_follow:
        return True

    print("\nWaypoint follower handoff is ready.")
    print("The robot is stopped after Nav2 staging and before custom waypoint following.")
    print("Place the temporary obstacle on the planned path now.")
    print("Safety requirements:")
    print("  - keep the path area clear except for the test obstacle")
    print("  - keep Ctrl+C and physical stop available")
    print(
        "Current pose: "
        f"x={current_pose.x:.3f}, y={current_pose.y:.3f}, yaw={current_pose.yaw_deg:.1f} deg"
    )
    if executable_waypoints:
        first = executable_waypoints[0]
        print(
            "First follower waypoint: "
            f"index={first.index}, x={first.x:.3f}, y={first.y:.3f}"
        )
    response = input_fn("Type RUN to start custom waypoint following: ").strip()
    return response == "RUN"

def print_dry_run(
    args,
    raw_waypoints,
    executable_waypoints,
    tracking_validation=None,
    lookahead_guard=None,
    *,
    context,
):
    COMMAND_SMOOTHING_RATE_LIMIT = context['COMMAND_SMOOTHING_RATE_LIMIT']
    LOOKAHEAD_GUARD_OFF = context['LOOKAHEAD_GUARD_OFF']
    POST_REPLAN_ESCAPE_ANGULAR_HINT_CAP_RADPS = context['POST_REPLAN_ESCAPE_ANGULAR_HINT_CAP_RADPS']
    POST_REPLAN_ESCAPE_COMPLETION_TOLERANCE_M = context['POST_REPLAN_ESCAPE_COMPLETION_TOLERANCE_M']
    POST_REPLAN_ESCAPE_MIN_TIMEOUT_SEC = context['POST_REPLAN_ESCAPE_MIN_TIMEOUT_SEC']
    POST_REPLAN_ESCAPE_NO_MOTION_EPS_M = context['POST_REPLAN_ESCAPE_NO_MOTION_EPS_M']
    POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_MAP_SEC = context['POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_MAP_SEC']
    POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_ODOM_SEC = context['POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_ODOM_SEC']
    POST_REPLAN_ESCAPE_STRAIGHT_UNTIL_PROGRESS_M = context['POST_REPLAN_ESCAPE_STRAIGHT_UNTIL_PROGRESS_M']
    POST_REPLAN_ESCAPE_TIMEOUT_MARGIN_SEC = context['POST_REPLAN_ESCAPE_TIMEOUT_MARGIN_SEC']
    POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG = context['POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG']
    POST_ROTATE_BRANCH_MIN_RELEASE_PROGRESS_M = context['POST_ROTATE_BRANCH_MIN_RELEASE_PROGRESS_M']
    POST_ROTATE_BRANCH_RELEASE_STABLE_SAMPLES = context['POST_ROTATE_BRANCH_RELEASE_STABLE_SAMPLES']
    PROJECTION_LOCK_PROGRESS_TOLERANCE_M = context['PROJECTION_LOCK_PROGRESS_TOLERANCE_M']
    PROJECTION_LOCK_REQUIRED_SAMPLES = context['PROJECTION_LOCK_REQUIRED_SAMPLES']
    ROTATE_ANCHOR_ROUTE_HEADING_EXIT_SAMPLES = context['ROTATE_ANCHOR_ROUTE_HEADING_EXIT_SAMPLES']
    ROUTE_HEADING_LOOKAHEAD_M = context['ROUTE_HEADING_LOOKAHEAD_M']
    format_optional_m = context['format_optional_m']
    post_replan_route_clearance_preview_distance_m = context[
        'post_replan_route_clearance_preview_distance_m'
    ]
    resolve_post_replan_escape_steering_mode = context[
        'resolve_post_replan_escape_steering_mode'
    ]
    print("Waypoint follower dry run")
    print(f"Waypoint CSV: {args.waypoints}")
    print(f"Raw waypoints: {len(raw_waypoints)}")
    print(f"Executable waypoints: {len(executable_waypoints)}")
    if executable_waypoints:
        first = executable_waypoints[0]
        last = executable_waypoints[-1]
        print(
            "First executable waypoint: "
            f"source index {first.index}, x={first.x:.3f}, y={first.y:.3f}"
        )
        print(
            "Last executable waypoint: "
            f"source index {last.index}, x={last.x:.3f}, y={last.y:.3f}"
        )
    print(f"Start selection: {args.start_selection}")
    print(f"Wait before follow: {'yes' if args.wait_before_follow else 'no'}")
    print(f"LiDAR map replan: {'enabled' if args.enable_lidar_map_replan else 'disabled'}")
    print(f"Log path: {args.results_csv}")
    if not args.verbose:
        print("Detailed route/config hidden; rerun with --verbose to print it.")
        return

    print(f"Map frame: {args.map_frame}")
    print(f"Base frame: {args.base_frame}, fallback: {args.fallback_base_frame}")
    print(f"cmd_vel topic: {args.cmd_vel_topic}")
    print(f"scan topic: {args.scan_topic}")
    print(f"AMCL topic: {args.amcl_topic}")
    print(f"odom topic: {args.odom_topic}")
    print(f"max odom age: {args.max_odom_age_sec:.3f} sec")
    print(f"Controller: {args.controller}")
    if tracking_validation is not None:
        print(f"controller={args.controller}")
        print(f"tracking_source={tracking_validation.source}")
        print(f"tracking_point_count={tracking_validation.point_count}")
        print(
            "tracking_endpoint_error_m="
            f"{format_optional_m(tracking_validation.endpoint_error_m)}"
        )
        if tracking_validation.start_projection_error_m is None:
            print(
                "tracking_start_error_m="
                f"{format_optional_m(tracking_validation.start_error_m)}"
            )
        else:
            print(
                "tracking_start_projection_error_m="
                f"{format_optional_m(tracking_validation.start_projection_error_m)}"
            )
        print(
            "tracking_validation_status="
            f"{tracking_validation.validation_status}"
        )
    print(f"Linear speed: {args.linear_speed:.3f} m/s")
    print(f"Max angular speed: {args.max_angular_speed:.3f} rad/s")
    print(f"Waypoint tolerance: {args.waypoint_tolerance_m:.3f} m")
    print(f"Goal tolerance: {args.goal_tolerance_m:.3f} m")
    if args.controller == "pure-pursuit":
        print(f"Path lookahead: {args.path_lookahead_m:.3f} m")
        print(
            "Pure-pursuit goal tolerance: "
            f"{args.pure_pursuit_goal_tolerance_m:.3f} m"
        )
        print(f"pure_pursuit_speed_profile={args.pure_pursuit_speed_profile}")
        print(f"pure_pursuit_forward_control={args.pure_pursuit_forward_control}")
        print(
            "pure_pursuit_path_profile_scheduling="
            f"{args.pure_pursuit_path_profile_scheduling}"
        )
        print(
            "pure_pursuit_path_profile_straight_speed_mps="
            f"{args.pure_pursuit_path_profile_straight_speed_mps:.3f}"
        )
        short_effective_cap_mps = min(
            abs(float(args.linear_speed)),
            abs(float(args.pure_pursuit_path_profile_short_speed_cap_mps)),
        )
        bend_effective_cap_mps = min(
            abs(float(args.linear_speed)),
            abs(float(args.pure_pursuit_path_profile_bend_speed_cap_mps)),
        )
        print(
            "pure_pursuit_path_profile_short_speed_cap_mps="
            f"{args.pure_pursuit_path_profile_short_speed_cap_mps:.3f}"
        )
        print(
            "pure_pursuit_path_profile_short_effective_speed_cap_mps="
            f"{short_effective_cap_mps:.3f}"
        )
        print(
            "pure_pursuit_path_profile_bend_speed_cap_mps="
            f"{args.pure_pursuit_path_profile_bend_speed_cap_mps:.3f}"
        )
        print(
            "pure_pursuit_path_profile_bend_effective_speed_cap_mps="
            f"{bend_effective_cap_mps:.3f}"
        )
        print(
            "pure_pursuit_route_heading_blend="
            f"{args.pure_pursuit_route_heading_blend:.3f}"
        )
        print(
            "pure_pursuit_cross_track_gain="
            f"{args.pure_pursuit_cross_track_gain:.3f}"
        )
        print(
            "pure_pursuit_cross_track_speed_floor_mps="
            f"{args.pure_pursuit_cross_track_speed_floor_mps:.3f}"
        )
        print(
            "pure_pursuit_max_cross_track_correction_deg="
            f"{args.pure_pursuit_max_cross_track_correction_deg:.3f}"
        )
        print(
            "pure_pursuit_angular_feasibility_speed_limit="
            f"{args.pure_pursuit_angular_feasibility_speed_limit}"
        )
        print(
            "pure_pursuit_angular_feasibility_margin="
            f"{args.pure_pursuit_angular_feasibility_margin:.3f}"
        )
        print(
            "pure_pursuit_default_linear_speed_resolved_mps="
            f"{args.linear_speed:.3f}"
        )
        print(
            "pure_pursuit_default_max_angular_speed_resolved_radps="
            f"{args.max_angular_speed:.3f}"
        )
        print("pure_pursuit_target_source=route_projection")
        print(
            "pure_pursuit_max_track_angular_speed_radps="
            f"{args.pure_pursuit_max_track_angular_speed_radps:.3f}"
        )
        print(
            "pure_pursuit_max_rotate_angular_speed_radps="
            f"{args.pure_pursuit_max_rotate_angular_speed_radps:.3f}"
        )
        print(
            "pure_pursuit_cross_track_warning_m="
            f"{args.pure_pursuit_cross_track_warning_m:.3f}"
        )
        print(
            "pure_pursuit_max_cross_track_error_m="
            f"{args.pure_pursuit_max_cross_track_error_m:.3f}"
        )
        print(
            "pure_pursuit_tracking_progress_tolerance_m="
            f"{args.pure_pursuit_tracking_progress_tolerance_m:.3f}"
        )
        print(
            "pure_pursuit_projection_lock_required_samples="
            f"{PROJECTION_LOCK_REQUIRED_SAMPLES}"
        )
        print(
            "pure_pursuit_projection_lock_progress_tolerance_m="
            f"{PROJECTION_LOCK_PROGRESS_TOLERANCE_M:.3f}"
        )
        print(
            "pure_pursuit_route_heading_lookahead_m="
            f"{ROUTE_HEADING_LOOKAHEAD_M:.3f}"
        )
        print(
            "pure_pursuit_route_heading_rotate_start_deg="
            f"{args.pure_pursuit_route_heading_rotate_start_deg:.3f}"
        )
        print(
            "pure_pursuit_route_heading_rotate_stop_deg="
            f"{args.pure_pursuit_route_heading_rotate_stop_deg:.3f}"
        )
        print(
            "pure_pursuit_post_rotate_branch_heading_tolerance_deg="
            f"{POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG:.3f}"
        )
        print(
            "pure_pursuit_post_rotate_branch_release_samples="
            f"{POST_ROTATE_BRANCH_RELEASE_STABLE_SAMPLES}"
        )
        print(
            "pure_pursuit_rotate_anchor_route_heading_exit_samples="
            f"{ROTATE_ANCHOR_ROUTE_HEADING_EXIT_SAMPLES}"
        )
        print(
            "pure_pursuit_post_rotate_branch_min_release_progress_m="
            f"{POST_ROTATE_BRANCH_MIN_RELEASE_PROGRESS_M:.3f}"
        )
        print(
            "pure_pursuit_max_lateral_accel_mps2="
            f"{args.pure_pursuit_max_lateral_accel_mps2:.3f}"
        )
        print(
            "pure_pursuit_turn_speed_margin="
            f"{args.pure_pursuit_turn_speed_margin:.3f}"
        )
        print(
            "pure_pursuit_heading_deadband_deg="
            f"{args.pure_pursuit_heading_deadband_deg:.3f}"
        )
        print(
            "pure_pursuit_lateral_deadband_m="
            f"{args.pure_pursuit_lateral_deadband_m:.3f}"
        )
        print(
            "pure_pursuit_curvature_limit_start_heading_error_deg="
            f"{args.pure_pursuit_curvature_limit_start_heading_error_deg:.3f}"
        )
        print(
            "pure_pursuit_curvature_limit_full_heading_error_deg="
            f"{args.pure_pursuit_curvature_limit_full_heading_error_deg:.3f}"
        )
        print(
            "pure_pursuit_rotate_start_heading_error_deg="
            f"{args.pure_pursuit_rotate_start_heading_error_deg:.3f}"
        )
        print(
            "pure_pursuit_rotate_stop_heading_error_deg="
            f"{args.pure_pursuit_rotate_stop_heading_error_deg:.3f}"
        )
        print(
            "pure_pursuit_min_curvature_linear_speed_mps="
            f"{args.pure_pursuit_min_curvature_linear_speed_mps:.3f}"
        )
        print(f"Tracking path CSV: {args.tracking_path_csv or 'none'}")
        print(f"pure_pursuit_lookahead_guard={args.pure_pursuit_lookahead_guard}")
        print(
            "pure_pursuit_min_guarded_lookahead_m="
            f"{args.pure_pursuit_min_guarded_lookahead_m:.3f}"
        )
        if args.pure_pursuit_lookahead_guard != LOOKAHEAD_GUARD_OFF:
            print(
                "lookahead_guard_static_inflation_radius_m="
                f"{args.pure_pursuit_lookahead_guard_static_inflation_radius_m:.3f}"
            )
            print("lookahead_guard_unknown_cells=blocked")
            print(
                "lookahead_guard_static_blocked_cell_count="
                f"{len(lookahead_guard.static_blocked_cells) if lookahead_guard else 'n/a'}"
            )
            print("lookahead_guard_status=configured")
            print("lookahead_guard_selected_target_distance_m=n/a")
            print("lookahead_guard_blocked_cell_count=n/a")
        print(
            "pure_pursuit_command_smoothing="
            f"{args.pure_pursuit_command_smoothing}"
        )
        if args.pure_pursuit_command_smoothing == COMMAND_SMOOTHING_RATE_LIMIT:
            print(
                "pure_pursuit_max_linear_accel_mps2="
                f"{args.pure_pursuit_max_linear_accel_mps2:.3f}"
            )
            print(
                "pure_pursuit_max_linear_decel_mps2="
                f"{args.pure_pursuit_max_linear_decel_mps2:.3f}"
            )
            print(
                "pure_pursuit_max_angular_accel_radps2="
                f"{args.pure_pursuit_max_angular_accel_radps2:.3f}"
            )
            print(
                "pure_pursuit_max_angular_decel_radps2="
                f"{args.pure_pursuit_max_angular_decel_radps2:.3f}"
            )
            print(
                "pure_pursuit_final_decel_distance_m="
                f"{args.pure_pursuit_final_decel_distance_m:.3f}"
            )
            print(
                "pure_pursuit_min_smoothed_linear_speed_mps="
                f"{args.pure_pursuit_min_smoothed_linear_speed_mps:.3f}"
            )
            print("pure_pursuit_smoothing_dt_clamp=[0,2/control_rate_hz]")
    print(f"RViz visualization: {'disabled' if args.no_rviz_visualization else 'enabled'}")
    if not args.no_rviz_visualization:
        print(f"  path topic: {args.rviz_path_topic}")
        print(f"  waypoint markers: {args.rviz_waypoint_marker_topic}")
        print(f"  obstacle markers: {args.rviz_obstacle_marker_topic}")
    if args.enable_lidar_map_replan:
        print(f"  artifact only: {'yes' if args.lidar_replan_artifact_only else 'no'}")
        print(f"  static map: {args.static_map}")
        print(f"  output dir: {args.replan_output_dir}")
        print(
            "  initial scans: "
            f"{args.run_local_map_initial_scan_mode} x "
            f"{args.run_local_map_initial_scan_count}"
        )
        print(f"  update mode: {args.run_local_map_update_mode}")
        print(f"  min hit count: {args.run_local_map_min_hit_count}")
        print(f"  inflation radius: {args.run_local_map_inflation_radius_m:.3f} m")
        print(f"  sparse retry count: {args.run_local_map_sparse_retry_count}")
        print(f"  prune behind distance: {args.run_local_map_prune_behind_distance_m:.3f} m")
        print(f"  post-replan recovery: {args.post_replan_recovery}")
        if args.post_replan_recovery == "on":
            print(f"  post-replan clear scans: {args.post_replan_clear_scan_samples}")
            print(f"  post-replan timeout: {args.post_replan_timeout_sec:.3f} sec")
            print(f"  post-replan escape distance: {args.post_replan_escape_distance_m:.3f} m")
            print(
                "  post-replan escape speed: "
                f"{args.post_replan_escape_linear_speed_mps:.3f} m/s"
            )
            print(
                "  post-replan escape steering mode: "
                f"{args.post_replan_escape_steering_mode} configured, "
                f"{resolve_post_replan_escape_steering_mode(args)} resolved"
            )
            print(
                "  post-replan escape completion tolerance: "
                f"{POST_REPLAN_ESCAPE_COMPLETION_TOLERANCE_M:.3f} m"
            )
            print(
                "  post-replan escape timeout margin: "
                f"{POST_REPLAN_ESCAPE_TIMEOUT_MARGIN_SEC:.3f} sec"
            )
            print(
                "  post-replan escape minimum timeout: "
                f"{POST_REPLAN_ESCAPE_MIN_TIMEOUT_SEC:.3f} sec"
            )
            print(
                "  post-replan escape angular hint cap: "
                f"{POST_REPLAN_ESCAPE_ANGULAR_HINT_CAP_RADPS:.3f} rad/s"
            )
            print(
                "  post-replan escape straight-until-progress: "
                f"{POST_REPLAN_ESCAPE_STRAIGHT_UNTIL_PROGRESS_M:.3f} m"
            )
            print(
                "  post-replan escape no-motion epsilon: "
                f"{POST_REPLAN_ESCAPE_NO_MOTION_EPS_M:.3f} m"
            )
            print(
                "  post-replan escape no-motion timeouts: "
                f"odom={POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_ODOM_SEC:.3f} sec, "
                f"map={POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_MAP_SEC:.3f} sec"
            )
            print(
                "  post-replan align heading error: "
                f"{args.post_replan_align_heading_error_deg:.1f} deg"
            )
            print(
                "  post-replan clearance mode: "
                f"{args.post_replan_clearance_mode}"
            )
            print(
                "  post-replan route clearance preview distance: "
                f"{args.post_replan_route_clearance_preview_distance_m:.3f} m "
                "configured, "
                f"{post_replan_route_clearance_preview_distance_m(args):.3f} m "
                "effective"
            )
    if args.start_selection == "path-progress":
        print(
            "Runtime route selection uses live TF after startup; "
            "the route below is a fixed-skip preview."
        )
    print("Executable route:")
    for index, waypoint in enumerate(executable_waypoints, start=1):
        print(f"  {index}. source index {waypoint.index}: x={waypoint.x:.3f}, y={waypoint.y:.3f}")

def parse_args(argv, *, context):
    COMMAND_SMOOTHING_MODES = context['COMMAND_SMOOTHING_MODES']
    DEFAULT_AMCL_TOPIC = context['DEFAULT_AMCL_TOPIC']
    DEFAULT_CONTROLLER = context['DEFAULT_CONTROLLER']
    DEFAULT_CONTROL_RATE_HZ = context['DEFAULT_CONTROL_RATE_HZ']
    DEFAULT_CMD_VEL_TOPIC = context['DEFAULT_CMD_VEL_TOPIC']
    DEFAULT_FORWARD_STOP_HEADING_ERROR_DEG = context['DEFAULT_FORWARD_STOP_HEADING_ERROR_DEG']
    DEFAULT_FORWARD_YAW_DEADBAND_DEG = context['DEFAULT_FORWARD_YAW_DEADBAND_DEG']
    DEFAULT_GOAL_TOLERANCE_M = context['DEFAULT_GOAL_TOLERANCE_M']
    DEFAULT_HARD_STOP_RANGE_M = context['DEFAULT_HARD_STOP_RANGE_M']
    DEFAULT_LINEAR_GAIN = context['DEFAULT_LINEAR_GAIN']
    DEFAULT_LINEAR_SPEED_MPS = context['DEFAULT_LINEAR_SPEED_MPS']
    DEFAULT_LOCALIZATION_RECOVERY_TIME_SEC = context['DEFAULT_LOCALIZATION_RECOVERY_TIME_SEC']
    DEFAULT_MAX_AMCL_AGE_SEC = context['DEFAULT_MAX_AMCL_AGE_SEC']
    DEFAULT_MAX_AMCL_VAR_X = context['DEFAULT_MAX_AMCL_VAR_X']
    DEFAULT_MAX_AMCL_VAR_Y = context['DEFAULT_MAX_AMCL_VAR_Y']
    DEFAULT_MAX_AMCL_VAR_YAW = context['DEFAULT_MAX_AMCL_VAR_YAW']
    DEFAULT_MAX_ANGULAR_SPEED_RADPS = context['DEFAULT_MAX_ANGULAR_SPEED_RADPS']
    DEFAULT_MAX_GOAL_SNAP_M = context['DEFAULT_MAX_GOAL_SNAP_M']
    DEFAULT_MAX_ODOM_AGE_SEC = context['DEFAULT_MAX_ODOM_AGE_SEC']
    DEFAULT_MAX_POSE_AGE_SEC = context['DEFAULT_MAX_POSE_AGE_SEC']
    DEFAULT_MAX_REPLAN_PATH_LENGTH_RATIO = context['DEFAULT_MAX_REPLAN_PATH_LENGTH_RATIO']
    DEFAULT_MAX_REPLAN_SCAN_AGE_SEC = context['DEFAULT_MAX_REPLAN_SCAN_AGE_SEC']
    DEFAULT_MAX_REPLAN_TF_AGE_SEC = context['DEFAULT_MAX_REPLAN_TF_AGE_SEC']
    DEFAULT_MAX_SCAN_AGE_SEC = context['DEFAULT_MAX_SCAN_AGE_SEC']
    DEFAULT_MAX_START_SNAP_M = context['DEFAULT_MAX_START_SNAP_M']
    DEFAULT_MAX_TF_UPDATE_GAP_SEC = context['DEFAULT_MAX_TF_UPDATE_GAP_SEC']
    DEFAULT_MAX_WAYPOINT_TIME_SEC = context['DEFAULT_MAX_WAYPOINT_TIME_SEC']
    DEFAULT_MIN_LINEAR_SPEED_MPS = context['DEFAULT_MIN_LINEAR_SPEED_MPS']
    DEFAULT_MIN_SCAN_RANGE_M = context['DEFAULT_MIN_SCAN_RANGE_M']
    DEFAULT_MIN_WAYPOINT_SPACING_M = context['DEFAULT_MIN_WAYPOINT_SPACING_M']
    DEFAULT_OBSTACLE_ANGLE_WINDOW_DEG = context['DEFAULT_OBSTACLE_ANGLE_WINDOW_DEG']
    DEFAULT_OBSTACLE_FORWARD_DISTANCE_M = context['DEFAULT_OBSTACLE_FORWARD_DISTANCE_M']
    DEFAULT_OBSTACLE_FORWARD_HALF_WIDTH_M = context['DEFAULT_OBSTACLE_FORWARD_HALF_WIDTH_M']
    DEFAULT_OBSTACLE_INFLATE_RADIUS_M = context['DEFAULT_OBSTACLE_INFLATE_RADIUS_M']
    DEFAULT_OBSTACLE_MIN_CLUSTER_SIZE = context['DEFAULT_OBSTACLE_MIN_CLUSTER_SIZE']
    DEFAULT_OBSTACLE_MIN_CLUSTER_WIDTH_M = context['DEFAULT_OBSTACLE_MIN_CLUSTER_WIDTH_M']
    DEFAULT_OBSTACLE_MIN_RANGE_M = context['DEFAULT_OBSTACLE_MIN_RANGE_M']
    DEFAULT_ODOM_FRAME = context['DEFAULT_ODOM_FRAME']
    DEFAULT_ODOM_TOPIC = context['DEFAULT_ODOM_TOPIC']
    DEFAULT_PATH_LOOKAHEAD_M = context['DEFAULT_PATH_LOOKAHEAD_M']
    DEFAULT_POST_REPLAN_ALIGN_HEADING_ERROR_DEG = context['DEFAULT_POST_REPLAN_ALIGN_HEADING_ERROR_DEG']
    DEFAULT_POST_REPLAN_CLEAR_SCAN_SAMPLES = context['DEFAULT_POST_REPLAN_CLEAR_SCAN_SAMPLES']
    DEFAULT_POST_REPLAN_ESCAPE_DISTANCE_M = context['DEFAULT_POST_REPLAN_ESCAPE_DISTANCE_M']
    DEFAULT_POST_REPLAN_ESCAPE_LINEAR_SPEED_MPS = context['DEFAULT_POST_REPLAN_ESCAPE_LINEAR_SPEED_MPS']
    DEFAULT_POST_REPLAN_ESCAPE_STEERING_MODE = context['DEFAULT_POST_REPLAN_ESCAPE_STEERING_MODE']
    DEFAULT_POST_REPLAN_CLEARANCE_MODE = context['DEFAULT_POST_REPLAN_CLEARANCE_MODE']
    DEFAULT_POST_REPLAN_ROUTE_CLEARANCE_PREVIEW_DISTANCE_M = context['DEFAULT_POST_REPLAN_ROUTE_CLEARANCE_PREVIEW_DISTANCE_M']
    DEFAULT_POST_REPLAN_RECOVERY = context['DEFAULT_POST_REPLAN_RECOVERY']
    DEFAULT_POST_REPLAN_TIMEOUT_SEC = context['DEFAULT_POST_REPLAN_TIMEOUT_SEC']
    DEFAULT_PURE_PURSUIT_ANGULAR_FEASIBILITY_MARGIN = context['DEFAULT_PURE_PURSUIT_ANGULAR_FEASIBILITY_MARGIN']
    DEFAULT_PURE_PURSUIT_ANGULAR_FEASIBILITY_SPEED_LIMIT = context['DEFAULT_PURE_PURSUIT_ANGULAR_FEASIBILITY_SPEED_LIMIT']
    DEFAULT_PURE_PURSUIT_COMMAND_SMOOTHING = context['DEFAULT_PURE_PURSUIT_COMMAND_SMOOTHING']
    DEFAULT_PURE_PURSUIT_CROSS_TRACK_GAIN = context['DEFAULT_PURE_PURSUIT_CROSS_TRACK_GAIN']
    DEFAULT_PURE_PURSUIT_CROSS_TRACK_SPEED_FLOOR_MPS = context['DEFAULT_PURE_PURSUIT_CROSS_TRACK_SPEED_FLOOR_MPS']
    DEFAULT_PURE_PURSUIT_CROSS_TRACK_WARNING_M = context['DEFAULT_PURE_PURSUIT_CROSS_TRACK_WARNING_M']
    DEFAULT_PURE_PURSUIT_CURVATURE_LIMIT_FULL_HEADING_ERROR_DEG = context['DEFAULT_PURE_PURSUIT_CURVATURE_LIMIT_FULL_HEADING_ERROR_DEG']
    DEFAULT_PURE_PURSUIT_CURVATURE_LIMIT_START_HEADING_ERROR_DEG = context['DEFAULT_PURE_PURSUIT_CURVATURE_LIMIT_START_HEADING_ERROR_DEG']
    DEFAULT_PURE_PURSUIT_FINAL_DECEL_DISTANCE_M = context['DEFAULT_PURE_PURSUIT_FINAL_DECEL_DISTANCE_M']
    DEFAULT_PURE_PURSUIT_FORWARD_CONTROL = context['DEFAULT_PURE_PURSUIT_FORWARD_CONTROL']
    DEFAULT_PURE_PURSUIT_HEADING_DEADBAND_DEG = context['DEFAULT_PURE_PURSUIT_HEADING_DEADBAND_DEG']
    DEFAULT_PURE_PURSUIT_LATERAL_DEADBAND_M = context['DEFAULT_PURE_PURSUIT_LATERAL_DEADBAND_M']
    DEFAULT_PURE_PURSUIT_LINEAR_SPEED_MPS = context['DEFAULT_PURE_PURSUIT_LINEAR_SPEED_MPS']
    DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD = context['DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD']
    DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD_STATIC_INFLATION_RADIUS_M = context['DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD_STATIC_INFLATION_RADIUS_M']
    DEFAULT_PURE_PURSUIT_MAX_ANGULAR_ACCEL_RADPS2 = context['DEFAULT_PURE_PURSUIT_MAX_ANGULAR_ACCEL_RADPS2']
    DEFAULT_PURE_PURSUIT_MAX_ANGULAR_DECEL_RADPS2 = context['DEFAULT_PURE_PURSUIT_MAX_ANGULAR_DECEL_RADPS2']
    DEFAULT_PURE_PURSUIT_MAX_ANGULAR_SPEED_RADPS = context['DEFAULT_PURE_PURSUIT_MAX_ANGULAR_SPEED_RADPS']
    DEFAULT_PURE_PURSUIT_MAX_CROSS_TRACK_CORRECTION_DEG = context['DEFAULT_PURE_PURSUIT_MAX_CROSS_TRACK_CORRECTION_DEG']
    DEFAULT_PURE_PURSUIT_MAX_CROSS_TRACK_ERROR_M = context['DEFAULT_PURE_PURSUIT_MAX_CROSS_TRACK_ERROR_M']
    DEFAULT_PURE_PURSUIT_MAX_LATERAL_ACCEL_MPS2 = context['DEFAULT_PURE_PURSUIT_MAX_LATERAL_ACCEL_MPS2']
    DEFAULT_PURE_PURSUIT_MAX_LINEAR_ACCEL_MPS2 = context['DEFAULT_PURE_PURSUIT_MAX_LINEAR_ACCEL_MPS2']
    DEFAULT_PURE_PURSUIT_MAX_LINEAR_DECEL_MPS2 = context['DEFAULT_PURE_PURSUIT_MAX_LINEAR_DECEL_MPS2']
    DEFAULT_PURE_PURSUIT_MAX_TRACK_ANGULAR_SPEED_RADPS = context['DEFAULT_PURE_PURSUIT_MAX_TRACK_ANGULAR_SPEED_RADPS']
    DEFAULT_PURE_PURSUIT_MIN_GUARDED_LOOKAHEAD_M = context['DEFAULT_PURE_PURSUIT_MIN_GUARDED_LOOKAHEAD_M']
    DEFAULT_PURE_PURSUIT_PATH_PROFILE_BEND_SPEED_CAP_MPS = context['DEFAULT_PURE_PURSUIT_PATH_PROFILE_BEND_SPEED_CAP_MPS']
    DEFAULT_PURE_PURSUIT_PATH_PROFILE_SCHEDULING = context['DEFAULT_PURE_PURSUIT_PATH_PROFILE_SCHEDULING']
    DEFAULT_PURE_PURSUIT_PATH_PROFILE_SHORT_SPEED_CAP_MPS = context['DEFAULT_PURE_PURSUIT_PATH_PROFILE_SHORT_SPEED_CAP_MPS']
    DEFAULT_PURE_PURSUIT_PATH_PROFILE_STRAIGHT_SPEED_MPS = context['DEFAULT_PURE_PURSUIT_PATH_PROFILE_STRAIGHT_SPEED_MPS']
    DEFAULT_PURE_PURSUIT_ROTATE_START_HEADING_ERROR_DEG = context['DEFAULT_PURE_PURSUIT_ROTATE_START_HEADING_ERROR_DEG']
    DEFAULT_PURE_PURSUIT_ROTATE_STOP_HEADING_ERROR_DEG = context['DEFAULT_PURE_PURSUIT_ROTATE_STOP_HEADING_ERROR_DEG']
    DEFAULT_PURE_PURSUIT_ROUTE_HEADING_BLEND = context['DEFAULT_PURE_PURSUIT_ROUTE_HEADING_BLEND']
    DEFAULT_PURE_PURSUIT_ROUTE_HEADING_ROTATE_START_DEG = context['DEFAULT_PURE_PURSUIT_ROUTE_HEADING_ROTATE_START_DEG']
    DEFAULT_PURE_PURSUIT_ROUTE_HEADING_ROTATE_STOP_DEG = context['DEFAULT_PURE_PURSUIT_ROUTE_HEADING_ROTATE_STOP_DEG']
    DEFAULT_PURE_PURSUIT_SPEED_PROFILE = context['DEFAULT_PURE_PURSUIT_SPEED_PROFILE']
    DEFAULT_PURE_PURSUIT_TRACKING_PROGRESS_TOLERANCE_M = context['DEFAULT_PURE_PURSUIT_TRACKING_PROGRESS_TOLERANCE_M']
    DEFAULT_PURE_PURSUIT_TURN_SPEED_MARGIN = context['DEFAULT_PURE_PURSUIT_TURN_SPEED_MARGIN']
    DEFAULT_REPLAN_OUTPUT_DIR = context['DEFAULT_REPLAN_OUTPUT_DIR']
    DEFAULT_REPLAN_TIMEOUT_SEC = context['DEFAULT_REPLAN_TIMEOUT_SEC']
    DEFAULT_RESULTS_CSV = context['DEFAULT_RESULTS_CSV']
    DEFAULT_ROBOT_FOOTPRINT_RADIUS_M = context['DEFAULT_ROBOT_FOOTPRINT_RADIUS_M']
    DEFAULT_ROTATE_START_HEADING_ERROR_DEG = context['DEFAULT_ROTATE_START_HEADING_ERROR_DEG']
    DEFAULT_ROTATE_STOP_HEADING_ERROR_DEG = context['DEFAULT_ROTATE_STOP_HEADING_ERROR_DEG']
    DEFAULT_ROTATION_STOP_RANGE_M = context['DEFAULT_ROTATION_STOP_RANGE_M']
    DEFAULT_RUN_LOCAL_MAP_CLEARANCE_MARGIN_M = context['DEFAULT_RUN_LOCAL_MAP_CLEARANCE_MARGIN_M']
    DEFAULT_RUN_LOCAL_MAP_CORRIDOR_CHECK_DISTANCE_M = context['DEFAULT_RUN_LOCAL_MAP_CORRIDOR_CHECK_DISTANCE_M']
    DEFAULT_RUN_LOCAL_MAP_INFLATION_RADIUS_M = context['DEFAULT_RUN_LOCAL_MAP_INFLATION_RADIUS_M']
    DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_COUNT = context['DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_COUNT']
    DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_MODE = context['DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_MODE']
    DEFAULT_RUN_LOCAL_MAP_MAX_REJECTED_RATIO = context['DEFAULT_RUN_LOCAL_MAP_MAX_REJECTED_RATIO']
    DEFAULT_RUN_LOCAL_MAP_MAX_SCAN_AGE_SEC = context['DEFAULT_RUN_LOCAL_MAP_MAX_SCAN_AGE_SEC']
    DEFAULT_RUN_LOCAL_MAP_MAX_TF_AGE_SEC = context['DEFAULT_RUN_LOCAL_MAP_MAX_TF_AGE_SEC']
    DEFAULT_RUN_LOCAL_MAP_MAX_UPDATES = context['DEFAULT_RUN_LOCAL_MAP_MAX_UPDATES']
    DEFAULT_RUN_LOCAL_MAP_MIN_HIT_COUNT = context['DEFAULT_RUN_LOCAL_MAP_MIN_HIT_COUNT']
    DEFAULT_RUN_LOCAL_MAP_MIN_USED_POINTS = context['DEFAULT_RUN_LOCAL_MAP_MIN_USED_POINTS']
    DEFAULT_RUN_LOCAL_MAP_PRUNE_BEHIND_DISTANCE_M = context['DEFAULT_RUN_LOCAL_MAP_PRUNE_BEHIND_DISTANCE_M']
    DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_ANGLE_WINDOW_DEG = context['DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_ANGLE_WINDOW_DEG']
    DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_COUNT = context['DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_COUNT']
    DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_DISTANCE_M = context['DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_DISTANCE_M']
    DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_HALF_WIDTH_M = context['DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_HALF_WIDTH_M']
    DEFAULT_RUN_LOCAL_MAP_UPDATE_MODE = context['DEFAULT_RUN_LOCAL_MAP_UPDATE_MODE']
    DEFAULT_RVIZ_OBSTACLE_MARKER_TOPIC = context['DEFAULT_RVIZ_OBSTACLE_MARKER_TOPIC']
    DEFAULT_RVIZ_PATH_TOPIC = context['DEFAULT_RVIZ_PATH_TOPIC']
    DEFAULT_RVIZ_WAYPOINT_MARKER_TOPIC = context['DEFAULT_RVIZ_WAYPOINT_MARKER_TOPIC']
    DEFAULT_SCAN_TOPIC = context['DEFAULT_SCAN_TOPIC']
    DEFAULT_SCAN_HALF_ANGLE_DEG = context['DEFAULT_SCAN_HALF_ANGLE_DEG']
    DEFAULT_SETTLE_SEC = context['DEFAULT_SETTLE_SEC']
    DEFAULT_STARTUP_TIMEOUT_SEC = context['DEFAULT_STARTUP_TIMEOUT_SEC']
    DEFAULT_START_ON_PATH_TOLERANCE_M = context['DEFAULT_START_ON_PATH_TOLERANCE_M']
    DEFAULT_START_SELECTION = context['DEFAULT_START_SELECTION']
    DEFAULT_STATIC_MAP = context['DEFAULT_STATIC_MAP']
    DEFAULT_TF_RECOVERY_TIME_SEC = context['DEFAULT_TF_RECOVERY_TIME_SEC']
    DEFAULT_TRACKING_ENDPOINT_TOLERANCE_M = context['DEFAULT_TRACKING_ENDPOINT_TOLERANCE_M']
    DEFAULT_TRACKING_MAX_SEGMENT_M = context['DEFAULT_TRACKING_MAX_SEGMENT_M']
    DEFAULT_TRACKING_START_TOLERANCE_M = context['DEFAULT_TRACKING_START_TOLERANCE_M']
    DEFAULT_WAYPOINTS_CSV = context['DEFAULT_WAYPOINTS_CSV']
    DEFAULT_WAYPOINT_TOLERANCE_M = context['DEFAULT_WAYPOINT_TOLERANCE_M']
    DEFAULT_YAW_GAIN = context['DEFAULT_YAW_GAIN']
    FORWARD_CONTROL_MODES = context['FORWARD_CONTROL_MODES']
    LOOKAHEAD_GUARD_MODES = context['LOOKAHEAD_GUARD_MODES']
    PATH_PROFILE_SCHEDULING_MODES = context['PATH_PROFILE_SCHEDULING_MODES']
    POST_REPLAN_RECOVERY_MODES = context['POST_REPLAN_RECOVERY_MODES']
    POST_REPLAN_CLEARANCE_MODES = context['POST_REPLAN_CLEARANCE_MODES']
    POST_REPLAN_ESCAPE_STEERING_MODES = context['POST_REPLAN_ESCAPE_STEERING_MODES']
    Path = context['Path']
    SPEED_PROFILE_MODES = context['SPEED_PROFILE_MODES']
    argparse = context['argparse']
    datetime = context['datetime']
    sys = context['sys']
    parse_argv = list(argv) if argv is not None else sys.argv[1:]
    max_angular_speed_explicit = any(
        token == "--max-angular-speed"
        or token.startswith("--max-angular-speed=")
        for token in parse_argv
    )
    parser = argparse.ArgumentParser(
        description="Follow planned A* waypoints using TF pose and /cmd_vel.",
    )
    parser.add_argument("--waypoints", default=DEFAULT_WAYPOINTS_CSV, type=Path)
    parser.add_argument("--run-id", help="Run ID for logging.")
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV, type=Path)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--fallback-base-frame", default="base_link")
    parser.add_argument("--cmd-vel-topic", default=DEFAULT_CMD_VEL_TOPIC)
    parser.add_argument("--scan-topic", default=DEFAULT_SCAN_TOPIC)
    parser.add_argument("--amcl-topic", default=DEFAULT_AMCL_TOPIC)
    parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
    parser.add_argument("--linear-speed", type=float)
    parser.add_argument("--min-linear-speed", default=DEFAULT_MIN_LINEAR_SPEED_MPS, type=float)
    parser.add_argument("--linear-gain", default=DEFAULT_LINEAR_GAIN, type=float)
    parser.add_argument("--max-angular-speed", type=float)
    parser.add_argument("--yaw-gain", default=DEFAULT_YAW_GAIN, type=float)
    parser.add_argument("--forward-yaw-deadband-deg", default=DEFAULT_FORWARD_YAW_DEADBAND_DEG, type=float)
    parser.add_argument("--forward-stop-heading-error-deg", default=DEFAULT_FORWARD_STOP_HEADING_ERROR_DEG, type=float)
    parser.add_argument("--waypoint-tolerance-m", default=DEFAULT_WAYPOINT_TOLERANCE_M, type=float)
    parser.add_argument("--goal-tolerance-m", default=DEFAULT_GOAL_TOLERANCE_M, type=float)
    parser.add_argument(
        "--controller",
        default=DEFAULT_CONTROLLER,
        choices=["stop-go", "pure-pursuit"],
    )
    parser.add_argument("--path-lookahead-m", default=DEFAULT_PATH_LOOKAHEAD_M, type=float)
    parser.add_argument("--pure-pursuit-goal-tolerance-m", type=float)
    parser.add_argument(
        "--pure-pursuit-lookahead-guard",
        default=DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD,
        choices=LOOKAHEAD_GUARD_MODES,
    )
    parser.add_argument(
        "--pure-pursuit-min-guarded-lookahead-m",
        default=DEFAULT_PURE_PURSUIT_MIN_GUARDED_LOOKAHEAD_M,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-lookahead-guard-static-inflation-radius-m",
        default=DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD_STATIC_INFLATION_RADIUS_M,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-command-smoothing",
        default=DEFAULT_PURE_PURSUIT_COMMAND_SMOOTHING,
        choices=COMMAND_SMOOTHING_MODES,
    )
    parser.add_argument(
        "--pure-pursuit-speed-profile",
        default=DEFAULT_PURE_PURSUIT_SPEED_PROFILE,
        choices=SPEED_PROFILE_MODES,
    )
    parser.add_argument(
        "--pure-pursuit-forward-control",
        default=DEFAULT_PURE_PURSUIT_FORWARD_CONTROL,
        choices=FORWARD_CONTROL_MODES,
    )
    parser.add_argument(
        "--pure-pursuit-path-profile-scheduling",
        default=DEFAULT_PURE_PURSUIT_PATH_PROFILE_SCHEDULING,
        choices=PATH_PROFILE_SCHEDULING_MODES,
    )
    parser.add_argument(
        "--pure-pursuit-path-profile-straight-speed-mps",
        default=DEFAULT_PURE_PURSUIT_PATH_PROFILE_STRAIGHT_SPEED_MPS,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-path-profile-short-speed-cap-mps",
        default=DEFAULT_PURE_PURSUIT_PATH_PROFILE_SHORT_SPEED_CAP_MPS,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-path-profile-bend-speed-cap-mps",
        default=DEFAULT_PURE_PURSUIT_PATH_PROFILE_BEND_SPEED_CAP_MPS,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-route-heading-blend",
        default=DEFAULT_PURE_PURSUIT_ROUTE_HEADING_BLEND,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-cross-track-gain",
        default=DEFAULT_PURE_PURSUIT_CROSS_TRACK_GAIN,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-cross-track-speed-floor-mps",
        default=DEFAULT_PURE_PURSUIT_CROSS_TRACK_SPEED_FLOOR_MPS,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-max-cross-track-correction-deg",
        default=DEFAULT_PURE_PURSUIT_MAX_CROSS_TRACK_CORRECTION_DEG,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-angular-feasibility-speed-limit",
        default=DEFAULT_PURE_PURSUIT_ANGULAR_FEASIBILITY_SPEED_LIMIT,
        choices=["on", "off"],
    )
    parser.add_argument(
        "--pure-pursuit-angular-feasibility-margin",
        default=DEFAULT_PURE_PURSUIT_ANGULAR_FEASIBILITY_MARGIN,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-max-lateral-accel-mps2",
        default=DEFAULT_PURE_PURSUIT_MAX_LATERAL_ACCEL_MPS2,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-turn-speed-margin",
        default=DEFAULT_PURE_PURSUIT_TURN_SPEED_MARGIN,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-heading-deadband-deg",
        default=DEFAULT_PURE_PURSUIT_HEADING_DEADBAND_DEG,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-lateral-deadband-m",
        default=DEFAULT_PURE_PURSUIT_LATERAL_DEADBAND_M,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-curvature-limit-start-heading-error-deg",
        default=DEFAULT_PURE_PURSUIT_CURVATURE_LIMIT_START_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-curvature-limit-full-heading-error-deg",
        default=DEFAULT_PURE_PURSUIT_CURVATURE_LIMIT_FULL_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-rotate-start-heading-error-deg",
        default=DEFAULT_PURE_PURSUIT_ROTATE_START_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-rotate-stop-heading-error-deg",
        default=DEFAULT_PURE_PURSUIT_ROTATE_STOP_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-route-heading-rotate-start-deg",
        default=DEFAULT_PURE_PURSUIT_ROUTE_HEADING_ROTATE_START_DEG,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-route-heading-rotate-stop-deg",
        default=DEFAULT_PURE_PURSUIT_ROUTE_HEADING_ROTATE_STOP_DEG,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-max-track-angular-speed-radps",
        default=DEFAULT_PURE_PURSUIT_MAX_TRACK_ANGULAR_SPEED_RADPS,
        type=float,
    )
    parser.add_argument("--pure-pursuit-max-rotate-angular-speed-radps", type=float)
    parser.add_argument(
        "--pure-pursuit-cross-track-warning-m",
        default=DEFAULT_PURE_PURSUIT_CROSS_TRACK_WARNING_M,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-max-cross-track-error-m",
        default=DEFAULT_PURE_PURSUIT_MAX_CROSS_TRACK_ERROR_M,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-tracking-progress-tolerance-m",
        default=DEFAULT_PURE_PURSUIT_TRACKING_PROGRESS_TOLERANCE_M,
        type=float,
    )
    parser.add_argument("--pure-pursuit-min-curvature-linear-speed-mps", type=float)
    parser.add_argument(
        "--pure-pursuit-max-linear-accel-mps2",
        default=DEFAULT_PURE_PURSUIT_MAX_LINEAR_ACCEL_MPS2,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-max-linear-decel-mps2",
        default=DEFAULT_PURE_PURSUIT_MAX_LINEAR_DECEL_MPS2,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-max-angular-accel-radps2",
        default=DEFAULT_PURE_PURSUIT_MAX_ANGULAR_ACCEL_RADPS2,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-max-angular-decel-radps2",
        default=DEFAULT_PURE_PURSUIT_MAX_ANGULAR_DECEL_RADPS2,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-final-decel-distance-m",
        default=DEFAULT_PURE_PURSUIT_FINAL_DECEL_DISTANCE_M,
        type=float,
    )
    parser.add_argument("--pure-pursuit-min-smoothed-linear-speed-mps", type=float)
    parser.add_argument("--tracking-path-csv", type=Path)
    parser.add_argument(
        "--tracking-endpoint-tolerance-m",
        default=DEFAULT_TRACKING_ENDPOINT_TOLERANCE_M,
        type=float,
    )
    parser.add_argument(
        "--tracking-start-tolerance-m",
        default=DEFAULT_TRACKING_START_TOLERANCE_M,
        type=float,
    )
    parser.add_argument(
        "--tracking-max-segment-m",
        default=DEFAULT_TRACKING_MAX_SEGMENT_M,
        type=float,
    )
    parser.add_argument("--allow-tracking-path-mismatch", action="store_true")
    parser.add_argument(
        "--rotate-start-heading-error-deg",
        default=DEFAULT_ROTATE_START_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument(
        "--rotate-stop-heading-error-deg",
        default=DEFAULT_ROTATE_STOP_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument("--min-waypoint-spacing-m", default=DEFAULT_MIN_WAYPOINT_SPACING_M, type=float)
    parser.add_argument(
        "--start-selection",
        default=DEFAULT_START_SELECTION,
        choices=["path-progress", "fixed-skip"],
    )
    parser.add_argument("--start-on-path-tolerance-m", default=DEFAULT_START_ON_PATH_TOLERANCE_M, type=float)
    parser.add_argument("--odom-frame", default=DEFAULT_ODOM_FRAME)
    parser.add_argument("--scan-half-angle-deg", default=DEFAULT_SCAN_HALF_ANGLE_DEG, type=float)
    parser.add_argument("--hard-stop-range-m", default=DEFAULT_HARD_STOP_RANGE_M, type=float)
    parser.add_argument("--min-scan-range-m", default=DEFAULT_MIN_SCAN_RANGE_M, type=float)
    parser.add_argument("--rotation-stop-range-m", default=DEFAULT_ROTATION_STOP_RANGE_M, type=float)
    parser.add_argument("--max-pose-age-sec", default=DEFAULT_MAX_POSE_AGE_SEC, type=float)
    parser.add_argument("--max-scan-age-sec", default=DEFAULT_MAX_SCAN_AGE_SEC, type=float)
    parser.add_argument("--max-amcl-age-sec", default=DEFAULT_MAX_AMCL_AGE_SEC, type=float)
    parser.add_argument("--max-odom-age-sec", default=DEFAULT_MAX_ODOM_AGE_SEC, type=float)
    parser.add_argument("--max-amcl-var-x", default=DEFAULT_MAX_AMCL_VAR_X, type=float)
    parser.add_argument("--max-amcl-var-y", default=DEFAULT_MAX_AMCL_VAR_Y, type=float)
    parser.add_argument("--max-amcl-var-yaw", default=DEFAULT_MAX_AMCL_VAR_YAW, type=float)
    parser.add_argument("--max-waypoint-time-sec", default=DEFAULT_MAX_WAYPOINT_TIME_SEC, type=float)
    parser.add_argument("--max-tf-update-gap-sec", default=DEFAULT_MAX_TF_UPDATE_GAP_SEC, type=float)
    parser.add_argument("--tf-recovery-time-sec", default=DEFAULT_TF_RECOVERY_TIME_SEC, type=float)
    parser.add_argument(
        "--localization-recovery-time-sec",
        default=DEFAULT_LOCALIZATION_RECOVERY_TIME_SEC,
        type=float,
    )
    parser.add_argument("--control-rate-hz", default=DEFAULT_CONTROL_RATE_HZ, type=float)
    parser.add_argument("--settle-sec", default=DEFAULT_SETTLE_SEC, type=float)
    parser.add_argument("--startup-timeout-sec", default=DEFAULT_STARTUP_TIMEOUT_SEC, type=float)
    parser.add_argument("--notes", default="follow_planned_waypoints")
    parser.add_argument("--fail-on-bad-localization", action="store_true")
    parser.add_argument("--pause-on-bad-localization", action="store_true")
    parser.add_argument("--require-amcl-startup", action="store_true")
    parser.add_argument("--fail-on-stale-tf", action="store_true")
    parser.add_argument("--no-skip-first-waypoint", action="store_true")
    parser.add_argument("--rviz-path-topic", default=DEFAULT_RVIZ_PATH_TOPIC)
    parser.add_argument(
        "--rviz-waypoint-marker-topic",
        default=DEFAULT_RVIZ_WAYPOINT_MARKER_TOPIC,
    )
    parser.add_argument(
        "--rviz-obstacle-marker-topic",
        default=DEFAULT_RVIZ_OBSTACLE_MARKER_TOPIC,
    )
    parser.add_argument("--no-rviz-visualization", action="store_true")
    parser.add_argument("--enable-lidar-map-replan", action="store_true")
    parser.add_argument("--lidar-replan-artifact-only", action="store_true")
    parser.add_argument("--static-map", default=DEFAULT_STATIC_MAP, type=Path)
    parser.add_argument("--replan-output-dir", default=DEFAULT_REPLAN_OUTPUT_DIR, type=Path)
    parser.add_argument("--max-replans", default=1, type=int)
    parser.add_argument("--replan-timeout-sec", default=DEFAULT_REPLAN_TIMEOUT_SEC, type=float)
    parser.add_argument("--max-replan-scan-age-sec", default=DEFAULT_MAX_REPLAN_SCAN_AGE_SEC, type=float)
    parser.add_argument("--max-replan-tf-age-sec", default=DEFAULT_MAX_REPLAN_TF_AGE_SEC, type=float)
    parser.add_argument("--allow-latest-tf-replan-fallback", action="store_true")
    parser.add_argument("--obstacle-forward-distance-m", default=DEFAULT_OBSTACLE_FORWARD_DISTANCE_M, type=float)
    parser.add_argument("--obstacle-forward-half-width-m", default=DEFAULT_OBSTACLE_FORWARD_HALF_WIDTH_M, type=float)
    parser.add_argument("--obstacle-angle-window-deg", default=DEFAULT_OBSTACLE_ANGLE_WINDOW_DEG, type=float)
    parser.add_argument("--obstacle-min-range-m", default=DEFAULT_OBSTACLE_MIN_RANGE_M, type=float)
    parser.add_argument("--robot-footprint-radius-m", default=DEFAULT_ROBOT_FOOTPRINT_RADIUS_M, type=float)
    parser.add_argument("--obstacle-min-cluster-size", default=DEFAULT_OBSTACLE_MIN_CLUSTER_SIZE, type=int)
    parser.add_argument("--obstacle-min-cluster-width-m", default=DEFAULT_OBSTACLE_MIN_CLUSTER_WIDTH_M, type=float)
    parser.add_argument("--obstacle-inflate-radius-m", default=DEFAULT_OBSTACLE_INFLATE_RADIUS_M, type=float)
    parser.add_argument("--max-start-snap-m", default=DEFAULT_MAX_START_SNAP_M, type=float)
    parser.add_argument("--max-goal-snap-m", default=DEFAULT_MAX_GOAL_SNAP_M, type=float)
    parser.add_argument(
        "--max-replan-path-length-ratio",
        default=DEFAULT_MAX_REPLAN_PATH_LENGTH_RATIO,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-initial-scan-mode",
        default=DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_MODE,
        choices=["none", "forward", "full"],
    )
    parser.add_argument(
        "--run-local-map-initial-scan-count",
        default=DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_COUNT,
        type=int,
    )
    parser.add_argument(
        "--run-local-map-update-mode",
        default=DEFAULT_RUN_LOCAL_MAP_UPDATE_MODE,
        choices=["none", "forward", "full"],
    )
    parser.add_argument(
        "--run-local-map-min-hit-count",
        default=DEFAULT_RUN_LOCAL_MAP_MIN_HIT_COUNT,
        type=int,
    )
    parser.add_argument(
        "--run-local-map-inflation-radius-m",
        default=DEFAULT_RUN_LOCAL_MAP_INFLATION_RADIUS_M,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-max-tf-age-sec",
        default=DEFAULT_RUN_LOCAL_MAP_MAX_TF_AGE_SEC,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-max-scan-age-sec",
        default=DEFAULT_RUN_LOCAL_MAP_MAX_SCAN_AGE_SEC,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-min-used-points",
        default=DEFAULT_RUN_LOCAL_MAP_MIN_USED_POINTS,
        type=int,
    )
    parser.add_argument(
        "--run-local-map-max-rejected-ratio",
        default=DEFAULT_RUN_LOCAL_MAP_MAX_REJECTED_RATIO,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-corridor-check-distance-m",
        default=DEFAULT_RUN_LOCAL_MAP_CORRIDOR_CHECK_DISTANCE_M,
        type=float,
    )
    parser.add_argument("--run-local-map-corridor-radius-m", type=float)
    parser.add_argument(
        "--run-local-map-clearance-margin-m",
        default=DEFAULT_RUN_LOCAL_MAP_CLEARANCE_MARGIN_M,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-max-updates",
        default=DEFAULT_RUN_LOCAL_MAP_MAX_UPDATES,
        type=int,
    )
    parser.add_argument(
        "--run-local-map-sparse-retry-count",
        default=DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_COUNT,
        type=int,
    )
    parser.add_argument(
        "--run-local-map-sparse-retry-forward-half-width-m",
        default=DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_HALF_WIDTH_M,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-sparse-retry-angle-window-deg",
        default=DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_ANGLE_WINDOW_DEG,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-sparse-retry-forward-distance-m",
        default=DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_DISTANCE_M,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-prune-behind-distance-m",
        default=DEFAULT_RUN_LOCAL_MAP_PRUNE_BEHIND_DISTANCE_M,
        type=float,
    )
    parser.add_argument(
        "--post-replan-recovery",
        default=DEFAULT_POST_REPLAN_RECOVERY,
        choices=POST_REPLAN_RECOVERY_MODES,
    )
    parser.add_argument(
        "--post-replan-clear-scan-samples",
        default=DEFAULT_POST_REPLAN_CLEAR_SCAN_SAMPLES,
        type=int,
    )
    parser.add_argument(
        "--post-replan-timeout-sec",
        default=DEFAULT_POST_REPLAN_TIMEOUT_SEC,
        type=float,
    )
    parser.add_argument(
        "--post-replan-escape-distance-m",
        default=DEFAULT_POST_REPLAN_ESCAPE_DISTANCE_M,
        type=float,
    )
    parser.add_argument(
        "--post-replan-escape-linear-speed-mps",
        default=DEFAULT_POST_REPLAN_ESCAPE_LINEAR_SPEED_MPS,
        type=float,
    )
    parser.add_argument(
        "--post-replan-escape-steering-mode",
        default=DEFAULT_POST_REPLAN_ESCAPE_STEERING_MODE,
        choices=POST_REPLAN_ESCAPE_STEERING_MODES,
    )
    parser.add_argument(
        "--post-replan-align-heading-error-deg",
        default=DEFAULT_POST_REPLAN_ALIGN_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument(
        "--post-replan-clearance-mode",
        default=DEFAULT_POST_REPLAN_CLEARANCE_MODE,
        choices=POST_REPLAN_CLEARANCE_MODES,
    )
    parser.add_argument(
        "--post-replan-route-clearance-preview-distance-m",
        default=DEFAULT_POST_REPLAN_ROUTE_CLEARANCE_PREVIEW_DISTANCE_M,
        type=float,
    )
    parser.add_argument("--run-local-map-artifact-prefix")
    parser.add_argument("--wait-before-follow", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed route/configuration output.",
    )
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args(parse_argv)

    if not args.run_id:
        args.run_id = datetime.now().strftime("waypoint_follow_%Y%m%d_%H%M%S")
    if args.linear_speed is None:
        args.linear_speed = (
            DEFAULT_PURE_PURSUIT_LINEAR_SPEED_MPS
            if args.controller == "pure-pursuit"
            else DEFAULT_LINEAR_SPEED_MPS
        )
    if args.max_angular_speed is None:
        args.max_angular_speed = (
            DEFAULT_PURE_PURSUIT_MAX_ANGULAR_SPEED_RADPS
            if args.controller == "pure-pursuit"
            else DEFAULT_MAX_ANGULAR_SPEED_RADPS
        )
    args.max_angular_speed_explicit = max_angular_speed_explicit
    if args.pure_pursuit_max_rotate_angular_speed_radps is None:
        args.pure_pursuit_max_rotate_angular_speed_radps = (
            args.max_angular_speed
            if args.controller == "pure-pursuit" and max_angular_speed_explicit
            else DEFAULT_PURE_PURSUIT_MAX_ANGULAR_SPEED_RADPS
        )
    if args.pure_pursuit_goal_tolerance_m is None:
        args.pure_pursuit_goal_tolerance_m = args.goal_tolerance_m
    if args.pure_pursuit_min_curvature_linear_speed_mps is None:
        args.pure_pursuit_min_curvature_linear_speed_mps = args.min_linear_speed
    if args.pure_pursuit_min_smoothed_linear_speed_mps is None:
        args.pure_pursuit_min_smoothed_linear_speed_mps = args.min_linear_speed
    validate_args(parser, args, context=context)
    return args

def validate_args(parser, args, *, context):
    FORWARD_CONTROL_ROUTE_DAMPED = context['FORWARD_CONTROL_ROUTE_DAMPED']
    SPEED_PROFILE_FIXED = context['SPEED_PROFILE_FIXED']
    positive_fields = [
        "linear_speed",
        "min_linear_speed",
        "linear_gain",
        "max_angular_speed",
        "yaw_gain",
        "waypoint_tolerance_m",
        "goal_tolerance_m",
        "path_lookahead_m",
        "pure_pursuit_goal_tolerance_m",
        "pure_pursuit_min_guarded_lookahead_m",
        "pure_pursuit_max_lateral_accel_mps2",
        "pure_pursuit_rotate_start_heading_error_deg",
        "pure_pursuit_rotate_stop_heading_error_deg",
        "pure_pursuit_route_heading_rotate_start_deg",
        "pure_pursuit_route_heading_rotate_stop_deg",
        "pure_pursuit_path_profile_straight_speed_mps",
        "pure_pursuit_path_profile_short_speed_cap_mps",
        "pure_pursuit_path_profile_bend_speed_cap_mps",
        "pure_pursuit_max_track_angular_speed_radps",
        "pure_pursuit_max_rotate_angular_speed_radps",
        "pure_pursuit_cross_track_speed_floor_mps",
        "pure_pursuit_cross_track_warning_m",
        "pure_pursuit_max_cross_track_error_m",
        "pure_pursuit_tracking_progress_tolerance_m",
        "pure_pursuit_angular_feasibility_margin",
        "tracking_endpoint_tolerance_m",
        "tracking_start_tolerance_m",
        "tracking_max_segment_m",
        "rotate_start_heading_error_deg",
        "rotate_stop_heading_error_deg",
        "scan_half_angle_deg",
        "hard_stop_range_m",
        "min_scan_range_m",
        "rotation_stop_range_m",
        "start_on_path_tolerance_m",
        "max_pose_age_sec",
        "max_scan_age_sec",
        "max_amcl_age_sec",
        "max_odom_age_sec",
        "max_amcl_var_x",
        "max_amcl_var_y",
        "max_amcl_var_yaw",
        "max_waypoint_time_sec",
        "max_tf_update_gap_sec",
        "tf_recovery_time_sec",
        "localization_recovery_time_sec",
        "control_rate_hz",
        "startup_timeout_sec",
        "replan_timeout_sec",
        "max_replan_scan_age_sec",
        "max_replan_tf_age_sec",
        "obstacle_forward_distance_m",
        "obstacle_forward_half_width_m",
        "obstacle_angle_window_deg",
        "obstacle_min_range_m",
        "robot_footprint_radius_m",
        "obstacle_min_cluster_width_m",
        "obstacle_inflate_radius_m",
        "max_start_snap_m",
        "max_goal_snap_m",
        "max_replan_path_length_ratio",
        "run_local_map_inflation_radius_m",
        "run_local_map_max_tf_age_sec",
        "run_local_map_max_scan_age_sec",
        "run_local_map_corridor_check_distance_m",
        "run_local_map_clearance_margin_m",
        "run_local_map_sparse_retry_forward_half_width_m",
        "run_local_map_sparse_retry_angle_window_deg",
        "run_local_map_sparse_retry_forward_distance_m",
        "run_local_map_prune_behind_distance_m",
        "post_replan_timeout_sec",
        "post_replan_escape_distance_m",
        "post_replan_escape_linear_speed_mps",
        "post_replan_align_heading_error_deg",
    ]
    for field in positive_fields:
        if getattr(args, field) <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.min_linear_speed > args.linear_speed:
        parser.error("--min-linear-speed must be <= --linear-speed")
    if args.rotate_stop_heading_error_deg >= args.rotate_start_heading_error_deg:
        parser.error("--rotate-stop-heading-error-deg must be < --rotate-start-heading-error-deg")
    if args.forward_yaw_deadband_deg < 0.0:
        parser.error("--forward-yaw-deadband-deg must be non-negative")
    if args.forward_yaw_deadband_deg >= args.forward_stop_heading_error_deg:
        parser.error("--forward-yaw-deadband-deg must be < --forward-stop-heading-error-deg")
    if args.forward_stop_heading_error_deg >= args.rotate_start_heading_error_deg:
        parser.error(
            "--forward-stop-heading-error-deg must be < "
            "--rotate-start-heading-error-deg"
        )
    if args.hard_stop_range_m >= args.min_scan_range_m:
        parser.error("--hard-stop-range-m must be < --min-scan-range-m")
    if args.hard_stop_range_m >= args.rotation_stop_range_m:
        parser.error("--hard-stop-range-m must be < --rotation-stop-range-m")
    if not (0.0 < args.scan_half_angle_deg <= 90.0):
        parser.error("--scan-half-angle-deg must be > 0 and <= 90")
    if args.settle_sec < 0.0:
        parser.error("--settle-sec must be non-negative")
    if args.max_replans < 1:
        parser.error("--max-replans must be >= 1")
    if args.run_local_map_initial_scan_count < 1:
        parser.error("--run-local-map-initial-scan-count must be >= 1")
    if args.run_local_map_min_hit_count < 1:
        parser.error("--run-local-map-min-hit-count must be >= 1")
    if args.run_local_map_min_used_points < 1:
        parser.error("--run-local-map-min-used-points must be >= 1")
    if not (0.0 <= args.run_local_map_max_rejected_ratio <= 1.0):
        parser.error("--run-local-map-max-rejected-ratio must be between 0 and 1")
    if (
        args.run_local_map_corridor_radius_m is not None
        and args.run_local_map_corridor_radius_m <= 0.0
    ):
        parser.error("--run-local-map-corridor-radius-m must be greater than zero")
    if args.run_local_map_max_updates < 1:
        parser.error("--run-local-map-max-updates must be >= 1")
    if args.run_local_map_sparse_retry_count < 0:
        parser.error("--run-local-map-sparse-retry-count must be >= 0")
    if args.run_local_map_sparse_retry_angle_window_deg > 90.0:
        parser.error("--run-local-map-sparse-retry-angle-window-deg must be <= 90")
    if args.post_replan_clear_scan_samples < 1:
        parser.error("--post-replan-clear-scan-samples must be >= 1")
    if args.post_replan_escape_linear_speed_mps > args.linear_speed:
        parser.error(
            "--post-replan-escape-linear-speed-mps must be <= --linear-speed"
        )
    if args.obstacle_min_cluster_size < 1:
        parser.error("--obstacle-min-cluster-size must be >= 1")
    if args.pure_pursuit_lookahead_guard_static_inflation_radius_m < 0.0:
        parser.error(
            "--pure-pursuit-lookahead-guard-static-inflation-radius-m "
            "must be non-negative"
        )
    non_negative_fields = [
        "pure_pursuit_max_linear_accel_mps2",
        "pure_pursuit_max_linear_decel_mps2",
        "pure_pursuit_max_angular_accel_radps2",
        "pure_pursuit_max_angular_decel_radps2",
        "pure_pursuit_min_smoothed_linear_speed_mps",
        "pure_pursuit_min_curvature_linear_speed_mps",
        "pure_pursuit_heading_deadband_deg",
        "pure_pursuit_lateral_deadband_m",
        "pure_pursuit_cross_track_gain",
        "pure_pursuit_max_cross_track_correction_deg",
        "post_replan_route_clearance_preview_distance_m",
    ]
    for field in non_negative_fields:
        if getattr(args, field) < 0.0:
            parser.error(f"--{field.replace('_', '-')} must be non-negative")
    if not (0.0 <= args.pure_pursuit_route_heading_blend <= 1.0):
        parser.error("--pure-pursuit-route-heading-blend must be between 0 and 1")
    if args.pure_pursuit_max_cross_track_correction_deg > 90.0:
        parser.error("--pure-pursuit-max-cross-track-correction-deg must be <= 90")
    if args.pure_pursuit_angular_feasibility_margin > 1.0:
        parser.error("--pure-pursuit-angular-feasibility-margin must be <= 1")
    if (
        args.controller == "pure-pursuit"
        and args.pure_pursuit_forward_control == FORWARD_CONTROL_ROUTE_DAMPED
        and args.pure_pursuit_speed_profile != SPEED_PROFILE_FIXED
    ):
        parser.error(
            "--pure-pursuit-forward-control route-damped requires "
            "--pure-pursuit-speed-profile fixed"
        )
    if not (0.0 < args.pure_pursuit_turn_speed_margin <= 1.0):
        parser.error("--pure-pursuit-turn-speed-margin must be > 0 and <= 1")
    if args.pure_pursuit_cross_track_warning_m > args.pure_pursuit_max_cross_track_error_m:
        parser.error(
            "--pure-pursuit-cross-track-warning-m must be <= "
            "--pure-pursuit-max-cross-track-error-m"
        )
    if not (
        args.pure_pursuit_heading_deadband_deg
        < args.pure_pursuit_curvature_limit_start_heading_error_deg
        < args.pure_pursuit_curvature_limit_full_heading_error_deg
    ):
        parser.error(
            "--pure-pursuit-heading-deadband-deg must be < "
            "--pure-pursuit-curvature-limit-start-heading-error-deg must be < "
            "--pure-pursuit-curvature-limit-full-heading-error-deg"
        )
    if (
        args.pure_pursuit_curvature_limit_full_heading_error_deg
        >= args.pure_pursuit_rotate_start_heading_error_deg
    ):
        parser.error(
            "--pure-pursuit-curvature-limit-full-heading-error-deg must be < "
            "--pure-pursuit-rotate-start-heading-error-deg"
        )
    if (
        args.pure_pursuit_rotate_stop_heading_error_deg
        >= args.pure_pursuit_rotate_start_heading_error_deg
    ):
        parser.error(
            "--pure-pursuit-rotate-stop-heading-error-deg must be < "
            "--pure-pursuit-rotate-start-heading-error-deg"
        )
    if (
        args.pure_pursuit_route_heading_rotate_stop_deg
        >= args.pure_pursuit_route_heading_rotate_start_deg
    ):
        parser.error(
            "--pure-pursuit-route-heading-rotate-stop-deg must be < "
            "--pure-pursuit-route-heading-rotate-start-deg"
        )
    if args.pure_pursuit_min_smoothed_linear_speed_mps > args.linear_speed:
        parser.error(
            "--pure-pursuit-min-smoothed-linear-speed-mps must be <= "
            "--linear-speed"
        )
    if args.pure_pursuit_min_curvature_linear_speed_mps > args.linear_speed:
        parser.error(
            "--pure-pursuit-min-curvature-linear-speed-mps must be <= "
            "--linear-speed"
        )
    if args.pure_pursuit_final_decel_distance_m <= args.goal_tolerance_m:
        parser.error(
            "--pure-pursuit-final-decel-distance-m must be greater than "
            "--goal-tolerance-m"
        )
