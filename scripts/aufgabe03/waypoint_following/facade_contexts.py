from __future__ import annotations


def rviz_message_types(facade):
    return facade["rviz_visualization"].RvizMessageTypes(
        point=facade["Point"],
        pose_stamped=facade["PoseStamped"],
        nav_path=facade["NavPath"],
        marker=facade["Marker"],
        marker_array=facade["MarkerArray"],
        qos_profile=facade["QoSProfile"],
        durability_policy=facade["DurabilityPolicy"],
    )


def rviz_node_context(facade):
    return facade["rviz_visualization"].RvizNodeContext(
        build_rviz_path_message=facade["build_rviz_path_message"],
        build_rviz_waypoint_markers=facade["build_rviz_waypoint_markers"],
        build_rviz_obstacle_markers=facade["build_rviz_obstacle_markers"],
    )


def node_setup_context(facade):
    return facade["node_setup"].NodeSetupContext(
        RuntimeDiagnostics=facade["RuntimeDiagnostics"],
        ReplanManager=facade["ReplanManager"],
        build_command_smoother=facade["build_command_smoother"],
        build_lookahead_guard=facade["build_lookahead_guard"],
        projection_lock_required_samples=facade[
            "PROJECTION_LOCK_REQUIRED_SAMPLES"
        ],
        projection_lock_progress_tolerance_m=facade[
            "PROJECTION_LOCK_PROGRESS_TOLERANCE_M"
        ],
        route_heading_lookahead_m=facade["ROUTE_HEADING_LOOKAHEAD_M"],
        post_rotate_branch_heading_tolerance_deg=facade[
            "POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG"
        ],
        post_rotate_branch_release_stable_samples=facade[
            "POST_ROTATE_BRANCH_RELEASE_STABLE_SAMPLES"
        ],
        post_replan_route_clearance_preview_distance_m=facade[
            "post_replan_route_clearance_preview_distance_m"
        ],
    )


def controller_runtime_context(facade):
    return facade["controller_runtime"].ControllerRuntimeContext(
        TrackingPathValidation=facade["TrackingPathValidation"],
        TwistCommand=facade["TwistCommand"],
        CommandSmoother=facade["CommandSmoother"],
        CommandSmoothingConfig=facade["CommandSmoothingConfig"],
        LookaheadGuard=facade["LookaheadGuard"],
        load_tracking_path_csv=facade["load_tracking_path_csv"],
        validate_tracking_path_geometry=facade[
            "validate_tracking_path_geometry"
        ],
        clamp=facade["clamp"],
        default_controller=facade["DEFAULT_CONTROLLER"],
        default_pure_pursuit_lookahead_guard=facade[
            "DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD"
        ],
        default_pure_pursuit_command_smoothing=facade[
            "DEFAULT_PURE_PURSUIT_COMMAND_SMOOTHING"
        ],
        lookahead_guard_off=facade["LOOKAHEAD_GUARD_OFF"],
        command_smoothing_off=facade["COMMAND_SMOOTHING_OFF"],
        command_smoothing_rate_limit=facade["COMMAND_SMOOTHING_RATE_LIMIT"],
        speed_profile_curvature_aware=facade[
            "SPEED_PROFILE_CURVATURE_AWARE"
        ],
        scheduler_status_deadband=facade["SCHEDULER_STATUS_DEADBAND"],
        projection_lock_required_samples=facade[
            "PROJECTION_LOCK_REQUIRED_SAMPLES"
        ],
        projection_lock_progress_tolerance_m=facade[
            "PROJECTION_LOCK_PROGRESS_TOLERANCE_M"
        ],
        route_heading_lookahead_m=facade["ROUTE_HEADING_LOOKAHEAD_M"],
        rotate_anchor_route_heading_exit_samples=facade[
            "ROTATE_ANCHOR_ROUTE_HEADING_EXIT_SAMPLES"
        ],
        post_rotate_branch_heading_tolerance_deg=facade[
            "POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG"
        ],
        post_rotate_branch_release_stable_samples=facade[
            "POST_ROTATE_BRANCH_RELEASE_STABLE_SAMPLES"
        ],
        post_rotate_branch_min_release_progress_m=facade[
            "POST_ROTATE_BRANCH_MIN_RELEASE_PROGRESS_M"
        ],
        post_rotate_branch_end_lateral_tolerance_m=facade[
            "POST_ROTATE_BRANCH_END_LATERAL_TOLERANCE_M"
        ],
        post_rotate_zero_linear_eps_mps=facade[
            "POST_ROTATE_ZERO_LINEAR_EPS_MPS"
        ],
    )


def ros_runtime_context(facade):
    return facade["ros_runtime"].RosRuntimeContext(
        rclpy=facade["rclpy"],
        Time=facade["Time"],
        Twist=facade["Twist"],
        blocked_error_type=facade["BlockedByScanError"],
        default_odom_frame=facade["DEFAULT_ODOM_FRAME"],
        stop_publish_count=facade["STOP_PUBLISH_COUNT"],
        stop_publish_hz=facade["STOP_PUBLISH_HZ"],
        fresh_scan_stamp_slack_sec=facade["replan_runtime"].FRESH_SCAN_STAMP_SLACK_SEC,
        scan_stamp_sec=facade["replan_runtime"].scan_stamp_sec,
        reset_command_smoother=facade["reset_command_smoother"],
        evaluate_scan_safety=facade["evaluate_scan_safety"],
        quaternion_to_yaw_deg=facade["quaternion_to_yaw_deg"],
        shortest_angle_delta_deg=facade["shortest_angle_delta_deg"],
    )


def ros_node_wiring_context(facade):
    return facade["ros_node_wiring"].RosNodeWiringContext(
        Twist=facade["Twist"],
        NavPath=facade["NavPath"],
        MarkerArray=facade["MarkerArray"],
        LaserScan=facade["LaserScan"],
        PoseWithCovarianceStamped=facade["PoseWithCovarianceStamped"],
        qos_profile_sensor_data=facade["qos_profile_sensor_data"],
        tf2_ros=facade["tf2_ros"],
        time_sleep=facade["time"].sleep,
        rviz_messages_available=facade["rviz_messages_available"],
        rviz_qos_profile=facade["rviz_qos_profile"],
    )


def follow_loop_context(facade):
    return facade["follow_loop"].FollowLoopContext(
        blocked_by_scan_error_type=facade["BlockedByScanError"],
        build_command_smoother=facade["build_command_smoother"],
        build_path_controller=facade["build_path_controller"],
        build_sparse_tracking_validation=facade[
            "build_sparse_tracking_validation"
        ],
        compat_follower_type=facade["WaypointFollower"],
        format_optional_m=facade["format_optional_m"],
        guard_block_signature=facade["guard_block_signature"],
        post_replan_recovery_escape=facade["POST_REPLAN_RECOVERY_ESCAPE"],
        post_replan_recovery_should_preempt_controller=facade[
            "post_replan_recovery_should_preempt_controller"
        ],
        publish_rviz_obstacles_if_available=facade[
            "publish_rviz_obstacles_if_available"
        ],
        publish_rviz_route_if_available=facade[
            "publish_rviz_route_if_available"
        ],
        rclpy=facade["rclpy"],
        replan_trigger_known_corridor=facade["REPLAN_TRIGGER_KNOWN_CORRIDOR"],
        replan_trigger_lookahead_guard=facade[
            "REPLAN_TRIGGER_LOOKAHEAD_GUARD"
        ],
        replan_trigger_scan_blockage=facade["REPLAN_TRIGGER_SCAN_BLOCKAGE"],
        reset_command_smoother=facade["reset_command_smoother"],
        reset_route_projection_controller=facade[
            "reset_route_projection_controller"
        ],
        route_state_type=facade["RouteState"],
        smoothed_step_command=facade["smoothed_step_command"],
        waypoint_timeout_error_type=facade["WaypointTimeoutError"],
    )


def run_session_context(facade):
    return facade["run_session"].RunSessionContext(
        parse_args=facade["parse_args"],
        load_waypoints=facade["load_waypoints"],
        prepare_executable_waypoints=facade["prepare_executable_waypoints"],
        prepare_tracking_setup=facade["prepare_tracking_setup"],
        build_lookahead_guard=facade["build_lookahead_guard"],
        print_dry_run=facade["print_dry_run"],
        require_motion_confirmation=facade["require_motion_confirmation"],
        wait_before_follow_confirmation=facade[
            "wait_before_follow_confirmation"
        ],
        select_executable_waypoints=facade["select_executable_waypoints"],
        WaypointFollower=facade["WaypointFollower"],
        BlockedByScanError=facade["BlockedByScanError"],
        WaypointTimeoutError=facade["WaypointTimeoutError"],
        notes_with_tracking_metadata=facade["notes_with_tracking_metadata"],
        notes_with_velocity_scheduler_metadata=facade[
            "notes_with_velocity_scheduler_metadata"
        ],
        notes_with_smoothing_metadata=facade["notes_with_smoothing_metadata"],
        notes_with_route_projection_metadata=facade[
            "notes_with_route_projection_metadata"
        ],
        notes_with_guard_metadata=facade["notes_with_guard_metadata"],
        notes_with_post_replan_recovery_metadata=facade[
            "notes_with_post_replan_recovery_metadata"
        ],
        build_log_row=facade["build_log_row"],
        append_csv_row=facade["append_csv_row"],
        CSV_HEADER=facade["CSV_HEADER"],
        rclpy=facade["rclpy"],
        sys_argv=facade["sys"].argv,
        stderr=facade["sys"].stderr,
        time_now_sec=facade["time"].time,
    )
