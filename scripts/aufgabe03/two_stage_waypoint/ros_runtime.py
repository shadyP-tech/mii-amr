import math
import time

try:
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
    from nav_msgs.msg import Odometry
    from nav2_msgs.action import FollowPath, NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    import tf2_ros
except ImportError:
    rclpy = None
    FollowPath = None
    NavigateToPose = None
    ActionClient = None
    Node = object
    qos_profile_sensor_data = None
    Time = None
    LaserScan = object
    Odometry = object
    tf2_ros = None

    class _FallbackStamp:
        sec = 0
        nanosec = 0

    class _FallbackHeader:
        def __init__(self):
            self.frame_id = ""
            self.stamp = _FallbackStamp()

    class _FallbackPosition:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class _FallbackOrientation:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.w = 1.0

    class _FallbackPose:
        def __init__(self):
            self.position = _FallbackPosition()
            self.orientation = _FallbackOrientation()

    class _FallbackPoseWithCovariance:
        def __init__(self):
            self.pose = _FallbackPose()
            self.covariance = [0.0] * 36

    class PoseWithCovarianceStamped:
        def __init__(self):
            self.header = _FallbackHeader()
            self.pose = _FallbackPoseWithCovariance()

    class Twist:
        def __init__(self):
            self.linear = _FallbackPosition()
            self.angular = _FallbackPosition()

from arena_active_spin import (
    ArenaActiveSpinConfig,
    run_arena_active_spin,
    write_diagnostics_json,
)
from arena_geometry_localizer import ArenaGeometryConfig

from .model import (
    STOP_PUBLISH_COUNT,
    STOP_PUBLISH_HZ,
    ArrivalCheck,
    Pose2D,
    ScanSafety,
    StabilityState,
)
from .pure import (
    amcl_stability_satisfied,
    amcl_validation_timed_out,
    arena_active_diagnostics_path,
    distance_pose_to_waypoint_path_m,
    evaluate_spin_scan_safety,
    goal_status_name,
    quaternion_to_yaw_deg,
    required_preflight_interfaces,
    shortest_angle_delta_deg,
    update_amcl_stability,
    validate_pose_prior_for_initialpose,
    yaw_to_quaternion_values,
)


def arena_active_config_from_args(args):
    arena_config = ArenaGeometryConfig(
        arena_length_m=args.arena_length_m,
        arena_width_m=args.arena_width_m,
        heater_side_width_m=args.arena_heater_wall_width_m,
        clean_side_width_m=args.arena_clean_wall_width_m,
        width_match_min_margin_m=args.arena_width_match_min_margin_m,
        max_short_wall_range_sum_error_m=args.arena_max_short_wall_range_sum_error_m,
        map_center_x=args.arena_map_center_x,
        map_center_y=args.arena_map_center_y,
        map_yaw_deg=args.arena_map_yaw_deg,
        heater_wall_side=args.heater_wall_side,
        min_wall_points=args.arena_min_wall_points,
        max_wall_separation_error_m=args.arena_max_wall_separation_error_m,
        max_line_rmse_m=args.arena_max_line_rmse_m,
        min_parallel_score=args.arena_min_parallel_score,
        min_short_wall_confidence=args.arena_min_short_wall_confidence,
        min_classification_margin=args.arena_min_classification_margin,
        forced_short_wall_side=args.arena_force_short_wall_side,
        forced_short_wall_type=args.arena_force_short_wall_type,
    )
    return ArenaActiveSpinConfig(
        run_id=args.run_id,
        diagnostics_path=arena_active_diagnostics_path(args),
        cmd_vel_topic=args.cmd_vel_topic,
        scan_topic=args.scan_topic,
        odom_topic=args.odom_topic,
        spin_direction=args.arena_active_spin_direction,
        angular_speed_rad_s=args.arena_active_angular_speed_rad_s,
        max_spin_sec=args.arena_active_max_spin_sec,
        spin_complete_tolerance_deg=args.arena_active_spin_complete_tolerance_deg,
        min_angular_progress_rad_s=args.arena_active_min_angular_progress_rad_s,
        progress_check_sec=args.arena_active_progress_check_sec,
        min_scan_samples=args.arena_active_min_scan_samples,
        max_odom_scan_age_sec=args.arena_active_max_odom_scan_age_sec,
        stop_settle_sec=args.arena_active_stop_settle_sec,
        min_front_clearance_m=args.arena_active_min_front_clearance_m,
        min_side_clearance_m=args.arena_active_min_side_clearance_m,
        min_rear_clearance_m=args.arena_active_min_rear_clearance_m,
        require_operator_confirmation=args.arena_active_require_operator_confirmation,
        allow_extra_cmd_vel_publishers=args.arena_active_allow_extra_cmd_vel_publishers,
        on_failure="abort",
        dry_run=False,
        range_stride=args.arena_active_range_stride,
        max_points=args.arena_active_max_points,
        control_rate_hz=args.control_rate_hz,
        recovery_mode=args.arena_active_recovery_mode,
        recovery_executor=args.arena_active_recovery_executor,
        enable_center_reposition=args.arena_active_enable_center_reposition,
        center_reposition_max_attempts=args.arena_active_center_reposition_max_attempts,
        center_reposition_target_nearest_short_wall_range_m=(
            args.arena_active_center_reposition_target_nearest_short_wall_range_m
        ),
        center_reposition_min_step_m=args.arena_active_center_reposition_min_step_m,
        center_reposition_max_step_m=args.arena_active_center_reposition_max_step_m,
        center_reposition_linear_speed_mps=(
            args.arena_active_center_reposition_linear_speed_mps
        ),
        center_reposition_angular_speed_rad_s=(
            args.arena_active_center_reposition_angular_speed_rad_s
        ),
        center_reposition_heading_tolerance_deg=(
            args.arena_active_center_reposition_heading_tolerance_deg
        ),
        center_reposition_min_front_clearance_m=(
            args.arena_active_center_reposition_min_front_clearance_m
        ),
        center_reposition_lateral_offset_threshold_m=(
            args.arena_active_center_reposition_lateral_offset_threshold_m
        ),
        center_reposition_lateral_target_offset_m=(
            args.arena_active_center_reposition_lateral_target_offset_m
        ),
        center_reposition_lateral_min_step_m=(
            args.arena_active_center_reposition_lateral_min_step_m
        ),
        center_reposition_lateral_max_step_m=(
            args.arena_active_center_reposition_lateral_max_step_m
        ),
        center_reposition_enable_heater_approach=(
            args.arena_active_center_reposition_enable_heater_approach
        ),
        center_reposition_heater_approach_max_attempts=(
            args.arena_active_center_reposition_heater_approach_max_attempts
        ),
        center_reposition_heater_approach_target_range_m=(
            args.arena_active_center_reposition_heater_approach_target_range_m
        ),
        center_reposition_heater_approach_min_selected_score=(
            args.arena_active_center_reposition_heater_approach_min_selected_score
        ),
        center_reposition_heater_approach_max_opposite_score=(
            args.arena_active_center_reposition_heater_approach_max_opposite_score
        ),
        center_reposition_heater_approach_min_delta=(
            args.arena_active_center_reposition_heater_approach_min_delta
        ),
        center_reposition_heater_approach_min_step_m=(
            args.arena_active_center_reposition_heater_approach_min_step_m
        ),
        center_reposition_heater_approach_max_step_m=(
            args.arena_active_center_reposition_heater_approach_max_step_m
        ),
        active_explore_max_attempts=args.arena_active_explore_max_attempts,
        active_explore_max_single_move_m=args.arena_active_explore_max_single_move_m,
        active_explore_max_total_distance_m=(
            args.arena_active_explore_max_total_distance_m
        ),
        active_explore_grid_resolution_m=(
            args.arena_active_explore_grid_resolution_m
        ),
        active_explore_grid_size_m=args.arena_active_explore_grid_size_m,
        active_explore_inflation_radius_m=(
            args.arena_active_explore_inflation_radius_m
        ),
        active_explore_unknown_blocked=args.arena_active_explore_unknown_blocked,
        active_explore_max_path_segments=args.arena_active_explore_max_path_segments,
        active_explore_side_bias=args.arena_active_explore_side_bias,
        arena_config=arena_config,
    )


def build_initial_pose_message(
    x,
    y,
    yaw_deg,
    var_x,
    var_y,
    var_yaw_rad2,
    frame_id="map",
    stamp=None,
):
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = 0.0
    qx, qy, qz, qw = yaw_to_quaternion_values(yaw_deg)
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    covariance = [0.0] * 36
    covariance[0] = float(var_x)
    covariance[7] = float(var_y)
    covariance[35] = float(var_yaw_rad2)
    msg.pose.covariance = covariance
    return msg


def pose2d_from_pose_msg(pose_msg, stamp_sec=None, frame_id=""):
    position = pose_msg.position
    orientation = pose_msg.orientation
    return Pose2D(
        x=float(position.x),
        y=float(position.y),
        yaw_deg=quaternion_to_yaw_deg(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ),
        stamp_sec=stamp_sec,
        frame_id=frame_id,
    )


def stamp_to_sec(stamp):
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def latest_tf_time():
    if Time is None:
        return None
    return Time()


def transform_to_pose2d(transform, frame_id):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    stamp_sec = stamp_to_sec(transform.header.stamp)
    return Pose2D(
        x=float(translation.x),
        y=float(translation.y),
        yaw_deg=quaternion_to_yaw_deg(rotation.x, rotation.y, rotation.z, rotation.w),
        stamp_sec=stamp_sec,
        frame_id=frame_id,
    )


class TwoStageCoordinator(Node):
    def __init__(self, args):
        if rclpy is None:
            raise RuntimeError("ROS 2 Python modules are unavailable. Source ROS 2 Humble first.")
        super().__init__("two_stage_waypoint_run")
        self.args = args
        self.last_scan = None
        self.last_scan_received_sec = None
        self.last_amcl = None
        self.last_amcl_received_sec = None
        self.active_goal_handle = None
        self.selected_base_frame = ""

        self.cmd_vel_pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            args.initial_pose_topic,
            10,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            args.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            args.amcl_topic,
            self.amcl_callback,
            10,
        )
        self.navigate_client = ActionClient(self, NavigateToPose, args.navigate_action)
        self.follow_path_client = None
        if (
            args.arena_active_recovery_mode == "active_explore"
            and args.arena_active_recovery_executor == "nav2_follow_path"
        ):
            if FollowPath is None:
                raise RuntimeError(
                    "nav2_msgs.action.FollowPath is unavailable; "
                    "source a Nav2 Humble environment first"
                )
            self.follow_path_client = ActionClient(self, FollowPath, args.follow_path_action)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def scan_callback(self, msg):
        self.last_scan = msg
        self.last_scan_received_sec = time.time()

    def amcl_callback(self, msg):
        self.last_amcl = msg
        self.last_amcl_received_sec = time.time()

    def publish_stop(self):
        self.cmd_vel_pub.publish(Twist())

    def stop_repeatedly(self):
        delay = 1.0 / STOP_PUBLISH_HZ
        for _ in range(STOP_PUBLISH_COUNT):
            if rclpy.ok():
                self.publish_stop()
            time.sleep(delay)

    def wait_for_future(self, future, timeout_sec, description):
        deadline = time.time() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.time() > deadline:
                raise RuntimeError(f"Timed out waiting for {description}")
            rclpy.spin_once(self, timeout_sec=0.1)
        if not future.done():
            raise RuntimeError(f"ROS shutdown while waiting for {description}")
        if future.exception() is not None:
            raise RuntimeError(f"{description} failed: {future.exception()}")
        return future.result()

    def wait_for_fresh_scan(self, timeout_sec):
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() <= deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.last_scan is None or self.last_scan_received_sec is None:
                continue
            if time.time() - self.last_scan_received_sec <= self.args.max_scan_age_sec:
                return self.last_scan
        raise RuntimeError(f"Timed out waiting for fresh {self.args.scan_topic}")

    def current_scan_safety(self):
        if self.last_scan is None:
            return ScanSafety(False, "missing_scan", 0, None)
        return evaluate_spin_scan_safety(
            self.last_scan.ranges,
            range_min=self.last_scan.range_min,
            range_max=self.last_scan.range_max,
            min_scan_range_m=self.args.spin_min_scan_range_m,
            min_valid_scan_count=self.args.spin_min_valid_scan_count,
        )

    def preflight_before_motion(self):
        requirements = required_preflight_interfaces(self.args)
        for action in requirements.actions:
            client = self.navigate_client
            if action == self.args.follow_path_action:
                client = self.follow_path_client
            if client is None or not client.wait_for_server(
                timeout_sec=self.args.preflight_timeout_sec,
            ):
                raise RuntimeError(f"Required action is unavailable: {action}")
        self.wait_for_fresh_scan(self.args.preflight_timeout_sec)
        safety = self.current_scan_safety()
        if not safety.ok:
            raise RuntimeError(f"Preflight scan safety failed: {safety.reason}")
        self.wait_for_initial_pose_subscriber(self.args.preflight_timeout_sec)

    def wait_for_initial_pose_subscriber(self, timeout_sec=None):
        if timeout_sec is None:
            timeout_sec = self.args.preflight_timeout_sec
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() <= deadline:
            if self.initial_pose_pub.get_subscription_count() > 0:
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError(
            "AMCL/localization is not listening on "
            f"{self.args.initial_pose_topic}"
        )

    def perform_arena_active_spin(self):
        return run_arena_active_spin(
            self,
            self.cmd_vel_pub,
            arena_active_config_from_args(self.args),
            rclpy,
            Twist,
            LaserScan,
            Odometry,
            qos_profile_sensor_data,
        )

    def publish_arena_active_initial_pose(self, pose_prior, arena_result):
        var_x, var_y, var_yaw = validate_pose_prior_for_initialpose(pose_prior)
        self.wait_for_initial_pose_subscriber(
            min(2.0, self.args.preflight_timeout_sec)
        )
        msg = build_initial_pose_message(
            pose_prior.x_m,
            pose_prior.y_m,
            math.degrees(pose_prior.yaw_rad),
            var_x,
            var_y,
            var_yaw,
            frame_id=self.args.map_frame,
        )
        for _ in range(3):
            self.initial_pose_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
        arena_result.diagnostics["initialpose"] = {
            "published": True,
            "x_m": pose_prior.x_m,
            "y_m": pose_prior.y_m,
            "yaw_rad": pose_prior.yaw_rad,
            "covariance": [float(value) for value in msg.pose.covariance],
        }
        write_diagnostics_json(arena_result.diagnostics_path, arena_result.diagnostics)
        return msg

    def amcl_pose2d(self, msg):
        stamp_sec = stamp_to_sec(msg.header.stamp)
        return pose2d_from_pose_msg(msg.pose.pose, stamp_sec=stamp_sec, frame_id=msg.header.frame_id)

    def wait_for_amcl_validation(self, timeout_sec, min_received_sec=None, min_settle_sec=0.0):
        start = time.time()
        state = StabilityState()
        processed_received_sec = None
        last_reason = state.reason
        while rclpy.ok():
            now = time.time()
            if amcl_validation_timed_out(start, now, timeout_sec):
                raise RuntimeError(
                    "Timed out waiting for AMCL validation: "
                    f"reason={last_reason}, stable_samples={state.stable_count}, "
                    f"quiet_duration_sec={state.quiet_duration_sec:.2f}, "
                    f"max_pose_jump_m={state.max_pose_jump_m:.4f}, "
                    f"max_yaw_jump_deg={state.max_yaw_jump_deg:.2f}"
                )
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.last_amcl is None or self.last_amcl_received_sec is None:
                continue
            if min_received_sec is not None and self.last_amcl_received_sec < min_received_sec:
                last_reason = "waiting_for_fresh_amcl"
                continue
            age_sec = now - self.last_amcl_received_sec
            if age_sec > self.args.max_amcl_age_sec:
                last_reason = "stale_amcl"
                continue
            if processed_received_sec != self.last_amcl_received_sec:
                processed_received_sec = self.last_amcl_received_sec
                pose = self.amcl_pose2d(self.last_amcl)
                state = update_amcl_stability(
                    state,
                    pose,
                    self.last_amcl.pose.covariance,
                    self.args.max_amcl_var_x,
                    self.args.max_amcl_var_y,
                    self.args.max_amcl_var_yaw_rad2,
                    self.args.max_stable_pose_jump_m,
                    self.args.max_stable_yaw_jump_deg,
                    sample_sec=self.last_amcl_received_sec,
                )
                last_reason = state.reason
            if amcl_stability_satisfied(
                state,
                self.args.stable_amcl_samples,
                min_settle_sec,
                now_sec=now,
            ):
                return state
        raise RuntimeError("ROS shutdown while waiting for AMCL validation")

    def transform_age_sec(self, transform):
        stamp_sec = stamp_to_sec(transform.header.stamp)
        if stamp_sec is None:
            return None
        return time.time() - stamp_sec

    def lookup_robot_pose_tf(
        self,
        target_frame,
        base_frames,
        timeout_sec,
        description="robot pose TF",
    ):
        deadline = time.time() + timeout_sec
        last_error = ""
        while rclpy.ok() and time.time() <= deadline:
            errors = []
            for frame in base_frames:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        target_frame,
                        frame,
                        latest_tf_time(),
                    )
                except Exception as exc:
                    errors.append(f"{frame}: {exc}")
                    continue

                age_sec = self.transform_age_sec(transform)
                if age_sec is not None and age_sec > self.args.max_pose_age_sec:
                    errors.append(
                        f"{frame}: stale_tf age={age_sec:.3f}s "
                        f"limit={self.args.max_pose_age_sec:.3f}s"
                    )
                    continue

                self.selected_base_frame = frame
                return transform, frame

            last_error = "; ".join(errors)
            rclpy.spin_once(self, timeout_sec=self.args.tf_lookup_retry_period_sec)
        if not rclpy.ok():
            raise RuntimeError(f"ROS shutdown while waiting for {description}")
        raise RuntimeError(
            f"Timed out waiting for {description} "
            f"{target_frame}->{'/'.join(base_frames)}: {last_error}"
        )

    def lookup_pose(self, timeout_sec=None, description="robot pose TF"):
        transform, frame = self.lookup_robot_pose_tf(
            target_frame=self.args.map_frame,
            base_frames=[self.args.base_frame, self.args.fallback_base_frame],
            timeout_sec=(
                timeout_sec if timeout_sec is not None else self.args.tf_lookup_timeout_sec
            ),
            description=description,
        )
        return transform_to_pose2d(transform, frame), frame

    def validate_post_localization_tf(self):
        return self.lookup_pose(
            timeout_sec=self.args.tf_ready_timeout_sec,
            description="post-localization TF",
        )

    def validate_post_amcl_pose_prior(self, pose_prior, observed_pose, arena_result, frame):
        prior_yaw_deg = math.degrees(pose_prior.yaw_rad)
        dx = observed_pose.x - pose_prior.x_m
        dy = observed_pose.y - pose_prior.y_m
        position_error_m = math.hypot(dx, dy)
        yaw_error_deg = abs(shortest_angle_delta_deg(prior_yaw_deg, observed_pose.yaw_deg))
        validation = {
            "ok": (
                position_error_m
                <= self.args.arena_active_max_post_amcl_prior_position_error_m
                and yaw_error_deg
                <= self.args.arena_active_max_post_amcl_prior_yaw_error_deg
            ),
            "frame": frame,
            "prior_x_m": pose_prior.x_m,
            "prior_y_m": pose_prior.y_m,
            "prior_yaw_deg": prior_yaw_deg,
            "observed_x_m": observed_pose.x,
            "observed_y_m": observed_pose.y,
            "observed_yaw_deg": observed_pose.yaw_deg,
            "dx_m": dx,
            "dy_m": dy,
            "position_error_m": position_error_m,
            "yaw_error_deg": yaw_error_deg,
            "max_position_error_m": (
                self.args.arena_active_max_post_amcl_prior_position_error_m
            ),
            "max_yaw_error_deg": self.args.arena_active_max_post_amcl_prior_yaw_error_deg,
        }
        arena_result.diagnostics["post_amcl_pose_prior_validation"] = validation
        write_diagnostics_json(arena_result.diagnostics_path, arena_result.diagnostics)
        if not validation["ok"]:
            raise RuntimeError(
                "Post-AMCL pose differs from arena prior: "
                f"position_error_m={position_error_m:.3f} "
                f"limit={validation['max_position_error_m']:.3f}, "
                f"yaw_error_deg={yaw_error_deg:.1f} "
                f"limit={validation['max_yaw_error_deg']:.1f}"
            )
        return validation

    def navigate_to_staging(self, staging_goal):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.args.map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = staging_goal.waypoint.x
        goal_msg.pose.pose.position.y = staging_goal.waypoint.y
        goal_msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion_values(staging_goal.yaw_deg)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        send_future = self.navigate_client.send_goal_async(goal_msg)
        goal_handle = self.wait_for_future(
            send_future,
            self.args.preflight_timeout_sec,
            "NavigateToPose goal acceptance",
        )
        if not goal_handle.accepted:
            raise RuntimeError("NavigateToPose goal was rejected")

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result = self.wait_for_future(
            result_future,
            self.args.nav_to_start_timeout_sec,
            "NavigateToPose result",
        )
        self.active_goal_handle = None
        status = int(result.status)
        status_name = goal_status_name(status)
        if status != 4:
            raise RuntimeError(f"NavigateToPose did not succeed: {status_name}")
        return status_name

    def cancel_active_goal(self):
        if self.active_goal_handle is None:
            return
        future = self.active_goal_handle.cancel_goal_async()
        try:
            self.wait_for_future(future, 2.0, "NavigateToPose cancellation")
        finally:
            self.active_goal_handle = None

    def verify_arrival(self, staging_goal, waypoints=None):
        pose, frame = self.lookup_pose(description="arrival TF")
        position_error = math.hypot(
            pose.x - staging_goal.waypoint.x,
            pose.y - staging_goal.waypoint.y,
        )
        yaw_error = abs(shortest_angle_delta_deg(pose.yaw_deg, staging_goal.yaw_deg))
        strict_position_ok = position_error <= self.args.arrival_tolerance_m
        strict_yaw_ok = yaw_error <= self.args.arrival_yaw_tolerance_deg
        distance_to_path_m = None
        if waypoints is not None:
            distance_to_path_m = distance_pose_to_waypoint_path_m(pose, waypoints)
        handoff_path_ok = (
            distance_to_path_m is not None
            and distance_to_path_m <= self.args.follower_start_on_path_tolerance_m
        )

        if not strict_position_ok and not handoff_path_ok:
            raise RuntimeError(
                "Arrival position check failed: "
                f"error={position_error:.3f} m, "
                f"limit={self.args.arrival_tolerance_m:.3f} m"
            )
        if not strict_yaw_ok and not handoff_path_ok:
            raise RuntimeError(
                "Arrival yaw check failed: "
                f"error={yaw_error:.1f} deg, "
                f"limit={self.args.arrival_yaw_tolerance_deg:.1f} deg"
            )
        if not strict_position_ok or not strict_yaw_ok:
            self.get_logger().warn(
                "Nav2 staging missed the strict arrival gate, but handoff is "
                "allowed because the robot is close to the waypoint path: "
                f"position_error={position_error:.3f} m, "
                f"yaw_error={yaw_error:.1f} deg, "
                f"distance_to_path={distance_to_path_m:.3f} m, "
                f"path_limit={self.args.follower_start_on_path_tolerance_m:.3f} m"
            )
        return ArrivalCheck(
            pose,
            frame,
            position_error,
            yaw_error,
            distance_to_path_m=distance_to_path_m,
            handoff_path_ok=handoff_path_ok,
            strict_position_ok=strict_position_ok,
            strict_yaw_ok=strict_yaw_ok,
        )
