from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from .models import AmclHealth, Pose2D


@dataclass(frozen=True)
class RosRuntimeContext:
    rclpy: Any
    Time: Any
    Twist: Any
    blocked_error_type: type[Exception]
    default_odom_frame: str
    stop_publish_count: int
    stop_publish_hz: float
    fresh_scan_stamp_slack_sec: float
    scan_stamp_sec: Callable[[Any], float | None]
    reset_command_smoother: Callable[[Any], None]
    evaluate_scan_safety: Callable[..., Any]
    quaternion_to_yaw_deg: Callable[[float, float, float, float], float]
    shortest_angle_delta_deg: Callable[[float, float], float]


class RecoverableHealthError(RuntimeError):
    def __init__(self, reason, timeout_sec, message):
        super().__init__(message)
        self.reason = reason
        self.timeout_sec = timeout_sec


def stamp_to_sec(stamp):
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def amcl_covariances(covariance):
    return float(covariance[0]), float(covariance[7]), float(covariance[35])


def evaluate_amcl_health(
    covariance,
    age_sec,
    max_age_sec,
    max_var_x,
    max_var_y,
    max_var_yaw,
    fail_on_bad_localization=False,
):
    cov_x, cov_y, cov_yaw = amcl_covariances(covariance)
    warnings = []
    if age_sec is None or age_sec > max_age_sec:
        warnings.append("stale_amcl")
    if cov_x > max_var_x:
        warnings.append("high_cov_x")
    if cov_y > max_var_y:
        warnings.append("high_cov_y")
    if cov_yaw > max_var_yaw:
        warnings.append("high_cov_yaw")
    return AmclHealth(
        ok=not warnings or not fail_on_bad_localization,
        warnings=warnings,
        cov_x=cov_x,
        cov_y=cov_y,
        cov_yaw=cov_yaw,
        age_sec=age_sec,
    )


def age_ok(age_sec, max_age_sec):
    return age_sec is not None and age_sec <= max_age_sec


def ordered_base_frames(base_frame, fallback_base_frame):
    frames = []
    for frame in [base_frame, fallback_base_frame]:
        if frame and frame not in frames:
            frames.append(frame)
    return frames


def transform_to_pose2d(transform, frame_id, context):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    stamp_sec = stamp_to_sec(transform.header.stamp)
    return Pose2D(
        x=float(translation.x),
        y=float(translation.y),
        yaw_deg=context.quaternion_to_yaw_deg(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ),
        stamp_sec=stamp_sec,
        frame_id=frame_id,
    )


def compose_2d_pose(parent_from_mid, mid_from_child, child_frame_id, context):
    parent_pose = transform_to_pose2d(
        parent_from_mid,
        parent_from_mid.header.frame_id,
        context,
    )
    child_pose = transform_to_pose2d(mid_from_child, child_frame_id, context)
    yaw_rad = math.radians(parent_pose.yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    x = parent_pose.x + cos_yaw * child_pose.x - sin_yaw * child_pose.y
    y = parent_pose.y + sin_yaw * child_pose.x + cos_yaw * child_pose.y
    yaw_deg = context.shortest_angle_delta_deg(
        0.0,
        parent_pose.yaw_deg + child_pose.yaw_deg,
    )
    return Pose2D(
        x=x,
        y=y,
        yaw_deg=yaw_deg,
        stamp_sec=child_pose.stamp_sec,
        frame_id=child_frame_id,
    )


def scan_callback(node, context, msg):
    node.last_scan = msg
    node.last_scan_received_sec = time.time()


def amcl_callback(node, context, msg):
    node.last_amcl = msg
    node.last_amcl_received_sec = time.time()


def odom_callback(node, context, msg):
    node.last_odom = msg
    node.last_odom_received_sec = time.time()
    header = getattr(msg, "header", None)
    pose = msg.pose.pose
    orientation = pose.orientation
    frame_id = getattr(header, "frame_id", "")
    child_frame_id = getattr(msg, "child_frame_id", "")
    node.last_odom_pose = Pose2D(
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw_deg=context.quaternion_to_yaw_deg(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ),
        stamp_sec=stamp_to_sec(getattr(header, "stamp", None)),
        frame_id=child_frame_id or frame_id,
    )
    node.last_odom_frame_id = frame_id
    node.last_odom_child_frame_id = child_frame_id


def fresh_direct_odom_pose(node, now_sec=None):
    pose = getattr(node, "last_odom_pose", None)
    received_sec = getattr(node, "last_odom_received_sec", None)
    if pose is None or received_sec is None:
        return None, None, "direct_odom_start_unavailable"
    if now_sec is None:
        now_sec = time.time()
    age_sec = max(0.0, now_sec - received_sec)
    if age_sec > float(node.args.max_odom_age_sec):
        return None, age_sec, "direct_odom_stale"
    return pose, age_sec, "none"


def publish_velocity(node, context, linear_x, angular_z):
    if linear_x != 0.0 or angular_z != 0.0:
        node.last_scan_block_budget_repair_signature = None
        node.last_lookahead_guard_block_signature = None
    else:
        context.reset_command_smoother(node)
    msg = context.Twist()
    msg.linear.x = linear_x
    msg.angular.z = angular_z
    node.pub.publish(msg)


def stop_repeatedly(node, context):
    context.reset_command_smoother(node)
    msg = context.Twist()
    sleep_sec = 1.0 / context.stop_publish_hz
    for _ in range(context.stop_publish_count):
        if context.rclpy.ok():
            node.pub.publish(msg)
        node.spin_for(sleep_sec)


def spin_once(node, context, timeout_sec):
    context.rclpy.spin_once(node, timeout_sec=timeout_sec)


def spin_for(node, context, duration_sec, step_sec=0.05):
    deadline = time.time() + max(0.0, duration_sec)
    while context.rclpy.ok() and time.time() < deadline:
        timeout_sec = min(step_sec, max(0.0, deadline - time.time()))
        context.rclpy.spin_once(node, timeout_sec=timeout_sec)


def wait_one_control_cycle(node, context):
    period_sec = 1.0 / node.args.control_rate_hz
    context.rclpy.spin_once(node, timeout_sec=period_sec)
    time.sleep(period_sec)


def wait_for_startup_gate(node, context, timeout_sec=None):
    if timeout_sec is None:
        timeout_sec = node.args.startup_timeout_sec
    require_amcl = (
        node.args.require_amcl_startup
        or node.args.fail_on_bad_localization
        or node.args.pause_on_bad_localization
    )
    start = time.time()
    while context.rclpy.ok():
        have_scan = node.last_scan is not None
        have_amcl = node.last_amcl is not None
        have_tf = False
        try:
            _pose, frame = node.lookup_pose()
            node.base_frame_used = frame
            have_tf = True
        except RuntimeError:
            have_tf = False

        if have_scan and have_tf and (have_amcl or not require_amcl):
            return
        if time.time() - start > timeout_sec:
            missing = []
            if not have_scan:
                missing.append("/scan")
            if require_amcl and not have_amcl:
                missing.append("/amcl_pose")
            if not have_tf:
                missing.append(
                    f"TF {node.args.map_frame}->{node.args.base_frame}/"
                    f"{node.args.fallback_base_frame}"
                )
            raise RuntimeError(
                "Timed out waiting for startup data: " + ", ".join(missing)
            )
        context.rclpy.spin_once(node, timeout_sec=0.1)
    raise RuntimeError("ROS shutdown during startup gate.")


def lookup_pose(node, context):
    errors = []
    for frame in ordered_base_frames(node.args.base_frame, node.args.fallback_base_frame):
        try:
            transform = node.tf_buffer.lookup_transform(
                node.args.map_frame,
                frame,
                context.Time(),
            )
            node.base_frame_used = frame
            return transform_to_pose2d(transform, frame, context), frame
        except Exception as exc:
            errors.append(f"{frame}: {exc}")
    odom_frame = getattr(node.args, "odom_frame", context.default_odom_frame)
    for frame in ordered_base_frames(node.args.base_frame, node.args.fallback_base_frame):
        try:
            map_from_odom = node.tf_buffer.lookup_transform(
                node.args.map_frame,
                odom_frame,
                context.Time(),
            )
            odom_from_base = node.tf_buffer.lookup_transform(
                odom_frame,
                frame,
                context.Time(),
            )
            node.base_frame_used = frame
            return compose_2d_pose(
                map_from_odom,
                odom_from_base,
                frame,
                context,
            ), frame
        except Exception as exc:
            errors.append(
                f"split {node.args.map_frame}->{odom_frame}->{frame}: {exc}"
            )
    raise RuntimeError("Could not lookup TF pose: " + "; ".join(errors))


def lookup_odom_pose(node, context):
    errors = []
    odom_frame = getattr(node.args, "odom_frame", context.default_odom_frame)
    lookup_time = context.Time() if callable(context.Time) else None
    for frame in ordered_base_frames(node.args.base_frame, node.args.fallback_base_frame):
        try:
            transform = node.tf_buffer.lookup_transform(
                odom_frame,
                frame,
                lookup_time,
            )
            return transform_to_pose2d(transform, frame, context)
        except Exception as exc:
            errors.append(f"{odom_frame}->{frame}: {exc}")
    raise RuntimeError("Could not lookup odom TF pose: " + "; ".join(errors))


def try_lookup_odom_pose(node, context):
    try:
        return node.lookup_odom_pose()
    except Exception:
        return None


def update_tf_tracking(node, context, pose):
    if pose.stamp_sec is None:
        return None
    now = time.time()
    if node.last_tf_stamp_sec is None or pose.stamp_sec != node.last_tf_stamp_sec:
        node.last_tf_stamp_sec = pose.stamp_sec
        node.last_tf_stamp_change_local_sec = now
    if node.last_tf_stamp_change_local_sec is None:
        node.last_tf_stamp_change_local_sec = now
    return now - node.last_tf_stamp_change_local_sec


def reset_tf_tracking(node, context):
    node.last_tf_stamp_sec = None
    node.last_tf_stamp_change_local_sec = None


def refresh_after_operator_wait(node, context, min_scan_stamp_sec, timeout_sec=None):
    node.reset_tf_tracking()
    timeout_sec = timeout_sec or node.args.startup_timeout_sec
    deadline = time.time() + timeout_sec
    while context.rclpy.ok() and time.time() <= deadline:
        context.rclpy.spin_once(node, timeout_sec=0.1)
        scan_stamp_sec = (
            None
            if node.last_scan is None
            else context.scan_stamp_sec(node.last_scan)
        )
        if (
            node.last_scan is not None
            and node.last_scan_received_sec is not None
            and node.last_scan_received_sec >= min_scan_stamp_sec
            and (
                scan_stamp_sec is None
                or scan_stamp_sec
                >= min_scan_stamp_sec - context.fresh_scan_stamp_slack_sec
            )
        ):
            return
    raise RuntimeError(
        "Timed out waiting for fresh stamped /scan after handoff pause."
    )


def current_amcl_health(node, context):
    if node.last_amcl is None:
        return AmclHealth(
            ok=not node.args.fail_on_bad_localization,
            warnings=["missing_amcl"],
            cov_x=None,
            cov_y=None,
            cov_yaw=None,
            age_sec=None,
        )
    age_sec = (
        None if node.last_amcl_received_sec is None
        else time.time() - node.last_amcl_received_sec
    )
    covariance = node.last_amcl.pose.covariance
    return evaluate_amcl_health(
        covariance,
        age_sec,
        node.args.max_amcl_age_sec,
        node.args.max_amcl_var_x,
        node.args.max_amcl_var_y,
        node.args.max_amcl_var_yaw,
        fail_on_bad_localization=node.args.fail_on_bad_localization,
    )


def check_health_or_raise(node, context):
    try:
        pose, frame = node.lookup_pose()
    except RuntimeError as exc:
        raise RecoverableHealthError(
            "tf_lookup",
            node.args.tf_recovery_time_sec,
            str(exc),
        ) from exc
    if pose.stamp_sec is None:
        raise RuntimeError("TF pose has no usable timestamp.")
    pose_age = time.time() - pose.stamp_sec
    node.diagnostics.tf_pose_age_sec = pose_age
    if pose_age > node.args.max_pose_age_sec:
        node.diagnostics.tf_stale_warning_count += 1
        message = f"TF pose is stale: age={pose_age:.3f} sec"
        if node.args.fail_on_stale_tf:
            raise RuntimeError(message)
        node.get_logger().warn(message)

    tf_update_gap_sec = node.update_tf_tracking(pose)
    if (
        tf_update_gap_sec is not None
        and tf_update_gap_sec > node.args.max_tf_update_gap_sec
    ):
        raise RecoverableHealthError(
            "tf_update_gap",
            node.args.tf_recovery_time_sec,
            "TF transform stamp stopped updating: "
            f"gap={tf_update_gap_sec:.3f} sec, "
            f"limit={node.args.max_tf_update_gap_sec:.3f} sec",
        )

    scan_age = (
        None if node.last_scan_received_sec is None
        else time.time() - node.last_scan_received_sec
    )
    if not age_ok(scan_age, node.args.max_scan_age_sec):
        raise RecoverableHealthError(
            "scan_stale",
            node.args.max_scan_age_sec,
            f"/scan is stale: age={scan_age}",
        )

    amcl_health = node.current_amcl_health()
    if amcl_health.warnings:
        node.diagnostics.localization_warning_count += 1
        message = "AMCL localization warning(s): " + ",".join(amcl_health.warnings)
        if not amcl_health.ok:
            raise RuntimeError(message)
        if node.args.pause_on_bad_localization:
            raise RecoverableHealthError(
                "bad_localization",
                node.args.localization_recovery_time_sec,
                message,
            )
        node.get_logger().warn(message)

    return pose, frame, amcl_health


def check_health_or_recover(node, context):
    while True:
        try:
            return node.check_health_or_raise()
        except RecoverableHealthError as exc:
            node.diagnostics.recovery_pause_count += 1
            node.stop_repeatedly()
            node.get_logger().warn(
                f"{exc}; pausing for up to {exc.timeout_sec:.1f} sec"
            )
            deadline = time.time() + exc.timeout_sec
            last_message = str(exc)
            while time.time() < deadline and context.rclpy.ok():
                context.rclpy.spin_once(node, timeout_sec=0.1)
                time.sleep(0.1)
                try:
                    return node.check_health_or_raise()
                except RecoverableHealthError as retry_exc:
                    last_message = str(retry_exc)
            raise RuntimeError(
                f"{exc.reason} did not recover within "
                f"{exc.timeout_sec:.1f} sec: {last_message}"
            )


def check_scan_or_raise(node, context, mode):
    if node.last_scan is None:
        raise RuntimeError("No /scan sample is available.")
    safety = node.evaluate_current_scan_safety(mode)
    if not safety.safe:
        raise context.blocked_error_type(safety)
    return safety


def evaluate_current_scan_safety(node, context, mode):
    if node.last_scan is None:
        raise RuntimeError("No /scan sample is available.")
    return context.evaluate_scan_safety(
        node.last_scan.ranges,
        node.last_scan.angle_min,
        node.last_scan.angle_increment,
        node.last_scan.range_min,
        node.last_scan.range_max,
        mode,
        node.args.scan_half_angle_deg,
        node.args.hard_stop_range_m,
        node.args.min_scan_range_m,
        node.args.rotation_stop_range_m,
    )
