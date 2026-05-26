#!/usr/bin/env python3
"""
ROS-facing orchestration for run-local obstacle replanning.

This module stops motion, reads fresh scan/TF state, converts sensor endpoints
into map-frame observations, and delegates all map/replan work to the ROS-free
lidar_obstacle_map module. It intentionally does not implement waypoint control.
"""

from __future__ import annotations

import math
import time

try:
    from rclpy.time import Time
except ImportError:
    Time = None

import lidar_obstacle_map


FRESH_SCAN_STAMP_SLACK_SEC = 0.25


def stamp_to_sec(stamp):
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def time_from_stamp(stamp):
    if Time is None:
        return stamp
    return Time.from_msg(stamp)


def latest_time():
    if Time is None:
        return None
    return Time()


def yaw_from_quaternion(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_yaw_rad(transform):
    rotation = transform.transform.rotation
    return yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)


def transform_point_2d(transform, x, y):
    translation = transform.transform.translation
    yaw = transform_yaw_rad(transform)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        float(translation.x) + cos_yaw * x - sin_yaw * y,
        float(translation.y) + sin_yaw * x + cos_yaw * y,
    )


def scan_frame_points(scan):
    return lidar_obstacle_map.scan_ranges_to_base_points(
        scan.ranges,
        scan.angle_min,
        scan.angle_increment,
        scan.range_min,
        scan.range_max,
    )


def scan_points_to_map_observations(scan, map_from_scan_transform, base_filter_config=None):
    observations = []
    for scan_point in scan_frame_points(scan):
        if (
            base_filter_config is not None
            and not lidar_obstacle_map.base_point_passes_roi(scan_point, base_filter_config)
        ):
            continue
        map_x, map_y = transform_point_2d(
            map_from_scan_transform,
            scan_point.x,
            scan_point.y,
        )
        observations.append(lidar_obstacle_map.MapFrameObservation(
            map_x,
            map_y,
            stamp_sec=stamp_to_sec(scan.header.stamp),
        ))
    return observations


def scan_points_to_base_frame(scan, map_from_scan_transform, robot_pose):
    base_points = []
    for scan_point in scan_frame_points(scan):
        map_x, map_y = transform_point_2d(
            map_from_scan_transform,
            scan_point.x,
            scan_point.y,
        )
        base_points.append(lidar_obstacle_map.map_point_to_base(
            map_x,
            map_y,
            robot_pose,
        ))
    return base_points


def lookup_map_from_scan_transform(node, args, scan):
    scan_frame = getattr(scan.header, "frame_id", "")
    if not scan_frame:
        raise RuntimeError("scan frame is empty")
    try:
        transform = node.tf_buffer.lookup_transform(
            args.map_frame,
            scan_frame,
            time_from_stamp(scan.header.stamp),
        )
        return transform, "timestamped"
    except Exception as timestamped_exc:
        if not args.allow_latest_tf_replan_fallback:
            raise RuntimeError(
                "timestamped TF lookup failed for LiDAR replan: "
                f"{timestamped_exc}"
            ) from timestamped_exc
        try:
            transform = node.tf_buffer.lookup_transform(
                args.map_frame,
                scan_frame,
                latest_time(),
            )
        except Exception as latest_exc:
            raise RuntimeError(
                "latest TF fallback failed for LiDAR replan: "
                f"timestamped={timestamped_exc}; latest={latest_exc}"
            ) from latest_exc
        return transform, "latest_fallback"


def replan_config_from_args(args):
    return lidar_obstacle_map.ObstacleOverlayConfig(
        forward_distance_m=args.obstacle_forward_distance_m,
        forward_half_width_m=args.obstacle_forward_half_width_m,
        angle_window_deg=args.obstacle_angle_window_deg,
        min_range_m=args.obstacle_min_range_m,
        robot_footprint_radius_m=args.robot_footprint_radius_m,
        min_cluster_size=args.obstacle_min_cluster_size,
        min_cluster_width_m=args.obstacle_min_cluster_width_m,
        inflate_radius_m=args.obstacle_inflate_radius_m,
        max_start_snap_m=args.max_start_snap_m,
        max_goal_snap_m=args.max_goal_snap_m,
        max_replan_path_length_ratio=args.max_replan_path_length_ratio,
    )


def run_local_config_from_args(args):
    inflation_radius = getattr(
        args,
        "run_local_map_inflation_radius_m",
        getattr(args, "obstacle_inflate_radius_m", 0.22),
    )
    return lidar_obstacle_map.RunLocalMapConfig(
        min_hit_count=getattr(args, "run_local_map_min_hit_count", 2),
        inflation_radius_m=inflation_radius,
        robot_footprint_radius_m=getattr(args, "robot_footprint_radius_m", 0.18),
        clearance_margin_m=getattr(args, "run_local_map_clearance_margin_m", 0.04),
        static_wall_exclusion_radius_m=getattr(
            args,
            "run_local_map_clearance_margin_m",
            0.04,
        ),
        min_used_points=getattr(args, "run_local_map_min_used_points", 3),
        max_rejected_ratio=getattr(args, "run_local_map_max_rejected_ratio", 0.90),
        max_updates=getattr(args, "run_local_map_max_updates", 3),
        max_replan_path_length_ratio=getattr(args, "max_replan_path_length_ratio", 3.0),
        max_start_snap_m=getattr(args, "max_start_snap_m", 0.20),
        max_goal_snap_m=getattr(args, "max_goal_snap_m", 0.30),
    )


def artifact_prefix_from_args(args, sequence=None):
    prefix = getattr(args, "run_local_map_artifact_prefix", None)
    if not prefix:
        prefix = f"{args.run_id}_run_local"
    if sequence is not None and sequence > 1:
        return f"{prefix}_{sequence:03d}"
    return prefix


def max_scan_age_from_args(args):
    return getattr(
        args,
        "run_local_map_max_scan_age_sec",
        getattr(args, "max_replan_scan_age_sec", 1.0),
    )


def max_tf_age_from_args(args):
    return getattr(
        args,
        "run_local_map_max_tf_age_sec",
        getattr(args, "max_replan_tf_age_sec", 1.0),
    )


def current_scan_age(node):
    if node.last_scan is None or node.last_scan_received_sec is None:
        return None
    return time.time() - node.last_scan_received_sec


def scan_stamp_sec(scan):
    return stamp_to_sec(getattr(getattr(scan, "header", None), "stamp", None))


def fresh_scan_or_error(
    node,
    args,
    min_scan_received_sec=None,
    min_scan_stamp_sec=None,
):
    if node.last_scan is None or node.last_scan_received_sec is None:
        raise RuntimeError("No /scan sample is available for LiDAR replan.")
    if (
        min_scan_received_sec is not None
        and node.last_scan_received_sec < min_scan_received_sec
    ):
        raise RuntimeError(
            f"{lidar_obstacle_map.RUN_LOCAL_FAILURE_STALE_SCAN}: "
            f"received={node.last_scan_received_sec:.3f}, "
            f"required_after={min_scan_received_sec:.3f}"
        )
    stamp_sec = scan_stamp_sec(node.last_scan)
    if (
        min_scan_stamp_sec is not None
        and stamp_sec is not None
        and stamp_sec < min_scan_stamp_sec - FRESH_SCAN_STAMP_SLACK_SEC
    ):
        raise RuntimeError(
            f"{lidar_obstacle_map.RUN_LOCAL_FAILURE_STALE_SCAN}: "
            f"stamp={stamp_sec:.3f}, "
            f"required_after={min_scan_stamp_sec:.3f}"
        )
    scan_age_sec = current_scan_age(node)
    if scan_age_sec is None or scan_age_sec > max_scan_age_from_args(args):
        raise RuntimeError(
            f"{lidar_obstacle_map.RUN_LOCAL_FAILURE_STALE_SCAN}: "
            f"age={scan_age_sec}, limit={max_scan_age_from_args(args):.3f}s"
        )
    return node.last_scan, scan_age_sec


def observations_from_latest_scan(
    node,
    args,
    scan_mode=None,
    min_scan_received_sec=None,
    min_scan_stamp_sec=None,
):
    scan, scan_age_sec = fresh_scan_or_error(
        node,
        args,
        min_scan_received_sec=min_scan_received_sec,
        min_scan_stamp_sec=min_scan_stamp_sec,
    )
    transform, lookup_mode = lookup_map_from_scan_transform(node, args, scan)
    tf_stamp_sec = stamp_to_sec(transform.header.stamp)
    tf_age_sec = None if tf_stamp_sec is None else time.time() - tf_stamp_sec
    if tf_age_sec is not None and tf_age_sec > max_tf_age_from_args(args):
        raise RuntimeError(
            f"{lidar_obstacle_map.RUN_LOCAL_FAILURE_STALE_TF}: "
            f"age={tf_age_sec:.3f}s, limit={max_tf_age_from_args(args):.3f}s"
        )
    mode = scan_mode or getattr(args, "run_local_map_update_mode", "forward")
    base_filter_config = replan_config_from_args(args) if mode == "forward" else None
    observations = scan_points_to_map_observations(scan, transform, base_filter_config)
    return observations, scan, scan_age_sec, tf_age_sec, lookup_mode


def collect_initial_observations(node, args):
    scan_count = getattr(args, "run_local_map_initial_scan_count", 5)
    observations = []
    latest_scan = None
    latest_scan_age = None
    latest_tf_age = None
    latest_lookup_mode = ""
    seen_stamps = set()
    deadline = time.time() + max(2.0, scan_count * 0.5)
    node.stop_repeatedly()
    min_scan_received_sec = time.time()
    min_scan_stamp_sec = min_scan_received_sec
    while len(seen_stamps) < scan_count and time.time() <= deadline:
        try:
            batch, scan, scan_age, tf_age, lookup_mode = observations_from_latest_scan(
                node,
                args,
                scan_mode="full",
                min_scan_received_sec=min_scan_received_sec,
                min_scan_stamp_sec=min_scan_stamp_sec,
            )
        except RuntimeError:
            if hasattr(node, "spin_once"):
                node.spin_once(0.1)
            elif "rclpy" in globals() and globals()["rclpy"] is not None:
                pass
            time.sleep(0.05)
            continue
        stamp = stamp_to_sec(scan.header.stamp)
        key = stamp if stamp is not None else node.last_scan_received_sec
        if key not in seen_stamps:
            seen_stamps.add(key)
            observations.extend(batch)
            latest_scan = scan
            latest_scan_age = scan_age
            latest_tf_age = tf_age
            latest_lookup_mode = lookup_mode
        if len(seen_stamps) >= scan_count:
            break
        if hasattr(node, "spin_once"):
            node.spin_once(0.1)
        time.sleep(0.05)
    if len(seen_stamps) < scan_count:
        raise RuntimeError(
            f"{lidar_obstacle_map.RUN_LOCAL_FAILURE_STALE_SCAN}: "
            f"collected={len(seen_stamps)}, required={scan_count}"
        )
    return observations, latest_scan, latest_scan_age, latest_tf_age, latest_lookup_mode, len(seen_stamps)


def apply_runtime_diagnostics(result, scan, scan_age_sec, tf_age_sec, lookup_mode):
    result.diagnostics.scan_frame = getattr(scan.header, "frame_id", "") if scan is not None else ""
    result.diagnostics.scan_age_sec = scan_age_sec
    result.diagnostics.tf_age_sec = tf_age_sec
    result.diagnostics.tf_lookup_mode = lookup_mode
    return result


def build_run_local_replan(
    args,
    observations,
    current_pose,
    goal_waypoint,
    old_remaining_waypoints,
    sequence=1,
    run_local_map=None,
):
    robot_pose = lidar_obstacle_map.Pose2D(
        current_pose.x,
        current_pose.y,
        current_pose.yaw_deg,
    )
    goal_pose = lidar_obstacle_map.Pose2D(
        goal_waypoint.x,
        goal_waypoint.y,
        0.0,
    )
    result = lidar_obstacle_map.build_run_local_replan_result(
        args.static_map,
        observations,
        robot_pose,
        goal_pose,
        args.run_id,
        output_dir=args.replan_output_dir,
        artifact_prefix=artifact_prefix_from_args(args, sequence=sequence),
        config=run_local_config_from_args(args),
        old_remaining_waypoints=old_remaining_waypoints,
        run_local_map=run_local_map,
    )
    replan_duration_sec = result.diagnostics.replan_duration_sec
    if (
        replan_duration_sec is not None
        and replan_duration_sec > args.replan_timeout_sec
    ):
        result.success = False
        result.reason = (
            "replan_timeout_exceeded "
            f"duration={replan_duration_sec:.3f}s "
            f"limit={args.replan_timeout_sec:.3f}s"
        )
    return result


def plan_existing_run_local_map(
    args,
    run_local_map,
    current_pose,
    goal_waypoint,
    old_remaining_waypoints,
    sequence=1,
):
    robot_pose = lidar_obstacle_map.Pose2D(
        current_pose.x,
        current_pose.y,
        current_pose.yaw_deg,
    )
    goal_pose = lidar_obstacle_map.Pose2D(
        goal_waypoint.x,
        goal_waypoint.y,
        0.0,
    )
    result = lidar_obstacle_map.plan_with_run_local_map(
        run_local_map,
        robot_pose,
        goal_pose,
        args.run_id,
        output_dir=args.replan_output_dir,
        artifact_prefix=artifact_prefix_from_args(args, sequence=sequence),
        old_remaining_waypoints=old_remaining_waypoints,
    )
    replan_duration_sec = result.diagnostics.replan_duration_sec
    if (
        replan_duration_sec is not None
        and replan_duration_sec > args.replan_timeout_sec
    ):
        result.success = False
        result.reason = (
            "replan_timeout_exceeded "
            f"duration={replan_duration_sec:.3f}s "
            f"limit={args.replan_timeout_sec:.3f}s"
        )
    return result


def perform_initial_run_local_replan(
    node,
    args,
    current_pose,
    goal_waypoint,
    old_remaining_waypoints,
):
    observations, scan, scan_age, tf_age, lookup_mode, collected_count = collect_initial_observations(
        node,
        args,
    )
    result = build_run_local_replan(
        args,
        observations,
        current_pose,
        goal_waypoint,
        old_remaining_waypoints,
        sequence=1,
        run_local_map=None,
    )
    result.diagnostics.run_local_initial_scan_count = collected_count
    return apply_runtime_diagnostics(result, scan, scan_age, tf_age, lookup_mode)


def update_run_local_map_from_latest_scan(node, args, current_pose, goal_waypoint, old_remaining_waypoints, sequence):
    observations, scan, scan_age, tf_age, lookup_mode = observations_from_latest_scan(
        node,
        args,
        scan_mode=getattr(args, "run_local_map_update_mode", "forward"),
    )
    result = build_run_local_replan(
        args,
        observations,
        current_pose,
        goal_waypoint,
        old_remaining_waypoints,
        sequence=sequence,
        run_local_map=getattr(node, "run_local_map", None),
    )
    return apply_runtime_diagnostics(result, scan, scan_age, tf_age, lookup_mode)


def perform_lidar_replan(
    node,
    args,
    current_pose,
    goal_waypoint,
    old_remaining_waypoints,
    sequence,
):
    node.stop_repeatedly()
    return update_run_local_map_from_latest_scan(
        node,
        args,
        current_pose,
        goal_waypoint,
        old_remaining_waypoints,
        sequence=sequence,
    )
