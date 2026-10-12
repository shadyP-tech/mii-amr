"""Stopped, fresh-scan, timestamped-TF orchestration for run-local A* replans."""

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


def scan_stamp_sec(scan):
    return stamp_to_sec(getattr(getattr(scan, "header", None), "stamp", None))


def now_sec(node):
    return node.now_sec() if hasattr(node, "now_sec") else time.time()


def time_from_stamp(stamp):
    return Time.from_msg(stamp) if Time is not None else stamp


def yaw_from_quaternion(qx, qy, qz, qw):
    if not all(math.isfinite(value) for value in (qx, qy, qz, qw)):
        raise RuntimeError("non-finite scan transform rotation")
    if abs(qx * qx + qy * qy + qz * qz + qw * qw - 1.0) > 0.1:
        raise RuntimeError("invalid scan transform quaternion")
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def transform_point_2d(transform, x, y):
    rotation = transform.transform.rotation
    translation = transform.transform.translation
    yaw = yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
    result = (float(translation.x) + math.cos(yaw) * x - math.sin(yaw) * y,
              float(translation.y) + math.sin(yaw) * x + math.cos(yaw) * y)
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError("non-finite scan transform position")
    return result


def scan_frame_points(scan):
    return lidar_obstacle_map.scan_ranges_to_base_points(
        scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max,
    )


def scan_points_to_map_observations(scan, map_from_scan_transform, base_filter_config=None,
                                    base_from_scan_transform=None):
    observations = []
    for point in scan_frame_points(scan):
        if base_filter_config is not None:
            if base_from_scan_transform is None:
                raise RuntimeError("forward obstacle ROI requires timestamped laser-to-base TF")
            base_x, base_y = transform_point_2d(base_from_scan_transform, point.x, point.y)
            base_point = lidar_obstacle_map.BaseFramePoint(base_x, base_y)
            if not lidar_obstacle_map.base_point_passes_roi(base_point, base_filter_config):
                continue
        x, y = transform_point_2d(map_from_scan_transform, point.x, point.y)
        observations.append(lidar_obstacle_map.MapFrameObservation(x, y, stamp_sec=scan_stamp_sec(scan)))
    return observations


def lookup_map_from_scan_transform(node, args, scan):
    frame = getattr(scan.header, "frame_id", "")
    if not frame:
        raise RuntimeError("scan frame is empty")
    try:
        transform = node.tf_buffer.lookup_transform(args.map_frame, frame, time_from_stamp(scan.header.stamp))
    except Exception as exc:
        raise RuntimeError(f"timestamped TF {args.map_frame}<-{frame} unavailable: {exc}") from exc
    # Scan endpoints are in the actual laser frame. A base-frame transform is
    # never a valid substitute, even when the laser and base are nearby.
    return transform, "timestamped"


def replan_config_from_args(args):
    return lidar_obstacle_map.ObstacleOverlayConfig(
        forward_distance_m=args.obstacle_forward_distance_m,
        forward_half_width_m=args.obstacle_forward_half_width_m,
        angle_window_deg=args.obstacle_angle_window_deg,
        min_range_m=args.obstacle_min_range_m,
        robot_footprint_radius_m=args.robot_footprint_radius_m,
    )


def run_local_config_from_args(args):
    return lidar_obstacle_map.RunLocalMapConfig(
        min_hit_count=args.run_local_map_min_hit_count,
        inflation_radius_m=args.run_local_map_inflation_radius_m,
        robot_footprint_radius_m=args.robot_footprint_radius_m,
        clearance_margin_m=args.run_local_map_clearance_margin_m,
        static_wall_exclusion_radius_m=args.run_local_map_clearance_margin_m,
        min_used_points=args.run_local_map_min_used_points,
        max_rejected_ratio=args.run_local_map_max_rejected_ratio,
        max_updates=args.run_local_map_max_updates,
        max_replan_path_length_ratio=args.max_replan_path_length_ratio,
        max_start_snap_m=args.max_start_snap_m,
        max_goal_snap_m=args.max_goal_snap_m,
        planner_inflate_radius_m=args.robot_footprint_radius_m + args.run_local_map_clearance_margin_m,
    )


def fresh_scan_or_error(node, args, min_scan_received_sec=None, min_scan_stamp_sec=None):
    scan = node.last_scan
    received = node.last_scan_received_sec
    if scan is None or received is None:
        raise RuntimeError("No /scan sample available for LiDAR replan")
    age = time.time() - received
    stamp = scan_stamp_sec(scan)
    stamp_age = None if stamp is None else now_sec(node) - stamp
    limit = min(args.max_replan_scan_age_sec, args.run_local_map_max_scan_age_sec)
    if (not math.isfinite(age) or age < 0 or age > limit or stamp is None or stamp <= 0
            or not math.isfinite(stamp_age) or not -FRESH_SCAN_STAMP_SLACK_SEC <= stamp_age <= limit
            or (min_scan_received_sec is not None and received <= min_scan_received_sec)
            or (min_scan_stamp_sec is not None and stamp < min_scan_stamp_sec)):
        raise RuntimeError(f"{lidar_obstacle_map.RUN_LOCAL_FAILURE_STALE_SCAN}: received_age={age}, stamp_age={stamp_age}")
    return scan, max(age, stamp_age)


def observations_from_latest_scan(node, args, scan_mode=None,
                                  min_scan_received_sec=None, min_scan_stamp_sec=None):
    scan, scan_age = fresh_scan_or_error(node, args, min_scan_received_sec, min_scan_stamp_sec)
    transform, lookup_mode = lookup_map_from_scan_transform(node, args, scan)
    stamp = stamp_to_sec(transform.header.stamp)
    tf_age = None if stamp is None else now_sec(node) - stamp
    limit = min(args.max_replan_tf_age_sec, args.run_local_map_max_tf_age_sec)
    if (stamp is None or stamp <= 0 or not math.isfinite(tf_age)
            or not -0.25 <= tf_age <= limit
            or abs(stamp - scan_stamp_sec(scan)) > FRESH_SCAN_STAMP_SLACK_SEC):
        raise RuntimeError(f"{lidar_obstacle_map.RUN_LOCAL_FAILURE_STALE_TF}: age={tf_age}")
    mode = scan_mode or args.run_local_map_update_mode
    roi = replan_config_from_args(args) if mode == "forward" else None
    base_transform = None
    if roi is not None:
        try:
            base_transform = node.tf_buffer.lookup_transform(
                node.base_frame_used, scan.header.frame_id, time_from_stamp(scan.header.stamp),
            )
        except Exception as exc:
            raise RuntimeError(f"forward obstacle ROI laser-to-base TF unavailable: {exc}") from exc
    observations = scan_points_to_map_observations(scan, transform, roi, base_transform)
    return observations, scan, scan_age, tf_age, lookup_mode


def collect_stopped_observations(node, args, scan_count, scan_mode,
                                 min_scan_received_sec=None, min_scan_stamp_sec=None):
    node.stop_repeatedly()
    received_after = time.time() if min_scan_received_sec is None else min_scan_received_sec
    stamped_after = now_sec(node) if min_scan_stamp_sec is None else min_scan_stamp_sec
    deadline = time.monotonic() + max(2.0, scan_count * 1.0)
    observations, seen = [], set()
    last_error = "no post-stop scan"
    latest = None
    while time.monotonic() < deadline:
        if not node.is_running():
            raise RuntimeError("ROS shutdown during replan scan collection")
        node.publish_velocity(0.0, 0.0)
        node.spin_once(0.05)
        try:
            batch, scan, age, tf_age, mode = observations_from_latest_scan(
                node, args, scan_mode, received_after, stamped_after,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            continue
        stamp = scan_stamp_sec(scan)
        if stamp not in seen:
            observations.extend(batch)
            seen.add(stamp)
            latest = (scan, age, tf_age, mode)
        if len(seen) == scan_count:
            return observations, *latest, len(seen)
    raise RuntimeError(f"Fresh replan scans timed out ({len(seen)}/{scan_count}): {last_error}")


def build_run_local_replan(args, observations, current_pose, goal_waypoint,
                           old_remaining_waypoints, sequence=1, run_local_map=None):
    started = time.monotonic()
    result = lidar_obstacle_map.build_run_local_replan_result(
        args.static_map, observations,
        lidar_obstacle_map.Pose2D(current_pose.x, current_pose.y, current_pose.yaw_deg),
        lidar_obstacle_map.Pose2D(goal_waypoint.x, goal_waypoint.y, 0.0),
        args.run_id, config=run_local_config_from_args(args),
        old_remaining_waypoints=old_remaining_waypoints, run_local_map=run_local_map,
    )
    if time.monotonic() - started > args.replan_timeout_sec:
        result.success = False
        result.reason = "replan_timeout_exceeded"
    return result


def _perform(node, args, current_pose, goal_waypoint, remaining, count, mode, sequence,
             min_scan_received_sec=None, min_scan_stamp_sec=None):
    observations, scan, age, tf_age, lookup_mode, collected = collect_stopped_observations(
        node, args, count, mode, min_scan_received_sec, min_scan_stamp_sec,
    )
    # Collection can take seconds. Use a newly gated robot pose for planning.
    current_pose = node.check_health_or_raise()
    result = build_run_local_replan(args, observations, current_pose, goal_waypoint, remaining,
                                   sequence, getattr(node, "run_local_map", None))
    result.diagnostics.scan_frame = scan.header.frame_id
    result.diagnostics.scan_age_sec = age
    result.diagnostics.tf_age_sec = tf_age
    result.diagnostics.tf_lookup_mode = lookup_mode
    result.diagnostics.run_local_initial_scan_count = collected
    return result


def perform_initial_run_local_replan(node, args, current_pose, goal_waypoint, old_remaining_waypoints):
    return _perform(node, args, current_pose, goal_waypoint, old_remaining_waypoints,
                    args.run_local_map_initial_scan_count, args.run_local_map_initial_scan_mode, 1)


def perform_lidar_replan(node, args, current_pose, goal_waypoint, old_remaining_waypoints,
                         sequence, min_scan_received_sec=None, min_scan_stamp_sec=None):
    return _perform(node, args, current_pose, goal_waypoint, old_remaining_waypoints,
                    args.run_local_map_initial_scan_count, args.run_local_map_update_mode, sequence,
                    min_scan_received_sec, min_scan_stamp_sec)
