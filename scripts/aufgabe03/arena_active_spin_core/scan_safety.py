from __future__ import annotations

import math

from arena_geometry_localizer import Pose2D, ScanSample

from .math_utils import normalize_angle_rad
from .models import ArenaActiveSpinConfig, PosePrior, SectorClearance


def yaw_from_quaternion(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def odom_pose_from_msg(msg):
    pose = msg.pose.pose
    orientation = pose.orientation
    return Pose2D(
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw_deg=math.degrees(
            yaw_from_quaternion(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
        ),
    )


def scan_sample_from_msg(msg, odom_pose):
    return ScanSample(
        ranges=list(msg.ranges),
        angle_min=float(msg.angle_min),
        angle_increment=float(msg.angle_increment),
        range_min=float(msg.range_min),
        range_max=float(msg.range_max),
        odom_pose=odom_pose,
    )


def valid_range(value, range_min, range_max):
    return (
        value is not None
        and math.isfinite(value)
        and value >= range_min
        and value <= range_max
    )


def angle_in_sector(angle_deg, ranges):
    return any(lower <= angle_deg <= upper for lower, upper in ranges)


def min_sector_range(scan, sectors):
    values = []
    for index, raw_range in enumerate(scan.ranges):
        if not valid_range(raw_range, scan.range_min, scan.range_max):
            continue
        angle_rad = scan.angle_min + index * scan.angle_increment
        angle_deg = math.degrees(normalize_angle_rad(angle_rad))
        if angle_in_sector(angle_deg, sectors):
            values.append(float(raw_range))
    return min(values) if values else None


def evaluate_clearance(scan, config: ArenaActiveSpinConfig):
    front = min_sector_range(scan, [(-30.0, 30.0)])
    left = min_sector_range(scan, [(60.0, 120.0)])
    right = min_sector_range(scan, [(-120.0, -60.0)])
    rear = min_sector_range(scan, [(150.0, 180.0), (-180.0, -150.0)])
    checks = [
        ("front_clearance_missing", "front_clearance_below_limit", front, config.min_front_clearance_m),
        ("left_clearance_missing", "left_clearance_below_limit", left, config.min_side_clearance_m),
        ("right_clearance_missing", "right_clearance_below_limit", right, config.min_side_clearance_m),
        ("rear_clearance_missing", "rear_clearance_below_limit", rear, config.min_rear_clearance_m),
    ]
    for missing_reason, low_reason, value, limit in checks:
        if value is None:
            return SectorClearance(False, missing_reason, front, left, right, rear)
        if value < limit:
            return SectorClearance(False, low_reason, front, left, right, rear)
    return SectorClearance(True, "ok", front, left, right, rear)


def evaluate_reposition_clearance(scan, config: ArenaActiveSpinConfig):
    front = min_sector_range(scan, [(-30.0, 30.0)])
    left = min_sector_range(scan, [(60.0, 120.0)])
    right = min_sector_range(scan, [(-120.0, -60.0)])
    rear = min_sector_range(scan, [(150.0, 180.0), (-180.0, -150.0)])
    checks = [
        (
            "front_clearance_missing",
            "front_clearance_below_limit",
            front,
            config.center_reposition_min_front_clearance_m,
        ),
        ("left_clearance_missing", "left_clearance_below_limit", left, config.min_side_clearance_m),
        ("right_clearance_missing", "right_clearance_below_limit", right, config.min_side_clearance_m),
        ("rear_clearance_missing", "rear_clearance_below_limit", rear, config.min_rear_clearance_m),
    ]
    for missing_reason, low_reason, value, limit in checks:
        if value is None:
            return SectorClearance(False, missing_reason, front, left, right, rear)
        if value < limit:
            return SectorClearance(False, low_reason, front, left, right, rear)
    return SectorClearance(True, "ok", front, left, right, rear)


def min_valid_scan_range(scan):
    values = [
        float(value)
        for value in getattr(scan, "ranges", [])
        if value is not None
        and math.isfinite(float(value))
        and float(value) >= float(scan.range_min)
        and float(value) <= float(scan.range_max)
    ]
    return min(values) if values else None


def dynamic_lateral_heading_from_scan(scan, current_yaw_rad):
    left = min_sector_range(scan, [(60.0, 120.0)])
    right = min_sector_range(scan, [(-120.0, -60.0)])
    if left is None or right is None:
        raise RuntimeError("center_reposition_dynamic_lateral_clearance_missing")
    if left >= right:
        return {
            "odom_heading_rad": normalize_angle_rad(current_yaw_rad + math.pi / 2.0),
            "direction": "left",
            "left_clearance_m": left,
            "right_clearance_m": right,
        }
    return {
        "odom_heading_rad": normalize_angle_rad(current_yaw_rad - math.pi / 2.0),
        "direction": "right",
        "left_clearance_m": left,
        "right_clearance_m": right,
    }


def covariance_list_from_localizer(covariance):
    values = [0.0] * 36
    values[0] = float(covariance["x_m2"])
    values[7] = float(covariance["y_m2"])
    values[35] = float(covariance["yaw_rad2"])
    return values


def pose_prior_from_localizer_result(result):
    pose = result.estimated_pose_prior
    covariance = result.estimated_covariance
    if pose is None or covariance is None:
        return None
    return PosePrior(
        x_m=float(pose.x),
        y_m=float(pose.y),
        yaw_rad=math.radians(float(pose.yaw_deg)),
        covariance=covariance_list_from_localizer(covariance),
    )
