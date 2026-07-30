"""Frame transforms and 180-degree-symmetric stand-axis geometry."""

from __future__ import annotations

import math
from typing import Sequence

from scripts.aufgabe04.perception.stand_axis_handoff.models import (
    ApproachPose,
    RigidTransform,
)


def axial_normalize_rad(angle_rad: float) -> float:
    if not math.isfinite(angle_rad):
        raise ValueError("axis angle must be finite")
    return 0.5 * math.atan2(math.sin(2.0 * angle_rad), math.cos(2.0 * angle_rad))


def axial_difference_rad(left_rad: float, right_rad: float) -> float:
    return abs(axial_normalize_rad(left_rad - right_rad))


def _normalized_quaternion(
    rotation_xyzw: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(rotation_xyzw) != 4:
        raise ValueError("quaternion must contain four values")
    values = tuple(float(value) for value in rotation_xyzw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm must be positive")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def rotate_vector(
    vector_xyz: Sequence[float],
    rotation_xyzw: Sequence[float],
) -> tuple[float, float, float]:
    if len(vector_xyz) != 3:
        raise ValueError("vector must contain three values")
    x, y, z = (float(value) for value in vector_xyz)
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError("vector must be finite")
    qx, qy, qz, qw = _normalized_quaternion(rotation_xyzw)
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


def transform_point(
    point_xyz: Sequence[float],
    transform: RigidTransform,
) -> tuple[float, float, float]:
    rotated = rotate_vector(point_xyz, transform.rotation_xyzw)
    return tuple(
        value + offset
        for value, offset in zip(rotated, transform.translation_xyz_m)
    )  # type: ignore[return-value]


def rectified_pixel_bearing_in_scan(
    *,
    u_px: float,
    v_px: float,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    scan_from_camera: RigidTransform,
) -> float:
    values = (u_px, v_px, fx_px, fy_px, cx_px, cy_px)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("pixel projection values must be finite")
    if fx_px <= 0.0 or fy_px <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    camera_ray = (
        (u_px - cx_px) / fx_px,
        (v_px - cy_px) / fy_px,
        1.0,
    )
    scan_ray = rotate_vector(camera_ray, scan_from_camera.rotation_xyzw)
    horizontal_norm = math.hypot(scan_ray[0], scan_ray[1])
    if horizontal_norm <= 1.0e-9:
        raise ValueError("camera ray has no stable scan-plane bearing")
    return math.atan2(scan_ray[1], scan_ray[0])


def camera_axis_in_scan(
    *,
    camera_yaw_rad: float,
    scan_from_camera: RigidTransform,
) -> float:
    """Transform a visible-face yaw into a scan-frame tangent axis.

    Camera optical coordinates use +x image-right, +y image-down, +z forward.
    The public camera-yaw convention is positive image-left.
    """

    if not math.isfinite(camera_yaw_rad):
        raise ValueError("camera yaw must be finite")
    normal_camera = (
        -math.sin(camera_yaw_rad),
        0.0,
        math.cos(camera_yaw_rad),
    )
    normal_scan = rotate_vector(normal_camera, scan_from_camera.rotation_xyzw)
    horizontal_norm = math.hypot(normal_scan[0], normal_scan[1])
    if horizontal_norm <= 1.0e-9:
        raise ValueError("camera face normal has no stable scan-plane projection")
    # Horizontal tangent z_hat x normal = (-normal_y, normal_x).
    return axial_normalize_rad(math.atan2(normal_scan[0], -normal_scan[1]))


def camera_face_normal_axis_in_scan(
    *,
    camera_face_normal_xyz: Sequence[float],
    scan_from_camera: RigidTransform,
) -> float:
    """Transform a metric PnP face normal into a scan-frame tangent axis."""

    normal_scan = rotate_vector(
        camera_face_normal_xyz,
        scan_from_camera.rotation_xyzw,
    )
    horizontal_norm = math.hypot(normal_scan[0], normal_scan[1])
    if horizontal_norm <= 1.0e-9:
        raise ValueError("camera face normal has no stable scan-plane projection")
    return axial_normalize_rad(math.atan2(normal_scan[0], -normal_scan[1]))


def approach_pose_from_axis(
    *,
    stand_center_xy_m: tuple[float, float],
    stand_axis_rad: float,
    stand_off_m: float,
) -> ApproachPose:
    if not math.isfinite(stand_off_m) or stand_off_m <= 0.0:
        raise ValueError("stand-off distance must be finite and positive")
    center_x, center_y = stand_center_xy_m
    if not all(math.isfinite(value) for value in (center_x, center_y)):
        raise ValueError("stand center must be finite")
    tangent = (math.cos(stand_axis_rad), math.sin(stand_axis_rad))
    normals = ((-tangent[1], tangent[0]), (tangent[1], -tangent[0]))
    toward_robot = (-center_x, -center_y)
    normal = max(
        normals,
        key=lambda value: value[0] * toward_robot[0] + value[1] * toward_robot[1],
    )
    approach_x = center_x + stand_off_m * normal[0]
    approach_y = center_y + stand_off_m * normal[1]
    yaw = math.atan2(center_y - approach_y, center_x - approach_x)
    return ApproachPose(approach_x, approach_y, yaw, stand_off_m)
