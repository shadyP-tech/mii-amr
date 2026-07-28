"""ROS-free pinhole and rigid-transform helpers for the real camera adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scripts.aufgabe04.navigation.models import Pose2D


@dataclass(frozen=True)
class CameraIntrinsics:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float


@dataclass(frozen=True)
class OpticalProjection:
    u_px: float
    v_px: float
    depth_m: float
    expected_size_px: float
    inside_image: bool


@dataclass(frozen=True)
class ImageRoi:
    x0: int
    y0: int
    x1: int
    y1: int
    expected_size_px: float


def intrinsics_from_camera_info(camera_info) -> CameraIntrinsics:
    projection = tuple(float(value) for value in camera_info.p)
    if len(projection) != 12:
        raise ValueError("CameraInfo projection matrix must contain 12 values")
    intrinsics = CameraIntrinsics(
        width_px=int(camera_info.width),
        height_px=int(camera_info.height),
        fx_px=projection[0],
        fy_px=projection[5],
        cx_px=projection[2],
        cy_px=projection[6],
    )
    validate_intrinsics(intrinsics)
    return intrinsics


def validate_intrinsics(intrinsics: CameraIntrinsics) -> None:
    values = (
        intrinsics.fx_px,
        intrinsics.fy_px,
        intrinsics.cx_px,
        intrinsics.cy_px,
    )
    if intrinsics.width_px <= 0 or intrinsics.height_px <= 0:
        raise ValueError("camera dimensions must be positive")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("camera intrinsics must be finite")
    if intrinsics.fx_px <= 0.0 or intrinsics.fy_px <= 0.0:
        raise ValueError("camera focal lengths must be positive")


def transform_point(
    point_xyz: Sequence[float],
    *,
    translation_xyz: Sequence[float],
    rotation_xyzw: Sequence[float],
) -> tuple[float, float, float]:
    """Apply a parent<-child transform to a point in the child frame."""

    if len(point_xyz) != 3 or len(translation_xyz) != 3 or len(rotation_xyzw) != 4:
        raise ValueError("rigid transform dimensions are invalid")
    point = tuple(float(value) for value in point_xyz)
    translation = tuple(float(value) for value in translation_xyz)
    quaternion = _normalized_quaternion(rotation_xyzw)
    if not all(math.isfinite(value) for value in (*point, *translation)):
        raise ValueError("rigid transform inputs must be finite")
    rotated = rotate_vector(point, quaternion)
    return tuple(
        rotated_value + translation_value
        for rotated_value, translation_value in zip(rotated, translation)
    )  # type: ignore[return-value]


def rotate_vector(
    vector_xyz: Sequence[float],
    rotation_xyzw: Sequence[float],
) -> tuple[float, float, float]:
    if len(vector_xyz) != 3 or len(rotation_xyzw) != 4:
        raise ValueError("quaternion rotation dimensions are invalid")
    x, y, z = (float(value) for value in vector_xyz)
    qx, qy, qz, qw = _normalized_quaternion(rotation_xyzw)
    # Expanded q * v * q^-1.
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


def project_optical_point(
    point_camera_xyz: Sequence[float],
    intrinsics: CameraIntrinsics,
    *,
    physical_size_m: float,
) -> OpticalProjection:
    """Project a point in REP-103 optical coordinates (+z forward)."""

    validate_intrinsics(intrinsics)
    if len(point_camera_xyz) != 3:
        raise ValueError("camera point must contain three values")
    x_right, y_down, depth = (float(value) for value in point_camera_xyz)
    if not all(math.isfinite(value) for value in (x_right, y_down, depth)):
        raise ValueError("camera point must be finite")
    if not math.isfinite(physical_size_m) or physical_size_m <= 0.0:
        raise ValueError("physical_size_m must be finite and positive")
    if depth <= 0.0:
        return OpticalProjection(math.nan, math.nan, depth, 0.0, False)
    u_px = intrinsics.cx_px + intrinsics.fx_px * x_right / depth
    v_px = intrinsics.cy_px + intrinsics.fy_px * y_down / depth
    expected = max(intrinsics.fx_px, intrinsics.fy_px) * physical_size_m / depth
    return OpticalProjection(
        u_px=u_px,
        v_px=v_px,
        depth_m=depth,
        expected_size_px=expected,
        inside_image=(
            0.0 <= u_px < intrinsics.width_px
            and 0.0 <= v_px < intrinsics.height_px
        ),
    )


def project_rectified_image_direction(
    top_camera_xyz: Sequence[float],
    bottom_camera_xyz: Sequence[float],
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """Project a camera-frame 3D line into normalized rectified-image direction.

    Points use REP-103 optical coordinates and the projection matrix already
    represented by ``CameraIntrinsics``. The returned direction runs from the
    projected top point toward the projected bottom point, matching the
    top-to-bottom ordering of the head's left and right image sides.
    """

    validate_intrinsics(intrinsics)
    if len(top_camera_xyz) != 3 or len(bottom_camera_xyz) != 3:
        raise ValueError("camera line endpoints must each contain three values")
    top = tuple(float(value) for value in top_camera_xyz)
    bottom = tuple(float(value) for value in bottom_camera_xyz)
    if not all(math.isfinite(value) for value in (*top, *bottom)):
        raise ValueError("camera line endpoints must be finite")
    if top[2] <= 0.0 or bottom[2] <= 0.0:
        raise ValueError("camera line endpoints must be in front of the camera")

    top_u = intrinsics.cx_px + intrinsics.fx_px * top[0] / top[2]
    top_v = intrinsics.cy_px + intrinsics.fy_px * top[1] / top[2]
    bottom_u = intrinsics.cx_px + intrinsics.fx_px * bottom[0] / bottom[2]
    bottom_v = intrinsics.cy_px + intrinsics.fy_px * bottom[1] / bottom[2]
    delta_u = bottom_u - top_u
    delta_v = bottom_v - top_v
    norm = math.hypot(delta_u, delta_v)
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise ValueError("projected camera line has no stable image direction")
    return delta_u / norm, delta_v / norm


def roi_from_projection(
    projection: OpticalProjection,
    intrinsics: CameraIntrinsics,
    *,
    padding_scale: float = 1.8,
    minimum_extent_px: float = 18.0,
) -> ImageRoi | None:
    validate_intrinsics(intrinsics)
    if not projection.inside_image or projection.depth_m <= 0.0:
        return None
    if not math.isfinite(padding_scale) or padding_scale < 1.0:
        raise ValueError("padding_scale must be finite and at least 1")
    extent = max(minimum_extent_px, projection.expected_size_px * padding_scale)
    half = extent / 2.0
    x0 = max(0, int(math.floor(projection.u_px - half)))
    y0 = max(0, int(math.floor(projection.v_px - half)))
    x1 = min(intrinsics.width_px, int(math.ceil(projection.u_px + half)))
    y1 = min(intrinsics.height_px, int(math.ceil(projection.v_px + half)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return ImageRoi(x0, y0, x1, y1, projection.expected_size_px)


def pose2d_from_transform(transform) -> Pose2D:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    quaternion = (rotation.x, rotation.y, rotation.z, rotation.w)
    forward = rotate_vector((1.0, 0.0, 0.0), quaternion)
    return Pose2D(
        float(translation.x),
        float(translation.y),
        math.atan2(forward[1], forward[0]),
    )


def optical_heading_from_transform(transform) -> float:
    """Return the map-frame heading of a camera optical +z axis."""

    rotation = transform.transform.rotation
    optical_forward = rotate_vector(
        (0.0, 0.0, 1.0),
        (rotation.x, rotation.y, rotation.z, rotation.w),
    )
    planar_norm = math.hypot(optical_forward[0], optical_forward[1])
    if planar_norm <= 1.0e-6:
        raise ValueError("camera optical axis has no stable map-plane projection")
    return math.atan2(optical_forward[1], optical_forward[0])


def _normalized_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    quaternion = tuple(float(value) for value in values)
    if len(quaternion) != 4 or not all(math.isfinite(value) for value in quaternion):
        raise ValueError("quaternion must contain four finite values")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm must be positive")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]
