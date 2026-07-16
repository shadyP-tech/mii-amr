"""Simulation-only LiDAR/QR seeded stand-head image crops."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HeadRoi:
    x0: int
    y0: int
    x1: int
    y1: int
    source: str
    expected_head_px: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class CameraTargetProjection:
    """Planar target geometry expressed at the simulated camera origin."""

    bearing_rad: float
    depth_m: float
    height_delta_m: float


def project_target_to_camera(
    *,
    robot_x_m: float,
    robot_y_m: float,
    robot_z_m: float,
    robot_yaw_rad: float,
    target_x_m: float,
    target_y_m: float,
    target_height_m: float,
    camera_forward_offset_m: float,
    camera_lateral_offset_m: float,
    camera_height_m: float,
    camera_yaw_offset_rad: float = 0.0,
) -> CameraTargetProjection:
    """Transform a known target centre into the camera's pinhole frame.

    The Gazebo Burger camera uses x-forward, y-left, z-up model coordinates.
    Image projection retains the historical simulation convention that a
    positive bearing moves toward increasing image columns.
    """

    values = (
        robot_x_m,
        robot_y_m,
        robot_z_m,
        robot_yaw_rad,
        target_x_m,
        target_y_m,
        target_height_m,
        camera_forward_offset_m,
        camera_lateral_offset_m,
        camera_height_m,
        camera_yaw_offset_rad,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("camera target projection inputs must be finite")
    dx_world = target_x_m - robot_x_m
    dy_world = target_y_m - robot_y_m
    cos_yaw = math.cos(robot_yaw_rad)
    sin_yaw = math.sin(robot_yaw_rad)
    forward_base = cos_yaw * dx_world + sin_yaw * dy_world
    lateral_base = -sin_yaw * dx_world + cos_yaw * dy_world
    forward_offset = forward_base - camera_forward_offset_m
    lateral_offset = lateral_base - camera_lateral_offset_m
    cos_camera = math.cos(camera_yaw_offset_rad)
    sin_camera = math.sin(camera_yaw_offset_rad)
    depth_m = cos_camera * forward_offset + sin_camera * lateral_offset
    lateral_m = -sin_camera * forward_offset + cos_camera * lateral_offset
    # ROS model coordinates use +y to the camera's left, while image columns
    # increase to the right.  Preserve the simulation image-bearing convention
    # documented above by negating the model-frame lateral component.
    bearing_rad = math.atan2(-lateral_m, depth_m)
    camera_world_height_m = robot_z_m + camera_height_m
    return CameraTargetProjection(
        bearing_rad=bearing_rad,
        depth_m=depth_m,
        height_delta_m=target_height_m - camera_world_height_m,
    )


def silhouette_close_kernel(expected_head_px: float, *, maximum: int = 7) -> int:
    """Scale edge closing conservatively without filling simulated QR texture."""

    if not math.isfinite(expected_head_px) or expected_head_px <= 0.0:
        raise ValueError("expected_head_px must be finite and positive")
    if maximum < 3:
        raise ValueError("maximum silhouette close kernel must be at least 3")
    maximum = maximum if maximum % 2 == 1 else maximum - 1
    scaled = max(3, int(round(expected_head_px * 0.05)) | 1)
    return min(maximum, scaled)


def silhouette_min_edge_height_px(
    expected_head_px: float,
    *,
    minimum: float = 5.0,
    maximum: float = 12.0,
) -> float:
    """Scale face-edge evidence without making the narrow stem undetectable."""

    values = (expected_head_px, minimum, maximum)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("silhouette edge-height inputs must be finite")
    if expected_head_px <= 0.0 or minimum <= 0.0 or maximum < minimum:
        raise ValueError("silhouette edge-height bounds must be positive and ordered")
    return max(minimum, min(maximum, 0.18 * expected_head_px))


def stand_head_roi(
    *,
    frame_width: int,
    frame_height: int,
    bearing_rad: float,
    distance_m: float | None,
    camera_fx_px: float,
    camera_fy_px: float | None = None,
    camera_cx_px: float,
    camera_cy_px: float,
    stand_face_size_m: float,
    camera_depth_m: float | None = None,
    target_height_delta_m: float | None = None,
    qr_corners_px: tuple[tuple[float, float], ...] = (),
    padding_scale: float = 2.2,
) -> HeadRoi | None:
    effective_fy_px = camera_fx_px if camera_fy_px is None else camera_fy_px
    if (
        frame_width <= 0
        or frame_height <= 0
        or camera_fx_px <= 0.0
        or effective_fy_px <= 0.0
        or stand_face_size_m <= 0.0
    ):
        return None
    half_horizontal_fov = math.atan2(frame_width / 2.0, camera_fx_px)
    if not math.isfinite(bearing_rad) or abs(bearing_rad) >= half_horizontal_fov:
        return None
    projection_depth_m = camera_depth_m if camera_depth_m is not None else distance_m
    expected = (
        max(camera_fx_px, effective_fy_px) * stand_face_size_m / projection_depth_m
        if projection_depth_m is not None and projection_depth_m > 0.0
        else 40.0
    )
    expected = max(16.0, min(float(min(frame_width, frame_height)), expected))
    if qr_corners_px:
        xs = [point[0] for point in qr_corners_px]
        ys = [point[1] for point in qr_corners_px]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        qr_extent = max(max(xs) - min(xs), max(ys) - min(ys))
        extent = max(expected, qr_extent * padding_scale)
        source = "qr_seeded"
    else:
        center_x = camera_cx_px + camera_fx_px * math.tan(bearing_rad)
        center_y = camera_cy_px
        if target_height_delta_m is not None:
            if not math.isfinite(target_height_delta_m):
                return None
            if camera_depth_m is None or camera_depth_m <= 0.0:
                return None
            center_y -= effective_fy_px * target_height_delta_m / camera_depth_m
        extent = expected * padding_scale
        source = "lidar_projected"
    half = max(18.0, extent / 2.0)
    x0 = max(0, int(math.floor(center_x - half)))
    y0 = max(0, int(math.floor(center_y - half)))
    x1 = min(frame_width, int(math.ceil(center_x + half)))
    y1 = min(frame_height, int(math.ceil(center_y + half)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return HeadRoi(x0, y0, x1, y1, source, expected)


def qr_corners_inside_roi(
    corners_px: tuple[tuple[float, float], ...],
    roi: HeadRoi | None,
) -> bool:
    """Return true only when every QR corner belongs to the selected target ROI."""

    if roi is None or len(corners_px) != 4:
        return False
    return all(
        roi.x0 <= x_px < roi.x1 and roi.y0 <= y_px < roi.y1
        for x_px, y_px in corners_px
    )
