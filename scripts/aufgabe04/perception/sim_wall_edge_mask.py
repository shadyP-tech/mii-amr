"""Simulation-only arena-wall exclusion masks for stand silhouette edges.

The known rectangular arena supplies the wall geometry.  A synchronized
LaserScan must confirm every projected wall ray, and closer returns carve a
protected foreground corridor so stands are never removed with the wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.simulation.sim_head_roi import project_target_to_camera


@dataclass(frozen=True)
class WallEdgeMaskResult:
    mask: object | None
    reason: str
    confirmed_wall_samples: int
    protected_foreground_samples: int
    scan_delta_sec: float | None


def ray_distance_to_arena_wall(
    *,
    origin_x_m: float,
    origin_y_m: float,
    ray_yaw_rad: float,
    arena: ArenaBounds,
    wall_thickness_m: float = 0.04,
) -> float | None:
    """Intersect a world ray with the inner faces of a rectangular arena."""

    arena.validate()
    if wall_thickness_m < 0.0:
        raise ValueError("wall_thickness_m must be non-negative")
    yaw = math.radians(arena.yaw_deg)
    dx = origin_x_m - arena.center_x_m
    dy = origin_y_m - arena.center_y_m
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    local_ray = ray_yaw_rad - yaw
    ray_x = math.cos(local_ray)
    ray_y = math.sin(local_ray)
    half_x = arena.length_m / 2.0 - wall_thickness_m / 2.0
    half_y = arena.width_m / 2.0 - wall_thickness_m / 2.0
    if half_x <= 0.0 or half_y <= 0.0:
        return None

    intersections: list[float] = []
    if abs(ray_x) > 1e-9:
        for boundary_x in (-half_x, half_x):
            distance = (boundary_x - local_x) / ray_x
            projected_y = local_y + distance * ray_y
            if distance > 0.0 and -half_y - 1e-6 <= projected_y <= half_y + 1e-6:
                intersections.append(distance)
    if abs(ray_y) > 1e-9:
        for boundary_y in (-half_y, half_y):
            distance = (boundary_y - local_y) / ray_y
            projected_x = local_x + distance * ray_x
            if distance > 0.0 and -half_x - 1e-6 <= projected_x <= half_x + 1e-6:
                intersections.append(distance)
    return min(intersections) if intersections else None


def _project_world_point(
    *,
    robot_x_m: float,
    robot_y_m: float,
    robot_z_m: float,
    robot_yaw_rad: float,
    point_x_m: float,
    point_y_m: float,
    point_z_m: float,
    camera_forward_offset_m: float,
    camera_lateral_offset_m: float,
    camera_height_m: float,
    camera_yaw_offset_rad: float,
    camera_fx_px: float,
    camera_fy_px: float,
    camera_cx_px: float,
    camera_cy_px: float,
) -> tuple[float, float, float] | None:
    projection = project_target_to_camera(
        robot_x_m=robot_x_m,
        robot_y_m=robot_y_m,
        robot_z_m=robot_z_m,
        robot_yaw_rad=robot_yaw_rad,
        target_x_m=point_x_m,
        target_y_m=point_y_m,
        target_height_m=point_z_m,
        camera_forward_offset_m=camera_forward_offset_m,
        camera_lateral_offset_m=camera_lateral_offset_m,
        camera_height_m=camera_height_m,
        camera_yaw_offset_rad=camera_yaw_offset_rad,
    )
    if projection.depth_m <= 1e-6:
        return None
    u_px = camera_cx_px + camera_fx_px * math.tan(projection.bearing_rad)
    v_px = camera_cy_px - camera_fy_px * projection.height_delta_m / projection.depth_m
    return u_px, v_px, projection.depth_m


def _cluster_foreground_columns(
    columns_px: list[float],
    *,
    max_gap_px: float,
    padding_px: int,
) -> tuple[tuple[int, int], ...]:
    """Merge adjacent projected scan returns into measured image spans."""

    if not columns_px:
        return ()
    ordered = sorted(value for value in columns_px if math.isfinite(value))
    if not ordered:
        return ()
    clusters: list[tuple[float, float]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value - previous > max_gap_px:
            clusters.append((start, previous))
            start = value
        previous = value
    clusters.append((start, previous))
    return tuple(
        (
            int(math.floor(start_px)) - padding_px,
            int(math.ceil(end_px)) + padding_px,
        )
        for start_px, end_px in clusters
    )


def _draw_projected_wall_column(
    cv2,
    mask,
    *,
    robot_x_m: float,
    robot_y_m: float,
    robot_z_m: float,
    robot_yaw_rad: float,
    wall_x_m: float,
    wall_y_m: float,
    wall_height_m: float,
    camera_forward_offset_m: float,
    camera_lateral_offset_m: float,
    camera_height_m: float,
    camera_yaw_offset_rad: float,
    camera_fx_px: float,
    camera_fy_px: float,
    camera_cx_px: float,
    camera_cy_px: float,
    frame_width: int,
    mask_line_width_px: int,
) -> bool:
    bottom = _project_world_point(
        robot_x_m=robot_x_m,
        robot_y_m=robot_y_m,
        robot_z_m=robot_z_m,
        robot_yaw_rad=robot_yaw_rad,
        point_x_m=wall_x_m,
        point_y_m=wall_y_m,
        point_z_m=0.0,
        camera_forward_offset_m=camera_forward_offset_m,
        camera_lateral_offset_m=camera_lateral_offset_m,
        camera_height_m=camera_height_m,
        camera_yaw_offset_rad=camera_yaw_offset_rad,
        camera_fx_px=camera_fx_px,
        camera_fy_px=camera_fy_px,
        camera_cx_px=camera_cx_px,
        camera_cy_px=camera_cy_px,
    )
    top = _project_world_point(
        robot_x_m=robot_x_m,
        robot_y_m=robot_y_m,
        robot_z_m=robot_z_m,
        robot_yaw_rad=robot_yaw_rad,
        point_x_m=wall_x_m,
        point_y_m=wall_y_m,
        point_z_m=wall_height_m,
        camera_forward_offset_m=camera_forward_offset_m,
        camera_lateral_offset_m=camera_lateral_offset_m,
        camera_height_m=camera_height_m,
        camera_yaw_offset_rad=camera_yaw_offset_rad,
        camera_fx_px=camera_fx_px,
        camera_fy_px=camera_fy_px,
        camera_cx_px=camera_cx_px,
        camera_cy_px=camera_cy_px,
    )
    if bottom is None or top is None:
        return False
    u_px = int(round((bottom[0] + top[0]) / 2.0))
    if not -mask_line_width_px <= u_px < frame_width + mask_line_width_px:
        return False
    cv2.line(
        mask,
        (u_px, int(round(bottom[1]))),
        (u_px, int(round(top[1]))),
        255,
        max(1, int(mask_line_width_px)),
    )
    return True


def build_confirmed_wall_exclusion_mask(
    cv2,
    numpy,
    *,
    scan: PlainLaserScan | None,
    image_stamp_sec: float | None,
    sync_tolerance_sec: float,
    robot_x_m: float,
    robot_y_m: float,
    robot_z_m: float,
    robot_yaw_rad: float,
    frame_width: int,
    frame_height: int,
    camera_fx_px: float,
    camera_fy_px: float,
    camera_cx_px: float,
    camera_cy_px: float,
    camera_forward_offset_m: float,
    camera_lateral_offset_m: float,
    camera_height_m: float,
    camera_yaw_offset_rad: float,
    lidar_forward_offset_m: float = 0.0,
    lidar_lateral_offset_m: float = 0.0,
    expected_scan_frame: str = "base_scan",
    arena: ArenaBounds = ArenaBounds(),
    wall_height_m: float = 0.40,
    wall_thickness_m: float = 0.04,
    wall_range_tolerance_m: float = 0.08,
    foreground_clearance_m: float = 0.06,
    foreground_support_mask=None,
    foreground_support_dilation_px: int = 7,
    mask_line_width_px: int = 7,
    min_confirmed_samples: int = 3,
) -> WallEdgeMaskResult:
    """Build a mask only from map-wall rays confirmed by the nearest scan."""

    if scan is None:
        return WallEdgeMaskResult(None, "synchronized_scan_unavailable", 0, 0, None)
    if (
        expected_scan_frame
        and scan.scan_frame_id
        and scan.scan_frame_id.lstrip("/") != expected_scan_frame.lstrip("/")
    ):
        return WallEdgeMaskResult(None, "unsupported_scan_frame", 0, 0, None)
    scan_delta_sec = None
    if image_stamp_sec is not None and scan.scan_stamp_sec is not None:
        scan_delta_sec = abs(float(image_stamp_sec) - float(scan.scan_stamp_sec))
        if scan_delta_sec > sync_tolerance_sec:
            return WallEdgeMaskResult(None, "scan_image_unsynchronized", 0, 0, scan_delta_sec)
    elif image_stamp_sec is not None:
        return WallEdgeMaskResult(None, "scan_stamp_unavailable", 0, 0, None)
    if frame_width <= 0 or frame_height <= 0:
        return WallEdgeMaskResult(None, "invalid_frame_size", 0, 0, scan_delta_sec)
    if camera_fx_px <= 0.0 or camera_fy_px <= 0.0:
        return WallEdgeMaskResult(None, "invalid_camera_intrinsics", 0, 0, scan_delta_sec)
    if (
        foreground_support_mask is not None
        and foreground_support_mask.shape[:2] != (frame_height, frame_width)
    ):
        return WallEdgeMaskResult(None, "foreground_support_shape_mismatch", 0, 0, scan_delta_sec)

    mask = numpy.zeros((frame_height, frame_width), dtype=numpy.uint8)
    foreground_columns_px: list[float] = []
    occluded_wall_points: list[tuple[float, float]] = []
    confirmed = 0
    protected = 0
    cos_robot = math.cos(robot_yaw_rad)
    sin_robot = math.sin(robot_yaw_rad)
    scan_origin_x = (
        robot_x_m
        + cos_robot * lidar_forward_offset_m
        - sin_robot * lidar_lateral_offset_m
    )
    scan_origin_y = (
        robot_y_m
        + sin_robot * lidar_forward_offset_m
        + cos_robot * lidar_lateral_offset_m
    )
    for index, raw_range in enumerate(scan.ranges):
        try:
            observed_range = float(raw_range)
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(observed_range)
            or observed_range < scan.range_min
            or observed_range > scan.range_max
        ):
            continue
        scan_bearing = scan.angle_min + index * scan.angle_increment
        world_yaw = robot_yaw_rad + scan_bearing
        predicted_wall_range = ray_distance_to_arena_wall(
            origin_x_m=scan_origin_x,
            origin_y_m=scan_origin_y,
            ray_yaw_rad=world_yaw,
            arena=arena,
            wall_thickness_m=wall_thickness_m,
        )
        if predicted_wall_range is None:
            continue
        observed_x = scan_origin_x + observed_range * math.cos(world_yaw)
        observed_y = scan_origin_y + observed_range * math.sin(world_yaw)

        if observed_range < predicted_wall_range - foreground_clearance_m:
            occluded_wall_points.append(
                (
                    scan_origin_x + predicted_wall_range * math.cos(world_yaw),
                    scan_origin_y + predicted_wall_range * math.sin(world_yaw),
                )
            )
            projected = _project_world_point(
                robot_x_m=robot_x_m,
                robot_y_m=robot_y_m,
                robot_z_m=robot_z_m,
                robot_yaw_rad=robot_yaw_rad,
                point_x_m=observed_x,
                point_y_m=observed_y,
                point_z_m=camera_height_m,
                camera_forward_offset_m=camera_forward_offset_m,
                camera_lateral_offset_m=camera_lateral_offset_m,
                camera_height_m=camera_height_m,
                camera_yaw_offset_rad=camera_yaw_offset_rad,
                camera_fx_px=camera_fx_px,
                camera_fy_px=camera_fy_px,
                camera_cx_px=camera_cx_px,
                camera_cy_px=camera_cy_px,
            )
            if projected is not None:
                u_px, _v_px, _depth_m = projected
                foreground_columns_px.append(u_px)
                protected += 1
            continue
        if abs(observed_range - predicted_wall_range) > wall_range_tolerance_m:
            continue

        wall_x = scan_origin_x + predicted_wall_range * math.cos(world_yaw)
        wall_y = scan_origin_y + predicted_wall_range * math.sin(world_yaw)
        if _draw_projected_wall_column(
            cv2,
            mask,
            robot_x_m=robot_x_m,
            robot_y_m=robot_y_m,
            robot_z_m=robot_z_m,
            robot_yaw_rad=robot_yaw_rad,
            wall_x_m=wall_x,
            wall_y_m=wall_y,
            wall_height_m=wall_height_m,
            camera_forward_offset_m=camera_forward_offset_m,
            camera_lateral_offset_m=camera_lateral_offset_m,
            camera_height_m=camera_height_m,
            camera_yaw_offset_rad=camera_yaw_offset_rad,
            camera_fx_px=camera_fx_px,
            camera_fy_px=camera_fy_px,
            camera_cx_px=camera_cx_px,
            camera_cy_px=camera_cy_px,
            frame_width=frame_width,
            mask_line_width_px=mask_line_width_px,
        ):
            confirmed += 1

    if confirmed < max(1, int(min_confirmed_samples)):
        return WallEdgeMaskResult(
            None,
            "insufficient_confirmed_wall_samples",
            confirmed,
            protected,
            scan_delta_sec,
        )
    # Once the same arena wall is confirmed by surrounding rays, project it
    # through foreground-occluded bearings too.  The 2D visual-support mask
    # below carves the actual stand pixels back out of this inferred surface.
    for wall_x, wall_y in occluded_wall_points:
        _draw_projected_wall_column(
            cv2,
            mask,
            robot_x_m=robot_x_m,
            robot_y_m=robot_y_m,
            robot_z_m=robot_z_m,
            robot_yaw_rad=robot_yaw_rad,
            wall_x_m=wall_x,
            wall_y_m=wall_y,
            wall_height_m=wall_height_m,
            camera_forward_offset_m=camera_forward_offset_m,
            camera_lateral_offset_m=camera_lateral_offset_m,
            camera_height_m=camera_height_m,
            camera_yaw_offset_rad=camera_yaw_offset_rad,
            camera_fx_px=camera_fx_px,
            camera_fy_px=camera_fy_px,
            camera_cx_px=camera_cx_px,
            camera_cy_px=camera_cy_px,
            frame_width=frame_width,
            mask_line_width_px=mask_line_width_px,
        )
    kernel_width = max(3, int(mask_line_width_px) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Cover anti-aliased/Canny pixels immediately around the projected wall
    # surface boundaries.  Foreground corridors are carved after this step.
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
        iterations=1,
    )
    if foreground_columns_px and foreground_support_mask is not None:
        projected_beam_spacing_px = max(
            1.0,
            camera_fx_px * math.tan(abs(scan.angle_increment)),
        )
        # The visual-support mask is dilated below, but clipping it against an
        # undilated beam corridor can still erase the outer half of a stand
        # edge exactly at that corridor boundary. Expand the measured interval
        # by the same effective support radius; only actual Canny support is
        # restored, so this does not unmask the whole image column.
        support_radius_px = max(
            0,
            (max(1, int(foreground_support_dilation_px)) + 1) // 2,
        )
        protected_intervals = _cluster_foreground_columns(
            foreground_columns_px,
            max_gap_px=max(4.0, 2.5 * projected_beam_spacing_px),
            padding_px=(
                max(3, int(math.ceil(1.25 * projected_beam_spacing_px)))
                + support_radius_px
            ),
        )
        corridor_mask = numpy.zeros_like(mask)
        for x0, x1 in protected_intervals:
            corridor_mask[:, max(0, x0) : min(frame_width, x1 + 1)] = 255
        support_kernel = max(1, int(foreground_support_dilation_px))
        if support_kernel > 1:
            support_kernel |= 1
            visual_support = cv2.dilate(
                foreground_support_mask,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (support_kernel, support_kernel),
                ),
                iterations=1,
            )
        else:
            visual_support = foreground_support_mask
        protected_pixels = cv2.bitwise_and(corridor_mask, visual_support)
        mask[protected_pixels > 0] = 0
    return WallEdgeMaskResult(mask, "ok", confirmed, protected, scan_delta_sec)
