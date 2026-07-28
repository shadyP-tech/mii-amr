"""Quadrilateral geometry, pose estimation, and debug rendering primitives."""

from __future__ import annotations

import math
from typing import Sequence

from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint,
    StandAxisImageEstimate,
)


def estimate_stand_axis_from_corners(
    corners: Sequence[ImagePoint],
    *,
    min_edge_height_px: float = 8.0,
    stand_width_m: float | None = None,
    stand_distance_m: float | None = None,
    camera_fx_px: float | None = None,
    camera_fy_px: float | None = None,
    camera_cx_px: float | None = None,
    camera_cy_px: float | None = None,
    stand_depth_m: float | None = None,
    stand_head_bottom_height_m: float | None = None,
    cv2=None,
    contour_area_px: float = 0.0,
    source: str = "corners",
) -> StandAxisImageEstimate:
    corners = order_corners(corners)
    top_left, top_right, bottom_right, bottom_left = corners
    left_height = _distance(top_left, bottom_left)
    right_height = _distance(top_right, bottom_right)
    if left_height < min_edge_height_px or right_height < min_edge_height_px:
        return _unusable("edge_too_short", corners=corners, contour_area_px=contour_area_px, source=source)

    ratio = left_height / right_height
    yaw_proxy = (ratio - 1.0) / (ratio + 1.0)
    closer_side = "left" if left_height > right_height else "right" if right_height > left_height else "equal"
    yaw_deg = _yaw_deg_from_square_pnp(
        cv2,
        corners,
        stand_width_m=stand_width_m,
        camera_fx_px=camera_fx_px,
        camera_fy_px=camera_fy_px,
        camera_cx_px=camera_cx_px,
        camera_cy_px=camera_cy_px,
    )
    if yaw_deg is None:
        yaw_deg = _yaw_deg_from_projected_width(corners, stand_width_m, stand_distance_m, camera_fx_px)
    if yaw_deg is None:
        yaw_deg = _yaw_deg_from_ratio(ratio, stand_width_m, stand_distance_m)

    return StandAxisImageEstimate(
        usable=True,
        reason="axis_estimated",
        mode="face_visible",
        corners=corners,
        axis_line=None,
        left_height_px=left_height,
        right_height_px=right_height,
        height_ratio=ratio,
        yaw_proxy=yaw_proxy,
        yaw_deg=yaw_deg,
        closer_side=closer_side,
        contour_area_px=contour_area_px,
        source=source,
    )


def quadrilateral_aspect_ratio(corners: Sequence[ImagePoint]) -> float:
    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    width = (_distance(top_left, top_right) + _distance(bottom_left, bottom_right)) / 2.0
    height = (_distance(top_left, bottom_left) + _distance(top_right, bottom_right)) / 2.0
    if height <= 0.0:
        return 0.0
    return width / height


def score_quadrilateral_candidate(corners: Sequence[ImagePoint], area_px: float) -> float:
    aspect_ratio = quadrilateral_aspect_ratio(corners)
    aspect_score = max(0.0, 1.0 - abs(math.log(max(aspect_ratio, 1e-6))))
    return area_px * (0.5 + 0.5 * aspect_score)


def wide_row_band(row_widths: Sequence[int], *, width_fraction: float = 0.60, max_gap: int = 3) -> tuple[int, int] | None:
    if not row_widths:
        return None
    max_width = max(row_widths)
    if max_width <= 0:
        return None
    threshold = max_width * width_fraction
    best = None
    best_length = -1
    start = None
    last_wide = None
    gap = 0
    for index, width in enumerate(row_widths):
        if width >= threshold:
            if start is None:
                start = index
            last_wide = index
            gap = 0
            continue
        if start is not None:
            gap += 1
            if gap > max_gap:
                end = last_wide if last_wide is not None else index - gap
                length = end - start + 1
                if length > best_length:
                    best = (start, end)
                    best_length = length
                start = None
                last_wide = None
                gap = 0
    if start is not None:
        end = last_wide if last_wide is not None else len(row_widths) - 1
        length = end - start + 1
        if length > best_length:
            best = (start, end)
    return best

def _largest_qr_quad(
    candidates: Sequence[
        tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]
    ],
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    """Select the nearest/largest visible QR instead of detector return order."""

    if not candidates:
        return None
    return max(candidates, key=_polygon_area)


def _scale_quadrilateral_about_center(
    corners: Sequence[ImagePoint],
    scale: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]:
    ordered = order_corners(corners)
    center_u = sum(point.u_px for point in ordered) / 4.0
    center_v = sum(point.v_px for point in ordered) / 4.0
    return order_corners(
        tuple(
            ImagePoint(
                center_u + (point.u_px - center_u) * scale,
                center_v + (point.v_px - center_v) * scale,
            )
            for point in ordered
        )
    )


def _corners_inside_image(corners: Sequence[ImagePoint], image_shape) -> bool:
    height, width = image_shape[:2]
    return all(0.0 <= point.u_px < width and 0.0 <= point.v_px < height for point in corners)


def _debug_outline_image(cv2, image_shape, corners: Sequence[ImagePoint]):
    import numpy

    outline = numpy.zeros(image_shape[:2], dtype=numpy.uint8)
    polygon = numpy.array(
        [[(int(round(point.u_px)), int(round(point.v_px))) for point in order_corners(corners)]],
        dtype=numpy.int32,
    )
    cv2.polylines(outline, polygon, isClosed=True, color=255, thickness=2)
    return outline


def _debug_rectangle_image(cv2, image_shape, corners: Sequence[ImagePoint]):
    return _debug_outline_image(cv2, image_shape, corners)


def _debug_rectangle_overlay_image(
    cv2,
    image_shape,
    corners: Sequence[ImagePoint],
    face_mask,
):
    """Show the accepted rectangle and the cutout that supports it together."""

    import numpy

    overlay = numpy.zeros(image_shape[:2], dtype=numpy.uint8)
    if face_mask is not None:
        overlay[numpy.asarray(face_mask) > 0] = 96
    rectangle = _debug_rectangle_image(cv2, image_shape, corners)
    overlay[rectangle > 0] = 255
    return overlay


def _debug_polygon_edge_cutout_image(
    cv2,
    edges,
    corners: Sequence[ImagePoint],
    *,
    thickness_px: int,
):
    import numpy

    cutout = numpy.zeros(edges.shape[:2], dtype=numpy.uint8)
    boundary = _debug_outline_image(cv2, edges.shape, corners)
    if thickness_px > 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness_px, thickness_px))
        boundary = cv2.dilate(boundary, kernel, iterations=1)
    return cv2.bitwise_and(edges, boundary, dst=cutout)

def estimate_edge_on_axis_from_line(
    start: ImagePoint,
    end: ImagePoint,
    *,
    min_edge_height_px: float = 8.0,
    source: str = "edge_on_line",
) -> StandAxisImageEstimate:
    top = start if start.v_px <= end.v_px else end
    bottom = end if top is start else start
    length_px = _distance(top, bottom)
    if length_px < min_edge_height_px:
        return _unusable("edge_on_line_too_short", source=source, axis_line=(top, bottom))
    return StandAxisImageEstimate(
        usable=True,
        reason="edge_on_approx_90_deg",
        mode="edge_on",
        corners=None,
        axis_line=(top, bottom),
        left_height_px=length_px,
        right_height_px=0.0,
        height_ratio=None,
        yaw_proxy=None,
        yaw_deg=None,
        closer_side="side_on",
        contour_area_px=0.0,
        source=source,
    )

def _polygon_area(corners: Sequence[ImagePoint]) -> float:
    ordered = order_corners(corners)
    area = 0.0
    for current, following in zip(ordered, ordered[1:] + ordered[:1]):
        area += current.u_px * following.v_px - following.u_px * current.v_px
    return abs(area) / 2.0


def _quadrilateral_corners(cv2, contour) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    perimeter = cv2.arcLength(contour, True)
    for epsilon_fraction in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(contour, epsilon_fraction * perimeter, True)
        if len(approx) == 4:
            points = [ImagePoint(float(point[0][0]), float(point[0][1])) for point in approx]
            return order_corners(points)
    return None


def order_corners(points: Sequence[ImagePoint]) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]:
    if len(points) != 4:
        raise ValueError("exactly four points are required")
    ordered_by_y = sorted(points, key=lambda point: (point.v_px, point.u_px))
    top = sorted(ordered_by_y[:2], key=lambda point: point.u_px)
    bottom = sorted(ordered_by_y[2:], key=lambda point: point.u_px)
    top_left, top_right = top
    bottom_left, bottom_right = bottom
    return top_left, top_right, bottom_right, bottom_left


def _yaw_deg_from_ratio(
    ratio: float,
    stand_width_m: float | None,
    stand_distance_m: float | None,
) -> float | None:
    if stand_width_m is None or stand_distance_m is None:
        return None
    if stand_width_m <= 0.0 or stand_distance_m <= 0.0:
        return None
    sin_yaw = (2.0 * stand_distance_m / stand_width_m) * ((ratio - 1.0) / (ratio + 1.0))
    if abs(sin_yaw) > 1.0:
        return None
    return math.degrees(math.asin(sin_yaw))


def _yaw_deg_from_square_pnp(
    cv2,
    corners: Sequence[ImagePoint],
    *,
    stand_width_m: float | None,
    camera_fx_px: float | None,
    camera_fy_px: float | None,
    camera_cx_px: float | None,
    camera_cy_px: float | None,
) -> float | None:
    if cv2 is None:
        return None
    if (
        stand_width_m is None
        or camera_fx_px is None
        or camera_fy_px is None
        or camera_cx_px is None
        or camera_cy_px is None
    ):
        return None
    if stand_width_m <= 0.0 or camera_fx_px <= 0.0 or camera_fy_px <= 0.0:
        return None
    try:
        import numpy
    except ImportError:
        return None

    half = stand_width_m / 2.0
    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    camera_matrix = numpy.array(
        [
            [camera_fx_px, 0.0, camera_cx_px],
            [0.0, camera_fy_px, camera_cy_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=numpy.float64,
    )
    distortion = numpy.zeros((4, 1), dtype=numpy.float64)

    def yaw_from_rotation_vector(rvec) -> float | None:
        rotation, _jacobian = cv2.Rodrigues(rvec)
        normal = rotation @ numpy.array(
            [[0.0], [0.0], [1.0]],
            dtype=numpy.float64,
        )
        normal_x = float(normal[0, 0])
        normal_z = float(normal[2, 0])
        if not math.isfinite(normal_x) or not math.isfinite(normal_z):
            return None
        # OpenCV optical +x points to image-right.  The public stand-axis
        # convention used by yaw_proxy, the metric fallbacks, ROS map yaw, and
        # the viewpoint planner is positive image-left / counterclockwise.
        # Convert handedness here so every estimator branch exposes the same
        # signed quantity to both the real viewer and simulation observer.
        return -math.degrees(math.atan2(normal_x, abs(normal_z)))

    # IPPE_SQUARE is specialized for a four-point coplanar square and returns
    # both planar-pose solutions.  Its object/image point order is fixed; the
    # image order below is therefore BL, BR, TR, TL rather than our normal
    # display order.  Select the physically visible solution with the lowest
    # measured reprojection error.
    if (
        hasattr(cv2, "solvePnPGeneric")
        and hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
    ):
        square_object_points = numpy.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=numpy.float64,
        )
        square_image_points = numpy.array(
            [
                [bottom_left.u_px, bottom_left.v_px],
                [bottom_right.u_px, bottom_right.v_px],
                [top_right.u_px, top_right.v_px],
                [top_left.u_px, top_left.v_px],
            ],
            dtype=numpy.float64,
        )
        try:
            generic_result = cv2.solvePnPGeneric(
                square_object_points,
                square_image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except Exception:
            generic_result = ()
        if generic_result and bool(generic_result[0]):
            best_pose = None
            rvecs = generic_result[1]
            tvecs = generic_result[2]
            for rvec, tvec in zip(rvecs, tvecs):
                tvec = numpy.asarray(tvec, dtype=numpy.float64).reshape(3, 1)
                if not math.isfinite(float(tvec[2, 0])) or float(tvec[2, 0]) <= 0.0:
                    continue
                try:
                    projected, _jacobian = cv2.projectPoints(
                        square_object_points,
                        rvec,
                        tvec,
                        camera_matrix,
                        distortion,
                    )
                except Exception:
                    continue
                residual = projected.reshape(-1, 2) - square_image_points
                reprojection_rmse = math.sqrt(
                    float(numpy.mean(numpy.sum(residual * residual, axis=1)))
                )
                yaw_deg = yaw_from_rotation_vector(rvec)
                if yaw_deg is None or not math.isfinite(reprojection_rmse):
                    continue
                if best_pose is None or reprojection_rmse < best_pose[0]:
                    best_pose = (reprojection_rmse, yaw_deg)
            if best_pose is not None:
                return best_pose[1]

    # Compatibility fallback for OpenCV builds without IPPE_SQUARE.
    object_points = numpy.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=numpy.float64,
    )
    image_points = numpy.array(
        [
            [top_left.u_px, top_left.v_px],
            [top_right.u_px, top_right.v_px],
            [bottom_right.u_px, bottom_right.v_px],
            [bottom_left.u_px, bottom_left.v_px],
        ],
        dtype=numpy.float64,
    )
    try:
        ok, rvec, _tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception:
        return None
    if not ok:
        return None
    return yaw_from_rotation_vector(rvec)


def _yaw_deg_from_projected_width(
    corners: Sequence[ImagePoint],
    stand_width_m: float | None,
    stand_distance_m: float | None,
    camera_fx_px: float | None,
) -> float | None:
    if stand_width_m is None or stand_distance_m is None or camera_fx_px is None:
        return None
    if stand_width_m <= 0.0 or stand_distance_m <= 0.0 or camera_fx_px <= 0.0:
        return None
    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    top_width = _distance(top_left, top_right)
    bottom_width = _distance(bottom_left, bottom_right)
    observed_width_px = (top_width + bottom_width) / 2.0
    expected_front_width_px = camera_fx_px * stand_width_m / stand_distance_m
    if observed_width_px <= 0.0 or expected_front_width_px <= 0.0:
        return None
    cos_yaw = observed_width_px / expected_front_width_px
    if cos_yaw > 1.0:
        return None
    cos_yaw = max(0.0, cos_yaw)
    magnitude_deg = math.degrees(math.acos(cos_yaw))

    left_height = _distance(top_left, bottom_left)
    right_height = _distance(top_right, bottom_right)
    height_proxy = (left_height - right_height) / max(left_height + right_height, 1e-6)
    top_center_x = (top_left.u_px + top_right.u_px) / 2.0
    bottom_center_x = (bottom_left.u_px + bottom_right.u_px) / 2.0
    shear_proxy = (bottom_center_x - top_center_x) / max(observed_width_px, 1e-6)
    sign_cue = height_proxy if abs(height_proxy) >= 0.01 else shear_proxy
    sign = -1.0 if sign_cue < 0.0 else 1.0
    return sign * magnitude_deg


def _distance(first: ImagePoint, second: ImagePoint) -> float:
    return math.hypot(second.u_px - first.u_px, second.v_px - first.v_px)


def _points_to_cv2(corners: Sequence[ImagePoint]):
    import numpy

    return numpy.array([[[point.u_px, point.v_px]] for point in corners], dtype=numpy.float32)


def _unusable(
    reason: str,
    *,
    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None = None,
    axis_line: tuple[ImagePoint, ImagePoint] | None = None,
    contour_area_px: float = 0.0,
    source: str = "unknown",
) -> StandAxisImageEstimate:
    return StandAxisImageEstimate(
        usable=False,
        reason=reason,
        mode="unavailable",
        corners=corners,
        axis_line=axis_line,
        left_height_px=0.0,
        right_height_px=0.0,
        height_ratio=None,
        yaw_proxy=None,
        yaw_deg=None,
        closer_side=None,
        contour_area_px=contour_area_px,
        source=source,
    )
