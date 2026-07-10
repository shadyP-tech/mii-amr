from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ImagePoint:
    u_px: float
    v_px: float


@dataclass(frozen=True)
class StandAxisImageEstimate:
    usable: bool
    reason: str
    mode: str
    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None
    axis_line: tuple[ImagePoint, ImagePoint] | None
    left_height_px: float
    right_height_px: float
    height_ratio: float | None
    yaw_proxy: float | None
    yaw_deg: float | None
    closer_side: str | None
    contour_area_px: float
    source: str = "unknown"


@dataclass(frozen=True)
class StandAxisEdgeDebugArtifacts:
    edges: object
    face_mask: object | None = None
    rectangle_mask: object | None = None


@dataclass(frozen=True)
class _SilhouetteFaceCandidate:
    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]
    face_mask: object


def estimate_stand_axis_from_mask(
    cv2,
    mask,
    *,
    min_area_px: float = 250.0,
    min_edge_height_px: float = 8.0,
    stand_width_m: float | None = None,
    stand_distance_m: float | None = None,
    camera_fx_px: float | None = None,
    camera_fy_px: float | None = None,
    camera_cx_px: float | None = None,
    camera_cy_px: float | None = None,
    stand_depth_m: float | None = None,
    stand_head_bottom_height_m: float | None = None,
) -> StandAxisImageEstimate:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _unusable("no_contour", source="color_mask")

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area_px:
        return _unusable("contour_too_small", contour_area_px=area, source="color_mask")

    corners = _quadrilateral_corners(cv2, contour)
    if corners is None:
        return _unusable("no_four_corner_contour", contour_area_px=area, source="color_mask")

    return estimate_stand_axis_from_corners(
        corners,
        min_edge_height_px=min_edge_height_px,
        stand_width_m=stand_width_m,
        stand_distance_m=stand_distance_m,
        camera_fx_px=camera_fx_px,
        camera_fy_px=camera_fy_px,
        camera_cx_px=camera_cx_px,
        camera_cy_px=camera_cy_px,
        stand_depth_m=stand_depth_m,
        stand_head_bottom_height_m=stand_head_bottom_height_m,
        cv2=cv2,
        contour_area_px=area,
        source="color_mask",
    )


def estimate_stand_axis_from_edges(
    cv2,
    frame,
    *,
    edge_preprocess: str = "outer_border",
    blur_kernel: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    dilate_iterations: int = 1,
    close_kernel: int = 5,
    close_iterations: int = 1,
    hough_threshold: int = 20,
    hough_min_line_length_px: int = 12,
    hough_max_line_gap_px: int = 8,
    min_boundary_line_length_px: float = 35.0,
    face_width_fraction: float = 0.60,
    min_face_area_fraction: float = 0.25,
    min_area_px: float = 250.0,
    min_edge_height_px: float = 8.0,
    min_aspect_ratio: float = 0.45,
    max_aspect_ratio: float = 1.80,
    front_face_to_qr_width_ratio: float | None = None,
    stand_width_m: float | None = None,
    stand_distance_m: float | None = None,
    camera_fx_px: float | None = None,
    camera_fy_px: float | None = None,
    camera_cx_px: float | None = None,
    camera_cy_px: float | None = None,
    stand_depth_m: float | None = None,
    stand_head_bottom_height_m: float | None = None,
) -> tuple[StandAxisImageEstimate, StandAxisEdgeDebugArtifacts]:
    edge_input = _edge_input_image(cv2, frame, edge_preprocess=edge_preprocess, blur_kernel=blur_kernel)
    frame_height, frame_width = frame.shape[:2]
    effective_camera_fy_px = camera_fy_px if camera_fy_px is not None else camera_fx_px
    effective_camera_cx_px = camera_cx_px if camera_cx_px is not None else (frame_width - 1.0) / 2.0
    effective_camera_cy_px = camera_cy_px if camera_cy_px is not None else (frame_height - 1.0) / 2.0
    edges = cv2.Canny(edge_input, canny_low, canny_high)
    if dilate_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)
    if close_kernel > 1 and close_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    adaptive_min_area_px = max(
        min_area_px,
        _largest_external_bounding_area(cv2, edges) * min_face_area_fraction,
    )

    qr_front_face = _front_face_from_qr_geometry(
        cv2,
        frame,
        edges,
        width_ratio=front_face_to_qr_width_ratio,
        min_area_px=adaptive_min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )
    if qr_front_face is not None:
        return (
            estimate_stand_axis_from_corners(
                qr_front_face.corners,
                min_edge_height_px=min_edge_height_px,
                stand_width_m=stand_width_m,
                stand_distance_m=stand_distance_m,
                camera_fx_px=camera_fx_px,
                camera_fy_px=effective_camera_fy_px,
                camera_cx_px=effective_camera_cx_px,
                camera_cy_px=effective_camera_cy_px,
                stand_depth_m=stand_depth_m,
                stand_head_bottom_height_m=stand_head_bottom_height_m,
                cv2=cv2,
                contour_area_px=_polygon_area(qr_front_face.corners),
                source="edge_qr_scaled_front",
            ),
            StandAxisEdgeDebugArtifacts(
                edges=edges,
                face_mask=qr_front_face.face_mask,
                rectangle_mask=_debug_rectangle_image(cv2, edges.shape, qr_front_face.corners),
            ),
        )

    plain_face = _plain_face_from_stem_cropped_edges(
        cv2,
        edges,
        min_area_px=adaptive_min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )
    if plain_face is not None:
        return (
            estimate_stand_axis_from_corners(
                plain_face.corners,
                min_edge_height_px=min_edge_height_px,
                stand_width_m=stand_width_m,
                stand_distance_m=stand_distance_m,
                camera_fx_px=camera_fx_px,
                camera_fy_px=effective_camera_fy_px,
                camera_cx_px=effective_camera_cx_px,
                camera_cy_px=effective_camera_cy_px,
                stand_depth_m=stand_depth_m,
                stand_head_bottom_height_m=stand_head_bottom_height_m,
                cv2=cv2,
                contour_area_px=_polygon_area(plain_face.corners),
                source="edge_plain_face_stem_anchor",
            ),
            StandAxisEdgeDebugArtifacts(
                edges=edges,
                face_mask=plain_face.face_mask,
                rectangle_mask=_debug_rectangle_image(cv2, edges.shape, plain_face.corners),
            ),
        )

    stem_face = _stem_anchored_face_from_edges(
        cv2,
        edges,
        min_area_px=adaptive_min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )
    if stem_face is not None:
        return (
            estimate_stand_axis_from_corners(
                stem_face.corners,
                min_edge_height_px=min_edge_height_px,
                stand_width_m=stand_width_m,
                stand_distance_m=stand_distance_m,
                camera_fx_px=camera_fx_px,
                camera_fy_px=effective_camera_fy_px,
                camera_cx_px=effective_camera_cx_px,
                camera_cy_px=effective_camera_cy_px,
                stand_depth_m=stand_depth_m,
                stand_head_bottom_height_m=stand_head_bottom_height_m,
                cv2=cv2,
                contour_area_px=_polygon_area(stem_face.corners),
                source="edge_stem_anchor",
            ),
            StandAxisEdgeDebugArtifacts(
                edges=edges,
                face_mask=stem_face.face_mask,
                rectangle_mask=_debug_rectangle_image(cv2, edges.shape, stem_face.corners),
            ),
        )

    silhouette_face = _face_quadrilateral_from_silhouette(
        cv2,
        edges,
        min_area_px=adaptive_min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        face_width_fraction=face_width_fraction,
    )
    if silhouette_face is not None:
        return (
            estimate_stand_axis_from_corners(
                silhouette_face.corners,
                min_edge_height_px=min_edge_height_px,
                stand_width_m=stand_width_m,
                stand_distance_m=stand_distance_m,
                camera_fx_px=camera_fx_px,
                camera_fy_px=effective_camera_fy_px,
                camera_cx_px=effective_camera_cx_px,
                camera_cy_px=effective_camera_cy_px,
                stand_depth_m=stand_depth_m,
                stand_head_bottom_height_m=stand_head_bottom_height_m,
                cv2=cv2,
                contour_area_px=_polygon_area(silhouette_face.corners),
                source="edge_silhouette",
            ),
            StandAxisEdgeDebugArtifacts(
                edges=edges,
                face_mask=silhouette_face.face_mask,
                rectangle_mask=_debug_rectangle_image(cv2, edges.shape, silhouette_face.corners),
            ),
        )

    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best: StandAxisImageEstimate | None = None
    best_score = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < adaptive_min_area_px:
            continue
        corners = _quadrilateral_corners(cv2, contour)
        if corners is None or not cv2.isContourConvex(_points_to_cv2(corners)):
            continue
        if _contour_has_lower_appendage(cv2, contour, corners):
            continue
        aspect_ratio = quadrilateral_aspect_ratio(corners)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        estimate = estimate_stand_axis_from_corners(
            corners,
            min_edge_height_px=min_edge_height_px,
            stand_width_m=stand_width_m,
            stand_distance_m=stand_distance_m,
            camera_fx_px=camera_fx_px,
            camera_fy_px=effective_camera_fy_px,
            camera_cx_px=effective_camera_cx_px,
            camera_cy_px=effective_camera_cy_px,
            stand_depth_m=stand_depth_m,
            stand_head_bottom_height_m=stand_head_bottom_height_m,
            cv2=cv2,
            contour_area_px=area,
            source="edges",
        )
        if not estimate.usable:
            continue
        score = score_quadrilateral_candidate(corners, area)
        if score > best_score:
            best = estimate
            best_score = score

    if best is not None:
        return best, StandAxisEdgeDebugArtifacts(edges=edges)

    line_corners = _quadrilateral_from_line_segments(
        cv2,
        edges,
        hough_threshold=hough_threshold,
        hough_min_line_length_px=hough_min_line_length_px,
        hough_max_line_gap_px=hough_max_line_gap_px,
        min_boundary_line_length_px=min_boundary_line_length_px,
        min_edge_height_px=min_edge_height_px,
        min_area_px=adaptive_min_area_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )
    if line_corners is not None:
        return (
            estimate_stand_axis_from_corners(
                line_corners,
                min_edge_height_px=min_edge_height_px,
                stand_width_m=stand_width_m,
                stand_distance_m=stand_distance_m,
                camera_fx_px=camera_fx_px,
                camera_fy_px=effective_camera_fy_px,
                camera_cx_px=effective_camera_cx_px,
                camera_cy_px=effective_camera_cy_px,
                stand_depth_m=stand_depth_m,
                stand_head_bottom_height_m=stand_head_bottom_height_m,
                cv2=cv2,
                contour_area_px=_polygon_area(line_corners),
                source="edge_lines",
            ),
            StandAxisEdgeDebugArtifacts(edges=edges),
        )

    edge_on = _edge_on_from_line_segments(
        cv2,
        edges,
        hough_threshold=hough_threshold,
        hough_min_line_length_px=hough_min_line_length_px,
        hough_max_line_gap_px=hough_max_line_gap_px,
        min_boundary_line_length_px=min_boundary_line_length_px,
        min_edge_height_px=min_edge_height_px,
    )
    if edge_on is not None:
        return edge_on, StandAxisEdgeDebugArtifacts(edges=edges)

    if best is None:
        return _unusable("no_edge_quadrilateral", source="edges"), StandAxisEdgeDebugArtifacts(edges=edges)


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


def _front_face_from_qr_geometry(
    cv2,
    frame,
    edges,
    *,
    width_ratio: float | None,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> _SilhouetteFaceCandidate | None:
    if width_ratio is None or width_ratio <= 1.0:
        return None
    qr_corners = _detect_qr_quad_corners(cv2, frame)
    if qr_corners is None:
        return None
    corners = _scale_quadrilateral_about_center(qr_corners, width_ratio)
    area = _polygon_area(corners)
    if area < min_area_px:
        return None
    if not _corners_inside_image(corners, edges.shape):
        return None
    aspect_ratio = quadrilateral_aspect_ratio(corners)
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return None
    estimate = estimate_stand_axis_from_corners(
        corners,
        min_edge_height_px=min_edge_height_px,
        contour_area_px=area,
        source="edge_qr_scaled_front",
    )
    if not estimate.usable:
        return None

    return _SilhouetteFaceCandidate(
        corners=corners,
        face_mask=_debug_outline_image(cv2, edges.shape, corners),
    )


def _detect_qr_quad_corners(cv2, frame) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    detector = cv2.QRCodeDetector()
    try:
        ok, points = detector.detect(frame)
    except Exception:
        ok, points = False, None
    if not ok or points is None:
        try:
            multi_result = detector.detectAndDecodeMulti(frame)
        except Exception:
            multi_result = ()
        points = multi_result[2] if len(multi_result) > 2 else None
        if points is None or len(points) == 0:
            return None
        points = points[0]
    try:
        flat_points = points.reshape(-1, 2)
    except Exception:
        return None
    if len(flat_points) < 4:
        return None
    return order_corners(tuple(ImagePoint(float(point[0]), float(point[1])) for point in flat_points[:4]))


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


def _expanded_head_edge_roi(
    cv2,
    edges,
    rough_corners: Sequence[ImagePoint],
    *,
    margin_px: int,
    stem_center_x: float | None = None,
    stem_top_y: float | None = None,
    min_edge_height_px: float = 8.0,
):
    import numpy

    ordered = order_corners(rough_corners)
    min_x = min(point.u_px for point in ordered)
    max_x = max(point.u_px for point in ordered)
    min_y = min(point.v_px for point in ordered)
    max_y = max(point.v_px for point in ordered)
    width = max_x - min_x
    height = max_y - min_y
    horizontal_margin = max(float(margin_px), 0.18 * width)
    top_margin = max(float(margin_px), 0.35 * height)
    bottom_margin = max(float(margin_px), 0.14 * height)

    x_min = max(0, int(math.floor(min_x - horizontal_margin)))
    x_max = min(edges.shape[1], int(math.ceil(max_x + horizontal_margin)) + 1)
    y_min = max(0, int(math.floor(min_y - top_margin)))
    y_max = min(edges.shape[0], int(math.ceil(max_y + bottom_margin)) + 1)

    edge_roi = numpy.zeros(edges.shape[:2], dtype=numpy.uint8)
    if x_max <= x_min or y_max <= y_min:
        return edge_roi
    edge_roi[y_min:y_max, x_min:x_max] = edges[y_min:y_max, x_min:x_max]

    if stem_center_x is None or stem_top_y is None:
        return edge_roi

    rough_bottom_y = max_y
    erase_from_y = int(round(min(rough_bottom_y + max(2.0, 0.25 * min_edge_height_px), stem_top_y)))
    if erase_from_y >= edge_roi.shape[0]:
        return edge_roi

    stem_half_width = max(5, int(round(min_edge_height_px * 1.2)))
    stem_x = int(round(stem_center_x))
    erase_left = max(0, stem_x - stem_half_width)
    erase_right = min(edge_roi.shape[1], stem_x + stem_half_width + 1)
    if erase_left < erase_right:
        edge_roi[erase_from_y:, erase_left:erase_right] = 0
    return edge_roi


def _connected_border_mask_and_corners(
    cv2,
    edges,
    edge_cutout,
    *,
    fallback_corners: Sequence[ImagePoint],
    min_edge_height_px: float,
):
    import numpy

    line_fit_corners = _cutout_outer_border_line_corners(
        cv2,
        edge_cutout,
        min_edge_height_px=min_edge_height_px,
    )
    if line_fit_corners is not None:
        return edge_cutout, line_fit_corners

    cutout_rect_corners = _cutout_min_area_rect_corners(
        cv2,
        edge_cutout,
        min_edge_height_px=min_edge_height_px,
    )
    if cutout_rect_corners is not None:
        return edge_cutout, cutout_rect_corners

    hull_corners = _outer_hull_corners(cv2, edge_cutout, min_edge_height_px=min_edge_height_px)
    if hull_corners is not None:
        return edge_cutout, hull_corners

    row_bounds = []
    max_span = 0.0
    for y_px in range(edge_cutout.shape[0]):
        columns = numpy.flatnonzero(edge_cutout[y_px, :])
        if len(columns) < 2:
            continue
        left = float(columns[0])
        right = float(columns[-1])
        span = right - left
        if span <= 0.0:
            continue
        row_bounds.append((float(y_px), left, right, span))
        max_span = max(max_span, span)

    if len(row_bounds) < 4 or max_span < min_edge_height_px:
        return edge_cutout, order_corners(fallback_corners)

    side_rows = [row for row in row_bounds if row[3] >= max(min_edge_height_px, 0.38 * max_span)]
    if len(side_rows) < 4:
        return edge_cutout, order_corners(fallback_corners)

    ys = numpy.array([row[0] for row in side_rows], dtype=numpy.float64)
    lefts = numpy.array([row[1] for row in side_rows], dtype=numpy.float64)
    rights = numpy.array([row[2] for row in side_rows], dtype=numpy.float64)
    spans = numpy.array([row[3] for row in side_rows], dtype=numpy.float64)
    broad_ys = ys[spans >= max(min_edge_height_px, 0.62 * max_span)]
    if len(broad_ys) >= 2:
        top_y = float(broad_ys.min())
        bottom_y = float(broad_ys.max())
    else:
        top_y = float(ys.min())
        bottom_y = float(ys.max())
    if bottom_y - top_y < min_edge_height_px:
        return edge_cutout, order_corners(fallback_corners)

    left_line = _fit_x_line_at_ys(ys, lefts)
    right_line = _fit_x_line_at_ys(ys, rights)
    top_line, bottom_line = _fit_top_bottom_y_lines(edge_cutout, min_edge_height_px=min_edge_height_px)
    if left_line is not None and right_line is not None and top_line is not None and bottom_line is not None:
        top_left = _intersect_x_of_y_line_with_y_of_x_line(left_line, top_line)
        top_right = _intersect_x_of_y_line_with_y_of_x_line(right_line, top_line)
        bottom_right = _intersect_x_of_y_line_with_y_of_x_line(right_line, bottom_line)
        bottom_left = _intersect_x_of_y_line_with_y_of_x_line(left_line, bottom_line)
        if None not in (top_left, top_right, bottom_right, bottom_left):
            corners = order_corners((top_left, top_right, bottom_right, bottom_left))
        else:
            corners = _outer_row_envelope_corners(
                fallback_corners,
                left_line,
                right_line,
                top_y=top_y,
                bottom_y=bottom_y,
                min_edge_height_px=min_edge_height_px,
            )
    else:
        corners = _outer_row_envelope_corners(
            fallback_corners,
            left_line,
            right_line,
            top_y=top_y,
            bottom_y=bottom_y,
            min_edge_height_px=min_edge_height_px,
        )
    if corners is None:
        return edge_cutout, order_corners(fallback_corners)

    return edge_cutout, corners


def _edge_pixels_inside_polygon(cv2, edges, corners: Sequence[ImagePoint], *, margin_px: int = 0):
    import numpy

    mask = numpy.zeros(edges.shape[:2], dtype=numpy.uint8)
    polygon = numpy.array(
        [[(int(round(point.u_px)), int(round(point.v_px))) for point in corners]],
        dtype=numpy.int32,
    )
    cv2.fillPoly(mask, polygon, 255)
    if margin_px > 0:
        kernel_size = max(1, int(margin_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.bitwise_and(edges, mask)


def _cutout_outer_border_line_corners(
    cv2,
    edge_cutout,
    *,
    min_edge_height_px: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    import numpy

    row_bounds = []
    max_span = 0.0
    for y_px in range(edge_cutout.shape[0]):
        columns = numpy.flatnonzero(edge_cutout[y_px, :])
        if len(columns) < 2:
            continue
        left = float(columns[0])
        right = float(columns[-1])
        span = right - left
        if span <= 0.0:
            continue
        row_bounds.append((float(y_px), left, right, span))
        max_span = max(max_span, span)
    if len(row_bounds) < 4 or max_span < min_edge_height_px:
        return None

    side_rows = [row for row in row_bounds if row[3] >= max(min_edge_height_px, 0.36 * max_span)]
    if len(side_rows) < 4:
        return None

    ys = numpy.array([row[0] for row in side_rows], dtype=numpy.float64)
    lefts = numpy.array([row[1] for row in side_rows], dtype=numpy.float64)
    rights = numpy.array([row[2] for row in side_rows], dtype=numpy.float64)
    spans = numpy.array([row[3] for row in side_rows], dtype=numpy.float64)
    broad_ys = ys[spans >= max(min_edge_height_px, 0.58 * max_span)]
    if len(broad_ys) < 2:
        return None
    top_y = float(broad_ys.min())
    bottom_y = float(broad_ys.max())
    if bottom_y - top_y < min_edge_height_px:
        return None

    top_band_px = max(3.0, min_edge_height_px * 0.55)
    bottom_band_px = max(3.0, min_edge_height_px * 0.65)
    min_horizontal_length_px = max(min_edge_height_px, 0.16 * max_span)
    top_line = _fit_y_line_from_border_segments(
        cv2,
        edge_cutout,
        target_y=top_y,
        band_px=top_band_px,
        min_length_px=min_horizontal_length_px,
        prefer_lower=False,
    )
    if top_line is None:
        top_line = _fit_y_line_from_extreme_column_points(
            edge_cutout,
            target_y=top_y,
            band_px=top_band_px,
            use_top=True,
        )
    bottom_line = _fit_y_line_from_border_segments(
        cv2,
        edge_cutout,
        target_y=bottom_y,
        band_px=bottom_band_px,
        min_length_px=max(6.0, 0.11 * max_span),
        prefer_lower=True,
    )
    if bottom_line is None:
        bottom_line = _fit_y_line_from_extreme_column_points(
            edge_cutout,
            target_y=bottom_y,
            band_px=bottom_band_px,
            use_top=False,
        )
    vertical_margin = max(3.0, 0.16 * (bottom_y - top_y))
    side_fit_rows = [
        row
        for row in side_rows
        if top_y + vertical_margin <= row[0] <= bottom_y - vertical_margin
    ]
    if len(side_fit_rows) < 4:
        side_fit_rows = side_rows
    side_ys = numpy.array([row[0] for row in side_fit_rows], dtype=numpy.float64)
    side_lefts = numpy.array([row[1] for row in side_fit_rows], dtype=numpy.float64)
    side_rights = numpy.array([row[2] for row in side_fit_rows], dtype=numpy.float64)
    left_line = _fit_x_line_at_ys(side_ys, side_lefts)
    right_line = _fit_x_line_at_ys(side_ys, side_rights)
    if left_line is None or right_line is None or top_line is None or bottom_line is None:
        return None

    top_left = _intersect_x_of_y_line_with_y_of_x_line(left_line, top_line)
    top_right = _intersect_x_of_y_line_with_y_of_x_line(right_line, top_line)
    bottom_right = _intersect_x_of_y_line_with_y_of_x_line(right_line, bottom_line)
    bottom_left = _intersect_x_of_y_line_with_y_of_x_line(left_line, bottom_line)
    if None in (top_left, top_right, bottom_right, bottom_left):
        return None
    corners = order_corners((top_left, top_right, bottom_right, bottom_left))
    top_left, top_right, bottom_right, bottom_left = corners
    if min(_distance(top_left, top_right), _distance(bottom_left, bottom_right)) < min_edge_height_px:
        return None
    if min(_distance(top_left, bottom_left), _distance(top_right, bottom_right)) < min_edge_height_px:
        return None
    return corners


def _fit_y_line_from_border_segments(
    cv2,
    edge_cutout,
    *,
    target_y: float,
    band_px: float,
    min_length_px: float,
    prefer_lower: bool,
) -> tuple[float, float] | None:
    import numpy

    if cv2 is None:
        return None
    hough_min_length = max(5, int(round(min_length_px)))
    segments = _line_segments_from_edges(
        cv2,
        edge_cutout,
        hough_threshold=8,
        hough_min_line_length_px=hough_min_length,
        hough_max_line_gap_px=max(4, int(round(band_px))),
    )
    if not segments:
        return None

    candidates = []
    for segment in segments:
        if abs(segment.angle_deg) > 38.0:
            continue
        y_mid = (segment.start.v_px + segment.end.v_px) / 2.0
        distance_from_target = abs(y_mid - target_y)
        if distance_from_target > band_px:
            continue
        if segment.length_px < min_length_px:
            continue
        y_bias = y_mid - target_y if prefer_lower else target_y - y_mid
        score = segment.length_px - 2.0 * distance_from_target + 0.35 * max(0.0, y_bias)
        candidates.append((score, segment))
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [segment for _score, segment in candidates[:4]]
    points = []
    for segment in selected:
        points.append((segment.start.u_px, segment.start.v_px))
        points.append((segment.end.u_px, segment.end.v_px))
    if len(points) < 4:
        return None

    xs = numpy.array([point[0] for point in points], dtype=numpy.float64)
    ys = numpy.array([point[1] for point in points], dtype=numpy.float64)
    if float(xs.max() - xs.min()) < max(3.0, min_length_px):
        return None
    slope, intercept = numpy.polyfit(xs, ys, 1)
    if not math.isfinite(float(slope)) or not math.isfinite(float(intercept)):
        return None
    return float(slope), float(intercept)


def _fit_y_line_from_extreme_column_points(
    edge_cutout,
    *,
    target_y: float,
    band_px: float,
    use_top: bool,
) -> tuple[float, float] | None:
    import numpy

    points = []
    lower = target_y - band_px
    upper = target_y + band_px
    for x_px in range(edge_cutout.shape[1]):
        rows = numpy.flatnonzero(edge_cutout[:, x_px])
        if len(rows) == 0:
            continue
        y_px = float(rows[0] if use_top else rows[-1])
        if lower <= y_px <= upper:
            points.append((float(x_px), y_px))
    if len(points) < 4:
        return None
    xs = numpy.array([point[0] for point in points], dtype=numpy.float64)
    ys = numpy.array([point[1] for point in points], dtype=numpy.float64)
    if float(xs.max() - xs.min()) < 1e-6:
        return None
    slope, intercept = numpy.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _cutout_min_area_rect_corners(
    cv2,
    edge_cutout,
    *,
    min_edge_height_px: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    points = cv2.findNonZero(edge_cutout)
    if points is None or len(points) < 4:
        return None
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    corners = order_corners(tuple(ImagePoint(float(point[0]), float(point[1])) for point in box))
    top_left, top_right, bottom_right, bottom_left = corners
    if min(_distance(top_left, top_right), _distance(bottom_left, bottom_right)) < min_edge_height_px:
        return None
    if min(_distance(top_left, bottom_left), _distance(top_right, bottom_right)) < min_edge_height_px:
        return None
    return corners


def _outer_hull_corners(
    cv2,
    edge_cutout,
    *,
    min_edge_height_px: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    points = cv2.findNonZero(edge_cutout)
    if points is None or len(points) < 4:
        return None
    hull = cv2.convexHull(points)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0.0:
        return None
    for epsilon_fraction in (0.01, 0.015, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, epsilon_fraction * perimeter, True)
        if len(approx) == 4:
            corners = order_corners(
                tuple(ImagePoint(float(point[0][0]), float(point[0][1])) for point in approx)
            )
            top_left, top_right, bottom_right, bottom_left = corners
            if min(_distance(top_left, top_right), _distance(bottom_left, bottom_right)) < min_edge_height_px:
                continue
            if min(_distance(top_left, bottom_left), _distance(top_right, bottom_right)) < min_edge_height_px:
                continue
            return corners
    return None


def _outer_row_envelope_corners(
    fallback_corners: Sequence[ImagePoint],
    left_line: tuple[float, float] | None,
    right_line: tuple[float, float] | None,
    *,
    top_y: float,
    bottom_y: float,
    min_edge_height_px: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    if left_line is None or right_line is None:
        return order_corners(fallback_corners)
    left_at_top = left_line[0] * top_y + left_line[1]
    left_at_bottom = left_line[0] * bottom_y + left_line[1]
    right_at_top = right_line[0] * top_y + right_line[1]
    right_at_bottom = right_line[0] * bottom_y + right_line[1]
    if min(right_at_top - left_at_top, right_at_bottom - left_at_bottom) < min_edge_height_px:
        return None
    return order_corners(
        (
            ImagePoint(left_at_top, top_y),
            ImagePoint(right_at_top, top_y),
            ImagePoint(right_at_bottom, bottom_y),
            ImagePoint(left_at_bottom, bottom_y),
        )
    )


def _fit_x_line_at_ys(ys, xs) -> tuple[float, float] | None:
    import numpy

    if len(ys) < 2:
        return None
    if abs(float(ys[-1]) - float(ys[0])) < 1e-6:
        return 0.0, float(xs.mean())
    slope, intercept = numpy.polyfit(ys, xs, 1)
    return float(slope), float(intercept)


def _fit_top_bottom_y_lines(edge_cutout, *, min_edge_height_px: float):
    import numpy

    column_bounds = []
    max_span = 0.0
    for x_px in range(edge_cutout.shape[1]):
        rows = numpy.flatnonzero(edge_cutout[:, x_px])
        if len(rows) < 2:
            continue
        top = float(rows[0])
        bottom = float(rows[-1])
        span = bottom - top
        if span <= 0.0:
            continue
        column_bounds.append((float(x_px), top, bottom, span))
        max_span = max(max_span, span)
    if len(column_bounds) < 4 or max_span < min_edge_height_px:
        return None, None

    boundary_columns = [column for column in column_bounds if column[3] >= max(min_edge_height_px, 0.35 * max_span)]
    if len(boundary_columns) < 4:
        return None, None

    xs = numpy.array([column[0] for column in boundary_columns], dtype=numpy.float64)
    tops = numpy.array([column[1] for column in boundary_columns], dtype=numpy.float64)
    bottoms = numpy.array([column[2] for column in boundary_columns], dtype=numpy.float64)
    if abs(float(xs[-1]) - float(xs[0])) < 1e-6:
        return None, None
    top_slope, top_intercept = numpy.polyfit(xs, tops, 1)
    bottom_slope, bottom_intercept = numpy.polyfit(xs, bottoms, 1)
    return (float(top_slope), float(top_intercept)), (float(bottom_slope), float(bottom_intercept))


def _intersect_x_of_y_line_with_y_of_x_line(
    x_of_y_line: tuple[float, float],
    y_of_x_line: tuple[float, float],
) -> ImagePoint | None:
    x_slope, x_intercept = x_of_y_line
    y_slope, y_intercept = y_of_x_line
    denominator = 1.0 - x_slope * y_slope
    if abs(denominator) < 1e-6:
        return None
    x = (x_slope * y_intercept + x_intercept) / denominator
    y = y_slope * x + y_intercept
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return ImagePoint(float(x), float(y))


def _debug_contour_edge_cutout_image(
    cv2,
    edges,
    contour,
    *,
    x_offset: int,
    y_offset: int,
    roi_width: int,
    roi_height: int,
    stem_center_x: float | None = None,
    stem_top_y: float | None = None,
    min_edge_height_px: float = 8.0,
):
    import numpy

    cutout = numpy.zeros(edges.shape[:2], dtype=numpy.uint8)
    y_end = min(edges.shape[0], y_offset + roi_height)
    x_end = min(edges.shape[1], x_offset + roi_width)
    if y_offset < 0 or x_offset < 0 or y_end <= y_offset or x_end <= x_offset:
        return cutout

    roi_edges = edges[y_offset:y_end, x_offset:x_end]
    boundary = numpy.zeros(roi_edges.shape[:2], dtype=numpy.uint8)
    boundary_thickness = max(3, int(round(min_edge_height_px * 0.55)))
    cv2.drawContours(boundary, [contour], -1, 255, thickness=boundary_thickness)
    selected = cv2.bitwise_and(roi_edges, boundary)

    if stem_center_x is not None and stem_top_y is not None:
        local_stem_x = int(round(stem_center_x - x_offset))
        local_stem_top = int(round(stem_top_y - y_offset))
        stem_half_width = max(4, int(round(min_edge_height_px * 0.85)))
        erase_y = max(0, local_stem_top - max(2, int(round(min_edge_height_px * 0.20))))
        erase_left = max(0, local_stem_x - stem_half_width)
        erase_right = min(selected.shape[1], local_stem_x + stem_half_width + 1)
        if erase_left < erase_right and erase_y < selected.shape[0]:
            selected[erase_y:, erase_left:erase_right] = 0

    cutout[y_offset:y_end, x_offset:x_end] = selected
    return cutout


def _plain_face_from_stem_cropped_edges(
    cv2,
    edges,
    *,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> _SilhouetteFaceCandidate | None:
    import numpy

    stem = _stem_anchor_from_edges(cv2, edges, min_edge_height_px=min_edge_height_px)
    if stem is None:
        return None
    stem_center_x, stem_top_y = stem
    contour_face = _plain_face_from_stem_head_contour(
        cv2,
        edges,
        stem_center_x=stem_center_x,
        stem_top_y=stem_top_y,
        min_area_px=min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )
    if contour_face is not None:
        return contour_face

    frame_height, frame_width = edges.shape[:2]
    search_height = max(35, int(round(0.60 * frame_height)))
    y_min = max(0, int(round(stem_top_y - search_height)))
    y_max = min(frame_height, int(round(stem_top_y + max(3.0, 0.5 * min_edge_height_px))))
    x_radius = max(35, int(round(0.30 * frame_width)))
    x_min = max(0, int(round(stem_center_x - x_radius)))
    x_max = min(frame_width, int(round(stem_center_x + x_radius)))
    if y_max <= y_min or x_max <= x_min:
        return None

    roi = edges[y_min:y_max, x_min:x_max]
    row_bounds = []
    max_span = 0.0
    for local_y in range(roi.shape[0]):
        global_y = y_min + local_y
        columns = numpy.flatnonzero(roi[local_y, :])
        if len(columns) < 2:
            continue
        left = float(columns[0] + x_min)
        right = float(columns[-1] + x_min)
        span = right - left
        if span <= 0.0:
            continue
        row_bounds.append((float(global_y), left, right, span))
        max_span = max(max_span, span)

    if len(row_bounds) < 4 or max_span < min_edge_height_px:
        return None

    broad_rows = [row for row in row_bounds if row[3] >= max(min_edge_height_px, 0.55 * max_span)]
    if len(broad_rows) < 4:
        return None

    ys = numpy.array([row[0] for row in broad_rows], dtype=numpy.float64)
    lefts = numpy.array([row[1] for row in broad_rows], dtype=numpy.float64)
    rights = numpy.array([row[2] for row in broad_rows], dtype=numpy.float64)
    top = float(ys.min())
    bottom = float(ys.max())
    if bottom - top < min_edge_height_px:
        return None

    left_at_top, left_at_bottom = _fit_boundary_x_at_ys(ys, lefts, top, bottom)
    right_at_top, right_at_bottom = _fit_boundary_x_at_ys(ys, rights, top, bottom)
    width_top = right_at_top - left_at_top
    width_bottom = right_at_bottom - left_at_bottom
    avg_width = (width_top + width_bottom) / 2.0
    height = bottom - top
    if width_top < min_edge_height_px or width_bottom < min_edge_height_px or avg_width < min_edge_height_px:
        return None
    left = min(left_at_top, left_at_bottom)
    right = max(right_at_top, right_at_bottom)
    width = right - left
    if stem_center_x < left + 0.18 * width or stem_center_x > right - 0.18 * width:
        return None

    aspect_ratio = avg_width / max(height, 1e-6)
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return None
    corners = order_corners(
        (
            ImagePoint(left_at_top, top),
            ImagePoint(right_at_top, top),
            ImagePoint(right_at_bottom, bottom),
            ImagePoint(left_at_bottom, bottom),
        )
    )
    area = _polygon_area(corners)
    if area < min_area_px:
        return None

    edge_cutout = _expanded_head_edge_roi(
        cv2,
        edges,
        corners,
        margin_px=max(10, int(round(min_edge_height_px * 1.8))),
        stem_center_x=stem_center_x,
        stem_top_y=stem_top_y,
        min_edge_height_px=min_edge_height_px,
    )
    face_mask, border_corners = _connected_border_mask_and_corners(
        cv2,
        edges,
        edge_cutout,
        fallback_corners=corners,
        min_edge_height_px=min_edge_height_px,
    )
    return _SilhouetteFaceCandidate(
        corners=border_corners,
        face_mask=face_mask,
    )


def _plain_face_from_stem_head_contour(
    cv2,
    edges,
    *,
    stem_center_x: float,
    stem_top_y: float,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> _SilhouetteFaceCandidate | None:
    import numpy

    frame_height, frame_width = edges.shape[:2]
    search_height = max(35, int(round(0.62 * frame_height)))
    y_min = max(0, int(round(stem_top_y - search_height)))
    y_max = min(frame_height, int(round(stem_top_y + max(4.0, 0.45 * min_edge_height_px))))
    x_radius = max(35, int(round(0.34 * frame_width)))
    x_min = max(0, int(round(stem_center_x - x_radius)))
    x_max = min(frame_width, int(round(stem_center_x + x_radius)))
    if y_max <= y_min or x_max <= x_min:
        return None

    roi = edges[y_min:y_max, x_min:x_max].copy()
    if cv2.countNonZero(roi) == 0:
        return None

    close_kernel_size = max(3, int(round(min_edge_height_px * 0.45)))
    if close_kernel_size % 2 == 0:
        close_kernel_size += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: _SilhouetteFaceCandidate | None = None
    best_score = -1.0
    for contour in contours:
        local_x, local_y, width, height = cv2.boundingRect(contour)
        if width < min_edge_height_px or height < min_edge_height_px:
            continue
        global_left = float(x_min + local_x)
        global_right = float(x_min + local_x + width)
        global_top = float(y_min + local_y)
        global_bottom = float(y_min + local_y + height)
        if not (global_left + 0.15 * width <= stem_center_x <= global_right - 0.15 * width):
            continue
        if global_bottom < stem_top_y - 0.25 * height:
            continue
        if global_top > stem_top_y - 0.35 * height:
            continue

        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area_px:
            continue
        corners = _quadrilateral_corners(cv2, contour)
        if corners is None:
            hull = cv2.convexHull(contour)
            corners = _quadrilateral_corners(cv2, hull)
        if corners is None:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            corners = tuple(ImagePoint(float(point[0]), float(point[1])) for point in box)
        corners = order_corners(
            tuple(
                ImagePoint(point.u_px + x_min, point.v_px + y_min)
                for point in corners
            )
        )
        aspect_ratio = quadrilateral_aspect_ratio(corners)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        area = _polygon_area(corners)
        if area < min_area_px:
            continue
        bottom_y = max(point.v_px for point in corners)
        top_y = min(point.v_px for point in corners)
        if bottom_y > stem_top_y + max(8.0, 0.12 * (bottom_y - top_y)):
            continue

        edge_cutout = _expanded_head_edge_roi(
            cv2,
            edges,
            corners,
            margin_px=max(10, int(round(min_edge_height_px * 1.8))),
            stem_center_x=stem_center_x,
            stem_top_y=stem_top_y,
            min_edge_height_px=min_edge_height_px,
        )
        face_mask, border_corners = _connected_border_mask_and_corners(
            cv2,
            edges,
            edge_cutout,
            fallback_corners=corners,
            min_edge_height_px=min_edge_height_px,
        )
        border_area = _polygon_area(border_corners)
        if border_area >= min_area_px:
            corners = border_corners
            area = border_area

        score = area * (1.0 + max(0.0, 1.0 - abs(stem_center_x - (global_left + global_right) / 2.0) / max(width, 1)))
        if score > best_score:
            best = _SilhouetteFaceCandidate(
                corners=corners,
                face_mask=face_mask,
            )
            best_score = score
    return best


def _stem_anchored_face_from_edges(
    cv2,
    edges,
    *,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> _SilhouetteFaceCandidate | None:
    import numpy

    stem = _stem_anchor_from_edges(cv2, edges, min_edge_height_px=min_edge_height_px)
    if stem is None:
        return None
    stem_center_x, stem_top_y = stem
    frame_height, frame_width = edges.shape[:2]
    search_height = max(35, int(round(0.65 * frame_height)))
    y_min = max(0, int(round(stem_top_y - search_height)))
    y_max = min(frame_height, int(round(stem_top_y + max(8.0, 0.08 * search_height))))
    x_radius = max(35, int(round(0.33 * frame_width)))
    x_min = max(0, int(round(stem_center_x - x_radius)))
    x_max = min(frame_width, int(round(stem_center_x + x_radius)))
    if y_max <= y_min or x_max <= x_min:
        return None

    roi = edges[y_min:y_max, x_min:x_max]
    row_bounds = []
    max_span = 0.0
    stem_top_limit = stem_top_y + max(4.0, min_edge_height_px)
    for local_y in range(roi.shape[0]):
        global_y = y_min + local_y
        if global_y > stem_top_limit:
            continue
        columns = numpy.flatnonzero(roi[local_y, :])
        if len(columns) < 2:
            continue
        left = float(columns[0] + x_min)
        right = float(columns[-1] + x_min)
        span = right - left
        if span <= 0.0:
            continue
        row_bounds.append((float(global_y), left, right, span))
        max_span = max(max_span, span)

    if len(row_bounds) < 4 or max_span < min_edge_height_px:
        return None

    min_span = max(min_edge_height_px, 0.45 * max_span)
    broad_rows = [row for row in row_bounds if row[3] >= min_span]
    if len(broad_rows) < 4:
        return None

    ys = numpy.array([row[0] for row in broad_rows], dtype=numpy.float64)
    lefts = numpy.array([row[1] for row in broad_rows], dtype=numpy.float64)
    rights = numpy.array([row[2] for row in broad_rows], dtype=numpy.float64)
    top = float(ys.min())
    bottom = float(ys.max())
    if bottom - top < min_edge_height_px:
        return None

    left_at_top, left_at_bottom = _fit_boundary_x_at_ys(ys, lefts, top, bottom)
    right_at_top, right_at_bottom = _fit_boundary_x_at_ys(ys, rights, top, bottom)
    width_top = right_at_top - left_at_top
    width_bottom = right_at_bottom - left_at_bottom
    avg_width = (width_top + width_bottom) / 2.0
    height = bottom - top
    if width_top < min_edge_height_px or width_bottom < min_edge_height_px or avg_width < min_edge_height_px:
        return None
    left = min(left_at_top, left_at_bottom)
    right = max(right_at_top, right_at_bottom)
    width = right - left
    if stem_center_x < left + 0.20 * width or stem_center_x > right - 0.20 * width:
        return None

    aspect_ratio = avg_width / max(height, 1e-6)
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return None
    corners = order_corners(
        (
            ImagePoint(left_at_top, top),
            ImagePoint(right_at_top, top),
            ImagePoint(right_at_bottom, bottom),
            ImagePoint(left_at_bottom, bottom),
        )
    )
    area = _polygon_area(corners)
    if area < min_area_px:
        return None

    return _SilhouetteFaceCandidate(
        corners=corners,
        face_mask=_debug_outline_image(cv2, edges.shape, corners),
    )


def _fit_boundary_x_at_ys(ys, xs, top_y: float, bottom_y: float) -> tuple[float, float]:
    import numpy

    if len(ys) < 2 or abs(float(ys[-1]) - float(ys[0])) < 1e-6:
        value = float(xs.mean()) if len(xs) else 0.0
        return value, value
    slope, intercept = numpy.polyfit(ys, xs, 1)
    return float(slope * top_y + intercept), float(slope * bottom_y + intercept)


def _stem_anchor_from_edges(cv2, edges, *, min_edge_height_px: float) -> tuple[float, float] | None:
    segments = _line_segments_from_edges(
        cv2,
        edges,
        hough_threshold=12,
        hough_min_line_length_px=max(12, int(round(min_edge_height_px * 3.0))),
        hough_max_line_gap_px=10,
    )
    frame_height, frame_width = edges.shape[:2]
    min_length = max(min_edge_height_px * 3.0, frame_height * 0.18)
    verticals = [
        segment
        for segment in segments
        if segment.length_px >= min_length
        and abs(abs(segment.angle_deg) - 90.0) <= 12.0
        and segment.y_min >= 0.30 * frame_height
        and segment.y_max >= 0.45 * frame_height
    ]
    if not verticals:
        return None

    best_pair = None
    best_pair_score = -1.0
    for index, left in enumerate(verticals):
        for right in verticals[index + 1 :]:
            separation = abs(left.x_mid - right.x_mid)
            if separation < 3.0 or separation > 0.16 * frame_width:
                continue
            overlap = _overlap_length(left.y_min, left.y_max, right.y_min, right.y_max)
            if overlap < 0.35 * min(left.length_px, right.length_px):
                continue
            top_y = min(left.y_min, right.y_min)
            center_x = (left.x_mid + right.x_mid) / 2.0
            score = overlap + 0.25 * (left.length_px + right.length_px) + max(0.0, frame_height - top_y) / frame_height
            if score > best_pair_score:
                best_pair = (center_x, top_y)
                best_pair_score = score
    if best_pair is not None:
        return best_pair

    best = max(verticals, key=lambda segment: segment.length_px * (1.0 + segment.y_max / max(frame_height, 1)))
    return best.x_mid, best.y_min


def _face_quadrilateral_from_silhouette(
    cv2,
    edges,
    *,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    face_width_fraction: float,
) -> _SilhouetteFaceCandidate | None:
    import numpy

    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_candidate = None
    best_score = -1.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < min_area_px:
            continue
        if width < min_edge_height_px or height < min_edge_height_px:
            continue

        component = numpy.zeros(edges.shape[:2], dtype=numpy.uint8)
        cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
        crop = component[y : y + height, x : x + width]
        row_widths = [int(cv2.countNonZero(crop[row_index, :])) for row_index in range(crop.shape[0])]
        band = wide_row_band(row_widths, width_fraction=face_width_fraction)
        if band is None:
            continue
        band_start, band_end = band
        if band_start > 0.30 * height:
            continue
        if band_end - band_start + 1 < min_edge_height_px:
            continue

        band_mask = crop[band_start : band_end + 1, :]
        band_mask, band_x_offset = _expand_band_mask_to_nearby_edges(
            cv2,
            edges,
            band_mask,
            x_offset=x,
            y_offset=y + band_start,
        )
        corners = _quadrilateral_from_mask_component(
            cv2,
            band_mask,
            x_offset=float(band_x_offset),
            y_offset=float(y + band_start),
        )
        if corners is None:
            continue
        aspect_ratio = quadrilateral_aspect_ratio(corners)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        if _polygon_area(corners) < min_area_px:
            continue
        stem_score = _lower_stem_support_score(cv2, crop, band_start, band_end)
        score = _polygon_area(corners) * (1.0 + 2.0 * stem_score)
        if score > best_score:
            best_candidate = _SilhouetteFaceCandidate(
                corners=corners,
                face_mask=_debug_outline_image(cv2, edges.shape, corners),
            )
            best_score = score
    return best_candidate


def _expand_band_mask_to_nearby_edges(cv2, edges, band_mask, *, x_offset: int, y_offset: int):
    import numpy

    height, width = band_mask.shape[:2]
    all_columns = numpy.flatnonzero(band_mask.max(axis=0))
    if len(all_columns) == 0:
        return band_mask, x_offset

    band_left = int(all_columns[0])
    band_right = int(all_columns[-1])
    band_width = max(1, band_right - band_left + 1)
    search_margin = max(8, int(round(0.35 * band_width)))
    global_left = max(0, x_offset + band_left - search_margin)
    global_right = min(edges.shape[1], x_offset + band_right + search_margin + 1)
    if global_right <= global_left:
        return band_mask, x_offset

    expanded = numpy.zeros((height, global_right - global_left), dtype=band_mask.dtype)
    for row_index in range(height):
        columns = numpy.flatnonzero(band_mask[row_index, :])
        if len(columns) == 0:
            continue
        left = x_offset + int(columns[0]) - global_left
        right = x_offset + int(columns[-1]) - global_left
        global_y = y_offset + row_index
        if global_y < 0 or global_y >= edges.shape[0]:
            continue
        support = numpy.flatnonzero(edges[global_y, global_left:global_right])
        if len(support) > 0:
            left = min(left, int(support[0]))
            right = max(right, int(support[-1]))
        left = max(0, left)
        right = min(expanded.shape[1] - 1, right)
        expanded[row_index, left : right + 1] = 255

    if cv2.countNonZero(expanded) == 0:
        return band_mask, x_offset
    return expanded, global_left


def _lower_stem_support_score(cv2, crop, band_start: int, band_end: int) -> float:
    import numpy

    face = crop[band_start : band_end + 1, :]
    face_columns = numpy.flatnonzero(face.max(axis=0))
    if len(face_columns) == 0:
        return 0.0

    face_width = max(1.0, float(face_columns[-1] - face_columns[0] + 1))
    face_center = (float(face_columns[0]) + float(face_columns[-1])) / 2.0
    face_height = max(1, band_end - band_start + 1)
    lower_start = band_end + 1
    lower_end = min(crop.shape[0], lower_start + max(8, int(round(face_height * 0.8))))
    if lower_start >= lower_end:
        return 0.0

    supported_rows = 0
    for row_index in range(lower_start, lower_end):
        row = crop[row_index, :]
        row_width = int(cv2.countNonZero(row))
        if row_width <= 0 or row_width > 0.45 * face_width:
            continue
        row_columns = numpy.flatnonzero(row)
        if len(row_columns) == 0:
            continue
        row_center = (float(row_columns[0]) + float(row_columns[-1])) / 2.0
        if abs(row_center - face_center) <= 0.28 * face_width:
            supported_rows += 1

    required_rows = max(4.0, face_height * 0.18)
    return min(1.0, supported_rows / required_rows)


def _quadrilateral_from_mask_component(
    cv2,
    mask,
    *,
    x_offset: float,
    y_offset: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    corners = _quadrilateral_corners(cv2, contour)
    if corners is None:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = tuple(ImagePoint(float(point[0]), float(point[1])) for point in box)

    return order_corners(
        tuple(
            ImagePoint(point.u_px + x_offset, point.v_px + y_offset)
            for point in corners
        )
    )


def _largest_external_bounding_area(cv2, edges) -> float:
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = 0.0
    for contour in contours:
        _x, _y, width, height = cv2.boundingRect(contour)
        largest = max(largest, float(width * height))
    return largest


def _edge_input_image(cv2, frame, *, edge_preprocess: str, blur_kernel: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if edge_preprocess == "outer_border":
        return _outer_border_edge_input(cv2, gray)
    if edge_preprocess == "gray":
        if blur_kernel > 1:
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        return gray
    raise ValueError(f"unsupported edge preprocess mode: {edge_preprocess}")


def _outer_border_edge_input(cv2, gray):
    # Suppress QR-code texture before Canny. The square outline and stem are
    # low-frequency structure; QR modules are high-frequency interior texture.
    smoothed = cv2.GaussianBlur(gray, (9, 9), 0)
    smoothed = cv2.medianBlur(smoothed, 7)
    return cv2.bilateralFilter(smoothed, 9, 50, 50)


@dataclass(frozen=True)
class _LineSegment:
    start: ImagePoint
    end: ImagePoint
    length_px: float
    angle_deg: float

    @property
    def y_min(self) -> float:
        return min(self.start.v_px, self.end.v_px)

    @property
    def y_max(self) -> float:
        return max(self.start.v_px, self.end.v_px)

    @property
    def x_min(self) -> float:
        return min(self.start.u_px, self.end.u_px)

    @property
    def x_max(self) -> float:
        return max(self.start.u_px, self.end.u_px)

    @property
    def x_mid(self) -> float:
        return (self.start.u_px + self.end.u_px) / 2.0

    def top_point(self) -> ImagePoint:
        return self.start if self.start.v_px <= self.end.v_px else self.end

    def bottom_point(self) -> ImagePoint:
        return self.start if self.start.v_px > self.end.v_px else self.end


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


def _quadrilateral_from_line_segments(
    cv2,
    edges,
    *,
    hough_threshold: int,
    hough_min_line_length_px: int,
    hough_max_line_gap_px: int,
    min_boundary_line_length_px: float,
    min_edge_height_px: float,
    min_area_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    segments = _line_segments_from_edges(
        cv2,
        edges,
        hough_threshold=hough_threshold,
        hough_min_line_length_px=hough_min_line_length_px,
        hough_max_line_gap_px=hough_max_line_gap_px,
    )
    if not segments:
        return None

    verticals = [
        segment
        for segment in segments
        if segment.length_px >= max(min_edge_height_px, min_boundary_line_length_px)
        and abs(abs(segment.angle_deg) - 90.0) <= 25.0
    ]
    horizontals = [
        segment
        for segment in segments
        if segment.length_px >= max(min_edge_height_px, min_boundary_line_length_px * 0.55)
        and abs(segment.angle_deg) <= 25.0
    ]

    best_corners = None
    best_score = -1.0
    for left in verticals:
        for right in verticals:
            if left.x_mid >= right.x_mid:
                continue
            width = right.x_mid - left.x_mid
            avg_height = (left.length_px + right.length_px) / 2.0
            if width < min_edge_height_px or avg_height < min_edge_height_px:
                continue
            if abs(left.length_px - right.length_px) > 0.55 * avg_height:
                continue
            if abs(left.y_min - right.y_min) > 0.45 * avg_height:
                continue
            if abs(left.y_max - right.y_max) > 0.45 * avg_height:
                continue

            corners = order_corners((left.top_point(), right.top_point(), right.bottom_point(), left.bottom_point()))
            aspect_ratio = quadrilateral_aspect_ratio(corners)
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
            area = _polygon_area(corners)
            if area < min_area_px:
                continue
            support = _horizontal_support_score(horizontals, corners)
            score = score_quadrilateral_candidate(corners, area) * (1.0 + 0.25 * support)
            if score > best_score:
                best_corners = corners
                best_score = score
    return best_corners


def _edge_on_from_line_segments(
    cv2,
    edges,
    *,
    hough_threshold: int,
    hough_min_line_length_px: int,
    hough_max_line_gap_px: int,
    min_boundary_line_length_px: float,
    min_edge_height_px: float,
) -> StandAxisImageEstimate | None:
    segments = _line_segments_from_edges(
        cv2,
        edges,
        hough_threshold=hough_threshold,
        hough_min_line_length_px=hough_min_line_length_px,
        hough_max_line_gap_px=hough_max_line_gap_px,
    )
    verticals = [
        segment
        for segment in segments
        if segment.length_px >= max(min_edge_height_px * 2.0, hough_min_line_length_px, min_boundary_line_length_px)
        and abs(abs(segment.angle_deg) - 90.0) <= 15.0
    ]
    if not verticals:
        return None
    frame_height = float(edges.shape[0])
    best = max(
        verticals,
        key=lambda segment: segment.length_px * (1.0 + max(0.0, (frame_height - segment.y_min) / frame_height)),
    )
    return estimate_edge_on_axis_from_line(
        best.top_point(),
        best.bottom_point(),
        min_edge_height_px=min_edge_height_px,
        source="edge_on_line",
    )


def _line_segments_from_edges(
    cv2,
    edges,
    *,
    hough_threshold: int,
    hough_min_line_length_px: int,
    hough_max_line_gap_px: int,
) -> tuple[_LineSegment, ...]:
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=hough_min_line_length_px,
        maxLineGap=hough_max_line_gap_px,
    )
    if raw_lines is None:
        return ()
    return tuple(_line_segment_from_hough(line[0]) for line in raw_lines)


def _line_segment_from_hough(values) -> _LineSegment:
    x1, y1, x2, y2 = (float(value) for value in values)
    start = ImagePoint(x1, y1)
    end = ImagePoint(x2, y2)
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    return _LineSegment(start, end, math.hypot(dx, dy), angle)


def _horizontal_support_score(horizontals: Sequence[_LineSegment], corners: Sequence[ImagePoint]) -> int:
    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    x_min = min(top_left.u_px, bottom_left.u_px)
    x_max = max(top_right.u_px, bottom_right.u_px)
    width = max(1.0, x_max - x_min)
    top_y = (top_left.v_px + top_right.v_px) / 2.0
    bottom_y = (bottom_left.v_px + bottom_right.v_px) / 2.0
    tolerance = max(5.0, 0.18 * width)
    support = 0
    for target_y in (top_y, bottom_y):
        if any(
            abs(((line.start.v_px + line.end.v_px) / 2.0) - target_y) <= tolerance
            and _overlap_length(line.x_min, line.x_max, x_min, x_max) >= 0.35 * width
            for line in horizontals
        ):
            support += 1
    return support


def _overlap_length(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def _contour_has_lower_appendage(cv2, contour, corners: Sequence[ImagePoint]) -> bool:
    _x, y, _w, h = cv2.boundingRect(contour)
    contour_bottom = y + h
    candidate_bottom = max(point.v_px for point in corners)
    candidate_height = max(point.v_px for point in corners) - min(point.v_px for point in corners)
    if candidate_height <= 0.0:
        return False
    return contour_bottom > candidate_bottom + 0.25 * candidate_height


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
        [[point.u_px, point.v_px] for point in order_corners(corners)],
        dtype=numpy.float64,
    )
    camera_matrix = numpy.array(
        [
            [camera_fx_px, 0.0, camera_cx_px],
            [0.0, camera_fy_px, camera_cy_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=numpy.float64,
    )
    distortion = numpy.zeros((4, 1), dtype=numpy.float64)
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
    rotation, _jacobian = cv2.Rodrigues(rvec)
    normal = rotation @ numpy.array([[0.0], [0.0], [1.0]], dtype=numpy.float64)
    normal_x = float(normal[0, 0])
    normal_z = float(normal[2, 0])
    if not math.isfinite(normal_x) or not math.isfinite(normal_z):
        return None
    return math.degrees(math.atan2(normal_x, abs(normal_z)))


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
