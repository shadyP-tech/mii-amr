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


def estimate_stand_axis_from_mask(
    cv2,
    mask,
    *,
    min_area_px: float = 250.0,
    min_edge_height_px: float = 8.0,
    stand_width_m: float | None = None,
    stand_distance_m: float | None = None,
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
    min_area_px: float = 250.0,
    min_edge_height_px: float = 8.0,
    min_aspect_ratio: float = 0.45,
    max_aspect_ratio: float = 1.80,
    stand_width_m: float | None = None,
    stand_distance_m: float | None = None,
) -> tuple[StandAxisImageEstimate, object]:
    edge_input = _edge_input_image(cv2, frame, edge_preprocess=edge_preprocess, blur_kernel=blur_kernel)
    edges = cv2.Canny(edge_input, canny_low, canny_high)
    if dilate_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)
    if close_kernel > 1 and close_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best: StandAxisImageEstimate | None = None
    best_score = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area_px:
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
        return best, edges

    silhouette_corners = _face_rectangle_from_silhouette(
        cv2,
        edges,
        min_area_px=min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        face_width_fraction=face_width_fraction,
    )
    if silhouette_corners is not None:
        return (
            estimate_stand_axis_from_corners(
                silhouette_corners,
                min_edge_height_px=min_edge_height_px,
                stand_width_m=stand_width_m,
                stand_distance_m=stand_distance_m,
                contour_area_px=_polygon_area(silhouette_corners),
                source="edge_silhouette",
            ),
            edges,
        )

    line_corners = _quadrilateral_from_line_segments(
        cv2,
        edges,
        hough_threshold=hough_threshold,
        hough_min_line_length_px=hough_min_line_length_px,
        hough_max_line_gap_px=hough_max_line_gap_px,
        min_boundary_line_length_px=min_boundary_line_length_px,
        min_edge_height_px=min_edge_height_px,
        min_area_px=min_area_px,
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
                contour_area_px=_polygon_area(line_corners),
                source="edge_lines",
            ),
            edges,
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
        return edge_on, edges

    if best is None:
        return _unusable("no_edge_quadrilateral", source="edges"), edges


def estimate_stand_axis_from_corners(
    corners: Sequence[ImagePoint],
    *,
    min_edge_height_px: float = 8.0,
    stand_width_m: float | None = None,
    stand_distance_m: float | None = None,
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


def _face_rectangle_from_silhouette(
    cv2,
    edges,
    *,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    face_width_fraction: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    import numpy

    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if float(cv2.contourArea(contour)) < min_area_px:
            continue
        x, y, width, height = cv2.boundingRect(contour)
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

        band_crop = crop[band_start : band_end + 1, :]
        ys, xs = numpy.where(band_crop > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x_min = x + float(xs.min())
        x_max = x + float(xs.max())
        y_min = y + float(band_start + ys.min())
        y_max = y + float(band_start + ys.max())
        corners = order_corners(
            (
                ImagePoint(x_min, y_min),
                ImagePoint(x_max, y_min),
                ImagePoint(x_max, y_max),
                ImagePoint(x_min, y_max),
            )
        )
        aspect_ratio = quadrilateral_aspect_ratio(corners)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        if _polygon_area(corners) < min_area_px:
            continue
        return corners
    return None


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
