"""Stem-anchored localization and head-candidate construction."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _distance,
    _polygon_area,
    _quadrilateral_corners,
    estimate_edge_on_axis_from_line,
    order_corners,
    quadrilateral_aspect_ratio,
    score_quadrilateral_candidate,
)
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint,
    StandAxisImageEstimate,
    _LineSegment,
    _SilhouetteFaceCandidate,
)
from scripts.aufgabe04.perception.stand_axis.raw_support import (
    _quadrilateral_edge_support,
    _raw_side_evidence_and_corners,
    _select_supported_head_corners,
)
from scripts.aufgabe04.perception.stand_structure_hypothesis import (
    evaluate_stand_structure,
)


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

    # ``fallback_corners`` remains part of this private call contract because
    # callers also use it for the final independent support check.  It must
    # never be returned here merely because every border fit failed.
    _ = fallback_corners

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
        return edge_cutout, None

    side_rows = [row for row in row_bounds if row[3] >= max(min_edge_height_px, 0.38 * max_span)]
    if len(side_rows) < 4:
        return edge_cutout, None

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
        return edge_cutout, None

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
                left_line,
                right_line,
                top_y=top_y,
                bottom_y=bottom_y,
                min_edge_height_px=min_edge_height_px,
            )
    else:
        corners = _outer_row_envelope_corners(
            left_line,
            right_line,
            top_y=top_y,
            bottom_y=bottom_y,
            min_edge_height_px=min_edge_height_px,
        )
    if corners is None:
        return edge_cutout, None

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
    left_line: tuple[float, float] | None,
    right_line: tuple[float, float] | None,
    *,
    top_y: float,
    bottom_y: float,
    min_edge_height_px: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    if left_line is None or right_line is None:
        return None
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
    # The connected localization contour may sit several pixels outside the
    # pre-morphology Canny border.  Keep the band wide enough to bridge that
    # known morphology offset, while still far narrower than the head interior
    # so label/QR texture cannot become rectangle evidence.
    boundary_thickness = max(
        5,
        min(11, int(round(min_edge_height_px * 1.10))),
    )
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


def _stem_local_x_bounds(
    frame_width: int,
    *,
    stem_center_x: float,
    min_edge_height_px: float,
) -> tuple[int, int]:
    """Bound a stem hypothesis to its own stand in a full camera frame."""

    # At close range 6.75 minimum-edge units still cover a ~100 px head. In a
    # 640 px full frame the fractional floor gives roughly 54 px on each side,
    # enough for the target head but not the neighbouring smaller stand seen in
    # the standalone-viewer flicker captures.
    x_radius = max(
        35.0,
        6.75 * min_edge_height_px,
        0.085 * frame_width,
    )
    x_min = max(0, int(math.floor(stem_center_x - x_radius)))
    x_max = min(frame_width, int(math.ceil(stem_center_x + x_radius)) + 1)
    return x_min, x_max


def _line_segment_x_at_y(segment, y_px: float) -> float | None:
    """Evaluate a non-horizontal Hough segment's infinite line at ``y_px``."""

    dy = segment.end.v_px - segment.start.v_px
    if abs(dy) <= 1e-6:
        return None
    fraction = (y_px - segment.start.v_px) / dy
    return segment.start.u_px + fraction * (
        segment.end.u_px - segment.start.u_px
    )


def _stem_owned_head_candidate_score(
    cv2,
    candidate: _SilhouetteFaceCandidate,
    *,
    stem_center_x: float,
    stem_top_y: float,
) -> float:
    """Rank accepted heads by raw support, square shape, and neck ownership."""

    corners = order_corners(candidate.corners)
    support = _quadrilateral_edge_support(cv2, candidate.face_mask, corners)
    xs = [point.u_px for point in corners]
    ys = [point.v_px for point in corners]
    width = max(max(xs) - min(xs), 1.0)
    height = max(max(ys) - min(ys), 1.0)
    center_x = (min(xs) + max(xs)) / 2.0
    bottom_y = max(ys)
    aspect_ratio = width / height
    square_score = 1.0 / (
        1.0 + abs(math.log(max(aspect_ratio, 1e-6)))
    )
    center_score = max(
        0.0,
        1.0 - abs(stem_center_x - center_x) / width,
    )
    bottom_score = max(
        0.0,
        1.0 - abs(stem_top_y - bottom_y) / height,
    )
    return (
        5.0 * support.mean
        + 1.5 * square_score
        + center_score
        + bottom_score
    )


def _stem_owned_head_from_line_segments(
    cv2,
    edges,
    *,
    measurement_edges,
    stem_center_x: float,
    stem_top_y: float,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> _SilhouetteFaceCandidate | None:
    """Select a closed outer head cycle owned by one detected stand stem.

    Canny legitimately retains QR modules, radiator slats, and window seams.
    Taking the first and last pixel of every row therefore lets unrelated
    components become head boundaries.  This proposal path instead pairs long
    side segments which straddle the stem and end at its head-to-neck
    transition.  The existing immutable-raw-edge gate still has to recover and
    support all four sides, so Hough lines are localization evidence only.
    """

    frame_height, frame_width = edges.shape[:2]
    x_min, x_max = _stem_local_x_bounds(
        frame_width,
        stem_center_x=stem_center_x,
        min_edge_height_px=min_edge_height_px,
    )
    search_height = max(35.0, 0.62 * frame_height)
    y_min = max(0.0, stem_top_y - search_height)
    y_max = min(
        float(frame_height - 1),
        stem_top_y + max(8.0, 0.75 * min_edge_height_px),
    )
    if x_max <= x_min or y_max <= y_min:
        return None

    segments = _line_segments_from_edges(
        cv2,
        edges,
        hough_threshold=12,
        hough_min_line_length_px=max(
            8,
            int(round(1.75 * min_edge_height_px)),
        ),
        hough_max_line_gap_px=max(
            4,
            int(round(0.75 * min_edge_height_px)),
        ),
    )
    minimum_side_length = max(
        2.0 * min_edge_height_px,
        0.16 * (y_max - y_min),
    )
    verticals = [
        segment
        for segment in segments
        if segment.length_px >= minimum_side_length
        and abs(abs(segment.angle_deg) - 90.0) <= 18.0
        and x_min <= segment.x_mid < x_max
        and segment.y_min >= y_min - min_edge_height_px
        and segment.y_min <= stem_top_y - min_edge_height_px
        and segment.y_max >= stem_top_y
        - max(12.0, 0.28 * segment.length_px)
        and segment.y_max <= y_max + min_edge_height_px
    ]
    if len(verticals) < 2:
        return None

    best = None
    best_score = -math.inf
    local_width = max(1.0, float(x_max - x_min))
    for left in verticals:
        if left.x_mid >= stem_center_x:
            continue
        for right in verticals:
            if right.x_mid <= stem_center_x:
                continue

            width = right.x_mid - left.x_mid
            if width < 2.0 * min_edge_height_px or width > 0.95 * local_width:
                continue
            center_margin = 0.12 * width
            if not (
                left.x_mid + center_margin
                <= stem_center_x
                <= right.x_mid - center_margin
            ):
                continue

            overlap = _overlap_length(
                left.y_min,
                left.y_max,
                right.y_min,
                right.y_max,
            )
            minimum_length = min(left.length_px, right.length_px)
            average_length = (left.length_px + right.length_px) / 2.0
            if overlap < 0.45 * minimum_length:
                continue
            if (
                abs(left.length_px - right.length_px)
                > 0.55 * average_length
            ):
                continue
            if abs(left.y_min - right.y_min) > 0.35 * average_length:
                continue
            if abs(left.y_max - right.y_max) > 0.35 * average_length:
                continue

            top_y = (left.y_min + right.y_min) / 2.0
            bottom_y = (left.y_max + right.y_max) / 2.0
            height = bottom_y - top_y
            if height < 2.0 * min_edge_height_px:
                continue
            if abs(bottom_y - stem_top_y) > max(14.0, 0.22 * height):
                continue

            left_at_top = _line_segment_x_at_y(left, top_y)
            left_at_bottom = _line_segment_x_at_y(left, bottom_y)
            right_at_top = _line_segment_x_at_y(right, top_y)
            right_at_bottom = _line_segment_x_at_y(right, bottom_y)
            if None in (
                left_at_top,
                left_at_bottom,
                right_at_top,
                right_at_bottom,
            ):
                continue
            rough_corners = order_corners(
                (
                    ImagePoint(float(left_at_top), top_y),
                    ImagePoint(float(right_at_top), top_y),
                    ImagePoint(float(right_at_bottom), bottom_y),
                    ImagePoint(float(left_at_bottom), bottom_y),
                )
            )
            aspect_ratio = quadrilateral_aspect_ratio(rough_corners)
            if (
                aspect_ratio < min_aspect_ratio
                or aspect_ratio > max_aspect_ratio
            ):
                continue
            area = _polygon_area(rough_corners)
            if area < min_area_px:
                continue

            face_mask, refitted_corners = _raw_side_evidence_and_corners(
                cv2,
                measurement_edges,
                rough_corners,
            )
            selected_corners, fit_reason, support = (
                _select_supported_head_corners(
                    cv2,
                    face_mask,
                    rough_corners,
                    refitted_corners,
                    image_shape=measurement_edges.shape,
                    stem_center_x=stem_center_x,
                    stem_top_y=stem_top_y,
                    min_aspect_ratio=min_aspect_ratio,
                    max_aspect_ratio=max_aspect_ratio,
                    allow_rough_fallback=False,
                )
            )
            if selected_corners is None or support is None:
                continue

            selected_aspect = quadrilateral_aspect_ratio(selected_corners)
            square_score = 1.0 / (
                1.0 + abs(math.log(max(selected_aspect, 1e-6)))
            )
            center_x = (
                min(point.u_px for point in selected_corners)
                + max(point.u_px for point in selected_corners)
            ) / 2.0
            selected_width = max(
                1.0,
                max(point.u_px for point in selected_corners)
                - min(point.u_px for point in selected_corners),
            )
            center_score = max(
                0.0,
                1.0 - abs(stem_center_x - center_x) / selected_width,
            )
            overlap_score = min(1.0, overlap / max(average_length, 1.0))
            extent_score = min(1.0, selected_width / local_width)
            score = (
                5.0 * support.mean
                + 1.5 * square_score
                + center_score
                + overlap_score
                + 0.5 * extent_score
            )
            if score > best_score:
                best = _SilhouetteFaceCandidate(
                    corners=selected_corners,
                    face_mask=face_mask,
                    rectangle_fit_reliable=True,
                    rectangle_fit_reason=fit_reason,
                )
                best_score = score
    return best


def _attach_structure_evidence(
    cv2,
    candidate: _SilhouetteFaceCandidate,
    *,
    measurement_edges,
    stem_center_x: float,
    stem_top_y: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> _SilhouetteFaceCandidate:
    """Validate a candidate against a current raw head-stem-base structure."""

    evidence = evaluate_stand_structure(
        cv2,
        measurement_edges,
        tuple((point.u_px, point.v_px) for point in candidate.corners),
        stem_center_x=stem_center_x,
        stem_top_y=stem_top_y,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )
    if candidate.rectangle_fit_reliable or not evidence.tracking_supported:
        return replace(candidate, structure_evidence=evidence)
    recovered = tuple(
        ImagePoint(float(u_px), float(v_px))
        for u_px, v_px in evidence.corners
    )
    return replace(
        candidate,
        corners=order_corners(recovered),
        face_mask=evidence.evidence_mask,
        rectangle_fit_reliable=True,
        rectangle_fit_reason=(
            "structure_owned_three_side_supported"
            if evidence.accepted
            else "structure_tracking_three_side_supported"
        ),
        structure_evidence=evidence,
    )

def _plain_face_from_stem_head_contour(
    cv2,
    edges,
    *,
    measurement_edges=None,
    stem_center_x: float,
    stem_top_y: float,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    fixed_parallel_side_direction: tuple[float, float] | None = None,
) -> _SilhouetteFaceCandidate | None:
    import numpy

    uses_independent_measurement = (
        measurement_edges is not None and measurement_edges is not edges
    )
    if measurement_edges is None:
        measurement_edges = edges
    if measurement_edges.shape[:2] != edges.shape[:2]:
        raise ValueError("measurement_edges must match localization edges")

    frame_height, frame_width = edges.shape[:2]
    search_height = max(35, int(round(0.62 * frame_height)))
    y_min = max(0, int(round(stem_top_y - search_height)))
    y_max = min(frame_height, int(round(stem_top_y + max(4.0, 0.45 * min_edge_height_px))))
    x_min, x_max = _stem_local_x_bounds(
        frame_width,
        stem_center_x=stem_center_x,
        min_edge_height_px=min_edge_height_px,
    )
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
    best_rank = (-1, -1.0)
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
        top_left, top_right, bottom_right, bottom_left = corners
        top_width = _distance(top_left, top_right)
        bottom_width = _distance(bottom_left, bottom_right)
        if (
            min(top_width, bottom_width)
            < 0.40 * max(top_width, bottom_width, 1e-6)
        ):
            # A wall branch joined to one head corner produces a trapezoid
            # hundreds of pixels wide on only one edge.  It is not a plausible
            # projected square face, even though its average aspect ratio can
            # accidentally fall inside the accepted range.
            continue
        area = _polygon_area(corners)
        if area < min_area_px:
            continue
        rough_area = area
        bottom_y = max(point.v_px for point in corners)
        top_y = min(point.v_px for point in corners)
        if bottom_y > stem_top_y + max(8.0, 0.12 * (bottom_y - top_y)):
            continue

        rough_corners = corners
        if uses_independent_measurement:
            face_mask, border_corners = _raw_side_evidence_and_corners(
                cv2,
                measurement_edges,
                rough_corners,
                fixed_parallel_side_direction=fixed_parallel_side_direction,
            )
        else:
            # Keep only original Canny edges near the selected head contour.
            # This connected-border path remains for same-domain callers; the
            # dual-edge path above deliberately does not depend on connectivity.
            edge_cutout = _debug_contour_edge_cutout_image(
                cv2,
                measurement_edges,
                contour,
                x_offset=x_min,
                y_offset=y_min,
                roi_width=roi.shape[1],
                roi_height=roi.shape[0],
                stem_center_x=stem_center_x,
                stem_top_y=stem_top_y,
                min_edge_height_px=min_edge_height_px,
            )
            head_gate = _edge_pixels_inside_polygon(
                cv2,
                measurement_edges,
                corners,
                margin_px=max(2, int(round(0.25 * min_edge_height_px))),
            )
            edge_cutout = cv2.bitwise_and(edge_cutout, head_gate)
            face_mask, border_corners = _connected_border_mask_and_corners(
                cv2,
                measurement_edges,
                edge_cutout,
                fallback_corners=rough_corners,
                min_edge_height_px=min_edge_height_px,
            )
        selected_corners, fit_reason, _support = _select_supported_head_corners(
            cv2,
            face_mask,
            rough_corners,
            border_corners,
            image_shape=measurement_edges.shape,
            stem_center_x=stem_center_x,
            stem_top_y=stem_top_y,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            allow_rough_fallback=not uses_independent_measurement,
        )
        candidate_corners = (
            selected_corners if selected_corners is not None else rough_corners
        )

        # Rank candidates by the robust pre-cutout quadrilateral.  Otherwise a
        # thin connected wall branch can enlarge the refitted border and beat
        # the actual stand head solely because its corrupted area is larger.
        score = rough_area * (
            1.0
            + max(
                0.0,
                1.0
                - abs(stem_center_x - (global_left + global_right) / 2.0)
                / max(width, 1),
            )
        )
        rank = (int(selected_corners is not None), score)
        if rank > best_rank:
            best = _SilhouetteFaceCandidate(
                corners=candidate_corners,
                face_mask=face_mask,
                rectangle_fit_reliable=selected_corners is not None,
                rectangle_fit_reason=fit_reason,
            )
            best_rank = rank
    return best

def _fit_boundary_x_at_ys(ys, xs, top_y: float, bottom_y: float) -> tuple[float, float]:
    import numpy

    if len(ys) < 2 or abs(float(ys[-1]) - float(ys[0])) < 1e-6:
        value = float(xs.mean()) if len(xs) else 0.0
        return value, value
    slope, intercept = numpy.polyfit(ys, xs, 1)
    return float(slope * top_y + intercept), float(slope * bottom_y + intercept)


def _stem_anchor_from_edges(cv2, edges, *, min_edge_height_px: float) -> tuple[float, float] | None:
    """Return the highest-ranked stem hypothesis for diagnostic callers."""

    candidates = _stem_anchor_candidates_from_edges(
        cv2,
        edges,
        min_edge_height_px=min_edge_height_px,
    )
    return candidates[0] if candidates else None


def _stem_anchor_candidates_from_edges(
    cv2,
    edges,
    *,
    min_edge_height_px: float,
) -> list[tuple[float, float]]:
    """Rank plausible stem anchors without treating Hough rank as truth.

    A wall seam can connect a head side to label texture and make that false
    vertical pair score slightly above the real, lower stem pair.  Consumers
    that derive orientation may therefore try these anchors in rank order and
    let independent rectangle edge support select the first valid head.
    """

    segments = _line_segments_from_edges(
        cv2,
        edges,
        hough_threshold=12,
        # The stem is narrower/shorter than a close-range head edge.  Requiring
        # three face-edge heights made the detector discard the actual stem
        # and then anchor on a long vertical side of the head instead.
        hough_min_line_length_px=max(8, int(round(min_edge_height_px * 1.5))),
        hough_max_line_gap_px=10,
    )
    frame_height, frame_width = edges.shape[:2]
    # The Gazebo full-frame stem occupies only about 10-13% of image height;
    # the earlier 15% minimum was appropriate for a tight projected ROI but
    # removed both real stem edges before pairing in a 640x480 frame.
    min_length = max(min_edge_height_px * 1.5, frame_height * 0.10)
    verticals = [
        segment
        for segment in segments
        if segment.length_px >= min_length
        and abs(abs(segment.angle_deg) - 90.0) <= 12.0
        and (
            segment.y_min >= 0.30 * frame_height
            # In a tight oblique ROI, morphology can merge a true stem edge
            # through head/label texture all the way to the upper head. Keep
            # that early-starting line only when it also reaches the bottom
            # portion where a real stand stem must continue; the sustained
            # transition gate below still has to validate the junction.
            or segment.y_max >= 0.80 * frame_height
        )
        # In a full 640x480 camera frame the visible stem terminates at the
        # base around 55-65% image height, while the head borders terminate
        # substantially higher.  The former 70% ROI-specific threshold
        # discarded the real full-frame Gazebo stem entirely.
        and segment.y_max >= 0.55 * frame_height
    ]
    if not verticals:
        return []

    scored_pairs = []
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
            lower_reach = (left.y_max + right.y_max) / max(2.0 * frame_height, 1.0)
            score = (
                overlap
                + 0.25 * (left.length_px + right.length_px)
                + 2.0 * lower_reach
                + max(0.0, frame_height - top_y) / frame_height
            )
            scored_pairs.append((score, center_x, top_y))

    candidates = []

    def append_distinct(anchor: tuple[float, float]) -> None:
        center_x, top_y = anchor
        if any(
            abs(center_x - known_x) <= 1.0 and abs(top_y - known_y) <= 2.0
            for known_x, known_y in candidates
        ):
            return
        candidates.append(anchor)

    for _score, center_x, line_top_y in sorted(scored_pairs, reverse=True):
        append_distinct(
            _resolved_stem_anchor(
                edges,
                center_x=center_x,
                line_top_y=line_top_y,
                min_edge_height_px=min_edge_height_px,
            )
        )

    # Preserve the former single-line fallback, but keep the other plausible
    # lower-reaching lines available when a false pair happened to exist.
    for segment in sorted(
        verticals,
        key=lambda item: (
            item.y_max,
            item.length_px,
            -item.y_min,
        ),
        reverse=True,
    ):
        append_distinct(
            _resolved_stem_anchor(
                edges,
                center_x=segment.x_mid,
                line_top_y=segment.y_min,
                min_edge_height_px=min_edge_height_px,
            )
        )
    return candidates


def _resolved_stem_anchor(
    edges,
    *,
    center_x: float,
    line_top_y: float,
    min_edge_height_px: float,
) -> tuple[float, float]:
    """Resolve a line top to a sustained head-to-stem transition when valid."""

    frame_height = edges.shape[0]
    transition_y = _stem_top_from_row_width_transition(
        edges,
        center_x=center_x,
        min_edge_height_px=min_edge_height_px,
    )
    line_anchor_y = min(
        frame_height - 1.0,
        line_top_y + min(2.0, 0.25 * min_edge_height_px),
    )
    use_late_transition = bool(
        transition_y is not None
        and transition_y > line_top_y + max(6.0, 0.08 * frame_height)
        and _has_sustained_stem_below_transition(
            edges,
            center_x=center_x,
            transition_y=transition_y,
            min_edge_height_px=min_edge_height_px,
        )
    )
    # Thick localization edges start paired Hough lines slightly inside the
    # lower head border, so advance their top by at most two pixels. Only a
    # much later sustained narrow run may override that line-derived anchor.
    return center_x, transition_y if use_late_transition else line_anchor_y


def _stem_top_from_row_width_transition(
    edges,
    *,
    center_x: float,
    min_edge_height_px: float,
) -> float | None:
    """Locate the broad-head to narrow-stem transition below a stem line pair."""

    import numpy

    frame_height, frame_width = edges.shape[:2]
    x_min, x_max = _stem_local_x_bounds(
        frame_width,
        stem_center_x=center_x,
        min_edge_height_px=min_edge_height_px,
    )
    local_width = max(1, x_max - x_min)
    # The head in the far-view flicker is only about 34 px wide. The old 80 px
    # full-frame broad threshold skipped it entirely. Scale against this stem's
    # local corridor and keep a clear margin over the expected narrow stem.
    broad_span = max(
        2.0 * min_edge_height_px,
        min(0.35 * local_width, 4.0 * min_edge_height_px),
    )
    narrow_span = max(
        1.25 * min_edge_height_px,
        min(0.18 * local_width, 2.5 * min_edge_height_px),
    )
    last_broad_y = None
    narrow_run_start = None
    narrow_run_length = 0
    for y_px in range(int(0.30 * frame_height), frame_height):
        columns = numpy.flatnonzero(edges[y_px, x_min:x_max])
        if len(columns) < 2:
            narrow_run_start = None
            narrow_run_length = 0
            continue
        left = float(columns[0] + x_min)
        right = float(columns[-1] + x_min)
        span = right - left
        if span >= broad_span and left <= center_x <= right:
            last_broad_y = y_px
            narrow_run_start = None
            narrow_run_length = 0
            continue
        if (
            last_broad_y is not None
            and span <= narrow_span
            and left - min_edge_height_px <= center_x <= right + min_edge_height_px
        ):
            if narrow_run_start is None:
                narrow_run_start = y_px
            narrow_run_length += 1
            if narrow_run_length >= 3:
                return float(narrow_run_start)
        else:
            narrow_run_start = None
            narrow_run_length = 0
    return None


def _has_sustained_stem_below_transition(
    edges,
    *,
    center_x: float,
    transition_y: float,
    min_edge_height_px: float,
) -> bool:
    """Confirm that a late width transition is a real lower stem.

    Hough segments on an oblique head can merge a head side, label edge, and
    stem into one long line whose reported top is far above the physical
    head/stem junction.  A genuine junction is still followed by a sustained
    narrow run around the same center.  A single arena edge crossing below the
    stand is not, so this check lets us trust the late transition without
    reviving the old floor/wall false anchor.
    """

    import numpy

    frame_height, frame_width = edges.shape[:2]
    start_y = max(0, int(math.floor(transition_y)))
    if start_y >= frame_height:
        return False
    x_min, x_max = _stem_local_x_bounds(
        frame_width,
        stem_center_x=center_x,
        min_edge_height_px=min_edge_height_px,
    )
    local_width = max(1, x_max - x_min)
    narrow_span = max(
        1.25 * min_edge_height_px,
        min(0.18 * local_width, 2.5 * min_edge_height_px),
    )
    required_run = max(
        4,
        int(math.ceil(min_edge_height_px)),
        int(math.ceil(0.08 * frame_height)),
    )
    max_gap = max(1, int(round(0.20 * min_edge_height_px)))
    supported_run = 0
    gap = 0
    search_end = min(
        frame_height,
        start_y + max(3 * required_run, int(math.ceil(0.40 * frame_height))),
    )
    for y_px in range(start_y, search_end):
        columns = numpy.flatnonzero(edges[y_px, x_min:x_max])
        supported = False
        if len(columns) >= 2:
            left = float(columns[0] + x_min)
            right = float(columns[-1] + x_min)
            span = right - left
            supported = (
                span <= narrow_span
                and left - min_edge_height_px <= center_x <= right + min_edge_height_px
            )
        if supported:
            supported_run += 1
            gap = 0
            if supported_run >= required_run:
                return True
            continue
        gap += 1
        if gap > max_gap:
            supported_run = 0
            gap = 0
    return False

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
    # Probabilistic Hough sampling otherwise depends on whatever OpenCV RNG
    # state an earlier frame/test left behind.  A fixed seed makes identical
    # edge masks yield identical stem/head candidates and prevents the live
    # diagnostic ROI from flickering between valid and unrelated rectangles.
    if hasattr(cv2, "setRNGSeed"):
        cv2.setRNGSeed(0)
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
