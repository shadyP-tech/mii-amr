"""Head-first raw-edge proposals for the square stand frame."""

from __future__ import annotations

import math

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _distance,
    order_corners,
    quadrilateral_aspect_ratio,
)
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint,
    _SilhouetteFaceCandidate,
)
from scripts.aufgabe04.perception.stand_axis.raw_support import (
    _quadrilateral_edge_support,
    _raw_side_evidence_and_corners,
)


def _short_centered_neck_support(edge_mask, corners) -> bool:
    """Check for a short post continuation below an already fitted head.

    This is intentionally a validation-only signal.  It cannot move a head
    corner or expand a quadrilateral into the stand stem.
    """

    import numpy

    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    width = (_distance(top_left, top_right) + _distance(bottom_left, bottom_right)) / 2.0
    height = (_distance(top_left, bottom_left) + _distance(top_right, bottom_right)) / 2.0
    if width <= 1.0 or height <= 1.0:
        return False
    center_x = (bottom_left.u_px + bottom_right.u_px) / 2.0
    bottom_y = (bottom_left.v_px + bottom_right.v_px) / 2.0
    x_radius = max(3, int(round(0.16 * width)))
    y_start = max(0, int(math.floor(bottom_y + 1.0)))
    y_end = min(edge_mask.shape[0], int(math.ceil(bottom_y + 0.42 * height)))
    if y_end - y_start < 3:
        return False
    x0 = max(0, int(math.floor(center_x - x_radius)))
    x1 = min(edge_mask.shape[1], int(math.ceil(center_x + x_radius)) + 1)
    if x1 <= x0:
        return False
    rows = numpy.any(edge_mask[y_start:y_end, x0:x1] > 0, axis=1)
    # A post can be broken by the head/ground junction, but it must retain a
    # short contiguous run; isolated QR or radiator pixels do not suffice.
    required_run = max(3, int(math.ceil(0.12 * height)))
    run = 0
    for present in rows:
        run = run + 1 if present else 0
        if run >= required_run:
            return True
    return False


def _head_first_face_from_edges(
    cv2,
    raw_edges,
    *,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    fixed_parallel_side_direction: tuple[float, float] | None,
) -> _SilhouetteFaceCandidate | None:
    """Propose a square head from paired top/bottom raw edge segments.

    Unlike the legacy path, this does not require a long paired stem before
    locating the head.  The outer top/bottom lines seed a compact square-like
    candidate, then ``_raw_side_evidence_and_corners`` independently measures
    all four sides from untouched Canny pixels.
    """

    import numpy

    lines = cv2.HoughLinesP(
        raw_edges,
        rho=1,
        theta=numpy.pi / 180.0,
        # Compression and Canny gaps fragment the physical head border more
        # often than they fragment the much longer radiator edges.  Keep this
        # permissive only at proposal time; the raw four-side support gate and
        # the tight square envelope below remain the acceptance decision.
        threshold=10,
        minLineLength=max(8, int(round(1.25 * min_edge_height_px))),
        maxLineGap=max(8, int(round(1.75 * min_edge_height_px))),
    )
    if lines is None:
        return None

    horizontals: list[tuple[float, float, float, float, float]] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < max(8.0, 1.5 * min_edge_height_px):
            continue
        if abs(math.degrees(math.atan2(dy, dx))) > 28.0:
            continue
        left_x, right_x = sorted((float(x1), float(x2)))
        horizontals.append((left_x, right_x, float(y1), float(y2), length))
    if len(horizontals) < 2:
        return None

    best: _SilhouetteFaceCandidate | None = None
    best_score = -math.inf
    frame_height, frame_width = raw_edges.shape[:2]
    for upper in horizontals:
        for lower in horizontals:
            if upper is lower:
                continue
            upper_left, upper_right, upper_y1, upper_y2, upper_length = upper
            lower_left, lower_right, lower_y1, lower_y2, lower_length = lower
            upper_y = (upper_y1 + upper_y2) / 2.0
            lower_y = (lower_y1 + lower_y2) / 2.0
            if lower_y <= upper_y:
                continue
            height = lower_y - upper_y
            if not 2.0 * min_edge_height_px <= height <= 0.62 * frame_height:
                continue
            overlap = min(upper_right, lower_right) - max(upper_left, lower_left)
            # The lower outer head edge is commonly split by the centred stem.
            # A short outer fragment is enough to locate its row; the raw
            # four-side gate below must still prove both bottom intervals.
            if overlap < max(0.20 * upper_length, 1.5 * min_edge_height_px):
                continue
            mean_width = upper_length
            # Do not let a QR module or inner printed frame seed the outer
            # stand head.  The real outer frame spans several edge-height
            # gates even at the far workstation position.
            if mean_width < max(4.5 * min_edge_height_px, 0.20 * frame_width):
                continue
            aspect = mean_width / max(height, 1.0)
            # Head-first proposals are deliberately much tighter than the
            # generic legacy 0.45--1.8 detector envelope.
            if not max(0.65, min_aspect_ratio) <= aspect <= min(1.35, max_aspect_ratio):
                continue
            if mean_width > 0.72 * frame_width:
                continue
            rough = order_corners(
                (
                    ImagePoint(upper_left, upper_y1),
                    ImagePoint(upper_right, upper_y2),
                    ImagePoint(upper_right, lower_y2),
                    ImagePoint(upper_left, lower_y1),
                )
            )
            face_mask, fitted = _raw_side_evidence_and_corners(
                cv2,
                raw_edges,
                rough,
                fixed_parallel_side_direction=fixed_parallel_side_direction,
            )
            if fitted is None:
                continue
            fitted = order_corners(fitted)
            fitted_aspect = quadrilateral_aspect_ratio(fitted)
            if not max(0.65, min_aspect_ratio) <= fitted_aspect <= min(1.35, max_aspect_ratio):
                continue
            support = _quadrilateral_edge_support(cv2, face_mask, fitted)
            if not support.accepted:
                continue
            if not _short_centered_neck_support(raw_edges, fitted):
                continue
            square_score = 1.0 / (1.0 + abs(math.log(max(fitted_aspect, 1.0e-6))))
            score = 5.0 * support.mean + square_score + min(1.0, mean_width / max(height, 1.0))
            if score > best_score:
                best = _SilhouetteFaceCandidate(
                    corners=fitted,
                    face_mask=face_mask,
                    rectangle_fit_reliable=True,
                    rectangle_fit_reason="head_first_raw_rectangle_supported",
                )
                best_score = score
    return best
