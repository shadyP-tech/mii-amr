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


# A heater/radiator produces many nearly vertical Hough lines.  The real head
# still needs a global search on initial acquisition, but raw four-side fitting
# every combinatorial rail pair makes the latest-frame ROS subscriber fall
# behind.  The bound applies only after cheap geometry ranking; the selected
# proposals remain independently validated against untouched Canny pixels.
_MAX_SIDE_FIRST_RAW_VERIFICATIONS = 16
_MAX_HORIZONTAL_RAW_VERIFICATIONS = 6
_MAX_PROJECTIVE_HEAD_YAW_DEG = 70.0
_MIN_PROJECTED_HEAD_ASPECT = math.cos(
    math.radians(_MAX_PROJECTIVE_HEAD_YAW_DEG)
)


def _head_first_aspect_bounds(
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    *,
    projective_real_camera: bool,
) -> tuple[float, float]:
    """Return bounds for a frontal square or its perspective trapezoid."""

    minimum = (
        _MIN_PROJECTED_HEAD_ASPECT
        if projective_real_camera
        else max(0.65, min_aspect_ratio)
    )
    return minimum, min(1.35, max_aspect_ratio)


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
    neck = edge_mask[y_start:y_end, x0:x1] > 0
    # A post can be broken by the head/ground junction, but it must retain a
    # short contiguous run on *both* outer post rails.  Checking merely for a
    # foreground pixel admits QR modules directly below an inner QR rectangle.
    required_run = max(3, int(math.ceil(0.12 * height)))
    # Keep the two rails farther apart than the 1--3 px Canny thickness of a
    # single line, while still accepting the narrow physical post.
    min_rail_gap = max(3, int(round(0.07 * width)))
    max_rail_gap = max(min_rail_gap + 1, int(round(0.34 * width)))
    for left_column in range(neck.shape[1]):
        for right_column in range(left_column + min_rail_gap, neck.shape[1]):
            if right_column - left_column > max_rail_gap:
                break
            paired_rows = neck[:, left_column] & neck[:, right_column]
            run = 0
            for present in paired_rows:
                run = run + 1 if present else 0
                if run >= required_run:
                    return True
    return False


def _head_candidate_from_rough_corners(
    cv2,
    raw_edges,
    rough_corners,
    *,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    fixed_parallel_side_direction: tuple[float, float] | None,
    bounded_endpoint_recovery: bool,
    edge_points=None,
) -> tuple[_SilhouetteFaceCandidate, float] | None:
    """Verify a coarse head proposal only with raw four-side evidence."""

    face_mask, fitted = _raw_side_evidence_and_corners(
        cv2,
        raw_edges,
        rough_corners,
        fixed_parallel_side_direction=fixed_parallel_side_direction,
        edge_points=edge_points,
        real_camera_endpoint_fraction=(
            0.15 if bounded_endpoint_recovery else None
        ),
        maximum_parallel_side_length_ratio=(
            1.45 if bounded_endpoint_recovery else None
        ),
    )
    if fitted is None:
        return None
    fitted = order_corners(fitted)
    fitted_aspect = quadrilateral_aspect_ratio(fitted)
    minimum_aspect, maximum_aspect = _head_first_aspect_bounds(
        min_aspect_ratio,
        max_aspect_ratio,
        projective_real_camera=bounded_endpoint_recovery,
    )
    if not minimum_aspect <= fitted_aspect <= maximum_aspect:
        return None
    support = _quadrilateral_edge_support(cv2, face_mask, fitted)
    if not support.accepted or not _short_centered_neck_support(raw_edges, fitted):
        return None
    width = (_distance(fitted[0], fitted[1]) + _distance(fitted[3], fitted[2])) / 2.0
    height = (_distance(fitted[0], fitted[3]) + _distance(fitted[1], fitted[2])) / 2.0
    square_score = 1.0 / (1.0 + abs(math.log(max(fitted_aspect, 1.0e-6))))
    # Prefer the outer physical frame when inner QR rectangles also have four
    # edges.  Size is secondary to raw support, never a synthetic expansion.
    score = 5.0 * support.mean + square_score + min(1.0, width / max(height, 1.0))
    return (
        _SilhouetteFaceCandidate(
            corners=fitted,
            face_mask=face_mask,
            rectangle_fit_reliable=True,
            rectangle_fit_reason="head_first_raw_rectangle_supported",
        ),
        score,
    )


def _side_first_head_candidates(
    cv2,
    raw_edges,
    *,
    lines,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    fixed_parallel_side_direction: tuple[float, float] | None,
    bounded_endpoint_recovery: bool,
    edge_points,
):
    """Yield raw-verified head candidates seeded by the parallel outer rails.

    The real stand invariant is a pair of parallel side rails.  Top and bottom
    may be perspective-sloped and the lower edge may be interrupted by the
    post, so they are fitted later from raw Canny evidence rather than being a
    Hough prerequisite.
    """

    frame_height, frame_width = raw_edges.shape[:2]
    side_segments = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < max(2.0 * min_edge_height_px, 10.0):
            continue
        # This keeps the rail proposal independent of exact image vertical,
        # while excluding top/bottom-like segments.
        if abs(dy) < 0.55 * abs(dx):
            continue
        if dy < 0.0:
            x1, y1, x2, y2 = x2, y2, x1, y1
            dx, dy = -dx, -dy
        direction = (dx / length, dy / length)
        side_segments.append(
            (
                ImagePoint(float(x1), float(y1)),
                ImagePoint(float(x2), float(y2)),
                direction,
                length,
            )
        )

    ranked_pairs = []
    minimum_aspect, maximum_aspect = _head_first_aspect_bounds(
        min_aspect_ratio,
        max_aspect_ratio,
        projective_real_camera=bounded_endpoint_recovery,
    )
    minimum_rail_separation = max(
        2.0 * min_edge_height_px,
        (0.05 if bounded_endpoint_recovery else 0.16) * frame_width,
    )
    for index, left in enumerate(side_segments):
        for right in side_segments[index + 1 :]:
            left_top, left_bottom, left_direction, left_length = left
            right_top, right_bottom, right_direction, right_length = right
            direction_cosine = abs(
                left_direction[0] * right_direction[0]
                + left_direction[1] * right_direction[1]
            )
            if direction_cosine < math.cos(math.radians(14.0)):
                continue
            if max(left_length, right_length) / min(left_length, right_length) > 1.55:
                continue
            left_center_x = (left_top.u_px + left_bottom.u_px) / 2.0
            right_center_x = (right_top.u_px + right_bottom.u_px) / 2.0
            if abs(right_center_x - left_center_x) < minimum_rail_separation:
                continue
            if left_center_x > right_center_x:
                left_top, right_top = right_top, left_top
                left_bottom, right_bottom = right_bottom, left_bottom
            width = (_distance(left_top, right_top) + _distance(left_bottom, right_bottom)) / 2.0
            height = (left_length + right_length) / 2.0
            aspect = width / max(height, 1.0)
            if not minimum_aspect <= aspect <= maximum_aspect:
                continue
            if height > 0.62 * frame_height or width > 0.72 * frame_width:
                continue
            # Ranking remains intentionally cheap: it uses only the physical
            # invariant (parallel rails) and the square-like extent.  Raw
            # Canny support, including the interrupted bottom edge and neck,
            # remains the authoritative acceptance decision below.
            parallel_score = (direction_cosine - math.cos(math.radians(14.0))) / (
                1.0 - math.cos(math.radians(14.0))
            )
            square_score = 1.0 / (1.0 + abs(math.log(max(aspect, 1.0e-6))))
            length_balance = min(left_length, right_length) / max(left_length, right_length)
            extent_score = min(1.0, height / max(2.0 * min_edge_height_px, 1.0))
            pair_center_x = (left_center_x + right_center_x) / 2.0
            # The caller has already reduced a real-camera frame to its
            # operator-configured candidate-search ROI.  Favor proposals near
            # that ROI's centre so repeated radiator ribs at its edge cannot
            # crowd out the stand before raw validation.  Simulation keeps a
            # global, unbiased fallback.
            center_score = (
                max(0.0, 1.0 - abs(pair_center_x - frame_width / 2.0) / (frame_width / 2.0))
                if fixed_parallel_side_direction is None
                else 0.0
            )
            score = (
                2.0 * parallel_score
                + square_score
                + length_balance
                + 0.15 * extent_score
                + 1.5 * center_score
            )
            ranked_pairs.append(
                (
                    score,
                    left_top,
                    right_top,
                    right_bottom,
                    left_bottom,
                )
            )

    # Do not spend the full raw-side fitting cost on radiator pairings.  A
    # stable sort keeps deterministic behavior when equally ranked Hough lines
    # occur in adjacent frames.
    ranked_pairs.sort(key=lambda proposal: proposal[0], reverse=True)
    for _score, left_top, right_top, right_bottom, left_bottom in ranked_pairs[
        :_MAX_SIDE_FIRST_RAW_VERIFICATIONS
    ]:
        verified = _head_candidate_from_rough_corners(
            cv2,
            raw_edges,
            order_corners((left_top, right_top, right_bottom, left_bottom)),
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            fixed_parallel_side_direction=fixed_parallel_side_direction,
            bounded_endpoint_recovery=bounded_endpoint_recovery,
            edge_points=edge_points,
        )
        if verified is not None:
            yield verified


def _head_first_face_from_edges(
    cv2,
    proposal_edges,
    *,
    measurement_edges=None,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    fixed_parallel_side_direction: tuple[float, float] | None,
    bounded_endpoint_recovery: bool = False,
) -> _SilhouetteFaceCandidate | None:
    """Propose from filtered topology, then measure only that proposal in raw Canny.

    The proposal image may contain colour-adaptive/topology filtering, whereas
    ``measurement_edges`` remains untouched raw Canny.  This separation keeps
    a heater rail from originating a global head hypothesis while preserving
    real head-border evidence during the local four-side refit.
    """

    import numpy

    if measurement_edges is None:
        measurement_edges = proposal_edges
    if measurement_edges.shape[:2] != proposal_edges.shape[:2]:
        raise ValueError("measurement_edges must match proposal_edges")

    lines = cv2.HoughLinesP(
        proposal_edges,
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
    locations = cv2.findNonZero(measurement_edges)
    if locations is None:
        return None
    edge_points = locations.reshape(-1, 2).astype(numpy.float64)

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
    best: _SilhouetteFaceCandidate | None = None
    best_score = -math.inf
    frame_height, frame_width = proposal_edges.shape[:2]
    # In the real-camera path, two observed parallel outer rails are the
    # strongest stand-head invariant.  Do not let an equally supported
    # top/bottom Hough pair switch the frame to a QR/interior quadrilateral.
    side_best: _SilhouetteFaceCandidate | None = None
    side_best_score = -math.inf
    for candidate, score in _side_first_head_candidates(
        cv2,
        measurement_edges,
        lines=lines,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        fixed_parallel_side_direction=fixed_parallel_side_direction,
        bounded_endpoint_recovery=bounded_endpoint_recovery,
        edge_points=edge_points,
    ):
        if score > side_best_score:
            side_best = candidate
            side_best_score = score
    if side_best is not None and fixed_parallel_side_direction is None:
        return side_best
    ranked_horizontal_pairs = []
    minimum_aspect, maximum_aspect = _head_first_aspect_bounds(
        min_aspect_ratio,
        max_aspect_ratio,
        projective_real_camera=bounded_endpoint_recovery,
    )
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
            minimum_horizontal_width = max(
                (
                    2.0
                    if bounded_endpoint_recovery
                    else 4.5
                )
                * min_edge_height_px,
                (
                    0.05
                    if bounded_endpoint_recovery
                    else 0.16
                )
                * frame_width,
            )
            if mean_width < minimum_horizontal_width:
                continue
            aspect = mean_width / max(height, 1.0)
            # Head-first proposals are deliberately much tighter than the
            # generic legacy 0.45--1.8 detector envelope.
            if not minimum_aspect <= aspect <= maximum_aspect:
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
            square_score = 1.0 / (1.0 + abs(math.log(max(aspect, 1.0e-6))))
            overlap_score = overlap / max(min(upper_length, lower_length), 1.0)
            length_balance = min(upper_length, lower_length) / max(upper_length, lower_length)
            ranked_horizontal_pairs.append(
                (
                    square_score + overlap_score + length_balance,
                    rough,
                )
            )

    # Only the real-camera fallback is bounded: the fixed-direction simulation
    # path is deterministic and retains its exhaustive validation behavior.
    # The real path has already been restricted to the configured candidate
    # ROI and is the one exposed to radiator-like edge clutter.
    ranked_horizontal_pairs.sort(key=lambda proposal: proposal[0], reverse=True)
    horizontal_limit = (
        _MAX_HORIZONTAL_RAW_VERIFICATIONS
        if fixed_parallel_side_direction is None
        else len(ranked_horizontal_pairs)
    )
    for _rank, rough in ranked_horizontal_pairs[:horizontal_limit]:
        verified = _head_candidate_from_rough_corners(
            cv2,
            measurement_edges,
            rough,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            fixed_parallel_side_direction=fixed_parallel_side_direction,
            bounded_endpoint_recovery=bounded_endpoint_recovery,
            edge_points=edge_points,
        )
        if verified is None:
            continue
        candidate, score = verified
        if score > best_score:
            best = candidate
            best_score = score
    if side_best is not None and side_best_score > best_score:
        best = side_best
    return best
