"""Raw-Canny side fitting and evidence gates for the observed head frame."""

from __future__ import annotations

import math
from typing import Sequence

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _corners_inside_image,
    _distance,
    _polygon_area,
    order_corners,
    quadrilateral_aspect_ratio,
)
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint,
    _QuadrilateralEdgeSupport,
)


def _quadrilateral_edge_support(
    cv2,
    edge_mask,
    corners: Sequence[ImagePoint],
    *,
    tolerance_px: float | None = None,
) -> _QuadrilateralEdgeSupport:
    """Measure how much of every proposed side is backed by cutout pixels.

    Sampling uses a distance transform so a fitted sub-pixel line may be a
    few pixels away from a thick Canny edge.  End points are trimmed because
    rounded/dilated corners are noisy.  The bottom is sampled in two outer
    intervals so the central stem attachment is an allowed notch, not false
    evidence against an otherwise real head border.
    """

    import numpy

    ordered = order_corners(corners)
    top_left, top_right, bottom_right, bottom_left = ordered
    mean_width = (
        _distance(top_left, top_right) + _distance(bottom_left, bottom_right)
    ) / 2.0
    mean_height = (
        _distance(top_left, bottom_left) + _distance(top_right, bottom_right)
    ) / 2.0
    if tolerance_px is None:
        tolerance_px = min(5.0, max(2.0, 0.03 * min(mean_width, mean_height)))
    tolerance_px = float(tolerance_px)

    if edge_mask is None or edge_mask.size == 0 or cv2.countNonZero(edge_mask) == 0:
        return _QuadrilateralEdgeSupport(0.0, 0.0, 0.0, 0.0, 0.0, tolerance_px)

    edge_pixels = numpy.asarray(edge_mask) > 0
    distance_input = numpy.where(edge_pixels, 0, 255).astype(numpy.uint8)
    distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)
    height, width = distance.shape[:2]

    def segment_support(
        start: ImagePoint,
        end: ImagePoint,
        start_fraction: float,
        end_fraction: float,
    ) -> float:
        segment_length = _distance(start, end)
        sample_count = max(
            8,
            int(math.ceil(segment_length * max(0.0, end_fraction - start_fraction)))
            + 1,
        )
        fractions = numpy.linspace(start_fraction, end_fraction, sample_count)
        xs = numpy.rint(
            start.u_px + fractions * (end.u_px - start.u_px)
        ).astype(numpy.int32)
        ys = numpy.rint(
            start.v_px + fractions * (end.v_px - start.v_px)
        ).astype(numpy.int32)
        xs = numpy.clip(xs, 0, max(0, width - 1))
        ys = numpy.clip(ys, 0, max(0, height - 1))
        return float(numpy.mean(distance[ys, xs] <= tolerance_px))

    return _QuadrilateralEdgeSupport(
        top=segment_support(top_left, top_right, 0.08, 0.92),
        right=segment_support(top_right, bottom_right, 0.08, 0.92),
        bottom_left=segment_support(bottom_left, bottom_right, 0.08, 0.40),
        bottom_right=segment_support(bottom_left, bottom_right, 0.60, 0.92),
        left=segment_support(top_left, bottom_left, 0.08, 0.92),
        tolerance_px=tolerance_px,
    )


def _validated_refitted_head_corners(
    rough_corners: Sequence[ImagePoint],
    refitted_corners: Sequence[ImagePoint],
    *,
    image_shape,
    stem_center_x: float,
    stem_top_y: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    """Validate a border refit without silently substituting rough geometry."""

    rough = order_corners(rough_corners)
    refitted = order_corners(refitted_corners)
    if not _corners_inside_image(refitted, image_shape):
        return None

    rough_area = _polygon_area(rough)
    refitted_area = _polygon_area(refitted)
    # A close simulation contour can seed on the 53.147 mm white panel while
    # raw Canny correctly refits the 69.930 mm outer board. Their nominal area
    # ratio is about 1.73, so the former 1.60 ceiling rejected the true head.
    # The subsequent local-corner and four-side support gates still prevent a
    # connected arena wall from becoming a rectangle.
    if rough_area <= 0.0 or not 0.55 * rough_area <= refitted_area <= 2.10 * rough_area:
        return None

    aspect_ratio = quadrilateral_aspect_ratio(refitted)
    if not min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
        return None

    top_left, top_right, bottom_right, bottom_left = refitted
    top_width = _distance(top_left, top_right)
    bottom_width = _distance(bottom_left, bottom_right)
    if min(top_width, bottom_width) < 0.45 * max(top_width, bottom_width, 1e-6):
        return None

    rough_xs = [point.u_px for point in rough]
    rough_ys = [point.v_px for point in rough]
    rough_width = max(rough_xs) - min(rough_xs)
    rough_height = max(rough_ys) - min(rough_ys)
    rough_extent = max(rough_width, rough_height, 1.0)
    margin = max(4.0, 0.30 * rough_extent)
    if any(
        point.u_px < min(rough_xs) - margin
        or point.u_px > max(rough_xs) + margin
        or point.v_px < min(rough_ys) - margin
        or point.v_px > max(rough_ys) + margin
        for point in refitted
    ):
        return None

    rough_center_x = (min(rough_xs) + max(rough_xs)) / 2.0
    rough_center_y = (min(rough_ys) + max(rough_ys)) / 2.0
    refitted_xs = [point.u_px for point in refitted]
    refitted_ys = [point.v_px for point in refitted]
    refitted_center_x = (min(refitted_xs) + max(refitted_xs)) / 2.0
    refitted_center_y = (min(refitted_ys) + max(refitted_ys)) / 2.0
    if math.hypot(
        refitted_center_x - rough_center_x,
        refitted_center_y - rough_center_y,
    ) > 0.25 * rough_extent:
        return None

    candidate_left = min(refitted_xs)
    candidate_right = max(refitted_xs)
    candidate_width = candidate_right - candidate_left
    if not (
        candidate_left + 0.12 * candidate_width
        <= stem_center_x
        <= candidate_right - 0.12 * candidate_width
    ):
        return None
    candidate_top = min(refitted_ys)
    candidate_bottom = max(refitted_ys)
    candidate_height = max(candidate_bottom - candidate_top, 1.0)
    if stem_top_y < candidate_bottom - 0.15 * candidate_height:
        return None
    if stem_top_y > candidate_bottom + 0.50 * candidate_height:
        return None
    return refitted


def _select_supported_head_corners(
    cv2,
    face_mask,
    rough_corners: Sequence[ImagePoint],
    refitted_corners: Sequence[ImagePoint] | None,
    *,
    image_shape,
    stem_center_x: float,
    stem_top_y: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    allow_rough_fallback: bool = True,
) -> tuple[
    tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None,
    str,
    _QuadrilateralEdgeSupport | None,
]:
    """Choose only geometry independently supported by the real cutout."""

    best_support: _QuadrilateralEdgeSupport | None = None
    if refitted_corners is not None:
        refitted = _validated_refitted_head_corners(
            rough_corners,
            refitted_corners,
            image_shape=image_shape,
            stem_center_x=stem_center_x,
            stem_top_y=stem_top_y,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )
        if refitted is not None:
            support = _quadrilateral_edge_support(cv2, face_mask, refitted)
            best_support = support
            if support.accepted:
                return refitted, "refitted_rectangle_edge_supported", support

    if allow_rough_fallback:
        # A topology-localized proposal may be retained only after the separate
        # evidence mask confirms all four proposed sides.  In the dual-edge
        # pipeline that mask contains untouched raw-Canny pixels selected in
        # narrow side bands, so morphology still cannot create pose evidence.
        rough = _validated_refitted_head_corners(
            rough_corners,
            rough_corners,
            image_shape=image_shape,
            stem_center_x=stem_center_x,
            stem_top_y=stem_top_y,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )
        if rough is not None:
            rough_support = _quadrilateral_edge_support(cv2, face_mask, rough)
            if best_support is None or rough_support.mean > best_support.mean:
                best_support = rough_support
            if rough_support.accepted:
                return rough, "rough_rectangle_edge_supported", rough_support

    return None, "head_rectangle_fit_unreliable", best_support

def _fit_raw_edge_side_in_band(
    cv2,
    edge_points,
    start: ImagePoint,
    end: ImagePoint,
    *,
    band_px: float,
    intervals: Sequence[tuple[float, float]],
    outward_sign: float = 0.0,
    fixed_direction: tuple[float, float] | None = None,
    minimum_coverage: float = 0.45,
):
    """Fit one proposed side from nearby raw pixels without connectivity.

    The topology side supplies only a search band. For every pixel-sized bin
    along that side, one untouched Canny pixel is selected by the requested
    outer/nearest policy and a robust line is fitted through those samples.
    This prevents a closed topology mask from measuring its own proposal.
    """

    import numpy

    dx = end.u_px - start.u_px
    dy = end.v_px - start.v_px
    length = math.hypot(dx, dy)
    if length <= 1e-6 or len(edge_points) == 0:
        return None, edge_points[:0]
    tangent = numpy.array([dx / length, dy / length], dtype=numpy.float64)
    normal = numpy.array([-tangent[1], tangent[0]], dtype=numpy.float64)
    relative = edge_points - numpy.array(
        [start.u_px, start.v_px],
        dtype=numpy.float64,
    )
    along_px = relative @ tangent
    along_fraction = along_px / length
    normal_offset = relative @ normal
    interval_mask = numpy.zeros(len(edge_points), dtype=bool)
    expected_length_px = 0.0
    for interval_start, interval_end in intervals:
        interval_mask |= (
            (along_fraction >= interval_start)
            & (along_fraction <= interval_end)
        )
        expected_length_px += max(0.0, interval_end - interval_start) * length
    candidate_mask = interval_mask & (numpy.abs(normal_offset) <= band_px)
    candidate_indices = numpy.flatnonzero(candidate_mask)
    candidates = edge_points[candidate_indices]
    if len(candidates) == 0:
        return None, candidates

    # A QR edge parallel to the outer border can enter the wider search band.
    # For known silhouette sides, retain the outermost raw pixel in each
    # tangent bin. This makes the topology proposal a search corridor only;
    # an interior QR edge cannot win merely because it lies closer to the
    # proposal. Legacy callers may still request closest-to-proposal samples
    # with outward_sign=0.
    # Do not use numpy.rint here: its ties-to-even rule collapses adjacent
    # integer pixels whenever the fitted line centre lies on a half pixel.
    candidate_bins = numpy.floor(
        along_px[candidate_indices] + 0.5
    ).astype(numpy.int32)
    candidate_offsets = normal_offset[candidate_indices]
    nearest_by_bin: dict[int, int] = {}
    if outward_sign:
        sample_order = numpy.argsort(-outward_sign * candidate_offsets)
    else:
        sample_order = numpy.argsort(numpy.abs(candidate_offsets))
    for local_index in sample_order:
        nearest_by_bin.setdefault(
            int(candidate_bins[local_index]),
            int(local_index),
        )
    selected = candidates[list(nearest_by_bin.values())]
    minimum_points = max(5, int(math.ceil(0.18 * expected_length_px)))
    if len(selected) < minimum_points:
        return None, candidates

    def robust_fit(points):
        fitted = cv2.fitLine(
            points.astype(numpy.float32).reshape(-1, 1, 2),
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        ).reshape(-1)
        direction = numpy.array(
            [float(fitted[0]), float(fitted[1])],
            dtype=numpy.float64,
        )
        direction_norm = float(numpy.linalg.norm(direction))
        if direction_norm <= 1e-9:
            return None
        direction /= direction_norm
        if float(direction @ tangent) < 0.0:
            direction *= -1.0
        point = numpy.array(
            [float(fitted[2]), float(fitted[3])],
            dtype=numpy.float64,
        )
        return point, direction

    def fixed_direction_fit(points):
        direction = numpy.asarray(fixed_direction, dtype=numpy.float64).reshape(2)
        direction_norm = float(numpy.linalg.norm(direction))
        if direction_norm <= 1e-9:
            return None
        direction /= direction_norm
        if float(direction @ tangent) < 0.0:
            direction *= -1.0
        angle_cosine = max(-1.0, min(1.0, float(direction @ tangent)))
        if math.degrees(math.acos(angle_cosine)) > 22.0:
            return None
        fit_normal = numpy.array(
            [-direction[1], direction[0]],
            dtype=numpy.float64,
        )
        offsets = points @ fit_normal
        offset = float(numpy.median(offsets))
        residuals = numpy.abs(offsets - offset)
        inlier_limit_px = max(1.25, min(2.5, 0.50 * band_px))
        inliers = points[residuals <= inlier_limit_px]
        if len(inliers) < minimum_points:
            return None
        offset = float(numpy.median(inliers @ fit_normal))
        along_center = float(numpy.median(inliers @ direction))
        point = along_center * direction + offset * fit_normal
        return point, direction

    first_fit = (
        fixed_direction_fit(selected)
        if fixed_direction is not None
        else robust_fit(selected)
    )
    if first_fit is None:
        return None, candidates
    fit_point, fit_direction = first_fit
    angle_cosine = max(-1.0, min(1.0, float(fit_direction @ tangent)))
    # A minimum-area/topology rectangle can be noticeably more axis-aligned
    # than the true perspective edge.  Keep enough angular freedom to recover
    # that raw edge while the narrow spatial band still rejects unrelated
    # structure.
    if fixed_direction is None and math.degrees(math.acos(angle_cosine)) > 22.0:
        return None, candidates

    fit_normal = numpy.array(
        [-fit_direction[1], fit_direction[0]],
        dtype=numpy.float64,
    )
    residuals = numpy.abs((selected - fit_point) @ fit_normal)
    inlier_limit_px = max(1.25, min(2.5, 0.50 * band_px))
    inliers = selected[residuals <= inlier_limit_px]
    if len(inliers) < minimum_points:
        return None, candidates
    final_fit = (
        fixed_direction_fit(inliers)
        if fixed_direction is not None
        else robust_fit(inliers)
    )
    if final_fit is None:
        return None, candidates
    fit_point, fit_direction = final_fit
    angle_cosine = max(-1.0, min(1.0, float(fit_direction @ tangent)))
    if fixed_direction is None and math.degrees(math.acos(angle_cosine)) > 20.0:
        return None, candidates

    fit_normal = numpy.array(
        [-fit_direction[1], fit_direction[0]],
        dtype=numpy.float64,
    )
    midpoint = numpy.array(
        [(start.u_px + end.u_px) / 2.0, (start.v_px + end.v_px) / 2.0],
        dtype=numpy.float64,
    )
    if abs(float((midpoint - fit_point) @ fit_normal)) > band_px + 1.0:
        return None, candidates

    candidate_residuals = numpy.abs((candidates - fit_point) @ fit_normal)
    evidence = candidates[candidate_residuals <= inlier_limit_px]
    projected_bins = numpy.unique(
        numpy.floor(
            (evidence - fit_point) @ fit_direction + 0.5
        ).astype(numpy.int32)
    )
    coverage = len(projected_bins) / max(expected_length_px, 1.0)
    if coverage < minimum_coverage:
        return None, evidence
    return (
        (
            float(fit_point[0]),
            float(fit_point[1]),
            float(fit_direction[0]),
            float(fit_direction[1]),
        ),
        evidence,
    )


def _image_line_intersection(first, second) -> ImagePoint | None:
    first_x, first_y, first_dx, first_dy = first
    second_x, second_y, second_dx, second_dy = second
    denominator = first_dx * second_dy - first_dy * second_dx
    if abs(denominator) < 0.15:
        return None
    offset_x = second_x - first_x
    offset_y = second_y - first_y
    scale = (offset_x * second_dy - offset_y * second_dx) / denominator
    x = first_x + scale * first_dx
    y = first_y + scale * first_dy
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return ImagePoint(float(x), float(y))


def _parallel_side_run_endpoints(
    edge_points,
    fitted_line,
    rough_start: ImagePoint,
    rough_end: ImagePoint,
    side_evidence,
    *,
    band_px: float,
):
    """Recover one head side's real endpoints from its aligned raw-edge run.

    A row-envelope proposal can be horizontally topped even when perspective
    makes the real head top strongly sloped. Once an outer side has been fitted
    with the calibrated parallel direction, its contiguous raw-Canny run gives
    the two corner heights without relying on that inaccurate top proposal.
    The search stays local to the rough side and must overlap the evidence that
    supported the fitted line, preventing a separate stem/base run from winning.
    """

    import numpy

    line_x, line_y, direction_x, direction_y = fitted_line
    direction = numpy.array([direction_x, direction_y], dtype=numpy.float64)
    direction_norm = float(numpy.linalg.norm(direction))
    if direction_norm <= 1e-9:
        return None
    direction /= direction_norm
    rough_tangent = numpy.array(
        [
            rough_end.u_px - rough_start.u_px,
            rough_end.v_px - rough_start.v_px,
        ],
        dtype=numpy.float64,
    )
    rough_length = float(numpy.linalg.norm(rough_tangent))
    if rough_length <= 1e-6:
        return None
    if float(direction @ rough_tangent) < 0.0:
        direction *= -1.0

    fit_point = numpy.array([line_x, line_y], dtype=numpy.float64)
    fit_normal = numpy.array([-direction[1], direction[0]], dtype=numpy.float64)
    along = (edge_points - fit_point) @ direction
    residuals = numpy.abs((edge_points - fit_point) @ fit_normal)
    rough_along = numpy.array(
        [
            (numpy.array([rough_start.u_px, rough_start.v_px]) - fit_point)
            @ direction,
            (numpy.array([rough_end.u_px, rough_end.v_px]) - fit_point)
            @ direction,
        ],
        dtype=numpy.float64,
    )
    search_margin = max(4.0, 0.35 * rough_length)
    residual_limit_px = max(1.25, min(2.5, 0.50 * band_px))
    local_mask = (
        (residuals <= residual_limit_px)
        & (along >= float(rough_along.min()) - search_margin)
        & (along <= float(rough_along.max()) + search_margin)
    )
    local_points = edge_points[local_mask]
    local_along = along[local_mask]
    if len(local_points) < 5:
        return None

    local_bins = numpy.floor(local_along + 0.5).astype(numpy.int32)
    unique_bins = numpy.unique(local_bins)
    if len(unique_bins) < 5:
        return None
    # Arena-wall suppression can cross an otherwise continuous outer side at
    # the wall/stand depth discontinuity.  In the live simulation capture it
    # removed roughly one quarter of the near side while leaving both the
    # upper run and the real bottom corner.  Keep those fragments in one local
    # side run.  The search is still constrained to the fitted line, to the
    # rough head neighbourhood, and to a run overlapping the original side
    # evidence, so this cannot jump across to the stem or another stand.
    maximum_gap_bins = max(3, int(math.ceil(0.30 * rough_length)))
    runs: list[tuple[int, int]] = []
    run_start = run_end = int(unique_bins[0])
    for raw_bin in unique_bins[1:]:
        current_bin = int(raw_bin)
        if current_bin - run_end > maximum_gap_bins:
            runs.append((run_start, run_end))
            run_start = current_bin
        run_end = current_bin
    runs.append((run_start, run_end))

    if len(side_evidence):
        evidence_bins = numpy.floor(
            (side_evidence - fit_point) @ direction + 0.5
        ).astype(numpy.int32)
    else:
        evidence_bins = numpy.empty((0,), dtype=numpy.int32)

    def run_rank(run: tuple[int, int]) -> tuple[int, int, int]:
        start_bin, end_bin = run
        overlap = int(
            numpy.count_nonzero(
                (evidence_bins >= start_bin) & (evidence_bins <= end_bin)
            )
        )
        span = end_bin - start_bin
        support = int(
            numpy.count_nonzero(
                (local_bins >= start_bin) & (local_bins <= end_bin)
            )
        )
        return overlap, span, support

    selected_start, selected_end = max(runs, key=run_rank)
    selected_overlap, selected_span, _selected_support = run_rank(
        (selected_start, selected_end)
    )
    minimum_overlap = max(5, int(math.ceil(0.25 * len(evidence_bins))))
    if selected_overlap < minimum_overlap:
        return None
    if selected_span < max(8.0, 0.55 * rough_length):
        return None

    run_mask = (local_bins >= selected_start) & (local_bins <= selected_end)
    run_points = local_points[run_mask]
    start_point = fit_point + float(selected_start) * direction
    end_point = fit_point + float(selected_end) * direction
    return (
        ImagePoint(float(start_point[0]), float(start_point[1])),
        ImagePoint(float(end_point[0]), float(end_point[1])),
        run_points,
    )


def _level_camera_endpoint_perspective_consistent(
    left_top: ImagePoint,
    left_bottom: ImagePoint,
    right_top: ImagePoint,
    right_bottom: ImagePoint,
    *,
    parallel_side_direction: tuple[float, float] = (0.0, 1.0),
) -> bool:
    """Reject a wall crossing masquerading as one parallel-side endpoint.

    The supplied direction is the common left/right side direction in image
    coordinates.  Projecting both cross-board edges onto that direction keeps
    the original level-camera perspective rule valid for a rolled camera:
    the upper edge has at least as much projected slope as the lower edge.
    A truncated side ending on the arena seam reverses that relationship.
    """

    direction_x, direction_y = (
        float(component) for component in parallel_side_direction
    )
    direction_norm = math.hypot(direction_x, direction_y)
    if (
        not math.isfinite(direction_norm)
        or direction_norm <= 1e-9
    ):
        raise ValueError(
            "parallel_side_direction must be a finite non-zero 2D direction"
        )
    direction_x /= direction_norm
    direction_y /= direction_norm

    top_delta_parallel = (
        (right_top.u_px - left_top.u_px) * direction_x
        + (right_top.v_px - left_top.v_px) * direction_y
    )
    bottom_delta_parallel = (
        (right_bottom.u_px - left_bottom.u_px) * direction_x
        + (right_bottom.v_px - left_bottom.v_px) * direction_y
    )
    mean_height = (
        _distance(left_top, left_bottom)
        + _distance(right_top, right_bottom)
    ) / 2.0
    tolerance_px = max(2.0, 0.04 * mean_height)
    if (
        top_delta_parallel * bottom_delta_parallel < 0.0
        and min(abs(top_delta_parallel), abs(bottom_delta_parallel))
        > tolerance_px
    ):
        return False
    return (
        abs(bottom_delta_parallel)
        <= abs(top_delta_parallel) + tolerance_px
    )


def _raw_side_evidence_and_corners(
    cv2,
    raw_edges,
    rough_corners: Sequence[ImagePoint],
    *,
    fixed_parallel_side_direction: tuple[float, float] | None = None,
):
    """Refit a common-sided head trapezoid from outer raw-Canny evidence.

    The left and right sides share one image direction. A calibrated caller
    may supply that direction exactly; other callers estimate one robust
    common direction from both raw sides. Top and bottom retain their own
    perspective slopes.
    """

    import numpy

    evidence_mask = numpy.zeros(raw_edges.shape[:2], dtype=numpy.uint8)
    locations = cv2.findNonZero(raw_edges)
    if locations is None:
        return evidence_mask, None
    edge_points = locations.reshape(-1, 2).astype(numpy.float64)
    top_left, top_right, bottom_right, bottom_left = order_corners(rough_corners)
    widths = (
        _distance(top_left, top_right),
        _distance(bottom_left, bottom_right),
    )
    heights = (
        _distance(top_left, bottom_left),
        _distance(top_right, bottom_right),
    )
    minimum_extent = min((*widths, *heights))
    band_px = float(max(3, min(6, int(round(0.08 * minimum_extent)))))
    parallel_side_band_px = band_px
    if fixed_parallel_side_direction is not None:
        # At close simulation viewpoints the connected topology can describe
        # the white-panel boundary rather than the board silhouette.  The
        # panel is 53.147 mm wide inside a 69.930 mm board, so either real
        # outer side may be about 15.8% of the panel width beyond that rough
        # proposal. Rasterized topology can sit another 2-3 px inside the panel
        # edge, so retain a 24% corridor for that localization error. Scale
        # only the calibrated parallel-side search; top/bottom retain the
        # narrow wall-safe corridor.
        parallel_side_band_px = float(
            max(
                band_px,
                min(24, int(round(0.24 * minimum_extent))),
            )
        )
    side_specs = {
        # Top and bottom both run left-to-right, so their fitted normal points
        # down in image coordinates. The silhouette exterior is therefore the
        # negative-normal side for the top and the positive-normal side for the
        # bottom. Reversing these signs selects inner frame/QR edges instead.
        "top": (top_left, top_right, ((0.08, 0.92),), -1.0, 0.50),
        "right": (top_right, bottom_right, ((0.08, 0.92),), -1.0, 0.55),
        # The stand stem legitimately interrupts the middle of the lower edge.
        "bottom": (
            bottom_left,
            bottom_right,
            ((0.08, 0.40), (0.60, 0.92)),
            1.0,
            0.40,
        ),
        "left": (top_left, bottom_left, ((0.08, 0.92),), 1.0, 0.55),
    }

    def fit_side(name: str, *, fixed_direction=None):
        start, end, intervals, outward_sign, minimum_coverage = side_specs[name]
        side_band_px = (
            parallel_side_band_px
            if name in ("left", "right")
            else band_px
        )
        return _fit_raw_edge_side_in_band(
            cv2,
            edge_points,
            start,
            end,
            band_px=side_band_px,
            intervals=intervals,
            outward_sign=outward_sign,
            fixed_direction=fixed_direction,
            minimum_coverage=minimum_coverage,
        )

    top, top_evidence = fit_side("top")
    bottom, bottom_evidence = fit_side("bottom")

    shared_direction = fixed_parallel_side_direction
    if shared_direction is None:
        preliminary_right, _right_evidence = fit_side("right")
        preliminary_left, _left_evidence = fit_side("left")
        if preliminary_right is not None and preliminary_left is not None:
            right_direction = numpy.array(preliminary_right[2:4], dtype=numpy.float64)
            left_direction = numpy.array(preliminary_left[2:4], dtype=numpy.float64)
            if float(right_direction @ left_direction) < 0.0:
                left_direction *= -1.0
            direction_cosine = max(
                -1.0,
                min(1.0, float(right_direction @ left_direction)),
            )
            if math.degrees(math.acos(direction_cosine)) <= 12.0:
                combined = right_direction + left_direction
                combined_norm = float(numpy.linalg.norm(combined))
                if combined_norm > 1e-9:
                    combined /= combined_norm
                    shared_direction = (float(combined[0]), float(combined[1]))

    if shared_direction is None:
        right = left = None
        right_evidence = left_evidence = edge_points[:0]
    else:
        right, right_evidence = fit_side(
            "right",
            fixed_direction=shared_direction,
        )
        left, left_evidence = fit_side(
            "left",
            fixed_direction=shared_direction,
        )

    parallel_endpoint_corners = None
    if (
        fixed_parallel_side_direction is not None
        and right is not None
        and left is not None
    ):
        # In the level simulation camera, the two fitted outer sides are the
        # most stable head evidence. Recover their complete raw runs and use
        # the endpoint pairs as accurate proposals for the perspective-sloped
        # top and bottom. This repairs row-envelope proposals without accepting
        # topology-only geometry or inferring an unsupported fourth side.
        left_run = _parallel_side_run_endpoints(
            edge_points,
            left,
            top_left,
            bottom_left,
            left_evidence,
            band_px=parallel_side_band_px,
        )
        right_run = _parallel_side_run_endpoints(
            edge_points,
            right,
            top_right,
            bottom_right,
            right_evidence,
            band_px=parallel_side_band_px,
        )
        if left_run is not None and right_run is not None:
            left_top, left_bottom, recovered_left_evidence = left_run
            right_top, right_bottom, recovered_right_evidence = right_run
            if not _level_camera_endpoint_perspective_consistent(
                left_top,
                left_bottom,
                right_top,
                right_bottom,
                parallel_side_direction=fixed_parallel_side_direction,
            ):
                left_run = right_run = None
            else:
                recovered_top, recovered_top_evidence = _fit_raw_edge_side_in_band(
                    cv2,
                    edge_points,
                    left_top,
                    right_top,
                    band_px=band_px,
                    intervals=((0.04, 0.96),),
                    outward_sign=0.0,
                    minimum_coverage=0.50,
                )
                recovered_bottom, recovered_bottom_evidence = (
                    _fit_raw_edge_side_in_band(
                        cv2,
                        edge_points,
                        left_bottom,
                        right_bottom,
                        band_px=band_px,
                        intervals=((0.04, 0.42), (0.58, 0.96)),
                        outward_sign=0.0,
                        minimum_coverage=0.40,
                    )
                )
                if recovered_top is not None and recovered_bottom is not None:
                    top = recovered_top
                    bottom = recovered_bottom
                    top_evidence = recovered_top_evidence
                    bottom_evidence = recovered_bottom_evidence
                    left_evidence = recovered_left_evidence
                    right_evidence = recovered_right_evidence
                    # The side-run endpoints are directly observed corner pixels.
                    # Keep them as the final corners; intersecting an infinite top
                    # fit extrapolated from only its middle fragment can otherwise
                    # overshoot a rounded outer corner by several pixels.
                    parallel_endpoint_corners = (
                        left_top,
                        right_top,
                        right_bottom,
                        left_bottom,
                    )

    lines = (top, right, bottom, left)
    evidences = (top_evidence, right_evidence, bottom_evidence, left_evidence)
    for line, evidence in zip(lines, evidences):
        if len(evidence):
            xs = numpy.rint(evidence[:, 0]).astype(numpy.int32)
            ys = numpy.rint(evidence[:, 1]).astype(numpy.int32)
            valid = (
                (xs >= 0)
                & (xs < evidence_mask.shape[1])
                & (ys >= 0)
                & (ys < evidence_mask.shape[0])
            )
            evidence_mask[ys[valid], xs[valid]] = 255

    if any(line is None for line in lines):
        return evidence_mask, None
    if parallel_endpoint_corners is not None:
        ordered_endpoint_corners = order_corners(parallel_endpoint_corners)
        if _quadrilateral_edge_support(
            cv2,
            evidence_mask,
            ordered_endpoint_corners,
        ).accepted:
            return evidence_mask, ordered_endpoint_corners
    corners = (
        _image_line_intersection(top, left),
        _image_line_intersection(top, right),
        _image_line_intersection(bottom, right),
        _image_line_intersection(bottom, left),
    )
    if any(point is None for point in corners):
        return evidence_mask, None
    ordered_corners = order_corners(corners)
    if (
        fixed_parallel_side_direction is not None
        and not _level_camera_endpoint_perspective_consistent(
            ordered_corners[0],
            ordered_corners[3],
            ordered_corners[1],
            ordered_corners[2],
            parallel_side_direction=fixed_parallel_side_direction,
        )
    ):
        return evidence_mask, None
    return evidence_mask, ordered_corners
