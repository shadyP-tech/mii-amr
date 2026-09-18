"""Select a coherent observed rail inside a metric projection corridor."""

from __future__ import annotations

from dataclasses import dataclass
import math


def coherent_metric_rail_points(
    cv2,
    candidates,
    candidate_bins,
    start,
    end,
    *,
    band_px: float,
    expected_length_px: float,
    minimum_coverage: float,
    fixed_direction=None,
    outward_sign=0.,
    color_support_mask=None,
):
    """Associate whole raw-edge rails before selecting one pixel per bin.

    Every returned point comes from ``candidates``, already clipped to the
    caller's side corridor and sampling intervals. Hough segments only propose
    lines: their interpolated pixels are never used as measurement evidence.
    Coverage of a coherent rail takes precedence over isolated pixels closer
    to the prediction. Comparable rails use projection distance to break ties;
    indistinguishable, spatially separate rails remain unobservable.
    A position-bounded physical head may instead request the enclosing rail,
    requiring at least 80 percent raw coverage in the same finite corridor.
    Colour support is a soft ranking cue worth at most one pixel, never a mask
    on returned evidence or a replacement for the raw corner checks.
    """

    import numpy

    if len(candidates) < 5:
        return candidates[:0]
    origin = numpy.floor(candidates.min(axis=0)).astype(numpy.int32)
    pixels = numpy.rint(candidates - origin).astype(numpy.int32)
    local_edges = numpy.zeros(
        (int(pixels[:, 1].max()) + 1, int(pixels[:, 0].max()) + 1),
        dtype=numpy.uint8,
    )
    local_edges[pixels[:, 1], pixels[:, 0]] = 255
    segments = cv2.HoughLinesP(
        local_edges,
        1.0,
        numpy.pi / 180.0,
        threshold=max(5, int(0.18 * expected_length_px)),
        minLineLength=max(4.0, 0.20 * expected_length_px),
        maxLineGap=max(2.0, min(5.0, 0.04 * expected_length_px)),
    )
    start_xy = numpy.array((start.u_px, start.v_px), dtype=numpy.float64)
    end_xy = numpy.array((end.u_px, end.v_px), dtype=numpy.float64)
    midpoint = (start_xy + end_xy) / 2.0
    tangent = end_xy - start_xy
    tangent /= numpy.linalg.norm(tangent)
    reference_direction = (
        tangent
        if fixed_direction is None
        else numpy.asarray(fixed_direction, dtype=numpy.float64)
    )
    reference_direction = reference_direction / numpy.linalg.norm(reference_direction)
    proposals = []
    if segments is not None:
        for segment in segments.reshape(-1, 4):
            proposals.append(
                (
                    segment[:2].astype(numpy.float64) + origin,
                    (segment[2:] - segment[:2]).astype(numpy.float64),
                )
            )
    # The lower head edge has two separate visible intervals around the stem.
    # Short Hough segments can quantize their slopes differently and lose one
    # interval on extrapolation. Also test the predicted/common direction at
    # offsets actually observed in this corridor; both intervals still need
    # independent raw support before the final quadrilateral can be accepted.
    reference_normal = numpy.array(
        (-reference_direction[1], reference_direction[0])
    )
    offsets = (candidates - midpoint) @ reference_normal
    offset_bins = numpy.floor(offsets + 0.5).astype(numpy.int32)
    for offset_bin in numpy.unique(offset_bins):
        offset = float(numpy.median(offsets[offset_bins == offset_bin]))
        if abs(offset) <= band_px:
            proposals.append(
                (midpoint + offset * reference_normal, reference_direction)
            )
    hypotheses = []
    for point, proposal_direction in proposals:
        direction = proposal_direction.copy()
        length = float(numpy.linalg.norm(direction))
        if length <= 0.0:
            continue
        direction /= length
        if float(direction @ tangent) < 0.0:
            direction *= -1.0
        angle = math.degrees(
            math.acos(min(1.0, abs(float(direction @ reference_direction))))
        )
        if angle > (12.0 if fixed_direction is not None else 20.0):
            continue
        normal = numpy.array((-direction[1], direction[0]))
        distance = abs(float((midpoint - point) @ normal))
        if distance > band_px:
            continue
        # A tight proposal band avoids combining two different parallel rails.
        # The existing robust fit and its coverage gates run on these raw
        # samples afterwards; this does not replace their acceptance checks.
        residuals = numpy.abs((candidates - point) @ normal)
        selected = residuals <= 1.25
        coverage = len(numpy.unique(candidate_bins[selected])) / max(
            expected_length_px, 1.0
        )
        if coverage < minimum_coverage:
            continue
        # Ignore a few rasterization bins when comparing complete alternatives.
        coverage_rank = round(min(1.0, coverage) * 20.0) / 20.0
        colour = 0.
        if color_support_mask is not None:
            locations = numpy.rint(candidates[selected]).astype(numpy.int32)
            colour = float(numpy.mean(color_support_mask[locations[:, 1], locations[:, 0]] > 0))
        if outward_sign and coverage >= .80:
            signed_offset = float((point-midpoint) @ reference_normal)*outward_sign
            # A soft hint worth at most one pixel of outward displacement.
            # Missing grey/overexposed mask support cannot delete a raw rail.
            rank = (1., signed_offset+colour, coverage_rank, -float(numpy.median(residuals[selected])))
        elif outward_sign:
            continue
        else:
            rank = (coverage_rank, -distance+colour, -float(numpy.median(residuals[selected])))
        hypotheses.append((rank, point, normal, selected))

    if not hypotheses:
        return candidates[:0]
    hypotheses.sort(key=lambda item: item[0], reverse=True)
    best_rank, best_point, best_normal, best_selected = hypotheses[0]
    for rank, point, normal, _selected in hypotheses[1:]:
        if rank[0] != best_rank[0] or abs(rank[1] - best_rank[1]) > 0.75:
            continue
        # Do not invent a measured rail halfway between equally plausible
        # parallel edges. Nearby Canny bands on the same rail may co-exist.
        best_nearest = midpoint + float((best_point - midpoint) @ best_normal) * best_normal
        other_nearest = midpoint + float((point - midpoint) @ normal) * normal
        # Two 2.5 px localization bands can describe the same thick Canny rail.
        if float(numpy.linalg.norm(best_nearest - other_nearest)) > 5.0:
            return candidates[:0]
    return candidates[best_selected]


@dataclass(frozen=True)
class MetricCornerArmSupport:
    """Raw support for the two incident rails at every semantic head corner."""

    radius_px: float
    minimum_bins: int
    bins_by_corner: dict[str, dict[str, int]]

    @property
    def accepted(self) -> bool:
        return all(
            count >= self.minimum_bins
            for arms in self.bins_by_corner.values()
            for count in arms.values()
        )


def metric_corner_arm_support(cv2, raw_edges, corners) -> MetricCornerArmSupport:
    """Count observed bins on both arms without discarding failed evidence.

    Long window/radiator segments may satisfy the trimmed side tests while
    their infinite-line intersections have no physical head corner nearby.
    Check the untouched image here: the side evidence mask deliberately omits
    endpoints. A corner pixel itself is not required, since heads are rounded.
    Counts use the original 1.5 px minimum longitudinal distance, bounded
    4–6 px radius, 2.5 px perpendicular allowance, and rounded pixel bins.
    """

    import numpy

    from scripts.aufgabe04.perception.stand_axis.geometry import _distance

    shortest_side = min(
        _distance(first, second)
        for first, second in zip(corners, corners[1:] + corners[:1])
    )
    radius = max(4.0, min(6.0, 0.08 * shortest_side))
    margin = int(math.ceil(radius + 2.0))
    height, width = raw_edges.shape[:2]
    names_and_arms = (
        ("head_top_left", ("left", "top")),
        ("head_top_right", ("top", "right")),
        ("head_bottom_right", ("right", "bottom")),
        ("head_bottom_left", ("bottom", "left")),
    )
    bins_by_corner = {}
    for index, corner in enumerate(corners):
        x0 = max(0, int(math.floor(corner.u_px)) - margin)
        y0 = max(0, int(math.floor(corner.v_px)) - margin)
        x1 = min(width, int(math.ceil(corner.u_px)) + margin + 1)
        y1 = min(height, int(math.ceil(corner.v_px)) + margin + 1)
        points = cv2.findNonZero(raw_edges[y0:y1, x0:x1])
        relative = (
            numpy.empty((0, 2), dtype=numpy.float64) if points is None
            else points.reshape(-1, 2).astype(numpy.float64)
        )
        relative += numpy.array((x0 - corner.u_px, y0 - corner.v_px))
        corner_name, arm_names = names_and_arms[index]
        bins_by_corner[corner_name] = {}
        for arm_name, neighbor in zip(
            arm_names, (corners[index - 1], corners[(index + 1) % 4])
        ):
            direction = numpy.array(
                (neighbor.u_px - corner.u_px, neighbor.v_px - corner.v_px)
            )
            direction /= numpy.linalg.norm(direction)
            along = relative @ direction
            normal = relative @ numpy.array((-direction[1], direction[0]))
            selected = (
                (along >= 1.5) & (along <= radius) & (numpy.abs(normal) <= 2.5)
            )
            bins = numpy.unique(
                numpy.floor(along[selected] + 0.5).astype(numpy.int32)
            )
            bins_by_corner[corner_name][arm_name] = len(bins)
    return MetricCornerArmSupport(radius, 2, bins_by_corner)


def observed_metric_corner_arms(cv2, raw_edges, corners) -> bool:
    """Compatibility predicate using the same raw support as diagnostics."""

    return metric_corner_arm_support(cv2, raw_edges, corners).accepted
