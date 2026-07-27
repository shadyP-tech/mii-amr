from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence


Point = tuple[float, float]


@dataclass(frozen=True)
class StandStructureEvidence:
    """Immutable-edge evidence for one head -> stem -> base hypothesis."""

    accepted: bool
    tracking_supported: bool
    reason: str
    head_top_support: float
    head_left_support: float
    head_right_support: float
    stem_left_support: float
    stem_right_support: float
    stem_span_px: float
    base_support: float
    base_span_px: float
    base_center_offset_px: float
    corners: tuple[Point, Point, Point, Point] | None
    evidence_mask: object | None = None

    def status_dict(self) -> dict[str, object]:
        values = asdict(self)
        values.pop("evidence_mask", None)
        return values


def _distance(start: Point, end: Point) -> float:
    return math.hypot(end[0] - start[0], end[1] - start[1])


def _segment_points_and_support(
    numpy,
    edge_points,
    start: Point,
    end: Point,
    *,
    band_px: float,
    start_fraction: float,
    end_fraction: float,
    bin_count: int = 24,
):
    direction = numpy.array(
        (end[0] - start[0], end[1] - start[1]),
        dtype=numpy.float64,
    )
    length = float(numpy.linalg.norm(direction))
    if length <= 1e-6:
        return edge_points[:0], 0.0
    unit = direction / length
    relative = edge_points - numpy.array(start, dtype=numpy.float64)
    along = relative @ unit
    normal = numpy.abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
    fractions = along / length
    selected = edge_points[
        (fractions >= start_fraction)
        & (fractions <= end_fraction)
        & (normal <= band_px)
    ]
    if not len(selected):
        return selected, 0.0
    selected_relative = selected - numpy.array(start, dtype=numpy.float64)
    selected_fractions = (selected_relative @ unit) / length
    bins = numpy.floor(
        (selected_fractions - start_fraction)
        / max(end_fraction - start_fraction, 1e-6)
        * bin_count
    ).astype(numpy.int32)
    bins = numpy.clip(bins, 0, bin_count - 1)
    support = len(numpy.unique(bins)) / float(bin_count)
    return selected, float(support)


def _fit_x_at_y(numpy, points, y: float) -> float | None:
    if len(points) < 4:
        return None
    ys = points[:, 1]
    xs = points[:, 0]
    if float(ys.max() - ys.min()) < 4.0:
        return None
    slope, intercept = numpy.polyfit(ys, xs, 1)
    return float(slope * y + intercept)


def _fit_y_at_x(numpy, points, x: float) -> float | None:
    if len(points) < 4:
        return None
    xs = points[:, 0]
    ys = points[:, 1]
    if float(xs.max() - xs.min()) < 4.0:
        return None
    slope, intercept = numpy.polyfit(xs, ys, 1)
    return float(slope * x + intercept)


def _empty(reason: str) -> StandStructureEvidence:
    return StandStructureEvidence(
        accepted=False,
        tracking_supported=False,
        reason=reason,
        head_top_support=0.0,
        head_left_support=0.0,
        head_right_support=0.0,
        stem_left_support=0.0,
        stem_right_support=0.0,
        stem_span_px=0.0,
        base_support=0.0,
        base_span_px=0.0,
        base_center_offset_px=math.inf,
        corners=None,
    )


def evaluate_stand_structure(
    cv2,
    raw_edges,
    rough_corners: Sequence[Point],
    *,
    stem_center_x: float,
    stem_top_y: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> StandStructureEvidence:
    """Recover a head with a missing bottom only when stem/base own it.

    Topology may provide ``rough_corners`` and the stem anchor. Acceptance is
    computed only from the supplied immutable raw edge mask. The derived lower
    head segment is a visualization/measurement boundary between independently
    observed side endpoints; it is never counted as raw bottom-edge support.
    """

    import numpy

    if raw_edges is None or raw_edges.size == 0:
        return _empty("structure_raw_edges_unavailable")
    locations = cv2.findNonZero(raw_edges)
    if locations is None:
        return _empty("structure_raw_edges_empty")
    if len(rough_corners) != 4:
        return _empty("structure_head_corner_count_invalid")

    top_left, top_right, bottom_right, bottom_left = tuple(
        (float(point[0]), float(point[1])) for point in rough_corners
    )
    width = (_distance(top_left, top_right) + _distance(bottom_left, bottom_right)) / 2.0
    height = (_distance(top_left, bottom_left) + _distance(top_right, bottom_right)) / 2.0
    if width < 8.0 or height < 8.0:
        return _empty("structure_head_too_small")

    edge_points = locations.reshape(-1, 2).astype(numpy.float64)
    band_px = float(max(3.0, min(7.0, 0.07 * min(width, height))))
    top_points, top_support = _segment_points_and_support(
        numpy,
        edge_points,
        top_left,
        top_right,
        band_px=band_px,
        start_fraction=0.04,
        end_fraction=0.96,
    )
    left_points, left_support = _segment_points_and_support(
        numpy,
        edge_points,
        top_left,
        bottom_left,
        band_px=band_px,
        start_fraction=0.02,
        end_fraction=1.08,
    )
    right_points, right_support = _segment_points_and_support(
        numpy,
        edge_points,
        top_right,
        bottom_right,
        band_px=band_px,
        start_fraction=0.02,
        end_fraction=1.08,
    )

    if top_support < 0.55:
        reason = "structure_head_top_unsupported"
    elif left_support < 0.55:
        reason = "structure_head_left_unsupported"
    elif right_support < 0.55:
        reason = "structure_head_right_unsupported"
    else:
        reason = "structure_head_three_sides_supported"

    evidence_mask = numpy.zeros(raw_edges.shape[:2], dtype=numpy.uint8)
    for points in (top_points, left_points, right_points):
        if len(points):
            xs = numpy.clip(
                numpy.rint(points[:, 0]).astype(numpy.int32),
                0,
                evidence_mask.shape[1] - 1,
            )
            ys = numpy.clip(
                numpy.rint(points[:, 1]).astype(numpy.int32),
                0,
                evidence_mask.shape[0] - 1,
            )
            evidence_mask[ys, xs] = 255

    if reason != "structure_head_three_sides_supported":
        return StandStructureEvidence(
            accepted=False,
            tracking_supported=False,
            reason=reason,
            head_top_support=top_support,
            head_left_support=left_support,
            head_right_support=right_support,
            stem_left_support=0.0,
            stem_right_support=0.0,
            stem_span_px=0.0,
            base_support=0.0,
            base_span_px=0.0,
            base_center_offset_px=math.inf,
            corners=None,
            evidence_mask=evidence_mask,
        )

    # Use independently observed side-run endpoints. The neck transition is a
    # sanity bound, not a synthetic source for either lower corner.
    left_bottom_y = float(numpy.percentile(left_points[:, 1], 96.0)) if len(left_points) else math.nan
    right_bottom_y = float(numpy.percentile(right_points[:, 1], 96.0)) if len(right_points) else math.nan
    left_top_y = float(numpy.percentile(left_points[:, 1], 4.0)) if len(left_points) else math.nan
    right_top_y = float(numpy.percentile(right_points[:, 1], 4.0)) if len(right_points) else math.nan
    if not all(
        math.isfinite(value)
        for value in (left_bottom_y, right_bottom_y, left_top_y, right_top_y)
    ):
        return StandStructureEvidence(
            **{
                **_empty("structure_head_side_endpoints_unavailable").__dict__,
                "head_top_support": top_support,
                "head_left_support": left_support,
                "head_right_support": right_support,
                "evidence_mask": evidence_mask,
            }
        )

    top_y = min(left_top_y, right_top_y)
    left_top_x = _fit_x_at_y(numpy, left_points, top_y)
    right_top_x = _fit_x_at_y(numpy, right_points, top_y)
    if left_top_x is None or right_top_x is None:
        return _empty("structure_head_side_fit_unavailable")
    top_left_y = _fit_y_at_x(numpy, top_points, left_top_x)
    top_right_y = _fit_y_at_x(numpy, top_points, right_top_x)
    if top_left_y is None or top_right_y is None:
        return _empty("structure_head_top_fit_unavailable")
    left_bottom_x = _fit_x_at_y(numpy, left_points, left_bottom_y)
    right_bottom_x = _fit_x_at_y(numpy, right_points, right_bottom_y)
    if left_bottom_x is None or right_bottom_x is None:
        return _empty("structure_head_lower_side_fit_unavailable")

    recovered = (
        (left_top_x, top_left_y),
        (right_top_x, top_right_y),
        (right_bottom_x, right_bottom_y),
        (left_bottom_x, left_bottom_y),
    )
    recovered_width = (
        _distance(recovered[0], recovered[1])
        + _distance(recovered[3], recovered[2])
    ) / 2.0
    recovered_height = (
        _distance(recovered[0], recovered[3])
        + _distance(recovered[1], recovered[2])
    ) / 2.0
    aspect_ratio = recovered_width / max(recovered_height, 1e-6)
    if not min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
        reason = "structure_head_aspect_invalid"
    recovered_bottom_y = (left_bottom_y + right_bottom_y) / 2.0
    if abs(stem_center_x - (left_bottom_x + right_bottom_x) / 2.0) > 0.20 * recovered_width:
        reason = "structure_stem_not_centered_under_head"
    if not (
        recovered_bottom_y - 0.20 * recovered_height
        <= stem_top_y
        <= recovered_bottom_y + 1.05 * recovered_height
    ):
        reason = "structure_stem_transition_misaligned"

    frame_height, frame_width = raw_edges.shape[:2]
    stem_x_radius = max(5.0, 0.18 * recovered_width)
    stem_y0 = max(recovered_bottom_y, min(stem_top_y, recovered_bottom_y + 0.30 * recovered_height))
    stem_y1 = min(float(frame_height - 1), stem_y0 + 2.40 * recovered_height)
    stem_points = edge_points[
        (edge_points[:, 1] >= stem_y0)
        & (edge_points[:, 1] <= stem_y1)
        & (edge_points[:, 0] >= stem_center_x - stem_x_radius)
        & (edge_points[:, 0] <= stem_center_x + stem_x_radius)
    ]
    left_stem = stem_points[stem_points[:, 0] < stem_center_x]
    right_stem = stem_points[stem_points[:, 0] > stem_center_x]

    def vertical_support(points) -> tuple[float, float]:
        if len(points) < 4 or stem_y1 <= stem_y0:
            return 0.0, 0.0
        bins = numpy.floor(
            (points[:, 1] - stem_y0) / max(stem_y1 - stem_y0, 1e-6) * 24
        ).astype(numpy.int32)
        bins = numpy.clip(bins, 0, 23)
        span = float(points[:, 1].max() - points[:, 1].min())
        return len(numpy.unique(bins)) / 24.0, span

    stem_left_support, left_stem_span = vertical_support(left_stem)
    stem_right_support, right_stem_span = vertical_support(right_stem)
    stem_span = min(left_stem_span, right_stem_span)
    if stem_left_support < 0.20 or stem_right_support < 0.20:
        reason = "structure_paired_stem_unsupported"
    if stem_span < 0.45 * recovered_height:
        reason = "structure_stem_too_short"

    base_y0 = stem_y0 + 0.45 * recovered_height
    base_points = edge_points[
        (edge_points[:, 1] >= base_y0)
        & (edge_points[:, 0] >= stem_center_x - 1.55 * recovered_width)
        & (edge_points[:, 0] <= stem_center_x + 1.55 * recovered_width)
    ]
    base_span = 0.0
    base_center_offset = math.inf
    base_support = 0.0
    if len(base_points) >= 8:
        row_ids = numpy.unique(numpy.rint(base_points[:, 1]).astype(numpy.int32))
        candidates = []
        for row in row_ids:
            row_points = base_points[
                numpy.abs(base_points[:, 1] - float(row)) <= 1.5
            ]
            if len(row_points) < 4:
                continue
            left = float(numpy.percentile(row_points[:, 0], 5.0))
            right = float(numpy.percentile(row_points[:, 0], 95.0))
            span = right - left
            center = (left + right) / 2.0
            if span >= 0.85 * recovered_width:
                candidates.append((span, center))
        if len(candidates) >= 4:
            base_left = float(numpy.percentile(base_points[:, 0], 5.0))
            base_right = float(numpy.percentile(base_points[:, 0], 95.0))
            base_span = base_right - base_left
            base_center = (base_left + base_right) / 2.0
            base_center_offset = abs(base_center - stem_center_x)
            span_score = min(1.0, base_span / max(1.20 * recovered_width, 1.0))
            center_score = max(
                0.0,
                1.0 - base_center_offset / max(0.45 * recovered_width, 1.0),
            )
            width_score = (
                1.0
                if base_span
                <= min(2.80 * recovered_width, 0.82 * frame_width)
                else 0.0
            )
            row_score = min(1.0, len(candidates) / 8.0)
            base_support = span_score * center_score * width_score * row_score
    if base_support < 0.55:
        reason = "structure_base_unsupported"

    for points in (left_stem, right_stem, base_points):
        if len(points):
            xs = numpy.clip(
                numpy.rint(points[:, 0]).astype(numpy.int32),
                0,
                evidence_mask.shape[1] - 1,
            )
            ys = numpy.clip(
                numpy.rint(points[:, 1]).astype(numpy.int32),
                0,
                evidence_mask.shape[0] - 1,
            )
            evidence_mask[ys, xs] = 255

    tracking_supported = bool(
        top_support >= 0.55
        and left_support >= 0.55
        and right_support >= 0.55
        and stem_left_support >= 0.20
        and stem_right_support >= 0.20
        and stem_span >= 0.45 * recovered_height
        and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio
        and reason not in {
            "structure_stem_not_centered_under_head",
            "structure_stem_transition_misaligned",
        }
    )
    accepted = tracking_supported and base_support >= 0.55
    return StandStructureEvidence(
        accepted=accepted,
        tracking_supported=tracking_supported,
        reason="structure_owned_head_supported" if accepted else reason,
        head_top_support=top_support,
        head_left_support=left_support,
        head_right_support=right_support,
        stem_left_support=stem_left_support,
        stem_right_support=stem_right_support,
        stem_span_px=stem_span,
        base_support=base_support,
        base_span_px=base_span,
        base_center_offset_px=base_center_offset,
        corners=recovered if tracking_supported else None,
        evidence_mask=evidence_mask,
    )
