"""Bounded, QR-neutral head localization before any metric pose exists.

A proposal only moves a crop. It has no angle, identity, or motion authority.
The caller must associate its bearing with the selected candidate and perform
its normal current-image metric fit, freshness, and consensus checks afterward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.geometry import _distance, order_corners
from scripts.aufgabe04.perception.stand_axis.head_candidates import _short_centered_neck_support
from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame

_MAX_LINES_PER_DIRECTION = 32
_MAX_RAW_VERIFICATIONS = 12


@dataclass(frozen=True)
class HeadProposal:
    """Current 2D corners and crop bounds, relative to the input search image."""

    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]
    bounds_xyxy: tuple[int, int, int, int]
    head_bounds_xyxy: tuple[float, float, float, float]
    center_u_px: float
    center_v_px: float
    observed_height_px: float
    expected_height_ratio: float
    center_offset_head_heights: float
    raw_edge_support: float
    geometry_quality: float


@dataclass(frozen=True)
class HeadProposalResult:
    proposal: HeadProposal | None
    reason: str
    considered_proposals: int = 0
    raw_verifications: int = 0
    locator: str | None = None


def _extent(corners):
    left, right, lower_right, lower_left = order_corners(corners)
    width = (_distance(left, right) + _distance(lower_left, lower_right)) / 2.0
    height = (_distance(left, lower_left) + _distance(right, lower_right)) / 2.0
    center = (
        sum(point.u_px for point in corners) / 4.0,
        sum(point.v_px for point in corners) / 4.0,
    )
    return width, height, center


def _line_groups(lines, *, expected_height, expected_center):
    groups = ([], [])
    if lines is None:
        return groups
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        first, last = (float(x1), float(y1)), (float(x2), float(y2))
        dx, dy = last[0] - first[0], last[1] - first[1]
        length = math.hypot(dx, dy)
        if not 0.55 * expected_height <= length <= 1.4 * expected_height:
            continue
        if (
            abs((first[0] + last[0]) / 2.0 - expected_center[0]) > 2.0 * expected_height
            or abs((first[1] + last[1]) / 2.0 - expected_center[1]) > 1.6 * expected_height
        ):
            continue
        if abs(dy) <= 0.30 * abs(dx):
            group, coordinate = groups[0], 0
        elif abs(dx) <= 0.30 * abs(dy):
            group, coordinate = groups[1], 1
        else:
            continue
        if first[coordinate] > last[coordinate]:
            first, last = last, first
        group.append((first, last))
    # Size, not proximity to the nominal center, orders the cheap locator.
    # Thus an off-center physical head is not displaced by tiny QR rectangles.
    for group in groups:
        group.sort(key=lambda pair: abs(math.log(math.dist(*pair) / expected_height)))
        del group[_MAX_LINES_PER_DIRECTION:]
    return groups


def _rough_proposals(groups, *, expected_height, expected_center, max_center_offset):
    ranked = []
    for direction, group in enumerate(groups):
        for index, first in enumerate(group):
            for second in group[index + 1:]:
                cross = 1 - direction
                first_mean = (first[0][cross] + first[1][cross]) / 2.0
                second_mean = (second[0][cross] + second[1][cross]) / 2.0
                upper, lower = (first, second) if first_mean < second_mean else (second, first)
                if max(abs(upper[i][direction] - lower[i][direction]) for i in (0, 1)) > 0.25 * expected_height:
                    continue
                points = (upper[0], upper[1], lower[1], lower[0])
                corners = order_corners(tuple(ImagePoint(*point) for point in points))
                width, height, center = _extent(corners)
                if (
                    not 0.70 <= height / expected_height <= 1.30
                    or not 0.35 <= width / max(height, 1.0) <= 1.35
                    or math.dist(center, expected_center) > max_center_offset * expected_height
                ):
                    continue
                score = abs(math.log(height / expected_height)) + 0.2 * math.dist(center, expected_center) / expected_height
                if any(max(_distance(a, b) for a, b in zip(corners, existing[1])) < 2.0 for existing in ranked):
                    continue
                ranked.append((score, corners))
    return sorted(ranked, key=lambda item: item[0])


def _same_head(first, second):
    """Near-identical/nested physical-border fits are one head, not two stands."""

    _width, height, center = _extent(first.corners)
    _other_width, other_height, other_center = _extent(second.corners)
    minimum = min(height, other_height)
    return (
        math.dist(center, other_center) <= 0.18 * minimum
        and max(height, other_height) <= 1.25 * minimum
        and all(
            _distance(a, b) <= 0.22 * minimum
            for a, b in zip(first.corners, second.corners)
        )
    )


def _proposal(measurement, shape, *, expected_height, expected_center):
    corners = order_corners(measurement.corners)
    _width, height, center = _extent(corners)
    x0 = min(point.u_px for point in corners)
    y0 = min(point.v_px for point in corners)
    x1 = max(point.u_px for point in corners)
    y1 = max(point.v_px for point in corners)
    margin = max(4.0, 0.12 * height)
    bounds = (
        max(0, int(math.floor(x0 - margin))),
        max(0, int(math.floor(y0 - margin))),
        min(shape[1], int(math.ceil(x1 + margin)) + 1),
        min(shape[0], int(math.ceil(y1 + 0.45 * height)) + 1),
    )
    support = float(measurement.support.mean)
    return HeadProposal(
        corners=corners,
        bounds_xyxy=bounds,
        head_bounds_xyxy=(x0, y0, x1, y1),
        center_u_px=center[0], center_v_px=center[1],
        observed_height_px=height,
        expected_height_ratio=height / expected_height,
        center_offset_head_heights=math.dist(center, expected_center) / expected_height,
        raw_edge_support=support,
        geometry_quality=support,
    )


def acquire_head_proposal(
    cv2,
    frame_bgr,
    *,
    expected_head_center_u_px: float,
    expected_head_center_v_px: float,
    expected_head_height_px: float,
    raw_edges=None,
    edge_preprocess: str = "channel_union",
    canny_low: int = 20,
    canny_high: int = 60,
    max_center_offset_fraction: float = 1.5,
) -> HeadProposalResult:
    """Find complete head borders within an already bounded candidate ROI.

    Grayscale line segments locate the outer rectangle without interpreting QR
    texture. If they produce no valid head, one bounded raw-line fallback can
    recover low-contrast backside rails. Neither line interpolation nor image
    morphology supplies evidence: all four rails/corner arms and the paired
    short neck are checked on untouched current-image Canny pixels.
    """

    expected_height = float(expected_head_height_px)
    center = (float(expected_head_center_u_px), float(expected_head_center_v_px))
    if (
        not all(math.isfinite(value) for value in (*center, expected_height, max_center_offset_fraction))
        or expected_height < 12.0
        or not 0.0 < max_center_offset_fraction <= 1.5
        or frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
        or min(frame_bgr.shape[:2]) < 12
    ):
        return HeadProposalResult(None, "head_proposal_input_invalid")
    if raw_edges is None:
        raw_edges = _canny_edges_from_frame(
            cv2, frame_bgr, edge_preprocess=edge_preprocess, blur_kernel=5,
            canny_low=canny_low, canny_high=canny_high,
        )
    if raw_edges.ndim != 2 or raw_edges.shape != frame_bgr.shape[:2]:
        raise ValueError("raw_edges must match the candidate search image")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    considered = verified = 0
    for locator in ("grayscale_lines", "raw_lines"):
        if locator == "raw_lines":
            if verified >= _MAX_RAW_VERIFICATIONS:
                break
            import numpy
            lines = cv2.HoughLinesP(
                raw_edges, 1.0, numpy.pi / 180.0,
                threshold=max(10, round(0.18 * expected_height)),
                minLineLength=max(8, round(0.55 * expected_height)),
                maxLineGap=max(2, round(0.08 * expected_height)),
            )
        rough = _rough_proposals(
            _line_groups(lines, expected_height=expected_height, expected_center=center),
            expected_height=expected_height, expected_center=center,
            max_center_offset=max_center_offset_fraction,
        )
        considered += len(rough)
        accepted = []
        # Share one budget across both locators; a difficult image cannot
        # double its number of expensive border fits through the fallback.
        for _score, corners in rough[:max(0, _MAX_RAW_VERIFICATIONS - verified)]:
            verified += 1
            measurement = refine_projected_head_border(
                cv2, raw_edges, corners, corridor_half_width_px=8.0,
            )
            if not measurement.accepted:
                continue
            proposal = _proposal(measurement, raw_edges.shape, expected_height=expected_height, expected_center=center)
            if (
                not 0.70 <= proposal.expected_height_ratio <= 1.30
                or proposal.center_offset_head_heights > max_center_offset_fraction
                or not _short_centered_neck_support(raw_edges, proposal.corners)
            ):
                continue
            if accepted and not all(_same_head(proposal, other) for other in accepted):
                return HeadProposalResult(None, "head_proposal_ambiguous", considered, verified, locator)
            accepted.append(proposal)
        if accepted:
            # Raw-supported nested borders describe one head. Prefer the
            # physically outer complete frame rather than its paper/QR inset.
            selected = max(accepted, key=lambda item: _extent(item.corners)[0] * item.observed_height_px)
            return HeadProposalResult(selected, "current_head_proposal", considered, verified, locator)
    return HeadProposalResult(None, "head_proposal_unavailable", considered, verified)
