"""Bounded, QR-neutral head localization before any metric pose exists.

A proposal only moves a crop. It has no angle, identity, or motion authority.
The caller must associate its bearing with the selected candidate and perform
its normal current-image metric fit, freshness, and consensus checks afterward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.geometry import _distance, order_corners
from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame
from scripts.aufgabe04.perception.stand_axis.joint_head_borders import rank_joint_head_borders
from scripts.aufgabe04.perception.stand_axis.head_outer_border import _encloses
from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import (
    bounded_head_acquisition, check_head_acquisition_deadline,
)

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
    joint_border_diagnostics: dict | None = None


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


def _rough_proposals(groups, *, expected_height, expected_center, max_center_offset,
                     max_vertical_center_offset=None, deadline_monotonic_sec=None):
    ranked = []

    def add(points):
        corners = order_corners(tuple(ImagePoint(*point) for point in points))
        width, height, center = _extent(corners)
        if (not 0.70 <= height / expected_height <= 1.30
                or not 0.35 <= width / max(height, 1.0) <= 1.35
                or math.dist(center, expected_center) > max_center_offset * expected_height):
            return
        if (max_vertical_center_offset is not None
                and abs(center[1] - expected_center[1]) > max_vertical_center_offset * expected_height):
            return
        score = abs(math.log(height / expected_height)) + 0.2 * math.dist(center, expected_center) / expected_height
        if any(max(_distance(a, b) for a, b in zip(corners, existing[1])) < 2.0 for existing in ranked):
            return
        ranked.append((score, corners))

    for direction, group in enumerate(groups):
        for index, first in enumerate(group):
            check_head_acquisition_deadline(deadline_monotonic_sec, "endpoint_hypotheses")
            for second in group[index + 1:]:
                cross = 1 - direction
                first_mean = (first[0][cross] + first[1][cross]) / 2.0
                second_mean = (second[0][cross] + second[1][cross]) / 2.0
                upper, lower = (first, second) if first_mean < second_mean else (second, first)
                if max(abs(upper[i][direction] - lower[i][direction]) for i in (0, 1)) > 0.25 * expected_height:
                    continue
                points = (upper[0], upper[1], lower[1], lower[0])
                add(points)
                # A low-contrast rail can have a truncated locator segment
                # although its original Canny pixels continue to the corner.
                # One union-extent seed corrects that endpoint bias. These
                # extrapolated positions are never measurement evidence.
                start = min(upper[0][direction], lower[0][direction])
                end = max(upper[1][direction], lower[1][direction])
                def extended(segment, value):
                    point = list(segment[0])
                    fraction = (value-segment[0][direction]) / (segment[1][direction]-segment[0][direction])
                    point[cross] += fraction*(segment[1][cross]-segment[0][cross])
                    point[direction] = value
                    return tuple(point)
                add((extended(upper, start), extended(upper, end),
                     extended(lower, end), extended(lower, start)))
    return sorted(ranked, key=lambda item: item[0])


def _same_head(first, second):
    """Near-identical/nested physical-border fits are one head, not two stands."""

    _width, height, center = _extent(first.corners)
    _other_width, other_height, other_center = _extent(second.corners)
    minimum = min(height, other_height)
    nearby = (
        math.dist(center, other_center) <= 0.18 * minimum
        and max(height, other_height) <= 1.25 * minimum
        and all(
            _distance(a, b) <= 0.22 * minimum
            for a, b in zip(first.corners, second.corners)
        )
    )
    if nearby:
        return True
    # A complete paper/printed inset is contained by the outer frame, even
    # when its centre or dimensions differ from the near-duplicate rail case.
    # Measured paper/symbol insets can be shorter than the physical frame;
    # an arbitrarily smaller nested stand or background box is not that cue.
    return (minimum / max(height, other_height) >= .70
            and math.dist(center, other_center) <= .30 * minimum
            and (_encloses(tuple(first.corners), tuple(second.corners), tolerance=2.)
                 or _encloses(tuple(second.corners), tuple(first.corners), tolerance=2.)))


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


@bounded_head_acquisition
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
    max_vertical_center_offset_fraction: float | None = None,
    proposal_filter=None,
    deadline_monotonic_sec: float | None = None,
) -> HeadProposalResult:
    """Find complete head borders within an already bounded candidate ROI.

    Independent grayscale/raw line fragments locate all four sides jointly.
    Current gradient and raw-support scores rank a bounded set of complete
    hypotheses, including complementary long-segment endpoint hints. Neither
    interpolation nor morphology supplies evidence: all four rails and corner
    arms are checked on current Canny pixels. Neck visibility is not required.

    The optional vertical band bounds only the locator's search area. A fresh
    candidate association callback may exclude independently verified heads
    before comparing alternatives; two qualifying heads remain ambiguous.
    Neither projection nor association supplies a fitted corner or angle.
    """

    expected_height = float(expected_head_height_px)
    center = (float(expected_head_center_u_px), float(expected_head_center_v_px))
    if (
        not all(math.isfinite(value) for value in (*center, expected_height, max_center_offset_fraction))
        or expected_height < 12.0
        or not 0.0 < max_center_offset_fraction <= 1.5
        or (max_vertical_center_offset_fraction is not None
            and (not math.isfinite(max_vertical_center_offset_fraction)
                 or not 0.0 < max_vertical_center_offset_fraction <= max_center_offset_fraction))
        or frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
        or min(frame_bgr.shape[:2]) < 12
    ):
        return HeadProposalResult(None, "head_proposal_input_invalid")
    check_head_acquisition_deadline(deadline_monotonic_sec, "preprocessing")
    if raw_edges is None:
        raw_edges = _canny_edges_from_frame(
            cv2, frame_bgr, edge_preprocess=edge_preprocess, blur_kernel=5,
            canny_low=canny_low, canny_high=canny_high,
        )
    if raw_edges.ndim != 2 or raw_edges.shape != frame_bgr.shape[:2]:
        raise ValueError("raw_edges must match the candidate search image")
    check_head_acquisition_deadline(deadline_monotonic_sec, "line_detection")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    locator_lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    endpoint_hints = _rough_proposals(
        _line_groups(locator_lines, expected_height=expected_height, expected_center=center),
        expected_height=expected_height, expected_center=center,
        max_center_offset=max_center_offset_fraction,
        max_vertical_center_offset=max_vertical_center_offset_fraction,
        deadline_monotonic_sec=deadline_monotonic_sec,
    )[:_MAX_RAW_VERIFICATIONS]
    hypotheses, diagnostics = rank_joint_head_borders(
        cv2, frame_bgr, raw_edges, expected_height=expected_height,
        expected_center=center, max_center_offset=max_center_offset_fraction,
        locator_lines=locator_lines, hint_corners=tuple(corners for _score, corners in endpoint_hints),
        max_vertical_center_offset=max_vertical_center_offset_fraction,
        deadline_monotonic_sec=deadline_monotonic_sec,
    )
    # Round-robin spatial groups keep another complete stand visible to the
    # ambiguity gate even when one head has many nested border alternatives.
    groups = []
    for hypothesis in hypotheses:
        _width, height, proposed_center = _extent(hypothesis.corners)
        for group in groups:
            _other_width, other_height, other_center = _extent(group[0].corners)
            if math.dist(proposed_center, other_center) <= .18 * min(height, other_height):
                group.append(hypothesis)
                break
        else:
            groups.append([hypothesis])
    ordered = [group[index] for index in range(max(map(len, groups), default=0))
               for group in groups if index < len(group)]
    hints = [item for item in hypotheses if item.locator == "paired_locator_endpoints"][:4]
    ordered = ordered[:_MAX_RAW_VERIFICATIONS - len(hints)] + hints
    accepted, records = [], []
    association_rejections = []
    for hypothesis in ordered[:_MAX_RAW_VERIFICATIONS]:
        check_head_acquisition_deadline(
            deadline_monotonic_sec, "strict_verification",
            considered_proposals=diagnostics["considered_closed_hypotheses"], raw_verifications=len(records))
        measurement = refine_projected_head_border(
            cv2, raw_edges, hypothesis.corners, corridor_half_width_px=8.0,
        )
        record = {"score": hypothesis.score, "locator": hypothesis.locator,
                  "minimum_raw_support": hypothesis.minimum_raw_support,
                  "mean_raw_support": hypothesis.mean_raw_support, "mean_gradient": hypothesis.mean_gradient,
                  "reason": measurement.reason, "accepted": measurement.accepted,
                  "corners": None if measurement.corners is None else
                      [(p.u_px, p.v_px) for p in measurement.corners],
                  "candidate_corners": None if measurement.candidate_corners is None else
                      [(p.u_px, p.v_px) for p in measurement.candidate_corners],
                  "verified_raw_support": None if measurement.support is None else
                      measurement.support.mean}
        records.append(record)
        if not measurement.accepted:
            continue
        proposal = _proposal(measurement, raw_edges.shape, expected_height=expected_height, expected_center=center)
        if (not .70 <= proposal.expected_height_ratio <= 1.30
                or proposal.center_offset_head_heights > max_center_offset_fraction
                or (max_vertical_center_offset_fraction is not None
                    and abs(proposal.center_v_px - center[1]) > max_vertical_center_offset_fraction * expected_height)):
            continue
        check_head_acquisition_deadline(deadline_monotonic_sec, "candidate_association",
            considered_proposals=diagnostics["considered_closed_hypotheses"], raw_verifications=len(records))
        if proposal_filter is not None and not proposal_filter(proposal):
            association_rejections.append([(p.u_px, p.v_px) for p in proposal.corners])
            continue
        accepted.append(proposal)
    diagnostics = {**diagnostics, "spatial_groups": len(groups), "strict_verifications": records,
                   "max_raw_verifications": _MAX_RAW_VERIFICATIONS,
                   "selection": "unavailable",
                   "max_vertical_center_offset_fraction": max_vertical_center_offset_fraction,
                   "association_filtered_heads": association_rejections,
                   "proposal_filter_applied": proposal_filter is not None}
    considered, verified = diagnostics["considered_closed_hypotheses"], len(records)
    check_head_acquisition_deadline(deadline_monotonic_sec, "head_selection",
        considered_proposals=considered, raw_verifications=verified)
    if accepted:
        selected = max(accepted, key=lambda item: _extent(item.corners)[0] * item.observed_height_px)
        if any(not _same_head(selected, other) for other in accepted):
            diagnostics["selection"] = "distinct_current_heads_ambiguous"
            return HeadProposalResult(None, "head_proposal_ambiguous", considered, verified,
                                      "joint_current_borders", diagnostics)
        # Preserve strong larger alternatives whose corner arms failed. They
        # remain diagnostics, never substitute corners or a preferred pose.
        # Temporal head consistency handles changes between admitted frames.
        selected_area = _extent(selected.corners)[0] * selected.observed_height_px
        unresolved_outer = []
        for record in records:
            if (record["reason"] != "model_corner_evidence_insufficient"
                    or record["candidate_corners"] is None
                    or record["verified_raw_support"] is None
                    or record["verified_raw_support"] < .90
                    or record["minimum_raw_support"] < .80):
                continue
            corners = tuple(ImagePoint(*point) for point in record["candidate_corners"])
            width, height, _center = _extent(corners)
            if (width * height >= 1.03 * selected_area
                    and _encloses(corners, tuple(selected.corners), tolerance=1.5)):
                unresolved_outer.append(record["candidate_corners"])
        if unresolved_outer:
            diagnostics["competing_outer_corners"] = unresolved_outer
        # Compare every admitted alternative before choosing the complete
        # enclosing frame. Projected proximity and solved yaw never rank it.
        diagnostics["selection"] = "maximal_verified_current_head"
        diagnostics["selected_corners"] = [(p.u_px, p.v_px) for p in selected.corners]
        check_head_acquisition_deadline(deadline_monotonic_sec, "complete_head_selection",
            considered_proposals=considered, raw_verifications=verified)
        return HeadProposalResult(selected, "current_head_proposal", considered, verified,
                                  "joint_current_borders", diagnostics)
    return HeadProposalResult(None, "head_proposal_candidate_association_rejected" if association_rejections
                              else "head_proposal_unavailable", considered, verified,
                              "joint_current_borders", diagnostics)
