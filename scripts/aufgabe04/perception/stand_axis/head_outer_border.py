"""Select the enclosing current head rails without neck or pose evidence.

Two bounded search scales locate current enclosing borders. Only
complete current raw borders and corner arms select a larger rectangle; no
interpolation, QR pose, neck cue or previous measurement supplies corners.
A lone quadrilateral cannot identify physical scale by itself. Its angle is
conditional on the candidate/model association performed by the observer.
"""

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _distance, _polygon_area, _well_formed_quadrilateral, order_corners,
)
from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


# Search policy, independent of the presence, dimensions or corners of a QR.
# These factors propose raw-pixel corridors; they never synthesize corners.
OUTER_HEAD_SEARCH_GROWTH_FACTORS = (1.10, 1.25)


@dataclass(frozen=True)
class HeadOuterBorderEvidence:
    accepted: bool
    reason: str
    original_corners: tuple[ImagePoint, ...] | None
    recovered_corners: tuple[ImagePoint, ...] | None
    profile_sha256: str
    attempted_growth_factors: tuple[float, ...] = ()
    recovered: bool = False
    selection_source: str = "current_enclosing_raw_borders"
    scale_identifiability: str = "conditional_on_candidate_and_measured_profile"
    neutral_proposal_corners: tuple[ImagePoint, ...] | None = None
    independently_resolved: bool = False
    proposal_inward_shift_px: float = 0.0
    attempted_raw_refinements: int = 0
    current_raw_alternatives: tuple[tuple[ImagePoint, ...], ...] = ()
    head_size_m: tuple[float, float] | None = None
    largest_inset_size_m: tuple[float, float] | None = None
    model_tolerance_m: float | None = None


def _encloses(outer, inner, tolerance=1.5):
    """Convex containment using signed edge distances, without fitted poses."""
    area = sum(a.u_px*b.v_px-b.u_px*a.v_px for a, b in zip(outer, outer[1:]+outer[:1]))
    sign = 1. if area >= 0 else -1.
    return all(sign*((b.u_px-a.u_px)*(p.v_px-a.v_px)-(b.v_px-a.v_px)*(p.u_px-a.u_px))
               >= -tolerance*max(_distance(a, b), 1.)
               for a, b in zip(outer, outer[1:]+outer[:1]) for p in inner)


def _inward_shift(selected, reference):
    """Largest perpendicular inward displacement of an original current rail."""
    selected, reference = tuple(order_corners(selected)), tuple(order_corners(reference))
    area = sum(a.u_px*b.v_px-b.u_px*a.v_px for a, b in zip(selected, selected[1:]+selected[:1]))
    sign = 1. if area >= 0. else -1.
    return max(0., max(-sign*((b.u_px-a.u_px)*(point.v_px-a.v_px)
                      -(b.v_px-a.v_px)*(point.u_px-a.u_px))/max(_distance(a, b), 1.)
                      for a, b, p, q in zip(selected, selected[1:]+selected[:1],
                                             reference, reference[1:]+reference[:1])
                      for point in (p, q)))


def _span(corners):
    a, b, c, d = order_corners(corners)
    return ((_distance(a, b)+_distance(c, d))/2.,
            (_distance(a, d)+_distance(b, c))/2.)


def _independent_enclosure(selected, original, head_size, inset_size, tolerance):
    """Reject the two edges of a paper rail as apparent head-frame recovery.

    Both dimensions must grow by at least the measured head/paper separation,
    reduced only by the profile's recorded metrology tolerance. A tiny larger
    contour is not independent evidence for the physical head boundary.
    """
    try:
        if (len(head_size) != 2 or len(inset_size) != 2
                or not all(math.isfinite(value) and value > 0. for value in (*head_size, *inset_size))
                or not math.isfinite(tolerance) or tolerance < 0.
                or any(inset+tolerance >= head for head, inset in zip(head_size, inset_size))):
            return False
        return all(outer/inner >= head/(inset+tolerance)
                   for outer, inner, head, inset in zip(
                       _span(selected), _span(original), head_size, inset_size))
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def validated_current_head_boundary(evidence, *, corners, profile_sha256,
                                    require_independent=False):
    """Bind an outer-boundary decision to its exact current fit and profile.

    This is a producer consistency check, not a substitute for candidate
    association, raw pixel verification, freshness or metric pose quality.
    """
    if not isinstance(evidence, HeadOuterBorderEvidence):
        return False
    try:
        selected = tuple(corners)
        original = tuple(evidence.original_corners)
        proposal = tuple(evidence.neutral_proposal_corners)
        for quad in (selected, original, proposal, *evidence.current_raw_alternatives):
            if (len(quad) != 4 or not all(math.isfinite(value) for point in quad
                    for value in (point.u_px, point.v_px))
                    or not _well_formed_quadrilateral(quad)):
                return False
        if not (evidence.accepted is True and evidence.profile_sha256 == profile_sha256
                and evidence.selection_source == "current_enclosing_raw_borders"
                and evidence.reason in {"maximal_current_head_border", "current_raw_outer_head_recovered"}
                and evidence.recovered_corners == selected
                and original in evidence.current_raw_alternatives
                and selected in evidence.current_raw_alternatives):
            return False
        independent = bool(evidence.recovered is True and evidence.independently_resolved is True
            and selected != original and _encloses(selected, original)
            and 1.03*_polygon_area(original) <= _polygon_area(selected)
            <= 1.70*_polygon_area(original)
            and _independent_enclosure(selected, original, evidence.head_size_m,
                                       evidence.largest_inset_size_m, evidence.model_tolerance_m))
        return not require_independent or independent
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return False


def current_head_boundary_eligible(estimate, debug):
    """Bind physical head pixels independently of every QR/marker diagnostic.

    Marker presence and marker-to-head span belong to side classification and
    identity only. They cannot reject, repair or promote this boundary proof.
    """
    evidence = getattr(debug, "head_outer_recovery", None)
    if not validated_current_head_boundary(evidence, corners=getattr(estimate, "corners", None),
            profile_sha256=getattr(estimate, "model_profile_sha256", None)):
        return False
    predicted = getattr(debug, "predicted_corners", None)
    quality = getattr(debug, "head_model_quality", None)
    return bool(evidence.profile_sha256 == getattr(debug, "model_profile_sha256", None)
                and (quality is None or evidence.head_size_m == quality.head_size_m)
                and (predicted is None or tuple(predicted) == evidence.neutral_proposal_corners))


def select_current_outer_head_border(cv2, raw_edges, *, model_profile, refinement,
                                     corridor_half_width_px, neutral_proposal_corners=None):
    """Prefer an enclosing complete border; missing outward pixels stay missing."""
    original = refinement.corners
    if not refinement.accepted or original is None:
        return refinement, HeadOuterBorderEvidence(
            False, "current_head_border_unavailable", original, None, model_profile.sha256)
    original = tuple(order_corners(original))
    proposal = tuple(order_corners(neutral_proposal_corners or original))
    center = (sum(p.u_px for p in proposal)/4, sum(p.v_px for p in proposal)/4)
    height = (_distance(proposal[0], proposal[3])+_distance(proposal[1], proposal[2]))/2
    # The two-pixel rail localization allowance is already used by the joint
    # current-gradient locator. Physical metrology adds its existing tolerance.
    inward_allowance = 2. + height*model_profile.tolerance_m/model_profile.head_height_m
    inward = _inward_shift(original, proposal)
    head_size = (model_profile.head_width_m, model_profile.head_height_m)
    inset_size = (model_profile.qr_panel_width_m, model_profile.qr_panel_height_m)
    selected, area = refinement, _polygon_area(original)
    tried, alternatives = [], [original]
    attempted = 0
    # Do not lose the original independently acquired outer frame when a wide
    # raw corridor selects an inner paper rail. The reference only locates a
    # narrower current-pixel fit; it never supplies accepted corners itself.
    if inward > inward_allowance:
        attempted += 1
        anchored = refine_projected_head_border(
            cv2, raw_edges, proposal,
            corridor_half_width_px=min(2., corridor_half_width_px))
        if anchored.accepted and anchored.corners is not None:
            anchored_corners = tuple(order_corners(anchored.corners))
            if (_encloses(anchored_corners, original)
                    and 1.03*area <= _polygon_area(anchored_corners) <= 1.70*area):
                selected, area = anchored, _polygon_area(anchored_corners)
                alternatives.append(anchored_corners)
    for growth in OUTER_HEAD_SEARCH_GROWTH_FACTORS:
        proposed = tuple(ImagePoint(center[0]+(p.u_px-center[0])*growth,
                                    center[1]+(p.v_px-center[1])*growth) for p in proposal)
        tried.append(growth)
        attempted += 1
        current = refine_projected_head_border(
            cv2, raw_edges, proposed, corridor_half_width_px=corridor_half_width_px)
        if not current.accepted or current.corners is None:
            continue
        corners = tuple(order_corners(current.corners))
        larger = _polygon_area(corners)
        shifted = math.hypot(sum(p.u_px for p in corners)/4-center[0],
                             sum(p.v_px for p in corners)/4-center[1])
        if (1.03*area <= larger <= 1.70*_polygon_area(original)
                and shifted <= max(2., .08*height) and _encloses(corners, original)
                and _encloses(corners, tuple(selected.corners))):
            selected, area = current, larger
            alternatives.append(corners)
    recovered = selected is not refinement
    remaining_inward = _inward_shift(tuple(selected.corners), proposal)
    accepted = remaining_inward <= inward_allowance
    return selected, HeadOuterBorderEvidence(
        accepted, ("current_physical_head_boundary_unresolved" if not accepted else
                   "current_raw_outer_head_recovered" if recovered else "maximal_current_head_border"),
        original, tuple(selected.corners), model_profile.sha256, tuple(tried), recovered,
        neutral_proposal_corners=proposal,
        independently_resolved=bool(accepted and recovered and _independent_enclosure(
            tuple(selected.corners), original, head_size, inset_size, model_profile.tolerance_m)),
        proposal_inward_shift_px=inward,
        attempted_raw_refinements=attempted,
        current_raw_alternatives=tuple(alternatives), head_size_m=head_size,
        largest_inset_size_m=inset_size, model_tolerance_m=model_profile.tolerance_m)


@dataclass(frozen=True)
class HeadMarkerBoundaryEvidence:
    accepted: bool
    reason: str
    observed_symbol_spans: tuple[float, float] | None = None
    expected_head_symbol_spans: tuple[float, float] | None = None
    expected_panel_symbol_spans: tuple[float, float] | None = None
    discrimination_margin: float | None = None
    supplies_angle: bool = False
    diagnostic_only: bool = True
    requests_reconsideration: bool = False


def check_current_head_marker_boundary(cv2, *, head_corners, qr_corners,
                                      marker_verified, model_profile):
    """Describe a marker span for offline diagnostics, never angle admission.

    This compatibility diagnostic compares measured head/paper proportions;
    its accepted flag describes only that ratio. Current raw borders, their
    measured 3D fit and candidate association establish a head independently.
    Neither a contradictory ratio nor missing QR changes that decision.
    """
    if not marker_verified or qr_corners is None or head_corners is None:
        return HeadMarkerBoundaryEvidence(True, "no_current_verified_symbol_scale_reference")
    if model_profile.qr_panel_width_m is None or model_profile.qr_panel_height_m is None:
        return HeadMarkerBoundaryEvidence(True, "measured_panel_dimensions_unavailable")
    import numpy as np
    head = tuple(order_corners(head_corners))
    if not _encloses(head, tuple(qr_corners), tolerance=1.5):
        return HeadMarkerBoundaryEvidence(True, "current_symbol_not_a_head_scale_reference")
    transform = cv2.getPerspectiveTransform(
        np.asarray([(p.u_px, p.v_px) for p in head], np.float32),
        np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.)), np.float32))
    normalized = cv2.perspectiveTransform(
        np.asarray([(p.u_px, p.v_px) for p in qr_corners], np.float32).reshape(1, 4, 2), transform)[0]
    observed = tuple(float(np.ptp(normalized[:, axis])) for axis in (0, 1))
    head_spans = (model_profile.qr_symbol_width_m/model_profile.head_width_m,
                  model_profile.qr_symbol_height_m/model_profile.head_height_m)
    panel_spans = (model_profile.qr_symbol_width_m/model_profile.qr_panel_width_m,
                   model_profile.qr_symbol_height_m/model_profile.qr_panel_height_m)
    minimum = min(_distance(a,b) for a,b in zip(head,head[1:]+head[:1]))
    margin = max(4./max(minimum,1.), model_profile.tolerance_m/min(model_profile.head_width_m,model_profile.head_height_m))
    inset = all(abs(value-panel)+margin < abs(value-physical)
                for value, panel, physical in zip(observed, panel_spans, head_spans))
    return HeadMarkerBoundaryEvidence(not inset,
        "current_border_matches_verified_qr_panel" if inset else "current_head_symbol_boundary_not_contradicted",
        observed, head_spans, panel_spans, margin)
