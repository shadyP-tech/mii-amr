"""Select the enclosing current head rails without neck or pose evidence.

Measured panel/symbol dimensions locate at most two outward searches. Only
complete current raw borders and corner arms select a larger rectangle; no
interpolation, QR pose, neck cue or previous measurement supplies corners.
A lone quadrilateral cannot identify physical scale by itself. Its angle is
conditional on the candidate/model association performed by the observer.
"""

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.geometry import _distance, _polygon_area, order_corners
from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


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


def _encloses(outer, inner, tolerance=1.5):
    """Convex containment using signed edge distances, without fitted poses."""
    area = sum(a.u_px*b.v_px-b.u_px*a.v_px for a, b in zip(outer, outer[1:]+outer[:1]))
    sign = 1. if area >= 0 else -1.
    return all(sign*((b.u_px-a.u_px)*(p.v_px-a.v_px)-(b.v_px-a.v_px)*(p.u_px-a.u_px))
               >= -tolerance*max(_distance(a, b), 1.)
               for a, b in zip(outer, outer[1:]+outer[:1]) for p in inner)


def select_current_outer_head_border(cv2, raw_edges, *, model_profile, refinement,
                                     corridor_half_width_px):
    """Prefer an enclosing complete border; missing outward pixels stay missing."""
    original = refinement.corners
    if not refinement.accepted or original is None:
        return refinement, HeadOuterBorderEvidence(
            False, "current_head_border_unavailable", original, None, model_profile.sha256)
    original = tuple(order_corners(original))
    center = (sum(p.u_px for p in original)/4, sum(p.v_px for p in original)/4)
    height = (_distance(original[0], original[3])+_distance(original[1], original[2]))/2
    growths = []
    for width, panel_height in ((model_profile.qr_panel_width_m, model_profile.qr_panel_height_m),
                                (model_profile.qr_symbol_width_m, model_profile.qr_symbol_height_m)):
        if width is None or panel_height is None or min(width, panel_height) <= 0:
            continue
        growth = max(model_profile.head_width_m/width, model_profile.head_height_m/panel_height)
        if math.isfinite(growth) and 1.03 <= growth <= 1.30 and growth not in growths:
            growths.append(growth)
    selected, area = refinement, _polygon_area(original)
    tried = []
    for growth in growths[:2]:
        proposed = tuple(ImagePoint(center[0]+(p.u_px-center[0])*growth,
                                    center[1]+(p.v_px-center[1])*growth) for p in original)
        tried.append(growth)
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
    recovered = selected is not refinement
    return selected, HeadOuterBorderEvidence(
        True, "current_raw_outer_head_recovered" if recovered else "maximal_current_head_border",
        original, tuple(selected.corners), model_profile.sha256, tuple(tried), recovered)


@dataclass(frozen=True)
class HeadMarkerBoundaryEvidence:
    accepted: bool
    reason: str
    observed_symbol_spans: tuple[float, float] | None = None
    expected_head_symbol_spans: tuple[float, float] | None = None
    expected_panel_symbol_spans: tuple[float, float] | None = None
    discrimination_margin: float | None = None
    supplies_angle: bool = False


def check_current_head_marker_boundary(cv2, *, head_corners, qr_corners,
                                      marker_verified, model_profile):
    """Veto an identified paper inset, without fitting or replacing an angle.

    A verified symbol supplies a current independent size reference. Compare
    its extent in the observed head quadrilateral to the measured head/paper
    alternatives; a two-pixel-per-side uncertainty margin prevents borderline
    geometry from being declared a paper border. Missing QR is not scale proof.
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
