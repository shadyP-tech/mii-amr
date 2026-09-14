"""Current head appearance evidence, independent of planar angle admission.

A complete rectangle without a decoded QR is not sufficient by itself. This
stage requires measured-profile, raw outer-border, projection and explicit
current marker checks. The observer separately verifies complete crop coverage,
unique candidate association, freshness and repeated stopped observations.
No value in this module grants an angle or motion authority.
"""

from dataclasses import dataclass, replace
import math

from scripts.aufgabe04.artifacts.backside_axis_observation import (
    BACKSIDE_AXIS_SAMPLE_SOURCE, MAXIMUM_HEAD_CENTER_ERROR_RATIO,
    MAXIMUM_HEAD_SCALE_RATIO, MINIMUM_BACKSIDE_FACE_CONFIDENCE,
    MINIMUM_HEAD_SCALE_RATIO,
)
from scripts.aufgabe04.perception.stand_axis.geometry import quadrilateral_aspect_ratio
from scripts.aufgabe04.perception.stand_axis.head_model_quality import (
    HeadModelQuality, MEASURED_HEAD_AXIS_SOURCE, MIN_HEAD_EDGE_PX,
)

MIN_NORMALIZED_ASPECT = math.cos(math.radians(70.0))
MAX_NORMALIZED_ASPECT = 1.35


@dataclass(frozen=True)
class HeadBacksideAppearance:
    accepted: bool
    reason: str
    profile_sha256: str | None
    corners: tuple | None
    head_scale_ratio: float | None = None
    head_center_error_ratio: float | None = None
    normalized_aspect: float | None = None
    confidence: float | None = None
    basis: str = "current_raw_head_and_explicit_marker_checks"
    neck_required: bool = False
    supplies_angle: bool = False
    motion_authorized: bool = False


def _gate_score(value, lower, upper):
    midpoint = (lower + upper) / 2.0
    return max(0.0, 1.0 - abs(value - midpoint) / ((upper - lower) / 2.0))


def head_appearance_confidence(quality, scale, center, aspect):
    """Keep the existing raw-border/projection confidence thresholds."""
    return min(1.0, max(0.0,
        .50 * quality.raw_border_support_mean
        + .20 * _gate_score(scale, MINIMUM_HEAD_SCALE_RATIO, MAXIMUM_HEAD_SCALE_RATIO)
        + .20 * max(0.0, 1.0 - center / MAXIMUM_HEAD_CENTER_ERROR_RATIO)
        + .10 * _gate_score(aspect, MIN_NORMALIZED_ASPECT, MAX_NORMALIZED_ASPECT)))


def assess_current_head_backside_appearance(
    estimate, debug, *, model_profile, camera,
    expected_center_u_px, expected_center_v_px, expected_height_px,
):
    """Assess appearance even when a current head has an ambiguous pose."""
    result = HeadBacksideAppearance(
        False, "backside_current_raw_head_required",
        estimate.model_profile_sha256, estimate.corners)
    quality = debug.head_model_quality
    corners = estimate.corners
    if (estimate.source not in {MEASURED_HEAD_AXIS_SOURCE, BACKSIDE_AXIS_SAMPLE_SOURCE}
            or estimate.evidence_state == "predicted_only"
            or debug.model_pose_fit_source != MEASURED_HEAD_AXIS_SOURCE
            or not isinstance(quality, HeadModelQuality)
            or quality.raw_corner_support_accepted is not True
            or quality.outer_border_verified is not True
            or type(quality.raw_border_support_mean) not in (int, float)
            or not .60 <= quality.raw_border_support_mean <= 1.0
            or corners is None or len(corners) != 4
            or any(not math.isfinite(v) for p in corners for v in (p.u_px, p.v_px))):
        return result
    if (not model_profile.committable or model_profile.environment != "physical"
            or model_profile.sha256 != estimate.model_profile_sha256
            or quality.profile_sha256 != model_profile.sha256):
        return replace(result, reason="backside_measured_physical_profile_required")
    if debug.qr_detected is not False or debug.qr_marker_verified is not False:
        return replace(result, reason="backside_current_marker_absence_required")
    boundary = getattr(debug, "head_marker_boundary", None)
    if boundary is not None and boundary.accepted is not True:
        return replace(result, reason="backside_head_boundary_contradicted")
    expected = (expected_center_u_px, expected_center_v_px, expected_height_px)
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in expected)
            or expected_height_px <= 0):
        return replace(result, reason="backside_candidate_projection_required")
    lengths = [math.hypot(b.u_px-a.u_px, b.v_px-a.v_px)
               for a, b in zip(corners, corners[1:]+corners[:1])]
    if min(lengths) < MIN_HEAD_EDGE_PX:
        return replace(result, reason="backside_head_pixel_span_insufficient")
    scale = (lengths[1] + lengths[3]) / (2 * expected_height_px)
    center = math.hypot(sum(p.u_px for p in corners)/4 - expected_center_u_px,
                        sum(p.v_px for p in corners)/4 - expected_center_v_px) / expected_height_px
    aspect = quadrilateral_aspect_ratio(corners) / (
        camera.fx_px / camera.fy_px * model_profile.head_width_m / model_profile.head_height_m)
    confidence = head_appearance_confidence(quality, scale, center, aspect)
    accepted = bool(
        MINIMUM_HEAD_SCALE_RATIO <= scale <= MAXIMUM_HEAD_SCALE_RATIO
        and 0 <= center <= MAXIMUM_HEAD_CENTER_ERROR_RATIO
        and MIN_NORMALIZED_ASPECT <= aspect <= MAX_NORMALIZED_ASPECT
        and MINIMUM_BACKSIDE_FACE_CONFIDENCE <= confidence <= 1.0)
    return replace(result, accepted=accepted,
                   reason=("current_head_appearance_backside_candidate" if accepted
                           else "backside_structure_or_projection_gates_rejected"),
                   head_scale_ratio=scale, head_center_error_ratio=center,
                   normalized_aspect=aspect, confidence=confidence)
