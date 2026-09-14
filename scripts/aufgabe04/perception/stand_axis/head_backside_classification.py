"""Classify a current measured head without supplying another angle estimate.

The four physical head corners own the angle. Raw outer-border/neck evidence,
candidate projection gates and current marker absence can separately label a
backside *candidate*. The observer still requires repeated fresh, stationary,
uniquely LiDAR-associated samples and vetoes any marker seen in that epoch.
"""

from dataclasses import dataclass, replace
import math

from scripts.aufgabe04.artifacts.backside_axis_observation import (
    BACKSIDE_AXIS_SAMPLE_SOURCE, BACKSIDE_MODEL_EVIDENCE_STATE, BACKSIDE_VISIBLE_FACE,
    MAXIMUM_HEAD_CENTER_ERROR_RATIO, MAXIMUM_HEAD_SCALE_RATIO,
    MINIMUM_BACKSIDE_FACE_CONFIDENCE, MINIMUM_HEAD_SCALE_RATIO,
)
from scripts.aufgabe04.perception.stand_axis.geometry import quadrilateral_aspect_ratio
from scripts.aufgabe04.perception.stand_axis.head_model_quality import (
    MEASURED_HEAD_AXIS_SOURCE, validated_head_model_quality,
)
from scripts.aufgabe04.perception.stand_axis.head_model_neck import HeadNeckJunction

MIN_NORMALIZED_ASPECT = math.cos(math.radians(70.0))
MAX_NORMALIZED_ASPECT = 1.35


@dataclass(frozen=True)
class HeadBacksideClassification:
    accepted: bool
    reason: str
    profile_sha256: str
    corners: tuple
    yaw_deg: float
    head_scale_ratio: float | None = None
    head_center_error_ratio: float | None = None
    normalized_aspect: float | None = None
    confidence: float | None = None
    angle_source: str = MEASURED_HEAD_AXIS_SOURCE


def _gate_score(value, lower, upper):
    midpoint = (lower + upper) / 2.0
    return max(0.0, 1.0 - abs(value - midpoint) / ((upper - lower) / 2.0))


def _face_confidence(quality, scale, center, aspect):
    return min(1.0, max(0.0,
        .50 * quality.raw_border_support_mean
        + .20 * _gate_score(scale, MINIMUM_HEAD_SCALE_RATIO, MAXIMUM_HEAD_SCALE_RATIO)
        + .20 * max(0.0, 1.0 - center / MAXIMUM_HEAD_CENTER_ERROR_RATIO)
        + .10 * _gate_score(aspect, MIN_NORMALIZED_ASPECT, MAX_NORMALIZED_ASPECT)))


def _classification_valid(evidence, estimate, debug):
    quality = getattr(debug, "head_model_quality", None)
    junction = getattr(debug, "head_neck_junction", None)
    if (not isinstance(evidence, HeadBacksideClassification)
            or evidence.accepted is not True
            or evidence.reason != "current_head_geometry_and_marker_absence"
            or evidence.angle_source != MEASURED_HEAD_AXIS_SOURCE
            or evidence.corners != estimate.corners or evidence.yaw_deg != estimate.yaw_deg
            or evidence.profile_sha256 != estimate.model_profile_sha256
            or not validated_head_model_quality(quality)
            or quality.profile_sha256 != evidence.profile_sha256
            or debug.qr_detected is not False or debug.qr_marker_verified is not False
            or not isinstance(junction, HeadNeckJunction) or junction.accepted is not True
            or junction.reason != "head_neck_junction_verified"):
        return False
    scale, center, aspect, confidence = (
        evidence.head_scale_ratio, evidence.head_center_error_ratio,
        evidence.normalized_aspect, evidence.confidence,
    )
    if any(type(v) not in (int, float) or not math.isfinite(v)
           for v in (scale, center, aspect, confidence)):
        return False
    return bool(
        MINIMUM_HEAD_SCALE_RATIO <= scale <= MAXIMUM_HEAD_SCALE_RATIO
        and 0.0 <= center <= MAXIMUM_HEAD_CENTER_ERROR_RATIO
        and MIN_NORMALIZED_ASPECT <= aspect <= MAX_NORMALIZED_ASPECT
        and MINIMUM_BACKSIDE_FACE_CONFIDENCE <= confidence <= 1.0
        and math.isclose(confidence, _face_confidence(quality, scale, center, aspect), abs_tol=1e-12)
    )


def is_classified_measured_head_backside(estimate, debug):
    """Recognize the combined source only with its bound classification proof."""
    proof = getattr(debug, "head_backside_classification", None)
    return bool(
        estimate.source == BACKSIDE_AXIS_SAMPLE_SOURCE
        and estimate.evidence_state == BACKSIDE_MODEL_EVIDENCE_STATE
        and debug.evidence_state == BACKSIDE_MODEL_EVIDENCE_STATE
        and debug.model_pose_fit_source == MEASURED_HEAD_AXIS_SOURCE
        and estimate.visible_face == BACKSIDE_VISIBLE_FACE
        and estimate.camera_face_normal_xyz is None
        and estimate.camera_face_center_xyz_m is None
        and _classification_valid(proof, estimate, debug)
        and estimate.visible_face_confidence == proof.confidence
        and debug.head_scale_ratio == proof.head_scale_ratio
        and debug.head_center_error_ratio == proof.head_center_error_ratio
    )


def classify_current_head_backside(
    estimate, debug, *, model_profile, camera,
    expected_center_u_px, expected_center_v_px, expected_height_px,
):
    """Attach side evidence to one admitted current fit; never refit its angle."""
    # Import locally to keep the admission validator independent of fitting.
    from scripts.aufgabe04.perception.stand_axis.head_model_admission import admit_measured_head_model

    if estimate.source != MEASURED_HEAD_AXIS_SOURCE:
        return estimate, debug
    proof = HeadBacksideClassification(
        False, "backside_current_head_geometry_required", estimate.model_profile_sha256,
        estimate.corners, estimate.yaw_deg,
    )
    def unchanged(reason):
        return estimate, replace(debug, head_backside_classification=replace(proof, reason=reason))

    if not admit_measured_head_model(
        estimate=estimate, debug=debug,
        yaw_rad=math.radians(estimate.yaw_deg) if estimate.yaw_deg is not None else math.nan,
    ).accepted:
        return unchanged(proof.reason)
    if debug.qr_detected is not False or debug.qr_marker_verified is not False:
        return unchanged("backside_current_marker_absence_required")
    if (not model_profile.committable or model_profile.environment != "physical"
            or model_profile.sha256 != estimate.model_profile_sha256):
        return unchanged("backside_measured_physical_profile_required")
    expected = (expected_center_u_px, expected_center_v_px, expected_height_px)
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in expected)
            or expected_height_px <= 0):
        return unchanged("backside_candidate_projection_required")
    corners = estimate.corners
    height = (math.hypot(corners[3].u_px - corners[0].u_px, corners[3].v_px - corners[0].v_px)
              + math.hypot(corners[2].u_px - corners[1].u_px, corners[2].v_px - corners[1].v_px)) / 2
    scale = height / expected_height_px
    center = math.hypot(sum(p.u_px for p in corners) / 4 - expected_center_u_px,
                        sum(p.v_px for p in corners) / 4 - expected_center_v_px) / expected_height_px
    aspect = quadrilateral_aspect_ratio(corners) / (
        camera.fx_px / camera.fy_px * model_profile.head_width_m / model_profile.head_height_m)
    confidence = _face_confidence(debug.head_model_quality, scale, center, aspect)
    proof = replace(proof, accepted=True, reason="current_head_geometry_and_marker_absence",
                    head_scale_ratio=scale, head_center_error_ratio=center,
                    normalized_aspect=aspect, confidence=confidence)
    if not _classification_valid(proof, estimate, debug):
        proof = replace(proof, accepted=False)
        return unchanged("backside_structure_or_projection_gates_rejected")
    return replace(
        estimate, source=BACKSIDE_AXIS_SAMPLE_SOURCE,
        reason="axis_estimated_current_measured_head_backside",
        evidence_state=BACKSIDE_MODEL_EVIDENCE_STATE, visible_face=BACKSIDE_VISIBLE_FACE,
        visible_face_confidence=confidence,
        # Symmetric head geometry never establishes a directed front normal.
        camera_face_normal_xyz=None, camera_face_center_xyz_m=None,
    ), replace(
        debug, evidence_state=BACKSIDE_MODEL_EVIDENCE_STATE,
        model_reason="axis_estimated_current_measured_head_backside",
        visible_face_reason=proof.reason, head_backside_classification=proof,
        head_scale_ratio=scale, head_center_error_ratio=center,
    )
