"""Adapt current measured-head pixels to a temporal veto, without smoothing."""

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.head_model_quality import (
    HeadModelQuality, MEASURED_HEAD_AXIS_SOURCE, MIN_HEAD_EDGE_PX,
    MAX_HEAD_REPROJECTION_RMSE_PX, MIN_HEAD_CORNER_SIGMA_PX, validated_head_model_quality,
)
from scripts.aufgabe04.perception.stand_axis.head_outer_border import current_head_boundary_eligible
from scripts.aufgabe04.real_robot.observer.contract import (
    BACKSIDE_AXIS_SAMPLE_SOURCE, REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE,
)
from scripts.aufgabe04.real_robot.observer.evidence import AxisWindowReview
from scripts.aufgabe04.real_robot.observer.head_temporal_consistency import HeadTemporalContext


MEASURED_HEAD_SOURCES = (MEASURED_HEAD_AXIS_SOURCE, BACKSIDE_AXIS_SAMPLE_SOURCE,
                         REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE)


@dataclass(frozen=True)
class CurrentHeadWindowInput:
    frame_stamp_sec: float
    model_sha256: str
    camera_signature: tuple
    corners_full_image: tuple
    projected_center_px: tuple
    expected_head_height_px: float
    yaw_rad: float
    plausible_yaws_rad: tuple


def current_head_window_input(estimate, debug, *, frame_stamp_sec, camera_signature,
                              roi, projected_center_px, expected_head_height_px):
    """Compare eligible physical heads, including their planar ambiguity.

    Raw-supported paper/panel corners are not alternative measurements of the
    physical head. Retained diagnostic IPPE poses cannot promote those rejected
    boundaries into this window or clear earlier valid head-angle evidence.
    """
    quality = getattr(debug, "head_model_quality", None)
    if (debug.model_pose_fit_source != MEASURED_HEAD_AXIS_SOURCE
            or estimate.source not in MEASURED_HEAD_SOURCES
            or estimate.evidence_state == "predicted_only"
            or debug.evidence_state == "predicted_only"
            or estimate.model_measurement_status != "measured"
            or debug.model_measurement_status != "measured"
            or not isinstance(quality, HeadModelQuality)
            or not estimate.model_profile_sha256
            or quality.profile_sha256 != estimate.model_profile_sha256
            or debug.model_profile_sha256 != estimate.model_profile_sha256
            or quality.raw_corner_support_accepted is not True
            or quality.outer_border_verified is not True
            or quality.raw_border_support_mean is None
            or not .60 <= quality.raw_border_support_mean <= 1.
            or estimate.corners is None or len(estimate.corners) != 4):
        return None
    accepted_head = validated_head_model_quality(quality)
    ambiguous_head = bool(
        quality.accepted is False and quality.axis_ambiguous is True
        and quality.reason == "head_model_planar_axis_ambiguous"
        and quality.pose_model == "measured_head_only"
        and quality.face_semantics == "undirected_plane"
        and quality.minimum_edge_length_px is not None
        and math.isfinite(quality.minimum_edge_length_px)
        and quality.minimum_edge_length_px >= MIN_HEAD_EDGE_PX
        and quality.reprojection_rmse_px is not None
        and 0. <= quality.reprojection_rmse_px <= MAX_HEAD_REPROJECTION_RMSE_PX)
    if not accepted_head and not ambiguous_head:
        return None
    if not current_head_boundary_eligible(estimate, debug):
        return None
    if estimate.reason in {"current_border_matches_verified_qr_panel",
                           "current_physical_head_boundary_unresolved"}:
        return None
    hypotheses = tuple(h for h in (getattr(debug, "head_pose_hypotheses", None) or ())
                       if h.positive_depth and math.isfinite(h.reprojection_rmse_px)
                       and 0 <= h.reprojection_rmse_px <= 2. and math.isfinite(h.yaw_deg))
    best_residual = min((h.reprojection_rmse_px for h in hypotheses), default=math.inf)
    # Match the producer's ambiguity neighborhood. Diagnostic alternatives
    # may veto older head angles; they never authorize a current measurement.
    residual_gap = max(MIN_HEAD_CORNER_SIGMA_PX, quality.reprojection_rmse_px) if ambiguous_head else .10
    plausible = tuple(math.radians(h.yaw_deg) for h in hypotheses
                      if h.reprojection_rmse_px <= best_residual + residual_gap)
    yaw = estimate.yaw_deg
    if type(yaw) in (int, float) and math.isfinite(yaw) and estimate.usable:
        current = math.radians(yaw)
    elif ambiguous_head and plausible:
        # Only ambiguity of a physical head may veto old angles. It remains
        # diagnostic and the caller still records no current axis for it.
        current = plausible[0]
    else:
        return None
    return CurrentHeadWindowInput(
        frame_stamp_sec, estimate.model_profile_sha256, camera_signature,
        tuple((p.u_px + roi.x0, p.v_px + roi.y0) for p in estimate.corners),
        projected_center_px, expected_head_height_px, current, plausible)


def review_current_head_window(tracker, current, *, snapshot):
    """Called only after common freshness, identity and stationary frame gates."""
    decision = tracker.observe(
        context=HeadTemporalContext(snapshot.target_key, current.model_sha256,
                                    current.camera_signature, snapshot.motion_epoch),
        stamp_sec=current.frame_stamp_sec, yaw_rad=current.yaw_rad,
        corners_full_image=current.corners_full_image,
        projected_center_px=current.projected_center_px,
        expected_head_height_px=current.expected_head_height_px,
        plausible_yaws_rad=current.plausible_yaws_rad)
    return AxisWindowReview(
        decision.current_sample_accepted,
        MEASURED_HEAD_SOURCES if decision.reset_axis_evidence else (),
        decision.reason), decision
