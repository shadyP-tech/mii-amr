"""Pure diagnostic payloads and profile-bound fallback geometry."""

from __future__ import annotations

from scripts.aufgabe04.perception.stand_axis.model_profile import StandModelProfile
from scripts.aufgabe04.perception.stand_axis.models import (
    StandAxisEdgeDebugArtifacts,
    StandAxisImageEstimate,
)


def resolved_fallback_face_to_qr_ratio(
    configured_ratio: float | None,
    stand_model_profile: StandModelProfile | None,
) -> float | None:
    """Keep fallback geometry consistent with a loaded metric model."""

    if stand_model_profile is None:
        return configured_ratio
    return (
        stand_model_profile.head_width_m
        / stand_model_profile.qr_symbol_width_m
    )


def metric_model_status_payload(
    *,
    profile: StandModelProfile | None,
    inputs_ready: bool,
    estimate: StandAxisImageEstimate | None,
    artifacts: StandAxisEdgeDebugArtifacts | None,
) -> dict[str, object]:
    """Serialize model acquisition/refinement state even when fallback wins."""

    enabled = profile is not None
    return {
        "enabled": enabled,
        "inputs_ready": bool(inputs_ready),
        "profile_id": None if profile is None else profile.profile_id,
        "profile_sha256": None if profile is None else profile.sha256,
        "measurement_status": (
            None if profile is None else profile.measurement_status
        ),
        "committable": bool(profile is not None and profile.committable),
        "usable": bool(estimate is not None and estimate.usable),
        "reason": (
            "metric_model_disabled"
            if profile is None
            else (
                "metric_inputs_unavailable"
                if not inputs_ready or estimate is None
                else estimate.reason
            )
        ),
        "evidence_state": None if estimate is None else estimate.evidence_state,
        "qr_detected": bool(artifacts is not None and artifacts.qr_detected),
        "qr_detection_scale": (
            None if artifacts is None else artifacts.qr_detection_scale
        ),
        "pose_seed_source": (
            None if artifacts is None else artifacts.pose_seed_source
        ),
        "predicted_corners_available": bool(
            artifacts is not None and artifacts.predicted_corners is not None
        ),
        "projected_landmarks": (
            []
            if artifacts is None or artifacts.projected_landmarks is None
            else sorted(artifacts.projected_landmarks)
        ),
        "pose_reprojection_rmse_px": (
            None if estimate is None else estimate.pose_reprojection_rmse_px
        ),
        "pose_ambiguity_gap_px": (
            None if estimate is None else estimate.pose_ambiguity_gap_px
        ),
        "refinement_support_mean": (
            None if artifacts is None else artifacts.refinement_support_mean
        ),
        "corridor_half_width_px": (
            None
            if artifacts is None
            else artifacts.model_corridor_half_width_px
        ),
        "pose_fit_source": (
            None if artifacts is None else artifacts.model_pose_fit_source
        ),
    }
