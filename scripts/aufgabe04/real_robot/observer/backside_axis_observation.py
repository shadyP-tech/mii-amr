"""Passive-observer builder for a model-backed backside-axis receipt.

Acquisition and temporal evidence remain observer responsibilities. The
artifact layer owns the shared schema and validator consumed by navigation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from scripts.aufgabe04.artifacts.backside_axis_observation import (
    BACKSIDE_AXIS_OBSERVATION_KIND,
    BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION,
    BACKSIDE_AXIS_SAMPLE_SOURCE,
    BACKSIDE_CLASSIFICATION_BASIS,
    BACKSIDE_MODEL_EVIDENCE_STATE,
    BACKSIDE_SAMPLE_GATE_KEYS,
    BACKSIDE_VISIBLE_FACE,
    PASSIVE_VIEWPOINT_OBSERVER_VERSION,
    REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE,
    TARGET_REGISTRATION_MODE_BOUNDED_CAMERA_LIDAR,
    TARGET_REGISTRATION_MODE_MAP_PROJECTION,
    validate_backside_axis_observation,
)


def build_backside_axis_observation(
    *,
    stream_id: str,
    stand_id: str,
    planning_frame: str,
    stand_x_m: float,
    stand_y_m: float,
    robot_x_m: float,
    robot_y_m: float,
    robot_yaw_rad: float,
    stand_axis_rad: float,
    axis_confidence: float,
    axis_sample_count: int,
    consensus_source: str,
    estimate_source: str,
    estimate_evidence_state: str,
    estimate_visible_face: str | None,
    visible_face_confidence: float,
    debug_qr_detected: bool,
    qr_texts: Sequence[str],
    evidence_qr_sample_count: int,
    evidence_tentative_qr_id: str | None,
    evidence_latched_qr_id: str | None,
    qr_marker_seen_in_stationary_epoch: bool,
    all_samples_stationary: bool,
    all_samples_synchronized: bool,
    all_samples_lidar_associated: bool,
    sensor_stamp_sec: float,
    stand_model_profile_sha256: str,
    stand_model_measurement_status: str,
    head_scale_ratio: float,
    head_center_error_ratio: float,
    pose_reprojection_rmse_px: float | None,
    pose_ambiguity_gap_px: float | None,
    robot_profile_sha256: str,
    calibration_profile_sha256: str,
    target_registration: Mapping[str, object],
) -> dict[str, object]:
    """Build schema 3 from repeated, registered current-frame evidence."""

    if not isinstance(target_registration, Mapping):
        raise ValueError("backside target registration is not a mapping")
    registration_mode = target_registration.get("mode")
    if registration_mode == TARGET_REGISTRATION_MODE_MAP_PROJECTION:
        required_consensus_source = BACKSIDE_AXIS_SAMPLE_SOURCE
    elif registration_mode == TARGET_REGISTRATION_MODE_BOUNDED_CAMERA_LIDAR:
        required_consensus_source = REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE
    else:
        raise ValueError("backside target registration mode is unsupported")
    if consensus_source != required_consensus_source:
        raise ValueError(
            "backside consensus source does not match target registration"
        )
    if estimate_source != BACKSIDE_AXIS_SAMPLE_SOURCE:
        raise ValueError("backside estimate source is not current-frame model evidence")
    if estimate_evidence_state != BACKSIDE_MODEL_EVIDENCE_STATE:
        raise ValueError("backside estimate is not fresh backside evidence")
    if estimate_visible_face != BACKSIDE_VISIBLE_FACE:
        raise ValueError("backside estimate did not classify a backside candidate")
    if debug_qr_detected is not False:
        raise ValueError("a QR marker is present in the current frame")
    if isinstance(qr_texts, (str, bytes)) or list(qr_texts) != []:
        raise ValueError("decoded QR text forbids a backside observation")
    if (
        isinstance(evidence_qr_sample_count, bool)
        or not isinstance(evidence_qr_sample_count, int)
        or evidence_qr_sample_count != 0
        or evidence_tentative_qr_id is not None
        or evidence_latched_qr_id is not None
    ):
        raise ValueError("the stationary evidence epoch contains QR evidence")
    if qr_marker_seen_in_stationary_epoch is not False:
        raise ValueError("the stationary evidence epoch contains a QR marker")

    sample_gate_evidence = {
        "all_samples_stationary": all_samples_stationary,
        "all_samples_synchronized": all_samples_synchronized,
        "all_samples_lidar_associated": all_samples_lidar_associated,
        # These are guaranteed by the exact accepted estimator source and the
        # marker/text/epoch checks above, then materialized for the consumer.
        "all_samples_current_frame_model_geometry": True,
        "all_samples_qr_marker_absent": True,
    }
    if any(
        sample_gate_evidence[name] is not True
        for name in BACKSIDE_SAMPLE_GATE_KEYS
    ):
        raise ValueError("not every backside axis sample passed the sensor gates")

    payload: dict[str, object] = {
        "schema_version": BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION,
        "observation_kind": BACKSIDE_AXIS_OBSERVATION_KIND,
        "motion_capability": "none",
        "stream_id": stream_id,
        "stand_id": stand_id,
        "planning_frame": planning_frame,
        "stand_center": {"x_m": stand_x_m, "y_m": stand_y_m},
        "robot_pose": {
            "x_m": robot_x_m,
            "y_m": robot_y_m,
            "yaw_rad": robot_yaw_rad,
        },
        "stand_axis_rad": stand_axis_rad,
        "axis_confidence": axis_confidence,
        "axis_sample_count": axis_sample_count,
        "axis_sample_source": consensus_source,
        "sensor_stamp_sec": sensor_stamp_sec,
        "observer_version": PASSIVE_VIEWPOINT_OBSERVER_VERSION,
        "visible_face": BACKSIDE_VISIBLE_FACE,
        "visible_face_source": estimate_source,
        "visible_face_confidence": visible_face_confidence,
        "classification_basis": BACKSIDE_CLASSIFICATION_BASIS,
        "qr_marker_detected": False,
        "qr_texts": [],
        "qr_absent_sample_count": axis_sample_count,
        "sample_gate_evidence": sample_gate_evidence,
        "stand_model_profile_sha256": stand_model_profile_sha256,
        "stand_model_measurement_status": stand_model_measurement_status,
        "model_evidence_state": estimate_evidence_state,
        "head_scale_ratio": head_scale_ratio,
        "head_center_error_ratio": head_center_error_ratio,
        "pose_reprojection_rmse_px": pose_reprojection_rmse_px,
        "pose_ambiguity_gap_px": pose_ambiguity_gap_px,
        "robot_profile_sha256": robot_profile_sha256,
        "calibration_profile_sha256": calibration_profile_sha256,
        "target_registration": dict(target_registration),
    }
    validate_backside_axis_observation(payload)
    return payload


__all__ = [
    "build_backside_axis_observation",
    "validate_backside_axis_observation",
]
