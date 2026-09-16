"""Bind QR-only discovery to its actual observation frame, without an axis.

This receipt records a place from which the robot read the candidate's QR. It
does not move the LiDAR candidate, create a facing recommendation, or authorize
navigation back to the recorded pose.
"""

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.camera_decision_geometry_binding import (
    require_projected_camera_candidate_binding,
)
from scripts.aufgabe04.real_robot.observer.qr_observation_binding import (
    load_bound_qr_observation_pose,
)
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256


def bind_qr_pose_discovery(*, observation, observation_frame, source_config,
                          source_registry, source_registry_sha256):
    """Authenticate the identity/pose and retain its immutable frame ancestry."""
    if (observation.recommendation_path is not None
            or observation.axis_observation_path is not None
            or observation.inspection_observation_path is not None):
        raise ValueError("QR observation pose must be a distinct terminal outcome")
    frame = observation_frame
    candidate = frame.candidate
    binding = frame.decision_binding
    fields = {} if binding is None else binding.to_receipt_fields()
    if source_registry is not None:
        if binding is None:
            raise ValueError("QR observation pose lacks its admitted candidate frame")
        projected = require_projected_camera_candidate_binding(
            fields, canonical_snapshot_path=source_config.snapshot_path,
            canonical_snapshot=source_config.snapshot, registry=source_registry,
            source_registry_sha256=source_registry_sha256,
            camera_snapshot_path=binding.camera_snapshot_path,
            projection_path=binding.projection_path, candidate_uid=candidate.candidate_uid,
        )
        if projected != candidate:
            raise ValueError("QR observation pose candidate differs from its frame projection")
    elif binding is not None or frame.planning_frame is not None:
        raise ValueError("QR observation pose frame lacks source registry ancestry")
    payload = load_bound_qr_observation_pose(
        observation.qr_observation_pose_path,
        candidate_uid=candidate.candidate_uid,
        stream_id=f"{source_config.session_id}_{candidate.candidate_uid}",
        planning_frame=frame.config.planning_frame,
        stand_x_m=candidate.geometry.x_m, stand_y_m=candidate.geometry.y_m,
        robot_profile_sha256=source_config.robot_profile_sha256,
        calibration_profile_sha256=source_config.calibration_profile_sha256,
    )
    if observation.qr_id != payload["qr_id"]:
        raise ValueError("QR observation pose identity differs from the observer result")
    return {
        "schema_version": 1,
        "record_kind": "qr_verified_observation_pose",
        "session_id": source_config.session_id,
        "candidate_uid": candidate.candidate_uid,
        "qr_id": payload["qr_id"],
        "planning_frame": payload["planning_frame"],
        "stand_center": payload["stand_center"],
        "robot_observation_pose": payload["robot_pose"],
        "sensor_stamp_sec": payload["sensor_stamp_sec"],
        "scan_stamp_sec": payload["scan_stamp_sec"],
        "stand_axis_rad": None,
        "facing_ready": False,
        "completion_scope": "discovery_only",
        "motion_authorized": False,
        "candidate_geometry_unchanged": True,
        "candidate_snapshot_sha256": candidate_snapshot_sha256(source_config.snapshot),
        "qr_observation_pose_json": str(observation.qr_observation_pose_path),
        "qr_verified_observation_pose_sha256": payload["qr_verified_observation_pose_sha256"],
        "robot_profile_sha256": payload["robot_profile_sha256"],
        "calibration_profile_sha256": payload["calibration_profile_sha256"],
        **fields,
    }


def write_qr_pose_discovery(path, record):
    return write_content_hashed_json(path, record, hash_field="candidate_qr_discovery_sha256")


def write_qr_pose_catalog(path, *, metadata, records):
    """Publish only fallback observations; facing records remain separate."""
    payload = {
        **metadata,
        "schema_version": 1,
        "catalog_kind": "real_autonomous_qr_observation_poses",
        "completion_scope": "discovery_only",
        "facing_ready": False,
        "motion_authorized": False,
        "stand_count": len(records),
        "records": sorted(records, key=lambda record: record["candidate_uid"]),
    }
    return write_content_hashed_json(path, payload, hash_field="qr_observation_pose_catalog_sha256")
