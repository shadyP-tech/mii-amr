"""Canonical schema-3 backside-axis receipts used by navigation tests."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidatePlanningFrame,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance,
    CandidatePoint2D,
    reproject_candidate_point,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)

from scripts.aufgabe04.navigation.approach.camera_axis_binding import (
    BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION,
    BACKSIDE_CLASSIFICATION_BASIS,
    BACKSIDE_CURRENT_FRAME_SOURCE,
    BACKSIDE_MODEL_EVIDENCE_STATE,
    BACKSIDE_VISIBLE_FACE,
    PASSIVE_VIEWPOINT_OBSERVER_VERSION,
    REAL_STAND_AXIS_OBSERVATION_KIND,
)
from scripts.aufgabe04.artifacts.backside_axis_observation import (
    TARGET_REGISTRATION_LIDAR_SOURCE_MAP,
    TARGET_REGISTRATION_MODE_MAP_PROJECTION,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    candidate_snapshot_sha256,
    new_candidate_snapshot,
    write_candidate_snapshot,
)


def backside_axis_payload(
    *,
    stand_id: str = "candidate_1",
    planning_frame: str = "map",
    stand_x_m: float = 0.0,
    stand_y_m: float = 0.0,
    robot_x_m: float = 0.0,
    robot_y_m: float = 0.7,
    robot_yaw_rad: float = 0.0,
    stand_axis_rad: float = 0.0,
) -> dict[str, object]:
    """Return a complete motion-neutral backside evidence fixture."""

    sample_count = 7
    return {
        "schema_version": BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION,
        "observation_kind": REAL_STAND_AXIS_OBSERVATION_KIND,
        "motion_capability": "none",
        "stream_id": "test_backside_axis_stream",
        "stand_id": stand_id,
        "planning_frame": planning_frame,
        "stand_center": {"x_m": stand_x_m, "y_m": stand_y_m},
        "robot_pose": {
            "x_m": robot_x_m,
            "y_m": robot_y_m,
            "yaw_rad": robot_yaw_rad,
        },
        "stand_axis_rad": stand_axis_rad,
        "axis_confidence": 0.90,
        "axis_sample_count": sample_count,
        "axis_sample_source": BACKSIDE_CURRENT_FRAME_SOURCE,
        "visible_face": BACKSIDE_VISIBLE_FACE,
        "visible_face_source": BACKSIDE_CURRENT_FRAME_SOURCE,
        "visible_face_confidence": 0.90,
        "classification_basis": BACKSIDE_CLASSIFICATION_BASIS,
        "qr_marker_detected": False,
        "qr_texts": [],
        "qr_absent_sample_count": sample_count,
        "model_evidence_state": BACKSIDE_MODEL_EVIDENCE_STATE,
        "observer_version": PASSIVE_VIEWPOINT_OBSERVER_VERSION,
        "stand_model_profile_sha256": "0" * 64,
        "stand_model_measurement_status": "measured",
        "sensor_stamp_sec": 123.5,
        "head_scale_ratio": 1.0,
        "head_center_error_ratio": 0.05,
        "target_registration": {
            "mode": TARGET_REGISTRATION_MODE_MAP_PROJECTION,
            "original_head_center_error_ratio": 0.05,
            "center_offset_limit_ratio": 0.55,
            "final_strict_head_center_error_ratio": 0.05,
            "map_bearing_rad": 0.0,
            "lidar_search_bearing_rad": 0.0,
            "camera_map_bearing_delta_rad": 0.0,
            "bearing_delta_limit_rad": math.radians(3.0),
            "lidar_search_bearing_source": (
                TARGET_REGISTRATION_LIDAR_SOURCE_MAP
            ),
            "unique_eligible_lidar_cluster_required": False,
            "eligible_lidar_cluster_count": 1,
        },
        "pose_reprojection_rmse_px": None,
        "pose_ambiguity_gap_px": None,
        "robot_profile_sha256": "1" * 64,
        "calibration_profile_sha256": "2" * 64,
        "sample_gate_evidence": {
            "all_samples_stationary": True,
            "all_samples_synchronized": True,
            "all_samples_lidar_associated": True,
            "all_samples_current_frame_model_geometry": True,
            "all_samples_qr_marker_absent": True,
        },
    }


def write_candidate_frame_projection_fixture(
    path: Path,
    *,
    candidate_uid: str,
    canonical_x_m: float,
    canonical_y_m: float,
    transform_x_m: float,
    transform_y_m: float,
    transform_yaw_rad: float,
    source_registry_sha256: str = "b" * 64,
    source_snapshot_path: Path | None = None,
) -> tuple[str, float, float]:
    """Write a production-shaped, strictly valid projection artifact."""

    source_snapshot_path = (
        path.parent / "canonical_candidate_snapshot.json"
        if source_snapshot_path is None
        else Path(source_snapshot_path)
    ).absolute()
    source_candidate = FrozenCandidate(
        candidate_uid=candidate_uid,
        geometry=CandidateGeometry(
            x_m=canonical_x_m,
            y_m=canonical_y_m,
            radius_m=0.06,
            uncertainty_m=0.02,
            keepout_radius_m=0.31,
        ),
        source=CandidateSource(
            source_kind="lidar/stand_coverage_survey",
            source_artifact_sha256=source_registry_sha256,
            detector_config_sha256="d" * 64,
            observation_ids=("fixture_observation",),
        ),
        confidence=0.90,
        hit_count=4,
        first_seen_sec=1.0,
        last_seen_sec=2.0,
    )
    source_snapshot = new_candidate_snapshot(
        snapshot_id="fixture_source_snapshot",
        created_unix_sec=3.0,
        planning_frame="map",
        map_bundle_sha256="e" * 64,
        candidates=(source_candidate,),
    )
    write_candidate_snapshot(source_snapshot_path, source_snapshot)
    provenance = CandidateFrameProvenance.from_frozen_map_observation(
        map_frame="map",
        odom_frame="odom",
        frozen_map_point=CandidatePoint2D(canonical_x_m, canonical_y_m),
        frozen_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
        source_evidence_id="fixture_frame_evidence",
    )
    transform = PlanarTransform2D(
        transform_x_m, transform_y_m, transform_yaw_rad
    )
    result = reproject_candidate_point(provenance, transform)
    current_x_m = result.current_map_point.x_m
    current_y_m = result.current_map_point.y_m
    projected_snapshot = replace(
        source_snapshot,
        candidates=(
            replace(
                source_candidate,
                geometry=replace(
                    source_candidate.geometry,
                    x_m=current_x_m,
                    y_m=current_y_m,
                ),
            ),
        ),
    )
    projected_snapshot_path = path.with_name(
        f"{path.stem}_candidate_snapshot.json"
    ).absolute()
    write_candidate_snapshot(projected_snapshot_path, projected_snapshot)
    planning_frame = CandidatePlanningFrame(
        current_pose=Pose2D(0.0, 0.0, 0.0),
        map_from_odom=transform,
    )
    digest = write_content_hashed_json(
        path,
        {
            "schema_version": 1,
            "source_candidate_snapshot_sha256": candidate_snapshot_sha256(
                source_snapshot
            ),
            "source_registry_sha256": source_registry_sha256,
            "projected_candidate_snapshot_sha256": (
                candidate_snapshot_sha256(projected_snapshot)
            ),
            "planning_frame_admission": planning_frame.to_evidence(),
            "candidate_reprojections": {candidate_uid: result.to_mapping()},
            "motion_authorized": False,
            "source_candidate_snapshot_path": str(source_snapshot_path),
            "projected_candidate_snapshot_path": str(
                projected_snapshot_path
            ),
        },
        hash_field="candidate_frame_projection_sha256",
    )
    return digest, current_x_m, current_y_m
