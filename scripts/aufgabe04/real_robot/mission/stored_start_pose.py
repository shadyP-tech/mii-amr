"""Resolve the exact Start QR and its durable, admitted robot pose.

Historical discovery evidence selects a target only. The return phase must
reproject it and certify a fresh route before issuing any motion permit.
"""

from dataclasses import dataclass
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, payload_sha256
from scripts.aufgabe04.navigation.approach.camera_decision_geometry_binding import (
    require_camera_recommendation_binding, require_projected_camera_candidate_binding,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.exact_two_camera_artifacts import (
    exact_two_camera_handoff_sha256, load_exact_two_camera_handoff,
)
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import REAL_VIEWPOINT_SOURCE, load_recommendation
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    StandSurveyRegistry, coverage_survey_plan_sha256, load_stand_survey_registry,
    stand_survey_registry_sha256,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.candidate.approach import CandidateApproachComplete, CandidateApproachConfig
from scripts.aufgabe04.real_robot.candidate.qr_goal_progress import validate_candidate_qr_goal_completion
from scripts.aufgabe04.real_robot.observer.qr_observation_binding import load_bound_qr_observation_pose
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256, load_candidate_snapshot
from scripts.aufgabe04.stations.server_identity_binding import load_observed_identities


@dataclass(frozen=True)
class StoredStartPose:
    candidate_uid: str
    pose: Pose2D
    source_frame: CandidatePlanningFrame
    registry: StandSurveyRegistry
    evidence: dict[str, object]


def _require_hash(actual, expected, label):
    if expected is None or actual != expected:
        raise ValueError(f"Start handoff {label} hash mismatch")


def load_stored_start_pose(
    completed: CandidateApproachComplete, config: CandidateApproachConfig,
) -> StoredStartPose:
    """Read stored artifacts, rejecting missing, ambiguous or substituted Start."""
    if not isinstance(completed, CandidateApproachComplete) or completed.motion_authorized:
        raise ValueError("Start handoff requires completed, stored camera exploration")
    required_paths = (
        completed.confirmed_candidate_snapshot_path, completed.observed_identities_path,
        completed.candidate_goal_progress_path, completed.qr_observation_catalog_path,
    )
    if any(path is None for path in required_paths):
        raise ValueError("Start handoff is missing stored completion artifacts")
    sources = [config.snapshot_path, *required_paths, completed.stand_facing_catalog_path]
    full_sha = candidate_snapshot_sha256(config.snapshot)
    _require_hash(candidate_snapshot_sha256(load_candidate_snapshot(config.snapshot_path)), full_sha, "full pool")
    confirmed = load_candidate_snapshot(completed.confirmed_candidate_snapshot_path)
    _require_hash(candidate_snapshot_sha256(confirmed), completed.confirmed_candidate_snapshot_sha256, "confirmed pool")
    observed = load_observed_identities(completed.observed_identities_path, candidate_snapshot=confirmed)
    _require_hash(payload_sha256(observed), completed.observed_identities_sha256, "observed identities")
    if observed["session_id"] != config.session_id:
        raise ValueError("Start handoff observed identities session mismatch")
    identities = observed["observed_qr_by_candidate"]
    goal = validate_candidate_qr_goal_completion(
        completed.candidate_goal_progress_path, candidate_snapshot=config.snapshot,
        confirmed_candidate_snapshot=confirmed, observed_qr_by_candidate=identities,
        expected_stand_count=completed.expected_stand_count,
    )
    _require_hash(payload_sha256(goal), completed.candidate_goal_progress_sha256, "completed QR goal")
    matches = [uid for uid, qr in identities.items() if qr == "Start"]
    if len(matches) != 1:
        raise ValueError("Start handoff requires exactly one admitted candidate with QR 'Start'")
    uid = matches[0]
    catalogs = []
    for path, digest, hash_field, kind in (
        (completed.stand_facing_catalog_path, completed.stand_facing_catalog_sha256,
         "stand_facing_catalog_sha256", "real_autonomous_stand_facing_poses"),
        (completed.qr_observation_catalog_path, completed.qr_observation_catalog_sha256,
         "qr_observation_pose_catalog_sha256", "real_autonomous_qr_observation_poses"),
    ):
        catalog = load_content_hashed_json(path, hash_field=hash_field)
        _require_hash(payload_sha256(catalog), digest, "pose catalog")
        expected = {
            "catalog_kind": kind, "session_id": config.session_id,
            "planning_frame": config.planning_frame, "map_bundle_sha256": config.plan.map_bundle_sha256,
            "coverage_plan_sha256": coverage_survey_plan_sha256(config.plan),
            "candidate_snapshot_sha256": full_sha,
            "confirmed_candidate_snapshot_sha256": completed.confirmed_candidate_snapshot_sha256,
            "observed_station_identities_sha256": completed.observed_identities_sha256,
            "candidate_goal_progress_sha256": completed.candidate_goal_progress_sha256,
            "robot_profile_sha256": config.robot_profile_sha256,
            "calibration_profile_sha256": config.calibration_profile_sha256,
        }
        if any(catalog.get(key) != value for key, value in expected.items()):
            raise ValueError("Start handoff pose catalog ancestry mismatch")
        catalogs.append((catalog, path, digest))
    records = [(record, catalog, path, digest) for catalog, path, digest in catalogs
               for record in catalog["records"] if record.get("candidate_uid") == uid or record.get("qr_id") == "Start"]
    if len(records) != 1 or records[0][0].get("candidate_uid") != uid or records[0][0].get("qr_id") != "Start":
        raise ValueError("Start handoff has missing or ambiguous stored poses")
    record, catalog, catalog_path, catalog_sha = records[0]
    registry_path = Path(catalog["source_registry_path"])
    registry = load_stand_survey_registry(registry_path, config.plan)
    _require_hash(stand_survey_registry_sha256(registry), catalog["source_registry_sha256"], "source registry")
    projection_path = Path(record["candidate_frame_projection_path"])
    camera_snapshot_path = Path(record["camera_candidate_snapshot_path"])
    handoff = None
    if config.exact_two_camera_handoff_path is not None:
        handoff = load_exact_two_camera_handoff(config.exact_two_camera_handoff_path)
        _require_hash(exact_two_camera_handoff_sha256(handoff), config.exact_two_camera_handoff_sha256, "camera handoff")
        sources.append(config.exact_two_camera_handoff_path)
    candidate = require_projected_camera_candidate_binding(
        record, canonical_snapshot_path=config.snapshot_path, canonical_snapshot=config.snapshot,
        registry=registry, source_registry_sha256=catalog["source_registry_sha256"],
        camera_snapshot_path=camera_snapshot_path, projection_path=projection_path, candidate_uid=uid,
        handoff=handoff,
    )
    projection = load_content_hashed_json(projection_path, hash_field="candidate_frame_projection_sha256")
    source_frame = CandidatePlanningFrame.from_evidence(projection["planning_frame_admission"])
    sources.extend((registry_path, projection_path, camera_snapshot_path))
    measured_center = None
    if catalog["catalog_kind"] == "real_autonomous_stand_facing_poses":
        observation_path = Path(record["recommendation_json"])
        require_camera_recommendation_binding(
            {**record, "camera_evidence_path": str(observation_path)},
            candidate=candidate, planning_frame=config.planning_frame,
        )
        recommendation = load_recommendation(
            observation_path, expected_frame=config.planning_frame,
            expected_source=REAL_VIEWPOINT_SOURCE, expected_simulation_only=False,
        )
        pose = Pose2D(**record["facing_pose"])
        if pose != recommendation.material_target.pose:
            raise ValueError("Start facing pose differs from admitted recommendation")
        measured_clearance = record.get("active_stand_clearance", {}).get("measured_center_clearance")
        if measured_clearance is not None:
            measured_center = {
                "x_m": recommendation.stand.center.x_m, "y_m": recommendation.stand.center.y_m,
                "uncertainty_m": recommendation.stand.uncertainty_m,
            }
        pose_kind = "geometry_validated_facing_pose"
    else:
        observation_path = Path(record["qr_observation_pose_json"])
        observation = load_bound_qr_observation_pose(
            observation_path, candidate_uid=uid, stream_id=f"{config.session_id}_{uid}",
            planning_frame=config.planning_frame, stand_x_m=candidate.geometry.x_m,
            stand_y_m=candidate.geometry.y_m, robot_profile_sha256=config.robot_profile_sha256,
            calibration_profile_sha256=config.calibration_profile_sha256,
        )
        _require_hash(observation["qr_verified_observation_pose_sha256"], record["qr_verified_observation_pose_sha256"], "QR observation")
        if observation["qr_id"] != "Start" or observation["robot_pose"] != record["robot_observation_pose"]:
            raise ValueError("Start viewing pose differs from admitted QR observation")
        pose = Pose2D(**record["robot_observation_pose"])
        pose_kind = "qr_verified_observation_pose"
    sources.append(observation_path)
    return StoredStartPose(uid, pose, source_frame, registry, {
        "qr_id": "Start", "candidate_uid": uid, "pose_kind": pose_kind,
        "catalog_path": str(catalog_path), "catalog_sha256": catalog_sha,
        "stored_pose": {"x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad},
        "source_planning_frame": source_frame.to_evidence(),
        **({} if measured_center is None else {"stored_measured_target_center": measured_center}),
        "source_artifacts": [{"path": str(Path(path).resolve()), "sha256": file_sha256(path)} for path in sources],
        "motion_authorized": False,
    })
