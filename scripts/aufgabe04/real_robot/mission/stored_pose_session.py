"""Read a completed exploration session without granting new motion authority."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import load_coverage_survey_plan
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import load_mission_leg_motion_authorization
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.navigation.execution.route_uncertainty_defaults import DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.real_robot.autonomous_runner.mission_config import _physical_clearance
from scripts.aufgabe04.real_robot.candidate.approach import CandidateApproachComplete, CandidateApproachConfig
from scripts.aufgabe04.real_robot.configuration.profile import RealRobotProfile, load_real_robot_profile, real_robot_profile_sha256
from scripts.aufgabe04.real_robot.configuration.site_contract import validate_physical_site_contract
from scripts.aufgabe04.real_robot.mission.stored_start_pose import StoredAdmittedPose, load_stored_admitted_poses
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256, load_candidate_snapshot


@dataclass(frozen=True)
class StoredPoseSession:
    completed: CandidateApproachComplete
    config: CandidateApproachConfig
    poses_by_qr: Mapping[str, StoredAdmittedPose]
    profile: RealRobotProfile


def _path(payload, key):
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"stored session requires {key}")
    # Artifact paths belong to their original execution context. Relocating
    # only the directory does not silently rewrite any authenticated ancestry.
    return Path(value)


def load_stored_pose_session(
    session_root: Path, robot_profile_path: Path, *, physical_site_path: Path | None = None,
) -> StoredPoseSession:
    """Authenticate every stored QR/facing pose and reconstruct planning inputs.

The old authorization supplies source identity and branch evidence only.
An independent tour must replace it with its own newly admitted scope, and
must separately establish that historical odometry has not been reset.
"""
    root = Path(session_root).resolve(strict=True)
    summary = json.loads((root / "mission_summary.json").read_text())
    if not isinstance(summary, dict) or summary.get("camera_validation_complete") is not True:
        raise ValueError("stored session has no completed camera exploration")
    if summary.get("motion_authorized") is not False:
        raise ValueError("stored session retains ambiguous motion authority")
    profile = load_real_robot_profile(robot_profile_path)
    site_path = physical_site_path or (
        Path(__file__).resolve().parents[4] / "docs" / "setups" / f"{profile.physical_site_id}.json"
    )
    site = validate_physical_site_contract(site_path, profile=profile,
        requested_expected_stand_count=summary.get("expected_stand_count"))
    survey_root = _path(summary, "survey_root")
    plan = load_coverage_survey_plan(survey_root / "coverage_plan.json")
    if plan.map_bundle_sha256 != site.map_bundle.bundle_sha256 or plan.planning_frame != profile.map_frame:
        raise ValueError("stored coverage plan differs from the current physical map/profile")
    snapshot_path = _path(summary, "candidate_snapshot")
    snapshot = load_candidate_snapshot(snapshot_path, required_map_bundle_sha256=plan.map_bundle_sha256)
    if candidate_snapshot_sha256(snapshot) != summary.get("candidate_snapshot_sha256"):
        raise ValueError("stored session full candidate snapshot hash mismatch")
    facing_path = _path(summary, "stand_facing_catalog")
    qr_path = _path(summary, "qr_observation_pose_catalog")
    facing = load_content_hashed_json(facing_path, hash_field="stand_facing_catalog_sha256")
    qr = load_content_hashed_json(qr_path, hash_field="qr_observation_pose_catalog_sha256")
    completed = CandidateApproachComplete(
        stand_count=summary["stand_count"], visit_order=tuple(summary.get("inspection_order", ())),
        identity_registry_path=(None if not summary.get("station_identity_registry") else _path(summary, "station_identity_registry")),
        identity_registry_sha256=summary.get("station_identity_registry_sha256"),
        stand_facing_catalog_path=facing_path, stand_facing_catalog_sha256=summary["stand_facing_catalog_sha256"],
        facing_records=tuple(facing["records"]), expected_stand_count=summary["expected_stand_count"],
        candidate_pool_count=summary["candidate_pool_count"],
        confirmed_candidate_snapshot_path=_path(summary, "confirmed_candidate_snapshot"),
        confirmed_candidate_snapshot_sha256=summary["confirmed_candidate_snapshot_sha256"],
        candidate_goal_progress_path=_path(summary, "candidate_goal_progress"),
        candidate_goal_progress_sha256=summary["candidate_goal_progress_sha256"],
        observed_identities_path=_path(summary, "observed_station_identities"),
        observed_identities_sha256=summary["observed_station_identities_sha256"],
        identity_binding_status=summary.get("identity_binding_status", "server_binding_pending"),
        qr_observation_catalog_path=qr_path, qr_observation_catalog_sha256=summary["qr_observation_pose_catalog_sha256"],
        qr_observation_records=tuple(qr["records"]),
    )
    authorization_path = root / "motion_authorization" / "mission_leg_motion_authorization.json"
    source_authorization = load_mission_leg_motion_authorization(authorization_path)
    runtime = profile.resolved_runtime()
    if (
        source_authorization.session_id != summary.get("session_id")
        or source_authorization.robot_id != profile.robot_id
        or source_authorization.namespace != runtime.namespace
        or source_authorization.cmd_vel_topic != runtime.cmd_vel_topic
        or source_authorization.semantic_map_id != site.site.map_measurement.semantic_map_id
    ):
        raise ValueError("stored session authorization identity differs from robot/site")
    stand_model = load_measured_physical_stand_model(_path(summary, "stand_model_profile"))
    if stand_model.sha256 != summary.get("stand_model_profile_sha256"):
        raise ValueError("stored session stand model hash mismatch")
    clearance = _physical_clearance(profile, approach_offset_m=.50, stand_model_profile=stand_model)
    config = CandidateApproachConfig(
        session_root=root, survey_root=survey_root, session_id=source_authorization.session_id,
        semantic_map_id=source_authorization.semantic_map_id, planning_frame=profile.map_frame,
        map_yaml=site.map_yaml_path, plan=plan, snapshot=snapshot, snapshot_path=snapshot_path,
        approach_offset_m=.50, inflation_radius_m=max(plan.config.inflation_radius_m, clearance["minimum_static_inflation_m"]),
        candidate_transit_radius_m=max(plan.config.candidate_keepout_radius_m, clearance["minimum_candidate_transit_radius_m"]),
        physical_clearance=clearance, uncertainty_sigma_multiplier=DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER,
        localization_branch_proof_id=source_authorization.localization_branch_proof_id,
        mission_leg_motion_authorization_json=authorization_path,
        expected_stand_count=completed.expected_stand_count, robot_radius_m=profile.robot_radius_m,
        robot_profile_sha256=real_robot_profile_sha256(profile), calibration_profile_sha256=profile.calibration_profile_sha256,
        require_uncertainty_aware_selection=True,
        exact_two_camera_handoff_path=(None if not summary.get("exact_two_camera_handoff") else _path(summary, "exact_two_camera_handoff")),
        exact_two_camera_handoff_sha256=summary.get("exact_two_camera_handoff_sha256"),
    )
    poses = load_stored_admitted_poses(completed, config)
    if len(poses) != completed.stand_count or len(snapshot.candidates) != completed.candidate_pool_count:
        raise ValueError("stored session completion counts disagree with authenticated poses/pool")
    session_sources = [{"path": str(Path(path).resolve()), "sha256": file_sha256(Path(path))}
        for path in (root / "mission_summary.json", robot_profile_path, site_path,
                     survey_root / "coverage_plan.json", _path(summary, "stand_model_profile"), authorization_path)]
    poses = {qr_id: replace(stored, evidence={**stored.evidence,
        "source_session_id": config.session_id,
        "source_artifacts": [*stored.evidence["source_artifacts"], *session_sources]})
        for qr_id, stored in poses.items()}
    return StoredPoseSession(completed, config, MappingProxyType(poses), profile)
