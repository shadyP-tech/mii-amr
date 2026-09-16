"""Checked, offline promotion of autonomous camera evidence into a frozen catalog.

Every source is bound by content hash. Recommendations are transformed from
per-observation map/odom frames into one authenticated common frame, then the
existing fixed-target planner checks the exact target and terminal corridor
against the occupancy map and the complete LiDAR hypothesis pool. This creates
survey evidence only; neither a server ACK nor motion authority is implied.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from scripts.aufgabe04.artifacts import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION, SurveyManifest, artifact_reference, write_survey_manifest,
)
from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, payload_sha256, write_content_hashed_json
from scripts.aufgabe04.artifacts.bounded_orientation import validated_bounded_orientation
from scripts.aufgabe04.navigation.approach.camera_decision_geometry_binding import (
    require_camera_recommendation_binding, require_projected_camera_candidate_binding,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import (
    DynamicApproachConfig, FaceNormalCandidate, minimum_static_obstacle_inflation_m, plan_fixed_approach,
)
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    REAL_VIEWPOINT_SOURCE, load_recommendation, normalize_angle,
    recommendation_axis_estimator, recommendation_uses_current_head_front, validate_recommendation,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    coverage_survey_plan_sha256, load_coverage_survey_plan, load_stand_survey_registry, stand_survey_registry_sha256,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import map_pose_to_odom, odom_pose_to_map
from scripts.aufgabe04.navigation.missions.plan_synchronized_viewpoint import (
    _known_stand_keepout_costmap, _prepend_certified_known_stand_egress, _validate_known_stand_route_clearance,
)
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle, write_frozen_map_bundle
from scripts.aufgabe04.perception.arrival_pose_estimator import arrival_pose_record_from_recommendation
from scripts.aufgabe04.real_robot.configuration.profile import (
    camera_calibration_sha256, load_camera_calibration, load_real_robot_profile, real_robot_profile_sha256,
)
from scripts.aufgabe04.stations.arrival_pose_catalog import (
    arrival_pose_catalog_sha256, freeze_arrival_pose_catalog, new_arrival_pose_catalog, upsert_arrival_pose, write_arrival_pose_catalog,
)
from scripts.aufgabe04.stations.arrival_pose_models import CatalogProvenance
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256, load_candidate_snapshot, write_candidate_snapshot
from scripts.aufgabe04.stations.server_identity_binding import (
    bind_observed_station_identities, load_observed_identities, load_server_qr_mapping_evidence,
)
from scripts.aufgabe04.stations.station_identity_registry import (
    load_station_identity_registry, station_identity_registry_sha256, write_station_identity_registry,
)


@dataclass(frozen=True)
class AutonomousCatalogInputs:
    facing_catalog: Path
    candidate_snapshot: Path  # Immutable complete source pool.
    confirmed_candidate_snapshot: Path
    observed_identities: Path
    source_stand_registry: Path
    coverage_plan: Path
    target_frame_projection: Path
    map_yaml: Path
    semantic_map_id: str
    robot_profile: Path
    camera_calibration: Path
    physical_site: Path
    server_qr_mapping_evidence: Path
    server_robot_id: str
    output_dir: Path
    source_identity_registry: Path | None = None


def _file_sha(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"source artifact must not be a symlink: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(payload) -> CandidatePlanningFrame:
    return CandidatePlanningFrame.from_evidence(payload["planning_frame_admission"])


def _project_recommendation(recommendation, source_frame, target_frame):
    def pose(value):
        return odom_pose_to_map(map_pose_to_odom(value, source_frame.map_from_odom), target_frame.map_from_odom)
    rotation = target_frame.map_from_odom.yaw_rad - source_frame.map_from_odom.yaw_rad
    measurement = recommendation.axis_measurement
    if recommendation_uses_current_head_front(recommendation):
        measurement = deepcopy(measurement)
        # These two fields are map-frame angles. Pixel geometry and the
        # camera-relative measurement retain their original sensor evidence.
        for field in ("stand_axis_rad", "camera_heading_rad"):
            measurement[field] = normalize_angle(measurement[field] + rotation)
    projected = replace(
        recommendation,
        stand=replace(recommendation.stand, center=pose(recommendation.stand.center)),
        robot_pose=pose(recommendation.robot_pose),
        face_candidates=tuple(replace(face, pose=pose(face.pose), outward_normal_rad=normalize_angle(face.outward_normal_rad + rotation)) for face in recommendation.face_candidates),
        material_target=replace(recommendation.material_target, pose=pose(recommendation.material_target.pose)),
        bounded_orientation=(None if recommendation.bounded_orientation is None else
                             validated_bounded_orientation(recommendation.bounded_orientation).rotated(rotation).payload()),
        axis_measurement=measurement,
    )
    validate_recommendation(projected)
    return projected


def promote_autonomous_arrival_catalog(
    inputs: AutonomousCatalogInputs, *, now_sec: float, max_recommendation_age_sec: float = 300.0,
) -> dict[str, object]:
    """Validate everything before publishing the complete immutable bundle."""
    if not math.isfinite(now_sec) or now_sec < 0:
        raise ValueError("catalog validation time must be finite and nonnegative")
    if not math.isfinite(max_recommendation_age_sec) or not 0 < max_recommendation_age_sec <= 300.0:
        raise ValueError("recommendation maximum age must be in (0, 300] seconds")
    facing = load_content_hashed_json(inputs.facing_catalog, hash_field="stand_facing_catalog_sha256")
    if facing.get("schema_version") != 1 or facing.get("catalog_kind") != "real_autonomous_stand_facing_poses":
        raise ValueError("unsupported autonomous facing catalog")
    # QR discovery may complete before physical orientation is available. A
    # viewing pose can support revisiting a stand, but is not a validated facing
    # target for logistics. Legacy complete geometry catalogs omit these fields.
    if "facing_complete" in facing and facing["facing_complete"] is not True:
        raise ValueError("QR discovery catalog lacks complete geometry-backed facing poses")
    if "qr_only_stand_count" in facing and (
        type(facing["qr_only_stand_count"]) is not int or facing["qr_only_stand_count"] != 0
    ):
        raise ValueError("QR-only observation poses cannot be promoted as facing targets")
    if facing.get("qr_only_candidate_uids"):
        raise ValueError("QR-only observation poses cannot be promoted as facing targets")
    records_for_readiness = facing.get("records")
    if isinstance(records_for_readiness, list) and any(isinstance(record, dict) and (
        record.get("evidence_kind") == "qr_verified_observation_pose"
        or record.get("facing_ready") is False
    ) for record in records_for_readiness):
        raise ValueError("QR-only observation record cannot be promoted as a facing target")
    profile = load_real_robot_profile(inputs.robot_profile)
    calibration = load_camera_calibration(inputs.camera_calibration)
    calibration_sha = camera_calibration_sha256(calibration)
    if calibration_sha != profile.calibration_profile_sha256 or facing.get("calibration_profile_sha256") != calibration_sha:
        raise ValueError("facing evidence, robot profile, and calibration hashes differ")
    if facing.get("robot_profile_sha256") != real_robot_profile_sha256(profile):
        raise ValueError("facing evidence robot profile hash differs")
    if _file_sha(inputs.physical_site) != profile.physical_site_sha256 or inputs.physical_site.stem != profile.physical_site_id:
        raise ValueError("physical site differs from robot profile")
    grid, bundle = load_occupancy_grid_with_bundle(inputs.map_yaml, semantic_map_id=inputs.semantic_map_id, planning_frame=profile.map_frame)
    plan = load_coverage_survey_plan(inputs.coverage_plan)
    if facing.get("coverage_plan_sha256") != coverage_survey_plan_sha256(plan):
        raise ValueError("coverage plan hash differs from facing evidence")
    if facing.get("map_bundle_sha256") != bundle.bundle_sha256 or plan.map_bundle_sha256 != bundle.bundle_sha256:
        raise ValueError("source occupancy map bundle mismatch")
    if facing.get("planning_frame") != profile.map_frame or plan.planning_frame != profile.map_frame:
        raise ValueError("source planning frame mismatch")
    source = load_candidate_snapshot(inputs.candidate_snapshot, required_map_bundle_sha256=bundle.bundle_sha256)
    confirmed = load_candidate_snapshot(inputs.confirmed_candidate_snapshot, required_map_bundle_sha256=bundle.bundle_sha256)
    if facing.get("candidate_snapshot_sha256") != candidate_snapshot_sha256(source) or facing.get("confirmed_candidate_snapshot_sha256") != candidate_snapshot_sha256(confirmed):
        raise ValueError("source or confirmed candidate snapshot hash mismatch")
    if confirmed != replace(source, candidates=tuple(candidate for candidate in source.candidates if candidate.candidate_uid in confirmed.candidate_uids)):
        raise ValueError("confirmed snapshot is not an unchanged subset of the full source pool")
    observations = load_observed_identities(inputs.observed_identities, candidate_snapshot=confirmed)
    if payload_sha256(observations) != facing.get("observed_station_identities_sha256"):
        raise ValueError("observed identity artifact hash differs from facing catalog")
    if observations["session_id"] != facing.get("session_id") or observations["observed_unix_sec"] > now_sec:
        raise ValueError("observation session or timestamp mismatch")
    records = facing.get("records")
    if not isinstance(records, list) or len(records) != len(confirmed.candidates) or {record.get("candidate_uid") for record in records} != set(confirmed.candidate_uids):
        raise ValueError("facing records must resolve exactly the confirmed candidate subset")
    if facing.get("stand_count") != len(records) or facing.get("expected_stand_count") != len(records) or facing.get("candidate_pool_count") != len(source.candidates):
        raise ValueError("facing catalog completion counts disagree")
    evidence = load_server_qr_mapping_evidence(inputs.server_qr_mapping_evidence, robot_id=inputs.server_robot_id, now_sec=now_sec)
    source_identity_sha = facing.get("station_identity_registry_sha256")
    if source_identity_sha:
        if inputs.source_identity_registry is None:
            raise ValueError("bound facing catalog requires its source identity registry")
        source_identity = load_station_identity_registry(inputs.source_identity_registry, candidate_snapshot=confirmed)
        if station_identity_registry_sha256(source_identity) != source_identity_sha:
            raise ValueError("source identity registry hash mismatch")
        rebound = bind_observed_station_identities(candidate_snapshot=confirmed, observed_qr_by_candidate=observations["observed_qr_by_candidate"], mapping_evidence=evidence, registry_id=source_identity.registry_id, now_sec=now_sec)
        if source_identity.mappings != rebound.mappings:
            raise ValueError("source identity registry conflicts with authoritative server mappings")
    registry = load_stand_survey_registry(inputs.source_stand_registry, plan)
    registry_sha = stand_survey_registry_sha256(registry)
    if any(candidate.frame_provenance is None or candidate.frame_provenance.map_frame != profile.map_frame or candidate.frame_provenance.odom_frame != profile.odom_frame for candidate in registry.candidates):
        raise ValueError("source registry frame identities differ from robot profile")
    if any(candidate.source.source_artifact_sha256 != registry_sha for candidate in source.candidates):
        raise ValueError("source candidate ancestry differs from sealed stand registry")
    if facing.get("source_registry_sha256") != registry_sha:
        raise ValueError("source stand registry hash differs from facing evidence")

    def bound_projection(receipt, uid):
        projection_path = Path(receipt["candidate_frame_projection_path"])
        snapshot_path = Path(receipt["camera_candidate_snapshot_path"])
        candidate = require_projected_camera_candidate_binding(
            receipt, canonical_snapshot_path=inputs.candidate_snapshot, canonical_snapshot=source,
            registry=registry, source_registry_sha256=registry_sha,
            camera_snapshot_path=snapshot_path, projection_path=projection_path, candidate_uid=uid,
        )
        payload = load_content_hashed_json(projection_path, hash_field="candidate_frame_projection_sha256")
        frame = _frame(payload)
        if frame.map_frame != profile.map_frame or frame.odom_frame != profile.odom_frame:
            raise ValueError("source or target projection frame identities differ from robot profile and registry")
        return candidate, frame

    target_projection = load_content_hashed_json(inputs.target_frame_projection, hash_field="candidate_frame_projection_sha256")
    target_receipt = {
        "candidate_frame_projection_path": str(inputs.target_frame_projection),
        "candidate_frame_projection_sha256": payload_sha256(target_projection),
        "camera_candidate_snapshot_path": target_projection["projected_candidate_snapshot_path"],
        "camera_candidate_snapshot_sha256": target_projection["projected_candidate_snapshot_sha256"],
    }
    _, target_frame = bound_projection(target_receipt, confirmed.candidate_uids[0])
    full_projected = load_candidate_snapshot(Path(target_receipt["camera_candidate_snapshot_path"]))
    projected_confirmed = replace(full_projected, candidates=tuple(candidate for candidate in full_projected.candidates if candidate.candidate_uid in confirmed.candidate_uids))
    identity = bind_observed_station_identities(
        candidate_snapshot=projected_confirmed, observed_qr_by_candidate=observations["observed_qr_by_candidate"],
        mapping_evidence=evidence, registry_id=f"{facing['session_id']}_catalog_identities", now_sec=now_sec,
    )
    config = DynamicApproachConfig(robot_radius_m=profile.robot_radius_m, scan_origin_to_base_offset_m=profile.scan_origin_to_base_offset_m, tracking_margin_m=0.05)
    inflation = max(plan.config.inflation_radius_m, minimum_static_obstacle_inflation_m(
        robot_radius_m=config.robot_radius_m, tracking_margin_m=config.tracking_margin_m,
        lidar_stop_distance_m=config.lidar_stop_distance_m, scan_origin_to_base_offset_m=config.scan_origin_to_base_offset_m,
        lidar_clearance_margin_m=config.lidar_clearance_margin_m,
    ))
    base_costmap = Costmap.from_occupancy_grid(grid).with_arena_bounds(plan.arena_bounds).with_inflation(inflation)
    checked_records = []
    checks = []
    for record in records:
        uid = record["candidate_uid"]
        if record.get("qr_id") != observations["observed_qr_by_candidate"][uid]:
            raise ValueError("facing record QR differs from observed identity evidence")
        if record.get("calibration_profile_sha256") != calibration_sha or record.get("robot_profile_sha256") != real_robot_profile_sha256(profile):
            raise ValueError("facing record calibration or robot profile mismatch")
        source_candidate, source_frame = bound_projection(record, uid)
        receipt = {**record, "camera_evidence_path": record["recommendation_json"]}
        require_camera_recommendation_binding(receipt, candidate=source_candidate, planning_frame=profile.map_frame)
        recommendation = load_recommendation(Path(record["recommendation_json"]), expected_frame=profile.map_frame, expected_source=REAL_VIEWPOINT_SOURCE, expected_simulation_only=False, now_unix_sec=now_sec, max_age_sec=max_recommendation_age_sec)
        selected_face = next(face for face in recommendation.face_candidates if face.face_id == recommendation.material_target.face_id)
        side = recommendation.side_evidence
        current_head_front = recommendation_uses_current_head_front(recommendation)
        if current_head_front and recommendation.axis_measurement["qr_id"] != record["qr_id"]:
            raise ValueError("current-head QR identity differs from the observed facing record")
        admitted_qr_policy = (
            current_head_front and side.kind == "qr_observation"
            and side.provenance == "real/onboard_camera_qr_observation"
        ) or (
            not current_head_front and recommendation.axis_sample_count >= 7
            and side.kind == "qr_consensus"
            and side.provenance == "real/onboard_camera_qr_consensus"
        )
        if (not admitted_qr_policy or not side.hard or not side.valid
                or side.face_id != selected_face.face_id or not selected_face.identity_resolved
                or recommendation.material_target.evidence_state != "hard_qr"):
            raise ValueError("catalog promotion requires validated current-head or seven-frame committed onboard QR face evidence")
        sensor_age_sec = now_sec - recommendation.sensor_stamp_sec
        if sensor_age_sec < 0 or sensor_age_sec > max_recommendation_age_sec:
            raise ValueError("original recommendation sensor stamp is stale or in the future")
        if not math.isclose(recommendation.sensor_stamp_sec, recommendation.observation_unix_sec, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("real recommendation observation time must preserve its original sensor stamp")
        recommendation = _project_recommendation(recommendation, source_frame, target_frame)
        candidate = full_projected.candidate_for(uid)
        if math.hypot(recommendation.stand.center.x_m - candidate.geometry.x_m, recommendation.stand.center.y_m - candidate.geometry.y_m) > 1e-6:
            raise ValueError("recommendation common-frame geometry differs from full projected pool")
        converted = arrival_pose_record_from_recommendation(
            recommendation, candidate_uid=uid, map_yaml_sha256=bundle.yaml_sha256, corridor_length_m=config.terminal_corridor_length_m,
            validated_unix_sec=now_sec, axis_sample_count=recommendation.axis_sample_count,
            estimator=recommendation_axis_estimator(recommendation), source="real/autonomous_checked_catalog",
        )
        converted = replace(converted, stand_id=identity.for_candidate(uid).server_station_id)
        target_config = replace(config, stand_radius_m=candidate.geometry.radius_m, stand_position_uncertainty_m=candidate.geometry.uncertainty_m, standoff_distance_m=converted.standoff_m, minimum_non_target_keepout_radius_m=candidate.geometry.keepout_radius_m)
        clearance = record.get("active_stand_clearance", {})
        required_standoff = clearance.get("minimum_active_standoff_m")
        if not isinstance(required_standoff, (float, int)) or not math.isfinite(required_standoff) or required_standoff <= 0 or converted.standoff_m + 1e-9 < required_standoff:
            raise ValueError("catalog target violates original active stand clearance")
        keepouts = tuple((item.geometry.x_m, item.geometry.y_m, replace(target_config, stand_radius_m=item.geometry.radius_m, stand_position_uncertainty_m=item.geometry.uncertainty_m, minimum_non_target_keepout_radius_m=item.geometry.keepout_radius_m).non_target_stand_keepout_radius_m) for item in full_projected.candidates if item.candidate_uid != uid)
        overlay = _known_stand_keepout_costmap(base_costmap, keepouts, start=target_frame.current_pose)
        fixed = FaceNormalCandidate(0, converted.face.outward_normal_rad, Pose2D(**asdict(converted.arrival_pose)), Pose2D(**asdict(converted.corridor_entry_pose)))
        result = plan_fixed_approach(overlay.costmap, overlay.egress_anchor or target_frame.current_pose, recommendation.stand.center, fixed, config=target_config)
        if result.plan is None:
            raise ValueError(f"{uid}: fixed target/corridor validation failed: {result.diagnostics.failure_reason}")
        result = _prepend_certified_known_stand_egress(result, source_start=target_frame.current_pose, overlay=overlay, target_stand=recommendation.stand.center, target_keepout_radius_m=target_config.stand_keepout_radius_m)
        clearances = _validate_known_stand_route_clearance(result.plan, overlay.keepouts)
        checked_records.append(converted)
        checks.append({"candidate_uid": uid, "camera_recommendation_sha256": record["camera_recommendation_sha256"], "candidate_frame_projection_sha256": record["candidate_frame_projection_sha256"], "known_stand_clearances": clearances, "fixed_target_and_corridor_validated": True,
                       "admission_policy": "current_head_and_bound_qr" if current_head_front else "seven_frame_qr_consensus",
                       "axis_sample_count": recommendation.axis_sample_count})

    survey_config = {"schema_version": 1, "config_kind": "checked_autonomous_arrival_catalog", "arena_bounds": plan.arena_bounds.to_metadata(), "dynamic_approach_config": asdict(config), "inflation_radius_m": inflation, "axis_sample_count": 7, "max_recommendation_age_sec": max_recommendation_age_sec, "motion_authorized": False}
    if any(check["admission_policy"] == "current_head_and_bound_qr" for check in checks):
        # Mixed receipt policies retain their real counts, rather than claiming
        # that a single current fit supplied seven temporal measurements.
        survey_config.pop("axis_sample_count")
        survey_config["axis_sample_counts_by_candidate"] = {
            check["candidate_uid"]: check["axis_sample_count"] for check in checks
        }
        survey_config["admission_policies_by_candidate"] = {
            check["candidate_uid"]: check["admission_policy"] for check in checks
        }
    config_sha = payload_sha256(survey_config)
    binding = {
        "schema_version": 1, "binding_kind": "checked_autonomous_arrival_catalog", "session_id": facing["session_id"],
        "source_facing_catalog_sha256": payload_sha256(facing), "source_candidate_snapshot_sha256": candidate_snapshot_sha256(source),
        "source_registry_sha256": registry_sha, "coverage_plan_sha256": coverage_survey_plan_sha256(plan),
        "target_frame_projection_sha256": payload_sha256(target_projection),
        "candidate_snapshot_sha256": candidate_snapshot_sha256(projected_confirmed), "obstacle_candidate_snapshot_sha256": candidate_snapshot_sha256(full_projected),
        "station_identity_registry_sha256": station_identity_registry_sha256(identity), "server_qr_mapping_evidence_sha256": evidence.sha256,
        "observed_station_identities_sha256": payload_sha256(observations), "calibration_profile_sha256": calibration_sha,
        "robot_profile_sha256": real_robot_profile_sha256(profile), "map_bundle_sha256": bundle.bundle_sha256,
        "survey_config_sha256": config_sha, "validation_unix_sec": now_sec, "checks": checks, "motion_authorized": False,
    }
    provenance = CatalogProvenance(
        planning_frame=profile.map_frame, map_yaml_sha256=bundle.yaml_sha256, world_id=profile.physical_site_id, world_sha256=profile.physical_site_sha256,
        session_id=facing["session_id"], environment="real", map_bundle_sha256=bundle.bundle_sha256,
        candidate_snapshot_sha256=candidate_snapshot_sha256(projected_confirmed), station_identity_registry_sha256=station_identity_registry_sha256(identity),
        survey_config_sha256=config_sha, calibration_profile_sha256=calibration_sha, survey_input_binding_sha256=payload_sha256(binding),
        obstacle_candidate_snapshot_sha256=candidate_snapshot_sha256(full_projected),
    )
    catalog = new_arrival_pose_catalog(catalog_id=f"{facing['session_id']}_checked_arrivals", provenance=provenance, expected_candidate_uids=confirmed.candidate_uids, created_unix_sec=now_sec)
    for record in checked_records:
        catalog = upsert_arrival_pose(catalog, record, updated_unix_sec=now_sec)
    catalog = freeze_arrival_pose_catalog(catalog, frozen_unix_sec=now_sec)
    digest = arrival_pose_catalog_sha256(catalog)
    manifest = SurveyManifest(
        schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION, manifest_id=f"survey_{digest[:16]}", created_unix_sec=now_sec,
        session_id=facing["session_id"], environment="real", planning_frame=profile.map_frame,
        map_bundle=artifact_reference("map_bundle", bundle.semantic_map_id, bundle.bundle_sha256),
        candidate_snapshot=artifact_reference("candidate_snapshot", projected_confirmed.snapshot_id, candidate_snapshot_sha256(projected_confirmed)),
        environment_descriptor=artifact_reference("physical_site", profile.physical_site_id, profile.physical_site_sha256),
        survey_config=artifact_reference("survey_config", f"survey_config_{config_sha[:16]}", config_sha),
        calibration_profile=artifact_reference("calibration_profile", calibration.calibration_id, calibration_sha),
        arrival_pose_catalog=artifact_reference("arrival_pose_catalog", catalog.catalog_id, digest),
    )
    # The directory is a new bundle: never replace a previous catalog revision.
    inputs.output_dir.mkdir(parents=True, exist_ok=False)
    write_candidate_snapshot(inputs.output_dir / "candidate_snapshot.json", projected_confirmed)
    write_candidate_snapshot(inputs.output_dir / "obstacle_candidate_snapshot.json", full_projected)
    write_station_identity_registry(inputs.output_dir / "station_identity_registry.json", identity)
    write_frozen_map_bundle(inputs.output_dir / "map_bundle.json", bundle)
    write_content_hashed_json(inputs.output_dir / f"survey_config_{config_sha}.json", survey_config, hash_field="survey_config_sha256")
    write_content_hashed_json(inputs.output_dir / "survey_input_binding.json", binding, hash_field="survey_input_binding_sha256")
    write_arrival_pose_catalog(inputs.output_dir / "arrival_pose_catalog.json", catalog)
    write_survey_manifest(inputs.output_dir / "survey_manifest.json", manifest)
    return {"catalog_sha256": digest, "catalog_path": str(inputs.output_dir / "arrival_pose_catalog.json"), "stand_count": len(checked_records), "obstacle_count": len(full_projected.candidates), "frozen": True, "motion_authorized": False}
