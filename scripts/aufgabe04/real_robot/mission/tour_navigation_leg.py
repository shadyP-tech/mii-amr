"""Planning and measured arrival for one immutable tour execution."""

from dataclasses import asdict
import math
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import CandidateRouteUncertaintyContext
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind, MAX_RETURN_TO_START_LEGS
from scripts.aufgabe04.navigation.execution.tour_replan_binding import tour_mission_leg_index
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import map_pose_to_odom, odom_pose_to_map
from scripts.aufgabe04.real_robot.candidate.approach import CandidateMotionLegRequest
from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import CandidateRouteUncertaintyReadinessRequest
from scripts.aufgabe04.real_robot.mission.stored_pose_navigation import arrival_errors, projected_target_evidence


def plan_tour_leg(stored, config, effects, *, frame, localization_path, overlay_path,
                  leg_root, visit_root, tour_id, visit_index, navigation):
    """Seal a fresh route and issue a request without publishing velocity."""
    radius = config.robot_radius_m
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0.:
        raise ValueError("stored navigation requires a finite positive robot radius")
    target, snapshot, snapshot_path, evidence = projected_target_evidence(stored, config, frame, leg_root)
    evidence.update(tour_id=tour_id, visit_index=visit_index, tour_navigation=navigation)
    uncertainty = effects.load_route_uncertainty_readiness(CandidateRouteUncertaintyReadinessRequest(
        preflight_json=localization_path, expected_start=frame.current_pose,
        planning_frame=frame.map_frame, odom_frame=frame.odom_frame,
        robot_radius_m=float(radius), sigma_multiplier=config.uncertainty_sigma_multiplier))
    if not isinstance(uncertainty, CandidateRouteUncertaintyContext):
        raise ValueError("stored navigation requires stopped route uncertainty evidence")
    stage_index = navigation["stage_index"]
    sealed = effects.plan_route(
        map_yaml=config.map_yaml, semantic_map_id=config.semantic_map_id, plan=config.plan,
        snapshot=snapshot, snapshot_path=snapshot_path, candidate_uid=stored.candidate_uid,
        start=frame.current_pose, target=target, output_dir=leg_root / "route",
        inflation_radius_m=config.inflation_radius_m, physical_clearance=config.physical_clearance,
        target_evidence=evidence, route_uncertainty_context=uncertainty,
        return_stage_index=stage_index, purpose="stored_pose_tour",
        temporary_obstacle_overlay_path=overlay_path)
    final = sealed["is_final_stage"]
    if not isinstance(final, bool):
        raise ValueError("stored navigation stage must declare final-stage status")
    stage_target = Pose2D(**sealed["stage_target_pose"])
    if not all(not isinstance(v, bool) and math.isfinite(v) for v in asdict(stage_target).values()):
        raise ValueError("stored navigation stage target must be finite")
    if final and any(error > 1e-9 for error in arrival_errors(stage_target, target)):
        raise ValueError("final stored navigation stage changed the stored target")
    if not final:
        if arrival_errors(frame.current_pose, stage_target)[0] < .20 - 1e-9:
            raise ValueError("intermediate stored navigation stage makes insufficient progress")
        if stage_index + 1 == MAX_RETURN_TO_START_LEGS:
            raise RuntimeError("stored navigation stage budget exhausted before final approach")
    request = CandidateMotionLegRequest(
        sealed={key: sealed[key] for key in ("route_csv", "diagnostics_json", "route_certificate_json")},
        run_id=f"{tour_id}_visit_{visit_index:03d}_execution_{navigation['execution_index']:03d}",
        session_root=visit_root, candidate_snapshot_path=Path(sealed["candidate_snapshot"]),
        uncertainty_map_yaml=config.map_yaml, uncertainty_sigma_multiplier=config.uncertainty_sigma_multiplier,
        localization_branch_proof_id=config.localization_branch_proof_id,
        mission_authorization_json=config.mission_leg_motion_authorization_json,
        session_id=tour_id, semantic_map_id=config.semantic_map_id,
        mission_leg_kind=MissionLegKind.STORED_POSE_TOUR,
        mission_leg_index=tour_mission_leg_index(visit_index, navigation["execution_index"]),
        target_id=stored.candidate_uid, permit_json_path=leg_root / "motion_permit.json")
    return request, stage_target, final, evidence


def verify_tour_leg_arrival(stored, effects, *, request, previous_frame, stage_target,
                            final_stage, leg_root, visit_index, leg_evidence):
    """Measure in stable odom, then project the exact stage into fresh map TF."""
    localization_path = leg_root / "arrival_localization.json"
    frame = effects.admit_planning_frame(localization_path)
    if (frame.map_frame, frame.odom_frame) != (previous_frame.map_frame, previous_frame.odom_frame):
        raise ValueError("stored navigation localization frame identities changed")
    target = odom_pose_to_map(map_pose_to_odom(stage_target, previous_frame.map_from_odom), frame.map_from_odom)
    distance, heading = arrival_errors(frame.current_pose, target)
    if distance > .08 or heading > .15:
        raise RuntimeError("stored navigation stopped outside the admitted pose arrival tolerance")
    progress = arrival_errors(map_pose_to_odom(previous_frame.current_pose, previous_frame.map_from_odom),
                              map_pose_to_odom(frame.current_pose, frame.map_from_odom))[0]
    if not final_stage and progress < .10:
        raise RuntimeError("stored navigation intermediate stop made no measured odom progress")
    leg_evidence.update(arrival_verified=True, arrival_planning_frame=frame.to_evidence(),
        arrival_localization_json=str(localization_path), arrival_target_pose=asdict(target),
        position_error_m=distance, heading_error_rad=heading, odom_progress_m=progress)
    arrival_path = leg_root / "arrival.json"
    write_content_hashed_json(arrival_path, {
        "schema_version": 1, "artifact_kind": "stored_pose_tour_leg_arrival",
        "session_id": request.session_id, "tour_id": request.session_id, "visit_index": visit_index,
        "candidate_uid": stored.candidate_uid, "qr_id": stored.evidence["qr_id"],
        "fastapi_request_ready": False, "motion_authorized": False, **leg_evidence,
    }, hash_field="stored_pose_tour_leg_arrival_sha256")
    return frame, arrival_path
