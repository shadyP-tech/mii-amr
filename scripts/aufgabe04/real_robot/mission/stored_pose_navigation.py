"""Fresh, bounded navigation to an authenticated stored pose; no server effects."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Callable

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame, project_candidate_snapshot_to_planning_frame
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import CandidateRouteUncertaintyContext
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind, MAX_RETURN_TO_START_LEGS
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import map_pose_to_odom, odom_pose_to_map
from scripts.aufgabe04.real_robot.candidate.approach import CandidateApproachConfig, CandidateMotionLegRequest
from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import CandidateRouteUncertaintyReadinessRequest
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome
from scripts.aufgabe04.real_robot.mission.start_return_readiness import load_stored_pose_tour_readiness
from scripts.aufgabe04.real_robot.mission.stored_start_pose import StoredAdmittedPose
from scripts.aufgabe04.stations.candidate_snapshot import write_candidate_snapshot


def _plan_route(**kwargs):
    from scripts.aufgabe04.navigation.approach.admitted_pose_route import plan_admitted_pose_route
    return plan_admitted_pose_route(**kwargs)


@dataclass(frozen=True)
class StoredPoseNavigationEffects:
    admit_planning_frame: Callable[[Path], CandidatePlanningFrame]
    run_motion_leg: Callable[[CandidateMotionLegRequest], MotionLegOutcome]
    plan_route: Callable[..., dict[str, object]] = _plan_route
    load_route_uncertainty_readiness: Callable[[CandidateRouteUncertaintyReadinessRequest], CandidateRouteUncertaintyContext] = load_stored_pose_tour_readiness


def target_in_frame(stored: StoredAdmittedPose, frame: CandidatePlanningFrame) -> Pose2D:
    if (frame.map_frame, frame.odom_frame) != (stored.source_frame.map_frame, stored.source_frame.odom_frame):
        raise ValueError("stored pose localization frame identities changed")
    return odom_pose_to_map(map_pose_to_odom(stored.pose, stored.source_frame.map_from_odom), frame.map_from_odom)


def arrival_errors(pose: Pose2D, target: Pose2D) -> tuple[float, float]:
    return (math.hypot(pose.x_m - target.x_m, pose.y_m - target.y_m),
            abs(math.remainder(pose.yaw_rad - target.yaw_rad, math.tau)))


def projected_target_evidence(stored, config, frame, root):
    target = target_in_frame(stored, frame)
    projection = project_candidate_snapshot_to_planning_frame(config.snapshot, stored.registry, frame)
    snapshot_path = root / "candidate_snapshot.json"
    snapshot_sha = write_candidate_snapshot(snapshot_path, projection.projected_snapshot)
    projection_path = root / "candidate_frame_projection.json"
    projection_sha = write_content_hashed_json(projection_path, {
        **projection.to_evidence(), "source_candidate_snapshot_path": str(config.snapshot_path),
        "projected_candidate_snapshot_path": str(snapshot_path),
    }, hash_field="candidate_frame_projection_sha256")
    evidence = {**stored.evidence, "target_pose": asdict(target), "planning_frame": config.planning_frame,
        "planning_frame_admission": frame.to_evidence(), "candidate_snapshot_sha256": snapshot_sha,
        "candidate_frame_projection_path": str(projection_path), "candidate_frame_projection_sha256": projection_sha}
    measured = stored.evidence.get("stored_measured_target_center")
    if measured is not None:
        point = odom_pose_to_map(map_pose_to_odom(
            Pose2D(measured["x_m"], measured["y_m"]), stored.source_frame.map_from_odom), frame.map_from_odom)
        evidence["measured_target_center"] = {
            "x_m": point.x_m, "y_m": point.y_m, "uncertainty_m": measured["uncertainty_m"]}
    return target, projection.projected_snapshot, snapshot_path, evidence


def execute_stored_pose_navigation(
    stored: StoredAdmittedPose, config: CandidateApproachConfig, effects: StoredPoseNavigationEffects,
    *, tour_session_id: str, visit_index: int, output_root: Path,
) -> dict[str, object]:
    """Visit one genuine QR identity under a new independent tour scope."""
    if (not isinstance(tour_session_id, str) or ".." in tour_session_id
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", tour_session_id)):
        raise ValueError("stored pose tour requires a safe tour session ID")
    if type(visit_index) is not int or visit_index < 0:
        raise ValueError("stored pose tour requires a nonnegative visit index")
    return execute_stored_pose_stages(lambda: stored, config, effects,
        output_root=output_root, session_id=tour_session_id,
        motion_root=Path(output_root), purpose="stored_pose_tour", visit_index=visit_index)


def execute_stored_pose_stages(
    load_target, config, effects, *, output_root: Path, session_id: str,
    motion_root: Path, purpose: str, visit_index: int = 0, report=lambda value: value,
    write_artifact=write_content_hashed_json,
) -> dict[str, object]:
    """Shared Start/tour stage engine. Reports are adapted at the boundary."""
    if purpose not in {"return_to_start", "stored_pose_tour"}:
        raise ValueError("unsupported stored pose navigation purpose")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    is_start = purpose == "return_to_start"
    summary = {"status": "failed_closed", "target_pose_reached": False, "arrival_verified": False,
        "motion_authorized": False, "motion_published": False, "leg_count": 0, "legs": []}
    if not is_start:
        summary.update(tour_id=session_id, visit_index=visit_index)
    try:
        stored = load_target()
        qr_id = stored.evidence["qr_id"]
        if not isinstance(qr_id, str) or not qr_id or (is_start and qr_id != "Start"):
            raise ValueError("stored target QR identity differs from navigation purpose")
        # Recheck source bytes even for an already-arrived visit, which has no
        # planner/child stage to repeat these checks before reporting arrival.
        for source in stored.evidence["source_artifacts"]:
            if file_sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError("stored target source artifact hash mismatch")
        localization_path = root / "planning_localization.json"
        frame = effects.admit_planning_frame(localization_path)
        target = target_in_frame(stored, frame)
        distance, heading = arrival_errors(frame.current_pose, target)
        already_arrived = distance <= .08 and heading <= .15
        evidence = stored.evidence
        summary.update(candidate_uid=stored.candidate_uid, qr_id=qr_id,
            pose_kind=evidence["pose_kind"], target_pose=asdict(target), planning_frame=frame.map_frame)
        for stage_index in range(MAX_RETURN_TO_START_LEGS):
            if distance <= .08 and heading <= .15:
                break
            radius = config.robot_radius_m
            if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0.:
                raise ValueError("stored navigation requires a finite positive robot radius")
            leg_root = root / "legs" / f"{stage_index:03d}"
            leg_root.mkdir(parents=True, exist_ok=False)
            target, snapshot, snapshot_path, evidence = projected_target_evidence(stored, config, frame, leg_root)
            if not is_start:
                evidence.update(tour_id=session_id, visit_index=visit_index)
            uncertainty = effects.load_route_uncertainty_readiness(CandidateRouteUncertaintyReadinessRequest(
                preflight_json=localization_path, expected_start=frame.current_pose,
                planning_frame=frame.map_frame, odom_frame=frame.odom_frame,
                robot_radius_m=float(radius), sigma_multiplier=config.uncertainty_sigma_multiplier))
            if not isinstance(uncertainty, CandidateRouteUncertaintyContext):
                raise ValueError("stored navigation requires stopped route uncertainty evidence")
            sealed = effects.plan_route(
                map_yaml=config.map_yaml, semantic_map_id=config.semantic_map_id, plan=config.plan,
                snapshot=snapshot, snapshot_path=snapshot_path, candidate_uid=stored.candidate_uid,
                start=frame.current_pose, target=target, output_dir=leg_root / "route",
                inflation_radius_m=config.inflation_radius_m, physical_clearance=config.physical_clearance,
                target_evidence=evidence, route_uncertainty_context=uncertainty, return_stage_index=stage_index,
                **({} if is_start else {"purpose": purpose}))
            final_stage = sealed["is_final_stage"]
            if not isinstance(final_stage, bool):
                raise ValueError("stored navigation stage must declare final-stage status")
            stage_target = Pose2D(**sealed["stage_target_pose"])
            if not all(not isinstance(v, bool) and math.isfinite(v) for v in asdict(stage_target).values()):
                raise ValueError("stored navigation stage target must be finite")
            target_distance, target_heading = arrival_errors(stage_target, target)
            if final_stage and (target_distance > 1e-9 or target_heading > 1e-9):
                raise ValueError("final stored navigation stage changed the stored target")
            if not final_stage:
                if arrival_errors(frame.current_pose, stage_target)[0] < .20 - 1e-9:
                    raise ValueError("intermediate stored navigation stage makes insufficient progress")
                if stage_index + 1 == MAX_RETURN_TO_START_LEGS:
                    raise RuntimeError("stored navigation leg budget exhausted before final approach")
            run_id = (f"{session_id}_return_to_start_{stage_index:03d}" if is_start
                      else f"{session_id}_visit_{visit_index:03d}_stage_{stage_index:03d}")
            outcome = effects.run_motion_leg(CandidateMotionLegRequest(
                sealed={key: value for key, value in sealed.items()
                        if key in {"route_csv", "diagnostics_json", "route_certificate_json"}},
                run_id=run_id, session_root=motion_root, candidate_snapshot_path=Path(sealed["candidate_snapshot"]),
                uncertainty_map_yaml=config.map_yaml, uncertainty_sigma_multiplier=config.uncertainty_sigma_multiplier,
                localization_branch_proof_id=config.localization_branch_proof_id,
                mission_authorization_json=config.mission_leg_motion_authorization_json,
                session_id=session_id, semantic_map_id=config.semantic_map_id,
                mission_leg_kind=MissionLegKind.RETURN_TO_START if is_start else MissionLegKind.STORED_POSE_TOUR,
                mission_leg_index=stage_index if is_start else visit_index * MAX_RETURN_TO_START_LEGS + stage_index,
                target_id=stored.candidate_uid, permit_json_path=leg_root / "motion_permit.json"))
            summary.update(run_id=run_id, leg_count=stage_index + 1,
                motion_published=summary["motion_published"] or outcome.motion_published)
            leg_evidence = {"stage_index": stage_index, "run_id": run_id, "final_stage": final_stage,
                "stage_target_pose": asdict(stage_target), "planning_frame": frame.to_evidence(),
                "planning_localization_json": str(localization_path), "status": outcome.status,
                "returncode": outcome.returncode, "motion_published": outcome.motion_published, "arrival_verified": False}
            summary["legs"].append(leg_evidence)
            if outcome.run_id != run_id or outcome.status != "completed" or outcome.returncode != 0:
                raise RuntimeError(f"stored navigation motion failed: {outcome.status}: {outcome.stop_reason}")
            previous_frame = frame
            localization_path = leg_root / "arrival_localization.json"
            frame = effects.admit_planning_frame(localization_path)
            if (frame.map_frame, frame.odom_frame) != (previous_frame.map_frame, previous_frame.odom_frame):
                raise ValueError("stored navigation localization frame identities changed")
            arrival_target = odom_pose_to_map(map_pose_to_odom(stage_target, previous_frame.map_from_odom), frame.map_from_odom)
            stage_distance, stage_heading = arrival_errors(frame.current_pose, arrival_target)
            if stage_distance > .08 or stage_heading > .15:
                raise RuntimeError("stored navigation stopped outside the admitted pose arrival tolerance")
            progress = arrival_errors(map_pose_to_odom(previous_frame.current_pose, previous_frame.map_from_odom),
                                      map_pose_to_odom(frame.current_pose, frame.map_from_odom))[0]
            if not final_stage and progress < .10:
                raise RuntimeError("stored navigation intermediate stop made no measured odom progress")
            leg_evidence.update(arrival_verified=True, arrival_planning_frame=frame.to_evidence(),
                arrival_localization_json=str(localization_path), arrival_target_pose=asdict(arrival_target),
                position_error_m=stage_distance, heading_error_rad=stage_heading, odom_progress_m=progress)
            write_artifact(leg_root / "arrival.json", {
                "schema_version": 1, "artifact_kind": "start_return_leg_arrival" if is_start else "stored_pose_tour_leg_arrival",
                "session_id": session_id, "candidate_uid": stored.candidate_uid, "qr_id": qr_id,
                "fastapi_request_ready": False, "motion_authorized": False, **leg_evidence,
                **({} if is_start else {"tour_id": session_id, "visit_index": visit_index}),
            }, hash_field="start_return_leg_arrival_sha256" if is_start else "stored_pose_tour_leg_arrival_sha256")
            target = target_in_frame(stored, frame)
            distance, heading = arrival_errors(frame.current_pose, target)
            if final_stage and (distance > .08 or heading > .15):
                raise RuntimeError("stored navigation final arrival differs from stored target pose")
        if distance > .08 or heading > .15:
            raise RuntimeError("stored navigation leg budget exhausted before reaching target")
        summary.update(status="already_at_target" if already_arrived else "completed", target_pose_reached=True,
            arrival_verified=True, arrival_pose=asdict(frame.current_pose), arrival_target_pose=asdict(target),
            arrival_position_error_m=distance, arrival_heading_error_rad=heading)
        write_artifact(root / "arrival.json", {
            "schema_version": 1, "artifact_kind": "admitted_start_arrival" if is_start else "stored_pose_tour_arrival",
            "session_id": session_id, **report(summary), "target_evidence": evidence,
            "arrival_planning_frame": frame.to_evidence(),
        }, hash_field="start_arrival_sha256" if is_start else "stored_pose_tour_arrival_sha256")
        return report(summary)
    except (AssertionError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        summary.update(error=str(exc), status="failed_closed", target_pose_reached=False, arrival_verified=False)
        write_artifact(root / "failure.json", {
            "schema_version": 1, "artifact_kind": "start_return_failure" if is_start else "stored_pose_tour_failure",
            "session_id": session_id, **report(summary),
        }, hash_field="start_return_failure_sha256" if is_start else "stored_pose_tour_failure_sha256")
        raise
