"""Return to the admitted Start pose after the camera artifacts are stored."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Callable

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidatePlanningFrame, project_candidate_snapshot_to_planning_frame,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind, MAX_RETURN_TO_START_LEGS
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import map_pose_to_odom, odom_pose_to_map
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachComplete, CandidateApproachConfig, CandidateMotionLegRequest,
)
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome
from scripts.aufgabe04.real_robot.mission.stored_start_pose import StoredStartPose, load_stored_start_pose
from scripts.aufgabe04.real_robot.mission.start_return_readiness import load_start_return_readiness
from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import CandidateRouteUncertaintyReadinessRequest
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import CandidateRouteUncertaintyContext
from scripts.aufgabe04.stations.candidate_snapshot import write_candidate_snapshot


def _plan_return(**kwargs):
    from scripts.aufgabe04.navigation.approach.admitted_pose_route import plan_admitted_pose_route
    return plan_admitted_pose_route(**kwargs)


@dataclass(frozen=True)
class StartReturnEffects:
    admit_planning_frame: Callable[[Path], CandidatePlanningFrame]
    run_motion_leg: Callable[[CandidateMotionLegRequest], MotionLegOutcome]
    load_target: Callable[[CandidateApproachComplete, CandidateApproachConfig], StoredStartPose] = load_stored_start_pose
    plan_route: Callable[..., dict[str, object]] = _plan_return
    load_route_uncertainty_readiness: Callable[
        [CandidateRouteUncertaintyReadinessRequest], CandidateRouteUncertaintyContext
    ] = load_start_return_readiness


def _target_in_frame(stored: StoredStartPose, frame: CandidatePlanningFrame) -> Pose2D:
    if (frame.map_frame, frame.odom_frame) != (stored.source_frame.map_frame, stored.source_frame.odom_frame):
        raise ValueError("Start return localization frame identities changed")
    return odom_pose_to_map(map_pose_to_odom(stored.pose, stored.source_frame.map_from_odom), frame.map_from_odom)


def _arrival_errors(pose: Pose2D, target: Pose2D) -> tuple[float, float]:
    return (math.hypot(pose.x_m - target.x_m, pose.y_m - target.y_m),
            abs(math.remainder(pose.yaw_rad - target.yaw_rad, math.tau)))


def _projected_target_evidence(stored, config, frame, root):
    """Project the original pool and exact Start target into this stopped epoch."""
    target = _target_in_frame(stored, frame)
    projection = project_candidate_snapshot_to_planning_frame(config.snapshot, stored.registry, frame)
    projected_path = root / "candidate_snapshot.json"
    projected_sha = write_candidate_snapshot(projected_path, projection.projected_snapshot)
    projection_path = root / "candidate_frame_projection.json"
    projection_sha = write_content_hashed_json(projection_path, {
        **projection.to_evidence(), "source_candidate_snapshot_path": str(config.snapshot_path),
        "projected_candidate_snapshot_path": str(projected_path),
    }, hash_field="candidate_frame_projection_sha256")
    evidence = {
        **stored.evidence, "target_pose": asdict(target), "planning_frame": config.planning_frame,
        "planning_frame_admission": frame.to_evidence(), "candidate_snapshot_sha256": projected_sha,
        "candidate_frame_projection_path": str(projection_path),
        "candidate_frame_projection_sha256": projection_sha,
    }
    measured = stored.evidence.get("stored_measured_target_center")
    if measured is not None:
        point = odom_pose_to_map(map_pose_to_odom(
            Pose2D(measured["x_m"], measured["y_m"]), stored.source_frame.map_from_odom,
        ), frame.map_from_odom)
        evidence["measured_target_center"] = {
            "x_m": point.x_m, "y_m": point.y_m, "uncertainty_m": measured["uncertainty_m"],
        }
    return target, projection.projected_snapshot, projected_path, evidence


def execute_start_return(
    completed: CandidateApproachComplete, config: CandidateApproachConfig, effects: StartReturnEffects,
) -> dict[str, object]:
    """Bounded independently admitted legs, then final arrival; no server effect."""
    root = config.session_root / "return_to_start"
    root.mkdir(parents=True, exist_ok=False)
    summary = {
        "return_to_start_status": "failed_closed", "start_pose_reached": False,
        "fastapi_request_ready": False, "fastapi_request_sent": False,
        "motion_authorized": False, "start_return_motion_published": False,
        "start_return_leg_count": 0, "start_return_legs": [],
    }
    try:
        stored = effects.load_target(completed, config)
        localization_path = root / "planning_localization.json"
        frame = effects.admit_planning_frame(localization_path)
        target = _target_in_frame(stored, frame)
        distance, heading = _arrival_errors(frame.current_pose, target)
        already_arrived = distance <= 0.08 and heading <= 0.15
        evidence = stored.evidence
        summary.update({
            "start_candidate_uid": stored.candidate_uid, "start_qr_id": "Start",
            "start_pose_kind": stored.evidence["pose_kind"], "start_target_pose": asdict(target),
            "start_target_planning_frame": frame.map_frame,
        })
        for stage_index in range(MAX_RETURN_TO_START_LEGS):
            if distance <= 0.08 and heading <= 0.15:
                break
            radius = config.robot_radius_m
            if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0.:
                raise ValueError("Start return requires a finite positive robot radius")
            leg_root = root / "legs" / f"{stage_index:03d}"
            leg_root.mkdir(parents=True, exist_ok=False)
            target, snapshot, snapshot_path, evidence = _projected_target_evidence(
                stored, config, frame, leg_root,
            )
            uncertainty = effects.load_route_uncertainty_readiness(CandidateRouteUncertaintyReadinessRequest(
                preflight_json=localization_path, expected_start=frame.current_pose,
                planning_frame=frame.map_frame, odom_frame=frame.odom_frame,
                robot_radius_m=float(radius), sigma_multiplier=config.uncertainty_sigma_multiplier,
            ))
            if not isinstance(uncertainty, CandidateRouteUncertaintyContext):
                raise ValueError("Start return requires stopped route uncertainty evidence")
            sealed = effects.plan_route(
                map_yaml=config.map_yaml, semantic_map_id=config.semantic_map_id, plan=config.plan,
                snapshot=snapshot, snapshot_path=snapshot_path,
                candidate_uid=stored.candidate_uid, start=frame.current_pose, target=target,
                output_dir=leg_root / "route", inflation_radius_m=config.inflation_radius_m,
                physical_clearance=config.physical_clearance, target_evidence=evidence,
                route_uncertainty_context=uncertainty, return_stage_index=stage_index,
            )
            final_stage = sealed["is_final_stage"]
            if not isinstance(final_stage, bool):
                raise ValueError("Start return stage must declare final-stage status")
            stage_target = Pose2D(**sealed["stage_target_pose"])
            if not all(math.isfinite(v) for v in asdict(stage_target).values()):
                raise ValueError("Start return stage target must be finite")
            target_distance, target_heading = _arrival_errors(stage_target, target)
            if final_stage and (target_distance > 1e-9 or target_heading > 1e-9):
                raise ValueError("final Start return stage changed the stored target")
            if not final_stage:
                if _arrival_errors(frame.current_pose, stage_target)[0] < 0.20 - 1e-9:
                    raise ValueError("intermediate Start return stage makes insufficient progress")
                if stage_index + 1 == MAX_RETURN_TO_START_LEGS:
                    raise RuntimeError("Start return leg budget exhausted before final approach")
            run_id = f"{config.session_id}_return_to_start_{stage_index:03d}"
            outcome = effects.run_motion_leg(CandidateMotionLegRequest(
                sealed={key: value for key, value in sealed.items() if isinstance(value, str)},
                run_id=run_id, session_root=config.session_root,
                candidate_snapshot_path=Path(sealed["candidate_snapshot"]),
                uncertainty_map_yaml=config.map_yaml, uncertainty_sigma_multiplier=config.uncertainty_sigma_multiplier,
                localization_branch_proof_id=config.localization_branch_proof_id,
                mission_authorization_json=config.mission_leg_motion_authorization_json,
                session_id=config.session_id, semantic_map_id=config.semantic_map_id,
                mission_leg_kind=MissionLegKind.RETURN_TO_START, mission_leg_index=stage_index,
                target_id=stored.candidate_uid, permit_json_path=leg_root / "motion_permit.json",
            ))
            summary.update({
                "start_return_run_id": run_id, "start_return_leg_count": stage_index + 1,
                "start_return_motion_published": summary["start_return_motion_published"] or outcome.motion_published,
            })
            leg_evidence = {
                "stage_index": stage_index, "run_id": run_id, "final_stage": final_stage,
                "stage_target_pose": asdict(stage_target), "planning_frame": frame.to_evidence(),
                "planning_localization_json": str(localization_path),
                "status": outcome.status, "returncode": outcome.returncode,
                "motion_published": outcome.motion_published, "arrival_verified": False,
            }
            summary["start_return_legs"].append(leg_evidence)
            if outcome.run_id != run_id or outcome.status != "completed" or outcome.returncode != 0:
                raise RuntimeError(f"Start return motion failed: {outcome.status}: {outcome.stop_reason}")
            previous_frame = frame
            localization_path = leg_root / "arrival_localization.json"
            frame = effects.admit_planning_frame(localization_path)
            if (frame.map_frame, frame.odom_frame) != (previous_frame.map_frame, previous_frame.odom_frame):
                raise ValueError("Start return localization frame identities changed")
            arrival_target = odom_pose_to_map(
                map_pose_to_odom(stage_target, previous_frame.map_from_odom), frame.map_from_odom,
            )
            stage_distance, stage_heading = _arrival_errors(frame.current_pose, arrival_target)
            if stage_distance > 0.08 or stage_heading > 0.15:
                raise RuntimeError("Start return stopped outside the admitted pose arrival tolerance")
            odom_before = map_pose_to_odom(previous_frame.current_pose, previous_frame.map_from_odom)
            odom_after = map_pose_to_odom(frame.current_pose, frame.map_from_odom)
            progress = _arrival_errors(odom_before, odom_after)[0]
            if not final_stage and progress < 0.10:
                raise RuntimeError("Start return intermediate stop made no measured odom progress")
            leg_evidence.update({
                "arrival_verified": True, "arrival_planning_frame": frame.to_evidence(),
                "arrival_localization_json": str(localization_path), "arrival_target_pose": asdict(arrival_target),
                "position_error_m": stage_distance, "heading_error_rad": stage_heading,
                "odom_progress_m": progress,
            })
            write_content_hashed_json(leg_root / "arrival.json", {
                "schema_version": 1, "artifact_kind": "start_return_leg_arrival",
                "session_id": config.session_id, "candidate_uid": stored.candidate_uid,
                "fastapi_request_ready": False, "motion_authorized": False, **leg_evidence,
            }, hash_field="start_return_leg_arrival_sha256")
            # No motion intervenes: this fresh arrival epoch is the next planning epoch.
            target = _target_in_frame(stored, frame)
            distance, heading = _arrival_errors(frame.current_pose, target)
            if final_stage and (distance > 0.08 or heading > 0.15):
                raise RuntimeError("Start return final arrival differs from stored Start pose")
        if distance > 0.08 or heading > 0.15:
            raise RuntimeError("Start return leg budget exhausted before reaching Start")
        summary.update({
            "return_to_start_status": "already_at_start" if already_arrived else "completed",
            "start_pose_reached": True, "fastapi_request_ready": True,
            "start_arrival_pose": asdict(frame.current_pose), "start_arrival_target_pose": asdict(target),
            "start_arrival_position_error_m": distance, "start_arrival_heading_error_rad": heading,
        })
        write_content_hashed_json(root / "arrival.json", {
            "schema_version": 1, "artifact_kind": "admitted_start_arrival", "session_id": config.session_id,
            **summary, "target_evidence": evidence, "arrival_planning_frame": frame.to_evidence(),
        }, hash_field="start_arrival_sha256")
        return summary
    except (AssertionError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        summary.update({
            "error": str(exc), "return_to_start_status": "failed_closed",
            "start_pose_reached": False, "fastapi_request_ready": False,
        })
        write_content_hashed_json(root / "failure.json", {
            "schema_version": 1, "artifact_kind": "start_return_failure", "session_id": config.session_id,
            **summary,
        }, hash_field="start_return_failure_sha256")
        raise
