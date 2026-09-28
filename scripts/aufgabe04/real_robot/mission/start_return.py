"""Compatibility boundary for the post-exploration return to exact Start."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import CandidateRouteUncertaintyContext
from scripts.aufgabe04.real_robot.candidate.approach import CandidateApproachComplete, CandidateApproachConfig, CandidateMotionLegRequest
from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import CandidateRouteUncertaintyReadinessRequest
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome
from scripts.aufgabe04.real_robot.mission.start_return_readiness import load_start_return_readiness
from scripts.aufgabe04.real_robot.mission.stored_start_pose import StoredStartPose, load_stored_start_pose
from scripts.aufgabe04.real_robot.mission.stored_pose_navigation import (
    _plan_route as _plan_return,
    arrival_errors as _arrival_errors,
    execute_stored_pose_stages,
    projected_target_evidence as _projected_target_evidence,
    target_in_frame as _target_in_frame,
)


@dataclass(frozen=True)
class StartReturnEffects:
    admit_planning_frame: Callable[[Path], CandidatePlanningFrame]
    run_motion_leg: Callable[[CandidateMotionLegRequest], MotionLegOutcome]
    load_target: Callable[[CandidateApproachComplete, CandidateApproachConfig], StoredStartPose] = load_stored_start_pose
    plan_route: Callable[..., dict[str, object]] = _plan_return
    load_route_uncertainty_readiness: Callable[[CandidateRouteUncertaintyReadinessRequest], CandidateRouteUncertaintyContext] = load_start_return_readiness


def _start_report(value):
    names = {
        "status": "return_to_start_status", "target_pose_reached": "start_pose_reached",
        "motion_published": "start_return_motion_published", "leg_count": "start_return_leg_count",
        "legs": "start_return_legs", "candidate_uid": "start_candidate_uid", "qr_id": "start_qr_id",
        "pose_kind": "start_pose_kind", "target_pose": "start_target_pose", "planning_frame": "start_target_planning_frame",
        "run_id": "start_return_run_id", "arrival_pose": "start_arrival_pose", "arrival_target_pose": "start_arrival_target_pose",
        "arrival_position_error_m": "start_arrival_position_error_m", "arrival_heading_error_rad": "start_arrival_heading_error_rad",
    }
    result = {names.get(key, key): item for key, item in value.items() if key != "arrival_verified"}
    if result["return_to_start_status"] == "already_at_target":
        result["return_to_start_status"] = "already_at_start"
    result.update(fastapi_request_ready=value["target_pose_reached"], fastapi_request_sent=False)
    return result


def execute_start_return(
    completed: CandidateApproachComplete, config: CandidateApproachConfig, effects: StartReturnEffects,
) -> dict[str, object]:
    """Keep the original Start-only identity, reports and output paths."""
    return execute_stored_pose_stages(lambda: effects.load_target(completed, config), config, effects,
        output_root=config.session_root / "return_to_start", session_id=config.session_id,
        motion_root=config.session_root, purpose="return_to_start", report=_start_report,
        write_artifact=write_content_hashed_json)
