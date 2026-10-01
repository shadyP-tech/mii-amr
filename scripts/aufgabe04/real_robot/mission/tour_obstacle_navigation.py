"""Stopped obstacle replanning for a tour; no ROS, HTTP, or steering policy."""

from dataclasses import asdict
from pathlib import Path
import re

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.navigation.execution.tour_replan_binding import (
    MAX_TOUR_DETOUR_REPLANS, MAX_TOUR_EXECUTIONS_PER_VISIT,
    is_replannable_tour_stop, tour_mission_leg_index, write_tour_terminal_evidence,
)
from scripts.aufgabe04.real_robot.mission.stored_pose_navigation import arrival_errors, target_in_frame
from scripts.aufgabe04.real_robot.mission.tour_navigation_leg import plan_tour_leg, verify_tour_leg_arrival


def execute_tour_obstacle_navigation(
    stored, config, effects, *, obstacle_map, capture_scan,
    tour_session_id: str, visit_index: int, output_root: Path,
    verify_start_handoff: bool = False,
):
    """Reach the original stored pose with at most two stopped obstacle detours.

    ``obstacle_map`` is shared across visits in one tour. Its published snapshots
    are frozen per execution; observations, clearing and expiry occur only here,
    while stopped. Each attempt gets its own route, permit and consumption slot.
    ``verify_start_handoff`` accepts the camera runner's already-completed
    return only after fresh stopped pose verification; it never dispatches motion.
    """
    if (not isinstance(tour_session_id, str) or ".." in tour_session_id
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", tour_session_id)):
        raise ValueError("stored pose tour requires a safe tour session ID")
    tour_mission_leg_index(visit_index, 0)
    if type(verify_start_handoff) is not bool:
        raise ValueError("verify_start_handoff must be boolean")
    if verify_start_handoff and (visit_index != 0 or stored.evidence.get("qr_id") != "Start"):
        raise ValueError("Start handoff verification is restricted to initial Start")
    if obstacle_map.tour_id != tour_session_id or obstacle_map.map_bundle_sha256 != config.plan.map_bundle_sha256:
        raise ValueError("temporary occupancy belongs to a different tour or map")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    summary = {"status": "failed_closed", "tour_id": tour_session_id, "visit_index": visit_index,
        "target_pose_reached": False, "arrival_verified": False, "motion_authorized": False,
        "motion_published": False, "leg_count": 0, "replan_count": 0, "legs": []}
    if verify_start_handoff:
        summary["initial_start_policy"] = "verify_camera_return"
    try:
        qr_id = stored.evidence["qr_id"]
        if not isinstance(qr_id, str) or not qr_id:
            raise ValueError("stored target QR identity must be nonempty")
        for source in stored.evidence["source_artifacts"]:
            if file_sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError("stored target source artifact hash mismatch")
        summary.update(candidate_uid=stored.candidate_uid, qr_id=qr_id, pose_kind=stored.evidence["pose_kind"])
        stage_index = replans = 0
        predecessor = None
        previous_path = previous_sha = ""
        evidence = stored.evidence
        for execution in range(MAX_TOUR_EXECUTIONS_PER_VISIT):
            leg_root = root / "legs" / f"{execution:03d}"
            leg_root.mkdir(parents=True, exist_ok=False)
            localization_path = leg_root / "planning_localization.json"
            frame = effects.admit_planning_frame(localization_path)
            target = target_in_frame(stored, frame)
            distance, heading = arrival_errors(frame.current_pose, target)
            # A stopped failure cannot become a successful arrival without a new
            # admitted execution, even if localization now places it near target.
            if execution == 0 and distance <= .08 and heading <= .15:
                break
            if verify_start_handoff:
                raise RuntimeError(
                    "camera exploration Start handoff is not at the stored pose "
                    f"(position error {distance:.3f} m, heading error {heading:.3f} rad); "
                    "finish the automatic return before starting the tour, or use "
                    "--drive-to-start to authorize a new approach"
                )
            capture_path = Path(capture_scan(leg_root / "scan_capture.json"))
            obstacle_map.update_from_capture(capture_path)
            overlay_path = obstacle_map.write_projection(leg_root / "temporary_obstacles.json", frame)
            if predecessor is not None:
                previous_path, previous_sha = write_tour_terminal_evidence(
                    predecessor["path"], outcome=predecessor["outcome"], request=predecessor["request"],
                    stage_index=predecessor["stage_index"], replan_count=predecessor["replan_count"],
                    final_stage=predecessor["final_stage"], arrival_path=predecessor["arrival_path"],
                    scan_capture_path=capture_path if predecessor["arrival_path"] is None else None)
            navigation = {"execution_index": execution, "stage_index": stage_index, "replan_count": replans,
                "previous_terminal_json": str(previous_path), "previous_terminal_sha256": previous_sha}
            request, stage_target, final, evidence = plan_tour_leg(stored, config, effects,
                frame=frame, localization_path=localization_path, overlay_path=overlay_path,
                leg_root=leg_root, visit_root=root, tour_id=tour_session_id,
                visit_index=visit_index, navigation=navigation)
            outcome = effects.run_motion_leg(request)
            leg = {"stage_index": stage_index, "execution_index": execution, "replan_count": replans,
                "run_id": request.run_id, "final_stage": final, "stage_target_pose": asdict(stage_target),
                "planning_frame": frame.to_evidence(), "planning_localization_json": str(localization_path),
                "temporary_obstacle_overlay_json": str(overlay_path), "scan_capture_json": str(capture_path),
                "status": outcome.status, "stop_reason": outcome.stop_reason, "returncode": outcome.returncode,
                "motion_published": outcome.motion_published, "arrival_verified": False}
            summary["legs"].append(leg)
            summary.update(leg_count=execution+1, motion_published=summary["motion_published"] or outcome.motion_published)
            if outcome.run_id != request.run_id:
                raise ValueError("stored navigation child run identity mismatch")
            predecessor = {"path": leg_root / "terminal.json", "outcome": outcome, "request": request,
                "stage_index": stage_index, "replan_count": replans, "final_stage": final, "arrival_path": None}
            if outcome.status != "completed" or outcome.returncode != 0:
                if not is_replannable_tour_stop(outcome):
                    raise RuntimeError(f"stored navigation motion failed: {outcome.status}: {outcome.stop_reason}")
                if replans >= MAX_TOUR_DETOUR_REPLANS:
                    raise RuntimeError("stored navigation obstacle replan budget exhausted")
                replans += 1
                summary["replan_count"] = replans
                continue
            frame, arrival_path = verify_tour_leg_arrival(stored, effects, request=request,
                previous_frame=frame, stage_target=stage_target, final_stage=final,
                leg_root=leg_root, visit_index=visit_index, leg_evidence=leg)
            predecessor["arrival_path"] = arrival_path
            target = target_in_frame(stored, frame)
            distance, heading = arrival_errors(frame.current_pose, target)
            if final:
                if distance > .08 or heading > .15:
                    raise RuntimeError("stored navigation final arrival differs from stored target pose")
                write_tour_terminal_evidence(predecessor["path"], outcome=outcome, request=request,
                    stage_index=stage_index, replan_count=replans, final_stage=True, arrival_path=arrival_path)
                break
            stage_index += 1
        else:
            raise RuntimeError("stored navigation execution budget exhausted before reaching target")
        summary.update(status="already_at_target" if not summary["leg_count"] else "completed",
            target_pose_reached=True, arrival_verified=True, arrival_pose=asdict(frame.current_pose),
            arrival_target_pose=asdict(target), arrival_position_error_m=distance, arrival_heading_error_rad=heading)
        write_content_hashed_json(root / "arrival.json", {
            "schema_version": 1, "artifact_kind": "stored_pose_tour_arrival", "session_id": tour_session_id,
            **summary, "target_evidence": evidence, "arrival_planning_frame": frame.to_evidence(),
        }, hash_field="stored_pose_tour_arrival_sha256")
        return summary
    except (AssertionError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        summary.update(error=str(exc), status="failed_closed", target_pose_reached=False, arrival_verified=False)
        write_content_hashed_json(root / "failure.json", {
            "schema_version": 1, "artifact_kind": "stored_pose_tour_failure", "session_id": tour_session_id, **summary,
        }, hash_field="stored_pose_tour_failure_sha256")
        raise
