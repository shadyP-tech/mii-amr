"""Bind local inspection policy to the existing candidate motion contracts.

The parent injects its planning-frame, route, recovery and arrival functions.
This module never publishes motion or bypasses a child preflight/permit.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

from scripts.aufgabe04.artifacts.candidate_inspection_observation import (
    load_candidate_inspection_observation,
)
from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, payload_sha256
from scripts.aufgabe04.navigation.approach.candidate_inspection_view import (
    write_candidate_inspection_view,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_models import (
    CandidatePreapproachUnreachableError,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.real_robot.candidate.inspection_execution import (
    CandidateInspectionEffects, CandidateInspectionRouteUnavailableError,
    execute_candidate_inspection,
)
from scripts.aufgabe04.real_robot.candidate.inspection_policy import novel_view
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)
from scripts.aufgabe04.real_robot.candidate.recovery_failure import CandidateStartupRecoveryError
from scripts.aufgabe04.real_robot.candidate.route_admission_deferral import (
    evaluate_candidate_route_admission_deferral,
)


def execute_local_candidate_inspection(
    *, observation_frame, source_config, effects, source_registry,
    candidate_root: Path, candidate_run_id: str, candidate_index: int,
    admit_arrival, admit_planning, move_certified_opposite, execute_motion,
    frame_type, request_type, observation_request_type,
):
    candidate_uid = observation_frame.candidate.candidate_uid
    seen_normals: list[float] = []
    motion_serial = 0

    def fresh_frame(root: Path):
        config, candidate, pose, planning, artifacts = admit_planning(
            source_config=source_config, effects=effects, source_registry=source_registry,
            candidate_uid=candidate_uid, candidate_root=root,
        )
        return frame_type(config, candidate, planning,
                          None if artifacts is None else artifacts.camera_decision_binding(), pose)

    def yaw(frame) -> float:
        return 0.0 if frame.planning_frame is None else frame.planning_frame.map_from_odom.yaw_rad

    def pose(frame):
        return (frame.observation_pose if frame.planning_frame is None
                else frame.planning_frame.current_pose)

    def normal(frame) -> float | None:
        current = pose(frame)
        if current is None:
            return None
        value = math.remainder(math.atan2(
            current.y_m - frame.candidate.geometry.y_m,
            current.x_m - frame.candidate.geometry.x_m,
        ) - yaw(frame), 2.0 * math.pi)
        if value not in seen_normals:
            seen_normals.append(value)
        return value

    def plan_and_move(frame, canonical_normal, root, index, source_path,
                      *, purpose="diverse_inspection", offset=None):
        nonlocal motion_serial
        frame = fresh_frame(root / "planning")
        current = pose(frame)
        if current is None:
            raise RuntimeError("inspection route lacks a fresh finite start pose")
        map_normal = math.remainder(canonical_normal + yaw(frame), 2.0 * math.pi)
        view_path = root / "inspection_view.json"
        write_candidate_inspection_view(
            view_path, snapshot=frame.config.snapshot, candidate_uid=candidate_uid,
            start=current, view_normal_rad=map_normal, purpose=purpose,
            view_index=index, source_observation_path=source_path,
        )
        request = request_type(
            map_yaml=frame.config.map_yaml, semantic_map_id=frame.config.semantic_map_id,
            plan=frame.config.plan, snapshot=frame.config.snapshot,
            snapshot_path=frame.config.snapshot_path, candidate_uid=candidate_uid,
            start=current, output_dir=root / "route",
            approach_offset_m=(source_config.approach_offset_m if offset is None else offset),
            inflation_radius_m=frame.config.inflation_radius_m,
            candidate_transit_radius_m=frame.config.candidate_transit_radius_m,
            physical_clearance=frame.config.physical_clearance,
            inspection_view_path=view_path,
        )
        try:
            sealed = effects.plan_preapproach(request)
        except ValueError as exc:
            # Static planning failure or the established prohibition on an
            # ambiguous zero-length route permits another useful proposal.
            text = str(exc)
            if not (isinstance(exc, CandidatePreapproachUnreachableError)
                    or text.startswith("candidate pre-approach A* failed")
                    or text == "target is blocked"
                    or text == "source route has fewer than two waypoints"):
                raise
            raise CandidateInspectionRouteUnavailableError(text) from exc
        summary_path = request.output_dir / "pipeline_summary.json"
        if summary_path.is_file():
            goal = json.loads(summary_path.read_text())["selected_approach_pose"]
            achieved = math.remainder(math.atan2(
                goal["y_m"] - frame.candidate.geometry.y_m,
                goal["x_m"] - frame.candidate.geometry.x_m,
            ) - yaw(frame), 2.0 * math.pi)
            if purpose == "diverse_inspection" and not novel_view(achieved, seen_normals):
                raise CandidateInspectionRouteUnavailableError(
                    "quantized inspection goal repeats an observed viewing direction"
                )
        elif frame.planning_frame is not None:
            raise RuntimeError("inspection route lacks materialized goal evidence")
        motion_serial += 1
        run_id = f"{candidate_run_id}_inspection_{motion_serial:03d}"
        try:
            execute_motion(
                config=frame.config, effects=effects, candidate_root=root,
                plan_request=request, initial_sealed=sealed, run_id=run_id,
                leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
                candidate_index=100000 + candidate_index * 1000 + motion_serial,
                target_id=candidate_uid, frame_source_config=source_config,
                source_registry=source_registry, plan_planning_frame=frame.planning_frame,
            )
        except CandidateStartupRecoveryError as exc:
            decision = evaluate_candidate_route_admission_deferral(
                exc, expected_initial_run_id=run_id,
            )
            if not decision.eligible:
                raise
            raise CandidateInspectionRouteUnavailableError(str(exc)) from exc
        return frame

    def admit(root, fallback_frame, index):
        result = admit_arrival(
            source_config=source_config, effects=effects, source_registry=source_registry,
            candidate_uid=candidate_uid, candidate_root=root, observation_attempt_index=0,
        )
        if result.planning_frame is None and result.observation_pose is None:
            result = replace(result, observation_pose=pose(fallback_frame))
        return result

    def admit_corrected(root, fallback_frame, index):
        """Correct a bearing-only miss before spending an observer slot."""
        error = None
        for correction in range(3):
            arrival_root = root if correction == 0 else root / f"alignment_{correction:02d}" / "arrival"
            try:
                return admit(arrival_root, fallback_frame, index)
            except CandidateObservationUnavailableError as exc:
                error = exc
                reasons = exc.status_evidence.get("reasons")
                if reasons != ["bearing_error_above_maximum"] or correction == 2:
                    raise
                evidence = exc.status_evidence
                robot = evidence["robot_pose"]
                target = evidence["target"]
                bearing = math.atan2(robot["y_m"] - target["y_m"],
                                     robot["x_m"] - target["x_m"])
                # Source arrival frame is admitted afresh, so recover its yaw
                # through the same frozen-frame projection evidence.
                projection = load_content_hashed_json(
                    Path(evidence["candidate_frame_projection_path"]),
                    hash_field="candidate_frame_projection_sha256",
                )
                if payload_sha256(projection) != evidence["candidate_frame_projection_sha256"]:
                    raise ValueError("arrival correction frame projection hash mismatch")
                source_yaw = projection["candidate_reprojections"][candidate_uid]["current_map_from_odom"]["yaw_rad"]
                correction_root = root / f"alignment_{correction + 1:02d}"
                try:
                    fallback_frame = plan_and_move(
                        fallback_frame, math.remainder(bearing - source_yaw, 2 * math.pi),
                        correction_root, index, None, purpose="arrival_alignment",
                        offset=float(evidence["measurements"]["range_m"]),
                    )
                except CandidateInspectionRouteUnavailableError:
                    # A degenerate same-position route is not expanded into
                    # fabricated motion; choose a genuinely different view.
                    fallback_frame = plan_and_move(
                        fallback_frame, math.remainder(bearing - source_yaw + math.pi / 4, 2 * math.pi),
                        correction_root / "diverse_fallback", index, None,
                    )
        raise error  # Defensive: the bounded loop returns or raises above.

    def move_view(frame, requested_normal, root, index, source_path):
        planned = plan_and_move(frame, requested_normal, root, index, source_path)
        return admit_corrected(root / "arrival", planned, index)

    def move_opposite(frame, observation, root, index):
        nonlocal motion_serial
        motion_serial += 1
        try:
            return move_certified_opposite(
                observation_frame=frame, observation=observation, source_config=source_config,
                effects=effects, source_registry=source_registry, candidate_root=root,
                candidate_run_id=f"{candidate_run_id}_inspection_{motion_serial:03d}",
                candidate_index=100000 + candidate_index * 1000 + motion_serial,
                observed_view_normals=tuple(seen_normals),
            )
        except CandidateObservationUnavailableError as exc:
            if exc.status_evidence.get("reasons") != ["bearing_error_above_maximum"]:
                raise
            return admit_corrected(root / "corrected_arrival", frame, index)

    def progress(frame, observation):
        evidence = load_candidate_inspection_observation(observation.inspection_observation_path)
        center = evidence["stand_center"]
        if evidence["candidate_uid"] != candidate_uid or evidence["planning_frame"] != frame.config.planning_frame:
            raise ValueError("inspection observation candidate/frame binding mismatch")
        if math.hypot(center["x_m"] - frame.candidate.geometry.x_m,
                      center["y_m"] - frame.candidate.geometry.y_m) > 1.0e-6:
            raise ValueError("inspection observation target center binding mismatch")
        return {**evidence, "inspection_observation_path": str(observation.inspection_observation_path)}

    try:
        initial = admit_corrected(candidate_root, observation_frame, 0)
    except CandidateInspectionRouteUnavailableError as exc:
        raise CandidateObservationUnavailableError(
            candidate_uid=candidate_uid, observation_attempt_index=0,
            reason="candidate_local_arrival_correction_unavailable",
            process_evidence={"observer_started": False, "reason": str(exc)},
            status_evidence={"motion_authorized": False},
        ) from exc
    return execute_candidate_inspection(
        candidate_uid=candidate_uid, candidate_root=candidate_root, initial_frame=initial,
        max_views=source_config.max_candidate_inspection_views,
        effects=CandidateInspectionEffects(
            capture=lambda frame, output, index: effects.capture_observation(
                observation_request_type(frame.candidate, output, index)),
            canonical_normal=normal, move_view=move_view, move_opposite=move_opposite,
            progress_evidence=progress,
        ),
    )
