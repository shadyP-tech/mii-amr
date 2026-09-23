"""Bind local inspection policy to the existing candidate motion contracts.

The parent injects its planning-frame, route, recovery and arrival functions.
This module never publishes motion or bypasses a child preflight/permit.
"""

from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
import json
import math
from pathlib import Path

from scripts.aufgabe04.artifacts.candidate_inspection_observation import (
    load_candidate_inspection_observation,
)
from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json, payload_sha256, write_content_hashed_json,
)
from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import (
    write_backside_axis_frame_projection,
)
from scripts.aufgabe04.navigation.approach.candidate_inspection_view import (
    write_candidate_inspection_view,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_models import (
    CandidatePreapproachUnreachableError,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.approach.candidate_arrival_admission import (
    CandidateArrivalAdmissionConfig, evaluate_candidate_arrival_admission,
)
from scripts.aufgabe04.real_robot.candidate.centering_execution import capture_with_centering
from scripts.aufgabe04.real_robot.candidate.retained_orientation import retain_orientation_after_arrival
from scripts.aufgabe04.navigation.planning.map_io import read_map_metadata
from scripts.aufgabe04.real_robot.candidate.inspection_execution import (
    CandidateInspectionEffects, CandidateInspectionRouteUnavailableError,
    execute_candidate_inspection,
)
from scripts.aufgabe04.real_robot.candidate.inspection_policy import novel_view
from scripts.aufgabe04.real_robot.candidate.camera_distance_recovery import (
    distance_recovery_goal_is_useful, select_camera_distance_recovery,
)
from scripts.aufgabe04.real_robot.candidate.inspection_route_search import (
    CandidateInspectionRouteSearch, bounded_inspection_standoffs,
)
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

    @contextmanager
    def phase(name, root, index, **details):
        """Leave a durable boundary even when a live effect never returns."""
        def emit(state, **extra):
            effects.event_sink(candidate_root / "inspection_handoff_events.jsonl", {
                "schema_version": 1, "event": name, "state": state,
                "candidate_uid": candidate_uid, "view_index": index,
                "output_dir": str(root), "timestamp_unix_sec": effects.clock(),
                "motion_authorized": False, **details, **extra,
            })
        emit("started")
        try:
            yield
        except BaseException as exc:
            try:
                emit("failed", exception_type=type(exc).__name__, detail=str(exc)[:1024])
            except Exception as log_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"Could not persist handoff failure: {log_error}")
            raise
        else:
            emit("returned")

    def capture_observation(request):
        with phase("observer_capture", request.output_dir, request.attempt_index):
            return effects.capture_observation(request)

    route_search = CandidateInspectionRouteSearch(
        candidate_uid,
        event_sink=lambda event: effects.event_sink(
            candidate_root / "inspection_route_proposals.jsonl", event,
        ),
    )

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
                      *, purpose="diverse_inspection", offset=None, camera_recovery=None):
        nonlocal motion_serial
        source_frame = frame
        frame = retain_orientation_after_arrival(source_frame, fresh_frame(root / "planning"), root / "planning")
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
            if isinstance(exc, CandidatePreapproachUnreachableError) and exc.candidate_uid != candidate_uid:
                raise
            raise CandidateInspectionRouteUnavailableError(
                text, reason_code="static_proposal_unavailable",
                evidence={"error_type": type(exc).__name__, "motion_published": False},
            ) from exc
        summary_path = request.output_dir / "pipeline_summary.json"
        if summary_path.is_file():
            goal = json.loads(summary_path.read_text())["selected_approach_pose"]
            achieved = math.remainder(math.atan2(
                goal["y_m"] - frame.candidate.geometry.y_m,
                goal["x_m"] - frame.candidate.geometry.x_m,
            ) - yaw(frame), 2.0 * math.pi)
            if purpose == "diverse_inspection" and not novel_view(achieved, seen_normals):
                raise CandidateInspectionRouteUnavailableError(
                    "quantized inspection goal repeats an observed viewing direction",
                    reason_code="quantized_view_already_observed",
                )
            if camera_recovery is not None and not distance_recovery_goal_is_useful(
                start_range_m=math.hypot(current.x_m - frame.candidate.geometry.x_m,
                                         current.y_m - frame.candidate.geometry.y_m),
                goal_range_m=math.hypot(goal["x_m"] - frame.candidate.geometry.x_m,
                                        goal["y_m"] - frame.candidate.geometry.y_m),
                requested_normal_rad=canonical_normal, achieved_normal_rad=achieved,
                minimum_range_m=camera_recovery.minimum_range_m,
                maximum_range_m=camera_recovery.maximum_range_m,
            ):
                raise CandidateInspectionRouteUnavailableError(
                    "quantized framing goal does not preserve bearing and increase useful range",
                    reason_code="camera_distance_goal_not_useful",
                )
        elif frame.planning_frame is not None or camera_recovery is not None:
            raise RuntimeError("inspection route lacks materialized goal evidence")
        motion_serial += 1
        run_id = f"{candidate_run_id}_inspection_{motion_serial:03d}"
        try:
            with phase("inspection_motion", root, index, purpose=purpose, run_id=run_id):
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
            raise CandidateInspectionRouteUnavailableError(
                str(exc), reason_code="no_motion_route_uncertainty_rejection",
                evidence=decision.to_event_fields(),
            ) from exc
        return frame

    def admit(root, fallback_frame, index):
        with phase("arrival_admission", root, index):
            result = admit_arrival(
                source_config=source_config, effects=effects, source_registry=source_registry,
                candidate_uid=candidate_uid, candidate_root=root, observation_attempt_index=0,
                allow_centering_acquisition=effects.run_centering_turn is not None,
            )
        if result.planning_frame is None and result.observation_pose is None:
            result = replace(result, observation_pose=pose(fallback_frame))
        return retain_orientation_after_arrival(fallback_frame, result, root)

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
        map_yaml = Path(source_config.map_yaml)
        if map_yaml.is_file():
            resolution = read_map_metadata(map_yaml).resolution
        else:
            # Preserve injected no-map offline effects without inventing a
            # raster margin. The production planner still requires the map
            # and fails closed if it is absent; no smaller pose is proposed.
            resolution = None
        offsets = bounded_inspection_standoffs(
            source_config.approach_offset_m,
            minimum_active_standoff_m=float(source_config.physical_clearance["minimum_active_standoff_m"]),
            candidate_transit_radius_m=source_config.candidate_transit_radius_m,
            map_resolution_m=resolution,
        )
        planned = route_search.move_direction(
            requested_normal_rad=requested_normal, standoffs=offsets, output_root=root,
            move=lambda offset, proposal_root: plan_and_move(
                frame, requested_normal, proposal_root, index, source_path, offset=offset,
            ),
        )
        # Arrival handling remains outside no-motion standoff fallback: the
        # successful child may have moved and its failures cannot retry radius.
        return admit_corrected(root / "arrival", planned, index)

    def move_opposite(frame, observation, root, index):
        nonlocal motion_serial
        motion_serial += 1
        try:
            arrival = move_certified_opposite(
                observation_frame=frame, observation=observation, source_config=source_config,
                effects=effects, source_registry=source_registry, candidate_root=root,
                candidate_run_id=f"{candidate_run_id}_inspection_{motion_serial:03d}",
                candidate_index=100000 + candidate_index * 1000 + motion_serial,
                observed_view_normals=tuple(seen_normals),
            )
        except CandidateObservationUnavailableError as exc:
            if exc.status_evidence.get("reasons") != ["bearing_error_above_maximum"]:
                raise
            arrival = admit_corrected(root / "corrected_arrival", frame, index)
        # Use the original proof and source frame even if arrival alignment
        # required a correction. Reprojection never fits another camera angle.
        if arrival.decision_binding is not None:
            source, target = frame.decision_binding, arrival.decision_binding
            if source is None:
                raise ValueError("retained backside axis lacks its source candidate frame")
            retained = root / "arrival_backside_orientation.json"
            write_backside_axis_frame_projection(retained,
                axis_evidence_path=observation.axis_observation_path,
                source_candidate_projection_path=source.projection_path,
                source_candidate_projection_sha256=source.projection_sha256,
                target_candidate_projection_path=target.projection_path,
                target_candidate_projection_sha256=target.projection_sha256,
                target_candidate_x_m=arrival.candidate.geometry.x_m,
                target_candidate_y_m=arrival.candidate.geometry.y_m)
            arrival = replace(arrival, retained_backside_axis_path=retained)
        return arrival

    def distance_recovery(frame, evidence):
        current = pose(frame)
        if current is None:
            return None
        return select_camera_distance_recovery(
            evidence.get("camera_framing"), candidate_uid=candidate_uid,
            current_range_m=math.hypot(current.x_m - frame.candidate.geometry.x_m,
                                      current.y_m - frame.candidate.geometry.y_m),
            preferred_range_m=source_config.approach_offset_m,
            maximum_allowed_range_m=(source_config.approach_offset_m
                                     + source_config.camera_arrival_range_slack_m),
        )

    def move_distance_recovery(frame, recovery, root, index, source_path):
        requested_normal = normal(frame)
        if requested_normal is None:
            raise RuntimeError("camera distance recovery lacks a finite observation pose")
        hint_path = root / "camera_framing_hint.json"
        write_content_hashed_json(
            hint_path, {**recovery.to_dict(), "source_observation_path": (
                None if source_path is None else str(source_path))},
            hash_field="camera_framing_recovery_sha256",
        )
        planned = route_search.move_direction(
            requested_normal_rad=requested_normal, standoffs=recovery.standoffs_m,
            output_root=root,
            move=lambda offset, proposal_root: plan_and_move(
                frame, requested_normal, proposal_root, index, hint_path,
                purpose="camera_distance_recovery", offset=offset, camera_recovery=recovery,
            ),
        )
        # Motion has completed: arrival failures are not no-motion proposal
        # failures and must not cause another radius attempt or blind orbit.
        return admit(root / "arrival", planned, index)

    def progress(frame, observation):
        evidence = load_candidate_inspection_observation(observation.inspection_observation_path)
        center = evidence["stand_center"]
        if evidence["candidate_uid"] != candidate_uid or evidence["planning_frame"] != frame.config.planning_frame:
            raise ValueError("inspection observation candidate/frame binding mismatch")
        if math.hypot(center["x_m"] - frame.candidate.geometry.x_m,
                      center["y_m"] - frame.candidate.geometry.y_m) > 1.0e-6:
            raise ValueError("inspection observation target center binding mismatch")
        return {**evidence, "inspection_observation_path": str(observation.inspection_observation_path)}

    def capture_frame(frame, output, index):
        retained = getattr(frame, "retained_backside_axis_path", None)
        if retained is not None:
            return capture_observation(observation_request_type(
                frame.candidate, output, index, allow_centering=False,
                timeout_sec=source_config.camera_timeout_sec,
                retained_backside_axis_path=retained,
                candidate_crop_snapshot_path=frame.decision_binding.camera_snapshot_path,
            ))
        return capture_observation(observation_request_type(frame.candidate, output, index,
            candidate_crop_snapshot_path=(None if frame.decision_binding is None
                else frame.decision_binding.camera_snapshot_path)))

    def centered_capture(frame, output, index):
        def capture(current, destination, view, enabled, timeout, not_before):
            return capture_observation(observation_request_type(
                current.candidate, destination, view, allow_centering=enabled,
                timeout_sec=timeout, observation_not_before_sec=not_before,
                retained_backside_axis_path=current.retained_backside_axis_path,
                candidate_crop_snapshot_path=(None if current.decision_binding is None
                    else current.decision_binding.camera_snapshot_path),
            ))

        def turn(current, advisory_path, root, serial, remaining, previous_result):
            outcome = effects.run_centering_turn(
                candidate=current.candidate, advisory_path=advisory_path,
                output_dir=root, view_id=f"{candidate_uid}:inspection:{index}",
                turn_index=serial, remaining_travel_rad=remaining,
                previous_result_path=previous_result,
            )
            # The sealed yaw-only child proves its signed turn and stopped
            # odometry. Reproject localization/candidate geometry afresh; old
            # map-facing yaw is not an admission criterion for this new view.
            updated = retain_orientation_after_arrival(current, fresh_frame(root / "arrival"), root / "arrival")
            decision = evaluate_candidate_arrival_admission(
                pose(updated), target_x_m=updated.candidate.geometry.x_m,
                target_y_m=updated.candidate.geometry.y_m,
                config=CandidateArrivalAdmissionConfig(
                    min_range_m=source_config.physical_clearance["minimum_active_standoff_m"],
                    max_range_m=source_config.approach_offset_m + source_config.camera_arrival_range_slack_m,
                    max_bearing_error_rad=math.pi,
                ),
            )
            evidence = {**decision.to_evidence_dict(), "candidate_uid": candidate_uid,
                        "admission_kind": "candidate_centering_reacquisition",
                        "turn_result_path": str(outcome.result_path),
                        "bearing_check_source": "sealed_inspection_turn",
                        "camera_centered": False, "requires_fresh_observation": True,
                        "motion_authorized": False}
            path = root / "post_turn_arrival.json"
            write_content_hashed_json(path, evidence,
                                      hash_field="candidate_centering_arrival_sha256")
            if not decision.accepted:
                raise CandidateObservationUnavailableError(
                    candidate_uid=candidate_uid, observation_attempt_index=index,
                    reason="candidate_centering_post_turn_range_rejected",
                    process_evidence={"arrival_path": str(path)}, status_evidence=evidence,
                )
            return updated, outcome

        return capture_with_centering(
            candidate_uid=candidate_uid, frame=frame, output_dir=output,
            view_index=index, timeout_sec=source_config.camera_timeout_sec,
            capture=capture, turn=turn,
        )

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
            capture=capture_frame,
            canonical_normal=normal, move_view=move_view, move_opposite=move_opposite,
            progress_evidence=progress, route_search_evidence=route_search.to_dict,
            distance_recovery=distance_recovery, move_distance_recovery=move_distance_recovery,
            capture_centered=centered_capture if effects.run_centering_turn is not None else None,
        ),
    )
