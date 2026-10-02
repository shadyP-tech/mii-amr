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
from scripts.aufgabe04.real_robot.candidate.target_admission import bind_current_lidar_target, require_frame_target
from scripts.aufgabe04.real_robot.candidate.retained_orientation import retain_orientation_after_arrival
from scripts.aufgabe04.navigation.planning.map_io import read_map_metadata
from scripts.aufgabe04.real_robot.candidate.inspection_execution import (
    CandidateInspectionEffects, CandidateInspectionRouteUnavailableError,
    execute_candidate_inspection,
)
from scripts.aufgabe04.real_robot.candidate.inspection_policy import (
    MINIMUM_VIEW_SEPARATION_RAD, novel_view,
)
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


def lidar_support_goal_is_useful(*, start, goal, target_x_m, target_y_m):
    """Require a separated view or a meaningful increase in spatial sampling.

    A small same-bearing displacement cannot establish independent geometry.
    This applies only to support probes, never a fitted camera-pose correction.
    """
    start_bearing = math.atan2(start.y_m - target_y_m, start.x_m - target_x_m)
    goal_bearing = math.atan2(goal["y_m"] - target_y_m, goal["x_m"] - target_x_m)
    separated = abs(math.remainder(goal_bearing - start_bearing, math.pi)) >= MINIMUM_VIEW_SEPARATION_RAD
    range_gain = (math.hypot(start.x_m - target_x_m, start.y_m - target_y_m)
                  - math.hypot(goal["x_m"] - target_x_m, goal["y_m"] - target_y_m))
    return separated or range_gain >= .05


def execute_local_candidate_inspection(
    *, observation_frame, source_config, effects, source_registry,
    candidate_root: Path, candidate_run_id: str, candidate_index: int,
    admit_arrival, admit_planning, move_certified_opposite, execute_motion,
    frame_type, request_type, observation_request_type,
):
    candidate_uid = observation_frame.candidate.candidate_uid
    seen_normals: list[float] = []
    motion_serial = 0
    active_view_index = 0

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
        with phase("planning_frame_admission", root, active_view_index):
            config, candidate, pose, planning, artifacts = admit_planning(
                source_config=source_config, effects=effects, source_registry=source_registry,
                candidate_uid=candidate_uid, candidate_root=root,
            )
        result = frame_type(config, candidate, planning,
                          None if artifacts is None else artifacts.camera_decision_binding(), pose,
                          localization_evidence_path=(None if planning is None else
                              root / "opposite_face_planning_localization.json"))
        return result

    def require_translation_support(frame, root):
        if getattr(source_config, "require_current_lidar_support", False):
            from scripts.aufgabe04.real_robot.candidate.approach import (
                _require_current_lidar_target, _validate_current_lidar_target_agreement,
            )
            estimate, support = _require_current_lidar_target(
                config=frame.config, effects=effects, planning_frame=frame.planning_frame, candidate_uid=candidate_uid,
                output_dir=root / "current_lidar_target", attempt_index=active_view_index)
            if frame.retained_backside_axis_path is not None:
                from scripts.aufgabe04.artifacts.current_target_estimate import planning_target_geometry
                from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import load_backside_axis_planning_observation
                axis = load_backside_axis_planning_observation(frame.retained_backside_axis_path)
                target = planning_target_geometry(frame.candidate, axis.validated_target_center)
                _validate_current_lidar_target_agreement(
                    estimate=estimate, support=support, candidate=frame.candidate,
                    target_geometry=target, attempt_index=active_view_index,
                )
            frame = bind_current_lidar_target(
                replace(frame, camera_target_geometry=None, camera_alignment=None,
                        retained_lidar_target=None, retained_survey_target=None,
                        current_lidar_target_path=None,
                        camera_target_geometry_evidence_path=None),
                evidence_path=Path(support["evidence_path"]),
            )
        return frame

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
                      *, purpose="diverse_inspection", offset=None, camera_recovery=None,
                      prepared_plan=None, selection_evidence=None, before_motion=None,
                      lidar_support_hint=None):
        nonlocal motion_serial
        if before_motion is not None:
            before_motion()
        source_frame = frame
        if prepared_plan is None:
            frame = retain_orientation_after_arrival(source_frame, fresh_frame(root / "planning"), root / "planning")
            frame = require_translation_support(frame, root / "planning")
        current = pose(frame)
        if current is None:
            raise RuntimeError("inspection route lacks a fresh finite start pose")
        map_normal = math.remainder(canonical_normal + yaw(frame), 2.0 * math.pi)
        view_path = root / "inspection_view.json"
        current_estimate = None
        current_support_path = None
        if prepared_plan is None and getattr(source_config, "require_current_lidar_support", False):
            from scripts.aufgabe04.real_robot.candidate.current_lidar_targets import load_current_lidar_target
            current_support_path = frame.current_lidar_target_path
            current_estimate = load_current_lidar_target(current_support_path,
                candidate_uid=candidate_uid, snapshot=frame.config.snapshot)
        elif prepared_plan is not None and getattr(source_config, "require_current_lidar_support", False):
            # Preparation must follow acquisition in this exact stopped frame.
            # Confirm the fixed camera fit, without rewriting its sealed point.
            from scripts.aufgabe04.real_robot.candidate.current_lidar_targets import load_current_lidar_target
            estimate = load_current_lidar_target(frame.current_lidar_target_path,
                candidate_uid=candidate_uid, snapshot=frame.config.snapshot)
            alignment = prepared_plan.camera_alignment
            if alignment is None or math.hypot(
                    estimate["x_m"]-alignment["center_x_m"],
                    estimate["y_m"]-alignment["center_y_m"]) > (
                    estimate["uncertainty_m"]+alignment["center_uncertainty_m"]):
                raise CandidateObservationUnavailableError(
                    candidate_uid=candidate_uid, observation_attempt_index=index,
                    reason="candidate_target_ineligible",
                    process_evidence={"observer_started": False, "motion_authorized": False},
                    status_evidence={"reason": "current_lidar_disagrees_with_prepared_alignment"},
                )
        write_candidate_inspection_view(
            view_path, snapshot=frame.config.snapshot, candidate_uid=candidate_uid,
            start=current, view_normal_rad=map_normal,
            purpose=(purpose if current_estimate is None else "current_lidar_target"),
            view_index=index, source_observation_path=source_path,
            camera_alignment=None if prepared_plan is None else prepared_plan.camera_alignment,
            validated_target_center=current_estimate, current_lidar_targets_path=current_support_path,
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
            prepared_plan=prepared_plan, selection_evidence=selection_evidence,
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
            if (purpose == "lidar_axis_hint" and prepared_plan is None
                    and not lidar_support_goal_is_useful(
                        start=current, goal=goal,
                        target_x_m=frame.candidate.geometry.x_m,
                        target_y_m=frame.candidate.geometry.y_m)):
                raise CandidateInspectionRouteUnavailableError(
                    "LiDAR support goal adds neither a separated view nor useful range improvement",
                    reason_code="lidar_support_goal_not_useful",
                    evidence={"motion_published": False, "selected_approach_pose": goal},
                )
            if lidar_support_hint is not None:
                from scripts.aufgabe04.real_robot.candidate.lidar_sampling import predicted_head_support
                hint, tf = lidar_support_hint, frame.planning_frame.map_from_odom
                center = hint["center_odom"]
                c, s = math.cos(tf.yaw_rad), math.sin(tf.yaw_rad)
                cx, cy = tf.x_m+c*center["x_m"]-s*center["y_m"], tf.y_m+s*center["x_m"]+c*center["y_m"]
                q = hint["scan_pose_robot"]
                gc, gs = math.cos(goal["yaw_rad"]), math.sin(goal["yaw_rad"])
                sx, sy = goal["x_m"]+gc*q["x_m"]-gs*q["y_m"], goal["y_m"]+gs*q["x_m"]+gc*q["y_m"]
                normal = hint["tangent_odom_rad"]+tf.yaw_rad+math.pi/2
                incident = abs(math.remainder(math.atan2(sy-cy, sx-cx)-normal, math.pi))
                predicted = predicted_head_support(distance_m=math.hypot(sx-cx, sy-cy),
                    incidence_rad=incident, angular_step_rad=hint["angular_step_rad"])
                worst = predicted_head_support(distance_m=math.hypot(sx-cx, sy-cy),
                    incidence_rad=incident+hint["angle_uncertainty_rad"], angular_step_rad=hint["angular_step_rad"])
                review = {**predicted, "worst_case_expected_return_count": worst["expected_return_count"],
                    "incidence_rad": incident, "source_hint": hint, "selected_approach_pose": goal,
                    "accepted": predicted["minimum_phase_return_count"] >= 4,
                    "head_alignment_verified": False, "motion_authorized": False}
                write_content_hashed_json(root / "lidar_support_goal_review.json", review,
                                         hash_field="lidar_support_goal_review_sha256")
                if not review["accepted"]:
                    raise CandidateInspectionRouteUnavailableError(
                        "Quantized support view predicts fewer than four head returns",
                        reason_code="lidar_support_goal_too_sparse", evidence=review)
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
        # A slow preflight/plan cannot authorize another optional recovery move
        # after its elapsed budget. An executing sealed leg keeps its own limits.
        if before_motion is not None:
            before_motion()
        motion_serial += 1
        run_id = f"{candidate_run_id}_inspection_{motion_serial:03d}"
        completed_frames = []
        try:
            with phase("inspection_motion", root, index, purpose=purpose, run_id=run_id):
                execute_motion(
                    config=frame.config, effects=effects, candidate_root=root,
                    plan_request=request, initial_sealed=sealed, run_id=run_id,
                    leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
                    candidate_index=100000 + candidate_index * 1000 + motion_serial,
                    target_id=candidate_uid, frame_source_config=source_config,
                    source_registry=source_registry, plan_planning_frame=frame.planning_frame,
                    completed_frame_sink=completed_frames.append,
                    retained_backside_axis_path=frame.retained_backside_axis_path,
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
        if not completed_frames and getattr(source_config, "require_current_lidar_support", False):
            raise RuntimeError("inspection motion did not return its completed target frame")
        completed = completed_frames[0] if completed_frames else frame
        if completed.retained_backside_axis_path is None and frame.retained_backside_axis_path is not None:
            # This original certified receipt is projected by arrival admission
            # into the completed route's new localization frame.
            completed = replace(completed, retained_backside_axis_path=frame.retained_backside_axis_path)
        return completed

    def admit(root, fallback_frame, index, *, refine_survey_target=False):
        with phase("arrival_admission", root, index):
            result = admit_arrival(
                source_config=source_config, effects=effects, source_registry=source_registry,
                candidate_uid=candidate_uid, candidate_root=root, observation_attempt_index=0,
                allow_centering_acquisition=effects.run_centering_turn is not None,
                **({"refine_survey_target": True}
                   if refine_survey_target and fallback_frame.retained_survey_target is not None
                   else {}),
                **({"target_source_frame": fallback_frame}
                   if (getattr(fallback_frame, "camera_alignment", None) is not None
                       or getattr(fallback_frame, "camera_target_geometry", None) is not None)
                   else {}),
                **({"retained_backside_axis_path": fallback_frame.retained_backside_axis_path}
                   if fallback_frame.retained_backside_axis_path is not None else {}),
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
        nonlocal active_view_index
        active_view_index = index
        require_frame_target(frame, evidence_path=output.with_name(output.name + "_target_admission.json"),
                             attempt_index=index)
        retained = getattr(frame, "retained_backside_axis_path", None)
        if retained is not None:
            return capture_observation(observation_request_type(
                frame.candidate, output, index, allow_centering=False,
                timeout_sec=source_config.camera_timeout_sec,
                retained_backside_axis_path=retained,
                candidate_crop_snapshot_path=frame.decision_binding.camera_snapshot_path,
                candidate_position_epoch_path=frame.decision_binding.projection_path,
            ))
        return capture_observation(observation_request_type(frame.candidate, output, index,
            candidate_crop_snapshot_path=(None if frame.decision_binding is None
                else frame.decision_binding.camera_snapshot_path),
            candidate_position_epoch_path=(None if frame.decision_binding is None else frame.decision_binding.projection_path)))

    def centered_capture(frame, output, index):
        nonlocal active_view_index
        active_view_index = index
        def capture(current, destination, view, enabled, timeout, not_before):
            require_frame_target(current,
                evidence_path=destination.with_name(destination.name + "_target_admission.json"),
                attempt_index=view)
            return capture_observation(observation_request_type(
                current.candidate, destination, view, allow_centering=enabled,
                timeout_sec=timeout, observation_not_before_sec=not_before,
                retained_backside_axis_path=current.retained_backside_axis_path,
                candidate_crop_snapshot_path=(None if current.decision_binding is None
                    else current.decision_binding.camera_snapshot_path),
                candidate_position_epoch_path=(None if current.decision_binding is None else current.decision_binding.projection_path),
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
            target_geometry = getattr(updated, "camera_target_geometry", None) or updated.candidate.geometry
            decision = evaluate_candidate_arrival_admission(
                pose(updated), target_x_m=target_geometry.x_m,
                target_y_m=target_geometry.y_m,
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

    recover_lidar = None
    if (effects.capture_lidar_view is not None and source_config.camera_calibration is not None
            and source_registry is not None and effects.admit_planning_frame is not None
            and effects.load_route_uncertainty_readiness is not None):
        from scripts.aufgabe04.real_robot.candidate.lidar_acquisition import create_lidar_camera_recovery
        from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import CandidateRouteUncertaintyReadinessRequest

        def load_alignment_uncertainty(frame):
            return effects.load_route_uncertainty_readiness(CandidateRouteUncertaintyReadinessRequest(
                preflight_json=frame.localization_evidence_path,
                expected_start=frame.planning_frame.current_pose,
                planning_frame=frame.planning_frame.map_frame, odom_frame=frame.planning_frame.odom_frame,
                robot_radius_m=source_config.robot_radius_m,
                sigma_multiplier=source_config.uncertainty_sigma_multiplier,
            ))

        def plan_lidar_move(frame, direction, root, serial, source_path, **kwargs):
            # LiDAR and camera search consume the same candidate route ledger.
            return route_search.move_direction(
                requested_normal_rad=direction, standoffs=(kwargs["offset"],),
                output_root=root,
                move=lambda offset, proposal_root: plan_and_move(
                    frame, direction, proposal_root, active_view_index, source_path,
                    **{**kwargs, "offset": offset}),
            )

        recovery = create_lidar_camera_recovery(
            source_config=source_config,
            source_registry=source_registry, effects=effects, candidate_root=candidate_root,
            fresh_frame=fresh_frame, plan_and_move=plan_lidar_move,
            load_uncertainty=load_alignment_uncertainty,
            require_translation_support=require_translation_support,
        )

        def recover_lidar(frame, root, index):
            nonlocal active_view_index
            active_view_index = index
            with phase("lidar_recovery", root, index):
                if route_search.budget_exhausted:
                    raise CandidateInspectionRouteUnavailableError(
                        "candidate route proposal budget exhausted before LiDAR recovery",
                        reason_code="route_proposal_budget_exhausted", evidence=route_search.to_dict(),
                    )
                updated, review, _ = recovery(frame)
                if not review["motion_completed"]:
                    return None
                if review["head_alignment_verified"] and review["camera_centered_verified"]:
                    # Fresh fitted geometry has already checked range and camera
                    # pose. Do not undo it with a base-to-old-centroid correction.
                    write_content_hashed_json(root / "candidate_arrival_admission.json", {
                        **review, "accepted": True,
                        "admission_kind": "fresh_lidar_calibrated_camera_alignment",
                        "requires_live_target_association": True,
                        "camera_centered": True, "motion_authorized": False,
                    }, hash_field="candidate_arrival_admission_sha256")
                    return updated
                # Reacquire the actual stopped arrival, but do not start a second
                # correction route inside this one-move recovery callback.
                return admit(root / "arrival", updated, index)

    try:
        # Refine the reached survey target once. Weak support leaves passive
        # camera acquisition available; further precision moves are separate.
        initial = (admit(candidate_root, observation_frame, 0, refine_survey_target=True)
                   if source_config.camera_calibration is not None
                   else admit_corrected(candidate_root, observation_frame, 0))
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
            recover_lidar=recover_lidar,
        ),
    )
