"""Optional stopped LiDAR recovery interleaved with camera inspection.

Local views add evidence only. Every displacement uses the existing candidate
route planner and motion gates; neither a fit nor this controller grants motion
authority or changes the frozen candidate registry.
"""
from __future__ import annotations

from dataclasses import replace
import math
import time

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.camera_candidate_selection import CameraCandidateSelectionConfig, NoFeasibleCameraCandidateError
from scripts.aufgabe04.navigation.approach.candidate_arrival_admission import CandidateArrivalAdmissionConfig, evaluate_candidate_arrival_admission
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.candidate_preapproach_selection import plan_and_select_camera_candidate
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import NoUncertaintyAdmittedCameraCandidateError
from scripts.aufgabe04.navigation.approach.lidar_inspection_hint import (
    derive_lidar_inspection_hints, fit_current_lidar_view, analyze_candidate_lidar_support)
from scripts.aufgabe04.navigation.approach.lidar_head_observability import (
    lidar_head_model_admission, verify_lidar_head_observability,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.candidate.inspection_route_search import CandidateInspectionRouteUnavailableError
from scripts.aufgabe04.real_robot.candidate.lidar_inspection_hints import load_camera_lidar_receipts
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256


MAX_PROBE_MOVES = 3
MAX_ALIGNMENT_MOVES = 2  # Initial placement and one measured arrival correction.
PROBE_STANDOFFS_M = (.55, .60, .65)
MAX_RECOVERY_ELAPSED_SEC = 120.
MAX_COHORTS_PER_STOPPED_VIEW = 3


class _RecoveryBudgetExpired(CandidateInspectionRouteUnavailableError):
    def __init__(self):
        super().__init__("Optional LiDAR recovery active-time budget exhausted",
                         reason_code="lidar_recovery_time_budget_exhausted")


def create_bounded_lidar_recovery(*, observe, move_probe, move_aligned, persist,
                                  move_sampling=None,
                                  monotonic=time.monotonic,
                                  budget_sec=MAX_RECOVERY_ELAPSED_SEC):
    """Return one-motion recovery steps with cumulative evidence and budgets.

    The caller must obtain camera evidence before the first call and after every
    completed move. Time between calls belongs to the camera, not this budget.
    Only typed, proven no-motion route rejections permit another proposal. A
    deadline never interrupts an in-flight effect or hides an unknown outcome.
    """
    if not math.isfinite(budget_sec) or budget_sec <= 0:
        raise ValueError("LiDAR recovery budget must be finite and positive")
    history = []
    probes = alignments = sampling_moves = serial = 0
    base_direction = None
    elapsed = 0.
    active_start = None
    terminal_reason = None

    def used_time():
        return elapsed + (0. if active_start is None else max(0., monotonic()-active_start))

    def recover(initial_frame):
        nonlocal probes, alignments, sampling_moves, serial, base_direction, elapsed, active_start, terminal_reason
        frame, hint, current_fit = initial_frame, None, None
        review = {"head_alignment_verified": False, "camera_centered_verified": False}
        motion_completed = False
        active_start = monotonic()

        def report(reason=None):
            return {"history": list(history), "probe_moves_attempted": probes,
                    "alignment_moves_attempted": alignments,
                    "sampling_moves_attempted": sampling_moves,
                    "head_alignment_verified": bool(review.get("head_alignment_verified")),
                    "camera_centered_verified": bool(review.get("camera_centered_verified")),
                    "motion_completed": motion_completed, "reason": reason or review.get("reason"),
                    "active_elapsed_sec": used_time(), "active_time_budget_sec": budget_sec,
                    "recovery_complete": terminal_reason is not None,
                    "motion_authorized": False, "stand_axis_authorized": False}

        def checkpoint(event, **details):
            if used_time() >= budget_sec:
                raise _RecoveryBudgetExpired()
            history.append({"event": event, **details})
            persist(report())

        def observation():
            nonlocal frame, hint, current_fit, review, serial
            checkpoint("stopped_observation_started", observation_index=serial)
            index = serial
            serial += 1
            frame, hint, current_fit, review = observe(frame, index, checkpoint)
            history.append({"event": "stopped_observation", **review})
            persist(report())

        try:
            if terminal_reason is not None:
                return frame, report(terminal_reason), None
            observation()
            while True:
                if review.get("acquisition_unavailable"):
                    terminal_reason = review.get("reason", "acquisition_unavailable")
                    break
                if review.get("head_alignment_verified") and review.get("camera_centered_verified"):
                    break
                if (hint is None and move_sampling is not None and sampling_moves == 0
                        and review.get("boundary_fragmentation_detected")):
                    sampling_moves += 1
                    kind = "scan_boundary_sampling"
                    move = lambda: move_sampling(frame, review, sampling_moves, checkpoint)
                elif hint is not None:
                    if alignments >= MAX_ALIGNMENT_MOVES:
                        terminal_reason = "alignment_correction_budget_exhausted"
                        break
                    alignments += 1
                    kind = "normal_alignment"
                    move = lambda: move_aligned(frame, hint, alignments, checkpoint)
                else:
                    if probes >= MAX_PROBE_MOVES:
                        terminal_reason = "independent_geometry_support_unavailable"
                        break
                    pose = frame.planning_frame.current_pose
                    target = frame.candidate.geometry
                    if base_direction is None:
                        base_direction = math.atan2(pose.y_m-target.y_m, pose.x_m-target.x_m) - frame.planning_frame.map_from_odom.yaw_rad
                    direction = base_direction + (0., math.radians(60), -math.radians(60))[probes]
                    if current_fit is not None:
                        normals = current_fit.normals(frame.config.snapshot, frame.candidate.candidate_uid)
                        canonical = [n-frame.planning_frame.map_from_odom.yaw_rad for n in normals]
                        nearest = min(canonical, key=lambda n: abs(math.remainder(n-base_direction, math.tau)))
                        direction = nearest + (0., math.radians(25), -math.radians(25))[probes]
                    probes += 1
                    kind = "support_view"
                    move = lambda: move_probe(frame, math.remainder(direction, math.tau), probes, checkpoint)
                checkpoint("motion_proposal", kind=kind, probe_moves_attempted=probes,
                           alignment_moves_attempted=alignments)
                try:
                    frame = move()
                except _RecoveryBudgetExpired:
                    raise
                except CandidateInspectionRouteUnavailableError as exc:
                    history.append({"event": "no_motion_route_unavailable", "kind": kind,
                                    "reason": exc.reason_code, "detail": str(exc)})
                    persist(report())
                    if exc.reason_code == "route_proposal_budget_exhausted":
                        terminal_reason = exc.reason_code
                        break
                    # A shared route ledger may wrap the deadline exception in
                    # its own exhausted-proposals error. Recheck active time
                    # before consuming any further proposal slot.
                    checkpoint("no_motion_route_returned", kind=kind)
                    continue
                motion_completed = True
                # Persist the successful return before attempting a fresh check.
                # Insufficient post-motion scans still require a camera turn.
                review = {"head_alignment_verified": False, "camera_centered_verified": False,
                          "reason": "post_motion_alignment_unverified"}
                history.append({"event": "motion_returned", "kind": kind})
                persist(report())
                observation()
                break  # Never dispatch a second successful move in this call.
        except _RecoveryBudgetExpired:
            terminal_reason = "lidar_recovery_time_budget_exhausted"
            review = {"head_alignment_verified": False, "camera_centered_verified": False}
            hint = None
            history.append({"event": "recovery_budget_exhausted", "motion_completed": motion_completed})
        except BaseException as exc:
            history.append({"event": "recovery_interrupted" if isinstance(exc, KeyboardInterrupt) else "recovery_failed",
                            "exception_type": type(exc).__name__, "detail": str(exc)[:1024],
                            "motion_completed": motion_completed})
            try:
                persist(report())
            except Exception as log_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"Could not persist LiDAR recovery failure: {log_error}")
            raise
        finally:
            elapsed += max(0., monotonic()-active_start)
            active_start = None
        result = report(terminal_reason)
        persist(result)
        return frame, result, hint

    return recover


def create_lidar_camera_recovery(*, source_config, source_registry, effects,
                                 candidate_root, fresh_frame, plan_and_move,
                                 load_uncertainty, monotonic=time.monotonic,
                                 budget_sec=MAX_RECOVERY_ELAPSED_SEC):
    """Build a lazy candidate-bound recovery session; construction has no effects."""
    from scripts.aufgabe04.real_robot.candidate.approach import _camera_alignment_uncertainty
    from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import (
        CandidateLidarCaptureRequest, CandidateLidarCaptureUnavailableError,
    )
    from scripts.aufgabe04.navigation.approach.lidar_alignment_arrival import verify_lidar_alignment_arrival
    from scripts.aufgabe04.real_robot.readiness.tour_scan_capture import TourScanCaptureError

    uid = None
    root = candidate_root / "lidar_head_acquisition"
    survey = None
    failures = {}
    local_views = []
    model_review = None
    last_fit = None
    last_capture = None
    revision = 0
    step = 0

    def persist(report):
        nonlocal revision
        payload = {**report, "candidate_uid": uid,
                   "source_candidate_snapshot_sha256": candidate_snapshot_sha256(source_config.snapshot),
                   "unavailable_survey_epochs": failures}
        path = root / "history" / f"revision_{revision:03d}.json"
        revision += 1
        write_content_hashed_json(path, payload, hash_field="lidar_acquisition_sha256")
        sink = getattr(effects, "event_sink", None)
        if sink is not None and report["history"]:
            sink(root / "recovery_events.jsonl", {"schema_version": 1, "candidate_uid": uid,
                 "timestamp_unix_sec": effects.clock(), "active_elapsed_sec": report["active_elapsed_sec"],
                 "motion_authorized": False, **report["history"][-1]})

    def all_receipts():
        return tuple(survey) + tuple(r for view in local_views for r in view.receipts)

    def observe(frame, serial, checkpoint):
        nonlocal survey, failures, model_review, last_fit, last_capture
        if model_review is None:
            model_review = lidar_head_model_admission(getattr(source_config, "measured_stand_model", None))
        if not model_review["accepted"]:
            return frame, None, None, {"head_alignment_verified": False, "camera_centered_verified": False,
                "reason": model_review["reason"], "model_admission": model_review,
                "acquisition_unavailable": True}
        if survey is None:
            checkpoint("survey_evidence_load")
            survey, failures = load_camera_lidar_receipts(
                survey_root=source_config.survey_root, plan=source_config.plan,
                snapshot=source_config.snapshot, registry=source_registry)
        checkpoint("observation_preflight", observation_index=serial)
        frame = fresh_frame(root / f"view_{serial:02d}" / "planning")
        route_context = load_uncertainty(frame)
        uncertainty = _camera_alignment_uncertainty(route_context)
        stopped_receipts = ()
        capture_paths = []
        support_fit = current_fit = hint = None
        review = {"head_alignment_verified": False, "camera_centered_verified": False,
                  "reason": "insufficient_head_geometry"}
        # Repeat passive measurements only at this stopped view. All scans,
        # including failures, remain in its support denominator. Another cohort
        # is not an independent viewpoint and does not relax any fit threshold.
        for cohort_index in range(MAX_COHORTS_PER_STOPPED_VIEW):
            checkpoint("scan_capture_started", observation_index=serial, cohort_index=cohort_index)
            floor = effects.clock()
            request = CandidateLidarCaptureRequest(
                plan=source_config.plan,
                candidate_snapshot_sha256=candidate_snapshot_sha256(frame.config.snapshot),
                candidate_uid=uid, viewpoint_id=f"local_{uid}_{serial:02d}",
                output_dir=root / f"view_{serial:02d}" / f"cohort_{cohort_index:02d}" / "capture",
                observation_not_before_sec=floor, planning_frame=frame.planning_frame,
                base_frame=source_config.camera_calibration.base_frame,
                scan_frame=source_config.lidar_scan_frame, scan_topic=source_config.lidar_scan_topic)
            try:
                captured = effects.capture_lidar_view(request)
            except (TourScanCaptureError, CandidateLidarCaptureUnavailableError) as exc:
                return frame, None, None, {**review, "reason": "fresh_scan_cohort_unavailable",
                    "detail": str(exc), "acquisition_unavailable": True,
                    "capture_evidence_paths": capture_paths}
            if (captured.candidate_uid != uid or captured.candidate_snapshot_sha256 != request.candidate_snapshot_sha256
                    or captured.viewpoint_id != request.viewpoint_id):
                raise ValueError("local LiDAR capture candidate binding mismatch")
            checkpoint("scan_capture_returned", observation_index=serial, cohort_index=cohort_index,
                       evidence_path=str(captured.evidence_path))
            current_fit = fit_current_lidar_view(snapshot=frame.config.snapshot, registry=source_registry,
                planning_frame=frame.planning_frame, candidate_uid=uid, receipts=captured.receipts)
            if current_fit is not None:
                center = current_fit.evidence["center_odom"]
                cx, cy, margin = center["x_m"], center["y_m"], current_fit.center_uncertainty_m + .04
            else:
                candidate = source_registry.candidate_for(uid)
                center = candidate.frame_provenance.canonical_odom_point
                cx, cy = center.x_m, center.y_m
                margin = candidate.radius_m + candidate.uncertainty_m + .04
            range_bound = max(math.hypot(r.frame_provenance.canonical_scan_pose_odom.x_m-cx,
                                        r.frame_provenance.canonical_scan_pose_odom.y_m-cy)
                              for r in captured.receipts) + margin
            mount_review = verify_lidar_head_observability(
                stand_model=source_config.measured_stand_model, base_frame=request.base_frame,
                mount_evidence=captured.mount_evidence, target_range_m=range_bound)
            if mount_review["accepted"] and mount_review["source_scan_stamps_sec"] != [r.scan_stamp_sec for r in captured.receipts]:
                raise ValueError("LiDAR head plane evidence differs from captured scan stamps")
            capture_paths.append(str(captured.evidence_path))
            review["head_observability"] = mount_review
            if not mount_review["accepted"]:
                return frame, None, None, {**review, "reason": mount_review["reason"],
                    "acquisition_unavailable": True, "capture_evidence_paths": capture_paths}
            local_views.append(captured)
            last_capture = captured
            stopped_receipts += captured.receipts
            support = analyze_candidate_lidar_support(registry=source_registry,
                candidate_uid=uid, receipts=captured.receipts)
            support_path = captured.evidence_path.parent / "head_support_diagnostics.json"
            write_content_hashed_json(support_path, support, hash_field="head_support_diagnostics_sha256")
            support_fit = fit_current_lidar_view(snapshot=frame.config.snapshot, registry=source_registry,
                planning_frame=frame.planning_frame, candidate_uid=uid, receipts=stopped_receipts)
            hints, diagnostics = derive_lidar_inspection_hints(
                snapshot=frame.config.snapshot, registry=source_registry,
                planning_frame=frame.planning_frame, receipts=all_receipts(),
                additional_viewpoint_ids=tuple(dict.fromkeys(v.viewpoint_id for v in local_views)),
                candidate_uids=(uid,))
            hint = hints.get(uid)
            review.update(fit_diagnostics=diagnostics.get(uid), capture_evidence_path=str(captured.evidence_path),
                          capture_evidence_paths=list(capture_paths), cohort_count=cohort_index+1,
                          examined_scan_count=len(stopped_receipts),
                          support_diagnostics_path=str(support_path),
                          boundary_fragmentation_detected=(support["boundary_fragmented_scan_count"] >=
                              math.ceil(len(captured.receipts)/2) and
                              all(r.scan_metadata is not None for r in captured.receipts)),
                          stopped_view_fit_usable=support_fit is not None,
                          newest_cohort_fit_usable=current_fit is not None)
            # The exact-time pose and verifier always use the NEWEST cohort.
            # Earlier samples can aid planning, but cannot certify a new arrival.
            tf, p = frame.planning_frame.map_from_odom, captured.latest_base_pose_odom
            c, s = math.cos(tf.yaw_rad), math.sin(tf.yaw_rad)
            actual = Pose2D(tf.x_m+c*p.x_m-s*p.y_m, tf.y_m+s*p.x_m+c*p.y_m,
                            math.remainder(tf.yaw_rad+p.yaw_rad, math.tau))
            actual_frame = CandidatePlanningFrame(actual, tf, frame.planning_frame.map_frame,
                                                  frame.planning_frame.odom_frame)
            frame = replace(frame, planning_frame=actual_frame, observation_pose=actual)
            if support_fit is not None:
                break
        if hint is not None and current_fit is not None and uncertainty is not None:
            checkpoint("arrival_verification", observation_index=serial)
            review = {**review, **verify_lidar_alignment_arrival(
                hint=hint, current_fit=current_fit, current_receipts=captured.receipts,
                snapshot=frame.config.snapshot, candidate_uid=uid,
                planning_frame=frame.planning_frame, calibration=source_config.camera_calibration,
                now_sec=effects.clock(), not_before_sec=floor,
                current_base_pose_odom=p, pose_stamp_sec=captured.pose_stamp_sec,
                localization_position_bound_m=uncertainty["localization_position_m"],
                localization_yaw_bound_rad=uncertainty["localization_yaw_rad"])}
            range_review = evaluate_candidate_arrival_admission(
                actual, target_x_m=current_fit.center_x_m, target_y_m=current_fit.center_y_m,
                config=CandidateArrivalAdmissionConfig(
                    min_range_m=source_config.physical_clearance["minimum_active_standoff_m"] + current_fit.center_uncertainty_m,
                    max_range_m=max(.65, source_config.approach_offset_m) + source_config.camera_arrival_range_slack_m,
                    max_bearing_error_rad=math.pi)).to_evidence_dict()
            review["physical_range_admission"] = range_review
            if not range_review["accepted"]:
                review.update(head_alignment_verified=False, accepted=False,
                              reason="fitted_head_arrival_range_rejected")
        last_fit = support_fit
        return frame, hint, support_fit, review

    def move_sampling(frame, review, serial, checkpoint):
        from scripts.aufgabe04.real_robot.candidate.lidar_sampling import MAX_TOTAL_TRAVEL_RAD
        checkpoint("sampling_turn_preflight", sampling_index=serial)
        outcome = effects.run_lidar_sampling_turn(
            candidate=frame.candidate, source_view_path=last_capture.evidence_path,
            snapshot_path=frame.config.snapshot_path, output_dir=root / f"sampling_{serial:02d}",
            before_motion=lambda: checkpoint("motion_dispatch", kind="scan_boundary_sampling", serial=serial))
        result = outcome.result
        if (result.get("purpose") != "candidate_lidar_sampling" or result.get("status") != "completed"
                or result.get("translation_commanded") is not False
                or result.get("total_angular_travel_rad", math.inf) > MAX_TOTAL_TRAVEL_RAD
                or result.get("stopped_at_sec", 0) <= last_capture.pose_stamp_sec):
            raise RuntimeError("LiDAR sampling turn lacks bounded fresh stopped evidence")
        # The next observation reacquires the actual pose and a new stopped epoch.
        return frame

    def move_probe(frame, normal, serial, checkpoint):
        pose = frame.planning_frame.current_pose
        target = frame.candidate.geometry
        distance = math.hypot(pose.x_m-target.x_m, pose.y_m-target.y_m)
        current_normal = math.atan2(pose.y_m-target.y_m, pose.x_m-target.x_m) - frame.planning_frame.map_from_odom.yaw_rad
        closest = min(.55, max(.50, source_config.approach_offset_m))
        if (.50 <= distance <= .65 and distance-closest < .05-1e-9
                and abs(math.remainder(normal-current_normal, math.tau)) < math.radians(10)):
            raise CandidateInspectionRouteUnavailableError(
                "Support probe repeats the current useful range and viewing direction",
                reason_code="redundant_lidar_support_probe")
        last = None
        standoffs = tuple(round(closest + i*.05, 3) for i in range(3))
        support_hint = None
        if last_fit is not None and last_capture is not None:
            from scripts.aufgabe04.real_robot.candidate.lidar_sampling import predicted_head_support
            center = last_fit.evidence["center_odom"]
            p, b = last_capture.receipts[-1].frame_provenance.canonical_scan_pose_odom, last_capture.latest_base_pose_odom
            dx, dy = p.x_m-b.x_m, p.y_m-b.y_m
            c, s = math.cos(b.yaw_rad), math.sin(b.yaw_rad)
            support_hint = {"center_odom": center, "tangent_odom_rad": last_fit.evidence["tangent_odom_rad"],
                "angle_uncertainty_rad": last_fit.angle_uncertainty_rad,
                "scan_pose_robot": {"x_m": c*dx+s*dy, "y_m": -s*dx+c*dy,
                                    "yaw_rad": math.remainder(p.yaw_rad-b.yaw_rad, math.tau)},
                "angular_step_rad": last_capture.receipts[-1].angle_increment_rad,
                "source_evidence_paths": [str(last_capture.evidence_path)], "motion_authorized": False}
            # Rank the closest viable standoff first. The adapter rechecks the
            # actual quantized goal and scanner lever arm before dispatch.
            incident = abs(math.remainder(normal-(support_hint["tangent_odom_rad"]+math.pi/2), math.pi))
            ranked = [(predicted_head_support(distance_m=offset-support_hint["scan_pose_robot"]["x_m"],
                       incidence_rad=incident, angular_step_rad=support_hint["angular_step_rad"])["expected_return_count"], offset)
                      for offset in standoffs]
            standoffs = tuple(offset for _, offset in sorted(ranked, reverse=True))
        for index, offset in enumerate(standoffs):
            checkpoint("support_route_preflight", probe_index=serial, standoff_index=index)
            try:
                return plan_and_move(frame, normal, root / f"probe_{serial:02d}_{index:02d}",
                    serial, None, purpose="lidar_axis_hint", offset=offset,
                    lidar_support_hint=support_hint,
                    before_motion=lambda: checkpoint("motion_dispatch", kind="support_view", serial=serial))
            except _RecoveryBudgetExpired:
                raise
            except CandidateInspectionRouteUnavailableError as exc:
                if exc.reason_code == "route_proposal_budget_exhausted":
                    raise
                last = exc
                checkpoint("support_route_returned_without_motion", probe_index=serial,
                           standoff_index=index)
        raise last

    def move_aligned(frame, hint, serial, checkpoint):
        # Capture provides the actual arrival pose. Routing reacquires its own
        # stopped pose/covariance together rather than reusing an earlier one.
        checkpoint("alignment_preflight", serial=serial)
        frame = fresh_frame(root / f"alignment_{serial:02d}" / "planning")
        context = load_uncertainty(frame)
        hints, _ = derive_lidar_inspection_hints(
            snapshot=frame.config.snapshot, registry=source_registry,
            planning_frame=frame.planning_frame,
            receipts=tuple(survey) + tuple(r for v in local_views for r in v.receipts),
            additional_viewpoint_ids=tuple(v.viewpoint_id for v in local_views),
            candidate_uids=(uid,),
        )
        hint = hints.get(uid)
        if hint is None:
            raise CandidateInspectionRouteUnavailableError("Head fit unavailable in fresh planning frame",
                                                          reason_code="head_geometry_unavailable")
        checkpoint("normal_route_planning", serial=serial)
        try:
            selected = plan_and_select_camera_candidate(
                map_yaml=frame.config.map_yaml, semantic_map_id=frame.config.semantic_map_id,
                plan=frame.config.plan, snapshot=frame.config.snapshot,
                current_pose=frame.planning_frame.current_pose, unresolved={uid},
                approach_offset_m=min(.65, max(.55, source_config.approach_offset_m)),
                inflation_radius_m=frame.config.inflation_radius_m,
                candidate_transit_radius_m=frame.config.candidate_transit_radius_m,
                physical_clearance=frame.config.physical_clearance,
                selection_config=CameraCandidateSelectionConfig(
                    source_config.camera_selection_linear_speed_mps,
                    source_config.camera_selection_angular_speed_radps, route_time_budget_enabled=True),
                route_uncertainty_context=context, lidar_inspection_hints={uid: hint},
                camera_calibration=source_config.camera_calibration,
                camera_alignment_uncertainty=_camera_alignment_uncertainty(context),
            )
        except (NoFeasibleCameraCandidateError, NoUncertaintyAdmittedCameraCandidateError) as exc:
            raise CandidateInspectionRouteUnavailableError(str(exc), reason_code="head_normal_route_unavailable") from exc
        prepared = selected.selected_plan
        if prepared.camera_alignment is None:
            raise CandidateInspectionRouteUnavailableError(
                "No normal-facing route satisfies the angular/clearance budget",
                reason_code="head_normal_route_unavailable", evidence=selected.to_evidence())
        return plan_and_move(frame, prepared.approach_bearing_rad-math.pi-frame.planning_frame.map_from_odom.yaw_rad,
            root / f"alignment_{serial:02d}", serial, None, purpose="lidar_axis_hint",
            offset=prepared.approach_offset_m, prepared_plan=prepared, selection_evidence=selected.to_evidence(),
            before_motion=lambda: checkpoint("motion_dispatch", kind="normal_alignment", serial=serial))

    controller = create_bounded_lidar_recovery(observe=observe, move_probe=move_probe,
        move_aligned=move_aligned, move_sampling=(move_sampling if getattr(effects, "run_lidar_sampling_turn", None) is not None else None),
        persist=persist, monotonic=monotonic, budget_sec=budget_sec)

    def recover(frame):
        nonlocal uid, step
        if uid is None:
            uid = frame.candidate.candidate_uid
        elif uid != frame.candidate.candidate_uid:
            raise ValueError("LiDAR recovery candidate binding changed")
        frame, report, hint = controller(frame)
        review_path = root / "steps" / f"step_{step:03d}" / "arrival_review.json"
        step += 1
        report = {**report, "candidate_uid": uid, "model_admission": model_review,
                  "candidate_snapshot_sha256": candidate_snapshot_sha256(frame.config.snapshot),
                  "candidate_frame_projection_path": str(frame.decision_binding.projection_path),
                  "candidate_frame_projection_sha256": frame.decision_binding.projection_sha256,
                  "arrival_review_path": str(review_path)}
        write_content_hashed_json(review_path, report, hash_field="lidar_alignment_arrival_sha256")
        return frame, report, hint

    return recover
