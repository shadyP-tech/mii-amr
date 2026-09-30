"""Bounded stopped LiDAR acquisition before camera inspection.

Local views add evidence only. Every displacement uses the existing candidate
route planner and motion gates; neither a fit nor this controller grants motion
authority or changes the frozen candidate registry.
"""
from __future__ import annotations

from dataclasses import replace
import math

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.camera_candidate_selection import CameraCandidateSelectionConfig, NoFeasibleCameraCandidateError
from scripts.aufgabe04.navigation.approach.candidate_arrival_admission import CandidateArrivalAdmissionConfig, evaluate_candidate_arrival_admission
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.candidate_preapproach_selection import plan_and_select_camera_candidate
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import NoUncertaintyAdmittedCameraCandidateError
from scripts.aufgabe04.navigation.approach.lidar_inspection_hint import derive_lidar_inspection_hints, fit_current_lidar_view
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


def run_bounded_lidar_acquisition(*, initial_frame, observe, move_probe, move_aligned, persist):
    """Pure control flow: failures cannot reset either finite motion budget.

    ``observe`` returns frame, multiview hint, current fit and arrival review.
    Motion callbacks may raise only a typed no-motion route-unavailable error
    to continue searching. Execution/interruption errors always propagate.
    """
    frame = initial_frame
    probes = alignments = 0
    history = []
    base_direction = None
    while True:
        frame, hint, current_fit, review = observe(frame, len(history))
        history.append({"event": "stopped_observation", **review})
        report = {"history": history, "probe_moves_attempted": probes,
                  "alignment_moves_attempted": alignments,
                  "head_alignment_verified": bool(review.get("head_alignment_verified")),
                  "camera_centered_verified": bool(review.get("camera_centered_verified")),
                  "motion_authorized": False, "stand_axis_authorized": False}
        persist(report)
        if (report["head_alignment_verified"] and report["camera_centered_verified"]) or review.get("acquisition_unavailable"):
            return frame, report
        if hint is not None:
            if alignments >= MAX_ALIGNMENT_MOVES:
                report["reason"] = "alignment_correction_budget_exhausted"
                persist(report)
                return frame, report
            alignments += 1
            kind = "normal_alignment"
            move = lambda: move_aligned(frame, hint, alignments)
        else:
            if probes >= MAX_PROBE_MOVES:
                report["reason"] = "independent_geometry_support_unavailable"
                persist(report)
                return frame, report
            pose = frame.planning_frame.current_pose
            target = frame.candidate.geometry
            if base_direction is None:
                base_direction = math.atan2(pose.y_m-target.y_m, pose.x_m-target.x_m) - frame.planning_frame.map_from_odom.yaw_rad
            direction = base_direction + (0., math.radians(60), -math.radians(60))[probes]
            if current_fit is not None:
                # A single supported tangent can improve the next measurement,
                # but never establish verified alignment on its own.
                normals = current_fit.normals(frame.config.snapshot, frame.candidate.candidate_uid)
                canonical = [n-frame.planning_frame.map_from_odom.yaw_rad for n in normals]
                nearest = min(canonical, key=lambda n: abs(math.remainder(n-base_direction, math.tau)))
                direction = nearest + (0., math.radians(30), -math.radians(30))[probes]
            probes += 1
            kind = "support_view"
            move = lambda: move_probe(frame, math.remainder(direction, math.tau), probes)
        history.append({"event": "motion_proposal", "kind": kind,
                        "probe_moves_attempted": probes, "alignment_moves_attempted": alignments})
        persist({**report, "history": history, "probe_moves_attempted": probes,
                 "alignment_moves_attempted": alignments})
        try:
            frame = move()
        except CandidateInspectionRouteUnavailableError as exc:
            history.append({"event": "no_motion_route_unavailable", "kind": kind,
                            "reason": exc.reason_code, "detail": str(exc)})


def prepare_lidar_camera_arrival(*, initial_frame, source_config, source_registry,
                                effects, candidate_root, fresh_frame, plan_and_move,
                                load_uncertainty):
    """Production adapter using bound cohorts and existing sealed motion effects."""
    from scripts.aufgabe04.real_robot.candidate.approach import _camera_alignment_uncertainty
    from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import (
        CandidateLidarCaptureRequest, CandidateLidarCaptureUnavailableError,
    )
    from scripts.aufgabe04.navigation.approach.lidar_alignment_arrival import verify_lidar_alignment_arrival
    from scripts.aufgabe04.real_robot.readiness.tour_scan_capture import TourScanCaptureError

    uid = initial_frame.candidate.candidate_uid
    root = candidate_root / "lidar_head_acquisition"
    model_review = lidar_head_model_admission(getattr(source_config, "measured_stand_model", None))
    if not model_review["accepted"]:
        report = {"head_alignment_verified": False, "camera_centered_verified": False,
                  "reason": model_review["reason"], "model_admission": model_review,
                  "candidate_uid": uid, "history": [], "probe_moves_attempted": 0,
                  "alignment_moves_attempted": 0, "motion_authorized": False,
                  "stand_axis_authorized": False}
        write_content_hashed_json(root / "arrival_review.json", report, hash_field="lidar_alignment_arrival_sha256")
        return initial_frame, report, None
    survey, failures = load_camera_lidar_receipts(
        survey_root=source_config.survey_root, plan=source_config.plan,
        snapshot=source_config.snapshot, registry=source_registry,
    )
    local_views = []
    last_hint = None

    def persist(report):
        # A separate append-only revision preserves consumed proposals on failure.
        path = root / "history" / f"revision_{persist.serial:03d}.json"
        persist.serial += 1
        write_content_hashed_json(path, {**report, "candidate_uid": uid,
            "source_candidate_snapshot_sha256": candidate_snapshot_sha256(source_config.snapshot),
            "unavailable_survey_epochs": failures}, hash_field="lidar_acquisition_sha256")
    persist.serial = 0

    def observe(frame, serial):
        nonlocal last_hint
        frame = fresh_frame(root / f"view_{serial:02d}" / "planning")
        route_context = load_uncertainty(frame)
        uncertainty = _camera_alignment_uncertainty(route_context)
        floor = effects.clock()
        request = CandidateLidarCaptureRequest(
            plan=source_config.plan,
            candidate_snapshot_sha256=candidate_snapshot_sha256(source_config.snapshot),
            candidate_uid=uid, viewpoint_id=f"local_{uid}_{serial:02d}",
            output_dir=root / f"view_{serial:02d}" / "capture",
            observation_not_before_sec=floor, planning_frame=frame.planning_frame,
            base_frame=source_config.camera_calibration.base_frame,
            scan_frame=source_config.lidar_scan_frame, scan_topic=source_config.lidar_scan_topic,
        )
        try:
            captured = effects.capture_lidar_view(request)
        except (TourScanCaptureError, CandidateLidarCaptureUnavailableError) as exc:
            return frame, None, None, {"head_alignment_verified": False,
                "reason": "fresh_scan_cohort_unavailable", "detail": str(exc),
                "acquisition_unavailable": True}
        if (captured.candidate_uid != uid or captured.candidate_snapshot_sha256 != request.candidate_snapshot_sha256
                or captured.viewpoint_id != request.viewpoint_id):
            raise ValueError("local LiDAR capture candidate binding mismatch")
        local_views.append(captured)
        receipts = tuple(survey) + tuple(r for v in local_views for r in v.receipts)
        hints, diagnostics = derive_lidar_inspection_hints(
            snapshot=frame.config.snapshot, registry=source_registry,
            planning_frame=frame.planning_frame, receipts=receipts,
            additional_viewpoint_ids=tuple(v.viewpoint_id for v in local_views),
            candidate_uids=(uid,),
        )
        hint = hints.get(uid)
        last_hint = hint
        current_fit = fit_current_lidar_view(snapshot=frame.config.snapshot, registry=source_registry,
            planning_frame=frame.planning_frame, candidate_uid=uid, receipts=captured.receipts)
        review = {"head_alignment_verified": False, "reason": "insufficient_head_geometry",
                  "fit_diagnostics": diagnostics.get(uid), "capture_evidence_path": str(captured.evidence_path)}
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
            mount_evidence=captured.mount_evidence, target_range_m=range_bound,
        )
        if mount_review["accepted"] and mount_review["source_scan_stamps_sec"] != [r.scan_stamp_sec for r in captured.receipts]:
            raise ValueError("LiDAR head plane evidence differs from captured scan stamps")
        review["head_observability"] = mount_review
        if not mount_review["accepted"]:
            last_hint = None
            return frame, None, None, {**review, "reason": mount_review["reason"],
                                       "acquisition_unavailable": True}
        tf, p = frame.planning_frame.map_from_odom, captured.latest_base_pose_odom
        c, s = math.cos(tf.yaw_rad), math.sin(tf.yaw_rad)
        actual = Pose2D(tf.x_m+c*p.x_m-s*p.y_m, tf.y_m+s*p.x_m+c*p.y_m,
                        math.remainder(tf.yaw_rad+p.yaw_rad, math.tau))
        actual_frame = CandidatePlanningFrame(actual, tf, frame.planning_frame.map_frame,
                                              frame.planning_frame.odom_frame)
        frame = replace(frame, planning_frame=actual_frame, observation_pose=actual)
        if hint is not None and current_fit is not None and uncertainty is not None:
            review = {**review, **verify_lidar_alignment_arrival(
                hint=hint, current_fit=current_fit, current_receipts=captured.receipts,
                snapshot=frame.config.snapshot, candidate_uid=uid,
                planning_frame=actual_frame, calibration=source_config.camera_calibration,
                now_sec=effects.clock(), not_before_sec=floor,
                current_base_pose_odom=p, pose_stamp_sec=captured.pose_stamp_sec,
                localization_position_bound_m=uncertainty["localization_position_m"],
                localization_yaw_bound_rad=uncertainty["localization_yaw_rad"],
            )}
            range_review = evaluate_candidate_arrival_admission(
                actual, target_x_m=current_fit.center_x_m, target_y_m=current_fit.center_y_m,
                config=CandidateArrivalAdmissionConfig(
                    min_range_m=source_config.physical_clearance["minimum_active_standoff_m"] + current_fit.center_uncertainty_m,
                    max_range_m=max(.65, source_config.approach_offset_m) + source_config.camera_arrival_range_slack_m,
                    max_bearing_error_rad=math.pi,
                ),
            ).to_evidence_dict()
            review["physical_range_admission"] = range_review
            if not range_review["accepted"]:
                review.update(head_alignment_verified=False, accepted=False,
                              reason="fitted_head_arrival_range_rejected")
        return frame, hint, current_fit, review

    def move_probe(frame, normal, serial):
        last = None
        for index, offset in enumerate(PROBE_STANDOFFS_M):
            try:
                return plan_and_move(frame, normal, root / f"probe_{serial:02d}_{index:02d}",
                                     serial, None, purpose="lidar_axis_hint", offset=offset)
            except CandidateInspectionRouteUnavailableError as exc:
                last = exc
        raise last

    def move_aligned(frame, hint, serial):
        # Capture provides the actual arrival pose. Routing reacquires its own
        # stopped pose/covariance together rather than reusing an earlier one.
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
            offset=prepared.approach_offset_m, prepared_plan=prepared, selection_evidence=selected.to_evidence())

    frame, report = run_bounded_lidar_acquisition(initial_frame=initial_frame, observe=observe,
        move_probe=move_probe, move_aligned=move_aligned, persist=persist)
    report = {**report, "candidate_uid": uid,
              "model_admission": model_review,
              "candidate_snapshot_sha256": candidate_snapshot_sha256(frame.config.snapshot),
              "candidate_frame_projection_path": str(frame.decision_binding.projection_path),
              "candidate_frame_projection_sha256": frame.decision_binding.projection_sha256}
    write_content_hashed_json(root / "arrival_review.json", report, hash_field="lidar_alignment_arrival_sha256")
    return frame, report, last_hint
