"""Missing visibility permits observation routes, never precise target motion."""
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.candidate_preapproach_selection import plan_and_select_camera_candidate
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.perception.lidar_scan_metadata import LidarScanMetadata
from scripts.aufgabe04.real_robot.candidate import approach
from scripts.aufgabe04.real_robot.candidate.current_lidar_targets import (
    HASH_FIELD, assess_current_lidar_targets, capture_current_lidar_targets,
    load_current_lidar_assessment, permits_survey_observation,
)
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import capture_candidate_lidar_view
from scripts.aufgabe04.real_robot.candidate.lidar_head_capture import HASH_FIELD as CAPTURE_HASH, head_capture_payload
from scripts.aufgabe04.real_robot.candidate.target_admission import (
    SURVEY_OBSERVATION_POLICY, bind_survey_observation_target, require_frame_target,
)
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome
from tests.aufgabe04.test_current_lidar_targets import fixture, raw_scans
from tests.aufgabe04 import test_current_lidar_approach_integration as integration_fixtures
from tests.aufgabe04 import test_lidar_inspection_planning as planning_fixtures

SENSOR = "scripts.aufgabe04.real_robot.candidate.current_lidar_targets"


@pytest.fixture
def integration():
    helper = integration_fixtures.CurrentLidarApproachIntegrationTests()
    helper.setUp()
    try:
        yield helper
    finally:
        helper.doCleanups()


def capture_support(config, planning, root, *, kind="occluded"):
    """Real immutable acquisition/replay; only the ROS sensor is substituted."""
    transform = planning.map_from_odom
    assert transform.yaw_rad == 0.
    base_x = planning.current_pose.x_m-transform.x_m
    base_y = planning.current_pose.y_m-transform.y_m
    scan_x = base_x+.04
    target_x = config.snapshot.candidates[0].geometry.x_m-transform.x_m
    def capture(request):
        scans = raw_scans(kinds=[kind]*8, shifts=[target_x-scan_x-1.]*8)
        for scan in scans:
            scan["scan_pose_odom"] = {"x_m": scan_x, "y_m": base_y, "yaw_rad": 0.}
            scan["base_pose_odom"] = {"x_m": base_x, "y_m": base_y, "yaw_rad": 0.}
            if kind == "missing":
                scan["ranges"] = [None]*len(scan["ranges"])
                scan["ranges"][0] = 2.  # Healthy sensor; only the target cone is empty.
                scan["scan_metadata"] = LidarScanMetadata(.5, 0., .1, "linear", (),
                    tuple("nan" if value is None else None for value in scan["ranges"])).to_mapping()
        payload = head_capture_payload(scans, tour_id=request.viewpoint_id,
            odom_frame="odom", base_frame=request.base_frame, scan_frame=request.scan_frame,
            captured_at_unix_sec=100.71)
        path = request.output_dir / "cohort.json"
        write_content_hashed_json(path, payload, hash_field=CAPTURE_HASH)
        clock = iter((100., 100.72))
        return capture_candidate_lidar_view(request, capture_cohort=lambda _: path,
            clock=lambda: next(clock))
    clock = iter((100., 100.72))
    return capture_current_lidar_targets(config,
        SimpleNamespace(clock=lambda: next(clock), capture_lidar_view=capture),
        planning, set(config.snapshot.candidate_uids), root)


def source(integration, *, kind="occluded"):
    config = integration.config()
    config = replace(config, camera_calibration=replace(config.camera_calibration, base_frame="base_footprint"))
    planning = integration.frame(-.04)
    artifacts = approach._materialize_candidate_frame_projection(source_config=config,
        source_registry=integration.registry(config), planning_frame=planning,
        output_root=integration.root / "source_projection")
    _, support = capture_support(artifacts.config, planning, integration.root / "support", kind=kind)
    frame = approach._CandidateObservationFrame(artifacts.config, artifacts.config.snapshot.candidates[0],
        planning, artifacts.camera_decision_binding(), observation_pose=planning.current_pose)
    return config, bind_survey_observation_target(frame, evidence_path=Path(support["evidence_path"])), support


@pytest.mark.parametrize("kind,eligible", [("occluded", True), ("wall", False),
    ("absent", False), ("ambiguous", False), ("stand", False)])
def test_only_missing_visibility_allows_survey_route(kind, eligible):
    _, evidence = assess_current_lidar_targets(**fixture(kinds=[kind]*8))
    assert permits_survey_observation(evidence["candidate_decisions"]["candidate_1"]) is eligible


def test_competing_candidate_and_unstable_or_incomplete_support_never_fall_back():
    _, evidence = assess_current_lidar_targets(**fixture(kinds=["occluded"]*8))
    decision = evidence["candidate_decisions"]["candidate_1"]
    for reason in ("competing_candidate", "ambiguous_clusters", "non_stand_cluster", "unsupported"):
        changed = {**decision, "scans": [{**decision["scans"][0], "reason": reason}, *decision["scans"][1:]]}
        assert not permits_survey_observation(changed)
    assert not permits_survey_observation({**decision, "scans": decision["scans"][:-1]})
    for reason in ("current_cluster_centers_unstable", "current_target_geometry_exceeds_bound",
                   "laser_plane_not_inside_measured_head"):
        assert not permits_survey_observation({**decision, "reasons": [*decision["reasons"], reason]})


@pytest.mark.parametrize("kind", ["occluded", "missing"])
def test_real_missing_cohort_replays_and_camera_arrival_uses_executed_survey_target(integration, kind):
    config, frame, support = source(integration, kind=kind)
    path = Path(support["evidence_path"])
    original = path.read_bytes()
    estimates, replay = load_current_lidar_assessment(path, snapshot=frame.config.snapshot)
    assert not estimates and permits_survey_observation(replay["candidate_decisions"]["candidate_0"])
    cameras = []
    effects = integration.effects(capture_observation=lambda request: cameras.append(request) or
        approach.CandidateObservation(request.output_dir / "recommendation.json", "QR_A", None))
    with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError("passive arrival rescan")):
        _, arrived = approach._capture_candidate_camera_result(observation_frame=frame,
            source_config=config, effects=effects, source_registry=integration.registry(config),
            candidate_root=integration.root / "arrival", candidate_run_id="survey_observation", candidate_index=0)
    assert len(cameras) == 1
    assert arrived.camera_target_geometry == arrived.candidate.geometry
    assert arrived.retained_survey_target == frame.retained_survey_target
    assert arrived.retained_lidar_target is None and arrived.current_lidar_target_path is None
    assert path.read_bytes() == original
    proof = json.loads(arrived.camera_target_geometry_evidence_path.read_text())
    assert proof["source_kind"] == SURVEY_OBSERVATION_POLICY
    assert proof["retained_survey_target"]["precise_motion_authorized"] is False


def test_survey_target_reprojects_original_geometry_and_rejects_tampering(integration):
    config, frame, support = source(integration)
    for index, angle in enumerate((.4, -.25)):
        tf = PlanarTransform2D(.2, -.1, angle)
        planning = CandidatePlanningFrame(Pose2D(.2+.3*math.cos(angle), -.1+.3*math.sin(angle), angle), tf)
        arrived = approach._admit_camera_arrival_geometry(source_config=config,
            effects=integration.effects(admit_planning_frame=lambda _: planning),
            source_registry=integration.registry(config), candidate_uid="candidate_0",
            candidate_root=integration.root / f"arrival_{index}", observation_attempt_index=0,
            target_source_frame=frame)
        assert arrived.camera_target_geometry.x_m == pytest.approx(.2+math.cos(angle))
        assert arrived.camera_target_geometry.y_m == pytest.approx(-.1+math.sin(angle))
        assert arrived.retained_survey_target.source_snapshot == frame.retained_survey_target.source_snapshot
        frame = arrived
    altered = replace(frame, camera_target_geometry=replace(frame.camera_target_geometry, x_m=.2))
    with pytest.raises(ValueError, match="geometry/provenance"):
        require_frame_target(altered, evidence_path=integration.root / "bad.json", attempt_index=0)
    path = Path(support["evidence_path"])
    payload = json.loads(path.read_text()); payload.pop(HASH_FIELD)
    payload["candidate_decisions"]["candidate_0"]["scans"][0]["reason"] = "unsupported"
    path.unlink(); write_content_hashed_json(path, payload, hash_field=HASH_FIELD)
    with pytest.raises(ValueError, match="source replay"):
        require_frame_target(frame, evidence_path=integration.root / "tampered.json", attempt_index=0)


def test_sparse_refinement_preserves_both_route_candidates_and_all_keepouts(tmp_path):
    kwargs, _ = planning_fixtures.LidarInspectionPlanningTest().fixture(tmp_path)
    first = kwargs["snapshot"].candidates[0]
    second = replace(first, candidate_uid="candidate_2", geometry=replace(first.geometry, y_m=.8),
                     source=replace(first.source, observation_ids=("observation_candidate_2",)))
    snapshot = replace(kwargs["snapshot"], candidates=(first, second))
    estimate = {"x_m": .52, "y_m": .8, "uncertainty_m": .08, "policy": "current_stopped_lidar_surface"}
    kwargs.update(snapshot=snapshot, unresolved=set(snapshot.candidate_uids),
        current_target_estimates={"candidate_2": estimate})
    seen = []
    from scripts.aufgabe04.navigation.approach.candidate_preapproach_compute import compute_candidate_preapproach_plan
    def compute(**arguments):
        prepared = compute_candidate_preapproach_plan(**arguments)
        seen.append(prepared)
        for candidate in snapshot.candidates:
            cell = prepared.dry_run.planning_costmap.world_to_grid(candidate.geometry.x_m, candidate.geometry.y_m)
            assert cell in prepared.dry_run.planning_costmap.blocked_cells
        return prepared
    with patch("scripts.aufgabe04.navigation.approach.candidate_preapproach_selection.compute_candidate_preapproach_plan",
               side_effect=compute):
        selection = plan_and_select_camera_candidate(**kwargs)
    plans = {p.candidate_uid: p for p in seen}
    assert set(plans) == set(snapshot.candidate_uids)
    assert plans["candidate_1"].approach_bearing_mode == "robot-to-stand"
    assert plans["candidate_1"].camera_alignment is None
    assert plans["candidate_2"].validated_target_center == estimate
    assert set(selection.to_evidence()["candidate_target_admission"]["eligible_candidate_uids"]) == set(snapshot.candidate_uids)


def test_occluded_selection_seals_existing_safe_observation_route(integration):
    config = integration.config()
    config = replace(config, camera_calibration=replace(config.camera_calibration, base_frame="base_footprint"))
    effects = integration.effects(admit_planning_frame=lambda _: integration.frame(-.04),
        select_initial_preapproach=approach._select_initial_preapproach)
    with patch(SENSOR + ".capture_current_lidar_targets", side_effect=lambda cfg, effects, frame, uids, root:
               capture_support(cfg, frame, root)):
        selected_config, start, selection, planning = approach._select_initial_candidate_with_localization_refresh(
            config=config, effects=effects, source_registry=integration.registry(config), candidate_index=0,
            eligible={"candidate_0"}, exact_two_support_by_uid=None)
    assert selection.prepared_plan.approach_bearing_mode == "robot-to-stand"
    assert selection.evidence["observation_target_source"] == SURVEY_OBSERVATION_POLICY
    request = approach.CandidatePreapproachRequest(selected_config.map_yaml, selected_config.semantic_map_id,
        selected_config.plan, selected_config.snapshot, selected_config.snapshot_path, "candidate_0", start,
        integration.root / "route", config.approach_offset_m, config.inflation_radius_m,
        config.candidate_transit_radius_m, config.physical_clearance,
        prepared_plan=selection.prepared_plan, selection_evidence=selection.evidence,
        survey_observation_support_path=Path(selection.evidence["current_lidar_support"]["evidence_path"]))
    sealed = approach._plan_preapproach_from_request(request)
    assert Path(sealed["route_csv"]).is_file()
    assert (request.output_dir / "preapproach_execution" / "route_certificate.json").is_file()
    completed = []
    approach._execute_candidate_motion(config=selected_config,
        effects=integration.effects(run_motion_leg=integration.fixtures._completed),
        candidate_root=integration.root, plan_request=request, initial_sealed=sealed,
        run_id="observation_only", leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
        candidate_index=0, target_id="candidate_0", plan_planning_frame=planning,
        completed_frame_sink=completed.append)
    assert completed[0].retained_survey_target is not None
    for forbidden in (replace(request, approach_normal_rad=0.),
                      replace(request, axis_observation_path=integration.root / "axis.json"),
                      replace(request, start=Pose2D(.2, 0., 0.))):
        with pytest.raises(ValueError):
            approach._plan_preapproach_from_request(forbidden)
    with pytest.raises(ValueError, match="precise or opposite"):
        approach._execute_candidate_motion(config=selected_config, effects=integration.effects(),
            candidate_root=integration.root, plan_request=request, initial_sealed=sealed,
            run_id="wrong_purpose", leg_kind=MissionLegKind.OPPOSITE_FACE,
            candidate_index=0, target_id="candidate_0", plan_planning_frame=planning)


@pytest.mark.parametrize("replacement_kind", ["occluded", "stand", "absent", "wall", "ambiguous"])
def test_replacement_uses_its_own_fresh_target_and_never_reuses_the_old_survey_receipt(integration, replacement_kind):
    config, frame, old_support = source(integration)
    config = replace(config, max_startup_reseals_per_leg=1)
    planned_config = replace(frame.config, max_startup_reseals_per_leg=1)
    fresh = CandidatePlanningFrame(Pose2D(.26, 0., 0.), PlanarTransform2D(.2, 0., 0.))
    root = integration.root / "replacement"
    plans, completed_frames, captures = [], [], []
    def capture(cfg, effects, planning, uids, output):
        result = capture_support(cfg, planning, output, kind=replacement_kind)
        captures.append(result[1])
        return result
    def admit(path):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}")
        return fresh
    def initial(request):
        return MotionLegOutcome(run_id=request.run_id, status="stopped",
            stop_reason="pose outside certified startup segment",
            stop_details={"source": "execution_route_certificate", "phase": "before_motion_confirmation",
                "reason": "pose outside certified startup segment", "fail_closed": True,
                "route_pose": {"x_m": .26, "y_m": 0., "yaw_rad": 0.}},
            motion_published=False, returncode=1, semantic_log_path=root / "initial.jsonl")
    def plan(request):
        plans.append(request)
        if replacement_kind == "occluded":
            approach._validate_survey_observation_request(request, fresh)
        return {"route_csv": "replacement.csv"}
    effects = integration.effects(admit_planning_frame=admit, run_motion_leg=initial,
        plan_preapproach=plan,
        run_startup_reseal_motion_leg=lambda request, attempt: integration.fixtures._startup_completed(request),
        wait_for_lidar_reacquisition=lambda _: None)
    request = approach.CandidatePreapproachRequest(planned_config.map_yaml, planned_config.semantic_map_id,
        planned_config.plan, planned_config.snapshot, planned_config.snapshot_path, "candidate_0", frame.planning_frame.current_pose,
        root / "initial_route", config.approach_offset_m, config.inflation_radius_m,
        config.candidate_transit_radius_m, config.physical_clearance,
        survey_observation_support_path=Path(old_support["evidence_path"]))
    kwargs = dict(config=planned_config, effects=effects, candidate_root=root, plan_request=request,
        initial_sealed={"route_csv": "initial.csv"}, run_id="survey_replacement",
        leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH, candidate_index=0, target_id="candidate_0",
        frame_source_config=config, source_registry=integration.registry(config),
        plan_planning_frame=frame.planning_frame, completed_frame_sink=completed_frames.append)
    with patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture):
        if replacement_kind in {"absent", "wall", "ambiguous"}:
            from scripts.aufgabe04.real_robot.candidate.recovery_failure import CandidateStartupRecoveryError
            with pytest.raises(CandidateStartupRecoveryError):
                approach._execute_candidate_motion(**kwargs)
            assert not plans and not completed_frames
            return
        outcome = approach._execute_candidate_motion(**kwargs)
    assert outcome.status == "completed" and len(plans) == len(completed_frames) == 1
    completed = completed_frames[0]
    assert completed.planning_frame == fresh
    assert completed.camera_target_geometry.x_m == pytest.approx(1.2)
    if replacement_kind == "occluded":
        assert plans[0].survey_observation_support_path == Path(captures[0]["evidence_path"])
        assert completed.retained_survey_target.evidence_path != Path(old_support["evidence_path"])
        assert completed.retained_survey_target.source_snapshot == plans[0].snapshot
        assert completed.retained_lidar_target is None
    else:
        assert plans[0].survey_observation_support_path is None
        assert completed.retained_survey_target is None
        assert completed.retained_lidar_target.evidence_path == Path(captures[0]["evidence_path"])


def test_full_phase_preserves_occluded_candidate_until_camera_observation(integration):
    config = integration.config()
    config = replace(config, camera_calibration=replace(config.camera_calibration, base_frame="base_footprint"))
    frames = iter((integration.frame(-.04), integration.frame(.3)))
    order = []
    def capture(cfg, effects, planning, uids, output):
        order.append("scan")
        return capture_support(cfg, planning, output)
    def camera(request):
        order.append("camera")
        return approach.CandidateObservation(request.output_dir / "recommendation.json", "QR_A", None)
    effects = integration.effects(admit_planning_frame=lambda _: next(frames),
        select_initial_preapproach=approach._select_initial_preapproach,
        plan_preapproach=approach._plan_preapproach_from_request,
        run_motion_leg=lambda request: order.append("motion") or integration.fixtures._completed(request),
        capture_observation=camera,
        validate_facing=lambda request: {"candidate_uid": request.candidate.candidate_uid},
        commit_decision=lambda request: None)
    with patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture):
        result = approach.execute_candidate_approach_phase(config, effects)
    assert result.stand_count == 1 and order == ["scan", "motion", "camera"]
    assert not any(event.get("event") == "camera_candidate_target_deferred" for event in integration.events)
