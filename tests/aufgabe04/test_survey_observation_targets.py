"""Survey hypotheses grant observation approaches; local scans refine arrival."""
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
    capture_current_lidar_targets,
)
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import capture_candidate_lidar_view
from scripts.aufgabe04.real_robot.candidate.lidar_head_capture import HASH_FIELD as CAPTURE_HASH, head_capture_payload
from scripts.aufgabe04.real_robot.candidate.target_admission import (
    SURVEY_OBSERVATION_POLICY, SURVEY_OBSERVATION_HASH_FIELD,
    bind_survey_observation_target, load_survey_observation_support,
    require_frame_target, write_survey_observation_support,
)
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome
from tests.aufgabe04.test_current_lidar_targets import raw_scans
from tests.aufgabe04.test_autonomous_candidate_runtime_recovery import _runtime_stop, _outcome
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


def capture_support(config, planning, root, *, kind="occluded", lateral_shift=0.):
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
            if lateral_shift:
                assert kind == "stand"
                distance_x = target_x-scan_x
                scan["ranges"] = [distance_x/math.cos(angle) if
                    abs(distance_x*math.tan(angle)-lateral_shift) <= .036 else None
                    for angle in (scan["angle_min"]+i*scan["angle_increment"]
                                  for i in range(len(scan["ranges"])))]
                scan["scan_metadata"] = LidarScanMetadata(.5, 0., .1, "linear", (),
                    tuple("nan" if value is None else None for value in scan["ranges"])).to_mapping()
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


def source(integration):
    config = integration.config()
    config = replace(config, camera_calibration=replace(config.camera_calibration, base_frame="base_footprint"))
    planning = integration.frame(-.04)
    artifacts = approach._materialize_candidate_frame_projection(source_config=config,
        source_registry=integration.registry(config), planning_frame=planning,
        output_root=integration.root / "source_projection")
    path = integration.root / "support" / "survey_observation.json"
    support = write_survey_observation_support(path, config=artifacts.config,
        planning_frame=planning, candidate_uid="candidate_0")
    frame = approach._CandidateObservationFrame(artifacts.config, artifacts.config.snapshot.candidates[0],
        planning, artifacts.camera_decision_binding(), observation_pose=planning.current_pose)
    return config, bind_survey_observation_target(frame, evidence_path=path), path, support


def test_survey_proof_needs_no_scan_and_preserves_observation_only_authority(integration):
    with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError("survey proof rescan")):
        _, frame, path, proof = source(integration)
    replay = load_survey_observation_support(path, candidate_uid="candidate_0", snapshot=frame.config.snapshot)
    assert replay == proof
    assert replay["source_kind"] == "survey_candidate_snapshot"
    assert replay["purpose"] == "observation_only"
    assert replay["precise_motion_authorized"] is False
    assert replay["keepouts_changed"] is False


@pytest.mark.parametrize("kind", ["occluded", "missing", "absent", "wall", "ambiguous"])
def test_weak_local_cohort_preserves_survey_target_and_still_starts_camera(integration, kind):
    config, frame, path, _ = source(integration)
    original = path.read_bytes()
    order = []
    effects = integration.effects(capture_observation=lambda request: cameras.append(request) or
        approach.CandidateObservation(request.output_dir / "recommendation.json", "QR_A", None))
    cameras = []
    def capture(cfg, effects, planning, uids, root):
        order.append("scan")
        assert len(order) == 1
        return capture_support(cfg, planning, root, kind=kind)
    with patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture) as scans:
        _, arrived = approach._capture_candidate_camera_result(observation_frame=frame,
            source_config=config, effects=effects, source_registry=integration.registry(config),
            candidate_root=integration.root / "arrival", candidate_run_id="survey_observation", candidate_index=0)
    assert len(cameras) == 1
    scans.assert_called_once()
    assert arrived.camera_target_geometry == arrived.candidate.geometry
    assert arrived.retained_survey_target == frame.retained_survey_target
    assert arrived.retained_lidar_target is None and arrived.current_lidar_target_path is None
    assert path.read_bytes() == original
    proof = json.loads(arrived.camera_target_geometry_evidence_path.read_text())
    assert proof["source_kind"] == SURVEY_OBSERVATION_POLICY
    assert proof["retained_survey_target"]["precise_motion_authorized"] is False
    # A later passive arrival refresh keeps the same target without another
    # opportunity to scan or refine it, even while it remains survey-bound.
    with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError("passive refresh rescan")):
        refreshed = approach._admit_camera_arrival_geometry(source_config=config,
            effects=integration.effects(), source_registry=integration.registry(config), candidate_uid="candidate_0",
            candidate_root=integration.root / "passive_refresh", observation_attempt_index=0,
            target_source_frame=arrived)
    assert refreshed.retained_survey_target == arrived.retained_survey_target


def test_survey_target_reprojects_original_geometry_and_rejects_tampering(integration):
    config, frame, path, _ = source(integration)
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
    payload = json.loads(path.read_text()); payload.pop(SURVEY_OBSERVATION_HASH_FIELD)
    payload["planning_frame"]["current_pose"]["x_m"] += .01
    path.unlink(); write_content_hashed_json(path, payload, hash_field=SURVEY_OBSERVATION_HASH_FIELD)
    with pytest.raises(ValueError, match="source binding"):
        require_frame_target(frame, evidence_path=integration.root / "tampered.json", attempt_index=0)


def test_valid_local_point_outside_acquisition_cone_cannot_veto_reached_camera_view(integration):
    config, source_frame, _, _ = source(integration)
    cameras, supports = [], []
    def capture(cfg, effects, planning, uids, output):
        result = capture_support(cfg, planning, output, kind="stand", lateral_shift=.12)
        assert "candidate_0" in result[0]
        supports.append(result[1])
        return result
    def turn(**kwargs):
        pytest.fail("optional target refinement must not cause a turn")
    effects = integration.effects(admit_planning_frame=lambda _: integration.frame(.5),
        run_centering_turn=turn, capture_observation=lambda request: cameras.append(request) or
            approach.CandidateObservation(request.output_dir / "recommendation.json", "QR_A", None))
    root = integration.root / "lateral_refinement"
    with patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture) as scans:
        _, arrived = approach._capture_candidate_camera_result(observation_frame=source_frame,
            source_config=config, effects=effects, source_registry=integration.registry(config),
            candidate_root=root, candidate_run_id="lateral_refinement", candidate_index=0)
    scans.assert_called_once()
    assert len(cameras) == 1
    assert arrived.camera_target_geometry == arrived.candidate.geometry
    assert arrived.retained_survey_target is source_frame.retained_survey_target
    assert arrived.retained_lidar_target is None
    evidence = json.loads((root / "candidate_arrival_admission.json").read_text())
    refinement = evidence["survey_target_refinement"]
    assert evidence["accepted"] and refinement["accepted"]
    assert refinement["adopted_for_camera_arrival"] is False
    rejection = refinement["arrival_rejection"]
    assert rejection["reasons"] == ["bearing_error_above_maximum"]
    assert rejection["measurements"]["absolute_bearing_error_rad"] > math.radians(10.)
    assert Path(supports[0]["evidence_path"]).is_file()


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


def test_explicit_survey_selection_suppresses_existing_precise_lidar_hints(tmp_path):
    kwargs, _ = planning_fixtures.LidarInspectionPlanningTest().fixture(tmp_path)
    assert kwargs["lidar_inspection_hints"]
    kwargs["current_target_estimates"] = {}
    selection = plan_and_select_camera_candidate(**kwargs)
    plan = selection.selected_plan
    assert plan.approach_bearing_mode == "robot-to-stand"
    assert plan.validated_target_center is None
    assert plan.camera_alignment is None


def test_survey_selection_seals_safe_observation_route_without_fresh_head_support(integration):
    config = integration.config()
    config = replace(config, camera_calibration=replace(config.camera_calibration, base_frame="base_footprint"))
    effects = integration.effects(admit_planning_frame=lambda _: integration.frame(-.04),
        select_initial_preapproach=approach._select_initial_preapproach)
    with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError("initial selection rescan")):
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
        survey_observation_support_path=Path(selection.evidence["survey_observation_support_path"]))
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
                      replace(request, start=Pose2D(.2, 0., 0.)),
                      replace(request, plan=replace(request.plan, survey_id="different_survey"))):
        with pytest.raises(ValueError):
            approach._plan_preapproach_from_request(forbidden)
    with pytest.raises(ValueError, match="precise or opposite"):
        approach._execute_candidate_motion(config=selected_config, effects=integration.effects(),
            candidate_root=integration.root, plan_request=request, initial_sealed=sealed,
            run_id="wrong_purpose", leg_kind=MissionLegKind.OPPOSITE_FACE,
            candidate_index=0, target_id="candidate_0", plan_planning_frame=planning)


@pytest.mark.parametrize("owner", ["startup", "runtime", "startup_runtime"])
def test_survey_reseal_refreshes_frame_proof_without_scan_or_precise_target_promotion(integration, owner):
    config, frame, old_path, _ = source(integration)
    authorization = integration.root / "authorization.json"
    authorization.write_text("{}")
    changes = dict(max_startup_reseals_per_leg=1,
        max_runtime_localization_reseals_per_leg=0 if owner == "startup" else 1,
        mission_motion_authorization_json=authorization)
    config = replace(config, **changes)
    planned_config = replace(frame.config, **changes)
    fresh = CandidatePlanningFrame(Pose2D(.26, 0., 0.), PlanarTransform2D(.2, 0., 0.))
    root = integration.root / "replacement"
    plans, completed_frames, replacements = [], [], []
    def admit(path):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}")
        return fresh
    def initial(request):
        if owner == "runtime":
            stopped = _runtime_stop(root, request.run_id)
            return replace(stopped,
                mission_leg_motion_permit_path=stopped.startup_reseal_motion_permit_path,
                mission_leg_motion_permit_sha256=stopped.startup_reseal_motion_permit_sha256,
                startup_reseal_motion_permit_path=None, startup_reseal_motion_permit_sha256="")
        return MotionLegOutcome(run_id=request.run_id, status="stopped",
            stop_reason="pose outside certified startup segment",
            stop_details={"source": "execution_route_certificate", "phase": "before_motion_confirmation",
                "reason": "pose outside certified startup segment", "fail_closed": True,
                "route_pose": {"x_m": .26, "y_m": 0., "yaw_rad": 0.}},
            motion_published=False, returncode=1, semantic_log_path=root / "initial.jsonl")
    def plan(request):
        plans.append(request)
        approach._validate_survey_observation_request(request, fresh)
        assert request.prepared_plan is None
        assert request.inspection_view_path is None
        assert request.axis_observation_path is None
        return {"route_csv": "replacement.csv"}
    def replacement(request, attempt):
        replacements.append(request)
        if owner == "startup_runtime" and len(replacements) == 1:
            return _runtime_stop(root, request.run_id)
        if owner == "startup":
            return integration.fixtures._startup_completed(request)
        return _outcome(root, run_id=request.run_id, status="completed", motion_published=True,
            permit_name="runtime_replacement.json", permit_digest="d" * 64)
    effects = integration.effects(admit_planning_frame=admit, run_motion_leg=initial,
        plan_preapproach=plan, run_startup_reseal_motion_leg=replacement,
        admit_runtime_localization=lambda path: fresh.current_pose,
        run_runtime_localization_reseal_motion_leg=replacement,
        wait_for_lidar_reacquisition=lambda _: None)
    request = approach.CandidatePreapproachRequest(planned_config.map_yaml, planned_config.semantic_map_id,
        planned_config.plan, planned_config.snapshot, planned_config.snapshot_path, "candidate_0", frame.planning_frame.current_pose,
        root / "initial_route", config.approach_offset_m, config.inflation_radius_m,
        config.candidate_transit_radius_m, config.physical_clearance,
        survey_observation_support_path=old_path)
    kwargs = dict(config=planned_config, effects=effects, candidate_root=root, plan_request=request,
        initial_sealed={"route_csv": "initial.csv"}, run_id="survey_replacement",
        leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH, candidate_index=0, target_id="candidate_0",
        frame_source_config=config, source_registry=integration.registry(config),
        plan_planning_frame=frame.planning_frame, completed_frame_sink=completed_frames.append)
    with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError("survey reseal rescan")):
        outcome = approach._execute_candidate_motion(**kwargs)
    assert outcome.status == "completed" and len(completed_frames) == 1
    assert len(plans) == (2 if owner == "startup_runtime" else 1)
    completed = completed_frames[0]
    assert completed.planning_frame == fresh
    assert completed.camera_target_geometry.x_m == pytest.approx(1.2)
    assert completed.retained_survey_target.evidence_path != old_path
    assert completed.retained_survey_target.evidence_path == plans[-1].survey_observation_support_path
    assert completed.retained_survey_target.source_snapshot == plans[-1].snapshot
    assert completed.retained_lidar_target is None
    assert len({item.survey_observation_support_path for item in plans}) == len(plans)


@pytest.mark.parametrize("kind", ["occluded", "absent", "wall"])
def test_full_phase_keeps_survey_candidate_until_camera_despite_weak_local_scans(integration, kind):
    config = integration.config()
    config = replace(config, camera_calibration=replace(config.camera_calibration, base_frame="base_footprint"))
    frames = iter((integration.frame(-.04), integration.frame(.3)))
    order = []
    def capture(cfg, effects, planning, uids, output):
        order.append("scan")
        assert order == ["select", "motion", "scan"]
        return capture_support(cfg, planning, output, kind=kind)
    def select(request):
        order.append("select")
        assert request.current_target_estimates == {}
        return approach._select_initial_preapproach(request)
    def camera(request):
        order.append("camera")
        return approach.CandidateObservation(request.output_dir / "recommendation.json", "QR_A", None)
    effects = integration.effects(admit_planning_frame=lambda _: next(frames),
        select_initial_preapproach=select,
        plan_preapproach=approach._plan_preapproach_from_request,
        run_motion_leg=lambda request: order.append("motion") or integration.fixtures._completed(request),
        capture_observation=camera,
        validate_facing=lambda request: {"candidate_uid": request.candidate.candidate_uid},
        commit_decision=lambda request: None)
    with patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture):
        result = approach.execute_candidate_approach_phase(config, effects)
    assert result.stand_count == 1 and order == ["select", "motion", "scan", "camera"]
    assert not any(event.get("event") == "camera_candidate_target_deferred" for event in integration.events)
