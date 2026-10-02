"""Passive recovery retains targets; a new aligned route requires fresh support."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scripts.aufgabe04.real_robot.candidate import lidar_acquisition as acquisition
from scripts.aufgabe04.real_robot.candidate.retained_orientation import retain_orientation_after_arrival
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from scripts.aufgabe04.real_robot.readiness.tour_scan_capture import TourScanCaptureError
from tests.aufgabe04 import test_candidate_lidar_acquisition as fixtures
from tests.aufgabe04 import test_candidate_lidar_handoff as handoff_fixtures


@pytest.fixture
def adapter():
    helper = fixtures.CandidateLidarAcquisitionAdapterTest()
    helper.setUp()
    try:
        yield helper
    finally:
        helper.doCleanups()


def test_passive_capture_failure_keeps_corrected_target_without_translation_support(adapter):
    original = replace(adapter.frame, camera_target_geometry=replace(
        adapter.frame.candidate.geometry, x_m=.02))
    adapter.source.require_current_lidar_support = True
    adapter.effects.capture_lidar_view = Mock(side_effect=TourScanCaptureError("sparse stopped scan"))
    support = Mock(side_effect=AssertionError("passive capture must not require translation support"))
    with patch.object(acquisition, "load_camera_lidar_receipts", return_value=((), {})):
        arrived, report, _ = adapter.make_adapter(require_translation_support=support)(original)
    assert arrived.camera_target_geometry == original.camera_target_geometry
    assert arrived.camera_target_geometry != arrived.candidate.geometry
    assert report["history"][-1]["reason"] == "fresh_scan_cohort_unavailable"
    assert adapter.effects.capture_lidar_view.call_count == 3
    support.assert_not_called()
    adapter.move.assert_not_called()


def test_verified_fit_replaces_old_target_without_retaining_its_proof(adapter):
    # This adapter test injects an already-bound source frame; proof validation
    # belongs to retention tests. A subsequent verified fit must retire that
    # binding rather than attaching the old receipt to its different center.
    old = replace(adapter.frame, camera_target_geometry=replace(
        adapter.frame.candidate.geometry, x_m=-.02),
        current_lidar_target_path=Path("prior_support.json"),
        retained_lidar_target=object(), camera_target_geometry_evidence_path=Path("prior_projection.json"),
        camera_alignment={"old": True})
    adapter.frame = old
    support = Mock(side_effect=AssertionError("verified passive fit must not plan translation"))
    with patch.object(acquisition, "retain_orientation_after_arrival", side_effect=lambda before, after, root: before), \
            patch.object(acquisition, "load_camera_lidar_receipts", return_value=(
                fixtures.line_receipts(increment=.01) + fixtures.line_receipts(2, increment=.01), {})):
        arrived, report, _ = adapter.make_adapter(require_translation_support=support)(old)
    assert report["head_alignment_verified"]
    assert report["camera_centered_verified"]
    assert arrived.camera_target_geometry != old.camera_target_geometry
    assert arrived.current_lidar_target_path is None
    assert arrived.retained_lidar_target is None
    assert arrived.camera_target_geometry_evidence_path is None
    assert arrived.camera_alignment is None
    support.assert_not_called()
    adapter.move.assert_not_called()


@pytest.mark.parametrize("callback_present", [False, True])
def test_alignment_missing_or_rejected_support_never_prepares_translation(adapter, callback_present):
    adapter.source.require_current_lidar_support = True
    denied = ValueError("current target absent")
    support = Mock(side_effect=denied)

    def direct_alignment(**callbacks):
        return lambda frame: callbacks["move_aligned"](frame, object(), 1, Mock())

    with patch.object(acquisition, "create_bounded_lidar_recovery", side_effect=direct_alignment), \
            patch.object(acquisition, "plan_and_select_camera_candidate") as plan:
        recovery = adapter.make_adapter(**({"require_translation_support": support} if callback_present else {}))
        error = ValueError if callback_present else RuntimeError
        with pytest.raises(error, match="current target absent|current target support admission"):
            recovery(adapter.frame)
    assert support.call_count == int(callback_present)
    plan.assert_not_called()
    adapter.move.assert_not_called()
    assert adapter.capture_requests == []


def test_alignment_support_runs_before_preparing_and_dispatching_route(adapter):
    adapter.source.require_current_lidar_support = True
    adapter.source.camera_selection_linear_speed_mps = .1
    adapter.source.camera_selection_angular_speed_radps = .3
    config = SimpleNamespace(snapshot=adapter.frame.config.snapshot, map_yaml=Path("map.yaml"),
        semantic_map_id="arena", plan=adapter.source.plan, inflation_radius_m=.1,
        candidate_transit_radius_m=.3, physical_clearance=adapter.source.physical_clearance)
    supported = replace(adapter.frame, config=config, camera_target_geometry=replace(
        adapter.frame.candidate.geometry, x_m=.015))
    events = []

    def support(frame, root):
        events.append("support")
        assert root.name == "planning"
        return supported

    prepared = SimpleNamespace(camera_alignment={"bound": True}, approach_bearing_rad=0., approach_offset_m=.55)
    selected = SimpleNamespace(selected_plan=prepared, to_evidence=lambda: {"selected": True})

    def plan(**kwargs):
        events.append("plan")
        assert kwargs["snapshot"] is supported.config.snapshot
        return selected

    def move(frame, *args, **kwargs):
        events.append("move")
        assert frame is supported
        assert kwargs["prepared_plan"] is prepared
        kwargs["before_motion"]()
        return frame

    def observe_then_align(**callbacks):
        def run(frame):
            observed, hint, _, _ = callbacks["observe"](frame, 0, Mock())
            moved = callbacks["move_aligned"](observed, hint, 1, Mock())
            return moved, {}, None
        return run

    adapter.move.side_effect = move
    with patch.object(acquisition, "create_bounded_lidar_recovery", side_effect=observe_then_align), \
            patch.object(acquisition, "load_camera_lidar_receipts", return_value=(
                fixtures.line_receipts(increment=.01) + fixtures.line_receipts(2, increment=.01), {})), \
            patch.object(acquisition, "plan_and_select_camera_candidate", side_effect=plan):
        arrived, _, _ = adapter.make_adapter(require_translation_support=support)(adapter.frame)
    assert arrived is supported
    assert events == ["support", "plan", "move"]
    assert len(adapter.capture_requests) == 1


def test_backside_refresh_preserves_independent_target_geometry(adapter):
    source = replace(adapter.frame, retained_backside_axis_path=Path("certified_axis.json"),
        camera_target_geometry=replace(adapter.frame.candidate.geometry, x_m=.02))
    with patch("scripts.aufgabe04.real_robot.candidate.retained_orientation.write_backside_axis_frame_projection") as project, \
            patch("scripts.aufgabe04.real_robot.candidate.retained_orientation.load_backside_axis_planning_observation",
                  return_value=SimpleNamespace(validated_target_center=None)):
        arrived = retain_orientation_after_arrival(source, adapter.frame, adapter.root / "arrival")
    project.assert_called_once()
    assert arrived.retained_backside_axis_path == adapter.root / "arrival/retained_backside_orientation.json"
    assert arrived.camera_target_geometry == source.camera_target_geometry
    assert arrived.camera_target_geometry_evidence_path.is_file()


def test_certified_backside_center_retains_its_own_bound_after_yaw(adapter):
    from scripts.aufgabe04.artifacts.current_target_estimate import planning_target_geometry
    estimate = dict(x_m=.12, y_m=0., uncertainty_m=.025,
                    policy="reconciled_metric_head_position_engineering_bound")
    source = replace(adapter.frame, retained_backside_axis_path=Path("certified_axis.json"),
        camera_target_geometry=planning_target_geometry(adapter.frame.candidate, estimate))
    assert estimate["x_m"] > source.candidate.geometry.radius_m + source.candidate.geometry.uncertainty_m
    fresh = replace(adapter.frame, planning_frame=replace(adapter.frame.planning_frame,
        current_pose=replace(adapter.frame.planning_frame.current_pose, yaw_rad=.1)))
    receipt = SimpleNamespace(validated_target_center=estimate)
    with patch("scripts.aufgabe04.real_robot.candidate.retained_orientation.write_backside_axis_frame_projection"), \
            patch("scripts.aufgabe04.real_robot.candidate.retained_orientation.load_backside_axis_planning_observation",
                  return_value=receipt) as load:
        arrived = retain_orientation_after_arrival(source, fresh, adapter.root / "after_yaw")
    assert load.call_count == 2
    assert arrived.camera_target_geometry == source.camera_target_geometry
    assert arrived.planning_frame == fresh.planning_frame
    assert arrived.retained_backside_axis_path == adapter.root / "after_yaw/retained_backside_orientation.json"
    assert arrived.retained_lidar_target is None
    assert arrived.current_lidar_target_path is None


@pytest.fixture
def inspection():
    helper = handoff_fixtures.CandidateLidarHandoffTest()
    helper.setUp()
    helper.source.inflation_radius_m = .1
    helper.source.candidate_transit_radius_m = .3
    helper.source.physical_clearance = {"minimum_active_standoff_m": .33}
    helper.source.snapshot_path = helper.root / "snapshot.json"
    helper.effects.plan_preapproach = Mock()
    try:
        yield helper
    finally:
        helper.doCleanups()


def inspection_hooks(helper):
    from scripts.aufgabe04.real_robot.candidate import inspection_adapters
    hooks = {}

    def capture_hooks(**kwargs):
        hooks.update(kwargs)
        return Mock()

    with patch.object(acquisition, "create_lidar_camera_recovery", side_effect=capture_hooks), \
            patch.object(inspection_adapters, "execute_candidate_inspection", return_value=None):
        inspection_adapters.execute_local_candidate_inspection(
            observation_frame=helper.initial, source_config=helper.source, effects=helper.effects,
            source_registry=helper.registry, candidate_root=helper.root,
            candidate_run_id="run", candidate_index=0, admit_arrival=helper.admit,
            admit_planning=lambda **kw: (helper.source, helper.initial.candidate,
                helper.initial.planning_frame.current_pose, helper.initial.planning_frame, None),
            move_certified_opposite=Mock(), execute_motion=helper.motion,
            frame_type=type(helper.initial), request_type=SimpleNamespace,
            observation_request_type=SimpleNamespace)
    return hooks


def test_disjoint_fresh_target_cannot_authorize_motion_with_retained_axis(inspection):
    inspection.source.require_current_lidar_support = True
    hooks = inspection_hooks(inspection)
    source = replace(inspection.initial, retained_backside_axis_path=Path("axis.json"))
    estimate = dict(x_m=.12, y_m=0., uncertainty_m=.02, policy="current_stopped_lidar_surface")
    axis = SimpleNamespace(validated_target_center=dict(x_m=-.10, y_m=0., uncertainty_m=.02,
        policy="reconciled_metric_head_position_engineering_bound"))
    with patch("scripts.aufgabe04.real_robot.candidate.approach._require_current_lidar_target",
               return_value=(estimate, {"evidence_path": "supported.json"})), \
            patch("scripts.aufgabe04.navigation.approach.backside_axis_frame_projection.load_backside_axis_planning_observation",
                  return_value=axis), \
            patch("scripts.aufgabe04.real_robot.candidate.inspection_adapters.bind_current_lidar_target") as bind:
        with pytest.raises(CandidateObservationUnavailableError) as error:
            hooks["require_translation_support"](source, inspection.root / "translation")
    assert error.value.status_evidence["reason"] == "current_lidar_disagrees_with_retained_target"
    assert error.value.process_evidence["motion_authorized"] is False
    bind.assert_not_called()
    inspection.effects.plan_preapproach.assert_not_called()
    inspection.motion.assert_not_called()


def test_completed_ordinary_inspection_move_keeps_original_axis_receipt(inspection):
    hooks = inspection_hooks(inspection)
    source = replace(inspection.initial, retained_backside_axis_path=Path("original_axis.json"))
    completed = replace(inspection.fresh, camera_target_geometry=replace(
        inspection.fresh.candidate.geometry, x_m=.02))

    def plan(request):
        request.output_dir.mkdir(parents=True)
        (request.output_dir / "pipeline_summary.json").write_text(json.dumps({
            "selected_approach_pose": {"x_m": -.6, "y_m": .1, "yaw_rad": 0.}}))
        return object()

    def move(**kwargs):
        kwargs["completed_frame_sink"](completed)

    inspection.effects.plan_preapproach.side_effect = plan
    inspection.motion.side_effect = move
    with patch("scripts.aufgabe04.real_robot.candidate.inspection_adapters.retain_orientation_after_arrival",
               side_effect=lambda before, after, root: replace(after,
                   retained_backside_axis_path=before.retained_backside_axis_path)):
        arrived = hooks["plan_and_move"](source, 0., inspection.root / "move", 1, None,
            offset=.55, purpose="arrival_alignment")
    inspection.motion.assert_called_once()
    assert arrived.retained_backside_axis_path == source.retained_backside_axis_path
    assert arrived.camera_target_geometry == completed.camera_target_geometry
    assert arrived.planning_frame == completed.planning_frame
