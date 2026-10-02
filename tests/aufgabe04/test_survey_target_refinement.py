"""An optional first-arrival cohort refines a point without becoming admission."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, write_content_hashed_json
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.real_robot.candidate import approach
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import (
    CandidateLidarCaptureUnavailableError, capture_candidate_lidar_view,
)
from scripts.aufgabe04.real_robot.candidate.lidar_head_capture import HASH_FIELD as CAPTURE_HASH, head_capture_payload
from scripts.aufgabe04.real_robot.candidate.survey_target_refinement import (
    HASH_FIELD, refine_survey_observation_target,
)
from scripts.aufgabe04.real_robot.candidate.target_admission import (
    bind_survey_observation_target, evaluate_target, require_frame_target,
    write_survey_observation_support,
)
from scripts.aufgabe04.real_robot.readiness.tour_scan_capture import TourScanCaptureError
from tests.aufgabe04.test_current_lidar_targets import raw_scans
from tests.aufgabe04 import test_current_lidar_approach_integration as integration_fixtures


@pytest.fixture
def integration():
    helper = integration_fixtures.CurrentLidarApproachIntegrationTests()
    helper.setUp()
    try:
        yield helper
    finally:
        helper.doCleanups()


def source(integration, *, arena_bounds=None):
    config = integration.config()
    config = replace(config, camera_calibration=replace(config.camera_calibration, base_frame="base_footprint"))
    if arena_bounds is not None:
        config = replace(config, plan=replace(config.plan, arena_bounds=arena_bounds))
    planning = integration.frame(-.04)
    artifacts = approach._materialize_candidate_frame_projection(source_config=config,
        source_registry=integration.registry(config), planning_frame=planning,
        output_root=integration.root / "source_projection")
    frame = approach._CandidateObservationFrame(artifacts.config, artifacts.config.snapshot.candidates[0],
        planning, artifacts.camera_decision_binding(), observation_pose=planning.current_pose)
    path = integration.root / "survey_observation_support.json"
    write_survey_observation_support(path, config=frame.config, planning_frame=planning,
        candidate_uid=frame.candidate.candidate_uid)
    return config, bind_survey_observation_target(frame, evidence_path=path), {"evidence_path": str(path)}


def fresh_effects(frame, *, kind="stand", shift=.12):
    planning = frame.planning_frame
    transform = planning.map_from_odom
    base_x = planning.current_pose.x_m-transform.x_m
    base_y = planning.current_pose.y_m-transform.y_m
    scan_x = base_x+.04
    target_x = frame.candidate.geometry.x_m-transform.x_m

    def capture(request):
        scans = raw_scans(kinds=[kind]*8, shifts=[target_x-scan_x-1.+shift]*8)
        for scan in scans:
            for key in ("stamp_sec", "received_at_unix_sec", "scan_pose_stamp_sec", "base_pose_stamp_sec"):
                scan[key] += 100.
            scan["head_plane_mount"]["exact_transform_stamp_sec"] += 100.
            scan["scan_pose_odom"] = {"x_m": scan_x, "y_m": base_y, "yaw_rad": 0.}
            scan["base_pose_odom"] = {"x_m": base_x, "y_m": base_y, "yaw_rad": 0.}
        payload = head_capture_payload(scans, tour_id=request.viewpoint_id,
            odom_frame="odom", base_frame=request.base_frame, scan_frame=request.scan_frame,
            captured_at_unix_sec=200.71)
        path = request.output_dir / "cohort.json"
        write_content_hashed_json(path, payload, hash_field=CAPTURE_HASH)
        clock = iter((200., 200.72))
        return capture_candidate_lidar_view(request, capture_cohort=lambda _: path,
            clock=lambda: next(clock))

    clock = iter((200., 200.72))
    return SimpleNamespace(clock=lambda: next(clock), capture_lidar_view=Mock(side_effect=capture))


def test_first_arrival_refines_from_one_real_cohort_and_binds_immutable_proof(integration):
    _, frame, support = source(integration)
    original_proof = Path(support["evidence_path"]).read_bytes()
    effects = fresh_effects(frame)
    output = integration.root / "refinement"

    refined, evidence = refine_survey_observation_target(
        frame, effects=effects, output_dir=output, attempt_index=0)

    effects.capture_lidar_view.assert_called_once()
    assert evidence["accepted"] and evidence["reason"] == "fresh_lidar_refinement_retained"
    assert refined.camera_target_geometry.x_m == pytest.approx(frame.candidate.geometry.x_m+.12)
    assert refined.retained_survey_target is None
    assert refined.retained_lidar_target is not None
    assert refined.camera_target_geometry_evidence_path is None
    assert refined.camera_alignment is None
    assert refined.candidate is frame.candidate and refined.config.snapshot is frame.config.snapshot
    assert Path(support["evidence_path"]).read_bytes() == original_proof
    assert require_frame_target(refined, evidence_path=output / "replayed_target.json", attempt_index=0).accepted
    saved = load_content_hashed_json(Path(evidence["evidence_path"]), hash_field=HASH_FIELD)
    assert saved["accepted"] and saved["maximum_cohort_count"] == 1
    assert not any(saved[key] for key in (
        "motion_authorized", "stand_axis_authorized", "head_alignment_verified", "camera_centered", "keepouts_changed"))


@pytest.mark.parametrize("kind", ["occluded", "absent", "ambiguous", "wall"])
def test_weak_or_ambiguous_or_nonstand_cohort_keeps_original_survey_frame(integration, kind):
    _, frame, _ = source(integration)
    effects = fresh_effects(frame, kind=kind)
    returned, evidence = refine_survey_observation_target(
        frame, effects=effects, output_dir=integration.root / "refinement", attempt_index=0)

    assert returned is frame and returned.retained_survey_target is frame.retained_survey_target
    assert not evidence["accepted"]
    assert evidence["reason"] == "fresh_lidar_refinement_not_supported"
    effects.capture_lidar_view.assert_called_once()
    assert Path(evidence["evidence_path"]).is_file()


def test_fresh_point_outside_arena_is_optional_and_does_not_replace_survey_target(integration):
    _, frame, _ = source(integration, arena_bounds=ArenaBounds(length_m=2.2, width_m=10.))
    assert evaluate_target(frame.config, frame.candidate, target_geometry=frame.camera_target_geometry).accepted
    effects = fresh_effects(frame)

    returned, evidence = refine_survey_observation_target(
        frame, effects=effects, output_dir=integration.root / "refinement", attempt_index=0)

    assert returned is frame and not evidence["accepted"]
    assert evidence["reason"] == "fresh_lidar_refinement_target_rejected"
    assert evidence["candidate_target_admission"]["reasons"] == ["target_static_map_incompatible"]
    effects.capture_lidar_view.assert_called_once()


@pytest.mark.parametrize("error", [CandidateLidarCaptureUnavailableError("expired cohort"),
    TourScanCaptureError("no stopped scans", diagnostics={"accepted_scan_count": 0})])
def test_transient_capture_absence_preserves_camera_view_without_retry(integration, error):
    _, frame, _ = source(integration)
    effects = fresh_effects(frame)
    effects.capture_lidar_view.side_effect = error
    returned, evidence = refine_survey_observation_target(
        frame, effects=effects, output_dir=integration.root / "refinement", attempt_index=0)

    assert returned is frame and not evidence["accepted"]
    assert evidence["reason"] == "fresh_lidar_refinement_unavailable"
    effects.capture_lidar_view.assert_called_once()


@pytest.mark.parametrize("error", [ValueError("capture source binding corrupt"),
    RuntimeError("capture configuration invalid"),
    TourScanCaptureError("ROS dependencies unavailable", retryable=False)])
def test_corrupt_or_unavailable_configuration_is_not_hidden(integration, error):
    _, frame, _ = source(integration)
    effects = fresh_effects(frame)
    effects.capture_lidar_view.side_effect = error
    output = integration.root / "refinement"
    with pytest.raises(type(error), match=str(error)):
        refine_survey_observation_target(frame, effects=effects, output_dir=output, attempt_index=0)
    effects.capture_lidar_view.assert_called_once()
    assert not (output / "survey_target_refinement.json").exists()


def test_changed_prior_proof_or_noninitial_call_cannot_trigger_refinement(integration):
    _, frame, support = source(integration)
    effects = fresh_effects(frame)
    with pytest.raises(ValueError, match="first survey-only"):
        refine_survey_observation_target(frame, effects=effects,
            output_dir=integration.root / "later", attempt_index=1)
    Path(support["evidence_path"]).write_text("{}")
    with pytest.raises(ValueError):
        refine_survey_observation_target(frame, effects=effects,
            output_dir=integration.root / "corrupt", attempt_index=0)
    effects.capture_lidar_view.assert_not_called()
