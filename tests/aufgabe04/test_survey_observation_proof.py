"""Survey observation authority is source-bound and needs no current scan."""
from copy import deepcopy
from dataclasses import replace
import math
from unittest.mock import patch

import pytest

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import coverage_survey_plan_sha256
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.real_robot.candidate.approach import _CandidateObservationFrame
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from scripts.aufgabe04.real_robot.candidate.target_admission import (
    SURVEY_OBSERVATION_HASH_FIELD, SURVEY_OBSERVATION_POLICY,
    bind_survey_observation_target, load_survey_observation_support,
    require_frame_target, retain_camera_target_geometry, write_survey_observation_support,
)
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256
from tests.aufgabe04 import test_autonomous_candidate_approach as fixtures
from tests.aufgabe04.test_candidate_target_admission import _with_advisory


def context(tmp_path):
    helper = fixtures.AutonomousCandidateApproachTest()
    config = helper._config(tmp_path, (helper._candidate("candidate_1", 1., 0.),))
    planning = CandidatePlanningFrame(Pose2D(.3, 0., 0.), PlanarTransform2D(0., 0., 0.))
    return config, planning


def write(tmp_path, config, planning):
    path = tmp_path / "survey_observation.json"
    proof = write_survey_observation_support(path, config=config,
        planning_frame=planning, candidate_uid="candidate_1")
    return path, proof


def test_scan_free_proof_retains_hypothesis_and_rechecks_live_map(tmp_path):
    config, planning = context(tmp_path)
    snapshot_before = candidate_snapshot_sha256(config.snapshot)
    with patch("scripts.aufgabe04.real_robot.candidate.current_lidar_targets.capture_current_lidar_targets",
               side_effect=AssertionError("survey admission cannot require current scans")), \
         patch("scripts.aufgabe04.real_robot.candidate.current_lidar_targets.load_current_lidar_assessment",
               side_effect=AssertionError("survey admission cannot require scan replay")):
        path, proof = write(tmp_path, config, planning)
        assert proof == load_survey_observation_support(path, candidate_uid="candidate_1", snapshot=config.snapshot)
        frame = bind_survey_observation_target(_CandidateObservationFrame(config,
            config.snapshot.candidates[0], planning, None), evidence_path=path)
        assert require_frame_target(frame, evidence_path=tmp_path / "arrival.json", attempt_index=0).accepted
    assert proof["policy"] == SURVEY_OBSERVATION_POLICY
    assert proof["coverage_plan_sha256"] == coverage_survey_plan_sha256(config.plan)
    assert proof["planning_frame"] == planning.to_evidence()
    assert proof["target_admission"]["accepted"] is True
    assert proof["purpose"] == "observation_only"
    assert not any(proof[key] for key in ("motion_authorized", "precise_motion_authorized", "keepouts_changed"))
    assert frame.camera_target_geometry == config.snapshot.candidates[0].geometry
    assert frame.retained_survey_target is not None and frame.retained_lidar_target is None
    assert frame.current_lidar_target_path is None and frame.camera_alignment is None
    assert candidate_snapshot_sha256(config.snapshot) == snapshot_before
    config.map_yaml.write_text(config.map_yaml.read_text().replace("resolution: 0.1", "resolution: 0.2"))
    with pytest.raises(ValueError, match="runtime map"):
        require_frame_target(frame, evidence_path=tmp_path / "changed_map.json", attempt_index=1)


@pytest.mark.parametrize("fault", ["wall", "outside", "morphology", "unknown_uid", "wrong_frame", "map_binding"])
def test_writer_rejects_unadmitted_targets_without_publishing_proof(tmp_path, fault):
    config, planning = context(tmp_path)
    candidate = config.snapshot.candidates[0]
    uid = candidate.candidate_uid
    if fault in {"wall", "outside"}:
        candidate = replace(candidate, geometry=replace(candidate.geometry, x_m=-5. if fault == "wall" else -6.))
    elif fault == "morphology":
        candidate = _with_advisory(candidate, map_hash=config.snapshot.map_bundle_sha256)
    elif fault == "unknown_uid":
        uid = "ghost_not_in_snapshot"
    elif fault == "wrong_frame":
        planning = replace(planning, map_frame="other_map")
    else:
        config = replace(config, plan=replace(config.plan, map_bundle_sha256="d"*64))
    config = replace(config, snapshot=replace(config.snapshot, candidates=(candidate,)))
    path = tmp_path / "rejected.json"
    with pytest.raises((ValueError, CandidateObservationUnavailableError)):
        write_survey_observation_support(path, config=config, planning_frame=planning, candidate_uid=uid)
    assert not path.exists()


@pytest.mark.parametrize("fault", ["policy", "source", "purpose", "authority", "numeric_false", "version",
    "candidate", "snapshot", "map", "geometry", "plan_hash", "frame", "extra", "decision",
    "decision_geometry", "clearance", "static_pose", "static_flags", "unknown_decision_field"])
def test_rehashed_false_or_mismatched_proof_is_rejected(tmp_path, fault):
    config, planning = context(tmp_path)
    _, original = write(tmp_path, config, planning)
    proof = deepcopy(original)
    changes = {"policy": ("policy", "current_stopped_lidar_surface"),
        "source": ("source_kind", "current_scan"), "purpose": ("purpose", "precise_alignment"),
        "authority": ("precise_motion_authorized", True), "numeric_false": ("motion_authorized", 0),
        "version": ("schema_version", True), "candidate": ("candidate_uid", "other"),
        "snapshot": ("candidate_snapshot_sha256", "a"*64), "map": ("map_bundle_sha256", "a"*64),
        "geometry": ("candidate_geometry_sha256", "a"*64), "plan_hash": ("coverage_plan_sha256", "unknown")}
    if fault in changes:
        key, value = changes[fault]; proof[key] = value
    elif fault == "frame":
        proof["planning_frame"]["map_frame"] = "other_map"
    elif fault == "extra":
        proof["current_lidar_support"] = {"accepted": True}
    elif fault == "decision":
        proof["target_admission"]["accepted"] = False
    elif fault == "decision_geometry":
        proof["target_admission"]["target_geometry_sha256"] = "a"*64
    elif fault == "unknown_decision_field":
        proof["target_admission"]["stand_confirmed"] = True
    else:
        static = proof["target_admission"]["static_map_evidence"]
        if fault == "clearance":
            static["static_map_clearance_m"] = .001
        elif fault == "static_pose":
            static["pose"]["x_m"] += .01
        else:
            static["admitted"] = 1
    path = tmp_path / "tampered.json"
    write_content_hashed_json(path, proof, hash_field=SURVEY_OBSERVATION_HASH_FIELD)
    with pytest.raises(ValueError):
        load_survey_observation_support(path, candidate_uid="candidate_1", snapshot=config.snapshot)


@pytest.mark.parametrize("fault", ["candidate", "geometry", "frame", "plan"])
def test_bind_rejects_wrong_candidate_epoch_or_plan(tmp_path, fault):
    config, planning = context(tmp_path)
    path, _ = write(tmp_path, config, planning)
    candidate = config.snapshot.candidates[0]
    camera_geometry = None
    if fault == "candidate":
        candidate = replace(candidate, geometry=replace(candidate.geometry, x_m=1.01))
    elif fault == "geometry":
        camera_geometry = replace(candidate.geometry, x_m=1.01)
    elif fault == "frame":
        planning = replace(planning, current_pose=Pose2D(.4, 0., 0.))
    else:
        config = replace(config, plan=replace(config.plan, survey_id="different_survey"))
    frame = _CandidateObservationFrame(config, candidate, planning, None, camera_target_geometry=camera_geometry)
    with pytest.raises(ValueError):
        bind_survey_observation_target(frame, evidence_path=path)


def test_retained_survey_proof_reprojects_original_geometry_and_detects_source_tampering(tmp_path):
    config, planning = context(tmp_path)
    path, proof = write(tmp_path, config, planning)
    source = bind_survey_observation_target(_CandidateObservationFrame(config,
        config.snapshot.candidates[0], planning, None), evidence_path=path)
    wrong_plan = replace(config, plan=replace(config.plan, survey_id="different_survey"))
    with pytest.raises(ValueError, match="original source binding"):
        require_frame_target(replace(source, config=wrong_plan),
            evidence_path=tmp_path / "wrong_plan.json", attempt_index=0)
    with pytest.raises(ValueError, match="coverage plan binding"):
        retain_camera_target_geometry(source, replace(source, config=wrong_plan,
            camera_target_geometry=None), evidence_path=tmp_path / "wrong_arrival_plan.json")
    for index, transform in enumerate((PlanarTransform2D(.2, -.1, .4), PlanarTransform2D(-.1, .2, -.3))):
        candidate = replace(source.candidate, geometry=replace(source.candidate.geometry,
            x_m=transform.x_m+math.cos(transform.yaw_rad), y_m=transform.y_m+math.sin(transform.yaw_rad)))
        arrival_config = replace(config, snapshot=replace(config.snapshot, candidates=(candidate,)))
        arrival = _CandidateObservationFrame(arrival_config, candidate,
            replace(planning, map_from_odom=transform), None)
        source = retain_camera_target_geometry(source, arrival, evidence_path=tmp_path / f"arrival_{index}.json")
        assert source.camera_target_geometry == candidate.geometry
        assert source.retained_survey_target.source_snapshot == config.snapshot
    proof["planning_frame"]["current_pose"]["x_m"] += .01
    path.unlink()
    write_content_hashed_json(path, proof, hash_field=SURVEY_OBSERVATION_HASH_FIELD)
    with pytest.raises(ValueError, match="original source binding"):
        require_frame_target(source, evidence_path=tmp_path / "altered.json", attempt_index=1)
