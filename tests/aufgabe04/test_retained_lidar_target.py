"""Arrival keeps a source-proven target without requiring a second head fit."""
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, write_content_hashed_json
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.real_robot.candidate.current_lidar_targets import capture_current_lidar_targets
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import capture_candidate_lidar_view
from scripts.aufgabe04.real_robot.candidate.lidar_head_capture import HASH_FIELD as CAPTURE_HASH, head_capture_payload
from scripts.aufgabe04.real_robot.candidate.target_admission import (
    bind_current_lidar_target, require_frame_target, retain_camera_target_geometry,
)
from tests.aufgabe04.test_current_lidar_targets import fixture, raw_scans
from tests.aufgabe04.test_coverage_visibility_reporting import _plan


@dataclass(frozen=True)
class Frame:
    config: object
    candidate: object
    planning_frame: object
    camera_target_geometry: object = None
    camera_alignment: object = None
    camera_target_geometry_evidence_path: object = None
    current_lidar_target_path: object = None
    retained_lidar_target: object = None


def bound_fixture(tmp_path, *, shift=.12):
    args = fixture(shifts=[shift]*8)
    config = SimpleNamespace(snapshot=args["snapshot"], plan=replace(_plan(), survey_id="survey"),
        camera_calibration=SimpleNamespace(base_frame="base_footprint"),
        lidar_scan_frame="base_scan", lidar_scan_topic="/scan", measured_stand_model=args["stand_model"])
    times = iter((100., 100.72))
    def capture(request):
        raw = head_capture_payload(raw_scans(shifts=[shift]*8), tour_id=request.viewpoint_id,
            odom_frame="odom", base_frame="base_footprint", scan_frame="base_scan", captured_at_unix_sec=100.71)
        path = request.output_dir / "scan_cohort.json"
        write_content_hashed_json(path, raw, hash_field=CAPTURE_HASH)
        clock = iter((100., 100.72))
        return capture_candidate_lidar_view(request, capture_cohort=lambda _: path, clock=lambda: next(clock))
    effects = SimpleNamespace(clock=lambda: next(times), capture_lidar_view=Mock(side_effect=capture))
    _, evidence = capture_current_lidar_targets(config, effects, args["planning_frame"],
        {"candidate_1"}, tmp_path / "support")
    frame = Frame(config, config.snapshot.candidates[0], args["planning_frame"])
    return bind_current_lidar_target(frame, evidence_path=Path(evidence["evidence_path"])), effects, evidence


def projected_frame(original, transform):
    # Fixture survey centers live at canonical odom (0, 0). Construct the
    # refreshed frozen snapshot independently of target reprojection code.
    candidate = replace(original.candidate, geometry=replace(original.candidate.geometry,
        x_m=transform.x_m, y_m=transform.y_m))
    snapshot = replace(original.config.snapshot, candidates=(candidate,))
    config = SimpleNamespace(**{**vars(original.config), "snapshot": snapshot})
    planning = replace(original.planning_frame, map_from_odom=transform,
        current_pose=Pose2D(transform.x_m-.5*math.cos(transform.yaw_rad),
                           transform.y_m-.5*math.sin(transform.yaw_rad), transform.yaw_rad))
    return Frame(config, candidate, planning)


def test_current_surface_outside_old_envelope_retains_original_proof(tmp_path):
    source, effects, evidence = bound_fixture(tmp_path)
    assert source.camera_target_geometry.x_m == pytest.approx(.12)
    assert source.camera_target_geometry.x_m > source.candidate.geometry.radius_m+source.candidate.geometry.uncertainty_m
    transform = PlanarTransform2D(1., -.2, math.pi/2)
    arrival = projected_frame(source, transform)
    retained = retain_camera_target_geometry(source, arrival, evidence_path=tmp_path / "arrival.json")
    assert retained.camera_target_geometry.x_m == pytest.approx(1.)
    assert retained.camera_target_geometry.y_m == pytest.approx(-.08)
    assert retained.camera_target_geometry.uncertainty_m == source.camera_target_geometry.uncertainty_m
    assert retained.retained_lidar_target is source.retained_lidar_target
    assert retained.current_lidar_target_path == Path(evidence["evidence_path"])
    assert retained.config.snapshot is arrival.config.snapshot
    assert retained.candidate.geometry.keepout_radius_m == source.candidate.geometry.keepout_radius_m
    assert retained.camera_alignment is None
    effects.capture_lidar_view.assert_called_once()
    proof = load_content_hashed_json(tmp_path / "arrival.json", hash_field="camera_target_geometry_projection_sha256")
    assert proof["source_kind"] == "current_stopped_lidar_surface"
    assert proof["retained_lidar_target"]["source_candidate_snapshot_sha256"] == evidence["candidate_snapshot_sha256"]
    assert not any(proof[k] for k in ("head_alignment_verified", "camera_centered", "motion_authorized", "keepouts_changed"))


def test_multiple_reprojections_use_original_snapshot_and_odom_point(tmp_path):
    source, effects, _ = bound_fixture(tmp_path)
    current = source
    for i, transform in enumerate((PlanarTransform2D(.4, -.5, 1.1),
                                   PlanarTransform2D(-.3, .2, -2.3),
                                   PlanarTransform2D(0., 0., 0.))):
        current = retain_camera_target_geometry(current, projected_frame(source, transform),
            evidence_path=tmp_path / f"projected_{i}.json")
        assert current.camera_target_geometry.x_m == pytest.approx(transform.x_m+.12*math.cos(transform.yaw_rad))
        assert current.camera_target_geometry.y_m == pytest.approx(transform.y_m+.12*math.sin(transform.yaw_rad))
        assert current.retained_lidar_target.source_snapshot is source.config.snapshot
    effects.capture_lidar_view.assert_called_once()


@pytest.mark.parametrize("mutation", ["uid", "map", "frame", "projection", "radius", "keepout", "population", "lineage"])
def test_retention_rejects_changed_arrival_binding(tmp_path, mutation):
    source, _, _ = bound_fixture(tmp_path)
    arrival = projected_frame(source, PlanarTransform2D(.3, .2, .6))
    candidate, snapshot, planning = arrival.candidate, arrival.config.snapshot, arrival.planning_frame
    if mutation == "uid":
        candidate = replace(candidate, candidate_uid="unrelated")
    elif mutation == "map":
        snapshot = replace(snapshot, map_bundle_sha256="b"*64)
    elif mutation == "frame":
        planning = replace(planning, odom_frame="other_odom")
    elif mutation in ("projection", "radius", "keepout"):
        key = {"projection": "x_m", "radius": "radius_m", "keepout": "keepout_radius_m"}[mutation]
        candidate = replace(candidate, geometry=replace(candidate.geometry, **{key: getattr(candidate.geometry, key)+.01}))
    elif mutation == "population":
        snapshot = replace(snapshot, candidates=(candidate, replace(candidate, candidate_uid="other")))
    else:
        candidate = replace(candidate, source=replace(candidate.source, observation_ids=("unrelated",)))
    if mutation in ("uid", "projection", "radius", "keepout", "lineage"):
        snapshot = replace(snapshot, candidates=(candidate,))
    arrival = replace(arrival, candidate=candidate, planning_frame=planning,
        config=SimpleNamespace(**{**vars(arrival.config), "snapshot": snapshot}))
    with pytest.raises(ValueError):
        retain_camera_target_geometry(source, arrival, evidence_path=tmp_path / "bad.json")


@pytest.mark.parametrize("mutation", ["geometry", "uncertainty", "path", "snapshot", "transform", "proof", "capture"])
def test_retention_rechecks_original_proof_and_source_geometry(tmp_path, mutation):
    source, _, evidence = bound_fixture(tmp_path)
    arrival = projected_frame(source, PlanarTransform2D(.3, .2, .6))
    if mutation in ("geometry", "uncertainty"):
        key = "x_m" if mutation == "geometry" else "uncertainty_m"
        source = replace(source, camera_target_geometry=replace(source.camera_target_geometry,
            **{key: getattr(source.camera_target_geometry, key)+.001}))
    elif mutation == "path":
        source = replace(source, current_lidar_target_path=tmp_path / "other.json")
    elif mutation == "snapshot":
        source = replace(source, retained_lidar_target=replace(source.retained_lidar_target,
            source_snapshot=replace(source.config.snapshot, snapshot_id="other")))
    elif mutation == "transform":
        source = replace(source, retained_lidar_target=replace(source.retained_lidar_target,
            source_planning_frame=replace(source.planning_frame, map_from_odom=PlanarTransform2D(.01, 0., 0.))))
    else:
        path = Path(evidence["evidence_path" if mutation == "proof" else "capture_path"])
        raw = json.loads(path.read_text())
        if mutation == "proof":
            raw["candidate_decisions"]["candidate_1"]["estimate"]["x_m"] += .001
        else:
            raw["scans"][0]["ranges"][50] += .001
        path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        retain_camera_target_geometry(source, arrival, evidence_path=tmp_path / "bad.json")


def test_initial_binding_requires_exact_source_planning_frame(tmp_path):
    source, _, evidence = bound_fixture(tmp_path)
    changed = replace(source, planning_frame=replace(source.planning_frame,
        current_pose=replace(source.planning_frame.current_pose, x_m=-.5)))
    with pytest.raises(ValueError, match="original source binding"):
        bind_current_lidar_target(changed, evidence_path=Path(evidence["evidence_path"]))


def test_unproven_camera_fit_keeps_original_tight_envelope(tmp_path):
    source, _, _ = bound_fixture(tmp_path)
    source = replace(source, retained_lidar_target=None, current_lidar_target_path=None)
    with pytest.raises(ValueError, match="outside candidate envelope"):
        retain_camera_target_geometry(source, projected_frame(source, PlanarTransform2D(0., 0., 0.)),
                                      evidence_path=tmp_path / "unproven.json")


def test_new_verified_arrival_geometry_supersedes_prior_target(tmp_path):
    source, _, _ = bound_fixture(tmp_path)
    arrival = projected_frame(source, PlanarTransform2D(.3, .2, .6))
    arrival = replace(arrival, camera_target_geometry=replace(arrival.candidate.geometry, x_m=.32))
    assert retain_camera_target_geometry(source, arrival, evidence_path=tmp_path / "unused.json") is arrival
    assert arrival.retained_lidar_target is None


@pytest.mark.parametrize("alter_geometry", [False, True])
def test_static_target_admission_revalidates_retained_proof_without_capture(tmp_path, alter_geometry):
    frame, effects, _ = bound_fixture(tmp_path)
    if alter_geometry:
        frame = replace(frame, camera_target_geometry=replace(frame.camera_target_geometry, x_m=.1))
    with patch("scripts.aufgabe04.real_robot.candidate.target_admission.require_target") as admit:
        if alter_geometry:
            with pytest.raises(ValueError, match="geometry/path binding"):
                require_frame_target(frame, evidence_path=tmp_path / "admission.json", attempt_index=1)
            admit.assert_not_called()
        else:
            require_frame_target(frame, evidence_path=tmp_path / "admission.json", attempt_index=1)
            assert admit.call_args.kwargs["target_geometry"] == frame.camera_target_geometry
    effects.capture_lidar_view.assert_called_once()
