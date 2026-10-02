"""Current support cannot be manufactured by clipping a historical envelope."""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.perception.lidar_scan_metadata import LidarScanMetadata
from scripts.aufgabe04.perception.lidar_visibility_evidence import lidar_visibility_receipt_from_scan
from scripts.aufgabe04.perception.lidar_visibility_frames import LidarVisibilityFrameProvenance
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.real_robot.candidate.current_lidar_targets import (
    HASH_FIELD, POLICY, assess_current_lidar_targets, capture_current_lidar_targets,
    load_current_lidar_target,
)
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import capture_candidate_lidar_view
from scripts.aufgabe04.real_robot.candidate.lidar_head_capture import HASH_FIELD as CAPTURE_HASH, head_capture_payload
from tests.aufgabe04.test_lidar_inspection_hint import hint_fixture
from tests.aufgabe04.test_coverage_visibility_reporting import _plan
from tests.aufgabe04.test_tour_scan_capture import sample


def model():
    return load_measured_physical_stand_model(Path(__file__).parents[2] /
        "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")


def raw_scans(kinds=None, shifts=None):
    kinds, shifts = kinds or ["stand"]*8, shifts or [0.]*8
    scans = []
    for n, (kind, shift) in enumerate(zip(kinds, shifts)):
        record = sample(100.+n*.1)
        ranges = []
        for i in range(101):
            angle = (i-50)*.01
            x = (.5 if kind == "occluded" else 2. if kind == "absent" else 1.+shift)
            distance = x/math.cos(angle)
            y = distance*math.sin(angle)
            present = (kind in ("wall", "occluded", "absent") or abs(y) <= .036)
            if kind == "ambiguous":
                present = .015 <= abs(y) <= .045
            ranges.append(distance if present else None)
        record.update(angle_min=-.5, angle_increment=.01, range_min=.05, range_max=8., ranges=ranges,
            base_pose_odom=dict(x_m=-1.04, y_m=0., yaw_rad=0.),
            scan_pose_odom=dict(x_m=-1., y_m=0., yaw_rad=0.))
        record["scan_metadata"] = LidarScanMetadata(.5, 0., .1, "linear", (),
            tuple("nan" if r is None else None for r in ranges)).to_mapping()
        record["head_plane_mount"] = dict(ground_frame="base_footprint", scan_height_above_ground_m=.182,
            scan_vertical_direction_x=0., scan_vertical_direction_y=0., scan_vertical_direction_z=1.,
            exact_transform_stamp_sec=record["stamp_sec"])
        scans.append(record)
    return scans


def fixture(*, kinds=None, shifts=None, transform=PlanarTransform2D(0., 0., 0.)):
    snapshot, _, frame = hint_fixture(transform)
    frame = replace(frame, current_pose=Pose2D(transform.x_m-1.04*math.cos(transform.yaw_rad),
        transform.y_m-1.04*math.sin(transform.yaw_rad), transform.yaw_rad))
    scans = raw_scans(kinds, shifts)
    receipts = []
    for i, scan in enumerate(scans):
        receipts.append(lidar_visibility_receipt_from_scan(receipt_id=f"scan_{i}", survey_id="survey",
            viewpoint_id="current", planning_frame="map", scan_frame="base_scan", scan_topic="/scan",
            map_bundle_sha256=snapshot.map_bundle_sha256, observer_config_sha256="f"*64,
            scan_stamp_sec=scan["stamp_sec"], pose_stamp_sec=scan["stamp_sec"],
            observer_clock_sec=scan["received_at_unix_sec"],
            scan_pose_map=Pose2D(transform.x_m-math.cos(transform.yaw_rad),
                transform.y_m-math.sin(transform.yaw_rad), transform.yaw_rad),
            angle_min_rad=scan["angle_min"], angle_increment_rad=scan["angle_increment"],
            range_min_m=scan["range_min"], range_max_m=scan["range_max"], ranges_m=scan["ranges"],
            frame_provenance=LidarVisibilityFrameProvenance("map", "odom", transform,
                Pose2D(-1., 0., 0.), "b"*64), scan_metadata=LidarScanMetadata.from_mapping(scan["scan_metadata"])))
    return dict(snapshot=snapshot, planning_frame=frame, receipts=tuple(receipts),
        candidate_uids=("candidate_1",), stand_model=model(), now_sec=100.72, not_before_sec=100.,
        mount_evidence=tuple({**s["head_plane_mount"], "stamp_sec": s["stamp_sec"]} for s in scans),
        base_frame="base_footprint")


def test_whole_wall_cannot_be_cropped_into_stand():
    estimates, evidence = assess_current_lidar_targets(**fixture(kinds=["wall"]*8))
    assert not estimates
    assert {s["reason"] for s in evidence["candidate_decisions"]["candidate_1"]["scans"]} == {"non_stand_cluster"}
    assert evidence["keepouts_changed"] is False
    assert evidence["candidate_rejection_authorized"] is False


@pytest.mark.parametrize("kind,reason", [("absent", "unsupported"), ("occluded", "occluded"),
                                         ("ambiguous", "ambiguous_clusters")])
def test_no_current_correspondence_defers_with_evidence(kind, reason):
    estimates, evidence = assess_current_lidar_targets(**fixture(kinds=[kind]*8))
    assert not estimates
    assert {s["reason"] for s in evidence["candidate_decisions"]["candidate_1"]["scans"]} == {reason}


def test_drifted_compact_surface_projects_through_current_transform():
    args = fixture(shifts=[.10]*8, transform=PlanarTransform2D(1., 2., math.pi/2))
    before = copy.deepcopy(args["snapshot"])
    estimates, evidence = assess_current_lidar_targets(**args)
    estimate = estimates["candidate_1"]
    assert estimate == pytest.approx({"x_m": 1., "y_m": 2.1, "uncertainty_m": .08, "policy": POLICY})
    assert args["snapshot"] == before
    decision = evidence["candidate_decisions"]["candidate_1"]
    assert decision["displacement_m"] == pytest.approx(.10)
    assert decision["center_odom"]["x_m"] == pytest.approx(.10)
    assert evidence["stand_axis_authorized"] is False


def test_competing_candidate_correspondence_is_not_resolved_by_uid_order():
    args = fixture(shifts=[.07]*8)
    candidate = args["snapshot"].candidates[0]
    other = replace(candidate, candidate_uid="candidate_2", geometry=replace(candidate.geometry, x_m=.12),
                    source=replace(candidate.source, observation_ids=("other",)))
    args["snapshot"] = replace(args["snapshot"], candidates=(candidate, other))
    estimates, evidence = assess_current_lidar_targets(**args)
    assert not estimates
    assert "ambiguous_current_target_correspondence" in evidence["candidate_decisions"]["candidate_1"]["reasons"]


@pytest.mark.parametrize("count,accepted", [(5, False), (6, True)])
def test_failed_scans_stay_in_support_denominator(count, accepted):
    estimates, evidence = assess_current_lidar_targets(**fixture(kinds=["stand"]*count+["absent"]*(8-count)))
    assert bool(estimates) is accepted
    assert evidence["candidate_decisions"]["candidate_1"]["supported_scan_count"] == count


def test_moving_cluster_centers_cannot_average_into_stable_target():
    estimates, evidence = assess_current_lidar_targets(**fixture(shifts=[-.04, .04]*4))
    assert not estimates
    assert "current_cluster_centers_unstable" in evidence["candidate_decisions"]["candidate_1"]["reasons"]


def test_two_adjacent_beams_support_presence_without_claiming_an_axis():
    args = fixture()
    updated = []
    for receipt in args["receipts"]:
        ranges = tuple(value if index in (49, 50) else None for index, value in enumerate(receipt.ranges_m))
        metadata = replace(receipt.scan_metadata, invalid_range_reasons=tuple("nan" if r is None else None for r in ranges))
        updated.append(replace(receipt, ranges_m=ranges, scan_metadata=metadata))
    args["receipts"] = tuple(updated)
    estimates, evidence = assess_current_lidar_targets(**args)
    assert estimates
    assert all(s["selected_cluster"]["point_count"] == 2 for s in evidence["candidate_decisions"]["candidate_1"]["scans"])
    assert evidence["stand_axis_authorized"] is False


def test_full_rotation_metadata_controls_seam_correspondence():
    for topology, accepted in (("full_rotation", True), ("linear", False)):
        args = fixture()
        count, step = 320, math.tau/320
        ranges = tuple(1/math.cos(i*step) if i in (0, 1, 318, 319) else None for i in range(count))
        metadata = LidarScanMetadata((count-1)*step, 0., .1, topology, (),
            tuple("nan" if r is None else None for r in ranges))
        args["receipts"] = tuple(replace(r, ranges_m=ranges, angle_min_rad=0.,
            angle_increment_rad=step, scan_metadata=metadata) for r in args["receipts"])
        estimates, evidence = assess_current_lidar_targets(**args)
        assert bool(estimates) is accepted
        if not accepted:
            assert "ambiguous_current_target_correspondence" in evidence["candidate_decisions"]["candidate_1"]["reasons"]


def test_distant_laser_plane_failure_does_not_block_supported_near_target():
    args = fixture()
    candidate = args["snapshot"].candidates[0]
    other = replace(candidate, candidate_uid="candidate_2", geometry=replace(candidate.geometry, x_m=2.),
                    source=replace(candidate.source, observation_ids=("other",)))
    args["snapshot"] = replace(args["snapshot"], candidates=(candidate, other))
    args["candidate_uids"] = ("candidate_1", "candidate_2")
    args["mount_evidence"] = tuple({**m, "scan_vertical_direction_x": math.sin(.015),
        "scan_vertical_direction_z": math.cos(.015)} for m in args["mount_evidence"])
    estimates, evidence = assess_current_lidar_targets(**args)
    assert set(estimates) == {"candidate_1"}
    assert "laser_plane_not_inside_measured_head" in evidence["candidate_decisions"]["candidate_2"]["reasons"]


@pytest.mark.parametrize("mutation", ["duplicate", "stale", "floor", "partial", "mount"])
def test_bad_cohort_is_systemic_error_not_candidate_absence(mutation):
    args = fixture()
    if mutation == "duplicate":
        args["receipts"] = (args["receipts"][0],)*8
    elif mutation == "stale":
        args["now_sec"] += 2.
    elif mutation == "floor":
        args["not_before_sec"] += .01
    elif mutation == "partial":
        args["receipts"] = args["receipts"][:7]
    else:
        args["mount_evidence"] = args["mount_evidence"][:7]
    with pytest.raises(ValueError):
        assess_current_lidar_targets(**args)


def captured_fixture(tmp_path, *, planning_shift=0.):
    args = fixture(shifts=[.06]*8)
    config = SimpleNamespace(snapshot=args["snapshot"], plan=replace(_plan(), survey_id="survey"),
        camera_calibration=SimpleNamespace(base_frame="base_footprint"),
        lidar_scan_frame="base_scan", lidar_scan_topic="/scan", measured_stand_model=args["stand_model"])
    times = iter((100., 100.72))
    def capture(request):
        scans = raw_scans(shifts=[.06]*8)
        raw = head_capture_payload(scans, tour_id=request.viewpoint_id, odom_frame="odom",
            base_frame="base_footprint", scan_frame="base_scan", captured_at_unix_sec=100.71)
        path = request.output_dir / "scan_cohort.json"
        write_content_hashed_json(path, raw, hash_field=CAPTURE_HASH)
        clock = iter((100., 100.72))
        return capture_candidate_lidar_view(request, capture_cohort=lambda _: path, clock=lambda: next(clock))
    effects = SimpleNamespace(clock=lambda: next(times), capture_lidar_view=Mock(side_effect=capture))
    planning = args["planning_frame"]
    planning = replace(planning, current_pose=replace(planning.current_pose, x_m=planning.current_pose.x_m+planning_shift))
    estimates, evidence = capture_current_lidar_targets(config, effects, planning,
        {"candidate_1"}, tmp_path / "support")
    return config, effects, estimates, evidence


def test_capture_once_and_replay_exact_source_bound_estimate(tmp_path):
    config, effects, estimates, evidence = captured_fixture(tmp_path)
    effects.capture_lidar_view.assert_called_once()
    assert load_current_lidar_target(evidence["evidence_path"], candidate_uid="candidate_1",
                                    snapshot=config.snapshot) == estimates["candidate_1"]


def test_capture_base_must_match_planning_start(tmp_path):
    with pytest.raises(ValueError, match="base pose differs from planning start"):
        captured_fixture(tmp_path, planning_shift=.02)


@pytest.mark.parametrize("mutation", ["capture", "estimate", "snapshot", "symlink"])
def test_loader_rejects_changed_sources_estimate_or_binding(tmp_path, mutation):
    config, _, _, evidence = captured_fixture(tmp_path)
    path = Path(evidence["evidence_path"])
    snapshot = config.snapshot
    if mutation == "capture":
        source = Path(evidence["capture_path"])
        raw = json.loads(source.read_text())
        raw["scans"][0]["ranges"][50] += .01
        source.write_text(json.dumps(raw))
    elif mutation == "estimate":
        raw = json.loads(path.read_text())
        raw.pop(HASH_FIELD)
        raw["candidate_decisions"]["candidate_1"]["estimate"]["x_m"] += .01
        path.unlink()
        write_content_hashed_json(path, raw, hash_field=HASH_FIELD)
    elif mutation == "snapshot":
        snapshot = replace(snapshot, snapshot_id="different")
    else:
        original = path.with_name("original.json")
        path.rename(original)
        path.symlink_to(original)
    with pytest.raises(ValueError):
        load_current_lidar_target(path, candidate_uid="candidate_1", snapshot=snapshot)
