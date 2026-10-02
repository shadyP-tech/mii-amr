"""Production eight-scan capture must reach the real recovery mount verifier."""

from dataclasses import asdict, replace

import pytest

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.perception.lidar_scan_metadata import LidarScanMetadata
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import capture_candidate_lidar_view
from scripts.aufgabe04.real_robot.candidate.lidar_head_capture import HASH_FIELD, head_capture_payload
from tests.aufgabe04 import test_candidate_lidar_acquisition as acquisition_fixtures
from tests.aufgabe04.test_coverage_visibility_reporting import _plan
from tests.aufgabe04.test_lidar_inspection_hint import line_receipts
from tests.aufgabe04.test_tour_scan_capture import sample


@pytest.fixture
def recovery():
    harness = acquisition_fixtures.CandidateLidarAcquisitionAdapterTest()
    harness.setUp()
    harness.now = 100.
    harness.source.plan = replace(_plan(), survey_id=harness.registry.survey_id)
    try:
        yield harness
    finally:
        harness.doCleanups()


def production_capture(harness, *, mount_count=8, final_height=.182):
    """Use the hashed production producer and adapter, not a fabricated view."""
    views = []

    def capture(request):
        receipt = line_receipts(increment=.01)[0]
        scans = []
        for index in range(8):
            scan_stamp = request.observation_not_before_sec + index*.1
            scan = sample(scan_stamp)
            scan.update(
                angle_min=receipt.angle_min_rad, angle_increment=receipt.angle_increment_rad,
                range_min=receipt.range_min_m, range_max=receipt.range_max_m,
                ranges=list(receipt.ranges_m),
                scan_pose_odom=asdict(receipt.frame_provenance.canonical_scan_pose_odom),
                base_pose_odom=asdict(receipt.frame_provenance.canonical_scan_pose_odom),
                scan_metadata=LidarScanMetadata(
                    receipt.angle_min_rad + (len(receipt.ranges_m)-1)*receipt.angle_increment_rad,
                    0., .1, "linear", (),
                    tuple("nan" if value is None else None for value in receipt.ranges_m),
                ).to_mapping(),
            )
            if index < mount_count:
                scan["head_plane_mount"] = {
                    "ground_frame": request.base_frame,
                    "scan_height_above_ground_m": final_height if index == 7 else .182,
                    "scan_vertical_direction_x": 0., "scan_vertical_direction_y": 0.,
                    "scan_vertical_direction_z": 1., "exact_transform_stamp_sec": scan_stamp,
                }
            scans.append(scan)
        raw = head_capture_payload(
            scans, tour_id=request.viewpoint_id, odom_frame=request.planning_frame.odom_frame,
            base_frame=request.base_frame, scan_frame=request.scan_frame,
            captured_at_unix_sec=scans[-1]["stamp_sec"]+.01,
        )
        path = request.output_dir / "scan_cohort.json"
        write_content_hashed_json(path, raw, hash_field=HASH_FIELD)
        end = scans[-1]["stamp_sec"]+.02
        clock = iter((request.observation_not_before_sec, end))
        view = capture_candidate_lidar_view(
            request, capture_cohort=lambda _: path, clock=lambda: next(clock),
        )
        harness.now = end
        views.append(view)
        return view

    harness.effects.capture_lidar_view = capture
    return views


def test_eight_scan_production_cohort_reaches_fresh_recovery_alignment(recovery):
    views = production_capture(recovery)
    returned, report, hint = recovery.run_adapter()
    assert len(views) == 1
    assert len(views[0].receipts) == len(views[0].mount_evidence) == 8
    assert report["head_alignment_verified"] and report["camera_centered_verified"], report
    assert hint is not None and returned.camera_target_geometry is not None
    stopped = next(event for event in report["history"] if event["event"] == "stopped_observation")
    mounts = stopped["head_observability"]
    assert mounts["accepted"] and mounts["source_scan_count"] == mounts["mount_record_count"] == 8
    assert mounts["source_scan_stamps_sec"] == [receipt.scan_stamp_sec for receipt in views[0].receipts]
    assert len(stopped["source_receipt_sha256s"]) == 8
    recovery.move.assert_not_called()


@pytest.mark.parametrize("mount_count", [0, 3, 7])
def test_incomplete_production_mount_cohort_cannot_recover_or_move(recovery, mount_count):
    views = production_capture(recovery, mount_count=mount_count)
    returned, report, hint = recovery.run_adapter()
    assert len(views[0].receipts) == 8
    assert len(views[0].mount_evidence) == mount_count
    assert report["reason"] == "complete_exact_scan_mount_records_required"
    assert report["recovery_complete"] and not report["head_alignment_verified"]
    assert hint is None and returned.camera_target_geometry is None
    recovery.move.assert_not_called()


def test_eighth_production_mount_still_enforces_measured_head_height(recovery):
    production_capture(recovery, final_height=.25)
    returned, report, hint = recovery.run_adapter()
    assert report["reason"] == "laser_plane_not_inside_measured_head"
    assert report["recovery_complete"] and not report["head_alignment_verified"]
    assert hint is None and returned.camera_target_geometry is None
    recovery.move.assert_not_called()
