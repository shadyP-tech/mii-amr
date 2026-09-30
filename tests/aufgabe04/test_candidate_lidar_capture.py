from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, write_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import (
    HASH_FIELD, CandidateLidarCaptureRequest, CandidateLidarCaptureUnavailableError,
    capture_candidate_lidar_view,
)
from scripts.aufgabe04.real_robot.readiness.tour_scan_contract import HASH_FIELD as COHORT_HASH_FIELD
from tests.aufgabe04 import test_coverage_visibility_reporting as visibility
from tests.aufgabe04.test_tour_scan_capture import payload as cohort_payload


class CandidateLidarCaptureTest(unittest.TestCase):
    def fixture(self, root):
        plan = visibility._plan()
        frame = CandidatePlanningFrame(Pose2D(1.,2.,.3), PlanarTransform2D(1.,2.,.3))
        output = root / "local"
        output.mkdir()
        request = CandidateLidarCaptureRequest(plan, "f"*64, "candidate_0001", "local_view_01",
            output, 100., frame, "base", "laser", "/scan")
        raw = cohort_payload()
        raw["tour_id"] = request.viewpoint_id
        path = output / "scan_cohort.json"
        self.publish(path, raw)
        return request, path, raw

    def publish(self, path, raw):
        path.unlink(missing_ok=True)
        write_content_hashed_json(path, raw, hash_field=COHORT_HASH_FIELD)

    def capture(self, request, path, *, end=100.22):
        clock = iter((100., end))
        return capture_candidate_lidar_view(request, capture_cohort=lambda seen: path,
                                           clock=lambda: next(clock))

    def test_local_cohort_bound_without_mutating_survey(self):
        with tempfile.TemporaryDirectory() as directory:
            request, path, raw = self.fixture(Path(directory))
            self.assertIsNone(request.plan.viewpoint_for(request.viewpoint_id))
            result = self.capture(request, path)
            self.assertEqual(len(result.receipts), 3)
            self.assertEqual(result.candidate_snapshot_sha256, request.candidate_snapshot_sha256)
            self.assertEqual(result.latest_base_pose_odom, Pose2D(0.,0.,0.))
            self.assertEqual(result.pose_stamp_sec, 100.2)
            self.assertEqual(result.captured_at_unix_sec, 100.21)
            self.assertEqual(result.mount_evidence, ())
            receipt = result.receipts[0]
            self.assertEqual(receipt.ranges_m, (.5,None,None,None,None,.75))
            self.assertAlmostEqual(receipt.scan_pose_map.x_m, 1.+.04*math.cos(.3))
            self.assertAlmostEqual(receipt.scan_pose_map.y_m, 2.+.04*math.sin(.3))
            self.assertEqual(receipt.frame_provenance.source_evidence_id, result.capture_sha256)
            bound = load_content_hashed_json(result.evidence_path, hash_field=HASH_FIELD)
            self.assertEqual(bound["source_evidence_kind"], "stationary_scan_cohort_sha256")
            self.assertFalse(bound["motion_authorized"])
            self.assertFalse(bound["stand_axis_authorized"])

    def test_exact_scan_mount_evidence_remains_bound_to_source_stamps(self):
        with tempfile.TemporaryDirectory() as directory:
            request, path, raw = self.fixture(Path(directory))
            request = replace(request, base_frame="base_footprint")
            raw["base_frame"] = "base_footprint"
            for scan in raw["scans"]:
                scan["head_plane_mount"] = {
                    "ground_frame": "base_footprint", "scan_height_above_ground_m": .25,
                    "scan_vertical_direction_x": 0., "scan_vertical_direction_y": 0.,
                    "scan_vertical_direction_z": 1., "exact_transform_stamp_sec": scan["stamp_sec"],
                }
            self.publish(path, raw)
            result = self.capture(request, path)
            expected = tuple({**scan["head_plane_mount"], "stamp_sec": scan["stamp_sec"]}
                             for scan in raw["scans"])
            self.assertEqual(result.mount_evidence, expected)
            bound = load_content_hashed_json(result.evidence_path, hash_field=HASH_FIELD)
            self.assertEqual(bound["mount_evidence"], list(expected))

    def test_partial_mount_evidence_is_not_filled_from_another_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            request, path, raw = self.fixture(Path(directory))
            request = replace(request, base_frame="base_footprint")
            raw["base_frame"] = "base_footprint"
            raw["scans"][0]["head_plane_mount"] = {
                "ground_frame": "base_footprint", "scan_height_above_ground_m": .25,
                "scan_vertical_direction_x": 0., "scan_vertical_direction_y": 0.,
                "scan_vertical_direction_z": 1., "exact_transform_stamp_sec": 100.,
            }
            self.publish(path, raw)
            result = self.capture(request, path)
            self.assertEqual(len(result.mount_evidence), 1)
            self.assertEqual(result.mount_evidence[0]["stamp_sec"], 100.)

    def test_prearrival_cohort_rejected_not_trimmed_to_two_scans(self):
        with tempfile.TemporaryDirectory() as directory:
            request, path, raw = self.fixture(Path(directory))
            request = replace(request, observation_not_before_sec=100.01)
            with self.assertRaisesRegex(CandidateLidarCaptureUnavailableError, "before the observation floor"):
                self.capture(request, path)

    def test_source_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            request, path, raw = self.fixture(Path(directory))
            persisted = json.loads(path.read_text())
            persisted["scans"][0]["ranges"][0] = .8
            path.write_text(json.dumps(persisted))
            with self.assertRaises(ValueError) as error:
                self.capture(request, path)
            self.assertNotIsInstance(error.exception, CandidateLidarCaptureUnavailableError)

    def test_exact_source_time_and_stationarity_revalidated_after_hash(self):
        mutations = (
            lambda raw: raw["scans"][0].update(scan_pose_stamp_sec=99.),
            lambda raw: raw["scans"][1].update(stamp_sec=100.),
            lambda raw: raw["scans"][-1]["base_pose_odom"].update(x_m=.016),
            lambda raw: raw["scans"][-1]["scan_pose_odom"].update(x_m=.046),
            lambda raw: raw["scans"].pop(),
        )
        for index, change in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                request, path, raw = self.fixture(Path(directory))
                change(raw)
                self.publish(path, raw)
                with self.assertRaises(ValueError) as error:
                    self.capture(request, path)
                self.assertNotIsInstance(error.exception, CandidateLidarCaptureUnavailableError)

    def test_frame_and_view_identity_mismatch_rejected(self):
        for key in ("tour_id", "odom_frame", "base_frame", "scan_frame", "artifact_kind"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                request, path, raw = self.fixture(Path(directory))
                raw[key] = "wrong"
                self.publish(path, raw)
                with self.assertRaisesRegex(ValueError, "identity differs") as error:
                    self.capture(request, path)
                self.assertNotIsInstance(error.exception, CandidateLidarCaptureUnavailableError)

    def test_old_completion_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            request, path, raw = self.fixture(Path(directory))
            with self.assertRaisesRegex(CandidateLidarCaptureUnavailableError, "stale"):
                self.capture(request, path, end=100.46)

    def test_future_cohort_remains_hard_validation_error(self):
        with tempfile.TemporaryDirectory() as directory:
            request, path, raw = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "future-dated") as error:
                self.capture(request, path, end=100.1)
            self.assertNotIsInstance(error.exception, CandidateLidarCaptureUnavailableError)

    def test_symlink_and_external_cohort_rejected(self):
        for kind in ("symlink", "external"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                request, path, raw = self.fixture(root)
                other = (request.output_dir if kind == "symlink" else root) / "other.json"
                if kind == "symlink":
                    other.symlink_to(path)
                else:
                    other.write_bytes(path.read_bytes())
                with self.assertRaises(ValueError):
                    self.capture(request, other)

    def test_existing_evidence_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            request, path, raw = self.fixture(Path(directory))
            result = self.capture(request, path)
            before = result.evidence_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "must be fresh"):
                self.capture(request, path)
            self.assertEqual(before, result.evidence_path.read_bytes())

    def test_invalid_request_bindings_fail_before_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            request, path, raw = self.fixture(Path(directory))
            for change in ({"candidate_snapshot_sha256": "invalid"}, {"candidate_uid": "../bad"},
                           {"observation_not_before_sec": float("nan")},
                           {"planning_frame": replace(request.planning_frame, map_frame="other_map")}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.capture(replace(request, **change), path)


if __name__ == "__main__":
    unittest.main()
