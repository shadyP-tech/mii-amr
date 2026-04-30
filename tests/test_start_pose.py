import csv
import math
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision_tracker"))

import start_pose  # noqa: E402


class StartPoseTest(unittest.TestCase):
    def setUp(self):
        self.ref = start_pose.StartPoseReference(x=1.0, y=2.0, yaw_deg=-179.0)

    def pose(
        self,
        x=1.0,
        y=2.0,
        yaw_deg=-179.0,
        valid_pose=True,
        num_detected=3,
        timestamp_epoch=100.0,
    ):
        return start_pose.TrackerPose(
            timestamp="",
            x=x,
            y=y,
            yaw_rad=math.radians(yaw_deg),
            yaw_deg=yaw_deg,
            valid_pose=valid_pose,
            num_detected=num_detected,
            timestamp_epoch=timestamp_epoch,
            file_mtime=timestamp_epoch,
        )

    def check(self, pose):
        return start_pose.check_start_pose(
            pose,
            ref=self.ref,
            position_tol_m=0.04,
            yaw_tol_deg=4.0,
            max_age_sec=1.0,
            required_markers=3,
            now=100.5,
        )

    def test_exact_reference_pose_passes(self):
        result = self.check(self.pose())
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["position_error_m"], 0.0)
        self.assertAlmostEqual(result["yaw_error_deg"], 0.0)

    def test_position_error_above_tolerance_fails(self):
        result = self.check(self.pose(x=1.041))
        self.assertFalse(result["accepted"])
        self.assertGreater(result["position_error_m"], 0.04)

    def test_yaw_error_above_tolerance_fails(self):
        result = self.check(self.pose(yaw_deg=-174.9))
        self.assertFalse(result["accepted"])
        self.assertGreater(abs(result["yaw_error_deg"]), 4.0)

    def test_yaw_wraparound_uses_shortest_angle(self):
        result = self.check(self.pose(yaw_deg=179.0))
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["yaw_error_deg"], -2.0)

    def test_stale_pose_fails(self):
        result = self.check(self.pose(timestamp_epoch=98.0))
        self.assertFalse(result["accepted"])
        self.assertFalse(result["fresh"])

    def test_invalid_pose_fails(self):
        result = self.check(self.pose(valid_pose=False))
        self.assertFalse(result["accepted"])
        self.assertFalse(result["valid_pose"])

    def test_insufficient_markers_fail(self):
        result = self.check(self.pose(num_detected=2))
        self.assertFalse(result["accepted"])
        self.assertFalse(result["enough_markers"])

    def test_appended_latest_pose_fields_parse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "latest_tracker_pose.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "x",
                    "y",
                    "yaw_rad",
                    "yaw_deg",
                    "valid_pose",
                    "num_detected",
                ])
                writer.writerow([
                    datetime.now().isoformat(),
                    "1.000000",
                    "2.000000",
                    f"{math.radians(179.0):.6f}",
                    "179.000000",
                    "1",
                    "3",
                ])

            pose = start_pose.read_latest_pose(path)
            self.assertIsNotNone(pose)
            self.assertTrue(pose.valid_pose)
            self.assertEqual(pose.num_detected, 3)
            self.assertAlmostEqual(pose.x, 1.0)
            self.assertAlmostEqual(pose.yaw_deg, 179.0)

    def test_latest_pose_path_is_repo_absolute(self):
        config_text = (ROOT / "vision_tracker" / "config.py").read_text()
        self.assertIn("PROJECT_ROOT = os.path.dirname(BASE_DIR)", config_text)
        self.assertIn(
            'LATEST_TRACKER_POSE_FILE = os.path.join(RESULTS_DIR, "latest_tracker_pose.csv")',
            config_text,
        )


if __name__ == "__main__":
    unittest.main()
