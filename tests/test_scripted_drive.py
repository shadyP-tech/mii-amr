import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import scripted_drive  # noqa: E402


class ScriptedDriveConfigTest(unittest.TestCase):
    def test_distance_parser_accepts_experiment_units(self):
        self.assertAlmostEqual(scripted_drive.parse_distance_m("30cm"), 0.30)
        self.assertAlmostEqual(scripted_drive.parse_distance_m("300 mm"), 0.30)
        self.assertAlmostEqual(scripted_drive.parse_distance_m("0.3m"), 0.30)

    def test_motion_configuration_uses_distance_and_speed(self):
        config = scripted_drive.configured_motion({
            "RUN_MODE": "linear-forward",
            "RUN_SPEED": "0.1",
            "RUN_DISTANCE": "30cm",
        })

        self.assertAlmostEqual(config["speed_mps"], 0.1)
        self.assertAlmostEqual(config["distance_m"], 0.3)
        self.assertAlmostEqual(config["duration_sec"], 3.0)

    def test_start_pose_validation_accepts_real_world_aligned_start(self):
        checks = scripted_drive.validation_config({})
        start_pose = {
            "x": 0.5,
            "y": 0.05,
            "yaw_deg": -179.0,
        }

        self.assertEqual(scripted_drive.validate_start_pose(start_pose, checks), [])

    def test_motion_validation_accepts_straight_30cm_run(self):
        checks = scripted_drive.validation_config({})
        start_pose = {
            "x": 0.5,
            "y": 0.05,
            "yaw_deg": 180.0,
        }
        final_pose = {
            "x": 0.2,
            "y": 0.052,
            "yaw_deg": -179.7,
        }

        summary = scripted_drive.motion_summary(start_pose, final_pose)

        self.assertAlmostEqual(summary["forward_m"], 0.3, places=3)
        self.assertEqual(
            scripted_drive.validate_motion(summary, 0.3, checks),
            [],
        )

    def test_motion_validation_rejects_broken_turning_sample(self):
        checks = scripted_drive.validation_config({})
        start_pose = {
            "x": 0.45370191190981723,
            "y": 2.1564552547831455,
            "yaw_deg": 80.58698843351725,
        }
        final_pose = {
            "x": 0.5146724276425879,
            "y": 2.0096499141409296,
            "yaw_deg": 22.11168732823179,
        }

        summary = scripted_drive.motion_summary(start_pose, final_pose)
        errors = scripted_drive.validate_motion(summary, 0.3, checks)

        self.assertGreater(len(errors), 0)
        self.assertTrue(any("forward distance" in error for error in errors))
        self.assertTrue(any("yaw drift" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
