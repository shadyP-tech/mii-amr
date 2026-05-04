import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import real_scripted_drive  # noqa: E402


class RealScriptedDriveConfigTest(unittest.TestCase):
    def test_default_motion_is_clockwise_90_degree_rotation(self):
        motion = real_scripted_drive.configured_motion({})

        self.assertEqual(motion["run_mode"], "rotate-in-place")
        self.assertEqual(motion["direction"], "clockwise")
        self.assertAlmostEqual(motion["command_angle_deg"], -90.0)
        self.assertAlmostEqual(motion["linear_x_mps"], 0.0)
        self.assertAlmostEqual(motion["angular_z_radps"], -0.3)
        self.assertAlmostEqual(motion["duration_sec"], math.pi / 2 / 0.3)
        self.assertEqual(motion["results_csv"], "results/real_rotation_runs.csv")

    def test_counterclockwise_rotation_uses_positive_angular_z(self):
        motion = real_scripted_drive.configured_motion({
            "RUN_MODE": "rotate-in-place",
            "RUN_ANGLE_DEG": "45",
            "RUN_ANGULAR_SPEED": "0.5",
        })

        self.assertEqual(motion["direction"], "counterclockwise")
        self.assertAlmostEqual(motion["angular_z_radps"], 0.5)
        self.assertAlmostEqual(motion["duration_sec"], math.radians(45.0) / 0.5)

    def test_linear_forward_mode_keeps_legacy_result_csv(self):
        motion = real_scripted_drive.configured_motion({
            "RUN_MODE": "linear-forward",
            "RUN_SPEED": "0.1",
            "RUN_DURATION_SEC": "3.0",
        })

        self.assertEqual(motion["run_mode"], "linear-forward")
        self.assertAlmostEqual(motion["linear_x_mps"], 0.1)
        self.assertAlmostEqual(motion["angular_z_radps"], 0.0)
        self.assertAlmostEqual(motion["duration_sec"], 3.0)
        self.assertEqual(motion["results_csv"], "results/real_scripted_drive_runs.csv")

    def test_pose_delta_handles_clockwise_yaw_wrap_and_drift(self):
        start_pose = {"x": 0.2, "y": 0.1, "yaw_deg": 90.0}
        final_pose = {"x": 0.203, "y": 0.096, "yaw_deg": 0.5}

        delta = real_scripted_drive.pose_delta(
            start_pose,
            final_pose,
            command_angle_deg=-90.0,
        )

        self.assertAlmostEqual(delta["yaw_change_deg"], -89.5)
        self.assertAlmostEqual(delta["yaw_error_deg"], 0.5)
        self.assertAlmostEqual(delta["dx"], 0.003)
        self.assertAlmostEqual(delta["dy"], -0.004)
        self.assertAlmostEqual(delta["position_drift_m"], 0.005)

    def test_invalid_rotation_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            real_scripted_drive.configured_motion({
                "RUN_MODE": "rotate-in-place",
                "RUN_ANGLE_DEG": "0",
            })


if __name__ == "__main__":
    unittest.main()
