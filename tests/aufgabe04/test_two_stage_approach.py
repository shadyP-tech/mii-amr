import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.compute_qr_facing_pose import main  # noqa: E402
from scripts.aufgabe04.navigation.models import Pose2D  # noqa: E402
from scripts.aufgabe04.navigation.two_stage_approach import (  # noqa: E402
    pre_approach_candidates,
    pre_approach_pose,
    qr_facing_pose_from_camera,
)


class TwoStageApproachTest(unittest.TestCase):
    def test_cli_supports_direct_script_execution(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/aufgabe04/navigation/compute_qr_facing_pose.py"),
                "--help",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--stand-axis-rad", completed.stdout)

    def test_preapproach_depends_on_robot_not_hidden_stand_yaw(self):
        pose = pre_approach_pose(Pose2D(0, 0), Pose2D(1, 0), offset_m=0.3)
        self.assertAlmostEqual(pose.x_m, 0.3)
        self.assertAlmostEqual(pose.y_m, 0.0)
        self.assertAlmostEqual(abs(pose.yaw_rad), math.pi)

    def test_preapproach_candidates_sample_both_directions_without_stand_yaw(self):
        candidates = pre_approach_candidates(
            Pose2D(0, 0), Pose2D(1, 0), offset_m=0.3
        )

        self.assertEqual(len(candidates), 8)
        self.assertAlmostEqual(candidates[0].x_m, 0.3)
        self.assertAlmostEqual(candidates[1].x_m, 0.3 / math.sqrt(2), places=6)
        self.assertAlmostEqual(candidates[1].y_m, 0.3 / math.sqrt(2), places=6)
        self.assertAlmostEqual(candidates[2].y_m, -0.3 / math.sqrt(2), places=6)
        for candidate in candidates:
            heading_to_stand = math.atan2(-candidate.y_m, -candidate.x_m)
            self.assertAlmostEqual(
                math.atan2(
                    math.sin(candidate.yaw_rad - heading_to_stand),
                    math.cos(candidate.yaw_rad - heading_to_stand),
                ),
                0.0,
            )

    def test_camera_qr_side_resolves_axis_normal_toward_observer(self):
        result = qr_facing_pose_from_camera(
            Pose2D(0, 0), Pose2D(1, 0), stand_axis_rad=math.pi / 2,
            side="qr_code_side", offset_m=0.3,
        )
        self.assertAlmostEqual(result.final_qr_approach.x_m, 0.3)
        self.assertAlmostEqual(result.final_qr_approach.y_m, 0.0)
        self.assertAlmostEqual(abs(result.final_qr_approach.yaw_rad), math.pi)

    def test_plain_side_selects_opposite_normal(self):
        result = qr_facing_pose_from_camera(
            Pose2D(0, 0), Pose2D(1, 0), stand_axis_rad=math.pi / 2,
            side="basic_color_side", offset_m=0.3,
        )
        self.assertAlmostEqual(result.final_qr_approach.x_m, -0.3)

    def test_cli_writes_final_pose_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pose.json"
            status = main([
                "--stand-x", "0", "--stand-y", "0", "--robot-x", "1", "--robot-y", "0",
                "--stand-axis-rad", str(math.pi / 2), "--side", "qr_code_side",
                "--output", str(output),
            ])
            self.assertEqual(status, 0)
            self.assertIn("final_qr_approach", output.read_text())


if __name__ == "__main__":
    unittest.main()
