import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.models import Pose2D  # noqa: E402
from scripts.aufgabe04.navigation.waypoint_controller import (  # noqa: E402
    ControllerConfig,
    compute_waypoint_command,
    forward_resume_target,
)


class WaypointControllerTest(unittest.TestCase):
    def test_resume_selects_next_forward_waypoint(self):
        waypoints = tuple(Pose2D(float(x), 0.0) for x in range(5))
        index, proximity = forward_resume_target(Pose2D(2.1, 0.0), waypoints)
        self.assertEqual(index, 3)
        self.assertAlmostEqual(proximity, 0.1)

    def test_fresh_route_starts_with_first_forward_waypoint(self):
        waypoints = (Pose2D(0.0, 0.0), Pose2D(1.0, 0.0))
        index, proximity = forward_resume_target(Pose2D(0.0, 0.0), waypoints)
        self.assertEqual(index, 1)
        self.assertEqual(proximity, 0.0)

    def test_final_position_requires_configured_final_yaw(self):
        step = compute_waypoint_command(
            Pose2D(1.0, 0.0, 0.0),
            (Pose2D(0.0, 0.0, float("nan")), Pose2D(1.0, 0.0, math.pi)),
            1,
            ControllerConfig(),
        )
        self.assertFalse(step.reached_goal)
        self.assertEqual(step.command.linear_x_mps, 0.0)
        self.assertNotEqual(step.command.angular_z_radps, 0.0)

        aligned = compute_waypoint_command(
            Pose2D(1.0, 0.0, math.pi),
            (Pose2D(0.0, 0.0, float("nan")), Pose2D(1.0, 0.0, math.pi)),
            1,
            ControllerConfig(),
        )
        self.assertTrue(aligned.reached_goal)

    def test_reports_normalized_heading_error_for_rotation_watchdog(self):
        step = compute_waypoint_command(
            Pose2D(0.0, 0.0, math.radians(170.0)),
            (Pose2D(0.0, 0.0), Pose2D(-1.0, -0.1)),
            0,
            ControllerConfig(),
        )

        self.assertEqual(step.target_index, 1)
        self.assertGreaterEqual(step.heading_error_rad, -math.pi)
        self.assertLessEqual(step.heading_error_rad, math.pi)
        self.assertEqual(step.command.linear_x_mps, 0.0)


if __name__ == "__main__":
    unittest.main()
