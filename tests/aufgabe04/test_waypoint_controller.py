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
)


class WaypointControllerTest(unittest.TestCase):
    def test_blends_forward_motion_through_corner(self):
        config = ControllerConfig(
            max_linear_mps=0.055,
            max_angular_radps=0.18,
            goal_tolerance_m=0.03,
            lookahead_distance_m=0.18,
        )
        waypoints = (
            Pose2D(0.0, 0.0),
            Pose2D(0.10, 0.0),
            Pose2D(0.10, 0.10),
        )

        step = compute_waypoint_command(Pose2D(0.05, 0.0, 0.0), waypoints, 0, config)

        self.assertFalse(step.reached_goal)
        self.assertGreater(step.command.linear_x_mps, 0.0)
        self.assertGreater(step.command.angular_z_radps, 0.0)
        self.assertEqual(step.pursuit_index, 2)

    def test_large_heading_error_rotates_in_place(self):
        config = ControllerConfig(max_linear_mps=0.055, stop_heading_error_rad=1.0)
        waypoints = (Pose2D(0.0, 0.0), Pose2D(0.30, 0.0))

        step = compute_waypoint_command(Pose2D(0.0, 0.0, math.pi), waypoints, 0, config)

        self.assertEqual(step.command.linear_x_mps, 0.0)
        self.assertLess(step.command.angular_z_radps, 0.0)

    def test_heading_error_scales_linear_speed_continuously(self):
        config = ControllerConfig(max_linear_mps=0.055, slow_heading_error_rad=0.75)
        waypoints = (Pose2D(0.0, 0.0), Pose2D(0.50, 0.0))

        straight = compute_waypoint_command(Pose2D(0.0, 0.0, 0.0), waypoints, 0, config)
        angled = compute_waypoint_command(Pose2D(0.0, 0.0, 0.5), waypoints, 0, config)

        self.assertGreater(straight.command.linear_x_mps, angled.command.linear_x_mps)
        self.assertGreater(angled.command.linear_x_mps, 0.0)

    def test_progress_advancement_is_limited_to_local_route_window(self):
        config = ControllerConfig(
            goal_tolerance_m=0.02,
            lookahead_distance_m=0.10,
            max_progress_advance_m=0.25,
        )
        waypoints = tuple(Pose2D(index * 0.10, 0.0) for index in range(6))

        step = compute_waypoint_command(Pose2D(0.32, 0.0, 0.0), waypoints, 0, config)

        self.assertEqual(step.target_index, 2)
        self.assertLess(step.target_index, 3)


if __name__ == "__main__":
    unittest.main()
