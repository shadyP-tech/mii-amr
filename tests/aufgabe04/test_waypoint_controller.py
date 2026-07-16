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
    reverse_staging_is_preferred,
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

    def test_final_waypoint_with_yaw_rotates_before_completion(self):
        config = ControllerConfig(goal_tolerance_m=0.03, heading_tolerance_rad=0.10)
        waypoints = (Pose2D(0.0, 0.0), Pose2D(0.10, 0.0, math.pi / 2.0))

        step = compute_waypoint_command(Pose2D(0.10, 0.0, 0.0), waypoints, 0, config)

        self.assertFalse(step.reached_goal)
        self.assertEqual(step.command.linear_x_mps, 0.0)
        self.assertGreater(step.command.angular_z_radps, 0.0)

    def test_final_waypoint_with_yaw_completes_inside_heading_tolerance(self):
        config = ControllerConfig(goal_tolerance_m=0.03, heading_tolerance_rad=0.10)
        waypoints = (Pose2D(0.0, 0.0), Pose2D(0.10, 0.0, math.pi / 2.0))

        step = compute_waypoint_command(
            Pose2D(0.10, 0.0, math.pi / 2.0 - 0.05), waypoints, 0, config
        )

        self.assertTrue(step.reached_goal)
        self.assertEqual(step.command.linear_x_mps, 0.0)
        self.assertEqual(step.command.angular_z_radps, 0.0)

    def test_nan_final_yaw_keeps_position_only_completion(self):
        config = ControllerConfig(goal_tolerance_m=0.03, heading_tolerance_rad=0.10)
        waypoints = (Pose2D(0.0, 0.0), Pose2D(0.10, 0.0, float("nan")))

        step = compute_waypoint_command(Pose2D(0.10, 0.0, math.pi), waypoints, 0, config)

        self.assertTrue(step.reached_goal)

    def test_lookahead_does_not_cross_into_heading_constrained_corridor(self):
        config = ControllerConfig(
            goal_tolerance_m=0.03,
            lookahead_distance_m=0.30,
            enforce_heading_corridor=True,
        )
        waypoints = (
            Pose2D(0.0, 0.0, float("nan")),
            Pose2D(0.10, 0.0, float("nan")),
            Pose2D(0.10, 0.05, math.pi / 2.0),
            Pose2D(0.10, 0.15, math.pi / 2.0),
        )

        transit = compute_waypoint_command(
            Pose2D(0.0, 0.0, 0.0), waypoints, 0, config
        )
        self.assertEqual(transit.pursuit_index, 1)

        corridor = compute_waypoint_command(
            Pose2D(0.10, 0.0, 0.0), waypoints, 2, config
        )
        self.assertEqual(corridor.command.linear_x_mps, 0.0)
        self.assertGreater(corridor.command.angular_z_radps, 0.0)
        self.assertEqual(corridor.pursuit_index, 2)

    def test_reverse_staging_keeps_body_aligned_then_corridor_drives_forward(self):
        waypoints = (
            Pose2D(0.0, 0.0, float("nan")),
            Pose2D(0.50, 0.0, float("nan")),
            Pose2D(0.50, 0.0, math.pi),
            Pose2D(0.35, 0.0, math.pi),
        )
        config = ControllerConfig(
            goal_tolerance_m=0.02,
            heading_tolerance_rad=0.10,
            enforce_heading_corridor=True,
            reverse_staging=True,
        )

        self.assertTrue(
            reverse_staging_is_preferred(Pose2D(0.0, 0.0, math.pi), waypoints)
        )
        staging = compute_waypoint_command(
            Pose2D(0.0, 0.0, math.pi), waypoints, 0, config
        )
        self.assertLess(staging.command.linear_x_mps, 0.0)
        self.assertAlmostEqual(staging.command.angular_z_radps, 0.0)

        handoff = compute_waypoint_command(
            Pose2D(0.50, 0.0, math.pi), waypoints, 1, config
        )
        self.assertEqual(handoff.command.linear_x_mps, 0.0)
        self.assertEqual(handoff.command.angular_z_radps, 0.0)
        self.assertEqual(handoff.target_index, 2)

        corridor = compute_waypoint_command(
            Pose2D(0.50, 0.0, math.pi), waypoints, handoff.target_index, config
        )
        self.assertGreater(corridor.command.linear_x_mps, 0.0)
        self.assertAlmostEqual(corridor.command.angular_z_radps, 0.0)

    def test_reverse_staging_is_not_selected_when_forward_matches_corridor(self):
        waypoints = (
            Pose2D(0.0, 0.0, float("nan")),
            Pose2D(0.50, 0.0, float("nan")),
            Pose2D(0.50, 0.0, 0.0),
            Pose2D(0.65, 0.0, 0.0),
        )

        self.assertFalse(
            reverse_staging_is_preferred(Pose2D(0.0, 0.0, 0.0), waypoints)
        )


if __name__ == "__main__":
    unittest.main()
