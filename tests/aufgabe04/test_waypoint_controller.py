import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.models import Pose2D  # noqa: E402
from scripts.aufgabe04.navigation.waypoint_controller import (  # noqa: E402
    ControllerConfig,
    StartEgressControlConfig,
    compute_start_egress_vertex_command,
    compute_waypoint_command,
    normalize_angle,
    reverse_staging_is_preferred,
)


class WaypointControllerTest(unittest.TestCase):
    def test_start_egress_lock_pursues_first_vertex_without_lookahead(self):
        config = ControllerConfig(
            goal_tolerance_m=0.08,
            lookahead_distance_m=0.18,
            max_progress_advance_m=0.45,
        )
        waypoints = (
            Pose2D(-0.131011, -0.270103, float("nan")),
            Pose2D(-0.195, -0.115, float("nan")),
            Pose2D(-0.595, -0.115, float("nan")),
        )
        start = Pose2D(-0.131011, -0.270103, -2.702)

        ordinary = compute_waypoint_command(start, waypoints, 0, config)
        locked = compute_start_egress_vertex_command(
            start,
            waypoints,
            1,
            config,
            reach_tolerance_m=0.02,
        )

        self.assertEqual(ordinary.pursuit_index, 2)
        self.assertIsNotNone(locked)
        self.assertEqual(locked.pursuit_index, 1)
        self.assertEqual(locked.target_index, 1)
        self.assertEqual(locked.command.linear_x_mps, 0.0)
        self.assertIsNone(
            compute_start_egress_vertex_command(
                Pose2D(-0.195, -0.115, 1.9),
                waypoints,
                1,
                config,
                reach_tolerance_m=0.02,
            )
        )

        post_egress_pose = Pose2D(-0.2297, -0.0940, 2.4126)
        turn = compute_waypoint_command(
            post_egress_pose,
            waypoints,
            2,
            config,
        )
        expected_error = normalize_angle(
            math.atan2(
                waypoints[2].y_m - post_egress_pose.y_m,
                waypoints[2].x_m - post_egress_pose.x_m,
            )
            - post_egress_pose.yaw_rad
        )
        self.assertEqual(turn.pursuit_index, 2)
        self.assertAlmostEqual(turn.controlled_heading_error_rad, expected_error)
        self.assertGreater(abs(turn.controlled_heading_error_rad), 0.70)

    def test_exact_keepout_egress_rotates_tightly_before_translation(self):
        """A->B egress must not cut the 0.30 m clearance certificate."""

        config = ControllerConfig(
            max_linear_mps=0.055,
            stop_heading_error_rad=1.25,
        )
        egress = StartEgressControlConfig(
            alignment_tolerance_rad=0.10,
            max_linear_mps=0.03,
        )
        waypoints = (
            Pose2D(-0.13217920580955628, -0.2693999965268561, float("nan")),
            Pose2D(-0.19499999999999984, -0.11499999999999977, float("nan")),
            Pose2D(-0.5949999999999998, -0.11499999999999977, float("nan")),
        )
        segment_heading = math.atan2(
            waypoints[1].y_m - waypoints[0].y_m,
            waypoints[1].x_m - waypoints[0].x_m,
        )

        large_error = compute_start_egress_vertex_command(
            Pose2D(waypoints[0].x_m, waypoints[0].y_m, -2.702412021176897),
            waypoints,
            1,
            config,
            egress_config=egress,
        )
        moderate_error = compute_start_egress_vertex_command(
            Pose2D(
                waypoints[0].x_m,
                waypoints[0].y_m,
                normalize_angle(segment_heading + 0.44),
            ),
            waypoints,
            1,
            config,
            egress_config=egress,
        )
        aligned = compute_start_egress_vertex_command(
            Pose2D(
                waypoints[0].x_m,
                waypoints[0].y_m,
                normalize_angle(segment_heading + 0.05),
            ),
            waypoints,
            1,
            config,
            egress_config=egress,
        )

        self.assertIsNotNone(large_error)
        self.assertIsNotNone(moderate_error)
        self.assertIsNotNone(aligned)
        self.assertEqual(large_error.command.linear_x_mps, 0.0)
        self.assertEqual(moderate_error.command.linear_x_mps, 0.0)
        self.assertGreater(aligned.command.linear_x_mps, 0.0)
        self.assertLessEqual(aligned.command.linear_x_mps, 0.03)
        self.assertEqual(large_error.pursuit_index, 1)
        self.assertEqual(moderate_error.pursuit_index, 1)
        self.assertEqual(aligned.pursuit_index, 1)

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
        self.assertEqual(step.target_index, 2)
        self.assertEqual(step.pursuit_index, 2)

    def test_certified_route_pursues_each_vertex_without_corner_shortcut(self):
        config = ControllerConfig(
            goal_tolerance_m=0.03,
            lookahead_distance_m=0.18,
            exact_vertex_pursuit=True,
        )
        waypoints = (
            Pose2D(0.0, 0.0),
            Pose2D(0.10, 0.0),
            Pose2D(0.10, 0.10),
        )

        step = compute_waypoint_command(
            Pose2D(0.05, 0.0, 0.0), waypoints, 1, config
        )

        self.assertFalse(step.reached_goal)
        self.assertEqual(step.target_index, 1)
        self.assertEqual(step.pursuit_index, 1)
        self.assertAlmostEqual(step.command.angular_z_radps, 0.0)

    def test_certified_route_aligns_at_exact_vertex_before_translating(self):
        """Retry-10 C egress must not arc outside its 3 cm route tube."""

        waypoints = (
            Pose2D(0.257876, 0.420872, float("nan")),
            Pose2D(0.255000, 0.385000, float("nan")),
            Pose2D(0.083339, 0.100503, float("nan")),
        )
        previous_segment_heading = math.atan2(
            waypoints[1].y_m - waypoints[0].y_m,
            waypoints[1].x_m - waypoints[0].x_m,
        )
        config = ControllerConfig(
            goal_tolerance_m=0.005,
            heading_tolerance_rad=0.25,
            exact_vertex_pursuit=True,
        )

        turn = compute_waypoint_command(
            Pose2D(
                waypoints[1].x_m,
                waypoints[1].y_m,
                previous_segment_heading,
            ),
            waypoints,
            1,
            config,
        )
        aligned = compute_waypoint_command(
            Pose2D(
                waypoints[1].x_m,
                waypoints[1].y_m,
                math.atan2(
                    waypoints[2].y_m - waypoints[1].y_m,
                    waypoints[2].x_m - waypoints[1].x_m,
                )
                + 0.10,
            ),
            waypoints,
            1,
            config,
        )

        self.assertEqual(turn.target_index, 2)
        self.assertEqual(turn.pursuit_index, 2)
        self.assertAlmostEqual(abs(turn.controlled_heading_error_rad), 0.4629, places=3)
        self.assertEqual(turn.command.linear_x_mps, 0.0)
        self.assertNotEqual(turn.command.angular_z_radps, 0.0)
        self.assertEqual(turn.progress_mode, "exact_vertex_alignment")
        self.assertGreater(aligned.command.linear_x_mps, 0.0)
        self.assertEqual(aligned.progress_mode, "path_tracking")

    def test_ordinary_route_retains_blended_motion_for_same_turn(self):
        waypoints = (
            Pose2D(0.255000, 0.385000, float("nan")),
            Pose2D(0.083339, 0.100503, float("nan")),
        )
        segment_heading = math.atan2(
            waypoints[1].y_m - waypoints[0].y_m,
            waypoints[1].x_m - waypoints[0].x_m,
        )
        step = compute_waypoint_command(
            Pose2D(
                waypoints[0].x_m,
                waypoints[0].y_m,
                segment_heading + 0.4629,
            ),
            waypoints,
            1,
            ControllerConfig(
                goal_tolerance_m=0.005,
                heading_tolerance_rad=0.25,
                lookahead_distance_m=0.0,
                exact_vertex_pursuit=False,
            ),
        )

        self.assertGreater(step.command.linear_x_mps, 0.0)
        self.assertEqual(step.progress_mode, "path_tracking")

    def test_intermediate_and_terminal_physical_tolerances_are_separate(self):
        config = ControllerConfig(
            goal_tolerance_m=0.02,
            terminal_goal_tolerance_m=0.005,
            heading_tolerance_rad=0.25,
            enforce_heading_corridor=True,
            exact_vertex_pursuit=True,
        )
        waypoints = (
            Pose2D(0.00, 0.0, 0.0),
            Pose2D(0.05, 0.0, 0.0),
            Pose2D(0.10, 0.0, 0.0),
        )

        passed_intermediate = compute_waypoint_command(
            Pose2D(0.0608, 0.0, 0.0),
            waypoints,
            1,
            config,
        )
        near_terminal = compute_waypoint_command(
            Pose2D(0.094, 0.0, 0.0),
            waypoints,
            2,
            config,
        )
        at_terminal = compute_waypoint_command(
            Pose2D(0.096, 0.0, 0.0),
            waypoints,
            2,
            config,
        )

        self.assertEqual(passed_intermediate.target_index, 2)
        self.assertGreater(passed_intermediate.command.linear_x_mps, 0.0)
        self.assertFalse(near_terminal.reached_goal)
        self.assertTrue(at_terminal.reached_goal)

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

    def test_long_leg_progress_latches_without_lookahead_reversal(self):
        """Exact e2e_006 leg 2 geometry must keep pursuing waypoint 3."""

        config = ControllerConfig(
            goal_tolerance_m=0.08,
            lookahead_distance_m=0.18,
            max_progress_advance_m=0.45,
        )
        waypoint_2 = Pose2D(-0.6449999999999996, -0.11499999999999977)
        waypoint_3 = Pose2D(
            -1.0323535974295406,
            -0.38939028132540787,
        )
        waypoints = (
            Pose2D(-0.19499999999999984, -0.11499999999999977),
            waypoint_2,
            waypoint_3,
        )
        segment_length = math.hypot(
            waypoint_3.x_m - waypoint_2.x_m,
            waypoint_3.y_m - waypoint_2.y_m,
        )
        self.assertAlmostEqual(segment_length, 0.47469235924695846)
        unit_x = (waypoint_3.x_m - waypoint_2.x_m) / segment_length
        unit_y = (waypoint_3.y_m - waypoint_2.y_m) / segment_length
        segment_yaw = math.atan2(unit_y, unit_x)

        inside_lookahead = Pose2D(
            waypoint_2.x_m + 0.179 * unit_x,
            waypoint_2.y_m + 0.179 * unit_y,
            segment_yaw,
        )
        outside_lookahead = Pose2D(
            waypoint_2.x_m + 0.195 * unit_x,
            waypoint_2.y_m + 0.195 * unit_y,
            segment_yaw,
        )

        first = compute_waypoint_command(
            inside_lookahead,
            waypoints,
            1,
            config,
        )
        second = compute_waypoint_command(
            outside_lookahead,
            waypoints,
            first.target_index,
            config,
        )

        self.assertEqual(first.target_index, 2)
        self.assertEqual(first.pursuit_index, 2)
        self.assertEqual(second.target_index, 2)
        self.assertEqual(second.pursuit_index, 2)
        self.assertAlmostEqual(
            first.distance_to_target_m,
            segment_length - 0.179,
        )
        self.assertAlmostEqual(
            second.distance_to_target_m,
            segment_length - 0.195,
        )
        self.assertGreater(first.command.linear_x_mps, 0.0)
        self.assertGreater(second.command.linear_x_mps, 0.0)
        self.assertAlmostEqual(first.command.angular_z_radps, 0.0, places=9)
        self.assertAlmostEqual(second.command.angular_z_radps, 0.0, places=9)

    def test_long_immediate_successor_is_eligible_but_cap_blocks_later_skip(self):
        config = ControllerConfig(
            goal_tolerance_m=0.02,
            lookahead_distance_m=0.0,
            max_progress_advance_m=0.45,
        )
        waypoints = (
            Pose2D(0.0, 0.0),
            Pose2D(0.47469235924695846, 0.0),
            Pose2D(0.5746923592469585, 0.0),
        )

        step = compute_waypoint_command(
            Pose2D(0.55, 0.0, 0.0),
            waypoints,
            0,
            config,
        )

        self.assertEqual(step.target_index, 1)
        self.assertEqual(step.pursuit_index, 1)

    def test_long_segment_latch_never_crosses_heading_handoff(self):
        config = ControllerConfig(
            goal_tolerance_m=0.08,
            lookahead_distance_m=0.18,
            max_progress_advance_m=0.45,
            enforce_heading_corridor=False,
        )
        waypoints = (
            Pose2D(0.0, 0.0, float("nan")),
            Pose2D(0.50, 0.0, math.pi),
        )

        step = compute_waypoint_command(
            Pose2D(0.179, 0.0, 0.0),
            waypoints,
            0,
            config,
        )

        self.assertEqual(step.target_index, 0)
        self.assertEqual(step.pursuit_index, 0)

    def test_retry09_route_reaches_exact_vertex_before_finite_yaw_handoff(self):
        config = ControllerConfig(
            goal_tolerance_m=0.03,
            lookahead_distance_m=0.18,
            max_progress_advance_m=0.45,
            enforce_heading_corridor=False,
        )
        waypoints = (
            Pose2D(-0.971168, -0.440402, float("nan")),
            Pose2D(-0.495, -0.115, float("nan")),
            Pose2D(-0.021299, -0.00703, -2.316),
        )
        segment_length = math.hypot(
            waypoints[1].x_m - waypoints[0].x_m,
            waypoints[1].y_m - waypoints[0].y_m,
        )
        unit_x = (waypoints[1].x_m - waypoints[0].x_m) / segment_length
        unit_y = (waypoints[1].y_m - waypoints[0].y_m) / segment_length
        before_handoff = Pose2D(
            waypoints[1].x_m - 0.179 * unit_x,
            waypoints[1].y_m - 0.179 * unit_y,
            math.atan2(unit_y, unit_x),
        )

        approach = compute_waypoint_command(
            before_handoff,
            waypoints,
            1,
            config,
        )
        handoff = compute_waypoint_command(
            Pose2D(
                waypoints[1].x_m,
                waypoints[1].y_m,
                before_handoff.yaw_rad,
            ),
            waypoints,
            approach.target_index,
            config,
        )

        self.assertEqual(approach.target_index, 1)
        self.assertEqual(approach.pursuit_index, 1)
        self.assertEqual(handoff.target_index, 2)
        self.assertEqual(handoff.pursuit_index, 2)
        self.assertEqual(handoff.progress_mode, "mode_handoff")
        self.assertEqual(handoff.command.linear_x_mps, 0.0)
        self.assertEqual(handoff.command.angular_z_radps, 0.0)

    def test_final_waypoint_with_yaw_rotates_before_completion(self):
        config = ControllerConfig(goal_tolerance_m=0.03, heading_tolerance_rad=0.10)
        waypoints = (Pose2D(0.0, 0.0), Pose2D(0.10, 0.0, math.pi / 2.0))

        step = compute_waypoint_command(Pose2D(0.10, 0.0, 0.0), waypoints, 0, config)

        self.assertFalse(step.reached_goal)
        self.assertEqual(step.command.linear_x_mps, 0.0)
        self.assertGreater(step.command.angular_z_radps, 0.0)

    def test_sim_sampling_tolerance_enters_terminal_yaw_before_circling(self):
        # Regression from the live Gazebo survey: the base was 1.69 cm from
        # the latched sampling point. A 1 cm gate kept translating and turned
        # left toward the point, away from the required camera-facing yaw.
        pose = Pose2D(0.2740, 0.4240, -2.3000)
        target = Pose2D(0.2806, 0.4089, 1.1295)

        tight = compute_waypoint_command(
            pose,
            (target,),
            0,
            ControllerConfig(
                goal_tolerance_m=0.01,
                heading_tolerance_rad=math.radians(5.0),
            ),
        )
        survey = compute_waypoint_command(
            pose,
            (target,),
            0,
            ControllerConfig(
                goal_tolerance_m=0.03,
                heading_tolerance_rad=math.radians(5.0),
            ),
        )

        self.assertGreater(tight.command.linear_x_mps, 0.0)
        self.assertGreater(tight.command.angular_z_radps, 0.0)
        self.assertEqual(survey.command.linear_x_mps, 0.0)
        self.assertLess(survey.command.angular_z_radps, 0.0)

    def test_final_waypoint_with_yaw_completes_inside_heading_tolerance(self):
        config = ControllerConfig(goal_tolerance_m=0.03, heading_tolerance_rad=0.10)
        waypoints = (Pose2D(0.0, 0.0), Pose2D(0.10, 0.0, math.pi / 2.0))

        step = compute_waypoint_command(
            Pose2D(0.10, 0.0, math.pi / 2.0 - 0.05), waypoints, 1, config
        )

        self.assertTrue(step.reached_goal)
        self.assertEqual(step.command.linear_x_mps, 0.0)
        self.assertEqual(step.command.angular_z_radps, 0.0)

    def test_nan_final_yaw_keeps_position_only_completion(self):
        config = ControllerConfig(goal_tolerance_m=0.03, heading_tolerance_rad=0.10)
        waypoints = (Pose2D(0.0, 0.0), Pose2D(0.10, 0.0, float("nan")))

        step = compute_waypoint_command(Pose2D(0.10, 0.0, math.pi), waypoints, 1, config)

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
