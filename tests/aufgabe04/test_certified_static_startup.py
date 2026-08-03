import unittest

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.simple_waypoint_follower import (
    certified_static_startup_decision,
)


class CertifiedStaticStartupDecisionTest(unittest.TestCase):
    def setUp(self):
        self.route = (
            Pose2D(0.00, 0.00),
            Pose2D(0.05, 0.00),
            Pose2D(0.10, 0.00),
        )

    def test_preserves_vertex_zero_when_pose_is_inside_its_tube(self):
        decision = certified_static_startup_decision(
            Pose2D(-0.01, 0.01),
            self.route,
            tracking_tube_radius_m=0.03,
        )

        self.assertTrue(decision.ok)
        self.assertEqual(decision.target_index, 0)
        self.assertEqual(decision.route_check.active_segment_start_index, 0)
        self.assertEqual(decision.route_check.active_segment_end_index, 0)

    def test_joins_only_the_certified_first_segment(self):
        decision = certified_static_startup_decision(
            Pose2D(0.04, 0.02),
            self.route,
            tracking_tube_radius_m=0.03,
        )

        self.assertTrue(decision.ok)
        self.assertEqual(decision.target_index, 1)
        self.assertEqual(decision.route_check.active_segment_start_index, 0)
        self.assertEqual(decision.route_check.active_segment_end_index, 1)
        self.assertLessEqual(
            decision.route_check.maximum_chord_distance_to_segment_m,
            0.03,
        )

    def test_accepts_the_recorded_stationary_amcl_pose_on_the_first_leg(self):
        recorded_route = (
            Pose2D(-0.445, -0.365),
            Pose2D(-0.495, -0.365),
            Pose2D(-0.545, -0.365),
        )
        decision = certified_static_startup_decision(
            Pose2D(-0.49488549918038205, -0.35208796985052626),
            recorded_route,
            tracking_tube_radius_m=0.03,
        )

        self.assertTrue(decision.ok)
        self.assertEqual(decision.target_index, 1)
        self.assertLess(
            decision.route_check.pose_distance_to_segment_m,
            0.013,
        )

    def test_fails_closed_outside_the_certified_first_segment(self):
        decision = certified_static_startup_decision(
            Pose2D(0.04, 0.031),
            self.route,
            tracking_tube_radius_m=0.03,
        )

        self.assertFalse(decision.ok)
        self.assertIsNone(decision.target_index)
        self.assertEqual(
            decision.route_check.reason,
            "pose left certified route tube",
        )
        self.assertEqual(decision.route_check.active_segment_start_index, 0)
        self.assertEqual(decision.route_check.active_segment_end_index, 1)


if __name__ == "__main__":
    unittest.main()
