import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.follower_safety import (  # noqa: E402
    cmd_vel_ownership_failure,
    finite_positive_min,
    initial_pose_failure,
    message_freshness_failure,
    obstacle_failure,
    waypoint_timeout_failure,
)
from scripts.aufgabe04.navigation.models import Pose2D  # noqa: E402


class FollowerSafetyTest(unittest.TestCase):
    def test_message_freshness_distinguishes_missing_stale_and_fresh(self):
        self.assertEqual(
            message_freshness_failure(
                "scan",
                has_message=False,
                receipt_age_sec=None,
                header_age_sec=None,
                max_age_sec=1.0,
            ),
            "missing scan",
        )
        self.assertEqual(
            message_freshness_failure(
                "odom",
                has_message=True,
                receipt_age_sec=0.2,
                header_age_sec=1.5,
                max_age_sec=1.0,
            ),
            "stale odom",
        )
        self.assertEqual(
            message_freshness_failure(
                "tf",
                has_message=True,
                receipt_age_sec=0.2,
                header_age_sec=0.9,
                max_age_sec=1.0,
            ),
            "",
        )

    def test_scan_obstacle_decision_uses_finite_positive_ranges(self):
        self.assertEqual(finite_positive_min([float("inf"), -1.0, 0.0, 0.35]), 0.35)
        self.assertEqual(obstacle_failure([float("inf"), 0.30], 0.20), "")
        self.assertEqual(obstacle_failure([0.19, 0.30], 0.20), "obstacle too close")
        self.assertEqual(obstacle_failure([], 0.20), "")

    def test_initial_pose_and_waypoint_timeout_fail_closed(self):
        first = Pose2D(0.0, 0.0, 0.0)

        self.assertEqual(initial_pose_failure(Pose2D(0.1, 0.0, 0.0), first, 0.35), "")
        self.assertEqual(
            initial_pose_failure(Pose2D(0.4, 0.0, 0.0), first, 0.35),
            "initial pose too far from first waypoint",
        )
        self.assertEqual(waypoint_timeout_failure(44.9, 45.0), "")
        self.assertEqual(waypoint_timeout_failure(45.1, 45.0), "waypoint timeout")

    def test_cmd_vel_ownership_is_exclusive_at_runtime(self):
        self_identity = "/aufgabe04_simple_waypoint_follower"

        self.assertEqual(cmd_vel_ownership_failure([self_identity], self_identity), "")
        self.assertEqual(
            cmd_vel_ownership_failure(
                [self_identity, "/teleop_keyboard", "/controller_server"],
                self_identity,
            ),
            "external cmd_vel publisher during run: /controller_server, /teleop_keyboard",
        )


if __name__ == "__main__":
    unittest.main()
