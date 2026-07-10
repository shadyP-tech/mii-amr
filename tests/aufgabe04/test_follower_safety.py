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
    rotation_progress_failure,
    is_non_allowlistable_direct_cmd_vel_publisher,
    startup_readiness_failure,
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

    def test_startup_readiness_reports_only_missing_inputs(self):
        self.assertEqual(
            startup_readiness_failure(scan_ready=False, odom_ready=True, pose_ready=False),
            "startup timeout waiting for scan, pose",
        )

    def test_rotation_watchdog_stops_timeout_and_no_progress(self):
        self.assertEqual(
            rotation_progress_failure(
                rotation_elapsed_sec=25.1,
                no_progress_elapsed_sec=0.2,
                max_rotation_sec=25.0,
                max_no_progress_sec=3.0,
            ),
            "rotation timeout",
        )
        self.assertEqual(
            rotation_progress_failure(
                rotation_elapsed_sec=5.0,
                no_progress_elapsed_sec=3.1,
                max_rotation_sec=25.0,
                max_no_progress_sec=3.0,
            ),
            "rotation stalled: heading error not decreasing",
        )
        self.assertEqual(
            rotation_progress_failure(
                rotation_elapsed_sec=5.0,
                no_progress_elapsed_sec=0.5,
                max_rotation_sec=25.0,
                max_no_progress_sec=3.0,
            ),
            "",
        )
        self.assertEqual(
            startup_readiness_failure(scan_ready=True, odom_ready=True, pose_ready=True),
            "",
        )

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

    def test_cmd_vel_ownership_never_allows_direct_nav2_publishers(self):
        self_identity = "/aufgabe04_simple_waypoint_follower"

        self.assertEqual(
            cmd_vel_ownership_failure(
                [self_identity, "/behavior_server", "/velocity_smoother"],
                self_identity,
                ["/behavior_server", "/velocity_smoother"],
            ),
            "external cmd_vel publisher during run: /behavior_server, /velocity_smoother",
        )
        self.assertEqual(
            cmd_vel_ownership_failure(
                [self_identity, "/behavior_server", "/teleop_keyboard"],
                self_identity,
                ["/behavior_server"],
            ),
            "external cmd_vel publisher during run: /behavior_server, /teleop_keyboard",
        )

    def test_nav2_direct_publishers_are_never_allowlistable(self):
        self.assertTrue(is_non_allowlistable_direct_cmd_vel_publisher("/behavior_server"))
        self.assertTrue(
            is_non_allowlistable_direct_cmd_vel_publisher("/robot1/velocity_smoother")
        )
        self.assertFalse(is_non_allowlistable_direct_cmd_vel_publisher("/robot1/cmd_vel_mux"))


if __name__ == "__main__":
    unittest.main()
