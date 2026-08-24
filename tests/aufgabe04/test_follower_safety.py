import sys
import unittest
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.control.follower_safety import (  # noqa: E402
    NO_VALID_FRONT_SECTOR_SCAN_RANGES,
    NO_VALID_SCAN_RANGES,
    OBSTACLE_TOO_CLOSE,
    cmd_vel_ownership_failure,
    finite_positive_min,
    front_sector_decision,
    initial_pose_failure,
    linear_scale_for_front_clearance,
    message_freshness_failure,
    obstacle_decision,
    obstacle_failure,
    sector_min_distance,
    stuck_progress_failure,
    waypoint_timeout_failure,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D  # noqa: E402


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
        self.assertEqual(
            message_freshness_failure(
                "tf",
                has_message=True,
                receipt_age_sec=0.1,
                header_age_sec=-0.6,
                max_age_sec=1.0,
                max_future_sec=0.25,
            ),
            "future-dated tf",
        )

    def test_scan_obstacle_decision_filters_laser_scan_bounds(self):
        self.assertEqual(finite_positive_min([float("inf"), -1.0, 0.0, 0.35]), 0.35)

        decision = obstacle_decision(
            [0.00000003, 0.30],
            0.20,
            range_min_m=0.12,
            range_max_m=3.5,
        )

        self.assertEqual(decision.stop_reason, "")
        self.assertEqual(decision.nearest_valid_range_m, 0.30)
        self.assertIsNone(decision.nearest_valid_bearing_rad)
        self.assertEqual(decision.valid_sample_count, 1)
        self.assertEqual(decision.rejected_below_min_count, 1)
        self.assertEqual(decision.threshold_m, 0.20)
        self.assertEqual(obstacle_failure([0.00000003, 0.30], 0.20, range_min_m=0.12), "")

    def test_scan_obstacle_decision_stops_on_valid_close_obstacle(self):
        decision = obstacle_decision([0.19, 0.30], 0.20, range_min_m=0.12, range_max_m=3.5)

        self.assertEqual(decision.stop_reason, OBSTACLE_TOO_CLOSE)
        self.assertEqual(decision.nearest_valid_range_m, 0.19)
        self.assertIsNone(decision.nearest_valid_bearing_rad)
        self.assertEqual(obstacle_failure([0.19, 0.30], 0.20, range_min_m=0.12), OBSTACLE_TOO_CLOSE)

    def test_scan_obstacle_decision_counts_above_max_and_non_finite(self):
        decision = obstacle_decision(
            [float("nan"), float("inf"), 4.0, 0.40],
            0.20,
            range_min_m=0.12,
            range_max_m=3.5,
        )

        self.assertEqual(decision.stop_reason, "")
        self.assertEqual(decision.nearest_valid_range_m, 0.40)
        self.assertEqual(decision.valid_sample_count, 1)
        self.assertEqual(decision.rejected_above_max_count, 1)
        self.assertEqual(decision.rejected_non_finite_count, 2)

    def test_scan_obstacle_decision_fails_closed_when_no_valid_ranges(self):
        decision = obstacle_decision(
            [0.0, 0.00000003, float("inf"), 4.0],
            0.20,
            range_min_m=0.12,
            range_max_m=3.5,
        )

        self.assertEqual(decision.stop_reason, NO_VALID_SCAN_RANGES)
        self.assertIsNone(decision.nearest_valid_range_m)
        self.assertEqual(decision.valid_sample_count, 0)
        self.assertEqual(decision.rejected_below_min_count, 2)
        self.assertEqual(decision.rejected_above_max_count, 1)
        self.assertEqual(decision.rejected_non_finite_count, 1)
        self.assertEqual(obstacle_failure([], 0.20), NO_VALID_SCAN_RANGES)

    def test_front_sector_and_slowdown_helpers_filter_laser_scan_bounds(self):
        ranges = [0.40, 0.80, 0.60, 0.90]

        self.assertEqual(sector_min_distance(ranges, 0.0, math.pi / 2.0, 0.0, 0.1), 0.40)
        self.assertEqual(sector_min_distance(ranges, 0.0, math.pi / 2.0, math.pi, 0.1), 0.60)
        self.assertIsNone(sector_min_distance([], 0.0, 1.0, 0.0, 0.5))
        self.assertEqual(linear_scale_for_front_clearance(None, 0.2, 0.4), 1.0)
        self.assertEqual(linear_scale_for_front_clearance(0.2, 0.2, 0.4), 0.0)
        self.assertEqual(linear_scale_for_front_clearance(0.4, 0.2, 0.4), 1.0)
        self.assertAlmostEqual(linear_scale_for_front_clearance(0.3, 0.2, 0.4), 0.5)

        self.assertIsNone(
            sector_min_distance([0.01, 0.40], -0.1, 0.1, -0.1, 0.01, range_min_m=0.12)
        )

    def test_front_sector_decision_clamps_unknown_front_clearance(self):
        decision = front_sector_decision(
            [0.01, 0.80, 0.90],
            -math.pi / 2.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            0.01,
            0.20,
            range_min_m=0.12,
            range_max_m=3.5,
        )

        self.assertEqual(decision.stop_reason, NO_VALID_FRONT_SECTOR_SCAN_RANGES)
        self.assertEqual(decision.valid_sample_count, 0)
        self.assertEqual(decision.rejected_below_min_count, 1)
        self.assertEqual(decision.source, "front_sector")

    def test_front_sector_decision_stops_on_valid_close_front_obstacle(self):
        decision = front_sector_decision(
            [0.19, 0.80, 0.90],
            -math.pi / 2.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            0.01,
            0.20,
            range_min_m=0.12,
            range_max_m=3.5,
        )

        self.assertEqual(decision.stop_reason, OBSTACLE_TOO_CLOSE)
        self.assertEqual(decision.nearest_valid_range_m, 0.19)
        self.assertAlmostEqual(
            decision.nearest_valid_bearing_rad,
            -math.pi / 2.0,
        )

    def test_sector_decision_supports_fail_closed_rear_motion_source(self):
        # 0 rad is index 2 and pi/-pi is index 0/4 for this synthetic scan.
        ranges = [0.19, 0.80, 0.25, 0.80, 0.19]
        rear = front_sector_decision(
            ranges,
            -math.pi,
            math.pi / 2.0,
            math.pi,
            0.1,
            0.20,
            range_min_m=0.12,
            range_max_m=3.5,
            source="rear_sector",
        )
        front = front_sector_decision(
            ranges,
            -math.pi,
            math.pi / 2.0,
            0.0,
            0.1,
            0.20,
            range_min_m=0.12,
            range_max_m=3.5,
        )

        self.assertEqual(rear.stop_reason, OBSTACLE_TOO_CLOSE)
        self.assertEqual(rear.source, "rear_sector")
        self.assertEqual(front.stop_reason, "")
        self.assertEqual(front.nearest_valid_range_m, 0.25)

    def test_initial_pose_and_waypoint_timeout_fail_closed(self):
        first = Pose2D(0.0, 0.0, 0.0)

        self.assertEqual(initial_pose_failure(Pose2D(0.1, 0.0, 0.0), first, 0.35), "")
        self.assertEqual(
            initial_pose_failure(Pose2D(0.4, 0.0, 0.0), first, 0.35),
            "initial pose too far from first waypoint",
        )
        self.assertEqual(waypoint_timeout_failure(44.9, 45.0), "")
        self.assertEqual(waypoint_timeout_failure(45.1, 45.0), "waypoint timeout")

    def test_stuck_progress_requires_motion_and_timeout(self):
        self.assertEqual(stuck_progress_failure(9.0, 8.0, False), "")
        self.assertEqual(stuck_progress_failure(7.9, 8.0, True), "")
        self.assertEqual(stuck_progress_failure(8.1, 8.0, True), "stuck no progress")

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
        self.assertEqual(
            cmd_vel_ownership_failure(
                [self_identity, "/behavior_server", "/velocity_smoother"],
                self_identity,
                ["/behavior_server", "/velocity_smoother"],
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()
