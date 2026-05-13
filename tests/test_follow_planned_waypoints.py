import argparse
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import follow_planned_waypoints as follower  # noqa: E402


def write_waypoints(path, rows, header=None):
    header = header or ["index", "world_x_m", "world_y_m"]
    with Path(path).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def default_args(**overrides):
    values = {
        "run_id": "test_run",
        "waypoints": Path("results/aufgabe03_waypoints.csv"),
        "linear_speed": follower.DEFAULT_LINEAR_SPEED_MPS,
        "min_linear_speed": follower.DEFAULT_MIN_LINEAR_SPEED_MPS,
        "linear_gain": follower.DEFAULT_LINEAR_GAIN,
        "max_angular_speed": follower.DEFAULT_MAX_ANGULAR_SPEED_RADPS,
        "yaw_gain": follower.DEFAULT_YAW_GAIN,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class FollowPlannedWaypointsTest(unittest.TestCase):
    def test_waypoint_csv_parsing_and_duplicate_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "waypoints.csv"
            write_waypoints(
                path,
                [
                    [0, 0.0, 0.0],
                    [1, 0.1, 0.0],
                    [2, 0.1, 0.0],
                    [3, 0.2, 0.0],
                ],
            )

            waypoints = follower.load_waypoints(path)

        self.assertEqual(len(waypoints), 3)
        self.assertEqual([wp.index for wp in waypoints], [0, 1, 3])
        self.assertEqual((waypoints[1].x, waypoints[1].y), (0.1, 0.0))

    def test_waypoint_csv_requires_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "waypoints.csv"
            write_waypoints(path, [[0, 0.0]], header=["index", "world_x_m"])

            with self.assertRaisesRegex(ValueError, "missing required"):
                follower.load_waypoints(path)

    def test_one_waypoint_after_skip_is_rejected(self):
        waypoints = [
            follower.Waypoint(0, 0.0, 0.0),
            follower.Waypoint(1, 0.2, 0.0),
        ]

        with self.assertRaisesRegex(ValueError, "at least two executable"):
            follower.prepare_executable_waypoints(
                waypoints,
                skip_first=True,
                min_spacing_m=0.0,
            )

    def test_downsampling_preserves_first_final_and_heading_change(self):
        waypoints = [
            follower.Waypoint(1, 0.00, 0.00),
            follower.Waypoint(2, 0.05, 0.00),
            follower.Waypoint(3, 0.10, 0.00),
            follower.Waypoint(4, 0.10, 0.05),
            follower.Waypoint(5, 0.10, 0.20),
        ]

        downsampled = follower.downsample_waypoints(waypoints, min_spacing_m=0.12)

        self.assertEqual([wp.index for wp in downsampled], [1, 3, 5])

    def test_quaternion_to_yaw_and_wraparound_error(self):
        yaw = follower.quaternion_to_yaw_deg(
            0.0,
            0.0,
            math.sin(math.radians(45.0)),
            math.cos(math.radians(45.0)),
        )

        self.assertAlmostEqual(yaw, 90.0)
        self.assertAlmostEqual(
            follower.shortest_angle_delta_deg(179.0, -179.0),
            2.0,
        )

    def test_target_heading_distance_and_tolerances(self):
        pose = follower.Pose2D(0.0, 0.0, 0.0)
        waypoint = follower.Waypoint(1, 0.3, 0.4)

        state = follower.target_state(pose, waypoint)

        self.assertAlmostEqual(state.distance_m, 0.5)
        self.assertAlmostEqual(state.heading_deg, math.degrees(math.atan2(0.4, 0.3)))
        self.assertFalse(
            follower.waypoint_reached(
                state.distance_m,
                is_final=False,
                waypoint_tolerance_m=0.08,
                goal_tolerance_m=0.10,
            )
        )
        self.assertTrue(
            follower.waypoint_reached(
                0.09,
                is_final=True,
                waypoint_tolerance_m=0.08,
                goal_tolerance_m=0.10,
            )
        )

    def test_rotate_hysteresis_mode_switching(self):
        self.assertTrue(
            follower.should_rotate(
                "forward",
                yaw_error_deg=16.0,
                start_threshold_deg=15.0,
                stop_threshold_deg=6.0,
            )
        )
        self.assertTrue(
            follower.should_rotate(
                "rotate",
                yaw_error_deg=7.0,
                start_threshold_deg=15.0,
                stop_threshold_deg=6.0,
            )
        )
        self.assertFalse(
            follower.should_rotate(
                "rotate",
                yaw_error_deg=5.0,
                start_threshold_deg=15.0,
                stop_threshold_deg=6.0,
            )
        )

    def test_velocity_command_clamps_linear_and_angular_speed(self):
        linear, angular = follower.velocity_command(
            distance_m=0.02,
            yaw_error_deg=90.0,
            rotate_mode=False,
            linear_speed_mps=0.05,
            min_linear_speed_mps=0.015,
            linear_gain=0.6,
            max_angular_speed_radps=0.30,
            yaw_gain=1.5,
        )

        self.assertAlmostEqual(linear, 0.015)
        self.assertAlmostEqual(angular, 0.30)

        linear, angular = follower.velocity_command(
            distance_m=1.0,
            yaw_error_deg=-90.0,
            rotate_mode=True,
            linear_speed_mps=0.05,
            min_linear_speed_mps=0.015,
            linear_gain=0.6,
            max_angular_speed_radps=0.30,
            yaw_gain=1.5,
        )

        self.assertEqual(linear, 0.0)
        self.assertAlmostEqual(angular, -0.30)

    def test_forward_scan_hard_and_soft_stop(self):
        hard = follower.evaluate_scan_safety(
            [0.15, 0.8, 0.9],
            angle_min=math.radians(-10),
            angle_increment=math.radians(10),
            range_min=0.1,
            range_max=4.0,
            mode="forward",
            scan_half_angle_deg=35.0,
            hard_stop_range_m=0.16,
            min_scan_range_m=0.24,
            rotation_stop_range_m=0.18,
        )
        soft = follower.evaluate_scan_safety(
            [0.20, 0.21, 0.22, 0.9],
            angle_min=math.radians(-15),
            angle_increment=math.radians(10),
            range_min=0.1,
            range_max=4.0,
            mode="forward",
            scan_half_angle_deg=35.0,
            hard_stop_range_m=0.16,
            min_scan_range_m=0.24,
            rotation_stop_range_m=0.18,
        )

        self.assertFalse(hard.safe)
        self.assertEqual(hard.reason, "hard_stop")
        self.assertFalse(soft.safe)
        self.assertEqual(soft.reason, "soft_stop")

    def test_rotation_scan_uses_full_scan(self):
        result = follower.evaluate_scan_safety(
            [0.5, 0.161],
            angle_min=math.radians(170),
            angle_increment=math.radians(10),
            range_min=0.1,
            range_max=4.0,
            mode="rotate",
            scan_half_angle_deg=35.0,
            hard_stop_range_m=0.16,
            min_scan_range_m=0.24,
            rotation_stop_range_m=0.18,
        )

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "soft_stop")

    def test_no_valid_scan_ranges_is_unsafe(self):
        result = follower.evaluate_scan_safety(
            [float("nan"), float("inf"), 0.05],
            angle_min=0.0,
            angle_increment=0.1,
            range_min=0.1,
            range_max=4.0,
            mode="forward",
            scan_half_angle_deg=35.0,
            hard_stop_range_m=0.16,
            min_scan_range_m=0.24,
            rotation_stop_range_m=0.18,
        )

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "no_valid_scan_ranges")

    def test_stale_age_checks(self):
        self.assertTrue(follower.age_ok(0.2, 0.5))
        self.assertFalse(follower.age_ok(0.6, 0.5))
        self.assertFalse(follower.age_ok(None, 0.5))

    def test_amcl_covariance_indices_and_warn_fail_behavior(self):
        covariance = [0.0] * 36
        covariance[0] = 0.06
        covariance[7] = 0.04
        covariance[35] = 0.11

        self.assertEqual(follower.amcl_covariances(covariance), (0.06, 0.04, 0.11))

        warning = follower.evaluate_amcl_health(
            covariance,
            age_sec=0.1,
            max_age_sec=1.0,
            max_var_x=0.05,
            max_var_y=0.05,
            max_var_yaw=0.10,
            fail_on_bad_localization=False,
        )
        failure = follower.evaluate_amcl_health(
            covariance,
            age_sec=0.1,
            max_age_sec=1.0,
            max_var_x=0.05,
            max_var_y=0.05,
            max_var_yaw=0.10,
            fail_on_bad_localization=True,
        )

        self.assertTrue(warning.ok)
        self.assertIn("high_cov_x", warning.warnings)
        self.assertIn("high_cov_yaw", warning.warnings)
        self.assertFalse(failure.ok)

    def test_ordered_base_frames_supports_fallback_without_duplicates(self):
        self.assertEqual(
            follower.ordered_base_frames("base_footprint", "base_link"),
            ["base_footprint", "base_link"],
        )
        self.assertEqual(
            follower.ordered_base_frames("base_link", "base_link"),
            ["base_link"],
        )

    def test_log_row_generation_for_statuses(self):
        args = default_args()
        scan = follower.ScanSafety(False, "soft_stop", 3, 0.2, 0.21)
        amcl = follower.AmclHealth(True, [], 0.01, 0.02, 0.03, 0.1)
        start = follower.Pose2D(0.0, 0.0, 0.0)
        final = follower.Pose2D(0.2, 0.1, 5.0)
        blocked = follower.Waypoint(2, 0.5, 0.0)

        for status in ["completed", "blocked", "timeout", "failed", "interrupted"]:
            with self.subTest(status=status):
                row = follower.build_log_row(
                    args,
                    waypoint_count=3,
                    reached_count=1,
                    status=status,
                    notes="test",
                    start_pose=start,
                    final_pose=final,
                    blocked_waypoint=blocked if status == "blocked" else None,
                    timeout_waypoint=blocked if status == "timeout" else None,
                    base_frame_used="base_footprint",
                    scan_safety=scan,
                    amcl_health=amcl,
                )
                values = dict(zip(follower.CSV_HEADER, row))
                self.assertEqual(values["status"], status)
                self.assertEqual(values["base_frame_used"], "base_footprint")
                if status == "blocked":
                    self.assertEqual(values["blocked_waypoint_index"], 2)
                if status == "timeout":
                    self.assertEqual(values["timeout_waypoint_index"], 2)

    def test_dry_run_main_avoids_ros_setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "waypoints.csv"
            write_waypoints(
                path,
                [
                    [0, 0.0, 0.0],
                    [1, 0.2, 0.0],
                    [2, 0.4, 0.0],
                ],
            )

            result = follower.main(["--waypoints", str(path), "--dry-run"])

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
