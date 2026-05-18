import argparse
import contextlib
import csv
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "aufgabe03"))

import follow_planned_waypoints as follower  # noqa: E402
import two_stage_waypoint_run as two_stage  # noqa: E402


def write_waypoints(path, rows):
    with Path(path).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "world_x_m", "world_y_m"])
        writer.writerows(rows)


class TwoStageWaypointRunTest(unittest.TestCase):
    def test_cli_parses_modes_overrides_timeouts_and_subprocess_paths(self):
        args = two_stage.parse_args(
            [
                "--dry-run",
                "--run-id",
                "two_stage_test",
                "--localization-mode",
                "known-start",
                "--initial-pose-x",
                "1.0",
                "--initial-pose-y",
                "2.0",
                "--initial-pose-yaw-deg",
                "30.0",
                "--global-localization-service",
                "/robot/reinitialize_global_localization",
                "--navigate-action",
                "/robot/navigate_to_pose",
                "--amcl-validation-timeout-sec",
                "12.0",
                "--known-start-validation-timeout-sec",
                "7.0",
                "--follower-script",
                "custom/follower.py",
                "--python-executable",
                "python-test",
            ]
        )

        self.assertEqual(args.localization_mode, "known-start")
        self.assertEqual(args.initial_pose_x, 1.0)
        self.assertEqual(args.initial_pose_y, 2.0)
        self.assertEqual(args.initial_pose_yaw_deg, 30.0)
        self.assertEqual(args.global_localization_service, "/robot/reinitialize_global_localization")
        self.assertEqual(args.navigate_action, "/robot/navigate_to_pose")
        self.assertEqual(args.amcl_validation_timeout_sec, 12.0)
        self.assertEqual(args.known_start_validation_timeout_sec, 7.0)
        self.assertEqual(args.follower_script, Path("custom/follower.py"))
        self.assertEqual(args.python_executable, "python-test")

    def test_known_start_requires_complete_initial_pose(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                two_stage.parse_args(["--localization-mode", "known-start"])

    def test_staging_goal_uses_waypoint_zero_and_yaw_toward_waypoint_one(self):
        waypoints = [
            two_stage.Waypoint(0, 0.0, 0.0),
            two_stage.Waypoint(1, 0.0, 1.0),
            two_stage.Waypoint(2, 1.0, 1.0),
        ]

        staging = two_stage.staging_goal_from_waypoints(waypoints)

        self.assertEqual(staging.waypoint.index, 0)
        self.assertAlmostEqual(staging.yaw_deg, 90.0)

    def test_waypoint_csv_requires_at_least_two_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "waypoints.csv"
            write_waypoints(path, [[0, 0.0, 0.0]])

            with self.assertRaisesRegex(ValueError, "at least two"):
                two_stage.load_waypoints(path)

    def test_global_preflight_does_not_require_map_to_base_tf(self):
        args = two_stage.parse_args(["--dry-run"])

        requirements = two_stage.required_preflight_interfaces(args)

        self.assertEqual(requirements.services, ["/reinitialize_global_localization"])
        self.assertEqual(requirements.actions, ["/navigate_to_pose"])
        self.assertEqual(requirements.topics, ["/scan"])
        self.assertFalse(requirements.requires_tf_before_localization)

    def test_known_start_initial_pose_message_sets_covariance_and_quaternion(self):
        msg = two_stage.build_initial_pose_message(
            x=1.0,
            y=-0.5,
            yaw_deg=90.0,
            var_x=0.05,
            var_y=0.04,
            var_yaw_rad2=0.1,
            frame_id="map",
        )

        self.assertEqual(msg.header.frame_id, "map")
        self.assertAlmostEqual(msg.pose.pose.position.x, 1.0)
        self.assertAlmostEqual(msg.pose.pose.position.y, -0.5)
        self.assertAlmostEqual(msg.pose.pose.orientation.z, math.sin(math.radians(45.0)))
        self.assertAlmostEqual(msg.pose.pose.orientation.w, math.cos(math.radians(45.0)))
        self.assertEqual(msg.pose.covariance[0], 0.05)
        self.assertEqual(msg.pose.covariance[7], 0.04)
        self.assertEqual(msg.pose.covariance[35], 0.1)

    def test_amcl_covariance_stability_pose_jump_and_timeout_helpers(self):
        covariance = [0.0] * 36
        covariance[0] = 0.01
        covariance[7] = 0.02
        covariance[35] = 0.03

        cov = two_stage.amcl_covariances(covariance)
        self.assertEqual((cov.x, cov.y, cov.yaw_rad2), (0.01, 0.02, 0.03))

        state = two_stage.StabilityState()
        state = two_stage.update_amcl_stability(
            state,
            two_stage.Pose2D(0.0, 0.0, 0.0),
            covariance,
            max_var_x=0.05,
            max_var_y=0.05,
            max_var_yaw_rad2=0.1,
            max_pose_jump_m=0.05,
            max_yaw_jump_deg=10.0,
        )
        state = two_stage.update_amcl_stability(
            state,
            two_stage.Pose2D(0.01, 0.0, 1.0),
            covariance,
            max_var_x=0.05,
            max_var_y=0.05,
            max_var_yaw_rad2=0.1,
            max_pose_jump_m=0.05,
            max_yaw_jump_deg=10.0,
        )
        self.assertEqual(state.stable_count, 2)

        state = two_stage.update_amcl_stability(
            state,
            two_stage.Pose2D(0.50, 0.0, 1.0),
            covariance,
            max_var_x=0.05,
            max_var_y=0.05,
            max_var_yaw_rad2=0.1,
            max_pose_jump_m=0.05,
            max_yaw_jump_deg=10.0,
        )
        self.assertEqual(state.stable_count, 1)
        self.assertEqual(state.reason, "pose_jump_above_threshold")
        self.assertTrue(two_stage.amcl_validation_timed_out(10.0, 71.0, 60.0))

    def test_scan_safety_aborts_on_unsafe_or_insufficient_ranges(self):
        unsafe = two_stage.evaluate_spin_scan_safety(
            [float("nan"), float("inf"), 0.22, 0.17],
            min_scan_range_m=0.18,
            min_valid_scan_count=2,
        )
        insufficient = two_stage.evaluate_spin_scan_safety(
            [float("nan"), float("inf"), 0.25],
            min_scan_range_m=0.18,
            min_valid_scan_count=2,
        )

        self.assertFalse(unsafe.ok)
        self.assertEqual(unsafe.reason, "unsafe_proximity")
        self.assertFalse(insufficient.ok)
        self.assertEqual(insufficient.reason, "insufficient_valid_scan")

    def test_follower_command_uses_path_progress_and_runner_without_shell(self):
        args = two_stage.parse_args(["--dry-run", "--run-id", "two_stage_test"])
        command = two_stage.build_follower_command(args)

        self.assertEqual(command[0], "python3")
        self.assertIn("--start-selection", command)
        self.assertEqual(command[command.index("--start-selection") + 1], "path-progress")
        self.assertIn("--max-amcl-var-yaw", command)

        captured = {}

        class Result:
            returncode = 0

        def fake_runner(cmd, check=False, shell=True):
            captured["cmd"] = cmd
            captured["check"] = check
            captured["shell"] = shell
            return Result()

        result = two_stage.run_follower_command(command, runner=fake_runner)

        self.assertEqual(result.returncode, 0)
        self.assertIs(captured["cmd"], command)
        self.assertFalse(captured["check"])
        self.assertFalse(captured["shell"])

    def test_pose_near_waypoint_zero_selects_waypoint_one_for_path_progress(self):
        waypoints = [
            follower.Waypoint(0, 0.0, 0.0),
            follower.Waypoint(1, 0.5, 0.0),
            follower.Waypoint(2, 1.0, 0.0),
        ]
        pose = follower.Pose2D(0.01, 0.0, 0.0)

        selection = follower.select_executable_waypoints(
            waypoints,
            pose,
            start_selection="path-progress",
            start_on_path_tolerance_m=0.25,
            waypoint_tolerance_m=0.12,
            goal_tolerance_m=0.12,
            min_spacing_m=0.0,
            skip_first=True,
        )

        self.assertEqual(selection.selected_waypoint_index, 1)
        self.assertEqual([waypoint.index for waypoint in selection.waypoints], [1, 2])

    def test_interrupt_cleanup_cancels_goal_and_publishes_stop(self):
        class FakeNode:
            def __init__(self):
                self.cancel_count = 0
                self.stop_count = 0

            def cancel_active_goal(self):
                self.cancel_count += 1

            def stop_repeatedly(self):
                self.stop_count += 1

        node = FakeNode()

        two_stage.cleanup_motion(node)

        self.assertEqual(node.cancel_count, 1)
        self.assertEqual(node.stop_count, 1)

    def test_log_row_contains_failure_status_and_follower_command(self):
        args = two_stage.parse_args(["--dry-run", "--run-id", "two_stage_test"])
        staging = two_stage.StagingGoal(two_stage.Waypoint(0, 0.0, 0.0), 0.0)
        diagnostics = two_stage.RunDiagnostics(
            timestamp="2026-05-18T10:00:00",
            start_wall_time="2026-05-18T10:00:00",
            end_wall_time="2026-05-18T10:00:05",
            duration_sec=5.0,
            status="failed",
            final_status_reason="test failure",
            follower_command="python3 follower.py",
            follower_return_code=1,
        )

        row = two_stage.build_log_row(args, staging, diagnostics)
        values = dict(zip(two_stage.CSV_HEADER, row))

        self.assertEqual(values["status"], "failed")
        self.assertEqual(values["final_status_reason"], "test failure")
        self.assertEqual(values["follower_command"], "python3 follower.py")
        self.assertEqual(values["follower_return_code"], 1)

    def test_dry_run_main_works_without_ros_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "waypoints.csv"
            write_waypoints(path, [[0, 0.0, 0.0], [1, 0.5, 0.0]])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = two_stage.main(
                    [
                        "--waypoints",
                        str(path),
                        "--run-id",
                        "two_stage_dry",
                        "--dry-run",
                    ]
                )

        self.assertEqual(result, 0)
        output = stdout.getvalue()
        self.assertIn("Selected waypoint 0", output)
        self.assertIn("Computed staging yaw", output)
        self.assertIn("Follower command:", output)
        self.assertIn("ROS imports available:", output)


if __name__ == "__main__":
    unittest.main()
