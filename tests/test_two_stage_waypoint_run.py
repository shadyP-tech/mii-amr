import argparse
import contextlib
import csv
import io
import math
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "aufgabe03"))

import arena_active_spin  # noqa: E402
import follow_planned_waypoints as follower  # noqa: E402
from two_stage_waypoint import cli as two_stage_cli  # noqa: E402
from two_stage_waypoint import experiment_log as two_stage_log  # noqa: E402
from two_stage_waypoint import model as two_stage_model  # noqa: E402
from two_stage_waypoint import pure as two_stage_pure  # noqa: E402
from two_stage_waypoint import ros_runtime as two_stage_ros  # noqa: E402


def write_waypoints(path, rows):
    with Path(path).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "world_x_m", "world_y_m"])
        writer.writerows(rows)


def stamp_from_sec(stamp_sec):
    seconds = int(stamp_sec)
    nanoseconds = int((stamp_sec - seconds) * 1_000_000_000)
    return argparse.Namespace(sec=seconds, nanosec=nanoseconds)


def make_transform(stamp_sec=None, x=0.0, y=0.0, yaw_deg=0.0):
    if stamp_sec is None:
        stamp_sec = time.time()
    half_yaw = math.radians(yaw_deg) / 2.0
    return argparse.Namespace(
        header=argparse.Namespace(stamp=stamp_from_sec(stamp_sec)),
        transform=argparse.Namespace(
            translation=argparse.Namespace(x=x, y=y, z=0.0),
            rotation=argparse.Namespace(
                x=0.0,
                y=0.0,
                z=math.sin(half_yaw),
                w=math.cos(half_yaw),
            ),
        ),
    )


def fake_tf_node(tf_buffer, **overrides):
    args = argparse.Namespace(
        map_frame="map",
        base_frame="base_footprint",
        fallback_base_frame="base_link",
        tf_lookup_timeout_sec=0.05,
        tf_lookup_retry_period_sec=0.001,
        tf_ready_timeout_sec=0.05,
        max_pose_age_sec=10.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    node = argparse.Namespace(args=args, tf_buffer=tf_buffer, selected_base_frame="")
    node.transform_age_sec = two_stage_ros.TwoStageCoordinator.transform_age_sec.__get__(
        node,
        type(node),
    )
    node.lookup_robot_pose_tf = two_stage_ros.TwoStageCoordinator.lookup_robot_pose_tf.__get__(
        node,
        type(node),
    )
    node.lookup_pose = two_stage_ros.TwoStageCoordinator.lookup_pose.__get__(node, type(node))
    node.validate_post_localization_tf = (
        two_stage_ros.TwoStageCoordinator.validate_post_localization_tf.__get__(
            node,
            type(node),
        )
    )
    return node


class TwoStageWaypointRunTest(unittest.TestCase):
    def test_cli_parses_arena_prior_overrides_and_subprocess_paths(self):
        args = two_stage_cli.parse_args(
            [
                "--dry-run",
                "--run-id",
                "arena_prior_test",
                "--navigate-action",
                "/robot/navigate_to_pose",
                "--arena-active-validation-timeout-sec",
                "12.0",
                "--amcl-settle-min-sec",
                "4.5",
                "--tf-ready-timeout-sec",
                "9.0",
                "--tf-lookup-timeout-sec",
                "11.0",
                "--tf-lookup-retry-period-sec",
                "0.2",
                "--follower-script",
                "custom/follower.py",
                "--python-executable",
                "python-test",
                "--arena-active-allow-extra-cmd-vel-publishers",
                "--arena-active-enable-center-reposition",
                "--arena-active-center-reposition-target-nearest-short-wall-range-m",
                "1.45",
                "--arena-active-center-reposition-max-step-m",
                "0.7",
                "--arena-force-short-wall-side",
                "axis_positive",
                "--arena-force-short-wall-type",
                "clean",
            ]
        )

        self.assertFalse(hasattr(args, "localization_mode"))
        self.assertEqual(
            args.results_csv,
            Path("results/aufgabe03/aufgabe03_arena_prior_two_stage_runs.csv"),
        )
        self.assertEqual(args.navigate_action, "/robot/navigate_to_pose")
        self.assertEqual(args.arena_active_validation_timeout_sec, 12.0)
        self.assertEqual(args.amcl_settle_min_sec, 4.5)
        self.assertEqual(args.tf_ready_timeout_sec, 9.0)
        self.assertEqual(args.tf_lookup_timeout_sec, 11.0)
        self.assertEqual(args.tf_lookup_retry_period_sec, 0.2)
        self.assertEqual(args.follower_script, Path("custom/follower.py"))
        self.assertEqual(args.python_executable, "python-test")
        self.assertTrue(args.arena_active_allow_extra_cmd_vel_publishers)
        self.assertTrue(args.arena_active_enable_center_reposition)
        self.assertEqual(
            args.arena_active_center_reposition_target_nearest_short_wall_range_m,
            1.45,
        )
        self.assertEqual(args.arena_active_center_reposition_max_step_m, 0.7)
        self.assertEqual(args.arena_force_short_wall_side, "axis_positive")
        self.assertEqual(args.arena_force_short_wall_type, "clean")

    def test_removed_legacy_flags_are_rejected(self):
        legacy_argvs = [
            ["--localization-mode", "arena-active"],
            ["--global-localization-service", "/reinitialize_global_localization"],
            ["--initial-pose-x", "0.0"],
            ["--known-start-validation-timeout-sec", "7.0"],
            ["--localization-spin-deg", "360"],
            ["--arena-active-on-failure", "global"],
            ["--arena-active-dry-run"],
        ]

        for argv in legacy_argvs:
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        two_stage_cli.parse_args(argv)

    def test_arena_active_force_short_wall_requires_side_and_type(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                two_stage_cli.parse_args(
                    [
                        "--arena-force-short-wall-side",
                        "axis_positive",
                    ]
                )

    def test_arena_prior_preflight_requires_scan_and_nav_for_real_runs(self):
        args = two_stage_cli.parse_args(["--dry-run"])

        requirements = two_stage_pure.required_preflight_interfaces(args)

        self.assertEqual(requirements.actions, ["/navigate_to_pose"])
        self.assertEqual(requirements.topics, ["/scan"])
        self.assertFalse(hasattr(requirements, "services"))
        self.assertFalse(hasattr(requirements, "requires_tf_before_localization"))

    def test_arena_active_diagnostics_path_defaults_next_to_results_csv(self):
        args = two_stage_cli.parse_args(
            [
                "--dry-run",
                "--run-id",
                "arena_prior_test",
                "--results-csv",
                "custom/logs/two_stage_runs.csv",
            ]
        )

        path = two_stage_pure.arena_active_diagnostics_path(args)

        self.assertEqual(
            path,
            Path("custom/logs/arena_prior_test_arena_active_result.json"),
        )

    def test_arena_active_diagnostics_path_allows_explicit_override(self):
        args = two_stage_cli.parse_args(
            [
                "--dry-run",
                "--run-id",
                "arena_prior_test",
                "--results-csv",
                "custom/logs/two_stage_runs.csv",
                "--arena-active-diagnostics-json",
                "diagnostics/arena.json",
            ]
        )

        self.assertEqual(
            two_stage_pure.arena_active_diagnostics_path(args),
            Path("diagnostics/arena.json"),
        )

    def test_arena_active_pose_prior_covariance_is_clamped_and_validated(self):
        covariance = [0.0] * 36
        covariance[0] = 1e-8
        covariance[7] = 1e-8
        covariance[35] = 1e-8
        pose_prior = arena_active_spin.PosePrior(
            x_m=0.1,
            y_m=0.2,
            yaw_rad=0.3,
            covariance=covariance,
        )

        var_x, var_y, var_yaw = two_stage_pure.validate_pose_prior_for_initialpose(pose_prior)

        self.assertEqual(var_x, two_stage_model.MIN_ARENA_ACTIVE_VAR_XY)
        self.assertEqual(var_y, two_stage_model.MIN_ARENA_ACTIVE_VAR_XY)
        self.assertEqual(var_yaw, two_stage_model.MIN_ARENA_ACTIVE_VAR_YAW_RAD2)

    def test_arena_active_invalid_pose_prior_is_rejected(self):
        covariance = [0.0] * 36
        covariance[0] = 0.01
        covariance[7] = 0.01
        covariance[35] = -0.1
        pose_prior = arena_active_spin.PosePrior(
            x_m=0.1,
            y_m=float("nan"),
            yaw_rad=0.3,
            covariance=covariance,
        )

        with self.assertRaisesRegex(RuntimeError, "non-finite pose"):
            two_stage_pure.validate_pose_prior_for_initialpose(pose_prior)

    def test_staging_goal_uses_waypoint_zero_and_yaw_toward_waypoint_one(self):
        waypoints = [
            two_stage_model.Waypoint(0, 0.0, 0.0),
            two_stage_model.Waypoint(1, 0.0, 1.0),
            two_stage_model.Waypoint(2, 1.0, 1.0),
        ]

        staging = two_stage_pure.staging_goal_from_waypoints(waypoints)

        self.assertEqual(staging.waypoint.index, 0)
        self.assertAlmostEqual(staging.yaw_deg, 90.0)

    def test_waypoint_csv_requires_at_least_two_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "waypoints.csv"
            write_waypoints(path, [[0, 0.0, 0.0]])

            with self.assertRaisesRegex(ValueError, "at least two"):
                two_stage_pure.load_waypoints(path)

    def test_initial_pose_message_sets_covariance_and_quaternion(self):
        msg = two_stage_ros.build_initial_pose_message(
            x=1.0,
            y=-0.5,
            yaw_deg=90.0,
            var_x=0.05,
            var_y=0.04,
            var_yaw_rad2=0.1,
            frame_id="map",
        )

        self.assertEqual(msg.header.frame_id, "map")
        self.assertEqual(msg.header.stamp.sec, 0)
        self.assertEqual(msg.header.stamp.nanosec, 0)
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

        cov = two_stage_pure.amcl_covariances(covariance)
        self.assertEqual((cov.x, cov.y, cov.yaw_rad2), (0.01, 0.02, 0.03))

        state = two_stage_model.StabilityState()
        state = two_stage_pure.update_amcl_stability(
            state,
            two_stage_model.Pose2D(0.0, 0.0, 0.0),
            covariance,
            max_var_x=0.05,
            max_var_y=0.05,
            max_var_yaw_rad2=0.1,
            max_pose_jump_m=0.05,
            max_yaw_jump_deg=10.0,
            sample_sec=10.0,
        )
        state = two_stage_pure.update_amcl_stability(
            state,
            two_stage_model.Pose2D(0.01, 0.0, 1.0),
            covariance,
            max_var_x=0.05,
            max_var_y=0.05,
            max_var_yaw_rad2=0.1,
            max_pose_jump_m=0.05,
            max_yaw_jump_deg=10.0,
            sample_sec=11.0,
        )
        self.assertEqual(state.stable_count, 2)
        self.assertAlmostEqual(state.quiet_duration_sec, 1.0)
        self.assertTrue(two_stage_pure.amcl_stability_satisfied(state, 2, 1.0))
        self.assertFalse(two_stage_pure.amcl_stability_satisfied(state, 2, 1.5))

        state = two_stage_pure.update_amcl_stability(
            state,
            two_stage_model.Pose2D(0.50, 0.0, 1.0),
            covariance,
            max_var_x=0.05,
            max_var_y=0.05,
            max_var_yaw_rad2=0.1,
            max_pose_jump_m=0.05,
            max_yaw_jump_deg=10.0,
            sample_sec=12.0,
        )
        self.assertEqual(state.stable_count, 1)
        self.assertEqual(state.reason, "pose_jump_above_threshold")
        self.assertAlmostEqual(state.quiet_duration_sec, 0.0)
        self.assertFalse(two_stage_pure.amcl_stability_satisfied(state, 1, 0.0))

        state = two_stage_pure.update_amcl_stability(
            state,
            two_stage_model.Pose2D(0.51, 0.0, 1.5),
            covariance,
            max_var_x=0.05,
            max_var_y=0.05,
            max_var_yaw_rad2=0.1,
            max_pose_jump_m=0.05,
            max_yaw_jump_deg=10.0,
            sample_sec=14.0,
        )
        self.assertEqual(state.stable_count, 2)
        self.assertEqual(state.reason, "stable")
        self.assertAlmostEqual(state.quiet_duration_sec, 2.0)
        self.assertTrue(two_stage_pure.amcl_stability_satisfied(state, 2, 2.0))
        self.assertTrue(two_stage_pure.amcl_stability_satisfied(state, 2, 3.0, now_sec=15.0))
        self.assertTrue(two_stage_pure.amcl_validation_timed_out(10.0, 71.0, 60.0))

    def test_amcl_stability_repeated_jumps_never_satisfy_settle_gate(self):
        covariance = [0.0] * 36
        covariance[0] = 0.01
        covariance[7] = 0.02
        covariance[35] = 0.03

        state = two_stage_model.StabilityState()
        poses = [
            two_stage_model.Pose2D(0.0, 0.0, 0.0),
            two_stage_model.Pose2D(0.20, 0.0, 0.0),
            two_stage_model.Pose2D(0.40, 0.0, 0.0),
            two_stage_model.Pose2D(0.60, 0.0, 0.0),
        ]
        for index, pose in enumerate(poses):
            state = two_stage_pure.update_amcl_stability(
                state,
                pose,
                covariance,
                max_var_x=0.05,
                max_var_y=0.05,
                max_var_yaw_rad2=0.1,
                max_pose_jump_m=0.05,
                max_yaw_jump_deg=10.0,
                sample_sec=20.0 + index,
            )

        self.assertEqual(state.reason, "pose_jump_above_threshold")
        self.assertEqual(state.stable_count, 1)
        self.assertGreater(state.max_pose_jump_m, 0.05)
        self.assertFalse(two_stage_pure.amcl_stability_satisfied(state, 1, 0.0))

    def test_amcl_stability_bad_covariance_resets_quiet_window(self):
        good_covariance = [0.0] * 36
        good_covariance[0] = 0.01
        good_covariance[7] = 0.02
        good_covariance[35] = 0.03
        bad_covariance = list(good_covariance)
        bad_covariance[7] = 0.20

        state = two_stage_model.StabilityState()
        state = two_stage_pure.update_amcl_stability(
            state,
            two_stage_model.Pose2D(0.0, 0.0, 0.0),
            good_covariance,
            max_var_x=0.05,
            max_var_y=0.05,
            max_var_yaw_rad2=0.1,
            max_pose_jump_m=0.05,
            max_yaw_jump_deg=10.0,
            sample_sec=30.0,
        )
        state = two_stage_pure.update_amcl_stability(
            state,
            two_stage_model.Pose2D(0.01, 0.0, 0.0),
            bad_covariance,
            max_var_x=0.05,
            max_var_y=0.05,
            max_var_yaw_rad2=0.1,
            max_pose_jump_m=0.05,
            max_yaw_jump_deg=10.0,
            sample_sec=31.0,
        )

        self.assertEqual(state.reason, "covariance_above_threshold")
        self.assertEqual(state.stable_count, 0)
        self.assertIsNone(state.stable_since_sec)
        self.assertEqual(state.quiet_duration_sec, 0.0)
        self.assertFalse(two_stage_pure.amcl_stability_satisfied(state, 1, 0.0))

    def test_scan_safety_aborts_on_unsafe_or_insufficient_ranges(self):
        unsafe = two_stage_pure.evaluate_spin_scan_safety(
            [float("nan"), float("inf"), 0.22, 0.17],
            min_scan_range_m=0.18,
            min_valid_scan_count=2,
        )
        insufficient = two_stage_pure.evaluate_spin_scan_safety(
            [float("nan"), float("inf"), 0.25],
            min_scan_range_m=0.18,
            min_valid_scan_count=2,
        )

        self.assertFalse(unsafe.ok)
        self.assertEqual(unsafe.reason, "unsafe_proximity")
        self.assertFalse(insufficient.ok)
        self.assertEqual(insufficient.reason, "insufficient_valid_scan")

    def test_preflight_requires_initial_pose_subscriber_before_motion(self):
        class FakeRclpy:
            spin_count = 0

            @staticmethod
            def ok():
                return True

            @classmethod
            def spin_once(cls, _node, timeout_sec=0.0):
                cls.spin_count += 1
                time.sleep(min(timeout_sec, 0.001))

        class FakeActionClient:
            def __init__(self):
                self.waited = False

            def wait_for_server(self, timeout_sec=0.0):
                self.waited = True
                self.timeout_sec = timeout_sec
                return True

        class FakePublisher:
            def get_subscription_count(self):
                return 0

        original_rclpy = two_stage_ros.rclpy
        two_stage_ros.rclpy = FakeRclpy
        try:
            args = two_stage_cli.parse_args(
                [
                    "--dry-run",
                    "--preflight-timeout-sec",
                    "0.005",
                ]
            )
            node = argparse.Namespace(
                args=args,
                navigate_client=FakeActionClient(),
                initial_pose_pub=FakePublisher(),
                last_scan=argparse.Namespace(
                    ranges=[0.5] * 30,
                    range_min=0.0,
                    range_max=4.0,
                ),
                last_scan_received_sec=time.time(),
            )
            node.wait_for_fresh_scan = (
                two_stage_ros.TwoStageCoordinator.wait_for_fresh_scan.__get__(
                    node,
                    type(node),
                )
            )
            node.current_scan_safety = (
                two_stage_ros.TwoStageCoordinator.current_scan_safety.__get__(
                    node,
                    type(node),
                )
            )
            node.wait_for_initial_pose_subscriber = (
                two_stage_ros.TwoStageCoordinator.wait_for_initial_pose_subscriber.__get__(
                    node,
                    type(node),
                )
            )
            node.preflight_before_motion = (
                two_stage_ros.TwoStageCoordinator.preflight_before_motion.__get__(
                    node,
                    type(node),
                )
            )

            with self.assertRaisesRegex(RuntimeError, "AMCL/localization is not listening"):
                node.preflight_before_motion()
        finally:
            two_stage_ros.rclpy = original_rclpy

        self.assertTrue(node.navigate_client.waited)
        self.assertGreaterEqual(FakeRclpy.spin_count, 1)

    def test_follower_command_uses_path_progress_and_runner_without_shell(self):
        args = two_stage_cli.parse_args(["--dry-run", "--run-id", "arena_prior_test"])
        command = two_stage_pure.build_follower_command(args)

        self.assertEqual(command[0], "python3")
        self.assertIn("--start-selection", command)
        self.assertEqual(command[command.index("--start-selection") + 1], "path-progress")
        self.assertIn("--startup-timeout-sec", command)
        self.assertEqual(command[command.index("--startup-timeout-sec") + 1], "20.0")
        self.assertIn("--max-amcl-var-yaw", command)
        self.assertNotIn("--enable-lidar-map-replan", command)

        captured = {}

        class Result:
            returncode = 0

        def fake_runner(cmd, check=False, shell=True):
            captured["cmd"] = cmd
            captured["check"] = check
            captured["shell"] = shell
            return Result()

        result = two_stage_cli.run_follower_command(command, runner=fake_runner)

        self.assertEqual(result.returncode, 0)
        self.assertIs(captured["cmd"], command)
        self.assertFalse(captured["check"])
        self.assertFalse(captured["shell"])

    def test_follower_command_passes_lidar_replan_flags_only_when_enabled(self):
        args = two_stage_cli.parse_args(
            [
                "--dry-run",
                "--run-id",
                "arena_prior_test",
                "--enable-lidar-map-replan",
                "--lidar-replan-artifact-only",
                "--static-map",
                "maps/test.yaml",
                "--replan-output-dir",
                "results/replan",
                "--max-replans",
                "2",
                "--run-local-map-initial-scan-count",
                "4",
                "--run-local-map-update-mode",
                "full",
                "--run-local-map-min-hit-count",
                "1",
                "--allow-latest-tf-replan-fallback",
            ]
        )

        command = two_stage_pure.build_follower_command(args)

        self.assertIn("--enable-lidar-map-replan", command)
        self.assertIn("--lidar-replan-artifact-only", command)
        self.assertIn("--allow-latest-tf-replan-fallback", command)
        self.assertEqual(command[command.index("--static-map") + 1], "maps/test.yaml")
        self.assertEqual(command[command.index("--replan-output-dir") + 1], "results/replan")
        self.assertEqual(command[command.index("--max-replans") + 1], "2")
        self.assertEqual(command[command.index("--run-local-map-initial-scan-count") + 1], "4")
        self.assertEqual(command[command.index("--run-local-map-update-mode") + 1], "full")
        self.assertEqual(command[command.index("--run-local-map-min-hit-count") + 1], "1")

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

        two_stage_cli.cleanup_motion(node)

        self.assertEqual(node.cancel_count, 1)
        self.assertEqual(node.stop_count, 1)

    def test_lookup_pose_retries_and_spins_until_transform_is_available(self):
        class FakeRclpy:
            spin_count = 0

            @classmethod
            def ok(cls):
                return True

            @classmethod
            def spin_once(cls, _node, timeout_sec=0.0):
                cls.spin_count += 1
                time.sleep(min(timeout_sec, 0.001))
                return None

        class DelayedTfBuffer:
            def __init__(self):
                self.calls = 0

            def lookup_transform(self, target_frame, source_frame, lookup_time):
                self.calls += 1
                self.last_target_frame = target_frame
                self.last_lookup_time = lookup_time
                if self.calls <= 2:
                    raise RuntimeError("map frame not ready")
                return make_transform(x=0.25, y=-0.1, yaw_deg=15.0)

        original_rclpy = two_stage_ros.rclpy
        two_stage_ros.rclpy = FakeRclpy
        try:
            tf_buffer = DelayedTfBuffer()
            node = fake_tf_node(tf_buffer)
            pose, frame = two_stage_ros.TwoStageCoordinator.validate_post_localization_tf(node)
        finally:
            two_stage_ros.rclpy = original_rclpy

        self.assertEqual(frame, "base_footprint")
        self.assertEqual(node.selected_base_frame, "base_footprint")
        self.assertAlmostEqual(pose.x, 0.25)
        self.assertAlmostEqual(pose.y, -0.1)
        self.assertAlmostEqual(pose.yaw_deg, 15.0)
        self.assertGreaterEqual(FakeRclpy.spin_count, 1)
        self.assertEqual(tf_buffer.last_target_frame, "map")

    def test_lookup_pose_falls_back_to_base_link_and_records_selected_frame(self):
        class FakeRclpy:
            @staticmethod
            def ok():
                return True

            @staticmethod
            def spin_once(_node, timeout_sec=0.0):
                return None

        class FallbackTfBuffer:
            def __init__(self):
                self.sources = []

            def lookup_transform(self, _target_frame, source_frame, _lookup_time):
                self.sources.append(source_frame)
                if source_frame == "base_footprint":
                    raise RuntimeError("base_footprint unavailable")
                return make_transform(x=1.0, y=2.0, yaw_deg=-30.0)

        original_rclpy = two_stage_ros.rclpy
        two_stage_ros.rclpy = FakeRclpy
        try:
            tf_buffer = FallbackTfBuffer()
            node = fake_tf_node(tf_buffer)
            pose, frame = two_stage_ros.TwoStageCoordinator.lookup_pose(node)
        finally:
            two_stage_ros.rclpy = original_rclpy

        self.assertEqual(frame, "base_link")
        self.assertEqual(node.selected_base_frame, "base_link")
        self.assertEqual(tf_buffer.sources, ["base_footprint", "base_link"])
        self.assertAlmostEqual(pose.x, 1.0)
        self.assertAlmostEqual(pose.y, 2.0)

    def test_lookup_pose_timeout_fails_cleanly(self):
        class FakeRclpy:
            @staticmethod
            def ok():
                return True

            @staticmethod
            def spin_once(_node, timeout_sec=0.0):
                time.sleep(min(timeout_sec, 0.001))

        class MissingTfBuffer:
            def lookup_transform(self, _target_frame, source_frame, _lookup_time):
                raise RuntimeError(f"{source_frame} missing")

        original_rclpy = two_stage_ros.rclpy
        two_stage_ros.rclpy = FakeRclpy
        try:
            node = fake_tf_node(
                MissingTfBuffer(),
                tf_lookup_timeout_sec=0.005,
                tf_lookup_retry_period_sec=0.001,
            )
            with self.assertRaisesRegex(RuntimeError, "Timed out waiting for robot pose TF"):
                two_stage_ros.TwoStageCoordinator.lookup_pose(node)
        finally:
            two_stage_ros.rclpy = original_rclpy

    def test_lookup_pose_rejects_stale_tf(self):
        class FakeRclpy:
            @staticmethod
            def ok():
                return True

            @staticmethod
            def spin_once(_node, timeout_sec=0.0):
                time.sleep(min(timeout_sec, 0.001))

        class StaleTfBuffer:
            def lookup_transform(self, _target_frame, _source_frame, _lookup_time):
                return make_transform(stamp_sec=time.time() - 20.0)

        original_rclpy = two_stage_ros.rclpy
        two_stage_ros.rclpy = FakeRclpy
        try:
            node = fake_tf_node(
                StaleTfBuffer(),
                max_pose_age_sec=1.0,
                tf_lookup_timeout_sec=0.005,
                tf_lookup_retry_period_sec=0.001,
            )
            with self.assertRaisesRegex(RuntimeError, "stale_tf"):
                two_stage_ros.TwoStageCoordinator.lookup_pose(node)
        finally:
            two_stage_ros.rclpy = original_rclpy

    def test_log_row_contains_clean_schema_and_follower_command(self):
        args = two_stage_cli.parse_args(["--dry-run", "--run-id", "arena_prior_test"])
        staging = two_stage_model.StagingGoal(two_stage_model.Waypoint(0, 0.0, 0.0), 0.0)
        diagnostics = two_stage_model.RunDiagnostics(
            timestamp="2026-05-18T10:00:00",
            start_wall_time="2026-05-18T10:00:00",
            end_wall_time="2026-05-18T10:00:05",
            duration_sec=5.0,
            status="failed",
            final_status_reason="test failure",
            follower_command="python3 follower.py",
            follower_return_code=1,
            arena_spin_duration_sec=2.0,
        )

        row = two_stage_log.build_log_row(args, staging, diagnostics)
        values = dict(zip(two_stage_model.CSV_HEADER, row))

        self.assertNotIn("localization_mode", values)
        self.assertNotIn("global_localization_service", values)
        self.assertNotIn("arena_localization_duration_sec", values)
        self.assertEqual(values["status"], "failed")
        self.assertEqual(values["final_status_reason"], "test failure")
        self.assertEqual(values["arena_spin_duration_sec"], 2.0)
        self.assertEqual(values["follower_command"], "python3 follower.py")
        self.assertEqual(values["follower_return_code"], 1)

    def test_dry_run_main_works_without_ros_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "waypoints.csv"
            write_waypoints(path, [[0, 0.0, 0.0], [1, 0.5, 0.0]])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = two_stage_cli.main(
                    [
                        "--waypoints",
                        str(path),
                        "--run-id",
                        "arena_prior_dry",
                        "--dry-run",
                    ]
                )

        self.assertEqual(result, 0)
        output = stdout.getvalue()
        self.assertIn("Arena-prior two-stage waypoint run dry run", output)
        self.assertIn("Selected waypoint 0", output)
        self.assertIn("Computed staging yaw", output)
        self.assertIn("Follower command:", output)
        self.assertIn("ROS imports available:", output)
        self.assertNotIn("Localization mode", output)
        self.assertNotIn("global localization", output.lower())
        self.assertNotIn("known-start", output)
        self.assertNotIn("arena-active-dry-run", output)

    def test_public_script_help_works_without_ros_graph(self):
        script = ROOT / "scripts" / "aufgabe03" / "two_stage_waypoint_run.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("arena-prior", result.stdout)
        self.assertNotIn("--localization-mode", result.stdout)
        self.assertNotIn("--global-localization-service", result.stdout)
        self.assertNotIn("--arena-active-dry-run", result.stdout)
        self.assertNotIn("known-start", result.stdout)


if __name__ == "__main__":
    unittest.main()
