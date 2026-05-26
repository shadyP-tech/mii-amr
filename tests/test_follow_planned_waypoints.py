import argparse
import contextlib
import csv
import io
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "aufgabe03"))

import follow_planned_waypoints as follower  # noqa: E402
import map_path_planner as planner  # noqa: E402


def write_waypoints(path, rows, header=None):
    header = header or ["index", "world_x_m", "world_y_m"]
    with Path(path).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def default_args(**overrides):
    values = {
        "run_id": "test_run",
        "waypoints": Path("results/aufgabe03/aufgabe03_waypoints.csv"),
        "linear_speed": follower.DEFAULT_LINEAR_SPEED_MPS,
        "min_linear_speed": follower.DEFAULT_MIN_LINEAR_SPEED_MPS,
        "linear_gain": follower.DEFAULT_LINEAR_GAIN,
        "max_angular_speed": follower.DEFAULT_MAX_ANGULAR_SPEED_RADPS,
        "yaw_gain": follower.DEFAULT_YAW_GAIN,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []
        self.errors = []

    def warn(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.infos.append(message)

    def error(self, message):
        self.errors.append(message)


class FakeStamp:
    def __init__(self, sec=0, nanosec=0):
        self.sec = sec
        self.nanosec = nanosec


class FakeHeader:
    def __init__(self):
        self.frame_id = ""
        self.stamp = None


class FakeVector3:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class FakeQuaternion:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0


class FakePose:
    def __init__(self):
        self.position = FakeVector3()
        self.orientation = FakeQuaternion()


class FakeColor:
    def __init__(self):
        self.r = 0.0
        self.g = 0.0
        self.b = 0.0
        self.a = 0.0


class FakePoint(FakeVector3):
    pass


class FakePoseStamped:
    def __init__(self):
        self.header = FakeHeader()
        self.pose = FakePose()


class FakeNavPath:
    def __init__(self):
        self.header = FakeHeader()
        self.poses = []


class FakeMarker:
    ADD = 0
    SPHERE = 2
    DELETEALL = 3
    CUBE_LIST = 6
    SPHERE_LIST = 7
    TEXT_VIEW_FACING = 9

    def __init__(self):
        self.header = FakeHeader()
        self.ns = ""
        self.id = 0
        self.type = 0
        self.action = self.ADD
        self.pose = FakePose()
        self.scale = FakeVector3()
        self.color = FakeColor()
        self.points = []
        self.text = ""


class FakeMarkerArray:
    def __init__(self, markers=None):
        self.markers = list(markers or [])


def install_fake_rviz_messages(testcase):
    originals = {
        "Point": follower.Point,
        "PoseStamped": follower.PoseStamped,
        "NavPath": follower.NavPath,
        "Marker": follower.Marker,
        "MarkerArray": follower.MarkerArray,
    }
    follower.Point = FakePoint
    follower.PoseStamped = FakePoseStamped
    follower.NavPath = FakeNavPath
    follower.Marker = FakeMarker
    follower.MarkerArray = FakeMarkerArray

    def restore():
        for name, value in originals.items():
            setattr(follower, name, value)

    testcase.addCleanup(restore)


def test_metadata(resolution=0.1, origin=(0.0, 0.0, 0.0)):
    return planner.MapMetadata(
        yaml_path=Path("test.yaml"),
        image_path=Path("test.pgm"),
        resolution=resolution,
        origin=origin,
        negate=0,
        occupied_thresh=0.65,
        free_thresh=0.25,
        mode="trinary",
    )


def free_map(width=10, height=10, resolution=0.1):
    return planner.OccupancyMap(
        metadata=test_metadata(resolution=resolution),
        width=width,
        height=height,
        cells=[
            [planner.CELL_FREE for _ in range(width)]
            for _ in range(height)
        ],
    )


class FakeHealthNode:
    def __init__(self, pose_age_sec=0.0, amcl_warnings=None, **overrides):
        args = default_args(
            max_pose_age_sec=1.0,
            max_scan_age_sec=1.0,
            max_amcl_age_sec=1.0,
            max_tf_update_gap_sec=5.0,
            tf_recovery_time_sec=0.01,
            localization_recovery_time_sec=0.01,
            fail_on_stale_tf=False,
            fail_on_bad_localization=False,
            pause_on_bad_localization=False,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        self.args = args
        self.pose = follower.Pose2D(
            0.0,
            0.0,
            0.0,
            stamp_sec=time.time() - pose_age_sec,
            frame_id="base_footprint",
        )
        self.amcl_health = follower.AmclHealth(
            ok=not args.fail_on_bad_localization,
            warnings=amcl_warnings or [],
            cov_x=0.01,
            cov_y=0.01,
            cov_yaw=0.01,
            age_sec=0.1,
        )
        self.last_scan_received_sec = time.time()
        self.diagnostics = follower.RuntimeDiagnostics(
            max_tf_update_gap_sec=args.max_tf_update_gap_sec,
        )
        self.last_tf_stamp_sec = None
        self.last_tf_stamp_change_local_sec = None
        self.logger = FakeLogger()

    def lookup_pose(self):
        return self.pose, "base_footprint"

    def current_amcl_health(self):
        return self.amcl_health

    def get_logger(self):
        return self.logger

    def update_tf_tracking(self, pose):
        return follower.WaypointFollower.update_tf_tracking(self, pose)


class FollowPlannedWaypointsTest(unittest.TestCase):
    def test_real_robot_defaults_match_slow_successful_profile(self):
        args = follower.parse_args(["--dry-run"])

        self.assertEqual(args.linear_speed, 0.03)
        self.assertEqual(args.min_linear_speed, 0.01)
        self.assertEqual(args.linear_gain, 0.25)
        self.assertEqual(args.max_angular_speed, 0.12)
        self.assertEqual(args.yaw_gain, 0.5)
        self.assertEqual(args.waypoint_tolerance_m, 0.12)
        self.assertEqual(args.goal_tolerance_m, 0.12)
        self.assertEqual(args.rotate_start_heading_error_deg, 20.0)
        self.assertEqual(args.rotate_stop_heading_error_deg, 4.0)
        self.assertEqual(args.max_pose_age_sec, 10.0)
        self.assertEqual(args.max_scan_age_sec, 8.0)
        self.assertEqual(args.max_amcl_age_sec, 15.0)
        self.assertEqual(args.startup_timeout_sec, 20.0)
        self.assertFalse(args.require_amcl_startup)
        self.assertEqual(args.max_waypoint_time_sec, 180.0)
        self.assertFalse(args.wait_before_follow)
        self.assertFalse(args.enable_lidar_map_replan)
        self.assertFalse(args.lidar_replan_artifact_only)
        self.assertEqual(args.run_local_map_initial_scan_mode, "forward")
        self.assertEqual(args.run_local_map_initial_scan_count, 5)
        self.assertEqual(args.run_local_map_update_mode, "forward")
        self.assertEqual(args.run_local_map_min_hit_count, 2)
        self.assertEqual(args.run_local_map_inflation_radius_m, 0.22)

    def test_lidar_replan_flags_parse_as_opt_in(self):
        args = follower.parse_args(
            [
                "--dry-run",
                "--enable-lidar-map-replan",
                "--lidar-replan-artifact-only",
                "--static-map",
                "maps/test.yaml",
                "--replan-output-dir",
                "results/replan",
                "--max-replans",
                "2",
                "--max-replan-scan-age-sec",
                "0.5",
                "--max-replan-tf-age-sec",
                "0.6",
                "--run-local-map-initial-scan-mode",
                "none",
                "--run-local-map-initial-scan-count",
                "3",
                "--run-local-map-update-mode",
                "full",
                "--run-local-map-min-hit-count",
                "1",
                "--run-local-map-inflation-radius-m",
                "0.15",
                "--run-local-map-corridor-check-distance-m",
                "0.5",
            ]
        )

        self.assertTrue(args.enable_lidar_map_replan)
        self.assertTrue(args.lidar_replan_artifact_only)
        self.assertEqual(args.static_map, Path("maps/test.yaml"))
        self.assertEqual(args.replan_output_dir, Path("results/replan"))
        self.assertEqual(args.max_replans, 2)
        self.assertEqual(args.max_replan_scan_age_sec, 0.5)
        self.assertEqual(args.max_replan_tf_age_sec, 0.6)
        self.assertEqual(args.run_local_map_initial_scan_mode, "none")
        self.assertEqual(args.run_local_map_initial_scan_count, 3)
        self.assertEqual(args.run_local_map_update_mode, "full")
        self.assertEqual(args.run_local_map_min_hit_count, 1)
        self.assertEqual(args.run_local_map_inflation_radius_m, 0.15)
        self.assertEqual(args.run_local_map_corridor_check_distance_m, 0.5)

    def test_rviz_visualization_flags_parse(self):
        args = follower.parse_args(["--dry-run"])

        self.assertFalse(args.no_rviz_visualization)
        self.assertEqual(args.rviz_path_topic, "/mii_amr/planned_path")
        self.assertEqual(args.rviz_waypoint_marker_topic, "/mii_amr/planned_waypoints")
        self.assertEqual(args.rviz_obstacle_marker_topic, "/mii_amr/run_local_obstacles")

        disabled = follower.parse_args(
            [
                "--dry-run",
                "--no-rviz-visualization",
                "--rviz-path-topic",
                "/custom/path",
                "--rviz-waypoint-marker-topic",
                "/custom/waypoints",
                "--rviz-obstacle-marker-topic",
                "/custom/obstacles",
            ]
        )

        self.assertTrue(disabled.no_rviz_visualization)
        self.assertEqual(disabled.rviz_path_topic, "/custom/path")
        self.assertEqual(disabled.rviz_waypoint_marker_topic, "/custom/waypoints")
        self.assertEqual(disabled.rviz_obstacle_marker_topic, "/custom/obstacles")

    def test_rviz_path_message_contains_current_pose_and_waypoints(self):
        install_fake_rviz_messages(self)
        stamp = FakeStamp(12, 34)
        pose = follower.Pose2D(0.1, 0.2, 90.0)
        waypoints = [
            follower.Waypoint(1, 0.5, 0.2),
            follower.Waypoint(2, 0.7, 0.4),
        ]

        path = follower.build_rviz_path_message(
            waypoints,
            "map",
            stamp,
            current_pose=pose,
        )

        self.assertEqual(path.header.frame_id, "map")
        self.assertIs(path.header.stamp, stamp)
        self.assertEqual(len(path.poses), 3)
        self.assertEqual(path.poses[0].pose.position.x, 0.1)
        self.assertEqual(path.poses[0].pose.position.y, 0.2)
        self.assertEqual(path.poses[1].pose.position.x, 0.5)
        self.assertEqual(path.poses[2].pose.position.y, 0.4)
        for pose_stamped in path.poses:
            self.assertEqual(pose_stamped.pose.orientation.x, 0.0)
            self.assertEqual(pose_stamped.pose.orientation.y, 0.0)
            self.assertEqual(pose_stamped.pose.orientation.z, 0.0)
            self.assertEqual(pose_stamped.pose.orientation.w, 1.0)

    def test_rviz_waypoint_markers_include_current_goal_and_labels(self):
        install_fake_rviz_messages(self)
        stamp = FakeStamp()
        waypoints = [
            follower.Waypoint(4, 0.2, 0.3),
            follower.Waypoint(5, 0.6, 0.7),
        ]

        marker_array = follower.build_rviz_waypoint_markers(
            waypoints,
            "map",
            stamp,
            current_waypoint_index=0,
        )

        markers = marker_array.markers
        self.assertEqual(markers[0].action, FakeMarker.DELETEALL)
        self.assertIn("planned_waypoints", [marker.ns for marker in markers])
        self.assertIn("current_waypoint", [marker.ns for marker in markers])
        self.assertIn("goal_waypoint", [marker.ns for marker in markers])
        labels = [
            marker.text
            for marker in markers
            if marker.ns == "planned_waypoint_labels"
        ]
        self.assertEqual(labels, ["4", "5"])

    def test_rviz_obstacle_markers_convert_cells_to_map_points(self):
        install_fake_rviz_messages(self)
        occ = free_map(width=6, height=6, resolution=0.1)
        run_local_map = follower.lidar_obstacle_map.RunLocalObstacleMap(
            occ,
            follower.lidar_obstacle_map.RunLocalMapConfig(
                min_hit_count=1,
                min_used_points=1,
                inflation_radius_m=0.1,
            ),
        )
        run_local_map.confirmed_raw_cells = {(1, 2)}
        run_local_map.inflated_obstacle_cells = {(1, 2), (2, 2)}

        marker_array = follower.build_rviz_obstacle_markers(
            run_local_map,
            "map",
            FakeStamp(),
            blocked_cells={(3, 4)},
        )

        markers = {marker.ns: marker for marker in marker_array.markers}
        self.assertIn("run_local_confirmed_obstacle_cells", markers)
        self.assertIn("run_local_inflated_obstacle_cells", markers)
        self.assertIn("run_local_blocked_corridor_cells", markers)
        confirmed_point = markers["run_local_confirmed_obstacle_cells"].points[0]
        self.assertAlmostEqual(confirmed_point.x, 0.15)
        self.assertAlmostEqual(confirmed_point.y, 0.25)
        blocked_point = markers["run_local_blocked_corridor_cells"].points[0]
        self.assertAlmostEqual(blocked_point.x, 0.35)
        self.assertAlmostEqual(blocked_point.y, 0.45)

    def test_default_corridor_check_does_not_double_inflate_replanned_path(self):
        occ = free_map(width=30, height=20, resolution=0.1)
        run_local_map = follower.lidar_obstacle_map.RunLocalObstacleMap(
            occ,
            follower.lidar_obstacle_map.RunLocalMapConfig(
                min_hit_count=1,
                min_used_points=1,
                inflation_radius_m=0.2,
                static_wall_exclusion_radius_m=0.0,
                max_start_snap_m=0.3,
                max_goal_snap_m=0.3,
            ),
        )
        run_local_map.add_observations(
            follower.lidar_obstacle_map.ObservationBatch([
                follower.lidar_obstacle_map.GridCellObservation(10, 10),
            ])
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = follower.lidar_obstacle_map.plan_with_run_local_map(
                run_local_map,
                follower.lidar_obstacle_map.Pose2D(0.55, 1.05, 0.0),
                follower.lidar_obstacle_map.Pose2D(1.65, 1.05, 0.0),
                "corridor_test",
                output_dir=Path(tmpdir),
            )
        waypoints = [
            follower.Waypoint(index, x, y)
            for index, x, y in result.waypoints
        ]

        class CorridorNode:
            corridor_blocked_cells = follower.WaypointFollower.corridor_blocked_cells

            def __init__(self):
                self.args = default_args(
                    run_local_map_corridor_check_distance_m=2.0,
                    run_local_map_corridor_radius_m=None,
                    run_local_map_inflation_radius_m=0.2,
                )
                self.run_local_map = run_local_map
                self.diagnostics = follower.RuntimeDiagnostics()

        node = CorridorNode()
        pose = follower.Pose2D(0.55, 1.05, 0.0)

        self.assertFalse(
            follower.WaypointFollower.corridor_blocked_cells(
                node,
                pose,
                waypoints,
            )
        )
        node.args.run_local_map_corridor_radius_m = 0.2
        self.assertTrue(
            follower.WaypointFollower.corridor_blocked_cells(
                node,
                pose,
                waypoints,
            )
        )

    def test_wait_before_follow_prompt_requires_run(self):
        args = follower.parse_args(["--dry-run", "--wait-before-follow"])
        pose = follower.Pose2D(0.1, 0.2, 15.0)
        waypoints = [follower.Waypoint(1, 0.5, 0.2)]

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            accepted = follower.wait_before_follow_confirmation(
                args,
                pose,
                waypoints,
                input_fn=lambda _prompt: "RUN",
            )
            rejected = follower.wait_before_follow_confirmation(
                args,
                pose,
                waypoints,
                input_fn=lambda _prompt: "stop",
            )

        self.assertTrue(accepted)
        self.assertFalse(rejected)
        self.assertIn("Waypoint follower handoff is ready", stdout.getvalue())

    def test_refresh_after_operator_wait_requires_new_scan_after_prompt(self):
        class Stamp:
            def __init__(self, stamp_sec):
                self.sec = int(stamp_sec)
                self.nanosec = int((stamp_sec - int(stamp_sec)) * 1_000_000_000)

        class Header:
            def __init__(self, stamp_sec):
                self.stamp = Stamp(stamp_sec)

        class Scan:
            def __init__(self, stamp_sec):
                self.header = Header(stamp_sec)

        class FakeRclpy:
            spin_count = 0

            @staticmethod
            def ok():
                return True

            @classmethod
            def spin_once(cls, node, timeout_sec=0.0):
                cls.spin_count += 1
                node.last_scan_received_sec = time.time()
                if cls.spin_count == 1:
                    node.last_scan = Scan(node.min_scan_received_sec - 10.0)
                else:
                    node.last_scan = Scan(time.time())

        class RefreshNode:
            reset_tf_tracking = follower.WaypointFollower.reset_tf_tracking
            refresh_after_operator_wait = follower.WaypointFollower.refresh_after_operator_wait

            def __init__(self):
                self.args = default_args(startup_timeout_sec=0.5)
                self.min_scan_received_sec = time.time()
                self.last_scan = Scan(self.min_scan_received_sec - 10.0)
                self.last_scan_received_sec = time.time() - 10.0
                self.last_tf_stamp_sec = 123.0
                self.last_tf_stamp_change_local_sec = time.time() - 10.0

        original_rclpy = follower.rclpy
        follower.rclpy = FakeRclpy
        try:
            node = RefreshNode()

            follower.WaypointFollower.refresh_after_operator_wait(
                node,
                node.min_scan_received_sec,
            )
        finally:
            follower.rclpy = original_rclpy

        self.assertEqual(FakeRclpy.spin_count, 2)
        self.assertGreaterEqual(
            follower.replan_runtime.scan_stamp_sec(node.last_scan),
            node.min_scan_received_sec - follower.replan_runtime.FRESH_SCAN_STAMP_SLACK_SEC,
        )
        self.assertIsNone(node.last_tf_stamp_sec)
        self.assertIsNone(node.last_tf_stamp_change_local_sec)

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

    def test_velocity_deadband_suppresses_tiny_angular_commands(self):
        linear, angular = follower.velocity_command(
            distance_m=1.0,
            yaw_error_deg=2.0,
            rotate_mode=False,
            linear_speed_mps=0.03,
            min_linear_speed_mps=0.01,
            linear_gain=0.25,
            max_angular_speed_radps=0.12,
            yaw_gain=0.5,
            forward_yaw_deadband_deg=4.0,
            forward_stop_heading_error_deg=18.0,
        )

        self.assertAlmostEqual(linear, 0.03)
        self.assertEqual(angular, 0.0)

    def test_velocity_scales_linear_speed_with_heading_error(self):
        linear, angular = follower.velocity_command(
            distance_m=1.0,
            yaw_error_deg=11.0,
            rotate_mode=False,
            linear_speed_mps=0.03,
            min_linear_speed_mps=0.01,
            linear_gain=0.25,
            max_angular_speed_radps=0.12,
            yaw_gain=0.5,
            forward_yaw_deadband_deg=4.0,
            forward_stop_heading_error_deg=18.0,
        )

        self.assertAlmostEqual(linear, 0.015)
        self.assertGreater(angular, 0.0)

    def test_velocity_stops_forward_motion_for_large_heading_error(self):
        linear, angular = follower.velocity_command(
            distance_m=1.0,
            yaw_error_deg=19.0,
            rotate_mode=False,
            linear_speed_mps=0.03,
            min_linear_speed_mps=0.01,
            linear_gain=0.25,
            max_angular_speed_radps=0.12,
            yaw_gain=0.5,
            forward_yaw_deadband_deg=4.0,
            forward_stop_heading_error_deg=18.0,
        )

        self.assertEqual(linear, 0.0)
        self.assertGreater(angular, 0.0)

    def test_invalid_heading_thresholds_are_rejected(self):
        invalid_cases = [
            ["--forward-yaw-deadband-deg", "18.0", "--forward-stop-heading-error-deg", "18.0"],
            ["--forward-stop-heading-error-deg", "20.0", "--rotate-start-heading-error-deg", "20.0"],
        ]

        for argv in invalid_cases:
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        follower.parse_args(argv)

    def test_path_progress_start_skips_already_reached_waypoints(self):
        waypoints = [
            follower.Waypoint(0, 0.0, 0.0),
            follower.Waypoint(1, 0.5, 0.0),
            follower.Waypoint(2, 1.0, 0.0),
            follower.Waypoint(3, 1.0, 0.5),
        ]
        pose = follower.Pose2D(0.52, 0.02, 0.0)

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

        self.assertEqual(selection.selected_segment_index, 1)
        self.assertEqual(selection.selected_waypoint_index, 2)
        self.assertEqual([wp.index for wp in selection.waypoints], [2, 3])
        self.assertAlmostEqual(selection.distance_to_path_m, 0.02)

    def test_path_progress_preserves_final_waypoint(self):
        waypoints = [
            follower.Waypoint(0, 0.0, 0.0),
            follower.Waypoint(1, 0.5, 0.0),
            follower.Waypoint(2, 1.0, 0.0),
        ]
        pose = follower.Pose2D(0.96, 0.01, 0.0)

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

        self.assertEqual([wp.index for wp in selection.waypoints], [2])

    def test_path_progress_off_path_start_fails_before_motion(self):
        waypoints = [
            follower.Waypoint(0, 0.0, 0.0),
            follower.Waypoint(1, 0.5, 0.0),
        ]
        pose = follower.Pose2D(0.2, 1.0, 0.0)

        with self.assertRaisesRegex(ValueError, "too far"):
            follower.select_executable_waypoints(
                waypoints,
                pose,
                start_selection="path-progress",
                start_on_path_tolerance_m=0.25,
                waypoint_tolerance_m=0.12,
                goal_tolerance_m=0.12,
                min_spacing_m=0.0,
                skip_first=True,
            )

    def test_fixed_skip_start_selection_preserves_old_behavior(self):
        waypoints = [
            follower.Waypoint(0, 0.0, 0.0),
            follower.Waypoint(1, 0.5, 0.0),
            follower.Waypoint(2, 1.0, 0.0),
        ]
        pose = follower.Pose2D(0.9, 0.0, 0.0)

        selection = follower.select_executable_waypoints(
            waypoints,
            pose,
            start_selection="fixed-skip",
            start_on_path_tolerance_m=0.25,
            waypoint_tolerance_m=0.12,
            goal_tolerance_m=0.12,
            min_spacing_m=0.0,
            skip_first=True,
        )

        self.assertIsNone(selection.selected_segment_index)
        self.assertEqual(selection.selected_waypoint_index, 1)
        self.assertEqual([wp.index for wp in selection.waypoints], [1, 2])

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

    def test_tf_update_gap_watchdog_detects_stalled_transform_stamp(self):
        dummy = argparse.Namespace(
            last_tf_stamp_sec=None,
            last_tf_stamp_change_local_sec=None,
        )
        pose = follower.Pose2D(0.0, 0.0, 0.0, stamp_sec=10.0)

        first_gap = follower.WaypointFollower.update_tf_tracking(dummy, pose)
        dummy.last_tf_stamp_change_local_sec -= 6.0
        stalled_gap = follower.WaypointFollower.update_tf_tracking(dummy, pose)
        updated_gap = follower.WaypointFollower.update_tf_tracking(
            dummy,
            follower.Pose2D(0.0, 0.0, 0.0, stamp_sec=11.0),
        )

        self.assertLess(first_gap, 0.1)
        self.assertGreater(stalled_gap, 5.0)
        self.assertLess(updated_gap, 0.1)

    def test_startup_gate_allows_missing_amcl_when_tf_and_scan_are_ready(self):
        class FakeRclpy:
            @staticmethod
            def ok():
                return True

        node = argparse.Namespace(
            args=default_args(
                startup_timeout_sec=0.01,
                map_frame="map",
                base_frame="base_footprint",
                fallback_base_frame="base_link",
                require_amcl_startup=False,
                fail_on_bad_localization=False,
                pause_on_bad_localization=False,
            ),
            last_scan=object(),
            last_amcl=None,
            base_frame_used="",
        )

        def lookup_pose():
            return follower.Pose2D(0.0, 0.0, 0.0), "base_footprint"

        node.lookup_pose = lookup_pose
        original_rclpy = follower.rclpy
        follower.rclpy = FakeRclpy
        try:
            follower.WaypointFollower.wait_for_startup_gate(node)
        finally:
            follower.rclpy = original_rclpy

        self.assertEqual(node.base_frame_used, "base_footprint")

    def test_startup_gate_can_require_amcl_for_strict_startup(self):
        class FakeRclpy:
            @staticmethod
            def ok():
                return True

            @staticmethod
            def spin_once(_node, timeout_sec=0.0):
                return None

        node = argparse.Namespace(
            args=default_args(
                startup_timeout_sec=0.001,
                map_frame="map",
                base_frame="base_footprint",
                fallback_base_frame="base_link",
                require_amcl_startup=True,
                fail_on_bad_localization=False,
                pause_on_bad_localization=False,
            ),
            last_scan=object(),
            last_amcl=None,
            base_frame_used="",
        )

        def lookup_pose():
            return follower.Pose2D(0.0, 0.0, 0.0), "base_footprint"

        node.lookup_pose = lookup_pose
        original_rclpy = follower.rclpy
        follower.rclpy = FakeRclpy
        try:
            with self.assertRaisesRegex(RuntimeError, "/amcl_pose"):
                follower.WaypointFollower.wait_for_startup_gate(node)
        finally:
            follower.rclpy = original_rclpy

    def test_missing_amcl_is_warning_unless_strict_localization_is_enabled(self):
        relaxed_node = argparse.Namespace(
            args=default_args(fail_on_bad_localization=False),
            last_amcl=None,
            last_amcl_received_sec=None,
        )
        strict_node = argparse.Namespace(
            args=default_args(fail_on_bad_localization=True),
            last_amcl=None,
            last_amcl_received_sec=None,
        )

        relaxed = follower.WaypointFollower.current_amcl_health(relaxed_node)
        strict = follower.WaypointFollower.current_amcl_health(strict_node)

        self.assertTrue(relaxed.ok)
        self.assertEqual(relaxed.warnings, ["missing_amcl"])
        self.assertFalse(strict.ok)
        self.assertEqual(strict.warnings, ["missing_amcl"])

    def test_stale_absolute_tf_age_warns_by_default(self):
        node = FakeHealthNode(pose_age_sec=2.0)

        pose, frame, _health = follower.WaypointFollower.check_health_or_raise(node)

        self.assertEqual(frame, "base_footprint")
        self.assertEqual(pose.frame_id, "base_footprint")
        self.assertEqual(node.diagnostics.tf_stale_warning_count, 1)
        self.assertEqual(len(node.logger.warnings), 1)

    def test_stale_absolute_tf_age_can_fail_strictly(self):
        node = FakeHealthNode(pose_age_sec=2.0, fail_on_stale_tf=True)

        with self.assertRaisesRegex(RuntimeError, "TF pose is stale"):
            follower.WaypointFollower.check_health_or_raise(node)

    def test_bad_amcl_covariance_can_pause_for_recovery(self):
        node = FakeHealthNode(
            amcl_warnings=["high_cov_x"],
            pause_on_bad_localization=True,
        )

        with self.assertRaises(follower.RecoverableHealthError) as context:
            follower.WaypointFollower.check_health_or_raise(node)

        self.assertEqual(context.exception.reason, "bad_localization")
        self.assertEqual(node.diagnostics.localization_warning_count, 1)

    def test_recovery_timeout_produces_clear_failure(self):
        class FakeRclpy:
            @staticmethod
            def ok():
                return True

            @staticmethod
            def spin_once(_node, timeout_sec=0.0):
                return None

        class RecoveryNode:
            def __init__(self):
                self.diagnostics = follower.RuntimeDiagnostics()
                self.stop_count = 0
                self.logger = FakeLogger()

            def check_health_or_raise(self):
                raise follower.RecoverableHealthError(
                    "tf_update_gap",
                    0.001,
                    "TF transform stamp stopped updating",
                )

            def stop_repeatedly(self):
                self.stop_count += 1

            def get_logger(self):
                return self.logger

        original_rclpy = follower.rclpy
        follower.rclpy = FakeRclpy
        try:
            node = RecoveryNode()
            with self.assertRaisesRegex(RuntimeError, "did not recover"):
                follower.WaypointFollower.check_health_or_recover(node)
        finally:
            follower.rclpy = original_rclpy

        self.assertEqual(node.stop_count, 1)
        self.assertEqual(node.diagnostics.recovery_pause_count, 1)

    def test_log_row_generation_for_statuses(self):
        args = default_args()
        scan = follower.ScanSafety(False, "soft_stop", 3, 0.2, 0.21)
        amcl = follower.AmclHealth(True, [], 0.01, 0.02, 0.03, 0.1)
        start = follower.Pose2D(0.0, 0.0, 0.0)
        final = follower.Pose2D(0.2, 0.1, 5.0)
        blocked = follower.Waypoint(2, 0.5, 0.0)
        diagnostics = follower.RuntimeDiagnostics(
            selected_start_segment_index=1,
            selected_start_waypoint_index=2,
            distance_to_path_m=0.03,
            tf_pose_age_sec=2.0,
            max_tf_update_gap_sec=5.0,
            tf_stale_warning_count=3,
            localization_warning_count=4,
            recovery_pause_count=1,
            max_abs_yaw_error_deg=12.0,
            yaw_error_sum_deg=18.0,
            yaw_error_count=3,
            rotate_seconds=1.2,
            forward_seconds=2.3,
            final_status_reason="test_reason",
        )

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
                    diagnostics=diagnostics,
                )
                values = dict(zip(follower.CSV_HEADER, row))
                self.assertEqual(values["status"], status)
                self.assertEqual(values["base_frame_used"], "base_footprint")
                self.assertEqual(values["selected_start_segment_index"], 1)
                self.assertEqual(values["selected_start_waypoint_index"], 2)
                self.assertEqual(values["final_status_reason"], "test_reason")
                self.assertIn("replan_count", values)
                self.assertIn("updated_map_yaml", values)
                if status == "blocked":
                    self.assertEqual(values["blocked_waypoint_index"], 2)
                if status == "timeout":
                    self.assertEqual(values["timeout_waypoint_index"], 2)

    def test_artifact_only_replan_returns_distinct_status_without_resuming(self):
        class FakeRclpy:
            @staticmethod
            def ok():
                return True

        args = follower.parse_args(
            [
                "--dry-run",
                "--enable-lidar-map-replan",
                "--lidar-replan-artifact-only",
            ]
        )
        node = argparse.Namespace(
            args=args,
            diagnostics=follower.RuntimeDiagnostics(),
            reached_count=0,
            start_pose=None,
            final_pose=None,
            last_amcl_health=None,
            last_scan_safety=None,
            base_frame_used="base_footprint",
            logger=FakeLogger(),
        )
        pose = follower.Pose2D(0.0, 0.0, 0.0, stamp_sec=time.time())
        amcl = follower.AmclHealth(True, [], 0.01, 0.01, 0.01, 0.1)
        safety = follower.ScanSafety(False, "soft_stop", 4, 0.2, 0.21)

        node.check_health_or_recover = lambda: (pose, "base_footprint", amcl)
        node.check_scan_or_raise = lambda _mode: (_ for _ in ()).throw(
            follower.BlockedByScanError(safety)
        )
        node.initialize_run_local_route = lambda _pose, waypoints: list(waypoints)
        node.replan_after_blockage = lambda _pose, _remaining, **_kwargs: [
            follower.Waypoint(10, 0.3, 0.0),
            follower.Waypoint(11, 0.5, 0.0),
        ]
        node.stop_repeatedly = lambda: None
        node.get_logger = lambda: node.logger

        original_rclpy = follower.rclpy
        follower.rclpy = FakeRclpy
        try:
            result = follower.WaypointFollower.follow_waypoints(
                node,
                [follower.Waypoint(1, 0.4, 0.0)],
            )
        finally:
            follower.rclpy = original_rclpy

        self.assertEqual(result["status"], "replan_artifact_only_complete")

    def test_live_replan_publishes_replacement_route(self):
        class ReplanRoutePublished(Exception):
            pass

        class FakeRclpy:
            @staticmethod
            def ok():
                return True

        args = follower.parse_args(["--dry-run", "--enable-lidar-map-replan"])
        node = argparse.Namespace(
            args=args,
            diagnostics=follower.RuntimeDiagnostics(),
            reached_count=0,
            start_pose=None,
            final_pose=None,
            last_amcl_health=None,
            last_scan_safety=None,
            base_frame_used="base_footprint",
            logger=FakeLogger(),
        )
        pose = follower.Pose2D(0.0, 0.0, 0.0, stamp_sec=time.time())
        amcl = follower.AmclHealth(True, [], 0.01, 0.01, 0.01, 0.1)
        safety = follower.ScanSafety(False, "soft_stop", 4, 0.2, 0.21)
        published_routes = []

        def publish_route(waypoints, current_pose=None, current_waypoint_index=0):
            route = [waypoint.index for waypoint in waypoints]
            published_routes.append(route)
            if route == [10, 11]:
                raise ReplanRoutePublished()

        node.check_health_or_recover = lambda: (pose, "base_footprint", amcl)
        node.initialize_run_local_route = lambda _pose, waypoints: list(waypoints)
        node.check_scan_or_raise = lambda _mode: (_ for _ in ()).throw(
            follower.BlockedByScanError(safety)
        )
        node.replan_after_blockage = lambda _pose, _remaining, **_kwargs: [
            follower.Waypoint(10, 0.3, 0.0),
            follower.Waypoint(11, 0.5, 0.0),
        ]
        node.stop_repeatedly = lambda: None
        node.get_logger = lambda: node.logger
        node.publish_rviz_route = publish_route

        original_rclpy = follower.rclpy
        follower.rclpy = FakeRclpy
        try:
            with self.assertRaises(ReplanRoutePublished):
                follower.WaypointFollower.follow_waypoints(
                    node,
                    [follower.Waypoint(1, 0.4, 0.0)],
                )
        finally:
            follower.rclpy = original_rclpy

        self.assertIn([10, 11], published_routes)

    def test_initial_run_local_empty_map_continues_with_static_route(self):
        class InitialMapNode:
            update_replan_diagnostics = follower.WaypointFollower.update_replan_diagnostics

            def __init__(self):
                self.args = default_args(run_local_map_initial_scan_mode="full")
                self.diagnostics = follower.RuntimeDiagnostics()
                self.logger = FakeLogger()
                self.stop_count = 0
                self.run_local_map = None

            def stop_repeatedly(self):
                self.stop_count += 1

            def get_logger(self):
                return self.logger

        waypoints = [
            follower.Waypoint(1, 0.4, 0.0),
            follower.Waypoint(2, 0.8, 0.0),
        ]
        run_local_map = object()
        initial_result = follower.lidar_obstacle_map.ReplanResult(
            success=False,
            reason=follower.lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS,
            diagnostics=follower.lidar_obstacle_map.ObstacleOverlayDiagnostics(),
            run_local_map=run_local_map,
        )
        original_initial_replan = follower.replan_runtime.perform_initial_run_local_replan
        follower.replan_runtime.perform_initial_run_local_replan = (
            lambda *_args, **_kwargs: initial_result
        )
        try:
            node = InitialMapNode()
            result = follower.WaypointFollower.initialize_run_local_route(
                node,
                follower.Pose2D(0.0, 0.0, 0.0),
                waypoints,
            )
        finally:
            follower.replan_runtime.perform_initial_run_local_replan = original_initial_replan

        self.assertEqual(result, waypoints)
        self.assertIsNone(node.run_local_map)
        self.assertEqual(node.diagnostics.replan_count, 0)
        self.assertEqual(
            node.diagnostics.last_replan_reason,
            follower.lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS,
        )
        self.assertEqual(len(node.logger.warnings), 1)
        self.assertGreaterEqual(node.stop_count, 1)

    def test_existing_run_local_map_without_confirmed_obstacles_rejects_replan(self):
        class EmptyRunLocalMap:
            confirmed_raw_cells = set()

        class ReplanNode:
            run_local_map = EmptyRunLocalMap()

        with self.assertRaisesRegex(
            RuntimeError,
            "lidar_replan_failed:no_confirmed_run_local_obstacles",
        ):
            follower.WaypointFollower.plan_with_existing_run_local_map(
                ReplanNode(),
                follower.Pose2D(0.0, 0.0, 0.0),
                [
                    follower.Waypoint(1, 0.4, 0.0),
                    follower.Waypoint(2, 0.8, 0.0),
                ],
            )

    def test_scan_blockage_requires_fresh_map_update_before_fallback_replan(self):
        class ReplanNode:
            replan_after_blockage = follower.WaypointFollower.replan_after_blockage
            plan_with_existing_run_local_map = (
                follower.WaypointFollower.plan_with_existing_run_local_map
            )
            update_replan_diagnostics = follower.WaypointFollower.update_replan_diagnostics
            validate_replan_result = follower.WaypointFollower.validate_replan_result
            replanned_waypoints_from_result = (
                follower.WaypointFollower.replanned_waypoints_from_result
            )
            first_motion_waypoint = follower.WaypointFollower.first_motion_waypoint

            def __init__(self):
                self.args = default_args(
                    max_replans=2,
                    run_local_map_update_mode="forward",
                    goal_tolerance_m=0.12,
                    waypoint_tolerance_m=0.12,
                    min_scan_range_m=0.24,
                    obstacle_forward_half_width_m=0.18,
                    robot_footprint_radius_m=0.18,
                )
                self.live_replan_attempt_count = 0
                self.diagnostics = follower.RuntimeDiagnostics()
                self.run_local_map = argparse.Namespace(confirmed_raw_cells={(1, 1)})
                self.logger = FakeLogger()

            def get_logger(self):
                return self.logger

        node = ReplanNode()
        old_remaining = [
            follower.Waypoint(1, 0.5, 0.0),
            follower.Waypoint(2, 1.0, 0.0),
        ]
        plan_called = False

        def fail_update(*_args, **_kwargs):
            raise RuntimeError("stale_tf: age=6.0s")

        def plan_existing(*_args, **_kwargs):
            nonlocal plan_called
            plan_called = True
            return follower.lidar_obstacle_map.ReplanResult(
                success=True,
                reason="run_local_replan_completed",
                waypoints=[(0, 0.0, 0.0), (1, 1.0, 0.0)],
            )

        original_update = follower.replan_runtime.perform_lidar_replan
        original_existing = follower.replan_runtime.plan_existing_run_local_map
        follower.replan_runtime.perform_lidar_replan = fail_update
        follower.replan_runtime.plan_existing_run_local_map = plan_existing
        try:
            with self.assertRaisesRegex(RuntimeError, "lidar_replan_failed:stale_tf"):
                follower.WaypointFollower.replan_after_blockage(
                    node,
                    follower.Pose2D(0.0, 0.0, 0.0),
                    old_remaining,
                    trigger=follower.REPLAN_TRIGGER_SCAN_BLOCKAGE,
                )
        finally:
            follower.replan_runtime.perform_lidar_replan = original_update
            follower.replan_runtime.plan_existing_run_local_map = original_existing

        self.assertFalse(plan_called)
        self.assertEqual(node.live_replan_attempt_count, 0)

    def test_known_corridor_blockage_can_replan_with_existing_map_after_update_failure(self):
        class ReplanNode:
            replan_after_blockage = follower.WaypointFollower.replan_after_blockage
            plan_with_existing_run_local_map = (
                follower.WaypointFollower.plan_with_existing_run_local_map
            )
            update_replan_diagnostics = follower.WaypointFollower.update_replan_diagnostics
            validate_replan_result = follower.WaypointFollower.validate_replan_result
            replanned_waypoints_from_result = (
                follower.WaypointFollower.replanned_waypoints_from_result
            )
            first_motion_waypoint = follower.WaypointFollower.first_motion_waypoint

            def __init__(self):
                self.args = default_args(
                    max_replans=2,
                    run_local_map_update_mode="forward",
                    goal_tolerance_m=0.12,
                    waypoint_tolerance_m=0.12,
                    min_scan_range_m=0.24,
                    obstacle_forward_half_width_m=0.18,
                    robot_footprint_radius_m=0.18,
                )
                self.live_replan_attempt_count = 0
                self.diagnostics = follower.RuntimeDiagnostics()
                self.run_local_map = argparse.Namespace(confirmed_raw_cells={(1, 1)})
                self.logger = FakeLogger()

            def get_logger(self):
                return self.logger

        node = ReplanNode()
        old_remaining = [
            follower.Waypoint(1, 0.5, 0.0),
            follower.Waypoint(2, 1.0, 0.0),
        ]
        plan_called = False

        def fail_update(*_args, **_kwargs):
            raise RuntimeError("stale_tf: age=6.0s")

        def plan_existing(*_args, **_kwargs):
            nonlocal plan_called
            plan_called = True
            return follower.lidar_obstacle_map.ReplanResult(
                success=True,
                reason="run_local_replan_completed",
                diagnostics=follower.lidar_obstacle_map.ObstacleOverlayDiagnostics(),
                waypoints=[(0, 0.0, 0.0), (1, 1.0, 0.0)],
                run_local_map=node.run_local_map,
            )

        original_update = follower.replan_runtime.perform_lidar_replan
        original_existing = follower.replan_runtime.plan_existing_run_local_map
        follower.replan_runtime.perform_lidar_replan = fail_update
        follower.replan_runtime.plan_existing_run_local_map = plan_existing
        try:
            replanned = follower.WaypointFollower.replan_after_blockage(
                node,
                follower.Pose2D(0.0, 0.0, 0.0),
                old_remaining,
                trigger=follower.REPLAN_TRIGGER_KNOWN_CORRIDOR,
            )
        finally:
            follower.replan_runtime.perform_lidar_replan = original_update
            follower.replan_runtime.plan_existing_run_local_map = original_existing

        self.assertTrue(plan_called)
        self.assertEqual([(wp.x, wp.y) for wp in replanned], [(0.0, 0.0), (1.0, 0.0)])
        self.assertEqual(node.live_replan_attempt_count, 1)
        self.assertEqual(len(node.logger.warnings), 1)

    def test_initial_run_local_path_failure_still_aborts(self):
        class InitialMapNode:
            update_replan_diagnostics = follower.WaypointFollower.update_replan_diagnostics
            validate_replan_result = follower.WaypointFollower.validate_replan_result

            def __init__(self):
                self.args = default_args(run_local_map_initial_scan_mode="full")
                self.diagnostics = follower.RuntimeDiagnostics()
                self.logger = FakeLogger()

            def stop_repeatedly(self):
                return None

            def get_logger(self):
                return self.logger

        initial_result = follower.lidar_obstacle_map.ReplanResult(
            success=False,
            reason=follower.lidar_obstacle_map.RUN_LOCAL_FAILURE_GOAL_BLOCKED,
            diagnostics=follower.lidar_obstacle_map.ObstacleOverlayDiagnostics(),
        )
        original_initial_replan = follower.replan_runtime.perform_initial_run_local_replan
        follower.replan_runtime.perform_initial_run_local_replan = (
            lambda *_args, **_kwargs: initial_result
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "goal_blocked"):
                follower.WaypointFollower.initialize_run_local_route(
                    InitialMapNode(),
                    follower.Pose2D(0.0, 0.0, 0.0),
                    [
                        follower.Waypoint(1, 0.4, 0.0),
                        follower.Waypoint(2, 0.8, 0.0),
                    ],
                )
        finally:
            follower.replan_runtime.perform_initial_run_local_replan = original_initial_replan

    def test_replan_validation_rejects_first_motion_waypoint_behind_robot(self):
        class ValidationNode:
            replanned_waypoints_from_result = (
                follower.WaypointFollower.replanned_waypoints_from_result
            )
            first_motion_waypoint = follower.WaypointFollower.first_motion_waypoint

            def __init__(self):
                self.args = default_args(
                    goal_tolerance_m=0.12,
                    waypoint_tolerance_m=0.12,
                    min_scan_range_m=0.24,
                    obstacle_forward_half_width_m=0.18,
                    robot_footprint_radius_m=0.18,
                )

        node = ValidationNode()
        result = follower.lidar_obstacle_map.ReplanResult(
            success=True,
            reason="replan_completed",
            waypoints=[
                (0, 0.0, 0.0),
                (1, -0.4, 0.0),
                (2, 0.0, 0.5),
            ],
        )
        current_pose = follower.Pose2D(0.0, 0.0, 0.0)
        old_remaining = [
            follower.Waypoint(1, 0.4, 0.0),
            follower.Waypoint(2, 0.0, 0.5),
        ]
        goal_waypoint = follower.Waypoint(2, 0.0, 0.5)

        with self.assertRaisesRegex(RuntimeError, "first_waypoint_behind_robot"):
            follower.WaypointFollower.validate_replan_result(
                node,
                result,
                current_pose,
                old_remaining,
                goal_waypoint,
            )

    def test_replan_validation_allows_near_forward_waypoint_on_clear_path(self):
        class ValidationNode:
            replanned_waypoints_from_result = (
                follower.WaypointFollower.replanned_waypoints_from_result
            )
            first_motion_waypoint = follower.WaypointFollower.first_motion_waypoint

            def __init__(self):
                self.args = default_args(
                    goal_tolerance_m=0.12,
                    waypoint_tolerance_m=0.12,
                    min_scan_range_m=0.24,
                    obstacle_forward_half_width_m=0.18,
                    robot_footprint_radius_m=0.18,
                )

        node = ValidationNode()
        result = follower.lidar_obstacle_map.ReplanResult(
            success=True,
            reason="run_local_replan_completed",
            waypoints=[
                (0, 0.0, 0.0),
                (1, 0.18, 0.0),
                (2, 0.6, 0.2),
            ],
            path_cells=[(0, 0), (1, 0), (2, 1)],
            inflated_obstacle_cells={(2, 0)},
        )
        current_pose = follower.Pose2D(0.0, 0.0, 0.0)
        old_remaining = [
            follower.Waypoint(1, 0.4, 0.0),
            follower.Waypoint(2, 0.6, 0.2),
        ]
        goal_waypoint = follower.Waypoint(2, 0.6, 0.2)

        replanned = follower.WaypointFollower.validate_replan_result(
            node,
            result,
            current_pose,
            old_remaining,
            goal_waypoint,
        )

        self.assertEqual([wp.index for wp in replanned], [0, 1, 2])

    def test_append_csv_row_migrates_old_header_by_appending_columns(self):
        args = default_args()
        row = follower.build_log_row(
            args,
            waypoint_count=3,
            reached_count=3,
            status="completed",
            notes="test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.csv"
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(follower.BASE_CSV_HEADER)
                writer.writerow(["old"] * len(follower.BASE_CSV_HEADER))

            follower.append_csv_row(path, follower.CSV_HEADER, row)

            with path.open(newline="") as file:
                rows = list(csv.reader(file))

        self.assertEqual(rows[0], follower.CSV_HEADER)
        self.assertEqual(len(rows[1]), len(follower.CSV_HEADER))
        self.assertEqual(len(rows[2]), len(follower.CSV_HEADER))

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

            with contextlib.redirect_stdout(io.StringIO()):
                result = follower.main(["--waypoints", str(path), "--dry-run"])

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
