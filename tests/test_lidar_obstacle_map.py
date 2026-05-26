import argparse
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "aufgabe03"))

import lidar_obstacle_map as overlay  # noqa: E402
import map_path_planner as planner  # noqa: E402
import replan_runtime  # noqa: E402


def metadata(resolution=0.1, origin=(0.0, 0.0, 0.0)):
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


def free_map(width=30, height=30, resolution=0.1):
    return planner.OccupancyMap(
        metadata=metadata(resolution=resolution),
        width=width,
        height=height,
        cells=[
            [planner.CELL_FREE for _ in range(width)]
            for _ in range(height)
        ],
    )


class LidarObstacleMapTest(unittest.TestCase):
    def test_base_point_to_map_handles_cardinal_yaws(self):
        point = overlay.BaseFramePoint(0.5, 0.0)
        cases = [
            (0.0, (1.5, 1.0)),
            (90.0, (1.0, 1.5)),
            (180.0, (0.5, 1.0)),
            (-90.0, (1.0, 0.5)),
        ]

        for yaw_deg, expected in cases:
            with self.subTest(yaw_deg=yaw_deg):
                result = overlay.base_point_to_map(
                    point,
                    overlay.Pose2D(1.0, 1.0, yaw_deg),
                )
                self.assertAlmostEqual(result[0], expected[0])
                self.assertAlmostEqual(result[1], expected[1])

    def test_overlay_filters_points_and_inflates_free_cells(self):
        occ = free_map()
        config = overlay.ObstacleOverlayConfig(
            min_cluster_size=3,
            min_cluster_width_m=0.05,
            inflate_radius_m=0.15,
        )
        points = [
            overlay.BaseFramePoint(0.45, -0.04),
            overlay.BaseFramePoint(0.45, 0.00),
            overlay.BaseFramePoint(0.45, 0.04),
            overlay.BaseFramePoint(-0.20, 0.0),
            overlay.BaseFramePoint(0.80, 0.0),
        ]

        updated, inflated, _rows, diagnostics = overlay.build_overlay_map(
            occ,
            points,
            overlay.Pose2D(1.0, 1.0, 0.0),
            config,
        )

        self.assertEqual(diagnostics.filtered_obstacle_points, 3)
        self.assertGreaterEqual(diagnostics.free_obstacle_cells, 1)
        self.assertGreater(len(inflated), diagnostics.free_obstacle_cells)
        self.assertEqual(occ.cells[10][14], planner.CELL_FREE)
        self.assertIn(planner.CELL_OCCUPIED, [cell for row in updated.cells for cell in row])

    def test_overlay_rejects_too_narrow_cluster(self):
        occ = free_map()
        config = overlay.ObstacleOverlayConfig(
            min_cluster_size=3,
            min_cluster_width_m=0.05,
        )
        points = [
            overlay.BaseFramePoint(0.45, 0.00),
            overlay.BaseFramePoint(0.451, 0.00),
            overlay.BaseFramePoint(0.452, 0.00),
        ]

        with self.assertRaisesRegex(overlay.ObstacleOverlayError, "obstacle_cluster_too_narrow"):
            overlay.build_overlay_map(
                occ,
                points,
                overlay.Pose2D(1.0, 1.0, 0.0),
                config,
            )

    def test_replan_result_writes_artifacts_without_modifying_original_map(self):
        occ = free_map(width=40, height=30)
        original_cells = [list(row) for row in occ.cells]
        points = [
            overlay.BaseFramePoint(0.45, -0.04),
            overlay.BaseFramePoint(0.45, 0.00),
            overlay.BaseFramePoint(0.45, 0.04),
        ]
        config = overlay.ObstacleOverlayConfig(
            min_cluster_size=3,
            min_cluster_width_m=0.05,
            inflate_radius_m=0.10,
            max_start_snap_m=0.30,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = overlay.build_replan_result(
                occ,
                points,
                overlay.Pose2D(1.0, 1.0, 0.0),
                overlay.Pose2D(2.0, 1.0, 0.0),
                "synthetic_test",
                output_dir=Path(tmpdir),
                config=config,
            )

            self.assertTrue(result.success, result.reason)
            self.assertTrue(Path(result.updated_map_yaml).exists())
            self.assertTrue(Path(result.updated_map_pgm).exists())
            self.assertTrue(Path(result.updated_path_csv).exists())
            self.assertTrue(Path(result.updated_waypoints_csv).exists())
            self.assertTrue(Path(result.updated_path_ppm).exists())
            self.assertTrue(Path(result.detected_obstacles_csv).exists())

        self.assertEqual(occ.cells, original_cells)

    def test_runtime_marks_result_failed_when_replan_timeout_is_exceeded(self):
        class Stamp:
            sec = int(time.time())
            nanosec = 0

        class Header:
            frame_id = "scan"
            stamp = Stamp()

        class Rotation:
            x = 0.0
            y = 0.0
            z = 0.0
            w = 1.0

        class Translation:
            x = 0.0
            y = 0.0
            z = 0.0

        class TransformBody:
            translation = Translation()
            rotation = Rotation()

        class Transform:
            header = Header()
            transform = TransformBody()

        class Buffer:
            def lookup_transform(self, *_args):
                return Transform()

        class Node:
            last_scan = argparse.Namespace(
                header=Header(),
                ranges=[0.45, 0.45, 0.45],
                angle_min=-0.08,
                angle_increment=0.08,
                range_min=0.05,
                range_max=4.0,
            )
            last_scan_received_sec = time.time()
            tf_buffer = Buffer()

            def stop_repeatedly(self):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                map_frame="map",
                allow_latest_tf_replan_fallback=False,
                max_replan_scan_age_sec=1.0,
                max_replan_tf_age_sec=10.0,
                replan_timeout_sec=0.0,
                static_map=free_map(width=40, height=30),
                replan_output_dir=Path(tmpdir),
                run_id="timeout_test",
                obstacle_forward_distance_m=0.55,
                obstacle_forward_half_width_m=0.18,
                obstacle_angle_window_deg=45.0,
                obstacle_min_range_m=0.12,
                robot_footprint_radius_m=0.18,
                obstacle_min_cluster_size=3,
                obstacle_min_cluster_width_m=0.05,
                obstacle_inflate_radius_m=0.10,
                run_local_map_min_hit_count=1,
                run_local_map_min_used_points=1,
                max_start_snap_m=0.30,
                max_goal_snap_m=0.30,
                max_replan_path_length_ratio=3.0,
            )

            result = replan_runtime.perform_lidar_replan(
                Node(),
                args,
                overlay.Pose2D(1.0, 1.0, 0.0),
                overlay.Pose2D(2.0, 1.0, 0.0),
                [
                    argparse.Namespace(x=1.0, y=1.0),
                    argparse.Namespace(x=2.0, y=1.0),
                ],
                sequence=1,
            )

        self.assertFalse(result.success)
        self.assertIn("replan_timeout_exceeded", result.reason)

    def test_runtime_live_replan_waits_for_scan_after_stop(self):
        def stamp_from_time(stamp_sec):
            return argparse.Namespace(
                sec=int(stamp_sec),
                nanosec=int((stamp_sec - int(stamp_sec)) * 1_000_000_000),
            )

        class Header:
            def __init__(self, stamp_sec):
                self.frame_id = "scan"
                self.stamp = stamp_from_time(stamp_sec)

        class Scan:
            def __init__(self, stamp_sec):
                self.header = Header(stamp_sec)
                self.ranges = [0.45, 0.45, 0.45]
                self.angle_min = -0.08
                self.angle_increment = 0.08
                self.range_min = 0.05
                self.range_max = 4.0

        class Rotation:
            x = 0.0
            y = 0.0
            z = 0.0
            w = 1.0

        class Translation:
            x = 0.0
            y = 0.0
            z = 0.0

        class TransformBody:
            translation = Translation()
            rotation = Rotation()

        class Transform:
            def __init__(self):
                self.header = Header(time.time())
                self.transform = TransformBody()

        class Buffer:
            def lookup_transform(self, *_args):
                return Transform()

        class Node:
            def __init__(self):
                self.last_scan = Scan(time.time() - 10.0)
                self.last_scan_received_sec = time.time() - 10.0
                self.tf_buffer = Buffer()
                self.stop_count = 0
                self.spin_count = 0

            def stop_repeatedly(self):
                self.stop_count += 1

            def spin_once(self, _timeout_sec):
                self.spin_count += 1
                self.last_scan = Scan(time.time())
                self.last_scan_received_sec = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                map_frame="map",
                allow_latest_tf_replan_fallback=False,
                run_local_map_update_mode="forward",
                run_local_map_max_scan_age_sec=1.0,
                run_local_map_max_tf_age_sec=10.0,
                replan_timeout_sec=5.0,
                static_map=free_map(width=40, height=30),
                replan_output_dir=Path(tmpdir),
                run_id="fresh_scan_test",
                obstacle_forward_distance_m=0.55,
                obstacle_forward_half_width_m=0.18,
                obstacle_angle_window_deg=45.0,
                obstacle_min_range_m=0.12,
                robot_footprint_radius_m=0.18,
                obstacle_min_cluster_size=3,
                obstacle_min_cluster_width_m=0.05,
                obstacle_inflate_radius_m=0.10,
                run_local_map_min_hit_count=1,
                run_local_map_min_used_points=1,
                max_start_snap_m=0.30,
                max_goal_snap_m=0.30,
                max_replan_path_length_ratio=3.0,
            )
            node = Node()

            result = replan_runtime.perform_lidar_replan(
                node,
                args,
                overlay.Pose2D(1.0, 1.0, 0.0),
                overlay.Pose2D(2.0, 1.0, 0.0),
                [
                    argparse.Namespace(x=1.0, y=1.0),
                    argparse.Namespace(x=2.0, y=1.0),
                ],
                sequence=1,
            )

        self.assertEqual(node.stop_count, 1)
        self.assertGreaterEqual(node.spin_count, 1)
        self.assertTrue(result.success, result.reason)
        self.assertLessEqual(result.diagnostics.scan_age_sec, 1.0)

    def test_runtime_initial_collection_uses_multiple_stopped_scans(self):
        now = int(time.time())

        class Stamp:
            def __init__(self, sec):
                self.sec = sec
                self.nanosec = 0

        class Header:
            def __init__(self, sec):
                self.frame_id = "scan"
                self.stamp = Stamp(sec)

        class Scan:
            def __init__(self, sec):
                self.header = Header(sec)
                self.ranges = [0.45]
                self.angle_min = 0.0
                self.angle_increment = 0.1
                self.range_min = 0.05
                self.range_max = 4.0

        class Rotation:
            x = 0.0
            y = 0.0
            z = 0.0
            w = 1.0

        class Translation:
            x = 0.0
            y = 0.0
            z = 0.0

        class TransformBody:
            translation = Translation()
            rotation = Rotation()

        class Transform:
            def __init__(self):
                self.header = Header(int(time.time()))
                self.transform = TransformBody()

        class Buffer:
            def lookup_transform(self, *_args):
                return Transform()

        class Node:
            def __init__(self):
                self.scans = [Scan(now), Scan(now + 1), Scan(now + 2)]
                self.scan_index = 0
                self.last_scan = self.scans[0]
                self.last_scan_received_sec = time.time()
                self.tf_buffer = Buffer()
                self.stop_count = 0
                self.spin_count = 0

            def stop_repeatedly(self):
                self.stop_count += 1

            def spin_once(self, _timeout_sec):
                self.spin_count += 1
                if self.scan_index + 1 < len(self.scans):
                    self.scan_index += 1
                    self.last_scan = self.scans[self.scan_index]
                    self.last_scan_received_sec = time.time()

        args = argparse.Namespace(
            map_frame="map",
            allow_latest_tf_replan_fallback=False,
            run_local_map_initial_scan_count=2,
            run_local_map_max_scan_age_sec=1.0,
            run_local_map_max_tf_age_sec=10.0,
        )
        node = Node()

        observations, _scan, _scan_age, _tf_age, _lookup_mode, collected = (
            replan_runtime.collect_initial_observations(node, args)
        )

        self.assertEqual(collected, 2)
        self.assertEqual(len(observations), 2)
        self.assertEqual(node.stop_count, 1)
        self.assertGreaterEqual(node.spin_count, 1)

    def test_runtime_initial_collection_retries_tf_offset_before_success(self):
        now = int(time.time())

        class Stamp:
            def __init__(self, sec):
                self.sec = sec
                self.nanosec = 0

        class Header:
            def __init__(self, sec):
                self.frame_id = "scan"
                self.stamp = Stamp(sec)

        class Scan:
            def __init__(self, sec):
                self.header = Header(sec)
                self.ranges = [0.45]
                self.angle_min = 0.0
                self.angle_increment = 0.1
                self.range_min = 0.05
                self.range_max = 4.0

        class Rotation:
            x = 0.0
            y = 0.0
            z = 0.0
            w = 1.0

        class Translation:
            x = 0.0
            y = 0.0
            z = 0.0

        class TransformBody:
            translation = Translation()
            rotation = Rotation()

        class Transform:
            def __init__(self):
                self.header = Header(int(time.time()))
                self.transform = TransformBody()

        class Buffer:
            def __init__(self):
                self.calls = 0

            def lookup_transform(self, *_args):
                self.calls += 1
                if self.calls <= 2:
                    raise RuntimeError("tf cache does not cover requested scan time")
                return Transform()

        class Node:
            def __init__(self):
                self.scans = [Scan(now), Scan(now + 1)]
                self.scan_index = 0
                self.last_scan = self.scans[0]
                self.last_scan_received_sec = time.time()
                self.tf_buffer = Buffer()
                self.spin_count = 0

            def stop_repeatedly(self):
                return None

            def spin_once(self, _timeout_sec):
                self.spin_count += 1
                if self.scan_index + 1 < len(self.scans):
                    self.scan_index += 1
                    self.last_scan = self.scans[self.scan_index]
                    self.last_scan_received_sec = time.time()

        args = argparse.Namespace(
            map_frame="map",
            allow_latest_tf_replan_fallback=True,
            run_local_map_initial_scan_count=1,
            run_local_map_max_scan_age_sec=1.0,
            run_local_map_max_tf_age_sec=10.0,
        )
        node = Node()

        observations, _scan, _scan_age, _tf_age, lookup_mode, collected = (
            replan_runtime.collect_initial_observations(node, args)
        )

        self.assertEqual(collected, 1)
        self.assertEqual(len(observations), 1)
        self.assertEqual(lookup_mode, "timestamped")
        self.assertGreaterEqual(node.spin_count, 1)
        self.assertEqual(node.tf_buffer.calls, 3)

    def test_run_local_observations_confirm_hits_without_mutating_static_map(self):
        occ = free_map(width=20, height=20)
        original_cells = [list(row) for row in occ.cells]
        run_map = overlay.RunLocalObstacleMap(
            occ,
            overlay.RunLocalMapConfig(
                min_hit_count=2,
                min_used_points=1,
                static_wall_exclusion_radius_m=0.0,
                inflation_radius_m=0.0,
            ),
        )

        diag1 = run_map.add_observations(overlay.ObservationBatch([
            overlay.MapFrameObservation(1.05, 1.05),
        ]))
        diag2 = run_map.add_observations(overlay.ObservationBatch([
            overlay.MapFrameObservation(1.05, 1.05),
        ]))

        self.assertTrue(diag1.update_accepted)
        self.assertTrue(diag2.update_accepted)
        self.assertEqual(len(run_map.confirmed_raw_cells), 1)
        self.assertEqual(occ.cells, original_cells)

    def test_run_local_filters_invalid_static_bounds_and_wall_band(self):
        occ = free_map(width=20, height=20)
        occ.cells[1][1] = planner.CELL_OCCUPIED
        run_map = overlay.RunLocalObstacleMap(
            occ,
            overlay.RunLocalMapConfig(
                min_hit_count=1,
                min_used_points=1,
                static_wall_exclusion_radius_m=0.11,
                inflation_radius_m=0.0,
            ),
        )

        diag = run_map.add_observations(overlay.ObservationBatch([
            overlay.MapFrameObservation(float("nan"), 0.0),
            overlay.GridCellObservation(-1, 0),
            overlay.GridCellObservation(1, 1),
            overlay.GridCellObservation(2, 1),
            overlay.GridCellObservation(10, 10),
        ]))

        self.assertTrue(diag.update_accepted)
        self.assertEqual(diag.rejected_invalid_range, 1)
        self.assertEqual(diag.rejected_bounds, 1)
        self.assertEqual(diag.rejected_static, 1)
        self.assertEqual(diag.rejected_wall_band, 1)
        self.assertEqual(run_map.confirmed_raw_cells, {(10, 10)})

    def wall_observations(self, wall_x, height, gap_y_values, repeats=2):
        observations = []
        gap = set(gap_y_values)
        for y in range(2, height - 2):
            if y in gap:
                continue
            for _ in range(repeats):
                observations.append(overlay.GridCellObservation(wall_x, y))
        return observations

    def test_run_local_wall_with_wide_hole_plans_through_gap(self):
        occ = free_map(width=60, height=40)
        run_map = overlay.RunLocalObstacleMap(
            occ,
            overlay.RunLocalMapConfig(
                min_hit_count=2,
                min_used_points=1,
                inflation_radius_m=0.20,
                static_wall_exclusion_radius_m=0.0,
                max_start_snap_m=0.5,
                max_goal_snap_m=0.5,
            ),
        )
        run_map.add_observations(overlay.ObservationBatch(
            self.wall_observations(30, occ.height, gap_y_values=range(17, 24))
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            result = overlay.plan_with_run_local_map(
                run_map,
                overlay.Pose2D(0.5, 2.0, 0.0),
                overlay.Pose2D(5.5, 2.0, 0.0),
                "wide_gap",
                output_dir=Path(tmpdir),
                old_remaining_waypoints=[
                    argparse.Namespace(x=0.5, y=2.0),
                    argparse.Namespace(x=5.5, y=2.0),
                ],
            )

        self.assertTrue(result.success, result.reason)
        self.assertFalse(set(result.path_cells).intersection(result.inflated_obstacle_cells))

    def test_run_local_wall_with_narrow_hole_reports_no_connected_path(self):
        occ = free_map(width=60, height=40)
        run_map = overlay.RunLocalObstacleMap(
            occ,
            overlay.RunLocalMapConfig(
                min_hit_count=2,
                min_used_points=1,
                inflation_radius_m=0.20,
                static_wall_exclusion_radius_m=0.0,
                max_start_snap_m=0.5,
                max_goal_snap_m=0.5,
            ),
        )
        run_map.add_observations(overlay.ObservationBatch(
            self.wall_observations(30, occ.height, gap_y_values=[20])
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            result = overlay.plan_with_run_local_map(
                run_map,
                overlay.Pose2D(0.5, 2.0, 0.0),
                overlay.Pose2D(5.5, 2.0, 0.0),
                "narrow_gap",
                output_dir=Path(tmpdir),
            )

        self.assertFalse(result.success)
        self.assertIn(
            result.diagnostics.run_local_no_path_reason,
            {
                overlay.RUN_LOCAL_FAILURE_NO_CONNECTED_PATH,
                "No path exists between start and goal",
            },
        )

    def test_run_local_start_disk_and_goal_blocking_are_explicit(self):
        occ = free_map(width=30, height=30)
        config = overlay.RunLocalMapConfig(
            min_hit_count=1,
            min_used_points=1,
            inflation_radius_m=0.20,
            static_wall_exclusion_radius_m=0.0,
        )

        start_map = overlay.RunLocalObstacleMap(occ, config)
        start_map.add_observations(overlay.ObservationBatch([
            overlay.GridCellObservation(10, 10),
        ]))
        with tempfile.TemporaryDirectory() as tmpdir:
            start_result = overlay.plan_with_run_local_map(
                start_map,
                overlay.Pose2D(1.0, 1.0, 0.0),
                overlay.Pose2D(2.0, 1.0, 0.0),
                "start_blocked",
                output_dir=Path(tmpdir),
            )

        goal_map = overlay.RunLocalObstacleMap(occ, config)
        goal_map.add_observations(overlay.ObservationBatch([
            overlay.GridCellObservation(20, 10),
        ]))
        with tempfile.TemporaryDirectory() as tmpdir:
            goal_result = overlay.plan_with_run_local_map(
                goal_map,
                overlay.Pose2D(1.0, 1.0, 0.0),
                overlay.Pose2D(2.0, 1.0, 0.0),
                "goal_blocked",
                output_dir=Path(tmpdir),
            )

        self.assertFalse(start_result.success)
        self.assertEqual(start_result.reason, overlay.RUN_LOCAL_FAILURE_START_IN_COLLISION)
        self.assertTrue(start_result.diagnostics.run_local_start_cell_blocked)
        self.assertFalse(goal_result.success)
        self.assertEqual(goal_result.reason, overlay.RUN_LOCAL_FAILURE_GOAL_BLOCKED)
        self.assertTrue(goal_result.diagnostics.run_local_goal_cell_blocked)

    def test_run_local_start_clearance_margin_does_not_abort_replan(self):
        occ = free_map(width=60, height=40, resolution=0.05)
        config = overlay.RunLocalMapConfig(
            min_hit_count=1,
            min_used_points=1,
            inflation_radius_m=0.22,
            robot_footprint_radius_m=0.18,
            clearance_margin_m=0.04,
            static_wall_exclusion_radius_m=0.0,
        )
        run_map = overlay.RunLocalObstacleMap(occ, config)
        run_map.add_observations(overlay.ObservationBatch([
            overlay.GridCellObservation(24, 20),
        ]))

        with tempfile.TemporaryDirectory() as tmpdir:
            result = overlay.plan_with_run_local_map(
                run_map,
                overlay.Pose2D(1.025, 1.025, 0.0),
                overlay.Pose2D(2.0, 1.025, 0.0),
                "start_margin",
                output_dir=Path(tmpdir),
            )

        self.assertTrue(result.success, result.reason)
        self.assertFalse(result.diagnostics.run_local_start_cell_blocked)
        self.assertFalse(set(result.path_cells).intersection(result.inflated_obstacle_cells))

    def test_run_local_cell_sources_and_corridor_block_detection(self):
        occ = free_map(width=20, height=20)
        occ.cells[0][0] = planner.CELL_UNKNOWN
        occ.cells[1][1] = planner.CELL_OCCUPIED
        run_map = overlay.RunLocalObstacleMap(
            occ,
            overlay.RunLocalMapConfig(
                min_hit_count=1,
                min_used_points=1,
                inflation_radius_m=0.11,
                static_wall_exclusion_radius_m=0.0,
            ),
        )
        run_map.add_observations(overlay.ObservationBatch([
            overlay.GridCellObservation(8, 10),
        ]))

        self.assertEqual(run_map.cell_source((1, 1)), overlay.CELL_SOURCE_STATIC_OCCUPIED)
        self.assertEqual(run_map.cell_source((0, 0)), overlay.CELL_SOURCE_UNKNOWN)
        self.assertEqual(run_map.cell_source((8, 10)), overlay.CELL_SOURCE_RUN_LOCAL_RAW)
        self.assertEqual(run_map.cell_source((9, 10)), overlay.CELL_SOURCE_RUN_LOCAL_INFLATED)
        self.assertEqual(run_map.cell_source((15, 15)), overlay.CELL_SOURCE_FREE)

        blocked = overlay.path_corridor_blocked_cells(
            occ,
            overlay.Pose2D(0.4, 1.0, 0.0),
            [argparse.Namespace(x=1.5, y=1.0)],
            run_map.inflated_obstacle_cells,
            check_distance_m=1.2,
            corridor_radius_m=0.05,
        )
        self.assertTrue(blocked)


if __name__ == "__main__":
    unittest.main()
