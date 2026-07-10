from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "aufgabe03"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lidar_obstacle_map  # noqa: E402
import map_path_planner as planner  # noqa: E402
from waypoint_following import controller_runtime  # noqa: E402
from waypoint_following import replanning  # noqa: E402
from waypoint_following.models import Pose2D, TrackingPathValidation, Waypoint  # noqa: E402
from waypoint_following.path_progress import (  # noqa: E402
    load_tracking_path_csv,
    validate_tracking_path_geometry,
)


def occupancy_map(tmpdir, width=12, height=8, blocked=()):
    metadata = planner.MapMetadata(
        yaml_path=Path(tmpdir) / "map.yaml",
        image_path=Path(tmpdir) / "map.pgm",
        resolution=0.05,
        origin=(0.0, 0.0, 0.0),
        negate=0,
        occupied_thresh=0.65,
        free_thresh=0.196,
        mode="trinary",
    )
    cells = [
        [planner.CELL_FREE for _x in range(width)]
        for _y in range(height)
    ]
    for x, y in blocked:
        cells[y][x] = planner.CELL_OCCUPIED
    return planner.OccupancyMap(metadata, width, height, cells)


def write_map(tmpdir, occupancy):
    planner.write_occupancy_map_copy(
        occupancy,
        occupancy.metadata.yaml_path,
        occupancy.metadata.image_path,
    )
    return occupancy.metadata.yaml_path


def max_segment_length(points):
    return max(
        math.hypot(points[index].x - points[index - 1].x, points[index].y - points[index - 1].y)
        for index in range(1, len(points))
    )


def heading_change_count(points):
    headings = []
    for index in range(1, len(points)):
        dx = points[index].x - points[index - 1].x
        dy = points[index].y - points[index - 1].y
        if math.hypot(dx, dy) <= 1e-12:
            continue
        headings.append(round(math.degrees(math.atan2(dy, dx)), 1))
    return sum(
        1
        for index in range(1, len(headings))
        if headings[index] != headings[index - 1]
    )


def waypoint_points(cells, occupancy):
    return [
        Waypoint(index, *planner.grid_to_world(cell[0], cell[1], occupancy.metadata))
        for index, cell in enumerate(cells)
    ]


def write_tracking_csv(path, points, occupancy):
    planner.write_path_csv(
        path,
        planner.build_world_path_rows(
            [(point.x, point.y) for point in points],
            occupancy.metadata,
        ),
    )


def runtime_context():
    return SimpleNamespace(
        TrackingPathValidation=TrackingPathValidation,
        default_controller="stop-go",
        load_tracking_path_csv=load_tracking_path_csv,
        validate_tracking_path_geometry=validate_tracking_path_geometry,
    )


class WarnLogger:
    def __init__(self):
        self.messages = []

    def warn(self, message):
        self.messages.append(message)


class PostReplanRouteBlockRepairHelperTests(unittest.TestCase):
    def test_default_extra_update_budget_allows_three_route_block_repairs(self):
        config = lidar_obstacle_map.RunLocalMapConfig(max_updates=3)
        node = SimpleNamespace(
            args=SimpleNamespace(),
            run_local_map=SimpleNamespace(update_count=3, config=config),
            post_replan_route_block_extra_update_count=0,
        )

        for expected_max_updates in (4, 5, 6):
            extra_used, original = (
                replanning.maybe_allow_post_replan_route_block_extra_update(node)
            )

            self.assertTrue(extra_used)
            self.assertIs(original, config)
            self.assertEqual(node.run_local_map.config.max_updates, expected_max_updates)
            node.run_local_map.update_count = expected_max_updates
            replanning.restore_post_replan_route_block_extra_update_budget(
                node,
                original,
            )

        self.assertEqual(node.post_replan_route_block_extra_update_count, 3)
        self.assertIs(node.run_local_map.config, config)
        with self.assertRaisesRegex(
            RuntimeError,
            "post_replan_route_block_repair_budget_exhausted",
        ):
            replanning.maybe_allow_post_replan_route_block_extra_update(node)

    def test_extra_update_budget_extends_and_restores_max_updates(self):
        config = lidar_obstacle_map.RunLocalMapConfig(max_updates=3)
        node = SimpleNamespace(
            args=SimpleNamespace(post_replan_route_block_extra_updates=1),
            run_local_map=SimpleNamespace(update_count=3, config=config),
            post_replan_route_block_extra_update_count=0,
        )

        extra_used, original = (
            replanning.maybe_allow_post_replan_route_block_extra_update(node)
        )

        self.assertTrue(extra_used)
        self.assertIs(original, config)
        self.assertIsNot(node.run_local_map.config, config)
        self.assertEqual(node.run_local_map.config.max_updates, 4)
        self.assertEqual(node.post_replan_route_block_extra_update_count, 1)

        replanning.restore_post_replan_route_block_extra_update_budget(
            node,
            original,
        )

        self.assertIs(node.run_local_map.config, config)
        self.assertEqual(node.run_local_map.config.max_updates, 3)
        with self.assertRaisesRegex(
            RuntimeError,
            "post_replan_route_block_repair_budget_exhausted",
        ):
            replanning.maybe_allow_post_replan_route_block_extra_update(node)

    def test_blocked_scan_min_times_require_strictly_newer_identity(self):
        min_received, min_stamp = replanning.blocked_scan_min_times((10.0, 20.0))

        self.assertGreater(min_stamp, 10.0)
        self.assertGreater(min_received, 20.0)

    def test_detailed_route_signature_uses_all_activated_waypoints(self):
        first = [
            Waypoint(0, 0.0, 0.0),
            Waypoint(1, 0.1, 0.0),
            Waypoint(2, 0.2, 0.0),
        ]
        second = [
            Waypoint(0, 0.0, 0.0),
            Waypoint(1, 0.1, 0.1),
            Waypoint(2, 0.2, 0.0),
        ]

        self.assertNotEqual(
            replanning.detailed_route_signature(None, first),
            replanning.detailed_route_signature(None, second),
        )


class TrackingPathSmoothingTests(unittest.TestCase):
    def test_shortcut_smoothing_preserves_endpoints_and_reduces_stair_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            occupancy = occupancy_map(tmpdir)
            blocked, _inflation = planner.inflate_blocked_cells(
                occupancy,
                0.0,
                block_unknown=True,
            )
            raw_cells = [
                (0, 0),
                (1, 1),
                (2, 1),
                (3, 2),
                (4, 2),
                (5, 3),
                (6, 3),
            ]
            raw_points = waypoint_points(raw_cells, occupancy)

            result = planner.smooth_cell_path_for_tracking(
                occupancy,
                blocked,
                raw_cells,
                spacing_m=0.05,
            )
            smoothed_points = [
                Waypoint(index, x, y)
                for index, (x, y) in enumerate(result.points)
            ]

            self.assertEqual(result.points[0], (raw_points[0].x, raw_points[0].y))
            self.assertEqual(result.points[-1], (raw_points[-1].x, raw_points[-1].y))
            self.assertLessEqual(result.smoothed_length_m, result.raw_length_m)
            self.assertLess(
                heading_change_count(smoothed_points),
                heading_change_count(raw_points),
            )
            self.assertLessEqual(max_segment_length(smoothed_points), 0.05 + 1e-9)

    def test_resampling_does_not_duplicate_final_endpoint(self):
        points = [(0.0, 0.0), (0.1, 0.0)]

        resampled = planner.resample_world_path(points, spacing_m=0.05)

        self.assertEqual(resampled[-1], points[-1])
        self.assertNotEqual(resampled[-2], points[-1])

    def test_shortcut_smoothing_does_not_cut_through_blocked_cells(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            occupancy = occupancy_map(tmpdir, blocked={(2, 0)})
            blocked, _inflation = planner.inflate_blocked_cells(
                occupancy,
                0.0,
                block_unknown=True,
            )
            raw_cells = [
                (0, 0),
                (1, 0),
                (1, 1),
                (2, 1),
                (3, 1),
                (3, 0),
                (4, 0),
            ]

            result = planner.smooth_cell_path_for_tracking(
                occupancy,
                blocked,
                raw_cells,
                spacing_m=0.05,
            )

            for index in range(1, len(result.points)):
                self.assertTrue(
                    planner.world_segment_is_clear(
                        occupancy,
                        blocked,
                        result.points[index - 1],
                        result.points[index],
                    )
                )

    def test_static_tracking_smoothing_writes_run_scoped_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            occupancy = occupancy_map(tmpdir)
            map_yaml = write_map(tmpdir, occupancy)
            raw_cells = [
                (0, 0),
                (1, 1),
                (2, 1),
                (3, 2),
                (4, 2),
                (5, 3),
                (6, 3),
            ]
            raw_points = waypoint_points(raw_cells, occupancy)
            tracking_csv = Path(tmpdir) / "tracking.csv"
            write_tracking_csv(tracking_csv, raw_points, occupancy)
            args = SimpleNamespace(
                controller="pure-pursuit",
                tracking_path_csv=tracking_csv,
                tracking_max_segment_m=0.30,
                tracking_endpoint_tolerance_m=0.10,
                tracking_start_tolerance_m=0.20,
                allow_tracking_path_mismatch=False,
                tracking_path_smoothing="shortcut",
                tracking_path_smoothing_spacing_m=0.05,
                static_map=map_yaml,
                pure_pursuit_lookahead_guard_static_inflation_radius_m=0.0,
                replan_output_dir=Path(tmpdir),
                run_id="static_smoothing_test",
            )

            tracking_points, validation = controller_runtime.prepare_tracking_setup(
                args,
                [raw_points[0], raw_points[-1]],
                runtime_context(),
            )

            artifact = Path(tmpdir) / "static_smoothing_test_tracking_path.csv"
            self.assertEqual(validation.source, "csv_smoothed")
            self.assertEqual(args.tracking_path_smoothing_status, "smoothed")
            self.assertEqual(args.tracking_path_smoothing_artifact, str(artifact))
            self.assertTrue(artifact.exists())
            self.assertGreaterEqual(len(tracking_points), 2)

    def test_static_tracking_smoothing_falls_back_to_raw_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            occupancy = occupancy_map(tmpdir, blocked={(1, 0)})
            map_yaml = write_map(tmpdir, occupancy)
            raw_cells = [(0, 0), (1, 0), (2, 0)]
            raw_points = waypoint_points(raw_cells, occupancy)
            tracking_csv = Path(tmpdir) / "tracking.csv"
            write_tracking_csv(tracking_csv, raw_points, occupancy)
            args = SimpleNamespace(
                controller="pure-pursuit",
                tracking_path_csv=tracking_csv,
                tracking_max_segment_m=0.30,
                tracking_endpoint_tolerance_m=0.10,
                tracking_start_tolerance_m=0.20,
                allow_tracking_path_mismatch=False,
                tracking_path_smoothing="shortcut",
                tracking_path_smoothing_spacing_m=0.05,
                static_map=map_yaml,
                pure_pursuit_lookahead_guard_static_inflation_radius_m=0.0,
                replan_output_dir=Path(tmpdir),
                run_id="fallback_test",
            )
            logger = WarnLogger()

            tracking_points, validation = controller_runtime.prepare_tracking_setup(
                args,
                [raw_points[0], raw_points[-1]],
                runtime_context(),
                logger=logger,
            )

            self.assertEqual(validation.source, "csv")
            self.assertEqual(args.tracking_path_smoothing_status, "fallback_raw")
            self.assertFalse((Path(tmpdir) / "fallback_test_tracking_path.csv").exists())
            self.assertEqual(
                [(point.x, point.y) for point in tracking_points],
                [(point.x, point.y) for point in raw_points],
            )
            self.assertTrue(logger.messages)

    def test_run_local_replan_keeps_raw_artifacts_and_adds_tracking_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            occupancy = occupancy_map(tmpdir)
            config = lidar_obstacle_map.RunLocalMapConfig(
                tracking_path_smoothing="shortcut",
                tracking_path_smoothing_spacing_m=0.05,
            )
            run_local_map = lidar_obstacle_map.RunLocalObstacleMap(
                occupancy,
                config,
            )
            result = lidar_obstacle_map.plan_with_run_local_map(
                run_local_map,
                lidar_obstacle_map.Pose2D(0.025, 0.025, 0.0),
                lidar_obstacle_map.Pose2D(0.325, 0.175, 0.0),
                "run_local_smoothing_test",
                output_dir=tmpdir,
                artifact_prefix="run_local_smoothing_test",
            )

            self.assertTrue(result.success)
            self.assertTrue(Path(result.updated_path_csv).exists())
            self.assertTrue(Path(result.updated_waypoints_csv).exists())
            self.assertTrue(Path(result.updated_tracking_path_csv).exists())
            self.assertEqual(
                result.diagnostics.tracking_path_smoothing_status,
                "smoothed",
            )
            self.assertGreaterEqual(len(result.path_points), 2)

    def test_replan_tracking_validation_falls_back_to_raw_smoothed_result(self):
        diagnostics = lidar_obstacle_map.ObstacleOverlayDiagnostics()
        diagnostics.tracking_path_smoothing_mode = "shortcut"
        diagnostics.tracking_path_smoothing_status = "smoothed"
        diagnostics.tracking_path_smoothing_raw_point_count = 3
        diagnostics.tracking_path_smoothing_smoothed_point_count = 2
        diagnostics.tracking_path_smoothing_raw_length_m = 0.10
        diagnostics.tracking_path_smoothing_smoothed_length_m = 1.0
        diagnostics.tracking_path_smoothing_artifact = "/tmp/smoothed.csv"
        result = SimpleNamespace(
            diagnostics=diagnostics,
            path_points=[
                Waypoint(0, 0.0, 0.0),
                Waypoint(1, 1.0, 0.0),
            ],
            raw_path_points=[
                Waypoint(0, 0.0, 0.0),
                Waypoint(1, 0.05, 0.0),
                Waypoint(2, 0.10, 0.0),
            ],
        )
        logger = WarnLogger()
        node = SimpleNamespace(
            args=SimpleNamespace(
                controller="pure-pursuit",
                tracking_max_segment_m=0.30,
                tracking_endpoint_tolerance_m=0.10,
                tracking_start_tolerance_m=0.20,
                allow_tracking_path_mismatch=False,
            ),
            get_logger=lambda: logger,
        )
        replanned = [
            Waypoint(0, 0.0, 0.0),
            Waypoint(1, 0.10, 0.0),
        ]

        replanning.remember_replan_tracking_replacement(
            node,
            result,
            replanned,
            Pose2D(0.0, 0.0, 0.0),
        )

        self.assertEqual(
            [(point.x, point.y) for point in node.last_replan_tracking_points],
            [(0.0, 0.0), (0.05, 0.0), (0.10, 0.0)],
        )
        self.assertEqual(node.args.tracking_path_smoothing_status, "fallback_raw")
        self.assertEqual(
            diagnostics.tracking_path_smoothing_reason.startswith(
                "smoothed_validation_failed:",
            ),
            True,
        )
        self.assertTrue(logger.messages)

    def test_route_projection_notes_include_cte_percentiles_and_limiter_counts(self):
        node = SimpleNamespace(
            diagnostics=SimpleNamespace(
                path_profile_status="",
                path_profile_speed_cap_mps=None,
                path_profile_lookahead_m=None,
                path_profile_distance_to_heading_break_m=None,
                path_profile_heading_break_delta_deg=None,
            ),
            max_cross_track_error_m=0.0,
            cross_track_error_sum_m=0.0,
            cross_track_error_count=0,
            cross_track_error_samples_m=[],
            max_route_heading_error_deg=0.0,
            angular_feasibility_sample_count=0,
            angular_feasibility_limited_count=0,
            angular_feasibility_min_scale=1.0,
            angular_feasibility_last_scale=None,
            angular_feasibility_max_raw_angular_z_radps=0.0,
            max_projection_backward_delta_m=0.0,
            max_rotate_anchor_backward_delta_m=0.0,
            max_rotate_anchor_forward_delta_m=0.0,
            last_rotate_anchor_aligned_samples=0,
            max_rotate_anchor_aligned_samples=0,
            pure_pursuit_rotate_anchor_activations=0,
            post_rotate_branch_rejected_wrong_heading_count=0,
            post_rotate_branch_max_heading_error_deg=0.0,
            post_rotate_branch_lock_activations=0,
            post_rotate_branch_ambiguity_failures=0,
            post_rotate_branch_target_clip_count=0,
            post_rotate_branch_heading_break_handoff_count=0,
            post_rotate_branch_physical_handoff_count=0,
            last_projection_acquisition_status="",
            last_projection_lock_sample_count=0,
            last_recorded_pure_pursuit_status=None,
            pure_pursuit_rotate_gate_entries=0,
            _current_path_controller=None,
        )
        step = SimpleNamespace(
            route_projection_result=SimpleNamespace(
                cross_track_error_m=0.10,
                heading_error_to_route_deg=5.0,
            ),
            route_heading_result=SimpleNamespace(
                heading_error_deg=5.0,
                source="route",
            ),
            forward_control_result=SimpleNamespace(
                angular_feasibility_limited=True,
                angular_feasibility_scale=0.5,
                raw_angular_z=0.18,
            ),
            pure_pursuit_rotate_reason="",
            pure_pursuit_rotate_source="",
            path_profile_result=None,
            pure_pursuit_status="tracking",
            mode="track",
        )
        controller_runtime.record_route_projection_result(
            node,
            step,
            SimpleNamespace(),
        )
        context = SimpleNamespace(
            default_controller="pure-pursuit",
            projection_lock_required_samples=3,
            projection_lock_progress_tolerance_m=0.02,
            rotate_anchor_route_heading_exit_samples=2,
            post_rotate_branch_heading_tolerance_deg=20.0,
            post_rotate_branch_release_stable_samples=2,
            post_rotate_branch_min_release_progress_m=0.04,
            post_rotate_branch_end_lateral_tolerance_m=0.05,
            post_rotate_zero_linear_eps_mps=0.001,
        )
        notes = controller_runtime.notes_with_route_projection_metadata(
            "notes",
            SimpleNamespace(controller="pure-pursuit"),
            node,
            context,
        )

        self.assertIn("pure_pursuit_p90_abs_cross_track_error_m=0.100", notes)
        self.assertIn("pure_pursuit_angular_feasibility_limited_count=1", notes)
        self.assertIn("pure_pursuit_angular_feasibility_min_scale=0.500", notes)


if __name__ == "__main__":
    unittest.main()
