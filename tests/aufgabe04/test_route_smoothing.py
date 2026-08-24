import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.planning.costmap import Costmap  # noqa: E402
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds  # noqa: E402
from scripts.aufgabe04.navigation.planning.exact_start_connector import (  # noqa: E402
    prepend_certified_exact_start,
)
from scripts.aufgabe04.navigation.planning.global_planner import plan_route  # noqa: E402
from scripts.aufgabe04.navigation.planning.map_io import (  # noqa: E402
    CELL_FREE,
    CELL_OCCUPIED,
    MapMetadata,
    OccupancyGrid,
)
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D  # noqa: E402
from scripts.aufgabe04.navigation.execution.route_context import build_station_route_dry_run  # noqa: E402
from scripts.aufgabe04.navigation.planning.route_smoothing import (  # noqa: E402
    greedy_line_of_sight_shortcut,
    segment_is_collision_free,
    smooth_plan_route_from_exact_start_with_summary,
    smooth_plan_route_result,
    smooth_plan_route_result_with_summary,
)
from scripts.aufgabe04.stations.models import Station  # noqa: E402
from scripts.aufgabe04.stations.models import StationPose  # noqa: E402


def grid_from_rows(rows, resolution=1.0):
    return OccupancyGrid(
        metadata=MapMetadata(
            yaml_path=Path("map.yaml"),
            image_path=Path("map.pgm"),
            resolution=resolution,
            origin=(0.0, 0.0, 0.0),
            negate=0,
            occupied_thresh=0.65,
            free_thresh=0.20,
            mode="trinary",
        ),
        width=len(rows[0]),
        height=len(rows),
        cells=tuple(tuple(row) for row in rows),
    )


class RouteSmoothingTest(unittest.TestCase):
    def test_open_astar_route_compacts_to_certified_endpoint_segment(self):
        costmap = Costmap.from_occupancy_grid(
            grid_from_rows([[CELL_FREE] * 5 for _ in range(5)])
        )
        result = plan_route(
            costmap,
            Pose2D(0.5, 0.5),
            Pose2D(4.5, 4.5),
            snap_radius_m=0.0,
        )

        smoothed = smooth_plan_route_result_with_summary(result, costmap=costmap)

        self.assertTrue(smoothed.summary.optimized)
        self.assertEqual(smoothed.summary.input_point_count, 5)
        self.assertEqual(smoothed.summary.output_point_count, 2)
        self.assertEqual(
            tuple(point.pose for point in smoothed.result.route.points),
            (Pose2D(0.5, 0.5, 0.0), Pose2D(4.5, 4.5, 0.0)),
        )
        self.assertAlmostEqual(
            smoothed.result.route.length_m,
            smoothed.summary.output_length_m,
        )

    def test_shortcut_does_not_cross_blocked_cell(self):
        rows = [[CELL_FREE] * 5 for _ in range(5)]
        rows[2][2] = CELL_OCCUPIED
        costmap = Costmap.from_occupancy_grid(grid_from_rows(rows))
        result = plan_route(
            costmap,
            Pose2D(0.5, 0.5),
            Pose2D(4.5, 4.5),
            snap_radius_m=0.0,
        )

        smoothed = smooth_plan_route_result(result, costmap=costmap)
        poses = tuple(point.pose for point in smoothed.route.points)

        self.assertNotEqual(poses, (Pose2D(0.5, 0.5, 0.0), Pose2D(4.5, 4.5, 0.0)))
        self.assertTrue(
            all(costmap.is_traversable(point.cell) for point in smoothed.route.points)
        )
        self.assertTrue(
            all(
                segment_is_collision_free(costmap, first.pose, second.pose)
                for first, second in zip(
                    smoothed.route.points,
                    smoothed.route.points[1:],
                )
            )
        )

    def test_exact_start_smoothing_checks_live_keepout_overlay(self):
        base = Costmap.from_occupancy_grid(
            grid_from_rows([[CELL_FREE] * 7 for _ in range(5)])
        )
        planning = base.with_blocked_cells({GridCell(3, 0)})
        exact_start = Pose2D(0.6, 0.5, 0.0)
        result = plan_route(
            planning,
            exact_start,
            Pose2D(6.5, 0.5, 0.0),
            snap_radius_m=0.0,
        )

        smoothed = smooth_plan_route_from_exact_start_with_summary(
            result,
            costmap=planning,
            exact_start=exact_start,
        )
        route = smoothed.result.route
        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.points[0].pose, exact_start)
        self.assertGreater(len(route.points), 2)
        self.assertTrue(
            all(
                segment_is_collision_free(planning, first.pose, second.pose)
                for first, second in zip(route.points, route.points[1:])
            )
        )
        with_connector, evidence = prepend_certified_exact_start(
            smoothed.result,
            base_costmap=base,
            start=exact_start,
            required_clearance_m=0.0,
        )
        self.assertFalse(evidence.required)
        self.assertTrue(evidence.validated)
        self.assertEqual(with_connector.route, route)

    def test_exact_start_smoothing_preserves_one_point_no_motion_route(self):
        costmap = Costmap.from_occupancy_grid(
            grid_from_rows([[CELL_FREE] * 3 for _ in range(3)])
        )
        exact_start = Pose2D(1.5, 1.5, 0.0)
        result = plan_route(
            costmap,
            exact_start,
            exact_start,
            snap_radius_m=0.0,
        )

        smoothed = smooth_plan_route_from_exact_start_with_summary(
            result,
            costmap=costmap,
            exact_start=exact_start,
        )

        self.assertIsNotNone(smoothed.result.route)
        self.assertEqual(len(smoothed.result.route.points), 1)
        self.assertEqual(smoothed.result.route.points[0].pose, exact_start)
        self.assertFalse(smoothed.summary.optimized)
        self.assertEqual(smoothed.summary.skipped_reason, "already_minimal")

    def test_protected_zero_length_handoff_is_retained(self):
        costmap = Costmap.from_occupancy_grid(
            grid_from_rows([[CELL_FREE] * 5 for _ in range(5)])
        )
        poses = (
            Pose2D(0.5, 0.5),
            Pose2D(1.5, 1.5),
            Pose2D(1.5, 1.5),
            Pose2D(4.5, 4.5),
        )

        shortened = greedy_line_of_sight_shortcut(
            costmap,
            poses,
            protected_indices=(1, 2),
        )

        self.assertEqual(shortened[1:3], poses[1:3])

    def test_route_context_records_line_of_sight_optimization_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "map.pgm").write_text(
                "P2\n5 5\n255\n" + " ".join(["255"] * 25) + "\n"
            )
            map_yaml = root / "map.yaml"
            map_yaml.write_text(
                "\n".join(
                    [
                        "image: map.pgm",
                        "resolution: 1.0",
                        "origin: [0.0, 0.0, 0.0]",
                        "negate: 0",
                        "occupied_thresh: 0.65",
                        "free_thresh: 0.20",
                        "mode: trinary",
                    ]
                )
                + "\n"
            )
            station_map = {
                "A": Station("A", StationPose(4.5, 4.5, 0.0), 0.0, 0.2),
            }

            dry_run = build_station_route_dry_run(
                map_yaml,
                ["A"],
                station_map=station_map,
                start=Pose2D(0.5, 0.5, 0.0),
                snap_radius_m=0.0,
                arena_bounds=ArenaBounds(
                    length_m=5.0,
                    width_m=5.0,
                    center_x_m=2.5,
                    center_y_m=2.5,
                ),
            )

        metadata = dry_run.metadata["line_of_sight_route_optimization"]
        self.assertTrue(metadata["enabled"])
        self.assertEqual(metadata["optimized_leg_count"], 1)
        self.assertLess(metadata["output_point_count"], metadata["input_point_count"])


if __name__ == "__main__":
    unittest.main()
