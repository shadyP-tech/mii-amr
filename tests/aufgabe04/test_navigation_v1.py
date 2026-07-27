import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.costmap import (  # noqa: E402
    CELL_SOURCE_INFLATED,
    CELL_SOURCE_RUN_LOCAL,
    Costmap,
)
from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds  # noqa: E402
from scripts.aufgabe04.navigation.global_planner import (  # noqa: E402
    FAILURE_GOAL_SNAP_FAILED,
    FAILURE_NO_PATH,
    FAILURE_START_SNAP_FAILED,
    STATUS_OK,
    plan_route,
)
from scripts.aufgabe04.navigation.map_io import (  # noqa: E402
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    MapMetadata,
    OccupancyGrid,
    load_occupancy_grid,
    read_pgm,
)
from scripts.aufgabe04.navigation.models import GridCell, Pose2D  # noqa: E402
from scripts.aufgabe04.navigation.route_context import build_station_route_dry_run  # noqa: E402
from scripts.aufgabe04.navigation.route_overlay import (  # noqa: E402
    RouteOverlayInput,
    render_route_overlay_svg,
    world_to_svg_units,
)
from scripts.aufgabe04.navigation.run_station_route import (  # noqa: E402
    build_parser as build_route_parser,
    main as run_station_route_main,
)
from scripts.aufgabe04.navigation.station_approach import navigation_targets_from_visits  # noqa: E402
from scripts.aufgabe04.stations.models import ApproachTarget, StationPose, StationVisit  # noqa: E402
from scripts.aufgabe04.stations.models import Station  # noqa: E402
from scripts.aufgabe04.stations.station_layout_io import write_station_layout_json  # noqa: E402


def metadata(resolution=1.0):
    return MapMetadata(
        yaml_path=Path("map.yaml"),
        image_path=Path("map.pgm"),
        resolution=resolution,
        origin=(0.0, 0.0, 0.0),
        negate=0,
        occupied_thresh=0.65,
        free_thresh=0.20,
        mode="trinary",
    )


def grid_from_rows(rows, resolution=1.0):
    return OccupancyGrid(
        metadata=metadata(resolution=resolution),
        width=len(rows[0]),
        height=len(rows),
        cells=tuple(tuple(row) for row in rows),
    )


class NavigationMapIoTest(unittest.TestCase):
    def test_loads_relative_p2_map_and_flips_image_y(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "tiny.pgm").write_text("P2\n2 2\n255\n0 255\n128 255\n")
            (root / "map.yaml").write_text(
                "\n".join(
                    [
                        "image: tiny.pgm",
                        "resolution: 0.5",
                        "origin: [0.0, 0.0, 0.0]",
                        "negate: 0",
                        "occupied_thresh: 0.65",
                        "free_thresh: 0.20",
                        "mode: trinary",
                    ]
                )
                + "\n"
            )

            grid = load_occupancy_grid(root / "map.yaml")

        self.assertEqual(grid.metadata.image_path.name, "tiny.pgm")
        self.assertEqual(grid.cells[1][0], CELL_OCCUPIED)
        self.assertEqual(grid.cells[0][0], CELL_UNKNOWN)
        self.assertEqual(grid.cells[0][1], CELL_FREE)

    def test_loads_p5_pgm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tiny.pgm"
            path.write_bytes(b"P5\n2 1\n255\n\x00\xff")

            image = read_pgm(path)

        self.assertEqual(image.width, 2)
        self.assertEqual(image.pixels, ((0, 255),))

    def test_rejects_missing_fields_nontrinary_and_nonzero_yaw(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "map.yaml").write_text("image: tiny.pgm\n")
            with self.assertRaisesRegex(ValueError, "missing required"):
                load_occupancy_grid(root / "map.yaml")

            base = [
                "image: tiny.pgm",
                "resolution: 0.5",
                "origin: [0.0, 0.0, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.20",
            ]
            (root / "tiny.pgm").write_text("P2\n1 1\n255\n255\n")
            (root / "map.yaml").write_text("\n".join(base + ["mode: scale"]) + "\n")
            with self.assertRaisesRegex(ValueError, "trinary"):
                load_occupancy_grid(root / "map.yaml")

            yaw_map = base.copy()
            yaw_map[2] = "origin: [0.0, 0.0, 0.1]"
            (root / "map.yaml").write_text("\n".join(yaw_map) + "\n")
            with self.assertRaisesRegex(ValueError, "zero-yaw"):
                load_occupancy_grid(root / "map.yaml")


class CostmapTest(unittest.TestCase):
    def test_unknown_is_blocked_by_default(self):
        costmap = Costmap.from_occupancy_grid(grid_from_rows([[CELL_UNKNOWN, CELL_FREE]]))

        self.assertTrue(costmap.is_blocked(GridCell(0, 0)))
        self.assertFalse(costmap.is_blocked(GridCell(1, 0)))

    def test_inflation_and_blocked_cells_are_immutable(self):
        costmap = Costmap.from_occupancy_grid(
            grid_from_rows(
                [
                    [CELL_FREE, CELL_FREE, CELL_FREE],
                    [CELL_FREE, CELL_OCCUPIED, CELL_FREE],
                    [CELL_FREE, CELL_FREE, CELL_FREE],
                ],
                resolution=0.1,
            )
        )

        inflated = costmap.with_inflation(0.1)
        run_local = costmap.with_blocked_cells([GridCell(0, 0)], source=CELL_SOURCE_RUN_LOCAL)

        self.assertFalse(costmap.is_blocked(GridCell(0, 1)))
        self.assertTrue(inflated.is_blocked(GridCell(0, 1)))
        self.assertEqual(inflated.cell_sources[GridCell(0, 1)], CELL_SOURCE_INFLATED)
        self.assertFalse(costmap.is_blocked(GridCell(0, 0)))
        self.assertEqual(run_local.cell_sources[GridCell(0, 0)], CELL_SOURCE_RUN_LOCAL)

    def test_inflation_uses_continuous_cell_square_clearance_at_diagonal(self):
        rows = [[CELL_FREE] * 9 for _ in range(9)]
        rows[1][1] = CELL_OCCUPIED
        costmap = Costmap.from_occupancy_grid(
            grid_from_rows(rows, resolution=0.05)
        )

        inflated = costmap.with_inflation(0.23)

        # Offset (4, 4) has centres 0.283 m apart, but the two cell squares
        # are only hypot(0.15, 0.15)=0.212 m apart and must be blocked.
        self.assertTrue(inflated.is_blocked(GridCell(5, 5)))
        # Offset (6, 6) has 0.354 m square-to-square clearance and remains
        # available, demonstrating that this is not a rectangular halo.
        self.assertFalse(inflated.is_blocked(GridCell(7, 7)))

    def test_inflation_rejects_nonfinite_or_negative_radius(self):
        costmap = Costmap.from_occupancy_grid(
            grid_from_rows([[CELL_FREE]], resolution=0.05)
        )
        for radius in (float("nan"), float("inf"), -0.01):
            with self.subTest(radius=radius):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    costmap.with_inflation(radius)

    def test_world_grid_conversion_uses_cell_centers(self):
        costmap = Costmap.from_occupancy_grid(grid_from_rows([[CELL_FREE]], resolution=0.5))

        self.assertEqual(costmap.world_to_grid(Pose2D(0.25, 0.25)), GridCell(0, 0))
        self.assertEqual(costmap.grid_to_world(GridCell(0, 0)), Pose2D(0.25, 0.25, 0.0))


class RouteOverlayTest(unittest.TestCase):
    def test_world_to_svg_units_uses_origin_and_image_y_flip(self):
        grid = OccupancyGrid(
            metadata=MapMetadata(
                yaml_path=Path("map.yaml"),
                image_path=Path("map.pgm"),
                resolution=0.5,
                origin=(-1.0, -2.0, 0.0),
                negate=0,
                occupied_thresh=0.65,
                free_thresh=0.20,
                mode="trinary",
            ),
            width=4,
            height=6,
            cells=tuple(tuple([CELL_FREE] * 4) for _ in range(6)),
        )

        bottom_left = world_to_svg_units(grid, Pose2D(-1.0, -2.0))
        top_right = world_to_svg_units(grid, Pose2D(1.0, 1.0))

        self.assertEqual(bottom_left.x, 0.0)
        self.assertEqual(bottom_left.y, 6.0)
        self.assertEqual(top_right.x, 4.0)
        self.assertEqual(top_right.y, 0.0)

    def test_render_svg_includes_validation_layers_and_metadata(self):
        grid = grid_from_rows([[CELL_FREE, CELL_OCCUPIED], [CELL_UNKNOWN, CELL_FREE]])
        result = plan_route(
            Costmap.from_occupancy_grid(grid),
            Pose2D(0.5, 0.5),
            Pose2D(1.5, 1.5),
            snap_radius_m=0.0,
        )
        station = Station("A", StationPose(1.5, 1.5, 0.0), 0.5, 0.25)
        visits = [
            StationVisit(
                "A",
                ApproachTarget("A", StationPose(1.0, 1.5, 0.0), stop_distance_m=0.5),
            )
        ]
        targets = tuple(navigation_targets_from_visits(visits, Costmap.from_occupancy_grid(grid)))

        svg = render_route_overlay_svg(
            RouteOverlayInput(
                grid=grid,
                arena_bounds=ArenaBounds(length_m=1.0, width_m=1.0),
                stations={"A": station},
                visits=visits,
                targets=targets,
                results=(result,),
                metadata={
                    "frame_id": "map",
                    "map_yaml": "map.yaml",
                    "map_image_sha256": "1234567890abcdef",
                    "origin": grid.metadata.origin,
                },
            )
        )

        self.assertIn('id="occupancy-map"', svg)
        self.assertIn('id="arena-bounds"', svg)
        self.assertIn('id="stations"', svg)
        self.assertIn('id="approach-targets"', svg)
        self.assertIn('id="planned-routes"', svg)
        self.assertIn("frame_id: map", svg)
        self.assertIn("visual check only", svg)


class GlobalPlannerTest(unittest.TestCase):
    def test_plans_success_route_with_diagnostics(self):
        costmap = Costmap.from_occupancy_grid(
            grid_from_rows(
                [
                    [CELL_FREE, CELL_FREE, CELL_FREE],
                    [CELL_FREE, CELL_FREE, CELL_FREE],
                    [CELL_FREE, CELL_FREE, CELL_FREE],
                ]
            )
        )

        result = plan_route(costmap, Pose2D(0.5, 0.5), Pose2D(2.5, 2.5), snap_radius_m=0.0)

        self.assertEqual(result.diagnostics.status, STATUS_OK)
        self.assertIsNotNone(result.route)
        self.assertGreater(result.route.length_m, 0.0)

    def test_no_diagonal_corner_cutting(self):
        costmap = Costmap.from_occupancy_grid(
            grid_from_rows(
                [
                    [CELL_FREE, CELL_OCCUPIED, CELL_FREE],
                    [CELL_OCCUPIED, CELL_FREE, CELL_FREE],
                    [CELL_FREE, CELL_FREE, CELL_FREE],
                ]
            )
        )

        result = plan_route(costmap, Pose2D(0.5, 0.5), Pose2D(2.5, 2.5), snap_radius_m=0.0)

        self.assertEqual(result.failure.reason, FAILURE_NO_PATH)

    def test_snap_success_and_snap_failure(self):
        costmap = Costmap.from_occupancy_grid(
            grid_from_rows([[CELL_OCCUPIED, CELL_FREE, CELL_FREE]], resolution=0.1)
        )

        success = plan_route(costmap, Pose2D(0.05, 0.05), Pose2D(0.25, 0.05), snap_radius_m=0.11)
        failure = plan_route(costmap, Pose2D(0.05, 0.05), Pose2D(0.25, 0.05), snap_radius_m=0.0)

        self.assertEqual(success.diagnostics.status, STATUS_OK)
        self.assertEqual(failure.failure.reason, FAILURE_START_SNAP_FAILED)

    def test_goal_snap_failure(self):
        costmap = Costmap.from_occupancy_grid(
            grid_from_rows([[CELL_FREE, CELL_FREE, CELL_OCCUPIED]], resolution=0.1)
        )

        result = plan_route(costmap, Pose2D(0.05, 0.05), Pose2D(0.25, 0.05), snap_radius_m=0.0)

        self.assertEqual(result.failure.reason, FAILURE_GOAL_SNAP_FAILED)


class StationDryRunTest(unittest.TestCase):
    def test_route_defaults_use_routes_subfolder(self):
        parser = build_route_parser()
        args = parser.parse_args(["--map", "map.yaml", "--stations", "A"])

        self.assertEqual(args.route_csv, Path("results/aufgabe04/routes/station_route.csv"))
        self.assertEqual(
            args.diagnostics_json,
            Path("results/aufgabe04/routes/station_route_diagnostics.json"),
        )
        self.assertIsNone(args.overlay_svg)

    def test_dry_run_does_not_apply_station_keepouts_as_transit_obstacles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "map.pgm").write_text("P2\n7 3\n255\n" + " ".join(["255"] * 21) + "\n")
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
                "A": Station("A", StationPose(6.0, 1.5, 0.0), 0.5, 0.2),
                "B": Station("B", StationPose(3.5, 1.5, 0.0), 0.0, 1.5),
            }

            dry_run = build_station_route_dry_run(
                map_yaml,
                ["A"],
                station_map=station_map,
                start=Pose2D(0.5, 1.5, 0.0),
                inflation_radius_m=0.0,
                snap_radius_m=0.0,
                arena_bounds=ArenaBounds(
                    length_m=7.0,
                    width_m=3.0,
                    center_x_m=3.5,
                    center_y_m=1.5,
                ),
            )

            self.assertIsNone(dry_run.results[0].failure)
            self.assertGreater(dry_run.results[0].route.length_m, 0.0)

    def test_station_target_validation_fails_closed_on_blocked_target(self):
        costmap = Costmap.from_occupancy_grid(grid_from_rows([[CELL_OCCUPIED]]))
        visits = [
            StationVisit(
                "A",
                ApproachTarget("A", StationPose(0.5, 0.5, 0.0), stop_distance_m=0.2),
            )
        ]

        with self.assertRaisesRegex(ValueError, "blocked"):
            navigation_targets_from_visits(visits, costmap)

    def test_dry_run_cli_emits_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "map.pgm").write_text("P2\n5 5\n255\n" + " ".join(["255"] * 25) + "\n")
            (root / "map.yaml").write_text(
                "\n".join(
                    [
                        "image: map.pgm",
                        "resolution: 0.5",
                        "origin: [-1.0, -1.0, 0.0]",
                        "negate: 0",
                        "occupied_thresh: 0.65",
                        "free_thresh: 0.20",
                        "mode: trinary",
                    ]
                )
                + "\n"
            )
            route_csv = root / "route.csv"
            diagnostics_json = root / "diagnostics.json"

            exit_code = run_station_route_main(
                [
                    "--map",
                    str(root / "map.yaml"),
                    "--stations",
                    "B",
                    "--start-x",
                    "-0.5",
                    "--start-y",
                    "-0.5",
                    "--route-csv",
                    str(route_csv),
                    "--diagnostics-json",
                    str(diagnostics_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(route_csv.exists())
            diagnostics = json.loads(diagnostics_json.read_text())
            self.assertEqual(diagnostics["metadata"]["stations"], ["B"])

    def test_dry_run_cli_uses_station_layout_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "map.pgm").write_text("P2\n7 7\n255\n" + " ".join(["255"] * 49) + "\n")
            (root / "map.yaml").write_text(
                "\n".join(
                    [
                        "image: map.pgm",
                        "resolution: 0.5",
                        "origin: [-1.5, -1.5, 0.0]",
                        "negate: 0",
                        "occupied_thresh: 0.65",
                        "free_thresh: 0.20",
                        "mode: trinary",
                    ]
                )
                + "\n"
            )
            layout_json = root / "layout.json"
            route_csv = root / "route.csv"
            diagnostics_json = root / "diagnostics.json"
            write_station_layout_json(
                layout_json,
                [Station("B", StationPose(-0.5, 0.5, 0.0), 0.5, 0.0)],
                {"seed": 99},
            )

            exit_code = run_station_route_main(
                [
                    "--map",
                    str(root / "map.yaml"),
                    "--stations",
                    "B",
                    "--station-layout-json",
                    str(layout_json),
                    "--start-x",
                    "-1.0",
                    "--start-y",
                    "-1.0",
                    "--route-csv",
                    str(route_csv),
                    "--diagnostics-json",
                    str(diagnostics_json),
                ]
            )
            diagnostics = json.loads(diagnostics_json.read_text())
            route_text = route_csv.read_text()

        self.assertEqual(exit_code, 0)
        self.assertEqual(diagnostics["metadata"]["station_layout_json"], str(layout_json))
        self.assertIn("-0.75,0.75", route_text)

    def test_dry_run_cli_writes_overlay_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "map.pgm").write_text("P2\n7 7\n255\n" + " ".join(["255"] * 49) + "\n")
            (root / "map.yaml").write_text(
                "\n".join(
                    [
                        "image: map.pgm",
                        "resolution: 0.5",
                        "origin: [-1.5, -1.5, 0.0]",
                        "negate: 0",
                        "occupied_thresh: 0.65",
                        "free_thresh: 0.20",
                        "mode: trinary",
                    ]
                )
                + "\n"
            )
            route_csv = root / "route.csv"
            diagnostics_json = root / "diagnostics.json"
            overlay_svg = root / "overlay.svg"
            overlay_metadata = root / "overlay_metadata.json"

            exit_code = run_station_route_main(
                [
                    "--map",
                    str(root / "map.yaml"),
                    "--stations",
                    "B",
                    "--start-x",
                    "-0.5",
                    "--start-y",
                    "-0.5",
                    "--route-csv",
                    str(route_csv),
                    "--diagnostics-json",
                    str(diagnostics_json),
                    "--overlay-svg",
                    str(overlay_svg),
                    "--overlay-metadata-json",
                    str(overlay_metadata),
                ]
            )

            svg = overlay_svg.read_text()
            metadata_payload = json.loads(overlay_metadata.read_text())

        self.assertEqual(exit_code, 0)
        self.assertIn('id="coordinate-frame-label"', svg)
        self.assertIn("frame_id: map", svg)
        self.assertEqual(metadata_payload["frame_id"], "map")

    def test_failed_overlay_is_gated_unless_explicitly_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pixels = [
                255, 255, 0, 255, 255,
                255, 255, 0, 255, 255,
                255, 255, 0, 255, 255,
            ]
            (root / "map.pgm").write_text("P2\n5 3\n255\n" + " ".join(str(value) for value in pixels) + "\n")
            (root / "map.yaml").write_text(
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
            layout_json = root / "layout.json"
            write_station_layout_json(
                layout_json,
                [Station("B", StationPose(4.5, 1.5, 0.0), 0.0, 0.0)],
                {"case": "blocked"},
            )
            blocked_overlay = root / "blocked.svg"
            allowed_overlay = root / "allowed.svg"
            common_args = [
                "--map",
                str(root / "map.yaml"),
                "--stations",
                "B",
                "--station-layout-json",
                str(layout_json),
                "--start-x",
                "0.5",
                "--start-y",
                "1.5",
                "--arena-length-m",
                "5.0",
                "--arena-width-m",
                "3.0",
                "--arena-center-x-m",
                "2.5",
                "--arena-center-y-m",
                "1.5",
                "--route-csv",
                str(root / "route.csv"),
                "--diagnostics-json",
                str(root / "diagnostics.json"),
            ]

            blocked_exit = run_station_route_main(common_args + ["--overlay-svg", str(blocked_overlay)])
            allowed_exit = run_station_route_main(
                common_args + ["--overlay-svg", str(allowed_overlay), "--allow-failed-overlay"]
            )
            allowed_text = allowed_overlay.read_text()

        self.assertEqual(blocked_exit, 1)
        self.assertFalse(blocked_overlay.exists())
        self.assertEqual(allowed_exit, 1)
        self.assertIn("FAILED/INCOMPLETE", allowed_text)

    def test_navigation_does_not_import_aufgabe03(self):
        navigation_dir = ROOT / "scripts" / "aufgabe04" / "navigation"
        offenders = []
        for path in navigation_dir.glob("*.py"):
            text = path.read_text()
            if "scripts.aufgabe03" in text or "map_path_planner" in text:
                offenders.append(path.name)

        self.assertEqual(offenders, [])

    def test_route_overlay_keeps_offline_import_boundary(self):
        text = (ROOT / "scripts" / "aufgabe04" / "navigation" / "route_overlay.py").read_text()
        banned = ["import rclpy", "nav_msgs", "geometry_msgs", "matplotlib", "PIL"]
        offenders = [pattern for pattern in banned if pattern in text]

        self.assertEqual(offenders, [])

    def test_station_modules_do_not_import_navigation(self):
        station_dir = ROOT / "scripts" / "aufgabe04" / "stations"
        offenders = []
        for path in station_dir.glob("*.py"):
            text = path.read_text()
            if "scripts.aufgabe04.navigation" in text:
                offenders.append(path.name)

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
