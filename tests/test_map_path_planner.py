import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import map_path_planner as planner  # noqa: E402


def metadata(**overrides):
    values = {
        "yaml_path": Path("test.yaml"),
        "image_path": Path("test.pgm"),
        "resolution": 0.5,
        "origin": (-1.0, -2.0, 0.0),
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
        "mode": "trinary",
    }
    values.update(overrides)
    return planner.MapMetadata(**values)


def occupancy_map(cells, resolution=1.0):
    height = len(cells)
    width = len(cells[0]) if height else 0
    return planner.OccupancyMap(
        metadata=metadata(resolution=resolution, origin=(0.0, 0.0, 0.0)),
        width=width,
        height=height,
        cells=cells,
    )


class MapPathPlannerTest(unittest.TestCase):
    def test_yaml_parsing_defaults_missing_mode_to_trinary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "map.yaml"
            yaml_path.write_text(
                "\n".join([
                    "image: map.pgm",
                    "resolution: 0.05",
                    "origin: [-2.82, -1.69, 0]",
                    "negate: 0",
                    "occupied_thresh: 0.65",
                    "free_thresh: 0.25",
                ])
            )

            parsed = planner.read_map_metadata(yaml_path)

        self.assertEqual(parsed.mode, "trinary")
        self.assertEqual(parsed.image_path, root / "map.pgm")
        self.assertAlmostEqual(parsed.resolution, 0.05)
        self.assertEqual(parsed.origin, (-2.82, -1.69, 0.0))

    def test_yaml_parsing_rejects_unsupported_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "map.yaml"
            yaml_path.write_text(
                "\n".join([
                    "image: map.pgm",
                    "mode: scale",
                    "resolution: 0.05",
                    "origin: [0, 0, 0]",
                    "negate: 0",
                    "occupied_thresh: 0.65",
                    "free_thresh: 0.25",
                ])
            )

            with self.assertRaisesRegex(ValueError, "Only trinary maps"):
                planner.read_map_metadata(yaml_path)

    def test_pgm_p2_parsing_with_comments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pgm_path = Path(tmpdir) / "map.pgm"
            pgm_path.write_text(
                "P2\n# comment\n3 2\n255\n0 127 255\n10 20 30\n"
            )

            image = planner.read_pgm(pgm_path)

        self.assertEqual(image.width, 3)
        self.assertEqual(image.height, 2)
        self.assertEqual(image.maxval, 255)
        self.assertEqual(image.pixels, [[0, 127, 255], [10, 20, 30]])

    def test_pgm_p5_parsing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pgm_path = Path(tmpdir) / "map.pgm"
            pgm_path.write_bytes(b"P5\n2 2\n255\n\x00\x7f\x80\xff")

            image = planner.read_pgm(pgm_path)

        self.assertEqual(image.width, 2)
        self.assertEqual(image.height, 2)
        self.assertEqual(image.pixels, [[0, 127], [128, 255]])

    def test_inclusive_occupancy_thresholds_negate_zero(self):
        meta = metadata(
            occupied_thresh=0.65,
            free_thresh=0.25,
            negate=0,
        )

        self.assertEqual(planner.pixel_to_cell(35, meta, maxval=100), planner.CELL_OCCUPIED)
        self.assertEqual(planner.pixel_to_cell(75, meta, maxval=100), planner.CELL_FREE)
        self.assertEqual(planner.pixel_to_cell(50, meta, maxval=100), planner.CELL_UNKNOWN)

    def test_inclusive_occupancy_thresholds_negate_one(self):
        meta = metadata(
            occupied_thresh=0.65,
            free_thresh=0.25,
            negate=1,
        )

        self.assertEqual(planner.pixel_to_cell(65, meta, maxval=100), planner.CELL_OCCUPIED)
        self.assertEqual(planner.pixel_to_cell(25, meta, maxval=100), planner.CELL_FREE)
        self.assertEqual(planner.pixel_to_cell(50, meta, maxval=100), planner.CELL_UNKNOWN)

    def test_image_grid_world_conversion_uses_y_axis_flip(self):
        meta = metadata(resolution=0.5, origin=(-1.0, -2.0, 0.0))

        self.assertEqual(planner.grid_to_image(2, 1, height=4), (2, 2))
        self.assertEqual(planner.image_to_grid(2, 2, height=4), (2, 1))
        self.assertEqual(planner.world_to_grid(0.25, -1.25, meta), (2, 1))
        self.assertEqual(planner.grid_to_world(2, 1, meta), (0.25, -1.25))

    def test_world_bounds_calculation(self):
        occ = occupancy_map(
            [
                [planner.CELL_FREE] * 3,
                [planner.CELL_FREE] * 3,
            ],
            resolution=0.5,
        )
        occ = planner.OccupancyMap(
            metadata=metadata(resolution=0.5, origin=(-1.0, -2.0, 0.0)),
            width=3,
            height=2,
            cells=occ.cells,
        )

        self.assertEqual(planner.world_bounds(occ), (-1.0, 0.5, -2.0, -1.0))

    def test_unknown_cells_are_blocked_by_default(self):
        occ = occupancy_map([
            [planner.CELL_FREE, planner.CELL_UNKNOWN],
        ])

        blocked = planner.base_blocked_cells(occ, block_unknown=True)

        self.assertIn((1, 0), blocked)
        self.assertNotIn((0, 0), blocked)

    def test_obstacle_inflation_radius_in_cells(self):
        occ = occupancy_map(
            [
                [planner.CELL_FREE] * 5,
                [planner.CELL_FREE] * 5,
                [planner.CELL_FREE, planner.CELL_FREE, planner.CELL_OCCUPIED, planner.CELL_FREE, planner.CELL_FREE],
                [planner.CELL_FREE] * 5,
                [planner.CELL_FREE] * 5,
            ],
            resolution=0.1,
        )

        blocked, inflation_cells = planner.inflate_blocked_cells(
            occ,
            inflate_radius_m=0.15,
            block_unknown=True,
        )

        self.assertEqual(inflation_cells, 2)
        self.assertIn((2, 2), blocked)
        self.assertIn((2, 4), blocked)
        self.assertIn((4, 2), blocked)
        self.assertNotIn((4, 4), blocked)

    def test_astar_finds_path_in_open_grid(self):
        occ = occupancy_map(
            [
                [planner.CELL_FREE] * 3,
                [planner.CELL_FREE] * 3,
                [planner.CELL_FREE] * 3,
            ],
        )

        path = planner.astar(occ, blocked=set(), start=(0, 0), goal=(2, 2))

        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 2))

    def test_astar_avoids_obstacles(self):
        occ = occupancy_map(
            [
                [planner.CELL_FREE, planner.CELL_FREE, planner.CELL_FREE],
                [planner.CELL_FREE, planner.CELL_OCCUPIED, planner.CELL_FREE],
                [planner.CELL_FREE, planner.CELL_FREE, planner.CELL_FREE],
            ],
        )
        blocked = planner.base_blocked_cells(occ)

        path = planner.astar(occ, blocked, start=(0, 1), goal=(2, 1))

        self.assertNotIn((1, 1), path)
        self.assertEqual(path[0], (0, 1))
        self.assertEqual(path[-1], (2, 1))

    def test_astar_prevents_diagonal_corner_cutting(self):
        occ = occupancy_map(
            [
                [planner.CELL_FREE, planner.CELL_OCCUPIED],
                [planner.CELL_OCCUPIED, planner.CELL_FREE],
            ],
        )
        blocked = planner.base_blocked_cells(occ)

        with self.assertRaisesRegex(ValueError, "No path exists"):
            planner.astar(occ, blocked, start=(0, 0), goal=(1, 1))

    def test_astar_unreachable_goal_fails_cleanly(self):
        occ = occupancy_map(
            [
                [planner.CELL_FREE, planner.CELL_OCCUPIED, planner.CELL_FREE],
            ],
        )
        blocked = planner.base_blocked_cells(occ)

        with self.assertRaisesRegex(ValueError, "No path exists"):
            planner.astar(occ, blocked, start=(0, 0), goal=(2, 0))

    def test_start_and_goal_snapping_within_radius(self):
        occ = occupancy_map(
            [
                [planner.CELL_OCCUPIED, planner.CELL_FREE, planner.CELL_FREE],
            ],
            resolution=0.1,
        )
        blocked = planner.base_blocked_cells(occ)

        snapped = planner.snap_to_traversable(
            occ,
            blocked,
            requested_cell=(0, 0),
            snap_radius_m=0.11,
        )

        self.assertEqual(snapped, (1, 0))

    def test_path_simplification_removes_same_direction_intermediate_cells(self):
        path = [(1, 1), (2, 1), (3, 1), (4, 2), (5, 3)]

        self.assertEqual(
            planner.simplify_path(path),
            [(1, 1), (3, 1), (5, 3)],
        )

    def test_csv_row_generation_and_cumulative_distance(self):
        meta = metadata(resolution=0.5, origin=(0.0, 0.0, 0.0))
        rows = planner.build_path_rows([(0, 0), (1, 0), (2, 1)], meta)

        self.assertEqual(rows[0][0], 0)
        self.assertEqual(rows[0][1:3], [0, 0])
        self.assertAlmostEqual(rows[1][-2], 0.5)
        self.assertAlmostEqual(rows[2][-1], 0.5 + math.sqrt(2) * 0.5)

    def test_ppm_writer_produces_valid_p6_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plot.ppm"
            planner.write_ppm(path, [[(255, 0, 0), (0, 255, 0)]])

            data = path.read_bytes()

        self.assertTrue(data.startswith(b"P6\n2 1\n255\n"))


if __name__ == "__main__":
    unittest.main()
