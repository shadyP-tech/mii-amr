import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.planning.generate_random_station_layout import (  # noqa: E402
    build_parser as build_generate_parser,
    main as generate_layout_main,
)
from scripts.aufgabe04.navigation.foundation.arena_bounds import (  # noqa: E402
    DEFAULT_ARENA_LENGTH_M,
    DEFAULT_ARENA_WIDTH_M,
    ArenaBounds,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D  # noqa: E402
from scripts.aufgabe04.navigation.planning.random_station_layout import (  # noqa: E402
    RandomStationLayoutConfig,
    generate_random_station_layout,
    station_ids_from_count,
)
from scripts.aufgabe04.stations.station_layout_io import station_layout_payload  # noqa: E402


def write_map(root: Path, rows, resolution=0.1, origin=(0.0, 0.0, 0.0)) -> Path:
    height = len(rows)
    width = len(rows[0])
    # rows are passed in grid order, but PGM stores top image row first.
    image_rows = list(reversed(rows))
    values = [str(value) for row in image_rows for value in row]
    (root / "map.pgm").write_text(f"P2\n{width} {height}\n255\n" + " ".join(values) + "\n")
    (root / "map.yaml").write_text(
        "\n".join(
            [
                "image: map.pgm",
                f"resolution: {resolution}",
                f"origin: [{origin[0]}, {origin[1]}, {origin[2]}]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.20",
                "mode: trinary",
            ]
        )
        + "\n"
    )
    return root / "map.yaml"


def free_rows(width, height):
    return [[255 for _ in range(width)] for _ in range(height)]


def rows_from_free_cells(width, height, free_cells):
    free = set(free_cells)
    return [
        [255 if (x, y) in free else 0 for x in range(width)]
        for y in range(height)
    ]


class RandomStationLayoutTest(unittest.TestCase):
    def test_generator_defaults_use_layouts_subfolder(self):
        parser = build_generate_parser()
        args = parser.parse_args(["--station-count", "1", "--seed", "42"])

        self.assertEqual(
            args.output_json,
            Path("results/aufgabe04/layouts/random_station_layout.json"),
        )
        self.assertIsNone(args.output_csv)
        self.assertEqual(args.arena_length_m, DEFAULT_ARENA_LENGTH_M)
        self.assertEqual(args.arena_width_m, DEFAULT_ARENA_WIDTH_M)
        self.assertEqual(args.arena_center_x, 0.0)
        self.assertEqual(args.arena_center_y, 0.0)
        self.assertEqual(args.arena_yaw_deg, 0.0)
        self.assertEqual(args.arena_margin_m, 0.0)

    def test_station_ids_from_count(self):
        self.assertEqual(station_ids_from_count(3), ("A", "B", "C"))
        with self.assertRaisesRegex(ValueError, "positive"):
            station_ids_from_count(0)

    def test_same_seed_produces_same_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_map(Path(tmpdir), free_rows(20, 20))
            config = RandomStationLayoutConfig(
                map_yaml=map_yaml,
                station_ids=("A", "B", "C"),
                seed=42,
                clearance_radius_m=0.0,
                min_station_distance_m=0.2,
                approach_offset_m=0.35,
                keepout_radius_m=0.1,
            )

            first = generate_random_station_layout(config)
            second = generate_random_station_layout(config)

        self.assertEqual(
            station_layout_payload(first.stations.values(), first.metadata),
            station_layout_payload(second.stations.values(), second.metadata),
        )

    def test_different_seed_changes_open_map_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_map(Path(tmpdir), free_rows(20, 20))
            base = dict(
                map_yaml=map_yaml,
                station_ids=("A", "B", "C"),
                clearance_radius_m=0.0,
                min_station_distance_m=0.2,
                approach_offset_m=0.35,
                keepout_radius_m=0.1,
            )

            first = generate_random_station_layout(RandomStationLayoutConfig(seed=1, **base))
            second = generate_random_station_layout(RandomStationLayoutConfig(seed=2, **base))

        first_poses = [station.pose for station in first.stations.values()]
        second_poses = [station.pose for station in second.stations.values()]
        self.assertNotEqual(first_poses, second_poses)

    def test_rejects_occupied_unknown_and_uses_only_free_cell(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_map(
                Path(tmpdir),
                [
                    [0, 128, 0, 128, 0, 128, 0, 128, 0, 128],
                    [128, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                    [0, 255, 255, 255, 255, 255, 255, 255, 255, 128],
                    [128, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                    [0, 255, 255, 255, 255, 255, 255, 255, 255, 128],
                    [128, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                    [0, 255, 255, 255, 255, 255, 255, 255, 255, 128],
                    [128, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                    [0, 255, 255, 255, 255, 255, 255, 255, 255, 128],
                    [128, 0, 128, 0, 128, 0, 128, 0, 128, 0],
                ],
                resolution=0.1,
            )

            result = generate_random_station_layout(
                RandomStationLayoutConfig(
                    map_yaml=map_yaml,
                    station_ids=("A",),
                    seed=7,
                    clearance_radius_m=0.0,
                    min_station_distance_m=0.0,
                    approach_offset_m=0.35,
                    keepout_radius_m=0.1,
                    yaw_mode="random",
                    max_attempts=100,
                )
            )

        station = result.stations["A"]
        self.assertGreaterEqual(station.pose.x_m, 0.1)
        self.assertLess(station.pose.x_m, 0.9)
        self.assertGreaterEqual(station.pose.y_m, 0.1)
        self.assertLess(station.pose.y_m, 0.9)

    def test_default_arena_bounds_are_recorded_and_enforced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_map(
                Path(tmpdir),
                free_rows(80, 80),
                resolution=0.05,
                origin=(-2.0, -2.0, 0.0),
            )

            result = generate_random_station_layout(
                RandomStationLayoutConfig(
                    map_yaml=map_yaml,
                    station_ids=("A", "B", "C"),
                    seed=11,
                    clearance_radius_m=0.0,
                    min_station_distance_m=0.2,
                    approach_offset_m=0.35,
                    keepout_radius_m=0.1,
                    max_attempts=1000,
                )
            )

        bounds = ArenaBounds()
        self.assertEqual(result.metadata["arena_bounds"]["length_m"], 3.90)
        self.assertEqual(result.metadata["arena_bounds"]["width_m"], 1.898)
        for station in result.stations.values():
            self.assertTrue(bounds.contains(Pose2D(station.pose.x_m, station.pose.y_m)))

    def test_rejects_map_free_cells_outside_default_arena_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Free cells are map-traversable but outside x <= 1.95 m.
            map_yaml = write_map(
                Path(tmpdir),
                rows_from_free_cells(
                    60,
                    60,
                    {(55, y) for y in range(25, 35)},
                ),
                resolution=0.05,
                origin=(-0.5, -1.5, 0.0),
            )

            with self.assertRaisesRegex(ValueError, "could not generate"):
                generate_random_station_layout(
                    RandomStationLayoutConfig(
                        map_yaml=map_yaml,
                        station_ids=("A",),
                        seed=9,
                        clearance_radius_m=0.0,
                        min_station_distance_m=0.0,
                        approach_offset_m=0.35,
                        keepout_radius_m=0.1,
                        yaw_mode="random",
                        max_attempts=100,
                    )
                )

    def test_enforces_start_distance_and_fails_when_impossible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_map(Path(tmpdir), free_rows(5, 5))

            with self.assertRaisesRegex(ValueError, "could not generate"):
                generate_random_station_layout(
                    RandomStationLayoutConfig(
                        map_yaml=map_yaml,
                        station_ids=("A",),
                        seed=3,
                        clearance_radius_m=0.0,
                        min_station_distance_m=0.0,
                        start=Pose2D(0.25, 0.25, 0.0),
                        min_start_distance_m=10.0,
                        approach_offset_m=0.35,
                        keepout_radius_m=0.1,
                        max_attempts=50,
                    )
                )

    def test_rejects_approach_offset_inside_keepout_margin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_map(Path(tmpdir), free_rows(10, 10), resolution=0.1)

            with self.assertRaisesRegex(ValueError, "approach offset"):
                generate_random_station_layout(
                    RandomStationLayoutConfig(
                        map_yaml=map_yaml,
                        station_ids=("A",),
                        seed=4,
                        approach_offset_m=0.30,
                        keepout_radius_m=0.20,
                    )
                )

    def test_final_keepout_clearance_validation_can_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_map(Path(tmpdir), free_rows(10, 10), resolution=0.1)

            with self.assertRaisesRegex(ValueError, "could not generate"):
                generate_random_station_layout(
                    RandomStationLayoutConfig(
                        map_yaml=map_yaml,
                        station_ids=("A",),
                        seed=5,
                        clearance_radius_m=0.40,
                        min_station_distance_m=0.0,
                        approach_offset_m=0.31,
                        keepout_radius_m=0.20,
                        max_attempts=50,
                    )
                )

    def test_cli_requires_seed_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_map(root, free_rows(20, 20), resolution=0.1)
            output_json = root / "layout.json"
            output_csv = root / "layout.csv"

            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as missing_seed:
                    generate_layout_main(["--map", str(map_yaml), "--station-count", "1"])

            exit_code = generate_layout_main(
                [
                    "--map",
                    str(map_yaml),
                    "--station-count",
                    "1",
                    "--seed",
                    "12",
                    "--clearance-radius-m",
                    "0.0",
                    "--approach-offset-m",
                    "0.35",
                    "--keepout-radius-m",
                    "0.1",
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ]
            )
            payload = json.loads(output_json.read_text())
            csv_exists = output_csv.exists()

        self.assertEqual(missing_seed.exception.code, 2)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["metadata"]["seed"], 12)
        self.assertTrue(csv_exists)


if __name__ == "__main__":
    unittest.main()
