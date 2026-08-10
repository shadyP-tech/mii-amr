from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from itertools import combinations
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.map_io import (
    CELL_FREE,
    MapMetadata,
    OccupancyGrid,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.plan_stand_coverage_survey import (
    main as plan_coverage_main,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    CoverageSurveyConfig,
    build_coverage_survey_plan,
    load_coverage_survey_plan,
    write_coverage_survey_plan,
)


MAP_HASH = "a" * 64
ARENA = ArenaBounds(length_m=4.0, width_m=2.0)
START = Pose2D(-1.5, 0.0, 0.0)


def free_grid() -> OccupancyGrid:
    width = 50
    height = 30
    return OccupancyGrid(
        metadata=MapMetadata(
            yaml_path=Path("map.yaml"),
            image_path=Path("map.pgm"),
            resolution=0.10,
            origin=(-2.5, -1.5, 0.0),
            negate=0,
            occupied_thresh=0.65,
            free_thresh=0.20,
            mode="trinary",
        ),
        width=width,
        height=height,
        cells=tuple(tuple([CELL_FREE] * width) for _ in range(height)),
    )


def write_free_map(root: Path) -> Path:
    width = 50
    height = 30
    (root / "map.pgm").write_text(
        f"P2\n{width} {height}\n255\n"
        + " ".join(["255"] * width * height)
        + "\n"
    )
    (root / "map.yaml").write_text(
        "\n".join(
            (
                "image: map.pgm",
                "resolution: 0.1",
                "origin: [-2.5, -1.5, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.20",
                "mode: trinary",
            )
        )
        + "\n"
    )
    return root / "map.yaml"


def build(config: CoverageSurveyConfig):
    return build_coverage_survey_plan(
        free_grid(),
        map_bundle_sha256=MAP_HASH,
        start=START,
        survey_id="exact_two_test",
        arena_bounds=ARENA,
        config=config,
    )


class ExactTwoCoveragePlanTest(unittest.TestCase):
    def test_exact_two_maximizes_union_then_shared_visibility(self):
        dense = build(CoverageSurveyConfig(lane_count=1))
        exact = build(
            CoverageSurveyConfig(
                lane_count=1,
                exact_inspection_point_count=2,
            )
        )
        dense_pairs = []
        for (first_index, first), (second_index, second) in combinations(
            enumerate(dense.viewpoints), 2
        ):
            first_visible = set(first.visible_cells)
            second_visible = set(second.visible_cells)
            shared_count = len(first_visible.intersection(second_visible))
            if first.cell == second.cell or shared_count == 0:
                continue
            union_count = len(first_visible.union(second_visible))
            dense_pairs.append(
                (
                    (-union_count, -shared_count, first_index, second_index),
                    (first.cell, second.cell),
                )
            )

        expected_pair = min(dense_pairs)[1]
        selected_pair = tuple(viewpoint.cell for viewpoint in exact.viewpoints)
        shared = set(exact.viewpoints[0].visible_cells).intersection(
            exact.viewpoints[1].visible_cells
        )

        self.assertEqual(len(exact.viewpoints), 2)
        self.assertEqual(selected_pair, expected_pair)
        self.assertEqual(len(set(selected_pair)), 2)
        self.assertTrue(shared)
        self.assertGreaterEqual(
            exact.planned_coverage_ratio,
            exact.config.coverage_threshold,
        )

    def test_dense_index_order_is_the_deterministic_final_tie_break(self):
        dense = build(
            CoverageSurveyConfig(lane_count=1, visibility_radius_m=10.0)
        )
        exact = build(
            CoverageSurveyConfig(
                lane_count=1,
                exact_inspection_point_count=2,
                visibility_radius_m=10.0,
            )
        )

        self.assertEqual(
            tuple(viewpoint.cell for viewpoint in exact.viewpoints),
            tuple(viewpoint.cell for viewpoint in dense.viewpoints[:2]),
        )

    def test_selector_rejects_malformed_or_unsupported_counts(self):
        for malformed in (True, 0, 1, 3, 2.0, "2"):
            with self.subTest(malformed=malformed):
                with self.assertRaisesRegex(
                    ValueError,
                    "exact_inspection_point_count must be exactly 2",
                ):
                    CoverageSurveyConfig(
                        lane_count=1,
                        exact_inspection_point_count=malformed,
                    ).validated()

    def test_selector_requires_the_centerline_dense_sequence(self):
        with self.assertRaisesRegex(ValueError, "requires lane_count=1"):
            CoverageSurveyConfig(
                lane_count=2,
                exact_inspection_point_count=2,
            ).validated()

    def test_no_distinct_pair_with_shared_visibility_fails_closed(self):
        with self.assertRaisesRegex(
            ValueError,
            "no valid exact-two inspection-point pair",
        ):
            build(
                CoverageSurveyConfig(
                    lane_count=1,
                    exact_inspection_point_count=2,
                    visibility_radius_m=0.04,
                    coverage_threshold=0.001,
                )
            )

    def test_selected_pair_still_has_to_pass_coverage_threshold(self):
        with self.assertRaisesRegex(ValueError, "coverage threshold"):
            build(
                CoverageSurveyConfig(
                    lane_count=1,
                    exact_inspection_point_count=2,
                    visibility_radius_m=1.0,
                    coverage_threshold=0.95,
                )
            )

    def test_selector_round_trips_and_unset_payload_remains_legacy_compatible(self):
        exact = build(
            CoverageSurveyConfig(
                lane_count=1,
                exact_inspection_point_count=2,
            )
        )
        dense = build(CoverageSurveyConfig(lane_count=1))
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exact_path = root / "exact.json"
            dense_path = root / "dense.json"
            write_coverage_survey_plan(exact_path, exact)
            write_coverage_survey_plan(dense_path, dense)
            exact_payload = json.loads(exact_path.read_text())
            dense_payload = json.loads(dense_path.read_text())
            loaded_exact = load_coverage_survey_plan(exact_path)
            loaded_dense = load_coverage_survey_plan(dense_path)

        self.assertEqual(
            exact_payload["config"]["exact_inspection_point_count"], 2
        )
        self.assertNotIn(
            "exact_inspection_point_count",
            dense_payload["config"],
        )
        self.assertEqual(loaded_exact, exact)
        self.assertIsNone(loaded_dense.config.exact_inspection_point_count)

    def test_persisted_string_count_is_not_coerced_to_an_integer(self):
        exact = build(
            CoverageSurveyConfig(
                lane_count=1,
                exact_inspection_point_count=2,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "malformed.json"
            write_coverage_survey_plan(path, exact)
            payload = json.loads(path.read_text())
            payload.pop("plan_sha256")
            payload["config"]["exact_inspection_point_count"] = "2"
            payload["plan_sha256"] = payload_sha256(payload)
            path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(
                ValueError,
                "exact_inspection_point_count must be exactly 2",
            ):
                load_coverage_survey_plan(path)

    def test_cli_persists_exact_inspection_point_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_path = write_free_map(root)
            output_dir = root / "survey"
            with redirect_stdout(StringIO()):
                status = plan_coverage_main(
                    [
                        "--map",
                        str(map_path),
                        "--start-x",
                        str(START.x_m),
                        "--start-y",
                        str(START.y_m),
                        "--survey-id",
                        "exact_two_cli",
                        "--output-dir",
                        str(output_dir),
                        "--lane-count",
                        "1",
                        "--exact-inspection-point-count",
                        "2",
                        "--arena-length-m",
                        str(ARENA.length_m),
                        "--arena-width-m",
                        str(ARENA.width_m),
                    ]
                )
            loaded = load_coverage_survey_plan(
                output_dir / "coverage_plan.json"
            )

        self.assertEqual(status, 0)
        self.assertEqual(loaded.config.exact_inspection_point_count, 2)
        self.assertEqual(len(loaded.viewpoints), 2)


if __name__ == "__main__":
    unittest.main()
