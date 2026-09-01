from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from itertools import combinations
import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    payload_sha256,
)
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.planning.map_io import (
    CELL_FREE,
    MapMetadata,
    OccupancyGrid,
    load_occupancy_grid_with_bundle,
)
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.coverage.exact_two_viewpoint_selection import (
    DEFAULT_EXACT_TWO_CANDIDATE_SPACING_M,
    DEFAULT_MINIMUM_EXACT_TWO_VIEWPOINT_BASELINE_M,
    ExactTwoViewpointCandidate,
    select_exact_two_viewpoint_cells,
)
from scripts.aufgabe04.navigation.missions.plan_stand_coverage_survey import (
    main as plan_coverage_main,
)
from scripts.aufgabe04.navigation.missions.startup_route_uncertainty_selection import (
    STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyConfig,
    build_coverage_survey_plan,
    load_coverage_survey_plan,
    write_coverage_survey_plan,
)
from scripts.aufgabe04.navigation.coverage.stand_discovery_route import (
    DEFAULT_TRACKING_TUBE_RADIUS_M,
    LONGITUDINALLY_DIVERSE_STAND_DISCOVERY_ROUTE_SOURCE,
    STAND_DISCOVERY_ROUTE_SOURCE,
    seal_stand_discovery_route,
    validate_stand_discovery_route_binding,
)
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg


MAP_HASH = "a" * 64
ARENA = ArenaBounds(length_m=4.0, width_m=2.0)
START = Pose2D(-1.5, 0.0, 0.0)
CANONICAL_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
AUDITED_STAND_XY = (-1.05464, -0.503014)


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


def exact_config(**changes: object) -> CoverageSurveyConfig:
    return replace(
        CoverageSurveyConfig(
            lane_count=1,
            exact_inspection_point_count=2,
            exact_two_candidate_spacing_m=(
                DEFAULT_EXACT_TWO_CANDIDATE_SPACING_M
            ),
            minimum_exact_two_viewpoint_baseline_m=(
                DEFAULT_MINIMUM_EXACT_TWO_VIEWPOINT_BASELINE_M
            ),
        ),
        **changes,
    )


def legacy_exact_plan():
    """Construct the shape emitted by the historical lane-one selector."""

    dense = build(CoverageSurveyConfig(lane_count=1))
    options = []
    for (first_index, first), (second_index, second) in combinations(
        enumerate(dense.viewpoints), 2
    ):
        first_visible = set(first.visible_cells)
        second_visible = set(second.visible_cells)
        shared_count = len(first_visible.intersection(second_visible))
        if first.cell == second.cell or shared_count == 0:
            continue
        union_count = len(first_visible.union(second_visible))
        options.append(
            (
                (-union_count, -shared_count, first_index, second_index),
                (first, second),
            )
        )
    viewpoints = min(options)[1]
    planned_covered = tuple(
        sorted(
            set(viewpoints[0].visible_cells).union(viewpoints[1].visible_cells)
        )
    )
    return replace(
        dense,
        config=CoverageSurveyConfig(
            lane_count=1,
            exact_inspection_point_count=2,
        ),
        viewpoints=viewpoints,
        planned_covered_cells=planned_covered,
        planned_coverage_ratio=len(planned_covered) / len(dense.surveyable_cells),
    )


class ExactTwoCoveragePlanTest(unittest.TestCase):
    def test_canonical_map_selects_safe_longitudinal_pair(self):
        grid, bundle = load_occupancy_grid_with_bundle(
            CANONICAL_MAP,
            semantic_map_id="arena_1p898x3p9_auto",
            planning_frame="map",
        )
        plan = build_coverage_survey_plan(
            grid,
            map_bundle_sha256=bundle.bundle_sha256,
            start=Pose2D(-1.644, -0.670, 0.011),
            survey_id="canonical_longitudinal_exact_two",
            arena_bounds=ArenaBounds(),
            config=exact_config(stop_spacing_m=0.70),
        )

        self.assertEqual(len(plan.viewpoints), 2)
        first, second = plan.viewpoints
        self.assertAlmostEqual(first.pose.x_m, -0.795, places=3)
        self.assertAlmostEqual(first.pose.y_m, -0.015, places=3)
        self.assertAlmostEqual(second.pose.x_m, 0.405, places=3)
        self.assertAlmostEqual(second.pose.y_m, -0.015, places=3)
        self.assertAlmostEqual(plan.planned_coverage_ratio, 0.9531, places=4)
        self.assertAlmostEqual(plan.config.stop_spacing_m, 0.70)
        self.assertAlmostEqual(plan.config.exact_two_candidate_spacing_m, 0.40)

        baseline_m = math.hypot(
            second.pose.x_m - first.pose.x_m,
            second.pose.y_m - first.pose.y_m,
        )
        self.assertGreaterEqual(baseline_m, 1.0)
        self.assertLessEqual(max(abs(first.pose.y_m), abs(second.pose.y_m)), 0.02)
        audited_clearances = tuple(
            math.hypot(
                viewpoint.pose.x_m - AUDITED_STAND_XY[0],
                viewpoint.pose.y_m - AUDITED_STAND_XY[1],
            )
            for viewpoint in plan.viewpoints
        )
        required_clearance_m = (
            plan.config.candidate_keepout_radius_m
            + DEFAULT_TRACKING_TUBE_RADIUS_M
        )
        self.assertGreater(min(audited_clearances), required_clearance_m)
        self.assertGreater(
            min(audited_clearances) - required_clearance_m,
            0.20,
        )

    def test_physical_stop_spacing_is_separate_from_candidate_sampling(self):
        first = build(exact_config(stop_spacing_m=0.70))
        second = build(exact_config(stop_spacing_m=0.25))

        self.assertEqual(
            tuple(viewpoint.cell for viewpoint in first.viewpoints),
            tuple(viewpoint.cell for viewpoint in second.viewpoints),
        )
        self.assertEqual(first.config.stop_spacing_m, 0.70)
        self.assertEqual(second.config.stop_spacing_m, 0.25)
        self.assertEqual(first.config.exact_two_candidate_spacing_m, 0.40)
        self.assertEqual(second.config.exact_two_candidate_spacing_m, 0.40)

    def test_exact_two_remains_center_corridor_and_meets_hard_gates(self):
        exact = build(exact_config())
        first, second = exact.viewpoints
        selected_pair = (first.cell, second.cell)
        shared = set(first.visible_cells).intersection(second.visible_cells)
        baseline_m = math.hypot(
            second.pose.x_m - first.pose.x_m,
            second.pose.y_m - first.pose.y_m,
        )

        self.assertEqual(len(set(selected_pair)), 2)
        self.assertTrue(shared)
        self.assertGreaterEqual(baseline_m, 1.0)
        self.assertAlmostEqual(first.pose.y_m, second.pose.y_m)
        self.assertGreaterEqual(
            exact.planned_coverage_ratio,
            exact.config.coverage_threshold,
        )

    def test_bearing_diversity_precedes_union_count(self):
        cells = tuple(GridCell(index, 0) for index in range(6))
        target_cells = cells[2:]
        world = {
            target_cells[0]: (0.0, 0.0),
            target_cells[1]: (0.0, 1.0),
            target_cells[2]: (0.0, -1.0),
            target_cells[3]: (1.0, 0.0),
        }
        # Pair 0->1 covers all targets but has weaker mean incidence diversity.
        # Pair 0->2 covers only the hard-threshold set but has better bearings.
        candidates = (
            ExactTwoViewpointCandidate(cells[0], -1.0, -0.5, target_cells),
            ExactTwoViewpointCandidate(cells[1], 1.0, 0.5, target_cells),
            ExactTwoViewpointCandidate(cells[5], 0.0, 0.5, target_cells[:3]),
        )

        self.assertEqual(
            select_exact_two_viewpoint_cells(
                candidates,
                surveyable_world_xy=world,
                coverage_threshold=0.75,
                minimum_viewpoint_baseline_m=0.5,
                start_x_m=-1.0,
                start_y_m=-0.5,
            ),
            (cells[0], cells[5]),
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

    def test_exact_two_requires_one_center_lane(self):
        with self.assertRaisesRegex(ValueError, "requires lane_count=1"):
            exact_config(lane_count=2).validated()

    def test_new_exact_two_requires_both_explicit_geometry_fields(self):
        with self.assertRaisesRegex(ValueError, "may be loaded or resumed"):
            build(
                CoverageSurveyConfig(
                    lane_count=1,
                    exact_inspection_point_count=2,
                )
            )
        for config in (
            CoverageSurveyConfig(
                lane_count=1,
                exact_inspection_point_count=2,
                exact_two_candidate_spacing_m=0.40,
            ),
            CoverageSurveyConfig(
                lane_count=1,
                exact_inspection_point_count=2,
                minimum_exact_two_viewpoint_baseline_m=1.0,
            ),
        ):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "must either both be set"):
                    config.validated()

    def test_exact_two_geometry_fields_are_finite_and_positive(self):
        field_names = (
            "exact_two_candidate_spacing_m",
            "minimum_exact_two_viewpoint_baseline_m",
        )
        for field_name in field_names:
            for malformed in (True, 0.0, -0.1, float("nan"), float("inf"), "1"):
                with self.subTest(field_name=field_name, malformed=malformed):
                    values = {
                        "exact_two_candidate_spacing_m": 0.40,
                        "minimum_exact_two_viewpoint_baseline_m": 1.0,
                    }
                    values[field_name] = malformed
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{field_name} must be finite and positive",
                    ):
                        CoverageSurveyConfig(
                            lane_count=1,
                            exact_inspection_point_count=2,
                            **values,
                        ).validated()

    def test_baseline_gate_uses_world_space_distance(self):
        first = GridCell(0, 0)
        second = GridCell(1, 0)
        targets = (GridCell(2, 0), GridCell(3, 0))
        candidates = (
            ExactTwoViewpointCandidate(first, -0.30, 0.0, targets),
            ExactTwoViewpointCandidate(second, 0.30, 0.0, targets),
        )
        geometry = {targets[0]: (0.0, 0.5), targets[1]: (0.0, -0.5)}

        selected = select_exact_two_viewpoint_cells(
            candidates,
            surveyable_world_xy=geometry,
            coverage_threshold=1.0,
            minimum_viewpoint_baseline_m=0.50,
            start_x_m=-0.30,
            start_y_m=0.0,
        )
        self.assertEqual(selected, (first, second))
        with self.assertRaisesRegex(ValueError, "minimum world-space"):
            select_exact_two_viewpoint_cells(
                candidates,
                surveyable_world_xy=geometry,
                coverage_threshold=1.0,
                minimum_viewpoint_baseline_m=0.70,
                start_x_m=-0.30,
                start_y_m=0.0,
            )

    def test_no_distinct_pair_with_shared_visibility_fails_closed(self):
        with self.assertRaisesRegex(
            ValueError,
            "no valid longitudinal exact-two inspection-point pair",
        ):
            build(
                exact_config(
                    visibility_radius_m=0.04,
                    coverage_threshold=0.001,
                )
            )

    def test_selected_pair_still_has_to_pass_coverage_threshold(self):
        with self.assertRaisesRegex(ValueError, "coverage threshold"):
            build(
                exact_config(
                    visibility_radius_m=1.0,
                    coverage_threshold=0.95,
                )
            )

    def test_selector_round_trips_and_legacy_lane_one_plan_still_loads(self):
        exact = build(exact_config(stop_spacing_m=0.70))
        legacy = legacy_exact_plan()
        dense = build(CoverageSurveyConfig(lane_count=1))
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exact_path = root / "exact.json"
            legacy_path = root / "legacy.json"
            dense_path = root / "dense.json"
            write_coverage_survey_plan(exact_path, exact)
            write_coverage_survey_plan(legacy_path, legacy)
            write_coverage_survey_plan(dense_path, dense)
            exact_payload = json.loads(exact_path.read_text())
            legacy_payload = json.loads(legacy_path.read_text())
            dense_payload = json.loads(dense_path.read_text())
            loaded_exact = load_coverage_survey_plan(exact_path)
            loaded_legacy = load_coverage_survey_plan(legacy_path)
            loaded_dense = load_coverage_survey_plan(dense_path)

        self.assertEqual(exact_payload["config"]["stop_spacing_m"], 0.70)
        self.assertEqual(
            exact_payload["config"]["exact_two_candidate_spacing_m"], 0.40
        )
        self.assertEqual(
            exact_payload["config"][
                "minimum_exact_two_viewpoint_baseline_m"
            ],
            1.0,
        )
        self.assertNotIn("exact_two_candidate_spacing_m", legacy_payload["config"])
        self.assertNotIn(
            "minimum_exact_two_viewpoint_baseline_m",
            legacy_payload["config"],
        )
        self.assertNotIn("exact_inspection_point_count", dense_payload["config"])
        self.assertEqual(loaded_exact, exact)
        self.assertEqual(loaded_legacy, legacy)
        self.assertIsNone(loaded_dense.config.exact_inspection_point_count)

    def test_persisted_string_count_is_not_coerced_to_an_integer(self):
        exact = build(exact_config())
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

    def test_persisted_baseline_gate_is_revalidated_on_load(self):
        exact = build(exact_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tampered_baseline_gate.json"
            write_coverage_survey_plan(path, exact)
            payload = json.loads(path.read_text())
            payload.pop("plan_sha256")
            payload["config"]["minimum_exact_two_viewpoint_baseline_m"] = 3.0
            payload["plan_sha256"] = payload_sha256(payload)
            path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(
                ValueError,
                "persisted minimum world-space viewpoint baseline",
            ):
                load_coverage_survey_plan(path)

    def test_cli_persists_centerline_exact_two_defaults(self):
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
                        "--stop-spacing-m",
                        "0.70",
                        "--arena-length-m",
                        str(ARENA.length_m),
                        "--arena-width-m",
                        str(ARENA.width_m),
                    ]
                )
            loaded = load_coverage_survey_plan(output_dir / "coverage_plan.json")

        self.assertEqual(status, 0)
        self.assertEqual(loaded.config.exact_inspection_point_count, 2)
        self.assertEqual(loaded.config.lane_count, 1)
        self.assertEqual(loaded.config.stop_spacing_m, 0.70)
        self.assertEqual(loaded.config.exact_two_candidate_spacing_m, 0.40)
        self.assertEqual(
            loaded.config.minimum_exact_two_viewpoint_baseline_m,
            1.0,
        )
        self.assertEqual(len(loaded.viewpoints), 2)

    def test_cli_persists_precheckpoint_exact_uncertainty_route_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_path = write_free_map(root)
            output_dir = root / "survey"
            preflight_path = root / "preplanning_localization.json"
            covariance = [0.0] * 36
            preflight_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "failures": [],
                        "route_pose": {
                            "frame_id": "map",
                            "child_frame_id": "base_footprint",
                            "x_m": START.x_m,
                            "y_m": START.y_m,
                            "yaw_rad": START.yaw_rad,
                        },
                        "stationary_amcl_samples": [
                            {"covariance": list(covariance)}
                            for _ in range(5)
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
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
                        "exact_two_uncertainty_selection",
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
                        "--startup-route-selection-preflight-json",
                        str(preflight_path),
                        "--startup-route-selection-robot-radius-m",
                        "0.105",
                        "--startup-route-selection-collision-margin-m",
                        "0.02",
                        "--startup-route-selection-tracking-tube-radius-m",
                        "0.03",
                        "--startup-route-selection-odom-drift-bound-m",
                        "0.02",
                        "--startup-route-selection-braking-latency-distance-m",
                        "0.015",
                        "--startup-route-selection-sigma-multiplier",
                        "2.0",
                        "--startup-route-selection-clearance-sample-spacing-m",
                        "0.005",
                    ]
                )
            selection_path = (
                output_dir / "startup_route_uncertainty_selection.json"
            )
            selection = load_content_hashed_json(
                selection_path,
                hash_field=STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD,
            )
            diagnostics = json.loads(
                (output_dir / "legs/leg_000_diagnostics.json").read_text()
            )

        self.assertEqual(status, 0)
        self.assertTrue(selection["selection"]["decision"]["ready"])
        self.assertEqual(len(selection["selection"]["options"]), 2)
        selected_id = selection["selection"]["decision"][
            "selected_option_id"
        ]
        self.assertEqual(
            diagnostics["metadata"]["target_viewpoint_id"],
            selected_id,
        )
        self.assertEqual(
            diagnostics["metadata"][
                "startup_route_uncertainty_selection"
            ]["selected_viewpoint_id"],
            selected_id,
        )
        self.assertFalse(selection["motion_authorized"])

    def test_longitudinal_plan_uses_bound_physical_route_source(self):
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
                        "longitudinal_route_source",
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
            self.assertEqual(status, 0)
            outputs = seal_stand_discovery_route(
                source_route_csv=output_dir / "legs/leg_000_route.csv",
                source_diagnostics_json=(
                    output_dir / "legs/leg_000_diagnostics.json"
                ),
                coverage_plan_path=output_dir / "coverage_plan.json",
                output_dir=root / "sealed",
            )
            diagnostics_path = Path(outputs["diagnostics_json"])
            diagnostics = json.loads(diagnostics_path.read_text())
            leg = load_route_leg(Path(outputs["route_csv"]), 0)
            accepted = validate_stand_discovery_route_binding(
                diagnostics_path,
                leg,
                coverage_plan_path=output_dir / "coverage_plan.json",
            )
            tampered = json.loads(diagnostics_path.read_text())
            tampered["metadata"]["source"] = STAND_DISCOVERY_ROUTE_SOURCE
            rejected = validate_stand_discovery_route_binding(
                diagnostics_path,
                leg,
                coverage_plan_path=output_dir / "coverage_plan.json",
                diagnostics_payload=tampered,
            )

        self.assertTrue(accepted.ok, accepted.failures)
        self.assertEqual(
            diagnostics["metadata"]["source"],
            LONGITUDINALLY_DIVERSE_STAND_DISCOVERY_ROUTE_SOURCE,
        )
        self.assertFalse(rejected.ok)
        self.assertIn(
            "stand discovery route source does not match persisted viewpoint geometry",
            rejected.failures,
        )


if __name__ == "__main__":
    unittest.main()
