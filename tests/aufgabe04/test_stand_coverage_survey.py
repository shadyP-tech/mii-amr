from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
from io import StringIO
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.exact_start_connector import (
    prepend_certified_exact_start,
)
from scripts.aufgabe04.navigation.global_planner import plan_route
from scripts.aufgabe04.navigation.map_io import (
    CELL_FREE,
    MapMetadata,
    OccupancyGrid,
    load_occupancy_grid_with_bundle,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.simple_waypoint_follower import (
    certified_static_startup_decision,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    STATUS_CONFIRMED,
    STATUS_PENDING_CAMERA,
    STATUS_PROVISIONAL,
    CoverageSurveyConfig,
    build_coverage_survey_plan,
    decide_candidate,
    fuse_confirmed_stands,
    load_coverage_survey_plan,
    load_stand_survey_registry,
    load_survey_progress,
    mark_viewpoint_visited,
    new_stand_survey_registry,
    new_survey_progress,
    plan_next_survey_leg,
    survey_status,
    visited_coverage_ratio,
    write_coverage_survey_plan,
    write_stand_survey_registry,
    write_survey_progress,
)
from scripts.aufgabe04.navigation.record_stand_coverage_stop import (
    main as record_coverage_stop_main,
)
from scripts.aufgabe04.navigation.record_stand_candidate_decision import (
    main as record_candidate_decision_main,
)
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand


MAP_HASH = "a" * 64


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
            [
                "image: map.pgm",
                "resolution: 0.1",
                "origin: [-2.5, -1.5, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.20",
                "mode: trinary",
            ]
        )
        + "\n"
    )
    return root / "map.yaml"


def stand(
    *,
    stand_id: str,
    x_m: float,
    y_m: float,
    observation_prefix: str,
    timestamp: float,
) -> ConfirmedStand:
    observation_ids = tuple(
        f"{observation_prefix}_{index}" for index in range(3)
    )
    return ConfirmedStand(
        stand_id=stand_id,
        x_m=x_m,
        y_m=y_m,
        confidence=0.8,
        hit_count=3,
        first_seen_sec=timestamp,
        last_seen_sec=timestamp + 0.2,
        first_confirmed_at_sec=timestamp + 0.2,
        source_observation_ids=observation_ids,
        provenance={},
    )


def plan(config: CoverageSurveyConfig | None = None):
    return build_coverage_survey_plan(
        free_grid(),
        map_bundle_sha256=MAP_HASH,
        start=Pose2D(-1.5, 0.0, 0.0),
        survey_id="survey_test",
        arena_bounds=ArenaBounds(length_m=4.0, width_m=2.0),
        config=config,
    )


class CoverageSurveyPlanTest(unittest.TestCase):
    def test_two_rail_plan_meets_coverage_gate_and_uses_distinct_lanes(self):
        survey = plan()

        lane_y = {round(item.pose.y_m, 1) for item in survey.viewpoints}
        self.assertEqual(len(lane_y), 2)
        self.assertGreaterEqual(survey.planned_coverage_ratio, 0.95)
        self.assertEqual(
            len(survey.viewpoint_ids),
            len(set(survey.viewpoint_ids)),
        )

    def test_visited_coverage_is_monotonic_and_not_candidate_count_based(self):
        survey = plan()
        progress = new_survey_progress(survey)
        ratios = []
        for viewpoint in survey.viewpoints:
            progress = mark_viewpoint_visited(
                survey, progress, viewpoint.viewpoint_id
            )
            ratios.append(visited_coverage_ratio(survey, progress))

        self.assertEqual(ratios, sorted(ratios))
        self.assertAlmostEqual(ratios[-1], survey.planned_coverage_ratio)
        status = survey_status(
            survey,
            progress,
            new_stand_survey_registry(survey),
        )
        self.assertTrue(status["coverage_complete"])
        self.assertTrue(status["exploration_complete"])

    def test_plan_progress_and_registry_round_trip(self):
        survey = plan()
        progress = mark_viewpoint_visited(
            survey,
            new_survey_progress(survey),
            survey.viewpoints[0].viewpoint_id,
        )
        registry = new_stand_survey_registry(survey)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_coverage_survey_plan(root / "plan.json", survey)
            write_survey_progress(root / "progress.json", progress, survey)
            write_stand_survey_registry(
                root / "registry.json", registry, survey
            )
            loaded_plan = load_coverage_survey_plan(root / "plan.json")
            loaded_progress = load_survey_progress(
                root / "progress.json", loaded_plan
            )
            loaded_registry = load_stand_survey_registry(
                root / "registry.json", loaded_plan
            )

        self.assertEqual(loaded_plan, survey)
        self.assertEqual(loaded_progress, progress)
        self.assertEqual(loaded_registry, registry)

    def test_live_amcl_start_is_prepended_and_accepted_by_startup_gate(self):
        map_path = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
        grid, bundle = load_occupancy_grid_with_bundle(
            map_path,
            semantic_map_id="arena_1p898x3p9_auto",
            planning_frame="map",
        )
        planned_start = Pose2D(
            -0.4297676901761188,
            -0.6484791305211309,
            1.603697893943687,
        )
        survey = build_coverage_survey_plan(
            grid,
            map_bundle_sha256=bundle.bundle_sha256,
            start=planned_start,
            survey_id="live_start_regression",
            config=CoverageSurveyConfig(lane_count=1, expected_stand_count=3),
        )
        leg = plan_next_survey_leg(
            grid,
            plan=survey,
            progress=new_survey_progress(survey),
            registry=new_stand_survey_registry(survey),
            current_pose=planned_start,
        )

        self.assertIsNotNone(leg)
        assert leg is not None and leg.route_result.route is not None
        self.assertTrue(leg.exact_start_connector.required)
        self.assertTrue(leg.exact_start_connector.validated)
        self.assertGreater(leg.exact_start_connector.minimum_margin_m, 0.039)
        self.assertEqual(leg.route_result.route.points[0].pose, planned_start)
        runtime_pose = Pose2D(
            -0.4074980823903188,
            -0.6491407250805117,
            1.6028881170512068,
        )
        startup = certified_static_startup_decision(
            runtime_pose,
            tuple(point.pose for point in leg.route_result.route.points),
            tracking_tube_radius_m=0.03,
        )
        self.assertTrue(startup.ok)
        self.assertEqual(startup.target_index, 1)

    def test_exact_start_connector_fails_closed_near_arena_wall(self):
        base = Costmap.from_occupancy_grid(free_grid()).with_arena_bounds(
            ArenaBounds(length_m=4.0, width_m=2.0)
        )
        planning = base.with_inflation(0.20)
        start = Pose2D(-1.0, -0.95, 0.0)
        result = plan_route(planning, start, Pose2D(0.0, 0.0), snap_radius_m=0.5)
        self.assertIsNotNone(result.route)

        with self.assertRaisesRegex(ValueError, "lacks continuous static clearance"):
            prepend_certified_exact_start(
                result,
                base_costmap=base,
                start=start,
                required_clearance_m=0.20,
            )

    def test_exact_cell_center_start_is_not_duplicated(self):
        base = Costmap.from_occupancy_grid(free_grid()).with_arena_bounds(
            ArenaBounds(length_m=4.0, width_m=2.0)
        )
        planning = base.with_inflation(0.20)
        start = Pose2D(-1.45, 0.05, 0.0)
        result = plan_route(planning, start, Pose2D(0.45, 0.05), snap_radius_m=0.3)
        self.assertIsNotNone(result.route)
        original_count = len(result.route.points)

        joined, evidence = prepend_certified_exact_start(
            result,
            base_costmap=base,
            start=start,
            required_clearance_m=0.20,
        )

        self.assertFalse(evidence.required)
        self.assertEqual(len(joined.route.points), original_count)

    def test_off_center_start_bypasses_same_cell_grid_center(self):
        base = Costmap.from_occupancy_grid(free_grid()).with_arena_bounds(
            ArenaBounds(length_m=4.0, width_m=2.0)
        )
        planning = base.with_inflation(0.20)
        start = Pose2D(-1.47, 0.03, 0.0)
        result = plan_route(planning, start, Pose2D(0.45, 0.05), snap_radius_m=0.3)
        self.assertIsNotNone(result.route)
        assert result.route is not None
        original_count = len(result.route.points)
        redundant_center = result.route.points[0]
        first_forward_vertex = result.route.points[1]
        self.assertEqual(redundant_center.cell, base.world_to_grid(start))

        joined, evidence = prepend_certified_exact_start(
            result,
            base_costmap=base,
            start=start,
            required_clearance_m=0.20,
        )

        assert joined.route is not None
        self.assertTrue(evidence.required)
        self.assertTrue(evidence.validated)
        self.assertEqual(evidence.anchor, first_forward_vertex.pose)
        self.assertEqual(joined.route.points[0].pose, start)
        self.assertEqual(joined.route.points[1].pose, first_forward_vertex.pose)
        self.assertNotEqual(joined.route.points[0].cell, joined.route.points[1].cell)
        self.assertNotIn(
            redundant_center.pose,
            tuple(point.pose for point in joined.route.points[1:]),
        )
        self.assertEqual(len(joined.route.points), original_count)

    def test_physical_start_regression_has_no_same_cell_reversal(self):
        map_path = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
        grid, bundle = load_occupancy_grid_with_bundle(
            map_path,
            semantic_map_id="arena_1p898x3p9_auto",
            planning_frame="map",
        )
        physical_start = Pose2D(
            -0.3674933407222252,
            -0.6078591295582066,
            1.660126876187633,
        )
        survey = build_coverage_survey_plan(
            grid,
            map_bundle_sha256=bundle.bundle_sha256,
            start=physical_start,
            survey_id="same_cell_reversal_regression",
            config=CoverageSurveyConfig(lane_count=1, expected_stand_count=5),
        )
        leg = plan_next_survey_leg(
            grid,
            plan=survey,
            progress=new_survey_progress(survey),
            registry=new_stand_survey_registry(survey),
            current_pose=physical_start,
        )

        self.assertIsNotNone(leg)
        assert leg is not None and leg.route_result.route is not None
        points = leg.route_result.route.points
        self.assertGreaterEqual(len(points), 3)
        self.assertEqual(points[0].pose, physical_start)
        self.assertNotEqual(points[0].cell, points[1].cell)
        first_heading = math.atan2(
            points[1].pose.y_m - points[0].pose.y_m,
            points[1].pose.x_m - points[0].pose.x_m,
        )
        second_heading = math.atan2(
            points[2].pose.y_m - points[1].pose.y_m,
            points[2].pose.x_m - points[1].pose.x_m,
        )
        turn_rad = abs(
            (second_heading - first_heading + math.pi)
            % (2.0 * math.pi)
            - math.pi
        )
        self.assertLess(turn_rad, math.pi / 2.0)


class StandSurveyRegistryTest(unittest.TestCase):
    def test_candidate_requires_distinct_viewpoints_before_camera_queue(self):
        survey = plan()
        registry = new_stand_survey_registry(survey)
        first = stand(
            stand_id="local_1",
            x_m=0.5,
            y_m=0.1,
            observation_prefix="first",
            timestamp=10.0,
        )
        registry = fuse_confirmed_stands(
            registry,
            (first,),
            viewpoint_id="survey_vp_001",
            config=survey.config,
        )

        self.assertEqual(len(registry.candidates), 1)
        candidate = registry.candidates[0]
        self.assertEqual(candidate.status, STATUS_PROVISIONAL)
        stable_uid = candidate.candidate_uid

        second = stand(
            stand_id="unrelated_local_id",
            x_m=0.54,
            y_m=0.08,
            observation_prefix="second",
            timestamp=20.0,
        )
        registry = fuse_confirmed_stands(
            registry,
            (second,),
            viewpoint_id="survey_vp_006",
            config=survey.config,
        )

        self.assertEqual(len(registry.candidates), 1)
        candidate = registry.candidates[0]
        self.assertEqual(candidate.candidate_uid, stable_uid)
        self.assertEqual(candidate.status, STATUS_PENDING_CAMERA)
        self.assertEqual(
            candidate.viewpoint_ids,
            ("survey_vp_001", "survey_vp_006"),
        )

    def test_replayed_observations_do_not_create_fake_viewpoint_diversity(self):
        survey = plan()
        first = stand(
            stand_id="local_1",
            x_m=0.5,
            y_m=0.1,
            observation_prefix="same",
            timestamp=10.0,
        )
        registry = fuse_confirmed_stands(
            new_stand_survey_registry(survey),
            (first,),
            viewpoint_id="survey_vp_001",
            config=survey.config,
        )
        replayed = fuse_confirmed_stands(
            registry,
            (first,),
            viewpoint_id="survey_vp_002",
            config=survey.config,
        )

        self.assertEqual(replayed, registry)
        self.assertEqual(replayed.candidates[0].status, STATUS_PROVISIONAL)

    def test_epoch_uses_global_assignment_when_greedy_would_lose_match(self):
        survey = plan()
        registry = fuse_confirmed_stands(
            new_stand_survey_registry(survey),
            (
                stand(
                    stand_id="first_a",
                    x_m=0.0,
                    y_m=0.0,
                    observation_prefix="first_a",
                    timestamp=10.0,
                ),
                stand(
                    stand_id="first_b",
                    x_m=0.20,
                    y_m=0.0,
                    observation_prefix="first_b",
                    timestamp=10.0,
                ),
            ),
            viewpoint_id="survey_vp_001",
            config=survey.config,
        )
        original_uids = tuple(
            candidate.candidate_uid for candidate in registry.candidates
        )
        contested = stand(
            stand_id="contested",
            x_m=0.08,
            y_m=0.0,
            observation_prefix="contested",
            timestamp=20.0,
        )
        first_only = stand(
            stand_id="first_only",
            x_m=0.09,
            y_m=-0.15,
            observation_prefix="first_only",
            timestamp=20.0,
        )

        updated = fuse_confirmed_stands(
            registry,
            (contested, first_only),
            viewpoint_id="survey_vp_002",
            config=survey.config,
        )
        reversed_input = fuse_confirmed_stands(
            registry,
            (first_only, contested),
            viewpoint_id="survey_vp_002",
            config=survey.config,
        )

        self.assertEqual(updated, reversed_input)
        self.assertEqual(len(updated.candidates), 2)
        self.assertEqual(
            tuple(candidate.candidate_uid for candidate in updated.candidates),
            original_uids,
        )
        self.assertIn(
            "first_only_0",
            updated.candidates[0].source_observation_ids,
        )
        self.assertIn(
            "contested_0",
            updated.candidates[1].source_observation_ids,
        )
        self.assertTrue(
            all(
                candidate.status == STATUS_PENDING_CAMERA
                for candidate in updated.candidates
            )
        )

    def test_camera_decisions_drive_exploration_completion(self):
        survey = plan(
            CoverageSurveyConfig(
                minimum_distinct_viewpoints=1,
                expected_stand_count=1,
            )
        )
        progress = new_survey_progress(survey)
        for viewpoint in survey.viewpoints:
            progress = mark_viewpoint_visited(
                survey, progress, viewpoint.viewpoint_id
            )
        registry = fuse_confirmed_stands(
            new_stand_survey_registry(survey),
            (
                stand(
                    stand_id="local",
                    x_m=0.0,
                    y_m=0.0,
                    observation_prefix="obs",
                    timestamp=10.0,
                ),
            ),
            viewpoint_id=survey.viewpoints[0].viewpoint_id,
            config=survey.config,
        )

        self.assertFalse(
            survey_status(survey, progress, registry)["exploration_complete"]
        )
        registry = decide_candidate(
            registry,
            registry.candidates[0].candidate_uid,
            status=STATUS_CONFIRMED,
        )
        self.assertTrue(
            survey_status(survey, progress, registry)["exploration_complete"]
        )

    def test_next_leg_is_replanned_around_provisional_keepout(self):
        survey = plan()
        progress = new_survey_progress(survey)
        registry = fuse_confirmed_stands(
            new_stand_survey_registry(survey),
            (
                stand(
                    stand_id="local",
                    x_m=0.0,
                    y_m=0.0,
                    observation_prefix="keepout",
                    timestamp=10.0,
                ),
            ),
            viewpoint_id=survey.viewpoints[-1].viewpoint_id,
            config=survey.config,
        )
        leg = plan_next_survey_leg(
            free_grid(),
            plan=survey,
            progress=progress,
            registry=registry,
            current_pose=Pose2D(-1.5, 0.0, 0.0),
        )

        self.assertIsNotNone(leg)
        route = leg.route_result.route
        self.assertIsNotNone(route)
        candidate = registry.candidates[0]
        for point in route.points:
            distance = (
                (point.pose.x_m - candidate.x_m) ** 2
                + (point.pose.y_m - candidate.y_m) ** 2
            ) ** 0.5
            self.assertGreaterEqual(
                distance + free_grid().metadata.resolution,
                candidate.keepout_radius_m,
            )

    def test_progress_hash_rejects_a_different_plan(self):
        survey = plan()
        progress = new_survey_progress(survey)
        changed = replace(
            survey,
            survey_id="other",
        )
        with self.assertRaises(ValueError):
            mark_viewpoint_visited(
                changed,
                progress,
                changed.viewpoints[0].viewpoint_id,
            )


class RecordCoverageStopCliTest(unittest.TestCase):
    def test_negative_observation_receipt_advances_coverage_and_replans(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_free_map(root)
            grid, bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id=map_yaml.stem,
                planning_frame="map",
            )
            survey = build_coverage_survey_plan(
                grid,
                map_bundle_sha256=bundle.bundle_sha256,
                start=Pose2D(-1.5, 0.0, 0.0),
                survey_id="survey_cli",
                arena_bounds=ArenaBounds(length_m=4.0, width_m=2.0),
            )
            survey_root = root / "survey"
            write_coverage_survey_plan(
                survey_root / "coverage_plan.json", survey
            )
            write_survey_progress(
                survey_root / "coverage_progress.json",
                new_survey_progress(survey),
                survey,
            )
            write_stand_survey_registry(
                survey_root / "stand_registry.json",
                new_stand_survey_registry(survey),
                survey,
            )
            viewpoint = survey.viewpoints[0]
            observations_path = root / "negative_observations.jsonl"
            observer_summary = root / "observer_summary.json"
            observer_summary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "motion_published": False,
                        "processed_scan_count": 5,
                        "accepted_observation_count": 0,
                        "map_bundle_sha256": bundle.bundle_sha256,
                        "planning_frame": "map",
                        "output_jsonl": str(observations_path),
                        "scan_frame_pose_in_planning_frame": {
                            "x_m": viewpoint.pose.x_m,
                            "y_m": viewpoint.pose.y_m,
                            "yaw_rad": viewpoint.pose.yaw_rad,
                        },
                    }
                )
            )

            with redirect_stdout(StringIO()):
                result = record_coverage_stop_main(
                    [
                        "--survey-root",
                        str(survey_root),
                        "--map",
                        str(map_yaml),
                        "--viewpoint-id",
                        viewpoint.viewpoint_id,
                        "--observer-summary-json",
                        str(observer_summary),
                    ]
                )
            progress = load_survey_progress(
                survey_root / "coverage_progress.json",
                survey,
            )

        self.assertEqual(result, 0)
        self.assertEqual(progress.visited_viewpoint_ids, (viewpoint.viewpoint_id,))

    def test_evidence_receipt_records_pending_camera_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            survey = plan(
                CoverageSurveyConfig(minimum_distinct_viewpoints=1)
            )
            survey_root = root / "survey"
            progress = new_survey_progress(survey)
            registry = fuse_confirmed_stands(
                new_stand_survey_registry(survey),
                (
                    stand(
                        stand_id="local",
                        x_m=0.0,
                        y_m=0.0,
                        observation_prefix="decision",
                        timestamp=10.0,
                    ),
                ),
                viewpoint_id=survey.viewpoints[0].viewpoint_id,
                config=survey.config,
            )
            candidate_uid = registry.candidates[0].candidate_uid
            write_coverage_survey_plan(
                survey_root / "coverage_plan.json", survey
            )
            write_survey_progress(
                survey_root / "coverage_progress.json",
                progress,
                survey,
            )
            write_stand_survey_registry(
                survey_root / "stand_registry.json",
                registry,
                survey,
            )
            receipt = root / "decision.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "survey_id": survey.survey_id,
                        "candidate_uid": candidate_uid,
                        "decision": "confirmed",
                        "decision_source": "operator",
                        "operator_confirmed": True,
                    }
                )
            )

            with redirect_stdout(StringIO()):
                result = record_candidate_decision_main(
                    [
                        "--survey-root",
                        str(survey_root),
                        "--decision-receipt-json",
                        str(receipt),
                    ]
                )
            loaded = load_stand_survey_registry(
                survey_root / "stand_registry.json",
                survey,
            )

        self.assertEqual(result, 0)
        self.assertEqual(loaded.candidates[0].status, STATUS_CONFIRMED)


if __name__ == "__main__":
    unittest.main()
