import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.aufgabe04.navigation.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.artifacts import (
    write_diagnostics_json,
    write_route_csv,
)
from scripts.aufgabe04.navigation.plan_stand_coverage_survey import (
    main as plan_coverage,
)
from scripts.aufgabe04.navigation.stand_blockage_replan import (
    blocker_candidate_uids,
    plan_blockage_route_to_viewpoint,
)
from scripts.aufgabe04.navigation.stand_discovery_route import (
    seal_stand_discovery_route,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    STATUS_PROVISIONAL,
    StandSurveyRegistry,
    SurveyCandidate,
    coverage_survey_plan_sha256,
    load_coverage_survey_plan,
)
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand


MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")


class StandBlockageReplanTest(unittest.TestCase):
    def _plan(self, root: Path):
        with redirect_stdout(StringIO()):
            status = plan_coverage(
                [
                    "--map",
                    str(MAP),
                    "--semantic-map-id",
                    "arena_1p898x3p9_auto",
                    "--planning-frame",
                    "map",
                    "--start-x",
                    "-0.80",
                    "--start-y",
                    "-0.465",
                    "--start-yaw",
                    str(math.pi),
                    "--survey-id",
                    "blockage_replan_test",
                    "--output-dir",
                    str(root),
                    "--lane-count",
                    "1",
                    "--stop-spacing-m",
                    "0.70",
                    "--candidate-keepout-radius-m",
                    "0.34",
                    "--expected-stand-count",
                    "1",
                ]
            )
        self.assertEqual(status, 0)
        return load_coverage_survey_plan(root / "coverage_plan.json")

    @staticmethod
    def _candidate(*, x_m=-1.07, y_m=-0.465):
        return SurveyCandidate(
            candidate_uid="survey_candidate_0001",
            x_m=x_m,
            y_m=y_m,
            radius_m=0.06,
            uncertainty_m=0.02,
            keepout_radius_m=0.34,
            confidence=0.9,
            hit_count=4,
            first_seen_sec=1.0,
            last_seen_sec=2.0,
            source_observation_ids=("obs_1", "obs_2", "obs_3"),
            viewpoint_ids=("blockage_001",),
            status=STATUS_PROVISIONAL,
        )

    def test_plans_exact_start_egress_away_from_blocking_stand(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "survey"
            plan = self._plan(root)
            candidate = self._candidate()
            registry = StandSurveyRegistry(
                schema_version=1,
                survey_id=plan.survey_id,
                planning_frame=plan.planning_frame,
                map_bundle_sha256=plan.map_bundle_sha256,
                candidates=(candidate,),
            )
            grid, _bundle = load_occupancy_grid_with_bundle(
                MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                planning_frame="map",
            )
            start = Pose2D(-0.80, -0.465, math.pi)
            target = plan.viewpoints[0]

            replanned = plan_blockage_route_to_viewpoint(
                grid,
                plan=plan,
                registry=registry,
                start=start,
                target_viewpoint_id=target.viewpoint_id,
                blocker_uids=(candidate.candidate_uid,),
                robot_radius_m=0.105,
            )

            route = replanned.route_result.route
            self.assertIsNotNone(route)
            assert route is not None
            self.assertAlmostEqual(route.points[0].pose.x_m, start.x_m)
            self.assertAlmostEqual(route.points[0].pose.y_m, start.y_m)
            start_distance = math.hypot(
                start.x_m - candidate.x_m,
                start.y_m - candidate.y_m,
            )
            anchor_distance = math.hypot(
                replanned.egress_anchor.x_m - candidate.x_m,
                replanned.egress_anchor.y_m - candidate.y_m,
            )
            self.assertGreater(anchor_distance, start_distance)
            self.assertGreater(replanned.minimum_egress_hard_clearance_m, 0.0)
            self.assertAlmostEqual(route.points[-1].pose.x_m, target.pose.x_m)
            self.assertAlmostEqual(route.points[-1].pose.y_m, target.pose.y_m)

    def test_replacement_route_passes_existing_sealing_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "survey"
            plan = self._plan(root)
            candidate = self._candidate()
            registry = StandSurveyRegistry(
                schema_version=1,
                survey_id=plan.survey_id,
                planning_frame=plan.planning_frame,
                map_bundle_sha256=plan.map_bundle_sha256,
                candidates=(candidate,),
            )
            grid, _bundle = load_occupancy_grid_with_bundle(
                MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                planning_frame="map",
            )
            target = plan.viewpoints[0]
            replanned = plan_blockage_route_to_viewpoint(
                grid,
                plan=plan,
                registry=registry,
                start=Pose2D(-0.80, -0.465, math.pi),
                target_viewpoint_id=target.viewpoint_id,
                blocker_uids=(candidate.candidate_uid,),
                robot_radius_m=0.105,
            )
            replacement_root = Path(tmp) / "replacement"
            route_path = replacement_root / "route.csv"
            diagnostics_path = replacement_root / "route_diagnostics.json"
            write_route_csv(
                route_path,
                (replanned.route_result,),
                final_yaw_by_leg={0: target.pose.yaw_rad},
            )
            write_diagnostics_json(
                diagnostics_path,
                (replanned.route_result,),
                metadata={
                    "schema_version": 1,
                    "route_kind": "stand_coverage_survey",
                    "motion_authorized": False,
                    "adaptive_blockage_replan": True,
                    "survey_id": plan.survey_id,
                    "plan_sha256": coverage_survey_plan_sha256(plan),
                    "map_bundle_sha256": plan.map_bundle_sha256,
                    "target_viewpoint_id": target.viewpoint_id,
                    "inflation_radius_m": plan.config.inflation_radius_m,
                    "candidate_keepout_count": 1,
                    "egress_anchor": {
                        "x_m": replanned.egress_anchor.x_m,
                        "y_m": replanned.egress_anchor.y_m,
                    },
                    "egress_distance_m": replanned.egress_distance_m,
                    "minimum_egress_hard_clearance_m": (
                        replanned.minimum_egress_hard_clearance_m
                    ),
                    "arena_boundary_overlay": True,
                    "arena_bounds": plan.arena_bounds.to_metadata(),
                },
            )

            sealed = seal_stand_discovery_route(
                source_route_csv=route_path,
                source_diagnostics_json=diagnostics_path,
                coverage_plan_path=root / "coverage_plan.json",
                output_dir=Path(tmp) / "sealed",
            )

            self.assertTrue(Path(sealed["route_csv"]).is_file())
            self.assertTrue(Path(sealed["route_certificate_json"]).is_file())

    def test_fails_closed_inside_hard_stand_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "survey"
            plan = self._plan(root)
            candidate = self._candidate(x_m=-1.00)
            registry = StandSurveyRegistry(
                schema_version=1,
                survey_id=plan.survey_id,
                planning_frame=plan.planning_frame,
                map_bundle_sha256=plan.map_bundle_sha256,
                candidates=(candidate,),
            )
            grid, _bundle = load_occupancy_grid_with_bundle(
                MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                planning_frame="map",
            )
            with self.assertRaisesRegex(ValueError, "hard stand exclusion"):
                plan_blockage_route_to_viewpoint(
                    grid,
                    plan=plan,
                    registry=registry,
                    start=Pose2D(-0.80, -0.465, math.pi),
                    target_viewpoint_id=plan.viewpoints[0].viewpoint_id,
                    blocker_uids=(candidate.candidate_uid,),
                    robot_radius_m=0.105,
                )

    def test_binds_only_new_near_frontal_stand(self):
        candidate = self._candidate()
        registry = StandSurveyRegistry(
            schema_version=1,
            survey_id="survey",
            planning_frame="map",
            map_bundle_sha256="a" * 64,
            candidates=(candidate,),
        )
        stand = ConfirmedStand(
            stand_id="detected_stand_01",
            x_m=candidate.x_m,
            y_m=candidate.y_m,
            confidence=0.9,
            hit_count=3,
            first_seen_sec=1.0,
            last_seen_sec=2.0,
            first_confirmed_at_sec=2.0,
            source_observation_ids=("obs_1", "obs_2", "obs_3"),
            provenance={},
        )

        selected = blocker_candidate_uids(
            registry,
            (stand,),
            Pose2D(-0.80, -0.465, math.pi),
        )

        self.assertEqual(selected, (candidate.candidate_uid,))


if __name__ == "__main__":
    unittest.main()
