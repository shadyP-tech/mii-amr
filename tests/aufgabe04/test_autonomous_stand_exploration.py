import json
from contextlib import redirect_stdout
from io import StringIO
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.plan_stand_coverage_survey import (
    main as plan_coverage,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.ros_preflight import RosPreflightResult
from scripts.aufgabe04.navigation.run_single_station_segment import (
    main as run_segment,
)
from scripts.aufgabe04.navigation.simple_waypoint_follower import (
    STATIC_PHYSICAL_ROUTE_KINDS,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    STATUS_PENDING_CAMERA,
    SurveyCandidate,
    load_coverage_survey_plan,
    load_stand_survey_registry,
    write_stand_survey_registry,
)
from scripts.aufgabe04.navigation.stand_discovery_route import (
    STAND_DISCOVERY_ROUTE_KIND,
    seal_stand_discovery_route,
    validate_stand_discovery_route_binding,
)
from scripts.aufgabe04.navigation.waypoint_csv import load_route_leg
from scripts.aufgabe04.real_robot.passive_viewpoint_node import _resolved_qr_id
from scripts.aufgabe04.real_robot.run_autonomous_stand_exploration import (
    MotionLegOutcome,
    _bounded_approach_offsets,
    _execute_coverage_leg_with_replans,
    _is_confirmable_stand_blockage,
    _opposite_face_normal,
    build_parser,
    candidate_snapshot_from_registry,
    plan_candidate_preapproach,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    write_candidate_snapshot,
)


MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")


class AutonomousStandExplorationTest(unittest.TestCase):
    def _planned_center_corridor(self, root: Path) -> None:
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
                    "0",
                    "--start-y",
                    "0",
                    "--start-yaw",
                    "0",
                    "--survey-id",
                    "autonomous_test",
                    "--output-dir",
                    str(root),
                    "--lane-count",
                    "1",
                    "--stop-spacing-m",
                    "0.70",
                    "--expected-stand-count",
                    "1",
                ]
            )
        self.assertEqual(status, 0)

    def test_single_center_lane_is_plannable_and_meets_coverage_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "survey"
            self._planned_center_corridor(root)
            plan = load_coverage_survey_plan(root / "coverage_plan.json")
            self.assertEqual(plan.config.lane_count, 1)
            self.assertGreaterEqual(plan.planned_coverage_ratio, 0.95)
            self.assertTrue(
                all(abs(viewpoint.pose.y_m) <= 0.35 for viewpoint in plan.viewpoints)
            )

    def test_center_corridor_leg_is_sealed_as_a_certified_physical_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "survey"
            sealed_root = Path(tmp) / "sealed"
            self._planned_center_corridor(root)
            outputs = seal_stand_discovery_route(
                source_route_csv=root / "legs/leg_000_route.csv",
                source_diagnostics_json=root / "legs/leg_000_diagnostics.json",
                coverage_plan_path=root / "coverage_plan.json",
                output_dir=sealed_root,
            )
            leg = load_route_leg(Path(outputs["route_csv"]), 0)
            status = validate_stand_discovery_route_binding(
                Path(outputs["diagnostics_json"]),
                leg,
                coverage_plan_path=root / "coverage_plan.json",
            )
            self.assertTrue(status.ok, status.failures)
            self.assertEqual(leg.route_kind, STAND_DISCOVERY_ROUTE_KIND)
            self.assertFalse(leg.simulation_only)
            self.assertTrue(leg.raw_waypoints[-1].protected)
            self.assertIn(
                STAND_DISCOVERY_ROUTE_KIND,
                STATIC_PHYSICAL_ROUTE_KINDS,
            )
            diagnostics = json.loads(
                Path(outputs["diagnostics_json"]).read_text()
            )
            self.assertTrue(diagnostics["metadata"]["motion_authorized"])

            with patch(
                "scripts.aufgabe04.navigation.run_single_station_segment."
                "run_ros_preflight",
                return_value=RosPreflightResult(
                    ok=True,
                    failures=[],
                    observations=[],
                    runtime_config={},
                ),
            ), patch(
                "scripts.aufgabe04.navigation.run_single_station_segment."
                "run_simple_waypoint_follower"
            ) as follower, redirect_stdout(StringIO()):
                runner_status = run_segment(
                    [
                        "--route-csv",
                        outputs["route_csv"],
                        "--diagnostics-json",
                        outputs["diagnostics_json"],
                        "--route-certificate-json",
                        outputs["route_certificate_json"],
                        "--coverage-plan",
                        str(root / "coverage_plan.json"),
                        "--leg-index",
                        "0",
                        "--semantic-log",
                        str(Path(tmp) / "events.jsonl"),
                        "--results-csv",
                        str(Path(tmp) / "results.csv"),
                        "--preflight-json",
                        str(Path(tmp) / "preflight.json"),
                        "--dry-run",
                    ]
                )
            self.assertEqual(runner_status, 0)
            follower.assert_not_called()

    def test_pending_registry_freezes_to_candidate_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "survey"
            self._planned_center_corridor(root)
            plan = load_coverage_survey_plan(root / "coverage_plan.json")
            registry_path = root / "stand_registry.json"
            registry = load_stand_survey_registry(registry_path, plan)
            candidate = SurveyCandidate(
                candidate_uid="survey_candidate_0001",
                x_m=0.2,
                y_m=0.3,
                radius_m=0.06,
                uncertainty_m=0.02,
                keepout_radius_m=0.31,
                confidence=0.9,
                hit_count=5,
                first_seen_sec=10.0,
                last_seen_sec=12.0,
                source_observation_ids=("obs_1", "obs_2"),
                viewpoint_ids=("survey_vp_001", "survey_vp_002"),
                status=STATUS_PENDING_CAMERA,
            )
            registry = type(registry)(
                schema_version=registry.schema_version,
                survey_id=registry.survey_id,
                planning_frame=registry.planning_frame,
                map_bundle_sha256=registry.map_bundle_sha256,
                candidates=(candidate,),
            )
            write_stand_survey_registry(registry_path, registry, plan)
            snapshot = candidate_snapshot_from_registry(
                registry,
                plan,
                registry_path=registry_path,
                snapshot_id="autonomous_snapshot",
            )
            self.assertEqual(snapshot.candidate_uids, (candidate.candidate_uid,))
            self.assertEqual(
                snapshot.candidates[0].source.observation_ids,
                candidate.source_observation_ids,
            )

    def test_auto_qr_requires_exactly_one_identity(self):
        self.assertEqual(_resolved_qr_id("auto", ("A",)), "A")
        self.assertIsNone(_resolved_qr_id("auto", ()))
        self.assertIsNone(_resolved_qr_id("auto", ("A", "B")))
        self.assertEqual(_resolved_qr_id("A", ("A",)), "A")
        self.assertIsNone(_resolved_qr_id("A", ("B",)))

    def test_axis_only_observation_selects_opposite_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            axis_path = Path(tmp) / "axis.json"
            axis_path.write_text(
                json.dumps(
                    {
                        "observation_kind": "real_stand_axis_without_qr",
                        "stand_axis_rad": 0.0,
                        "stand_center": {"x_m": 0.0, "y_m": 0.0},
                        "robot_pose": {
                            "x_m": 0.0,
                            "y_m": 0.7,
                            "yaw_rad": -math.pi / 2.0,
                        },
                    }
                )
            )
            selected = _opposite_face_normal(axis_path)
            error = math.atan2(
                math.sin(selected + math.pi / 2.0),
                math.cos(selected + math.pi / 2.0),
            )
            self.assertAlmostEqual(error, 0.0, places=6)

    def test_real_hardware_checkpoint_options_are_explicit(self):
        args = build_parser().parse_args(
            [
                "--robot-profile",
                "robot.json",
                "--camera-calibration",
                "camera.json",
                "--physical-site",
                "site.json",
                "--coverage-leg-limit",
                "1",
                "--stop-after-coverage",
                "--execute",
            ]
        )
        self.assertEqual(args.coverage_leg_limit, 1)
        self.assertTrue(args.stop_after_coverage)
        self.assertTrue(args.execute)

    def test_near_front_stuck_stop_is_eligible_for_stand_confirmation(self):
        outcome = MotionLegOutcome(
            run_id="blocked",
            status="stopped",
            stop_reason="stuck no progress",
            stop_details={
                "front_clearance": {"nearest_valid_range_m": 0.248},
            },
            motion_published=True,
            returncode=1,
            semantic_log_path=Path("events.jsonl"),
        )
        route_tube = MotionLegOutcome(
            **{
                **outcome.__dict__,
                "stop_reason": "pose left certified route tube",
            }
        )
        far_obstacle = MotionLegOutcome(
            **{
                **outcome.__dict__,
                "stop_details": {
                    "front_clearance": {"nearest_valid_range_m": 0.50},
                },
            }
        )

        self.assertTrue(_is_confirmable_stand_blockage(outcome))
        self.assertFalse(_is_confirmable_stand_blockage(route_tube))
        self.assertFalse(_is_confirmable_stand_blockage(far_obstacle))

    def test_coverage_leg_stops_observes_replans_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_route = root / "route.csv"
            source_diagnostics = root / "diagnostics.json"
            source_route.write_text("source\n")
            source_diagnostics.write_text("{}\n")
            replacement_route = root / "replacement_route.csv"
            replacement_diagnostics = root / "replacement_diagnostics.json"
            replacement_route.write_text("replacement\n")
            replacement_diagnostics.write_text("{}\n")
            blocked = MotionLegOutcome(
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason="stuck no progress",
                stop_details={
                    "front_clearance": {"nearest_valid_range_m": 0.248},
                },
                motion_published=True,
                returncode=1,
                semantic_log_path=root / "blocked.jsonl",
            )
            completed = MotionLegOutcome(
                run_id="mission_coverage_000_replan_001",
                status="completed",
                stop_reason="",
                stop_details={},
                motion_published=True,
                returncode=0,
                semantic_log_path=root / "completed.jsonl",
            )
            args = SimpleNamespace(
                session_id="mission",
                max_blockage_replans_per_leg=2,
                map=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
            )
            profile = SimpleNamespace(robot_radius_m=0.105)
            sealed = {
                "route_csv": str(source_route),
                "diagnostics_json": str(source_diagnostics),
                "route_certificate_json": str(root / "certificate.json"),
            }
            with patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "seal_stand_discovery_route",
                return_value=sealed,
            ) as seal, patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "_run_motion_leg",
                side_effect=(blocked, completed),
            ) as run, patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "_capture_lidar_epoch",
                return_value=root / "observer_summary.json",
            ) as observe, patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "record_blockage_replan",
                return_value={
                    "route_csv": str(replacement_route),
                    "diagnostics_json": str(replacement_diagnostics),
                    "summary_json": str(root / "replan_summary.json"),
                    "blockage_epoch_json": str(root / "epoch.json"),
                },
            ) as replan:
                _execute_coverage_leg_with_replans(
                    profile=profile,
                    args=args,
                    session_root=root / "session",
                    survey_root=root / "survey",
                    plan_path=root / "coverage_plan.json",
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                    source_route=source_route,
                    source_diagnostics=source_diagnostics,
                )

            self.assertEqual(run.call_count, 2)
            self.assertEqual(seal.call_count, 2)
            observe.assert_called_once()
            replan.assert_called_once()
            second_seal = seal.call_args_list[1].kwargs
            self.assertEqual(second_seal["source_route_csv"], replacement_route)
            self.assertEqual(
                second_seal["source_diagnostics_json"],
                replacement_diagnostics,
            )

    def test_opposite_inspection_offsets_never_cross_physical_minimum(self):
        offsets = _bounded_approach_offsets(0.70, 0.32)
        self.assertEqual(offsets[0], 0.70)
        self.assertEqual(offsets[-1], 0.32)
        self.assertTrue(all(value >= 0.32 for value in offsets))
        self.assertEqual(tuple(sorted(offsets, reverse=True)), offsets)

    def test_opposite_face_route_is_bound_to_axis_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            survey_root = root / "survey"
            self._planned_center_corridor(survey_root)
            plan = load_coverage_survey_plan(
                survey_root / "coverage_plan.json"
            )
            registry_path = survey_root / "stand_registry.json"
            registry = load_stand_survey_registry(registry_path, plan)
            candidate = SurveyCandidate(
                candidate_uid="survey_candidate_0001",
                x_m=0.2,
                y_m=0.0,
                radius_m=0.06,
                uncertainty_m=0.02,
                keepout_radius_m=0.31,
                confidence=0.9,
                hit_count=5,
                first_seen_sec=10.0,
                last_seen_sec=12.0,
                source_observation_ids=("obs_1", "obs_2"),
                viewpoint_ids=("survey_vp_001", "survey_vp_002"),
                status=STATUS_PENDING_CAMERA,
            )
            registry = type(registry)(
                schema_version=registry.schema_version,
                survey_id=registry.survey_id,
                planning_frame=registry.planning_frame,
                map_bundle_sha256=registry.map_bundle_sha256,
                candidates=(candidate,),
            )
            write_stand_survey_registry(registry_path, registry, plan)
            snapshot = candidate_snapshot_from_registry(
                registry,
                plan,
                registry_path=registry_path,
                snapshot_id="opposite_route_snapshot",
            )
            snapshot_path = root / "candidate_snapshot.json"
            write_candidate_snapshot(snapshot_path, snapshot)
            axis_path = root / "axis_observation.json"
            axis_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "observation_kind": "real_stand_axis_without_qr",
                        "stand_id": candidate.candidate_uid,
                        "planning_frame": plan.planning_frame,
                        "stand_center": {"x_m": 0.2, "y_m": 0.0},
                        "robot_pose": {
                            "x_m": 0.2,
                            "y_m": 0.55,
                            "yaw_rad": -math.pi / 2.0,
                        },
                        "stand_axis_rad": 0.0,
                    }
                )
            )
            outputs = plan_candidate_preapproach(
                map_yaml=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                plan=plan,
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                candidate_uid=candidate.candidate_uid,
                start=Pose2D(0.2, 0.55, -math.pi / 2.0),
                output_dir=root / "opposite_route",
                approach_offset_m=0.55,
                inflation_radius_m=0.25,
                candidate_transit_radius_m=0.31,
                physical_clearance={
                    "minimum_active_standoff_m": 0.26,
                    "minimum_candidate_transit_radius_m": 0.31,
                    "minimum_static_inflation_m": 0.25,
                },
                approach_normal_rad=-math.pi / 2.0,
                axis_observation_path=axis_path,
            )
            diagnostics = json.loads(
                Path(outputs["diagnostics_json"]).read_text()
            )
            self.assertEqual(
                diagnostics["metadata"]["approach_bearing_mode"],
                "camera-axis-face",
            )
            self.assertTrue(Path(outputs["route_certificate_json"]).exists())


if __name__ == "__main__":
    unittest.main()
