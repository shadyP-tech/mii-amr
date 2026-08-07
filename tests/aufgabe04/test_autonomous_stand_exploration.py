import json
from contextlib import redirect_stdout
from dataclasses import replace
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
    _admit_preplanning_localization,
    _bounded_approach_offsets,
    _capture_lidar_epoch,
    _execute_coverage_leg_with_replans,
    _is_resealable_startup_mismatch,
    _motion_outcome_from_log,
    _opposite_face_normal,
    _replan_startup_source,
    _runner_command,
    _run_motion_leg,
    build_parser,
    candidate_snapshot_from_registry,
    plan_candidate_preapproach,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    write_candidate_snapshot,
)


MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")


class AutonomousStandExplorationTest(unittest.TestCase):
    @staticmethod
    def _profile():
        return SimpleNamespace(
            robot_id="turtlebot1",
            namespace="",
            scan_topic="scan",
            odom_topic="odom",
            cmd_vel_topic="cmd_vel",
            amcl_topic="amcl_pose",
            map_frame="map",
            odom_frame="odom",
            base_frame="base_footprint",
            localization_source="amcl",
            max_linear_speed_mps=0.055,
            max_angular_speed_radps=0.18,
            robot_radius_m=0.105,
        )

    def test_motion_outcome_preserves_preflight_failure_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            semantic_log = Path(tmp) / "events.jsonl"
            semantic_log.write_text(
                json.dumps(
                    {
                        "event": "preflight_failed",
                        "run_id": "coverage_000",
                        "failures": [
                            "stationary AMCL stability: position_std=0.0940m/0.0150m"
                        ],
                        "observations": [],
                        "runtime_config": {"localization_source": "amcl"},
                    }
                )
                + "\n"
            )

            outcome = _motion_outcome_from_log(
                semantic_log,
                run_id="coverage_000",
                returncode=1,
            )

        self.assertEqual(outcome.status, "preflight_failed")
        self.assertIn("position_std=0.0940m", outcome.stop_reason)
        self.assertFalse(outcome.motion_published)
        self.assertTrue(outcome.stop_details["fail_closed"])

    def test_preplanning_localization_binds_start_to_admitted_pose(self):
        runtime = SimpleNamespace(namespace="")
        preflight = RosPreflightResult(
            ok=True,
            failures=[],
            observations=[],
            runtime_config={"localization_source": "amcl"},
            route_pose={
                "frame_id": "map",
                "child_frame_id": "base_footprint",
                "x_m": -0.4,
                "y_m": -0.6,
                "yaw_rad": 1.7,
            },
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
            "run_ros_preflight",
            return_value=preflight,
        ) as run_preflight:
            root = Path(tmp)
            start = _admit_preplanning_localization(runtime, root)
            evidence = json.loads(
                (root / "preflight/preplanning_localization.json").read_text()
            )

        self.assertEqual(start, Pose2D(-0.4, -0.6, 1.7))
        self.assertTrue(evidence["ok"])
        self.assertEqual(
            run_preflight.call_args.kwargs[
                "max_stationary_amcl_position_std_m"
            ],
            0.30,
        )
        self.assertEqual(
            run_preflight.call_args.kwargs[
                "max_stationary_amcl_position_spread_m"
            ],
            0.015,
        )
        self.assertEqual(
            run_preflight.call_args.kwargs[
                "max_localization_tf_future_sec"
            ],
            1.1,
        )

    def test_preplanning_localization_fails_before_route_planning(self):
        runtime = SimpleNamespace(namespace="")
        preflight = RosPreflightResult(
            ok=False,
            failures=["stationary AMCL stability: position_std=0.049m/0.015m"],
            observations=[],
            runtime_config={"localization_source": "amcl"},
            route_pose=None,
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
            "run_ros_preflight",
            return_value=preflight,
        ):
            root = Path(tmp)
            with self.assertRaisesRegex(
                RuntimeError,
                "preplanning localization admission failed",
            ):
                _admit_preplanning_localization(runtime, root)
            evidence = json.loads(
                (root / "preflight/preplanning_localization.json").read_text()
            )

        self.assertFalse(evidence["ok"])
        self.assertIn("position_std=0.049m", evidence["failures"][0])

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
                    route_pose={
                        "frame_id": "map",
                        "child_frame_id": "base_footprint",
                        "x_m": leg.raw_waypoints[0].pose.x_m,
                        "y_m": leg.raw_waypoints[0].pose.y_m,
                        "yaw_rad": 0.0,
                    },
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

    def test_center_corridor_rejects_nonfinal_heading_constraints(self):
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
            intermediate = leg.raw_waypoints[1]
            malformed_waypoint = replace(
                intermediate,
                pose=Pose2D(
                    intermediate.pose.x_m,
                    intermediate.pose.y_m,
                    0.0,
                ),
            )
            malformed_leg = replace(
                leg,
                raw_waypoints=(
                    leg.raw_waypoints[0],
                    malformed_waypoint,
                    *leg.raw_waypoints[2:],
                ),
            )

            status = validate_stand_discovery_route_binding(
                Path(outputs["diagnostics_json"]),
                malformed_leg,
                coverage_plan_path=root / "coverage_plan.json",
            )

        self.assertFalse(status.ok)
        self.assertIn(
            "stand discovery non-final waypoint yaw must be unconstrained",
            status.failures,
        )

    def test_discovery_dry_run_rejects_stale_start_before_confirmation(self):
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
            events = Path(tmp) / "events.jsonl"
            with patch(
                "scripts.aufgabe04.navigation.run_single_station_segment."
                "run_ros_preflight",
                return_value=RosPreflightResult(
                    ok=True,
                    failures=[],
                    observations=[],
                    runtime_config={},
                    route_pose={
                        "frame_id": "map",
                        "child_frame_id": "base_footprint",
                        "x_m": leg.raw_waypoints[0].pose.x_m - 0.10,
                        "y_m": leg.raw_waypoints[0].pose.y_m,
                        "yaw_rad": 0.0,
                    },
                ),
            ), patch(
                "scripts.aufgabe04.navigation.run_single_station_segment."
                "run_simple_waypoint_follower"
            ) as follower, redirect_stdout(StringIO()):
                status = run_segment(
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
                        str(events),
                        "--results-csv",
                        str(Path(tmp) / "results.csv"),
                        "--preflight-json",
                        str(Path(tmp) / "preflight.json"),
                        "--dry-run",
                    ]
                )
            self.assertEqual(status, 1)
            follower.assert_not_called()
            payloads = [
                json.loads(line)
                for line in events.read_text().splitlines()
                if line.strip()
            ]
            rejected = next(
                item for item in payloads
                if item["event"] == "startup_route_rejected"
            )
            self.assertFalse(rejected["motion_published"])
            self.assertEqual(
                rejected["stop_details"]["phase"],
                "before_motion_confirmation",
            )

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
        self.assertEqual(args.max_startup_reseals_per_leg, 3)
        self.assertTrue(args.stop_after_coverage)
        self.assertTrue(args.execute)

    def test_coverage_runner_receives_persistent_replan_contract(self):
        profile = self._profile()
        command = _runner_command(
            profile=profile,
            route_csv=Path("route.csv"),
            diagnostics_json=Path("diagnostics.json"),
            certificate_json=Path("certificate.json"),
            run_id="mission_coverage_002",
            session_root=Path("session"),
            coverage_plan=Path("survey/coverage_plan.json"),
            coverage_transient_replan={
                "survey_root": Path("survey"),
                "session_root": Path("session"),
                "map_yaml": MAP,
                "semantic_map_id": "arena_1p898x3p9_auto",
                "target_viewpoint_id": "survey_vp_003",
                "robot_radius_m": 0.105,
                "max_replans": 3,
                "leg_index": 2,
            },
            dry_run=False,
        )

        self.assertIn("--coverage-transient-replan-leg-index", command)
        self.assertEqual(
            command[command.index("--coverage-transient-replan-leg-index") + 1],
            "2",
        )
        self.assertEqual(
            float(
                command[
                    command.index("--omnidirectional-hard-stop-distance-m")
                    + 1
                ]
            ),
            0.125,
        )

    def test_runner_command_enables_complete_uncertainty_aware_odom_contract(self):
        command = _runner_command(
            profile=self._profile(),
            route_csv=Path("route.csv"),
            diagnostics_json=Path("diagnostics.json"),
            certificate_json=Path("map_certificate.json"),
            run_id="mission_coverage_000",
            session_root=Path("session"),
            dry_run=True,
            uncertainty_map_yaml=MAP,
            localization_branch_proof_id="known_start_marker_20260807",
            odom_execution_certificate_json=Path("odom_certificate.json"),
            uncertainty_budget_json=Path("uncertainty_budget.json"),
        )

        self.assertEqual(
            command[command.index("--execution-pose-frame") + 1], "odom"
        )
        self.assertEqual(
            command[
                command.index("--localization-branch-proof-id") + 1
            ],
            "known_start_marker_20260807",
        )
        self.assertEqual(
            command[command.index("--uncertainty-robot-radius-m") + 1],
            "0.105",
        )
        self.assertEqual(
            command[command.index("--max-stationary-amcl-position-std-m") + 1],
            "0.30",
        )
        self.assertEqual(
            command[command.index("--preflight-json") + 1],
            "session/preflight/mission_coverage_000_dry.json",
        )

    def test_runner_command_rejects_partial_odom_contract(self):
        with self.assertRaisesRegex(ValueError, "must be complete"):
            _runner_command(
                profile=self._profile(),
                route_csv=Path("route.csv"),
                diagnostics_json=Path("diagnostics.json"),
                certificate_json=Path("map_certificate.json"),
                run_id="mission_coverage_000",
                session_root=Path("session"),
                dry_run=True,
                uncertainty_map_yaml=MAP,
            )

    def test_lidar_epoch_uses_completed_odom_execution_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            certificate = root / "execute_certificate.json"
            certificate.write_text("{}\n")
            args = SimpleNamespace(
                map=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                lidar_epoch_sec=1.0,
            )

            def write_summary(command, **_kwargs):
                summary = Path(command[command.index("--summary-json") + 1])
                summary.write_text(
                    json.dumps({"processed_scan_count": 1}) + "\n"
                )
                return SimpleNamespace(returncode=0)

            with patch(
                "scripts.aufgabe04.real_robot."
                "run_autonomous_stand_exploration.subprocess.run",
                side_effect=write_summary,
            ) as run:
                summary = _capture_lidar_epoch(
                    profile=self._profile(),
                    args=args,
                    survey_root=root / "survey",
                    viewpoint_id="survey_vp_001",
                    odom_execution_certificate_path=certificate,
                )

        command = run.call_args.args[0]
        self.assertEqual(
            command[
                command.index("--odom-execution-certificate-json") + 1
            ],
            str(certificate),
        )
        self.assertEqual(summary.name, "observer_summary.json")

    def test_coverage_leg_uses_one_runner_with_in_process_replanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_route = root / "route.csv"
            source_diagnostics = root / "diagnostics.json"
            source_route.write_text("source\n")
            source_diagnostics.write_text("{}\n")
            completed = MotionLegOutcome(
                run_id="mission_coverage_000",
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
                return_value=completed,
            ) as run, patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "_capture_lidar_epoch",
            ) as observe:
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

            self.assertEqual(run.call_count, 1)
            self.assertEqual(seal.call_count, 1)
            observe.assert_not_called()
            recovery = run.call_args.kwargs["coverage_transient_replan"]
            self.assertEqual(recovery["target_viewpoint_id"], "survey_vp_001")
            self.assertEqual(recovery["max_replans"], 2)
            self.assertEqual(recovery["robot_radius_m"], 0.105)

    def test_startup_mismatch_replans_and_requires_fresh_confirmation(self):
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
            rejected = MotionLegOutcome(
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason="pose outside certified startup segment",
                stop_details={
                    "source": "execution_route_certificate",
                    "phase": "before_motion_confirmation",
                    "route_pose": {
                        "frame_id": "map",
                        "child_frame_id": "base_footprint",
                        "x_m": -0.5044625,
                        "y_m": -0.6255536,
                        "yaw_rad": 1.7007652,
                    },
                },
                motion_published=False,
                returncode=1,
                semantic_log_path=root / "rejected.jsonl",
            )
            completed = MotionLegOutcome(
                run_id="mission_coverage_000_startup_reseal_001",
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
                max_startup_reseals_per_leg=2,
                map=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
            )
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
                side_effect=(rejected, completed),
            ) as run, patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "_replan_startup_source",
                return_value={
                    "route_csv": str(replacement_route),
                    "diagnostics_json": str(replacement_diagnostics),
                    "summary_json": str(root / "reseal_summary.json"),
                },
            ) as replan, patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "_capture_lidar_epoch"
            ) as observe:
                _execute_coverage_leg_with_replans(
                    profile=SimpleNamespace(robot_radius_m=0.105),
                    args=args,
                    session_root=root / "session",
                    survey_root=root / "survey",
                    plan_path=root / "coverage_plan.json",
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                    source_route=source_route,
                    source_diagnostics=source_diagnostics,
                )

            self.assertTrue(_is_resealable_startup_mismatch(rejected))
            self.assertEqual(run.call_count, 2)
            self.assertFalse(
                run.call_args_list[0].kwargs["require_fresh_confirmation"]
            )
            self.assertTrue(
                run.call_args_list[1].kwargs["require_fresh_confirmation"]
            )
            self.assertEqual(seal.call_count, 2)
            self.assertEqual(
                seal.call_args_list[1].kwargs["source_route_csv"],
                replacement_route,
            )
            replan.assert_called_once()
            observe.assert_not_called()
            adaptive = [
                json.loads(line)
                for line in (root / "session/adaptive_replans.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                adaptive[-1]["event"], "startup_pose_route_resealed"
            )
            self.assertTrue(adaptive[-1]["fresh_confirmation_required"])

    def test_runtime_blockage_failure_never_relaunches_the_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_route = root / "route.csv"
            source_diagnostics = root / "diagnostics.json"
            source_route.write_text("route\n")
            source_diagnostics.write_text("{}\n")
            blocked = MotionLegOutcome(
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason="stuck no progress",
                stop_details={
                    "front_clearance": {
                        "nearest_valid_range_m": 0.248,
                        "nearest_valid_bearing_rad": 0.0,
                        "source": "front_sector",
                    },
                },
                motion_published=True,
                returncode=1,
                semantic_log_path=root / "blocked.jsonl",
            )
            args = SimpleNamespace(
                session_id="mission",
                max_blockage_replans_per_leg=2,
                max_startup_reseals_per_leg=2,
                map=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
            )
            profile = SimpleNamespace(
                robot_radius_m=0.105,
                namespace="",
                amcl_topic="amcl_pose",
                map_frame="map",
            )
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
                return_value=blocked,
            ) as run:
                with self.assertRaisesRegex(RuntimeError, "stuck no progress"):
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

            self.assertEqual(run.call_count, 1)
            self.assertEqual(seal.call_count, 1)

    def test_dry_precheck_mismatch_is_returned_without_execution_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "startup_rejected"
            event_path = root / "run_events" / f"{run_id}.jsonl"
            event_path.parent.mkdir(parents=True)
            event_path.write_text(
                json.dumps(
                    {
                        "event": "safety_stop",
                        "run_id": run_id,
                        "status": "stopped",
                        "stop_reason": "pose outside certified startup segment",
                        "stop_details": {
                            "source": "execution_route_certificate",
                            "phase": "before_motion_confirmation",
                            "route_pose": {
                                "frame_id": "map",
                                "child_frame_id": "base_footprint",
                                "x_m": -0.5044625,
                                "y_m": -0.6255536,
                                "yaw_rad": 1.7007652,
                            },
                        },
                        "motion_published": False,
                    }
                )
                + "\n"
            )
            sealed = {
                "route_csv": str(root / "route.csv"),
                "diagnostics_json": str(root / "diagnostics.json"),
                "route_certificate_json": str(root / "certificate.json"),
            }
            with patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "subprocess.run",
                return_value=SimpleNamespace(returncode=1),
            ) as run:
                outcome = _run_motion_leg(
                    profile=self._profile(),
                    sealed=sealed,
                    run_id=run_id,
                    session_root=root,
                    execute=True,
                    coverage_plan=root / "coverage_plan.json",
                )

            self.assertTrue(_is_resealable_startup_mismatch(outcome))
            self.assertEqual(run.call_count, 1)

    def test_resealed_route_refusal_never_launches_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sealed = {
                "route_csv": str(root / "route.csv"),
                "diagnostics_json": str(root / "diagnostics.json"),
                "route_certificate_json": str(root / "certificate.json"),
            }
            with patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ) as run, patch("builtins.input", return_value="NO"), redirect_stdout(
                StringIO()
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "operator did not authorize resealed route",
                ):
                    _run_motion_leg(
                        profile=self._profile(),
                        sealed=sealed,
                        run_id="resealed",
                        session_root=root,
                        execute=True,
                        coverage_plan=root / "coverage_plan.json",
                        require_fresh_confirmation=True,
                    )

            self.assertEqual(run.call_count, 1)

    def test_live_leg_keeps_child_runner_typed_run_interactive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "interactive_live_leg"
            sealed = {
                "route_csv": str(root / "route.csv"),
                "diagnostics_json": str(root / "diagnostics.json"),
                "route_certificate_json": str(root / "certificate.json"),
            }
            calls = 0

            def run_process(_command, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    event_path = root / "run_events" / f"{run_id}.jsonl"
                    event_path.parent.mkdir(parents=True, exist_ok=True)
                    event_path.write_text(
                        json.dumps(
                            {
                                "event": "motion_completed",
                                "run_id": run_id,
                                "status": "completed",
                                "stop_reason": "",
                                "stop_details": {},
                                "motion_published": True,
                            }
                        )
                        + "\n"
                    )
                return SimpleNamespace(returncode=0)

            with patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "subprocess.run",
                side_effect=run_process,
            ) as run:
                outcome = _run_motion_leg(
                    profile=self._profile(),
                    sealed=sealed,
                    run_id=run_id,
                    session_root=root,
                    execute=True,
                )

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(run.call_count, 2)
        self.assertNotIn("input", run.call_args_list[1].kwargs)

    def test_startup_reseal_replans_full_a_star_leg_from_rejected_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            survey_root = root / "survey"
            self._planned_center_corridor(survey_root)
            plan_path = survey_root / "coverage_plan.json"
            summary = json.loads(
                (survey_root / "survey_summary.json").read_text()
            )
            current = Pose2D(0.04, 0.02, 0.1)
            rejected = MotionLegOutcome(
                run_id="rejected",
                status="stopped",
                stop_reason="pose outside certified startup segment",
                stop_details={
                    "source": "execution_route_certificate",
                    "phase": "before_motion_confirmation",
                    "route_pose": {
                        "frame_id": "map",
                        "child_frame_id": "base_footprint",
                        "x_m": current.x_m,
                        "y_m": current.y_m,
                        "yaw_rad": current.yaw_rad,
                    },
                },
                motion_published=False,
                returncode=1,
                semantic_log_path=root / "events.jsonl",
            )
            outputs = _replan_startup_source(
                map_yaml=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                survey_root=survey_root,
                plan_path=plan_path,
                expected_target_viewpoint_id=summary["next_viewpoint_id"],
                current_pose=current,
                rejected_outcome=rejected,
                reseal_index=1,
                output_dir=root / "reseal_source",
            )
            leg = load_route_leg(Path(outputs["route_csv"]), 0)
            self.assertAlmostEqual(leg.raw_waypoints[0].pose.x_m, current.x_m)
            self.assertAlmostEqual(leg.raw_waypoints[0].pose.y_m, current.y_m)
            diagnostics = json.loads(
                Path(outputs["diagnostics_json"]).read_text()
            )
            self.assertTrue(diagnostics["metadata"]["startup_reseal"])
            self.assertTrue(
                diagnostics["metadata"]["exact_start_connector"]["validated"]
            )
            sealed = seal_stand_discovery_route(
                source_route_csv=Path(outputs["route_csv"]),
                source_diagnostics_json=Path(outputs["diagnostics_json"]),
                coverage_plan_path=plan_path,
                output_dir=root / "sealed_reseal",
            )
            self.assertTrue(Path(sealed["route_certificate_json"]).is_file())

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
