import argparse
import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.aufgabe04.simulation.run_arrival_pose_survey import (
    _candidate_start_pose,
    _catalog_provenance,
    _load_candidates,
    _load_snapshot_candidates,
    _observer_command,
    _planner_command,
    _runner_command,
    _survey_config_payload,
    _survey_one,
    _survey_completion_available,
    _survey_stream_id,
    _validate_heading_contract,
    _validate_target_distance,
    _wait_for_route,
    build_parser,
    main,
    SurveyCandidate,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.route_revision_store import RouteRevisionStore
from scripts.aufgabe04.navigation.viewpoint_sampling_contract import (
    VIEWPOINT_SAMPLING_CONTRACT_NAME,
    VIEWPOINT_SAMPLING_CONTRACT_VERSION,
)
from scripts.aufgabe04.stations.arrival_pose_catalog import (
    new_arrival_pose_catalog,
    write_arrival_pose_catalog,
)
from scripts.aufgabe04.stations.arrival_pose_models import CatalogProvenance
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    candidate_snapshot_sha256,
    new_candidate_snapshot,
)
from scripts.aufgabe04.stations.station_identity_registry import (
    StationIdentity,
    new_station_identity_registry,
)


class RunArrivalPoseSurveyTest(unittest.TestCase):
    def test_snapshot_geometry_is_passed_exactly_to_observer(self):
        frozen = FrozenCandidate(
            candidate_uid="candidate_a",
            geometry=CandidateGeometry(1.0, 2.0, 0.09, 0.045, 0.36),
            source=CandidateSource(
                "lidar/stand_confirmation",
                "1" * 64,
                "2" * 64,
                ("observation_a",),
            ),
            confidence=0.95,
            hit_count=8,
            first_seen_sec=10.0,
            last_seen_sec=11.0,
        )
        snapshot = new_candidate_snapshot(
            snapshot_id="candidate_snapshot_001",
            created_unix_sec=12.0,
            planning_frame="odom",
            map_bundle_sha256="3" * 64,
            candidates=(frozen,),
        )
        registry = new_station_identity_registry(
            registry_id="station_registry_001",
            created_unix_sec=13.0,
            candidate_snapshot_sha256=candidate_snapshot_sha256(snapshot),
            source_artifact_sha256="4" * 64,
            expected_candidate_uids=("candidate_a",),
            mappings=(StationIdentity("candidate_a", "A", "station_A"),),
        )
        candidate = _load_snapshot_candidates(snapshot, registry)[0]
        args = build_parser().parse_args(
            [
                "--candidate-snapshot", "snapshot.json",
                "--map", "map.yaml",
                "--world", "world.world",
                "--output-dir", "survey",
                "--catalog", "catalog.json",
                "--session-id", "session_001",
            ]
        )

        command = _observer_command(
            args, candidate, Path("survey/candidate_a"), "survey-a"
        )

        self.assertEqual(candidate.keepout_radius_m, 0.36)
        self.assertEqual(command[command.index("--stand-radius-m") + 1], "0.09")
        self.assertEqual(
            command[command.index("--stand-uncertainty-m") + 1], "0.045"
        )
        self.assertEqual(
            command[command.index("--target-distance-m") + 1], "0.33"
        )
        self.assertEqual(
            command[command.index("--sampling-arrival-tolerance-m") + 1],
            "0.017",
        )
        self.assertEqual(
            command[command.index("--tangential-correction-gain") + 1],
            "0.5",
        )

    def test_commands_are_simulation_only_and_planner_uses_survey_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidates_path = root / "candidates.json"
            candidates_path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_uid": "candidate_a",
                                "stand_id": "A",
                                "x_m": 1.0,
                                "y_m": 2.0,
                            },
                            {
                                "candidate_uid": "candidate_b",
                                "stand_id": "B",
                                "x_m": 2.0,
                                "y_m": 1.0,
                            },
                        ]
                    }
                )
            )
            candidates = _load_candidates(candidates_path)
            args = build_parser().parse_args(
                [
                    "--candidates-json", str(candidates_path),
                    "--map", str(root / "map.yaml"),
                    "--world", str(root / "world.world"),
                    "--output-dir", str(root / "survey"),
                    "--catalog", str(root / "catalog.json"),
                    "--session-id", "session_001",
                ]
            )
            output = root / "survey" / "candidate_a"

            observer = _observer_command(args, candidates[0], output, "survey-a")
            planner = _planner_command(
                args, candidates[0], candidates, output, "survey-a", "a" * 64
            )
            runner = _runner_command(args, candidates[0], output)

        self.assertIn("sim_synchronized_viewpoint_node.py", observer[1])
        self.assertEqual(
            observer[observer.index("--target-distance-m") + 1],
            "0.33",
        )
        self.assertEqual(_survey_config_payload(args)["target_distance_m"], 0.33)
        self.assertEqual(
            _survey_config_payload(args)["sampling_arrival_tolerance_m"],
            0.017,
        )
        self.assertEqual(
            observer[
                observer.index(
                    "--viewpoint-sampling-terminal-heading-hold-tolerance-m"
                )
                + 1
            ],
            "0.02",
        )
        self.assertEqual(
            observer[
                observer.index(
                    "--viewpoint-sampling-terminal-heading-target-envelope-radius-m"
                )
                + 1
            ],
            "0.03",
        )
        self.assertEqual(
            _survey_config_payload(args)["tangential_correction_gain"],
            0.5,
        )
        self.assertEqual(
            observer[observer.index("--max-center-error-deg") + 1],
            "12.0",
        )
        self.assertEqual(
            observer[observer.index("--max-tangential-step-deg") + 1],
            "20.0",
        )
        self.assertEqual(
            planner[planner.index("--target-yaw-threshold-deg") + 1],
            "4.0",
        )
        self.assertEqual(
            _survey_config_payload(args)[
                "observer_max_center_error_deg"
            ],
            12.0,
        )
        self.assertEqual(
            _survey_config_payload(args)[
                "observer_max_tangential_step_deg"
            ],
            20.0,
        )
        self.assertEqual(
            _survey_config_payload(args)[
                "planner_target_yaw_threshold_deg"
            ],
            4.0,
        )
        self.assertEqual(
            _survey_config_payload(args)["viewpoint_sampling_contract_name"],
            VIEWPOINT_SAMPLING_CONTRACT_NAME,
        )
        self.assertEqual(
            _survey_config_payload(args)["viewpoint_sampling_contract_version"],
            VIEWPOINT_SAMPLING_CONTRACT_VERSION,
        )
        self.assertRegex(
            _survey_config_payload(args)[
                "viewpoint_sampling_contract_source_sha256"
            ],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(
            _survey_config_payload(args)[
                "viewpoint_sampling_terminal_heading_target_envelope_radius_m"
            ],
            0.03,
        )
        self.assertEqual(
            _survey_config_payload(args)[
                "viewpoint_sampling_terminal_heading_minimum_stand_distance_m"
            ],
            0.31,
        )
        self.assertAlmostEqual(
            _survey_config_payload(args)[
                "viewpoint_sampling_terminal_heading_maximum_stand_distance_m"
            ],
            0.35,
        )
        self.assertIn("--workflow-mode", planner)
        self.assertEqual(planner[planner.index("--workflow-mode") + 1], "survey-only")
        self.assertIn("--arrival-pose-catalog", planner)
        self.assertEqual(
            planner[planner.index("--lidar-clearance-margin-m") + 1],
            "0.02",
        )
        self.assertEqual(planner.count("--expected-candidate-uid"), 2)
        self.assertEqual(planner.count("--known-stand-keepout"), 1)
        observer_feedback_path = observer[
            observer.index("--axis-acquisition-feedback-json") + 1
        ]
        planner_feedback_path = planner[
            planner.index("--axis-acquisition-feedback-json") + 1
        ]
        self.assertEqual(observer_feedback_path, planner_feedback_path)
        self.assertEqual(
            Path(observer_feedback_path).name,
            "axis_acquisition_feedback.json",
        )
        self.assertNotIn("--axis-acquisition-feedback-json", runner)
        self.assertEqual(
            observer[
                observer.index("--axis-acquisition-arrival-tolerance-m") + 1
            ],
            "0.1",
        )
        self.assertEqual(
            planner[
                planner.index("--axis-acquisition-arrival-tolerance-m") + 1
            ],
            "0.1",
        )
        frozen_feedback = _survey_config_payload(args)
        self.assertEqual(
            frozen_feedback["axis_acquisition_feedback_scope"],
            "per_candidate_observer_planner_sidecar",
        )
        self.assertFalse(
            frozen_feedback["axis_acquisition_feedback_is_motion_input"]
        )
        self.assertEqual(
            frozen_feedback["axis_acquisition_feedback_schema_version"],
            1,
        )
        self.assertNotIn(
            "axis_acquisition_feedback_path",
            frozen_feedback,
        )
        keepout_indexes = [
            index
            for index, value in enumerate(planner)
            if value == "--known-stand-keepout"
        ]
        self.assertEqual(
            [planner[index + 1 : index + 4] for index in keepout_indexes],
            [["2.0", "1.0", "0.26"]],
        )
        self.assertIn("--allow-sim-time", runner)
        self.assertEqual(
            runner[runner.index("--run-id") + 1],
            "survey_session_001_candidate_a",
        )
        self.assertIn("--preflight-observation-window-sec", runner)
        self.assertEqual(
            runner[runner.index("--preflight-observation-window-sec") + 1],
            "6.0",
        )
        self.assertIn("--initial-sensor-wait-sec", runner)
        self.assertEqual(
            runner[runner.index("--viewpoint-sampling-timeout-sec") + 1],
            "180.0",
        )
        self.assertEqual(
            runner[
                runner.index("--viewpoint-sampling-target-timeout-sec") + 1
            ],
            "60.0",
        )
        self.assertEqual(
            runner[
                runner.index("--viewpoint-sampling-goal-tolerance-m") + 1
            ],
            "0.017",
        )
        self.assertEqual(
            runner[
                runner.index(
                    "--viewpoint-sampling-terminal-heading-hold-tolerance-m"
                )
                + 1
            ],
            "0.02",
        )
        self.assertEqual(
            runner[
                runner.index("--viewpoint-sampling-target-distance-m") + 1
            ],
            "0.33",
        )
        self.assertEqual(
            runner[
                runner.index(
                    "--viewpoint-sampling-terminal-heading-target-envelope-radius-m"
                )
                + 1
            ],
            "0.03",
        )
        self.assertIn("--dynamic-route-refresh-sec", runner)

    def test_simulation_odom_after_stale_tf_is_disabled_by_default(self):
        args = build_parser().parse_args(
            [
                "--candidates-json", "candidates.json",
                "--map", "map.yaml",
                "--world", "world.world",
                "--output-dir", "survey",
                "--catalog", "catalog.json",
                "--session-id", "session_001",
            ]
        )
        candidate = SurveyCandidate("candidate_a", "A", 1.0, 2.0)

        runner = _runner_command(
            args,
            candidate,
            Path("survey/candidate_a"),
        )

        self.assertFalse(args.allow_simulation_odom_after_stale_tf)
        self.assertIs(
            _survey_config_payload(args)[
                "allow_simulation_odom_after_stale_tf"
            ],
            False,
        )
        self.assertNotIn(
            "--allow-simulation-odom-after-stale-tf",
            runner,
        )

    def test_simulation_odom_after_stale_tf_opt_in_is_frozen_and_forwarded_once(
        self,
    ):
        args = build_parser().parse_args(
            [
                "--candidates-json", "candidates.json",
                "--map", "map.yaml",
                "--world", "world.world",
                "--output-dir", "survey",
                "--catalog", "catalog.json",
                "--session-id", "session_001",
                "--allow-simulation-odom-after-stale-tf",
            ]
        )
        candidate = SurveyCandidate("candidate_a", "A", 1.0, 2.0)

        runner = _runner_command(
            args,
            candidate,
            Path("survey/candidate_a"),
        )

        self.assertTrue(args.allow_simulation_odom_after_stale_tf)
        self.assertIs(
            _survey_config_payload(args)[
                "allow_simulation_odom_after_stale_tf"
            ],
            True,
        )
        self.assertEqual(
            runner.count("--allow-simulation-odom-after-stale-tf"),
            1,
        )

    def test_explicit_observer_and_planner_heading_contract_is_forwarded(self):
        args = build_parser().parse_args(
            [
                "--candidates-json", "candidates.json",
                "--map", "map.yaml",
                "--world", "world.world",
                "--output-dir", "survey",
                "--catalog", "catalog.json",
                "--session-id", "session_001",
                "--observer-max-center-error-deg", "14",
                "--observer-max-tangential-step-deg", "18",
                "--planner-target-yaw-threshold-deg", "3",
            ]
        )
        candidate = SurveyCandidate("candidate_a", "A", 1.0, 2.0)

        _validate_heading_contract(args)
        observer = _observer_command(
            args,
            candidate,
            Path("survey/candidate_a"),
            "survey-a",
        )
        planner = _planner_command(
            args,
            candidate,
            (candidate,),
            Path("survey/candidate_a"),
            "survey-a",
            "a" * 64,
        )
        frozen = _survey_config_payload(args)

        self.assertEqual(
            observer[observer.index("--max-center-error-deg") + 1],
            "14.0",
        )
        self.assertEqual(
            observer[observer.index("--max-tangential-step-deg") + 1],
            "18.0",
        )
        self.assertEqual(
            planner[planner.index("--target-yaw-threshold-deg") + 1],
            "3.0",
        )
        self.assertEqual(frozen["observer_max_center_error_deg"], 14.0)
        self.assertEqual(
            frozen["observer_max_tangential_step_deg"],
            18.0,
        )
        self.assertEqual(frozen["planner_target_yaw_threshold_deg"], 3.0)

    def test_heading_contract_fails_closed_on_invalid_thresholds(self):
        common = [
            "--candidates-json", "candidates.json",
            "--map", "map.yaml",
            "--world", "world.world",
            "--output-dir", "survey",
            "--catalog", "catalog.json",
            "--session-id", "session_001",
        ]
        valid = build_parser().parse_args(
            common + ["--planner-target-yaw-threshold-deg", "6.999"]
        )
        _validate_heading_contract(valid)

        center_not_above_follower = build_parser().parse_args(
            common + ["--observer-max-center-error-deg", "5"]
        )
        with self.assertRaisesRegex(
            ValueError,
            "observer_max_center_error_deg must be strictly greater",
        ):
            _validate_heading_contract(center_not_above_follower)

        for value in ("7", "8"):
            args = build_parser().parse_args(
                common
                + ["--planner-target-yaw-threshold-deg", value]
            )
            with self.subTest(planner_threshold=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "planner_target_yaw_threshold_deg must be strictly less",
                ):
                    _validate_heading_contract(args)

        for option, field in (
            (
                "--observer-max-center-error-deg",
                "observer_max_center_error_deg",
            ),
            (
                "--observer-max-tangential-step-deg",
                "observer_max_tangential_step_deg",
            ),
            (
                "--planner-target-yaw-threshold-deg",
                "planner_target_yaw_threshold_deg",
            ),
        ):
            for value in ("0", "-1", "nan", "inf"):
                args = build_parser().parse_args(
                    common + [option, value]
                )
                with self.subTest(option=option, value=value):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{field} must be finite and positive",
                    ):
                        _validate_heading_contract(args)

    def test_explicit_lidar_clearance_margin_is_forwarded_and_bound(self):
        args = build_parser().parse_args(
            [
                "--candidates-json", "candidates.json",
                "--map", "map.yaml",
                "--world", "world.world",
                "--output-dir", "survey",
                "--catalog", "catalog.json",
                "--session-id", "session_001",
                "--lidar-clearance-margin-m", "0.035",
                "--arena-length-m", "4.2",
                "--arena-width-m", "2.1",
                "--arena-center-x-m", "0.1",
                "--arena-center-y-m", "-0.2",
                "--arena-yaw-deg", "3.0",
                "--arena-margin-m", "0.01",
            ]
        )
        candidate = SurveyCandidate("candidate_a", "A", 1.0, 2.0)

        planner = _planner_command(
            args,
            candidate,
            (candidate,),
            Path("survey/candidate_a"),
            "survey-a",
            "a" * 64,
        )

        self.assertEqual(
            planner[planner.index("--lidar-clearance-margin-m") + 1],
            "0.035",
        )
        self.assertEqual(
            _survey_config_payload(args)["lidar_clearance_margin_m"],
            0.035,
        )
        self.assertEqual(
            _survey_config_payload(args)["arena_bounds"],
            {
                "length_m": 4.2,
                "width_m": 2.1,
                "center_x_m": 0.1,
                "center_y_m": -0.2,
                "yaw_deg": 3.0,
                "margin_m": 0.01,
            },
        )
        for option, value in (
            ("--arena-length-m", "4.2"),
            ("--arena-width-m", "2.1"),
            ("--arena-center-x-m", "0.1"),
            ("--arena-center-y-m", "-0.2"),
            ("--arena-yaw-deg", "3.0"),
            ("--arena-margin-m", "0.01"),
        ):
            self.assertEqual(planner[planner.index(option) + 1], value)

    def test_explicit_target_distance_is_persisted_and_forwarded(self):
        args = build_parser().parse_args(
            [
                "--candidates-json", "candidates.json",
                "--map", "map.yaml",
                "--world", "world.world",
                "--output-dir", "survey",
                "--catalog", "catalog.json",
                "--session-id", "session_001",
                "--target-distance-m", "0.34",
                "--viewpoint-sampling-goal-tolerance-m", "0.01",
            ]
        )
        candidate = SurveyCandidate("candidate_a", "A", 1.0, 2.0)

        observer = _observer_command(
            args,
            candidate,
            Path("survey/candidate_a"),
            "survey-a",
        )

        self.assertEqual(
            observer[observer.index("--target-distance-m") + 1],
            "0.34",
        )
        self.assertEqual(_survey_config_payload(args)["target_distance_m"], 0.34)

    def test_target_envelope_accepts_exact_planner_and_observer_boundaries(self):
        candidate = SurveyCandidate("candidate_a", "A", 1.0, 2.0)
        common = [
            "--candidates-json", "candidates.json",
            "--map", "map.yaml",
            "--world", "world.world",
            "--output-dir", "survey",
            "--catalog", "catalog.json",
            "--session-id", "session_001",
        ]

        default_boundary = build_parser().parse_args(common)
        lower_boundary = build_parser().parse_args(
            common
            + [
                "--target-distance-m", "0.32",
                "--viewpoint-sampling-goal-tolerance-m", "0.01",
                "--sampling-arrival-tolerance-m", "0.01",
                "--viewpoint-sampling-terminal-heading-hold-tolerance-m",
                "0.01",
            ]
        )
        upper_boundary = build_parser().parse_args(
            common
            + [
                "--target-distance-m", "0.34",
                "--viewpoint-sampling-goal-tolerance-m", "0.01",
                "--sampling-arrival-tolerance-m", "0.01",
                "--viewpoint-sampling-terminal-heading-hold-tolerance-m",
                "0.01",
            ]
        )

        self.assertEqual(
            default_boundary.viewpoint_sampling_goal_tolerance_m,
            0.017,
        )
        self.assertEqual(
            default_boundary.sampling_arrival_tolerance_m,
            default_boundary.viewpoint_sampling_goal_tolerance_m,
        )
        self.assertEqual(
            default_boundary
            .viewpoint_sampling_terminal_heading_hold_tolerance_m,
            0.02,
        )
        self.assertEqual(
            default_boundary
            .viewpoint_sampling_terminal_heading_target_envelope_radius_m,
            0.03,
        )
        for args in (default_boundary, lower_boundary, upper_boundary):
            with self.subTest(
                target=args.target_distance_m,
                tolerance=args.viewpoint_sampling_goal_tolerance_m,
            ):
                _validate_target_distance(args, (candidate,))

    def test_target_envelope_rejects_each_cross_stage_boundary_violation(self):
        candidate = SurveyCandidate("candidate_a", "A", 1.0, 2.0)
        common = [
            "--candidates-json", "candidates.json",
            "--map", "map.yaml",
            "--world", "world.world",
            "--output-dir", "survey",
            "--catalog", "catalog.json",
            "--session-id", "session_001",
        ]
        below_planner = build_parser().parse_args(
            common
            + [
                "--target-distance-m", "0.32",
                "--viewpoint-sampling-goal-tolerance-m", "0.010001",
                "--sampling-arrival-tolerance-m", "0.010001",
                "--viewpoint-sampling-terminal-heading-hold-tolerance-m",
                "0.010001",
            ]
        )
        above_observer = build_parser().parse_args(
            common
            + [
                "--target-distance-m", "0.34",
                "--viewpoint-sampling-goal-tolerance-m", "0.010001",
                "--sampling-arrival-tolerance-m", "0.010001",
                "--viewpoint-sampling-terminal-heading-hold-tolerance-m",
                "0.010001",
            ]
        )
        observer_min_boundary = build_parser().parse_args(
            common
            + [
                "--target-distance-m", "0.30",
                "--viewpoint-sampling-goal-tolerance-m", "0.02",
                "--min-obstacle-distance-m", "0.01",
            ]
        )
        below_observer_min = build_parser().parse_args(
            common
            + [
                "--target-distance-m", "0.299999",
                "--viewpoint-sampling-goal-tolerance-m", "0.018",
                "--min-obstacle-distance-m", "0.01",
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            r"lower bound 0\.309989.*minimum_lidar_standoff_m=0\.310000",
        ):
            _validate_target_distance(below_planner, (candidate,))
        with self.assertRaisesRegex(
            ValueError,
            r"upper bound 0\.350011.*maximum distance 0\.350000",
        ):
            _validate_target_distance(above_observer, (candidate,))
        _validate_target_distance(observer_min_boundary, (candidate,))
        with self.assertRaisesRegex(
            ValueError,
            r"lower bound 0\.279989.*minimum distance 0\.280000",
        ):
            _validate_target_distance(below_observer_min, (candidate,))

    def test_sampling_arrival_tolerance_is_forwarded_frozen_and_bounded(self):
        candidate = SurveyCandidate("candidate_a", "A", 1.0, 2.0)
        common = [
            "--candidates-json", "candidates.json",
            "--map", "map.yaml",
            "--world", "world.world",
            "--output-dir", "survey",
            "--catalog", "catalog.json",
            "--session-id", "session_001",
        ]
        args = build_parser().parse_args(
            common
            + [
                "--viewpoint-sampling-goal-tolerance-m", "0.01",
                "--sampling-arrival-tolerance-m", "0.009",
                "--viewpoint-sampling-terminal-heading-hold-tolerance-m",
                "0.01",
                "--viewpoint-sampling-terminal-heading-target-envelope-radius-m",
                "0.025",
            ]
        )

        _validate_target_distance(args, (candidate,))
        observer = _observer_command(
            args,
            candidate,
            Path("survey/candidate_a"),
            "survey-a",
        )
        self.assertEqual(
            observer[observer.index("--sampling-arrival-tolerance-m") + 1],
            "0.009",
        )
        self.assertEqual(
            observer[
                observer.index(
                    "--viewpoint-sampling-terminal-heading-hold-tolerance-m"
                )
                + 1
            ],
            "0.01",
        )
        self.assertEqual(
            observer[
                observer.index(
                    "--viewpoint-sampling-terminal-heading-target-envelope-radius-m"
                )
                + 1
            ],
            "0.025",
        )
        self.assertEqual(
            _survey_config_payload(args)["sampling_arrival_tolerance_m"],
            0.009,
        )
        self.assertEqual(
            _survey_config_payload(args)[
                "viewpoint_sampling_terminal_heading_hold_tolerance_m"
            ],
            0.01,
        )
        self.assertEqual(
            _survey_config_payload(args)[
                "viewpoint_sampling_terminal_heading_target_envelope_radius_m"
            ],
            0.025,
        )

        for sampling_tolerance, goal_tolerance, effective_tolerance in (
            ("0.011", "0.01", "0.010000"),
            ("0.019", "0.02", "0.018000"),
        ):
            invalid = build_parser().parse_args(
                common
                + [
                    "--viewpoint-sampling-goal-tolerance-m",
                    goal_tolerance,
                    "--sampling-arrival-tolerance-m",
                    sampling_tolerance,
                ]
            )
            with self.subTest(
                sampling_tolerance=sampling_tolerance,
                goal_tolerance=goal_tolerance,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "effective follower entry tolerance "
                    + effective_tolerance,
                ):
                    _validate_target_distance(invalid, (candidate,))

    def test_sampling_arrival_tolerance_must_be_finite_and_positive(self):
        candidate = SurveyCandidate("candidate_a", "A", 1.0, 2.0)
        common = [
            "--candidates-json", "candidates.json",
            "--map", "map.yaml",
            "--world", "world.world",
            "--output-dir", "survey",
            "--catalog", "catalog.json",
            "--session-id", "session_001",
        ]
        for value in ("0", "-0.001", "nan", "inf"):
            args = build_parser().parse_args(
                common + ["--sampling-arrival-tolerance-m", value]
            )
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "sampling_arrival_tolerance_m must be finite and positive",
                ):
                    _validate_target_distance(args, (candidate,))

    def test_terminal_heading_target_envelope_is_bounded(self):
        candidate = SurveyCandidate("candidate_a", "A", 1.0, 2.0)
        common = [
            "--candidates-json", "candidates.json",
            "--map", "map.yaml",
            "--world", "world.world",
            "--output-dir", "survey",
            "--catalog", "catalog.json",
            "--session-id", "session_001",
        ]
        for value in ("0.016", "0.030001", "nan", "inf"):
            args = build_parser().parse_args(
                common
                + [
                    "--viewpoint-sampling-terminal-heading-target-envelope-radius-m",
                    value,
                ]
            )
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "target_envelope_radius_m",
                ):
                    _validate_target_distance(args, (candidate,))

    def test_incompatible_target_distance_fails_before_artifacts_or_processes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidates_path = root / "candidates.json"
            candidates_path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_uid": "candidate_a",
                                "stand_id": "A",
                                "x_m": 1.0,
                                "y_m": 2.0,
                            }
                        ]
                    }
                )
            )
            output_dir = root / "survey"
            map_bundle = MagicMock()
            map_bundle.bundle_sha256 = "a" * 64

            with patch(
                "scripts.aufgabe04.simulation.run_arrival_pose_survey."
                "load_occupancy_grid_with_bundle",
                return_value=(MagicMock(), map_bundle),
            ), patch(
                "scripts.aufgabe04.simulation.run_arrival_pose_survey."
                "write_frozen_map_bundle",
            ) as write_map_bundle, patch(
                "scripts.aufgabe04.simulation.run_arrival_pose_survey."
                "subprocess.Popen",
            ) as popen:
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "--candidates-json", str(candidates_path),
                            "--allow-legacy-candidate-json",
                            "--map", str(root / "map.yaml"),
                            "--world", str(root / "world.world"),
                            "--output-dir", str(output_dir),
                            "--catalog", str(root / "catalog.json"),
                            "--session-id", "session_001",
                            "--target-distance-m", "0.30",
                        ]
                    )

        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(output_dir.exists())
        write_map_bundle.assert_not_called()
        popen.assert_not_called()

    def test_planner_can_use_a_fresh_live_start_for_each_candidate(self):
        args = build_parser().parse_args(
            [
                "--candidates-json", "candidates.json",
                "--map", "map.yaml",
                "--world", "world.world",
                "--output-dir", "survey",
                "--catalog", "catalog.json",
                "--session-id", "session_001",
            ]
        )
        candidate = SurveyCandidate("candidate_a", "A", 1.0, 2.0)
        live_start = Pose2D(0.4, -0.2, 1.3)

        command = _planner_command(
            args,
            candidate,
            (candidate,),
            Path("survey/candidate_a"),
            "survey-a",
            "a" * 64,
            live_start,
        )

        self.assertIn("--start-x=0.4", command)
        self.assertIn("--start-y=-0.2", command)
        self.assertIn("--start-yaw=1.3", command)

    def test_planner_accepts_near_zero_negative_live_start_coordinates(self):
        args = build_parser().parse_args(
            [
                "--candidates-json", "candidates.json",
                "--map", "map.yaml",
                "--world", "world.world",
                "--output-dir", "survey",
                "--catalog", "catalog.json",
                "--session-id", "session_001",
            ]
        )
        candidate = SurveyCandidate("candidate_a", "A", 1.0, 2.0)
        live_start = Pose2D(-4.7e-06, -5.0e-06, -2.9e-05)

        command = _planner_command(
            args,
            candidate,
            (candidate,),
            Path("survey/candidate_a"),
            "survey-a",
            "a" * 64,
            live_start,
        )
        parser = argparse.ArgumentParser(add_help=False)
        for option in ("--start-x", "--start-y", "--start-yaw"):
            parser.add_argument(option, type=float, required=True)
        parsed, _ = parser.parse_known_args(command[2:])

        self.assertAlmostEqual(parsed.start_x, live_start.x_m)
        self.assertAlmostEqual(parsed.start_y, live_start.y_m)
        self.assertAlmostEqual(parsed.start_yaw, live_start.yaw_rad)

    def test_live_start_reader_is_used_only_when_requested(self):
        args = build_parser().parse_args(
            [
                "--candidates-json", "candidates.json",
                "--map", "map.yaml",
                "--world", "world.world",
                "--output-dir", "survey",
                "--catalog", "catalog.json",
                "--session-id", "session_001",
                "--initial-start-x", "1.0",
                "--initial-start-y", "2.0",
                "--initial-start-yaw", "0.5",
            ]
        )
        self.assertEqual(_candidate_start_pose(args), Pose2D(1.0, 2.0, 0.5))

        args.refresh_start_from_tf = True
        with patch(
            "scripts.aufgabe04.simulation.run_arrival_pose_survey."
            "read_current_tf_pose",
            return_value=Pose2D(3.0, 4.0, -0.2),
        ) as reader:
            self.assertEqual(
                _candidate_start_pose(args),
                Pose2D(3.0, 4.0, -0.2),
            )
        reader.assert_called_once_with(
            target_frame="odom",
            source_frame="base_footprint",
            timeout_sec=3.0,
            lookup_timeout_sec=0.2,
            use_sim_time=True,
        )

    def test_duplicate_candidate_uid_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.json"
            path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {"candidate_uid": "same", "x_m": 0, "y_m": 0},
                            {"candidate_uid": "same", "x_m": 1, "y_m": 1},
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                _load_candidates(path)

    def test_candidate_keepout_radius_override_is_passed_to_planner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidates_path = root / "candidates.json"
            candidates_path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_uid": "candidate_a",
                                "x_m": 1.0,
                                "y_m": 2.0,
                            },
                            {
                                "candidate_uid": "candidate_b",
                                "x_m": 2.0,
                                "y_m": 1.0,
                                "keepout_radius_m": 0.31,
                            }
                        ]
                    }
                )
            )
            candidates = _load_candidates(candidates_path)
            args = build_parser().parse_args(
                [
                    "--candidates-json", str(candidates_path),
                    "--map", str(root / "map.yaml"),
                    "--world", str(root / "world.world"),
                    "--output-dir", str(root / "survey"),
                    "--catalog", str(root / "catalog.json"),
                    "--session-id", "session_001",
                ]
            )

            planner = _planner_command(
                args,
                candidates[0],
                candidates,
                root / "survey" / "candidate_a",
                "survey-a",
                "a" * 64,
            )

        index = planner.index("--known-stand-keepout")
        self.assertEqual(planner[index + 1 : index + 4], ["2.0", "1.0", "0.31"])
        self.assertEqual(planner.count("--expected-candidate-uid"), 2)

    def test_stale_manifest_from_same_session_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "survey.manifest.json"
            stream_id = _survey_stream_id("session_001", "candidate_a")
            store = RouteRevisionStore(
                manifest,
                stream_id=stream_id,
                writer_id=f"planner-{stream_id}",
                now_fn=lambda: 100.0,
            )
            store.publish_active(
                "leg_index,point_index\n0,0\n",
                {"legs": []},
                target_revision=1,
                observation_unix_sec=100.0,
                source_robot_pose={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                target={"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0, "face_id": "near"},
                evidence={},
                previous_route_length_m=0.0,
                new_route_length_m=1.0,
                safety_diagnostics={},
            )
            with patch(
                "scripts.aufgabe04.simulation.run_arrival_pose_survey.time.sleep",
                return_value=None,
            ):
                with self.assertRaises(TimeoutError):
                    _wait_for_route(
                        manifest,
                        stream_id,
                        0.002,
                        not_before_unix_sec=101.0,
                    )

    def test_timeout_terminates_follower_gracefully_before_other_processes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidates_path = root / "candidates.json"
            candidates_path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_uid": "candidate_a",
                                "stand_id": "A",
                                "x_m": 1.0,
                                "y_m": 2.0,
                            }
                        ]
                    }
                )
            )
            args = build_parser().parse_args(
                [
                    "--candidates-json", str(candidates_path),
                    "--map", str(root / "map.yaml"),
                    "--world", str(root / "world.world"),
                    "--output-dir", str(root / "survey"),
                    "--catalog", str(root / "catalog.json"),
                    "--session-id", "session_001",
                    "--candidate-timeout-sec", "0.01",
                ]
            )
            candidate = _load_candidates(candidates_path)[0]
            provenance = CatalogProvenance(
                "odom", "a" * 64, "world", "b" * 64, "session_001", "simulation"
            )
            observer = MagicMock()
            planner = MagicMock()
            runner = MagicMock()
            observer.poll.return_value = None
            planner.poll.return_value = None
            runner.poll.return_value = None
            observer.wait.return_value = 0
            planner.wait.return_value = 0
            runner.wait.side_effect = [
                subprocess.TimeoutExpired("runner", 0.01),
                0,
            ]

            with patch(
                "scripts.aufgabe04.simulation.run_arrival_pose_survey.subprocess.Popen",
                side_effect=(observer, planner, runner),
            ), patch(
                "scripts.aufgabe04.simulation.run_arrival_pose_survey._wait_for_route",
                return_value="active",
            ):
                with self.assertRaises(subprocess.TimeoutExpired):
                    _survey_one(
                        args,
                        candidate,
                        (candidate,),
                        "b" * 64,
                        provenance,
                    )

        runner.terminate.assert_called_once_with()
        runner.kill.assert_not_called()
        planner.terminate.assert_called_once_with()
        observer.terminate.assert_called_once_with()

    def test_nonzero_follower_is_accepted_only_after_matching_terminal_handoff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidates_path = root / "candidates.json"
            candidates_path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_uid": "candidate_a",
                                "stand_id": "A",
                                "x_m": 1.0,
                                "y_m": 2.0,
                            }
                        ]
                    }
                )
            )
            args = build_parser().parse_args(
                [
                    "--candidates-json", str(candidates_path),
                    "--map", str(root / "map.yaml"),
                    "--world", str(root / "world.world"),
                    "--output-dir", str(root / "survey"),
                    "--catalog", str(root / "catalog.json"),
                    "--session-id", "session_001",
                ]
            )
            candidate = _load_candidates(candidates_path)[0]
            provenance = CatalogProvenance(
                "odom", "a" * 64, "world", "b" * 64, "session_001", "simulation"
            )
            observer = MagicMock()
            planner = MagicMock()
            runner = MagicMock()
            observer.poll.return_value = None
            planner.poll.return_value = None
            runner.poll.return_value = 2
            observer.wait.return_value = 0
            planner.wait.return_value = 0
            runner.wait.return_value = 2
            catalog = MagicMock()
            catalog.record_for.return_value = object()

            with patch(
                "scripts.aufgabe04.simulation.run_arrival_pose_survey.subprocess.Popen",
                side_effect=(observer, planner, runner),
            ), patch(
                "scripts.aufgabe04.simulation.run_arrival_pose_survey._wait_for_route",
                return_value="active",
            ), patch(
                "scripts.aufgabe04.simulation.run_arrival_pose_survey._survey_completion_available",
                return_value=True,
            ) as completed, patch(
                "scripts.aufgabe04.simulation.run_arrival_pose_survey.load_arrival_pose_catalog",
                return_value=catalog,
            ):
                _survey_one(
                    args,
                    candidate,
                    (candidate,),
                    "b" * 64,
                    provenance,
                )

        completed.assert_called_once()
        runner.terminate.assert_not_called()

    def test_completion_proof_rejects_wrong_catalog_provenance(self):
        provenance = CatalogProvenance(
            "odom", "a" * 64, "world", "b" * 64, "session_001", "simulation"
        )
        with patch(
            "scripts.aufgabe04.simulation.run_arrival_pose_survey.load_arrival_pose_catalog",
            side_effect=ValueError("provenance mismatch"),
        ) as loader, patch(
            "scripts.aufgabe04.simulation.run_arrival_pose_survey.read_route_revision"
        ) as reader:
            completed = _survey_completion_available(
                catalog_path=Path("catalog.json"),
                provenance=provenance,
                manifest_path=Path("manifest.json"),
                stream_id="survey-stream",
                candidate_uid="candidate_a",
            )

        self.assertFalse(completed)
        loader.assert_called_once_with(
            Path("catalog.json"), required_provenance=provenance
        )
        reader.assert_not_called()

    def test_main_rejects_catalog_from_another_session_before_spawning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidates = root / "candidates.json"
            candidates.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_uid": "candidate_a",
                                "stand_id": "A",
                                "x_m": 1.0,
                                "y_m": 2.0,
                            }
                        ]
                    }
                )
            )
            map_path = root / "map.yaml"
            world_path = root / "world.world"
            map_path.write_text("map")
            world_path.write_text("world")
            catalog_path = root / "catalog.json"
            map_hash = hashlib.sha256(map_path.read_bytes()).hexdigest()
            world_hash = hashlib.sha256(world_path.read_bytes()).hexdigest()
            catalog = new_arrival_pose_catalog(
                catalog_id="sim_arrival_survey",
                provenance=CatalogProvenance(
                    "odom",
                    map_hash,
                    world_path.stem,
                    world_hash,
                    "another_session",
                    "simulation",
                ),
                expected_candidate_uids=("candidate_a",),
                created_unix_sec=100.0,
            )
            write_arrival_pose_catalog(catalog_path, catalog)

            with patch(
                "scripts.aufgabe04.simulation.run_arrival_pose_survey.subprocess.Popen"
            ) as popen:
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "--candidates-json", str(candidates),
                            "--map", str(map_path),
                            "--world", str(world_path),
                            "--output-dir", str(root / "survey"),
                            "--catalog", str(catalog_path),
                            "--session-id", "session_001",
                        ]
                    )

        self.assertEqual(raised.exception.code, 2)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
