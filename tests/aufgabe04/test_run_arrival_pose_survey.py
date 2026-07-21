import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.aufgabe04.simulation.run_arrival_pose_survey import (
    _catalog_provenance,
    _load_candidates,
    _observer_command,
    _planner_command,
    _runner_command,
    _survey_one,
    _survey_completion_available,
    _survey_stream_id,
    _wait_for_route,
    build_parser,
    main,
)
from scripts.aufgabe04.navigation.route_revision_store import RouteRevisionStore
from scripts.aufgabe04.stations.arrival_pose_catalog import (
    new_arrival_pose_catalog,
    write_arrival_pose_catalog,
)
from scripts.aufgabe04.stations.arrival_pose_models import CatalogProvenance


class RunArrivalPoseSurveyTest(unittest.TestCase):
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
        self.assertIn("--workflow-mode", planner)
        self.assertEqual(planner[planner.index("--workflow-mode") + 1], "survey-only")
        self.assertIn("--arrival-pose-catalog", planner)
        self.assertEqual(planner.count("--expected-candidate-uid"), 2)
        self.assertEqual(planner.count("--known-stand-keepout"), 1)
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
        self.assertIn("--dynamic-route-refresh-sec", runner)

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
