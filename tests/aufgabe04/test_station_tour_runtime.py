"""Standalone entrypoint boundaries: artifact preview, RUN and navigation only."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.mission import station_tour_runtime as runtime


@dataclass(frozen=True)
class TourConfig:
    semantic_map_id: str = "arena"
    localization_branch_proof_id: str = "branch"
    mission_leg_motion_authorization_json: Path = Path("old-exploration-authorization")
    plan: object = field(default_factory=lambda: SimpleNamespace(map_bundle_sha256="a"*64))
    inflation_radius_m: float = .25
    physical_clearance: dict = field(default_factory=lambda: {"minimum_static_inflation_m": .25})


class StationTourRuntimeTest(unittest.TestCase):
    def session(self):
        saved = SimpleNamespace(candidate_uid="candidate", pose=Pose2D(1., 2., .3),
            source_frame=SimpleNamespace(to_evidence=lambda: {"map_frame": "map"}),
            evidence={"pose_kind": "qr_verified_observation_pose"})
        return SimpleNamespace(poses_by_qr={qr: saved for qr in (
            "Start", "QR_001", "QR_002", "QR_003", "QR_004")},
            profile=SimpleNamespace(robot_radius_m=.105), config=TourConfig())

    def arguments(self, root):
        return ["--exploration-session", str(root / "camera"),
                "--robot-profile", str(root / "profile.json"),
                "--server-robot-id", "team_01", "--tour-id", "tour_test",
                "--output-root", str(root / "tours")]

    def test_preview_authenticates_artifacts_without_ros_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("scripts.aufgabe04.real_robot.mission.stored_pose_session.load_stored_pose_session",
                       return_value=self.session()) as load, \
                 patch.object(runtime, "execute_tour") as execute, \
                 patch.object(runtime, "build_navigation_effects") as effects, \
                 patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(runtime.main(self.arguments(root)), 0)
            load.assert_called_once_with((root / "camera").resolve(), (root / "profile.json").resolve())
            execute.assert_not_called()
            effects.assert_not_called()
            inputs = json.loads((root / "tours/tour_test/inputs.json").read_text())
            self.assertEqual(inputs["randomize_request"], {"qr_count": 4, "stations": 3})
            self.assertTrue(inputs["cover_all_stands"])
            self.assertFalse(inputs["physical_cargo_actions"])
            self.assertEqual(inputs["initial_start_policy"], "verify_camera_return")
            self.assertEqual(inputs["clearance_policy"]["robot_radius_m"], .105)
            self.assertEqual(inputs["clearance_policy"]["initial_planning_inflation_m"], .25)
            self.assertTrue(inputs["clearance_policy"]["uncertainty_and_braking_reserves_retained"])

    def test_execute_requires_unloaded_and_continuous_odom_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            for extra in ([], ["--confirm-unloaded"], ["--confirm-odom-continuity"]):
                with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    runtime.main(self.arguments(Path(directory)) + ["--execute", *extra])
                self.assertEqual(caught.exception.code, 2)

    def test_existing_run_is_never_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "tours/tour_test"
            previous.mkdir(parents=True)
            (previous / "inputs.json").write_text("previous")
            with patch("scripts.aufgabe04.real_robot.mission.stored_pose_session.load_stored_pose_session",
                       return_value=self.session()), redirect_stdout(io.StringIO()):
                self.assertEqual(runtime.main(self.arguments(root)), 2)
            self.assertEqual([p.name for p in previous.iterdir()], ["inputs.json"])
            self.assertEqual((previous / "inputs.json").read_text(), "previous")

    def test_bundle_incompatible_tour_id_rejected_before_effects(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as caught:
            runtime.main(self.arguments(Path(directory)) + ["--tour-id", "tour..invalid"])
        self.assertEqual(caught.exception.code, 2)

    def test_authorized_execute_uses_new_root_and_preserves_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self.session()
            before = Path.cwd()
            with patch("scripts.aufgabe04.real_robot.mission.stored_pose_session.load_stored_pose_session",
                       return_value=session), \
                 patch.object(runtime, "execute_tour", return_value={"status": "completed"}) as execute, \
                 redirect_stdout(io.StringIO()):
                result = runtime.main(self.arguments(root) + ["--execute", "--confirm-unloaded",
                                                              "--confirm-odom-continuity", "--server-plan-only"])
            self.assertEqual(result, 0)
            self.assertEqual(Path.cwd(), before)
            self.assertIs(execute.call_args.args[0], session)
            self.assertTrue(execute.call_args.args[1].server_plan_only)
            self.assertEqual(execute.call_args.args[2:], ((root / "tours/tour_test").resolve(), "tour_test"))

    def test_execute_wires_real_orchestrator_to_new_authorization_and_all_saved_poses(self):
        self._assert_execute_start_policy(drive_to_start=False)

    def test_explicit_drive_to_start_allows_initial_navigation(self):
        self._assert_execute_start_policy(drive_to_start=True)

    def test_unverified_start_handoff_follows_plan_requests_but_prevents_arrival_reports(self):
        self._assert_execute_start_policy(drive_to_start=False, fail_handoff=True)

    def test_declined_run_has_no_localization_navigation_or_server_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(drive_to_start=False)
            effects = Mock()
            with patch.object(runtime, "build_navigation_effects", return_value=effects) as build, \
                 patch("builtins.input", return_value="STOP"), \
                 patch("scripts.aufgabe04.task_client.station_tour_client.StationTourClient") as client, \
                 patch("scripts.aufgabe04.navigation.execution.mission_leg_motion_permit.write_mission_leg_motion_authorization") as authorize, \
                 redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "did not authorize"):
                runtime.execute_tour(self.session(), args, Path(directory), "tour_test")
            build.assert_not_called()
            effects.admit_planning_frame.assert_not_called()
            effects.run_motion_leg.assert_not_called()
            client.assert_not_called()
            authorize.assert_not_called()

    def test_invalid_source_artifacts_fail_before_run_and_server_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("scripts.aufgabe04.real_robot.mission.stored_pose_session.load_stored_pose_session",
                       side_effect=ValueError("source hash changed")), \
                 patch("builtins.input") as confirm, \
                 patch.object(runtime, "build_navigation_effects") as effects, \
                 patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(runtime.main(self.arguments(root) + ["--execute", "--confirm-unloaded",
                    "--confirm-odom-continuity"]), 2)
            confirm.assert_not_called()
            effects.assert_not_called()

    def _assert_execute_start_policy(self, *, drive_to_start, fail_handoff=False):
        from tests.aufgabe04.test_station_tour import FakeClient
        from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
            MissionLegKind, load_mission_leg_motion_authorization,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            session = self.session()
            session.config = TourConfig()
            session.profile = SimpleNamespace(robot_id="robot", robot_radius_m=.105, resolved_runtime=lambda: SimpleNamespace(
                namespace="robot", cmd_vel_topic="/robot/cmd_vel", odom_frame="odom"))
            args = SimpleNamespace(server_base_url="http://fixture.invalid", server_robot_id="robot",
                                   http_timeout_sec=5., stations=3, server_plan_only=False,
                                   drive_to_start=drive_to_start)
            client = FakeClient([])
            effects = Mock()
            maps = []
            def arrived(stored, config, received_effects, *, tour_session_id, visit_index, output_root,
                        obstacle_map, capture_scan, verify_start_handoff):
                self.assertIs(received_effects, effects)
                maps.append(obstacle_map)
                self.assertEqual(obstacle_map.tour_id, tour_session_id)
                self.assertTrue(callable(capture_scan))
                self.assertEqual(config.mission_leg_motion_authorization_json, root / "motion_authorization/tour.json")
                qr_id = next(qr for qr, value in session.poses_by_qr.items() if value is stored)
                if visit_index == 0:
                    self.assertEqual(client.events[:2], [("randomize", 4, 3), ("get_mappings",)])
                    self.assertTrue((root / "server/frozen_plan.json").exists())
                client.events.append(("navigate", qr_id, visit_index))
                self.assertEqual(output_root, root / "visits" / f"{visit_index:03d}")
                self.assertEqual(verify_start_handoff, visit_index == 0 and not drive_to_start)
                if fail_handoff:
                    self.assertEqual(qr_id, "Start")
                    self.assertTrue(verify_start_handoff)
                    raise RuntimeError("Start handoff is not at the stored pose")
                return {"arrival_verified": True, "qr_id": qr_id}
            # Give each QR a distinguishable immutable saved-target object.
            session.poses_by_qr = {qr: SimpleNamespace(qr_id=qr) for qr in session.poses_by_qr}
            authorized = []
            def confirm(_):
                self.assertEqual(client.events, [])
                self.assertEqual(effects.mock_calls, [])
                authorized.append(True)
                return "RUN"
            def build(*_):
                self.assertTrue(authorized)
                return effects
            output = io.StringIO()
            with patch.object(runtime, "build_navigation_effects", side_effect=build), \
                 patch("builtins.input", side_effect=confirm), \
                 patch("scripts.aufgabe04.task_client.station_tour_client.StationTourClient", return_value=client), \
                 patch("scripts.aufgabe04.real_robot.mission.tour_obstacle_navigation.execute_tour_obstacle_navigation",
                       side_effect=arrived) as navigate, redirect_stdout(output):
                if fail_handoff:
                    with self.assertRaisesRegex(RuntimeError, "Start handoff"):
                        runtime.execute_tour(session, args, root, "tour_test")
                    self.assertEqual(client.events, [("randomize", 4, 3), ("get_mappings",), ("navigate", "Start", 0)])
                    self.assertEqual(client.reported, [])
                    self.assertEqual(navigate.call_count, 1)
                    effects.admit_planning_frame.assert_not_called()
                    self.assertIn("Server plan: validated", output.getvalue())
                    return
                result = runtime.execute_tour(session, args, root, "tour_test")
            self.assertTrue(result["all_saved_stands_visited"])
            self.assertTrue(result["server_mission_finished"])
            self.assertEqual(navigate.call_count, 8)
            self.assertTrue(all(value is maps[0] for value in maps))
            self.assertEqual(len(client.reported), 5)
            effects.admit_planning_frame.assert_not_called()
            self.assertIn("Server request: POST /api/v1/robots/robot/plan/randomize", output.getvalue())
            self.assertIn("Server accepted Start; next target: QR_003", output.getvalue())
            authorization = load_mission_leg_motion_authorization(root / "motion_authorization/tour.json")
            self.assertEqual(authorization.session_id, "tour_test")
            self.assertEqual(authorization.allowed_leg_kinds, (MissionLegKind.STORED_POSE_TOUR,))
            self.assertEqual(session.config.mission_leg_motion_authorization_json, Path("old-exploration-authorization"))

    def test_cli_start_handoff_is_verify_only_unless_opted_into_driving(self):
        parser = runtime.build_parser()
        arguments = self.arguments(Path("/tmp/tour"))
        self.assertFalse(parser.parse_args(arguments).drive_to_start)
        self.assertTrue(parser.parse_args([*arguments, "--drive-to-start"]).drive_to_start)

    def test_navigation_adapter_never_requests_camera_readiness(self):
        module = "scripts.aufgabe04.real_robot.autonomous_runner.runtime"
        with patch(module + "._run_motion_leg", return_value="outcome") as motion, \
             patch(module + "._admit_candidate_planning_frame", return_value="frame") as capture:
            profile = SimpleNamespace(resolved_runtime=lambda: "runtime")
            effects = runtime.build_navigation_effects(profile, Path("/tmp/tour"))
            self.assertEqual(effects.admit_planning_frame(Path("/tmp/tour/fresh.json")), "frame")
            request = SimpleNamespace(sealed={"route_csv": "route"}, run_id="visit",
                session_root=Path("/tmp/tour"), candidate_snapshot_path=Path("snapshot"),
                uncertainty_map_yaml=Path("map.yaml"), uncertainty_sigma_multiplier=2.,
                localization_branch_proof_id="branch", mission_authorization_json=Path("new_auth"),
                session_id="new_tour", semantic_map_id="arena", mission_leg_kind="stored_pose_tour",
                mission_leg_index=8, target_id="candidate", permit_json_path=Path("permit"))
            self.assertEqual(effects.run_motion_leg(request), "outcome")
            self.assertIsNone(motion.call_args.kwargs["sensor_timing_readiness_phase"])
            self.assertTrue(motion.call_args.kwargs["stored_pose_tour_obstacle_monitor"])
            self.assertEqual(motion.call_args.kwargs["mission_leg_permit_context"].session_id, "new_tour")
            capture.assert_called_once_with("runtime", Path("/tmp/tour"), evidence_path=Path("/tmp/tour/fresh.json"))
