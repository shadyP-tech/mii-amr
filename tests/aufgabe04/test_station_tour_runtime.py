"""Standalone entrypoint boundaries: artifact preview, RUN and navigation only."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
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


class StationTourRuntimeTest(unittest.TestCase):
    def session(self):
        saved = SimpleNamespace(candidate_uid="candidate", pose=Pose2D(1., 2., .3),
            source_frame=SimpleNamespace(to_evidence=lambda: {"map_frame": "map"}),
            evidence={"pose_kind": "qr_verified_observation_pose"})
        return SimpleNamespace(poses_by_qr={qr: saved for qr in (
            "Start", "QR_001", "QR_002", "QR_003", "QR_004")})

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
        from tests.aufgabe04.test_station_tour import FakeClient
        from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
            MissionLegKind, load_mission_leg_motion_authorization,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            session = self.session()
            session.config = TourConfig()
            session.profile = SimpleNamespace(robot_id="robot", resolved_runtime=lambda: SimpleNamespace(
                namespace="robot", cmd_vel_topic="/robot/cmd_vel"))
            args = SimpleNamespace(server_base_url="http://fixture.invalid", server_robot_id="robot",
                                   http_timeout_sec=5., stations=3, server_plan_only=False)
            client = FakeClient([])
            effects = Mock()
            def arrived(stored, config, received_effects, *, tour_session_id, visit_index, output_root):
                self.assertIs(received_effects, effects)
                self.assertEqual(config.mission_leg_motion_authorization_json, root / "motion_authorization/tour.json")
                qr_id = next(qr for qr, value in session.poses_by_qr.items() if value is stored)
                self.assertEqual(output_root, root / "visits" / f"{visit_index:03d}")
                return {"arrival_verified": True, "qr_id": qr_id}
            # Give each QR a distinguishable immutable saved-target object.
            session.poses_by_qr = {qr: SimpleNamespace(qr_id=qr) for qr in session.poses_by_qr}
            with patch.object(runtime, "build_navigation_effects", return_value=effects), \
                 patch("builtins.input", return_value="RUN"), \
                 patch("scripts.aufgabe04.task_client.station_tour_client.StationTourClient", return_value=client), \
                 patch("scripts.aufgabe04.real_robot.mission.stored_pose_navigation.execute_stored_pose_navigation",
                       side_effect=arrived) as navigate, redirect_stdout(io.StringIO()):
                result = runtime.execute_tour(session, args, root, "tour_test")
            self.assertTrue(result["all_saved_stands_visited"])
            self.assertTrue(result["server_mission_finished"])
            self.assertEqual(navigate.call_count, 8)
            self.assertEqual(len(client.reported), 5)
            authorization = load_mission_leg_motion_authorization(root / "motion_authorization/tour.json")
            self.assertEqual(authorization.session_id, "tour_test")
            self.assertEqual(authorization.allowed_leg_kinds, (MissionLegKind.STORED_POSE_TOUR,))
            self.assertEqual(session.config.mission_leg_motion_authorization_json, Path("old-exploration-authorization"))

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
            self.assertEqual(motion.call_args.kwargs["mission_leg_permit_context"].session_id, "new_tour")
            capture.assert_called_once_with("runtime", Path("/tmp/tour"), evidence_path=Path("/tmp/tour/fresh.json"))
