"""Discovery/server boundary exercised through the candidate state machine."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachEffects, CandidateObservation, execute_candidate_approach_phase,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.stations.server_identity_binding import (
    seal_server_qr_mapping_evidence, write_server_qr_mapping_evidence,
    load_observed_identities,
)
from scripts.aufgabe04.stations.station_identity_registry import load_station_identity_registry
from tests.aufgabe04 import test_autonomous_candidate_approach as fixtures


class CandidateServerBindingTests(unittest.TestCase):
    def fixture(self, root):
        fixture = fixtures.AutonomousCandidateApproachTest()
        config = fixture._config(root, (fixture._candidate("candidate_a", .2, 0.),))
        calls = []
        def motion(request):
            calls.append(request)
            return fixture._completed(request)
        effects = CandidateApproachEffects(
            select_initial_preapproach=fixture._nearest_selection,
            read_current_pose=lambda: Pose2D(0., 0., 0.),
            plan_preapproach=lambda request: {"route_csv": "route.csv"},
            run_motion_leg=motion,
            capture_observation=lambda request: CandidateObservation(
                request.output_dir / "recommendation.json", "Qr_A", None),
            validate_facing=lambda request: {"candidate_uid": request.candidate.candidate_uid},
            commit_decision=lambda request: None, clock=lambda: 10.,
        )
        return config, effects, calls

    def mapping(self, root):
        payload = [{
            "robot_id": "ServerRobot", "mode": "test", "processing_sequence": ["Station.Mixed"],
            "plan_steps": ["Station.Mixed"], "expanded_path": ["Station.Mixed"],
            "next_job_index": 0, "next_step_index": 0,
            "generated_at": "1970-01-01T00:00:01Z",
            "qr_mappings": [{"robot_id": "ServerRobot", "qr_code_id": "Qr_A",
                             "station_id": "Station.Mixed", "station_type": "processing",
                             "display_name": "Processing station"}],
        }]
        evidence = seal_server_qr_mapping_evidence(payload, robot_id="ServerRobot", captured_unix_sec=2.)
        path = root / "server_mapping.json"
        write_server_qr_mapping_evidence(path, evidence)
        return path

    def test_discovery_without_mapping_never_invents_station_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, effects, calls = self.fixture(Path(tmp))
            result = execute_candidate_approach_phase(config, effects)
            observed = load_observed_identities(result.observed_identities_path,
                                                candidate_snapshot=config.snapshot)
            self.assertEqual(observed["observed_qr_by_candidate"], {"candidate_a": "Qr_A"})
            self.assertIsNone(result.identity_registry_path)
            self.assertFalse((config.session_root / "station_identity_registry.json").exists())
            summary = result.to_mission_summary_fields()
            self.assertTrue(summary["goal_completed"])
            self.assertIsNone(summary["station_identity_registry"])
            self.assertEqual(summary["identity_binding_status"], "server_binding_pending")
            self.assertFalse(summary["motion_authorized"])
            self.assertEqual(len(calls), 1)

    def test_explicit_server_mapping_is_used_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, effects, _ = self.fixture(root)
            config = replace(config, server_qr_mapping_evidence_path=self.mapping(root),
                             server_robot_id="ServerRobot")
            result = execute_candidate_approach_phase(config, effects)
            registry = load_station_identity_registry(result.identity_registry_path,
                                                       candidate_snapshot=config.snapshot)
            identity = registry.for_candidate("candidate_a")
            self.assertEqual(identity.qr_id, "Qr_A")
            self.assertEqual(identity.server_station_id, "Station.Mixed")
            self.assertEqual(result.identity_binding_status, "server_bound")

    def test_wrong_robot_or_unpaired_evidence_fails_before_motion(self):
        for robot in (None, "wrong_robot"):
            with self.subTest(robot=robot), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config, effects, calls = self.fixture(root)
                config = replace(config, server_qr_mapping_evidence_path=self.mapping(root), server_robot_id=robot)
                with self.assertRaises(ValueError):
                    execute_candidate_approach_phase(config, effects)
                self.assertEqual(calls, [])

    def test_mapping_expiration_during_discovery_prevents_registry_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, effects, _ = self.fixture(root)
            config = replace(config, server_qr_mapping_evidence_path=self.mapping(root), server_robot_id="ServerRobot")
            now = [10.]
            capture = effects.capture_observation
            def delayed_capture(request):
                now[0] = 4000.
                return capture(request)
            effects = replace(effects, clock=lambda: now[0], capture_observation=delayed_capture)
            with self.assertRaisesRegex(ValueError, "stale"):
                execute_candidate_approach_phase(config, effects)
            self.assertFalse((config.session_root / "station_identity_registry.json").exists())
            self.assertTrue((config.session_root / "observed_station_identities.json").exists())


if __name__ == "__main__":
    unittest.main()
