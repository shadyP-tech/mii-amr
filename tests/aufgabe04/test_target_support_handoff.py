"""Authenticated observer target deferral through the production capture parent."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.autonomous_runner import runtime
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from scripts.aufgabe04.real_robot.observer.process import PassiveObserverProcessEvidence
from scripts.aufgabe04.real_robot.observer.target_support_failure import (
    REASON, TargetSupportFailureWindow, target_support_binding_sha256,
)
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256, write_candidate_snapshot
from tests.aufgabe04 import test_autonomous_candidate_approach as approach_fixtures
from tests.aufgabe04.test_autonomous_camera_capture import _args, _write_measured_model


class TargetSupportHandoffTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture = approach_fixtures.AutonomousCandidateApproachTest()
        self.candidate = fixture._candidate("survey_candidate_0005", 1.2, -.3)
        self.config = fixture._config(self.root, (self.candidate,))
        write_candidate_snapshot(self.config.snapshot_path, self.config.snapshot)
        self.model = _write_measured_model(self.root)
        self.profile = SimpleNamespace(
            map_frame="map", base_frame="base_footprint", scan_frame="base_scan",
            camera_optical_frame="camera", calibration_profile_sha256="c" * 64,
        )
        self.binding = {
            "candidate_uid": self.candidate.candidate_uid,
            "stream_id": "session_001_survey_candidate_0005",
            "target_key": "session_001_survey_candidate_0005:survey_candidate_0005:1.200000000:-0.300000000",
            "planning_frame": "map", "stand_center": {"x_m": 1.2, "y_m": -.3},
            "candidate_snapshot_sha256": candidate_snapshot_sha256(self.config.snapshot),
            "robot_profile_sha256": "b" * 64, "calibration_profile_sha256": "c" * 64,
            "stand_model_profile_sha256": load_measured_physical_stand_model(self.model).sha256,
        }

    def receipt(self):
        """Use real scan association and the real bounded policy to earn a receipt."""
        window = TargetSupportFailureWindow()
        result = None
        for index in range(7):
            stamp = 100. + .9 * index
            scan = PlainLaserScan(
                ranges=(1.2,) * 61, angle_min=-.3, angle_max=.3, angle_increment=.01,
                range_min=.1, range_max=5., scan_frame_id="base_scan",
                scan_stamp_sec=stamp, receipt_sec=stamp, scan_topology_profile="linear",
            )
            association = associate_candidate_lidar_target(
                scan, map_bearing_rad=0., cone_half_angle_rad=math.radians(15.),
                accepted_range_m=(.32, .54), now_sec=stamp + .01, max_scan_age_sec=.5,
            )
            result = window.observe(
                association=asdict(association), target_binding=self.binding, now_sec=stamp + .01,
                frame={"frame_stamp_sec": stamp, "scan_stamp_sec": stamp,
                       "robot_pose": {"x_m": .7, "y_m": -.3, "yaw_rad": 0.},
                       "motion_epoch": 0, "tf_validated": True, "poisoned": False,
                       "motion_epoch_reset": False, "frame_accepted": False},
            )
        self.assertIsNotNone(result, window.metadata)
        return result

    def capture(self, name, *, process_changes=None, receipt_mutator=None,
                status_mutator=None, receipt_mode="valid", artifact_name=None,
                observation_not_before_sec=None, snapshot_path=None):
        output = self.root / name

        def completed(**kwargs):
            path = output / "target_support_failure.json"
            payload = self.receipt()
            if receipt_mutator is not None:
                receipt_mutator(payload)
            digest = write_content_hashed_json(path, payload, hash_field="target_support_failure_sha256")
            if receipt_mode == "missing":
                path.unlink()
            elif receipt_mode == "tampered":
                tampered = json.loads(path.read_text())
                tampered["sample_count"] += 1
                path.write_text(json.dumps(tampered))
            status = {
                "state": "target_reconciliation_required", "reason": REASON,
                "observation_evidence": {"poisoned": False, "target_key": self.binding["target_key"],
                                         "motion_epoch": 0}, "motion_capability": "none",
                "target_support_failure": {"path": str(path), "sha256": digest},
            }
            if status_mutator is not None:
                status_mutator(status)
            (output / "observer_status.json").write_text(json.dumps(status))
            if artifact_name is not None:
                (output / artifact_name).write_text("{}")
            fields = {
                "completion_kind": "child_exit", "artifact_kind": None, "artifact_path": None,
                "deadline_expired": False, "returncode": 0,
                "cleanup_actions": ("exit_observed",), "signals_sent": (),
            }
            fields.update(process_changes or {})
            if fields["artifact_kind"] is not None:
                fields["artifact_path"] = output / "recommendation.json"
            return PassiveObserverProcessEvidence(**fields)

        with patch.object(runtime.subprocess, "Popen"), \
                patch.object(runtime, "real_robot_profile_sha256", return_value="b" * 64), \
                patch.object(runtime, "monitor_passive_observer_process", side_effect=completed):
            return runtime._capture_camera_recommendation(
                profile=self.profile, args=_args(self.model), candidate=self.candidate,
                output_dir=output, observation_attempt_index=2,
                candidate_crop_snapshot_path=self.config.snapshot_path if snapshot_path is None else snapshot_path,
                observation_not_before_sec=observation_not_before_sec,
            )

    def test_clean_bound_target_failure_becomes_terminal_typed_deferral_after_reap(self):
        with self.assertRaises(CandidateObservationUnavailableError) as raised:
            self.capture("valid")
        error = raised.exception
        self.assertEqual(error.reason, "candidate_target_ineligible")
        self.assertEqual(error.candidate_uid, self.candidate.candidate_uid)
        self.assertEqual(error.observation_attempt_index, 2)
        self.assertEqual(error.process_evidence["completion_kind"], "child_exit")
        self.assertEqual(error.status_evidence["state"], "target_reconciliation_required")
        self.assertIn("target_support_failure", error.status_evidence)
        self.assertTrue((self.root / "valid/observer_process.json").exists())

    def test_failed_or_forced_child_lifecycle_cannot_become_candidate_local_deferral(self):
        variants = (
            {"returncode": 1},
            {"returncode": -15, "signals_sent": ("SIGTERM",)},
            {"signals_sent": ("SIGINT",)},
            {"completion_kind": "deadline", "deadline_expired": True,
             "returncode": 130, "signals_sent": ("SIGINT",)},
            {"completion_kind": "artifact", "artifact_kind": "recommendation"},
        )
        for index, changes in enumerate(variants):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError) as raised:
                self.capture(f"lifecycle_{index}", process_changes=changes)
            self.assertNotIsInstance(raised.exception, CandidateObservationUnavailableError)

    def test_receipt_hash_source_and_negative_evidence_are_required(self):
        variants = (
            {"receipt_mode": "missing"},
            {"receipt_mode": "tampered"},
            {"status_mutator": lambda p: p["target_support_failure"].update(sha256="a" * 64)},
            {"status_mutator": lambda p: p["observation_evidence"].update(poisoned=True)},
            {"status_mutator": lambda p: p["observation_evidence"].update(target_key="another_target")},
            {"status_mutator": lambda p: p["observation_evidence"].update(motion_epoch=1)},
            {"status_mutator": lambda p: p.update(motion_capability="camera_centering")},
            {"receipt_mutator": lambda p: p["samples"][0]["frame"].update(tf_validated=False)},
            {"receipt_mutator": lambda p: p["samples"][0]["association"].update(in_range_sample_count=1)},
            {"snapshot_path": self.root / "missing_snapshot.json"},
            {"observation_not_before_sec": 101.},
        )
        for index, arguments in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(RuntimeError) as raised:
                self.capture(f"receipt_{index}", **arguments)
            self.assertNotIsInstance(raised.exception, CandidateObservationUnavailableError)

    def test_rehashed_wrong_target_or_profile_cannot_authorize_continuation(self):
        variants = {
            "candidate_uid": "other", "stream_id": "other_stream", "target_key": "other_target",
            "planning_frame": "odom", "stand_center": {"x_m": 1.3, "y_m": -.3},
            "candidate_snapshot_sha256": "a" * 64, "robot_profile_sha256": "a" * 64,
            "calibration_profile_sha256": "a" * 64, "stand_model_profile_sha256": "a" * 64,
        }
        for key, value in variants.items():
            def mutate(payload):
                payload["target_binding"][key] = value
                payload["target_binding_sha256"] = target_support_binding_sha256(payload["target_binding"])
            with self.subTest(field=key), self.assertRaises(RuntimeError) as raised:
                self.capture(f"binding_{key}", receipt_mutator=mutate)
            self.assertNotIsInstance(raised.exception, CandidateObservationUnavailableError)

    def test_conflicting_positive_artifact_cannot_mask_terminal_failure(self):
        for name in ("recommendation.json", "candidate_centering.json", "axis_observation.json",
                     "inspection_observation.json", "qr_observation_pose.json"):
            with self.subTest(artifact=name), self.assertRaises(RuntimeError) as raised:
                self.capture("conflict_" + name, artifact_name=name)
            self.assertNotIsInstance(raised.exception, CandidateObservationUnavailableError)


if __name__ == "__main__":
    unittest.main()
