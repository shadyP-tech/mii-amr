"""The QR-only terminal receipt cannot erase process or attempt failures."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.aufgabe04.real_robot.autonomous_runner import runtime
from scripts.aufgabe04.real_robot.observer.qr_observation_binding import (
    load_bound_qr_observation_pose,
)
from scripts.aufgabe04.real_robot.observer.process import PassiveObserverProcessEvidence
from tests.aufgabe04.test_autonomous_camera_capture import (
    _args, _candidate, _write_measured_model,
)


class QrObservationTransportTests(unittest.TestCase):
    @staticmethod
    def profile():
        return SimpleNamespace(
            map_frame="map", base_frame="base_footprint", scan_frame="base_scan",
            camera_optical_frame="camera", calibration_profile_sha256="c" * 64,
        )

    def capture(self, output, model, *, returncode=0, status="qr_observation_pose_committed"):
        def completed(**kwargs):
            (output / "observer_status.json").write_text(json.dumps({"state": status}))
            return PassiveObserverProcessEvidence(
                completion_kind="artifact", artifact_kind="qr_verified_observation_pose",
                artifact_path=output / "qr_observation_pose.json", deadline_expired=False,
                returncode=returncode, cleanup_actions=("graceful_wait",), signals_sent=(),
            )

        with patch.object(runtime.subprocess, "Popen") as popen, \
                patch.object(runtime, "real_robot_profile_sha256", return_value="b" * 64), \
                patch.object(runtime, "monitor_passive_observer_process", side_effect=completed) as monitor:
            result = runtime._capture_camera_recommendation(
                profile=self.profile(), args=_args(model), candidate=_candidate(), output_dir=output,
            )
            self.assertEqual(monitor.call_args.kwargs["qr_observation_pose_path"],
                             output / "qr_observation_pose.json")
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--qr-observation-pose-json") + 1],
                             str(output / "qr_observation_pose.json"))
            return result

    @patch.object(runtime, "load_bound_qr_observation_pose")
    def test_bound_qr_pose_is_terminal_without_axis(self, loader):
        loader.return_value = {"qr_id": "QR_003"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.capture(root / "attempt", _write_measured_model(root))
            self.assertEqual(result, (None, "QR_003", None, None,
                                      root / "attempt" / "qr_observation_pose.json"))
            self.assertTrue((root / "attempt" / "observer_process.json").exists())
        bindings = loader.call_args.kwargs
        self.assertEqual(bindings["candidate_uid"], _candidate().candidate_uid)
        self.assertEqual(bindings["stream_id"], "session_001_survey_candidate_0004")
        self.assertEqual(bindings["robot_profile_sha256"], "b" * 64)
        self.assertEqual(bindings["calibration_profile_sha256"], "c" * 64)
        self.assertEqual(bindings["camera_frame"], "camera")
        self.assertEqual(bindings["stand_x_m"], 1.2)

    @patch.object(runtime, "load_bound_qr_observation_pose")
    def test_qr_receipt_does_not_mask_child_or_terminal_status_failure(self, loader):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = _write_measured_model(root)
            for index, (code, state) in enumerate((
                (7, "qr_observation_pose_committed"),
                (-15, "qr_observation_pose_committed"), (0, "localization_failed"),
            )):
                with self.subTest(code=code, state=state), self.assertRaisesRegex(
                    RuntimeError, "without successful observer completion",
                ):
                    self.capture(root / f"attempt_{index}", model, returncode=code, status=state)
        loader.assert_not_called()

    @patch.object(runtime, "load_bound_qr_observation_pose",
                  side_effect=RuntimeError("invalid QR observation pose receipt: hash mismatch"))
    def test_invalid_fallback_is_terminal_and_does_not_become_local_retry(self, loader):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                self.capture(root / "attempt", _write_measured_model(root))
        self.assertEqual(loader.call_count, 1)


class QrObservationBindingTests(unittest.TestCase):
    @staticmethod
    def payload():
        return {
            "candidate_uid": "candidate", "stream_id": "session_candidate",
            "planning_frame": "map", "stand_center": {"x_m": 1.2, "y_m": -.3},
            "robot_profile_sha256": "b" * 64, "calibration_profile_sha256": "c" * 64,
            "stand_model_profile_sha256": "d" * 64,
            "localization_provenance": {"map_frame": "map", "base_frame": "base",
                                        "scan_frame": "scan", "camera_frame": "camera"},
        }

    @staticmethod
    def arguments():
        return dict(candidate_uid="candidate", stream_id="session_candidate", planning_frame="map",
                    stand_x_m=1.2, stand_y_m=-.3, robot_profile_sha256="b" * 64,
                    calibration_profile_sha256="c" * 64, stand_model_profile_sha256="d" * 64,
                    base_frame="base", scan_frame="scan", camera_frame="camera")

    @patch("scripts.aufgabe04.real_robot.observer.qr_observation_binding.load_qr_verified_observation_pose")
    def test_receipt_is_bound_to_parent_candidate_frame_and_profiles(self, loader):
        loader.return_value = self.payload()
        self.assertIs(load_bound_qr_observation_pose(Path("receipt"), **self.arguments()),
                      loader.return_value)
        for field in ("candidate_uid", "stream_id", "planning_frame", "robot_profile_sha256",
                      "calibration_profile_sha256", "stand_model_profile_sha256",
                      "base_frame", "scan_frame", "camera_frame", "stand_x_m", "stand_y_m"):
            args = self.arguments()
            args[field] = 123.0 if field in {"stand_x_m", "stand_y_m"} else "other"
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "not bound"):
                load_bound_qr_observation_pose(Path("receipt"), **args)

    @patch("scripts.aufgabe04.real_robot.observer.qr_observation_binding.load_qr_verified_observation_pose",
           side_effect=ValueError("content hash mismatch"))
    def test_invalid_artifact_never_enters_binding(self, loader):
        with self.assertRaisesRegex(RuntimeError, "invalid QR observation pose receipt"):
            load_bound_qr_observation_pose(Path("receipt"), **self.arguments())


if __name__ == "__main__":
    unittest.main()
