import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.real_robot.autonomous_runner import runtime
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)
from scripts.aufgabe04.real_robot.observer.process import (
    PassiveObserverProcessEvidence,
)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        robot_profile=Path("robot.json"),
        camera_calibration=Path("camera.json"),
        session_id="session_001",
        final_facing_offset_m=0.35,
        axis_sample_count=7,
        stand_model_profile=None,
        camera_timeout_sec=90.0,
    )


def _candidate() -> SimpleNamespace:
    return SimpleNamespace(
        candidate_uid="survey_candidate_0004",
        geometry=SimpleNamespace(
            x_m=1.2,
            y_m=-0.3,
            radius_m=0.06,
            uncertainty_m=0.02,
        ),
    )


class AutonomousCameraCaptureTests(unittest.TestCase):
    @patch.object(runtime.subprocess, "Popen")
    @patch.object(runtime, "monitor_passive_observer_process")
    def test_reaped_candidate_local_deadline_becomes_typed_deferral(
        self,
        monitor,
        _popen,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "attempt"

            def expire(**kwargs):
                status_path = kwargs["recommendation_path"].parent / (
                    "observer_status.json"
                )
                status_path.write_text(
                    json.dumps(
                        {
                            "state": "lidar_target_mismatch",
                            "axis_consensus": {
                                "sample_count": 2,
                                "peak_sample_count": 6,
                                "required_sample_count": 7,
                            },
                            "candidate_lidar_association": {
                                "nearest_range_delta_m": 0.094,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return PassiveObserverProcessEvidence(
                    completion_kind="deadline",
                    artifact_kind=None,
                    artifact_path=None,
                    deadline_expired=True,
                    returncode=130,
                    cleanup_actions=("send_sigint", "wait_after_sigint"),
                    signals_sent=("SIGINT",),
                )

            monitor.side_effect = expire
            with self.assertRaises(CandidateObservationUnavailableError) as caught:
                runtime._capture_camera_recommendation(
                    profile=object(),
                    args=_args(),
                    candidate=_candidate(),
                    output_dir=output_dir,
                    observation_attempt_index=1,
                )

        error = caught.exception
        self.assertEqual(error.candidate_uid, "survey_candidate_0004")
        self.assertEqual(error.observation_attempt_index, 1)
        self.assertEqual(error.process_evidence["completion_kind"], "deadline")
        self.assertEqual(error.status_evidence["state"], "lidar_target_mismatch")
        self.assertEqual(error.status_evidence["peak_consensus_sample_count"], 6)
        self.assertIn("nearest_lidar_range_delta_m=0.094", str(error))

    @patch.object(runtime.subprocess, "Popen")
    @patch.object(runtime, "monitor_passive_observer_process")
    def test_axis_artifact_is_terminal_and_command_enables_event_history(
        self,
        monitor,
        popen,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "attempt"
            axis_path = output_dir / "axis_observation.json"

            def complete_with_axis(**kwargs):
                kwargs["axis_observation_path"].write_text(
                    "{}", encoding="utf-8"
                )
                return PassiveObserverProcessEvidence(
                    completion_kind="artifact",
                    artifact_kind="axis_observation",
                    artifact_path=kwargs["axis_observation_path"],
                    deadline_expired=False,
                    returncode=0,
                    cleanup_actions=("graceful_wait",),
                    signals_sent=(),
                )

            monitor.side_effect = complete_with_axis
            result = runtime._capture_camera_recommendation(
                profile=object(),
                args=_args(),
                candidate=_candidate(),
                output_dir=output_dir,
            )

            process_payload = load_content_hashed_json(
                output_dir / "observer_process.json",
                hash_field="observer_process_evidence_sha256",
            )

        self.assertEqual(result, (None, None, axis_path))
        command = popen.call_args.args[0]
        self.assertIn("--status-events-jsonl", command)
        self.assertEqual(
            command[command.index("--status-events-jsonl") + 1],
            str(output_dir / "observer_events.jsonl"),
        )
        self.assertEqual(process_payload["completion_kind"], "artifact")
        self.assertEqual(process_payload["artifact_kind"], "axis_observation")

    @patch.object(runtime.subprocess, "Popen")
    @patch.object(runtime, "monitor_passive_observer_process")
    def test_recommendation_returns_one_bound_qr_identity(
        self,
        monitor,
        _popen,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "attempt"

            def complete_with_recommendation(**kwargs):
                recommendation_path = kwargs["recommendation_path"]
                recommendation_path.write_text("{}", encoding="utf-8")
                (recommendation_path.parent / "observer_status.json").write_text(
                    json.dumps(
                        {
                            "state": "recommendation_committed",
                            "qr_texts": ["station_04"],
                        }
                    ),
                    encoding="utf-8",
                )
                return PassiveObserverProcessEvidence(
                    completion_kind="artifact",
                    artifact_kind="recommendation",
                    artifact_path=recommendation_path,
                    deadline_expired=False,
                    returncode=0,
                    cleanup_actions=("graceful_wait",),
                    signals_sent=(),
                )

            monitor.side_effect = complete_with_recommendation
            result = runtime._capture_camera_recommendation(
                profile=object(),
                args=_args(),
                candidate=_candidate(),
                output_dir=output_dir,
            )

        self.assertEqual(
            result,
            (output_dir / "recommendation.json", "station_04", None),
        )

    @patch.object(runtime.subprocess, "Popen")
    @patch.object(runtime, "monitor_passive_observer_process")
    def test_deadline_reports_final_tf_retry_evidence(
        self,
        monitor,
        _popen,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "attempt"

            def expire(**kwargs):
                status_path = kwargs["recommendation_path"].parent / (
                    "observer_status.json"
                )
                status_path.write_text(
                    json.dumps(
                        {
                            "state": "tf_retry_exhausted",
                            "reason": "future extrapolation by 0.001858 sec",
                            "axis_consensus": {
                                "sample_count": 3,
                                "required_sample_count": 7,
                            },
                            "tf_retry": {"retry_count": 8},
                            "tf_retry_elapsed_sec": 0.16,
                            "retry_exhausted": True,
                        }
                    ),
                    encoding="utf-8",
                )
                return PassiveObserverProcessEvidence(
                    completion_kind="deadline",
                    artifact_kind=None,
                    artifact_path=None,
                    deadline_expired=True,
                    returncode=130,
                    cleanup_actions=("send_sigint", "wait_after_sigint"),
                    signals_sent=("SIGINT",),
                )

            monitor.side_effect = expire
            with self.assertRaises(RuntimeError) as caught:
                runtime._capture_camera_recommendation(
                    profile=object(),
                    args=_args(),
                    candidate=_candidate(),
                    output_dir=output_dir,
                )

            process_payload = load_content_hashed_json(
                output_dir / "observer_process.json",
                hash_field="observer_process_evidence_sha256",
            )

        message = str(caught.exception)
        self.assertIn("deadline expired", message)
        self.assertIn("state=tf_retry_exhausted", message)
        self.assertIn("consensus=3/7", message)
        self.assertIn("tf_retry_count=8", message)
        self.assertEqual(process_payload["completion_kind"], "deadline")


if __name__ == "__main__":
    unittest.main()
