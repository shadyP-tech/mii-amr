import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.perception.stand_axis.model_profile import (
    load_measured_physical_stand_model,
    stand_model_from_payload,
    write_stand_model,
)
from scripts.aufgabe04.real_robot.autonomous_runner import runtime
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)
from scripts.aufgabe04.real_robot.observer.process import (
    PassiveObserverProcessEvidence,
)
from tests.aufgabe04.backside_axis_fixture import backside_axis_payload


def _write_measured_model(root: Path) -> Path:
    path = root / "measured_physical_stand.json"
    write_stand_model(
        path,
        stand_model_from_payload(
            {
                "schema_version": 2,
                "profile_id": "physical_test_v2",
                "environment": "physical",
                "measurement_status": "measured",
                "head_width_m": 0.078,
                "head_height_m": 0.078,
                "head_depth_m": 0.006,
                "qr_symbol_width_m": 0.062,
                "qr_symbol_height_m": 0.062,
                "qr_panel_width_m": 0.071,
                "qr_panel_height_m": 0.071,
                "qr_center_x_m": 0.0,
                "qr_center_y_m": 0.0,
                "head_top_height_m": 0.210,
                "base_width_m": 0.153,
                "base_depth_m": 0.153,
                "tolerance_m": 0.002,
                "source": "direct test metrology",
            }
        ),
    )
    return path


def _args(model_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        robot_profile=Path("robot.json"),
        camera_calibration=Path("camera.json"),
        session_id="session_001",
        final_facing_offset_m=0.35,
        axis_sample_count=7,
        stand_model_profile=model_path,
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


def _bound_backside_axis_payload(model_path: Path) -> dict[str, object]:
    candidate = _candidate()
    payload = backside_axis_payload(
        stand_id=candidate.candidate_uid,
        planning_frame="map",
        stand_x_m=candidate.geometry.x_m,
        stand_y_m=candidate.geometry.y_m,
    )
    payload["stand_model_profile_sha256"] = (
        load_measured_physical_stand_model(model_path).sha256
    )
    return payload


class AutonomousCameraCaptureTests(unittest.TestCase):
    @patch.object(runtime.subprocess, "Popen")
    def test_missing_model_fails_before_attempt_artifacts_or_process(
        self,
        popen,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "attempt"
            args = _args(Path("unused-model.json"))
            args.stand_model_profile = None

            with self.assertRaisesRegex(
                RuntimeError,
                "requires a measured physical stand model",
            ):
                runtime._capture_camera_recommendation(
                    profile=object(),
                    args=args,
                    candidate=_candidate(),
                    output_dir=output_dir,
                )

            self.assertFalse(output_dir.exists())
            popen.assert_not_called()

    @patch.object(runtime.subprocess, "Popen")
    @patch.object(runtime, "monitor_passive_observer_process")
    def test_reaped_candidate_local_deadline_becomes_typed_deferral(
        self,
        monitor,
        _popen,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "attempt"

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
                    args=_args(_write_measured_model(root)),
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
            root = Path(tmp)
            output_dir = root / "attempt"
            axis_path = output_dir / "axis_observation.json"
            model_path = _write_measured_model(root)
            axis_payload = _bound_backside_axis_payload(model_path)

            def complete_with_axis(**kwargs):
                kwargs["axis_observation_path"].write_text(
                    json.dumps(axis_payload), encoding="utf-8"
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
                profile=SimpleNamespace(map_frame="map"),
                args=_args(model_path),
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
        self.assertIn("--stand-model-profile", command)
        self.assertNotIn("--stand-face-size-m", command)
        self.assertAlmostEqual(
            float(command[command.index("--stand-radius-m") + 1]),
            0.06,
        )
        self.assertAlmostEqual(
            float(command[command.index("--stand-head-center-height-m") + 1]),
            0.171,
        )
        self.assertEqual(
            command[command.index("--status-events-jsonl") + 1],
            str(output_dir / "observer_events.jsonl"),
        )
        self.assertEqual(process_payload["completion_kind"], "artifact")
        self.assertEqual(process_payload["artifact_kind"], "axis_observation")

    @patch.object(runtime.subprocess, "Popen")
    @patch.object(runtime, "monitor_passive_observer_process")
    def test_axis_artifact_rejects_candidate_binding_tampering(
        self,
        monitor,
        _popen,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = _write_measured_model(root)
            model_sha256 = load_measured_physical_stand_model(
                model_path
            ).sha256
            tampered_sha256 = (
                ("0" if model_sha256[0] != "0" else "1")
                + model_sha256[1:]
            )
            cases = (
                ("stand_id", "stand_id", "different_candidate"),
                ("planning_frame", "planning_frame", "odom"),
                ("stand_x", "stand_center.x_m", 1.200002),
                ("stand_y", "stand_center.y_m", -0.300002),
                (
                    "model_hash",
                    "stand_model_profile_sha256",
                    tampered_sha256,
                ),
            )

            for label, expected_field, tampered_value in cases:
                with self.subTest(label=label):
                    payload = _bound_backside_axis_payload(model_path)
                    if label == "stand_x":
                        payload["stand_center"]["x_m"] = tampered_value
                    elif label == "stand_y":
                        payload["stand_center"]["y_m"] = tampered_value
                    else:
                        payload[expected_field] = tampered_value

                    def complete_with_axis(**kwargs):
                        kwargs["axis_observation_path"].write_text(
                            json.dumps(payload), encoding="utf-8"
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
                    with self.assertRaises(RuntimeError) as caught:
                        runtime._capture_camera_recommendation(
                            profile=SimpleNamespace(map_frame="map"),
                            args=_args(model_path),
                            candidate=_candidate(),
                            output_dir=root / f"attempt_{label}",
                        )
                    self.assertIn(expected_field, str(caught.exception))

    @patch.object(runtime.subprocess, "Popen")
    @patch.object(runtime, "monitor_passive_observer_process")
    def test_axis_artifact_rejects_invalid_backside_contract(
        self,
        monitor,
        _popen,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = _write_measured_model(root)
            payload = _bound_backside_axis_payload(model_path)
            payload["qr_marker_detected"] = True

            def complete_with_axis(**kwargs):
                kwargs["axis_observation_path"].write_text(
                    json.dumps(payload), encoding="utf-8"
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
            with self.assertRaisesRegex(
                RuntimeError,
                "invalid backside axis receipt",
            ):
                runtime._capture_camera_recommendation(
                    profile=SimpleNamespace(map_frame="map"),
                    args=_args(model_path),
                    candidate=_candidate(),
                    output_dir=root / "attempt_invalid_contract",
                )

    @patch.object(runtime.subprocess, "Popen")
    @patch.object(runtime, "monitor_passive_observer_process")
    def test_recommendation_returns_one_bound_qr_identity(
        self,
        monitor,
        _popen,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "attempt"

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
                args=_args(_write_measured_model(root)),
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
            root = Path(tmp)
            output_dir = root / "attempt"

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
                    args=_args(_write_measured_model(root)),
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
