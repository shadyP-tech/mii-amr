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
from scripts.aufgabe04.real_robot.candidate.inspection_execution import (
    CandidateInspectionEffects,
    execute_candidate_inspection,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationDeferralLedger,
    CandidateObservationUnavailableError,
)
from scripts.aufgabe04.real_robot.observer.process import (
    PassiveObserverProcessEvidence,
)
from tests.aufgabe04.backside_axis_fixture import backside_axis_payload
from tests.aufgabe04.observer_timeout_fixture import recorded_backside_timeout_status


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
    @patch.object(runtime, "monitor_passive_observer_process")
    @patch.object(runtime, "real_robot_profile_sha256", return_value="b" * 64)
    @patch.object(runtime, "load_bound_camera_inspection")
    def test_intermediate_inspection_is_bound_after_child_reap(
        self, load_bound, profile_hash, monitor, popen
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = _write_measured_model(root)
            output = root / "attempt"
            inspection_path = output / "inspection_observation.json"
            monitor.return_value = PassiveObserverProcessEvidence(
                completion_kind="artifact",
                artifact_kind="inspection_observation",
                artifact_path=inspection_path,
                deadline_expired=False,
                returncode=0,
                cleanup_actions=("graceful_wait",),
                signals_sent=(),
            )

            def inspect_after_reap(*args, **kwargs):
                self.assertEqual(monitor.call_count, 1)
                self.assertTrue((output / "observer_process.json").exists())
                return {"qr_id": "QR_001", "completion_authorized": False}

            load_bound.side_effect = inspect_after_reap
            result = runtime._capture_camera_recommendation(
                profile=SimpleNamespace(
                    map_frame="map", calibration_profile_sha256="c" * 64
                ),
                args=_args(model_path), candidate=_candidate(), output_dir=output,
            )
            self.assertEqual(result, (None, "QR_001", None, inspection_path))
            bound = load_bound.call_args.kwargs
            self.assertEqual(bound["candidate_uid"], _candidate().candidate_uid)
            self.assertEqual(bound["stream_id"], "session_001_survey_candidate_0004")
            self.assertEqual(bound["stand_x_m"], 1.2)
            self.assertEqual(bound["stand_y_m"], -0.3)
            self.assertEqual(bound["robot_profile_sha256"], "b" * 64)
            self.assertEqual(bound["calibration_profile_sha256"], "c" * 64)
            self.assertEqual(
                bound["stand_model_profile_sha256"],
                load_measured_physical_stand_model(model_path).sha256,
            )
            self.assertEqual(
                monitor.call_args.kwargs["inspection_observation_path"],
                inspection_path,
            )
            self.assertIn("--inspection-observation-json", popen.call_args.args[0])
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--capture-history-dir") + 1], str(output / "capture_history"))
            self.assertEqual(command[command.index("--capture-max-frames") + 1], "64")
            self.assertEqual(command[command.index("--capture-max-bytes") + 1], "33554432")
            self.assertEqual(command[command.index("--scan-topology-profile") + 1], "linear")

    @patch.object(runtime, "_capture_camera_recommendation")
    def test_capture_adapter_preserves_inspection_path_and_legacy_success(self, capture):
        request = SimpleNamespace(candidate=_candidate(), output_dir=Path("out"), attempt_index=2)
        for result in (
            (Path("recommendation.json"), "QR_004", None),
            (None, "QR_001", None, Path("inspection.json")),
        ):
            with self.subTest(result=result):
                capture.return_value = result
                observation = runtime._capture_candidate_observation(
                    profile=object(), args=object(), request=request
                )
                self.assertEqual(observation.recommendation_path, result[0])
                self.assertEqual(observation.qr_id, result[1])
                self.assertEqual(
                    observation.inspection_observation_path,
                    result[3] if len(result) == 4 else None,
                )

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
        self.assertNotIsInstance(
            caught.exception, CandidateObservationUnavailableError
        )

    @patch.object(runtime.subprocess, "Popen")
    @patch.object(runtime, "monitor_passive_observer_process")
    def test_rejected_frames_at_deadline_preserve_other_candidate_retries(
        self,
        monitor,
        _popen,
    ) -> None:
        # Reduced evidence from the 2026-09-07 run: the final transient TF
        # status followed 272 transformed but LiDAR-rejected observations.
        status_payload = {
            "state": "tf_pending_exact_time",
            "reason": "future extrapolation by 0.005416 sec",
            "axis_consensus": {
                "sample_count": 0,
                "peak_sample_count": 0,
                "required_sample_count": 7,
            },
            "observation_evidence": {
                "accepted_frame_count": 0,
                "lidar_rejection_count": 272,
                "soft_miss_count": 346,
                "last_soft_miss_reason": "lidar_target_not_associated",
                "poisoned": False,
                "poison_reason": None,
            },
            "tf_retry_attempt_summary": {
                "attempted_tuple_count": 127,
                "exhausted_tuple_count": 2,
            },
            "retry_exhausted": False,
        }

        def expire(**kwargs):
            (kwargs["recommendation_path"].parent / "observer_status.json").write_text(
                json.dumps(status_payload), encoding="utf-8"
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

        ledger = CandidateObservationDeferralLedger(
            [f"survey_candidate_{index:04d}" for index in range(1, 6)],
            max_attempts_per_candidate=2,
        )
        for index in (3, 1, 2, 4, 5):
            uid = f"survey_candidate_{index:04d}"
            ledger.select(uid)
            if index in (3, 5):
                ledger.mark_resolved({"qr_id": f"QR_{index}"})
            else:
                ledger.mark_unavailable(
                    CandidateObservationUnavailableError(
                        candidate_uid=uid,
                        observation_attempt_index=0,
                        reason="first-pass candidate observation unavailable",
                        process_evidence={"completion_kind": "deadline"},
                        status_evidence={"state": "metric_model_measurement_unavailable"},
                    )
                )
        self.assertTrue(ledger.advance_pass())
        ledger.select("survey_candidate_0001")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = _candidate()
            candidate.candidate_uid = "survey_candidate_0001"
            monitor.side_effect = expire
            with self.assertRaises(CandidateObservationUnavailableError) as caught:
                runtime._capture_camera_recommendation(
                    profile=object(),
                    args=_args(_write_measured_model(root)),
                    candidate=candidate,
                    output_dir=root / "retry_attempt",
                )
            process_payload = load_content_hashed_json(
                root / "retry_attempt" / "observer_process.json",
                hash_field="observer_process_evidence_sha256",
            )

        error = caught.exception
        self.assertEqual(error.status_evidence["accepted_frame_count"], 0)
        self.assertEqual(error.status_evidence["lidar_rejection_count"], 272)
        self.assertIn("lidar_rejections=272", str(error))
        self.assertEqual(process_payload["completion_kind"], "deadline")
        self.assertFalse(error.to_failure_fields()["motion_continues_authorized"])
        ledger.mark_unavailable(error)
        state = ledger.selection_state()
        self.assertEqual(
            state.eligible_candidate_uids,
            ("survey_candidate_0002", "survey_candidate_0004"),
        )
        self.assertFalse(state.complete)
        self.assertFalse(state.terminal_incomplete)
        for uid in state.eligible_candidate_uids:
            ledger.select(uid)
            ledger.mark_resolved({"qr_id": f"QR_{uid}"})
        final_state = ledger.selection_state()
        self.assertTrue(final_state.terminal_incomplete)
        self.assertFalse(final_state.complete)
        self.assertEqual(final_state.unresolved_candidate_uids, (candidate.candidate_uid,))

    @patch.object(runtime.subprocess, "Popen")
    @patch.object(runtime, "monitor_passive_observer_process")
    def test_recorded_stale_deadline_records_each_view_and_respects_inspection_budget(
        self, monitor, _popen
    ) -> None:
        def expire(**kwargs):
            (kwargs["recommendation_path"].parent / "observer_status.json").write_text(
                json.dumps(recorded_backside_timeout_status()), encoding="utf-8"
            )
            return PassiveObserverProcessEvidence(
                completion_kind="deadline", artifact_kind=None, artifact_path=None,
                deadline_expired=True, returncode=130,
                cleanup_actions=("send_sigint", "wait_after_sigint"),
                signals_sent=("SIGINT",),
            )

        monitor.side_effect = expire
        for max_views in (2, 8):
            with self.subTest(max_views=max_views), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = _args(_write_measured_model(root))
                captures, failures, moves = [], [], []

                def capture(frame, output, index):
                    captures.append(index)
                    try:
                        return runtime._capture_camera_recommendation(
                            profile=object(), args=args, candidate=_candidate(),
                            output_dir=output, observation_attempt_index=index,
                        )
                    except CandidateObservationUnavailableError as exc:
                        failures.append(exc)
                        raise

                def move(frame, normal, output, index, source):
                    moves.append(index)
                    return normal

                with self.assertRaises(CandidateObservationUnavailableError) as caught:
                    execute_candidate_inspection(
                        candidate_uid=_candidate().candidate_uid, candidate_root=root,
                        initial_frame=0.0, max_views=max_views,
                        effects=CandidateInspectionEffects(
                            capture=capture, canonical_normal=lambda frame: frame,
                            move_view=move,
                            move_opposite=lambda *args: self.fail("stale axis authorized opposite view"),
                            progress_evidence=lambda *args: {},
                        ),
                    )

                progress = json.loads((root / "inspection_progress.json").read_text())
                self.assertEqual(captures, list(range(max_views)))
                self.assertEqual(len(moves), max_views - 1)
                self.assertEqual(progress["local_view_count"], max_views)
                self.assertEqual(progress["termination_reason"], "view_budget_exhausted")
                self.assertFalse(progress["joint_observation_ready"])
                self.assertEqual(caught.exception.reason, "candidate_local_inspection_exhausted")
                for failure in failures:
                    self.assertEqual(failure.status_evidence["accepted_frame_count"], 0)
                    self.assertEqual(failure.status_evidence["consensus_sample_count"], 0)
                    self.assertEqual(
                        failure.status_evidence["timeout_classification_basis"],
                        "accumulated_transform_ready_candidate_frames",
                    )
                    self.assertFalse(failure.to_failure_fields()["motion_continues_authorized"])

    @patch.object(runtime.subprocess, "Popen")
    @patch.object(runtime, "monitor_passive_observer_process")
    def test_trailing_tf_wait_after_candidate_frames_becomes_typed_deferral(
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
                            "state": "tf_pending_exact_time",
                            "reason": "future extrapolation by 0.015919 sec",
                            "axis_consensus": {
                                "sample_count": 0,
                                "peak_sample_count": 0,
                                "required_sample_count": 7,
                            },
                            "observation_evidence": {
                                "accepted_frame_count": 340,
                                "lidar_rejection_count": 0,
                                "soft_miss_count": 0,
                            },
                            "tf_retry_attempt_summary": {
                                "attempted_tuple_count": 342,
                                "exhausted_tuple_count": 104,
                            },
                            "retry_exhausted": False,
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
            with self.assertRaises(
                CandidateObservationUnavailableError
            ) as caught:
                runtime._capture_camera_recommendation(
                    profile=object(),
                    args=_args(_write_measured_model(root)),
                    candidate=_candidate(),
                    output_dir=output_dir,
                )

        error = caught.exception
        self.assertEqual(
            error.status_evidence["state"],
            "tf_pending_exact_time",
        )
        self.assertEqual(error.status_evidence["accepted_frame_count"], 340)
        self.assertEqual(
            error.status_evidence["timeout_classification_basis"],
            "accumulated_transform_ready_candidate_frames",
        )
        self.assertIn("accepted_candidate_frames=340", str(error))


if __name__ == "__main__":
    unittest.main()
