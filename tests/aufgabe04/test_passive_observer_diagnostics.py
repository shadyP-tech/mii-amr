from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.real_robot.observer.diagnostics import (
    candidate_local_observer_timeout_basis,
    format_passive_observer_failure,
    is_candidate_local_observer_timeout,
    load_passive_observer_status,
)
from scripts.aufgabe04.real_robot.observer.process import (
    PassiveObserverProcessEvidence,
)
from tests.aufgabe04.observer_timeout_fixture import (
    recorded_backside_timeout_status, recorded_front_timeout_status,
)


class PassiveObserverDiagnosticsTests(unittest.TestCase):
    def _load_payload(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "observer_status.json"
            status_path.write_text(json.dumps(payload), encoding="utf-8")
            return load_passive_observer_status(status_path)

    def _process(self, completion_kind="deadline"):
        artifact = completion_kind == "artifact"
        return PassiveObserverProcessEvidence(
            completion_kind=completion_kind,
            artifact_kind="axis_observation" if artifact else None,
            artifact_path=Path("axis.json") if artifact else None,
            deadline_expired=completion_kind == "deadline",
            returncode=130 if completion_kind == "deadline" else 1,
            cleanup_actions=("wait_after_sigint",),
            signals_sent=("SIGINT",),
        )

    def test_opposite_identity_states_enter_bounded_recovery_but_not_after_conflict_or_crash(self):
        for state in ('opposite_identity_collecting','opposite_identity_crop_conflict','opposite_identity_unavailable'):
            with self.subTest(state=state):
                status=self._load_payload(dict(state=state,observation_evidence=dict(
                    accepted_frame_count=360,lidar_rejection_count=44,poisoned=False)))
                self.assertTrue(is_candidate_local_observer_timeout(process=self._process(),status=status))
                self.assertFalse(is_candidate_local_observer_timeout(process=self._process(),
                    status=replace(status,observation_evidence_poisoned=True)))
                self.assertFalse(is_candidate_local_observer_timeout(process=replace(self._process(),returncode=1),status=status))
                message=format_passive_observer_failure(candidate_uid='candidate',process=self._process(),
                    status=status,process_evidence_path=Path('process.json'))
                self.assertIn('retained backside angle',message)
                self.assertNotIn('without a usable axis',message)

    def test_loads_retry_and_consensus_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "observer_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "state": "tf_pending_exact_time",
                        "reason": "requested time is 1.858 ms ahead",
                        "axis_consensus": {
                            "sample_count": 2,
                            "required_sample_count": 7,
                        },
                        "tf_retry": {"retry_count": 4},
                        "observation_evidence": {
                            "accepted_frame_count": 12,
                            "lidar_rejection_count": 3,
                            "soft_miss_count": 5,
                            "last_soft_miss_reason": "camera_lidar_skew",
                        },
                        "tf_retry_elapsed_sec": 0.061,
                        "retry_exhausted": False,
                    }
                ),
                encoding="utf-8",
            )

            status = load_passive_observer_status(status_path)

        self.assertEqual(status.state, "tf_pending_exact_time")
        self.assertEqual(status.consensus_sample_count, 2)
        self.assertEqual(status.consensus_required_sample_count, 7)
        self.assertEqual(status.tf_retry_count, 4)
        self.assertEqual(status.tf_retry_elapsed_sec, 0.061)
        self.assertEqual(status.accepted_frame_count, 12)
        self.assertEqual(status.lidar_rejection_count, 3)
        self.assertEqual(status.soft_miss_count, 5)
        self.assertEqual(
            status.last_soft_miss_reason,
            "camera_lidar_skew",
        )
        self.assertFalse(status.retry_exhausted)
        self.assertIsNone(status.load_error)

    def test_qr_binding_reason_survives_terminal_tf_status(self):
        diagnostic = dict(decoded_frame_count=71, texts=["QR_004"], accepted=False,
            reason="camera_map_bearing_interval_exceeds_limit", diagnostic_only=True)
        status = self._load_payload(dict(state="tf_retry_exhausted", qr_binding_diagnostic=diagnostic))
        self.assertEqual(status.qr_binding_diagnostic, diagnostic)
        message = format_passive_observer_failure(candidate_uid="candidate", process=self._process(),
            status=status, process_evidence_path=Path("process.json"))
        self.assertIn("71", message)
        self.assertIn("camera_map_bearing_interval_exceeds_limit", message)

    def test_missing_status_is_explicit_and_does_not_raise(self) -> None:
        status = load_passive_observer_status(Path("missing-status.json"))

        self.assertEqual(status.state, "no_status")
        self.assertIn("missing", status.load_error)

    def test_registered_wrapper_does_not_change_legacy_diagnostic_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "observer_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "state": "collecting_consensus",
                        "candidate_lidar_association": {
                            "nearest_range_delta_m": 0.031,
                        },
                        "camera_registered_candidate_lidar_association": {
                            "associated": True,
                            "search_association": {
                                "nearest_range_delta_m": 0.400,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            status = load_passive_observer_status(status_path)

        self.assertAlmostEqual(status.nearest_lidar_range_delta_m, 0.031)

    def test_invalid_status_is_explicit_and_does_not_mask_child_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "observer_status.json"
            status_path.write_text("[]", encoding="utf-8")

            status = load_passive_observer_status(status_path)

        self.assertEqual(status.state, "invalid_status")
        self.assertIn("JSON object", status.load_error)

    def test_failure_distinguishes_deadline_and_includes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "observer_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "state": "tf_retry_exhausted",
                        "reason": "future extrapolation",
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
            status = load_passive_observer_status(status_path)
            process_path = Path(tmp) / "observer_process.json"
            process = PassiveObserverProcessEvidence(
                completion_kind="deadline",
                artifact_kind=None,
                artifact_path=None,
                deadline_expired=True,
                returncode=130,
                cleanup_actions=("send_sigint", "wait_after_sigint"),
                signals_sent=("SIGINT",),
            )

            message = format_passive_observer_failure(
                candidate_uid="survey_candidate_0004",
                process=process,
                status=status,
                process_evidence_path=process_path,
            )

        self.assertIn("deadline expired", message)
        self.assertIn("child_returncode=130", message)
        self.assertIn("state=tf_retry_exhausted", message)
        self.assertIn("consensus=3/7", message)
        self.assertIn("tf_retry_count=8", message)
        self.assertIn("retry_exhausted=true", message)
        self.assertIn(str(process_path), message)

    def test_only_candidate_local_reaped_deadline_is_deferrable(self) -> None:
        process = PassiveObserverProcessEvidence(
            completion_kind="deadline",
            artifact_kind=None,
            artifact_path=None,
            deadline_expired=True,
            returncode=130,
            cleanup_actions=("send_sigint", "wait_after_sigint"),
            signals_sent=("SIGINT",),
        )
        local_status = load_passive_observer_status(Path("missing-status.json"))
        local_status = type(local_status)(
            state="lidar_target_mismatch",
            reason=None,
            consensus_sample_count=0,
            consensus_required_sample_count=7,
            tf_retry_count=0,
            tf_retry_elapsed_sec=None,
            retry_exhausted=None,
            load_error=None,
        )
        systemic_status = type(local_status)(
            state="tf_retry_exhausted",
            reason=None,
            consensus_sample_count=0,
            consensus_required_sample_count=7,
            tf_retry_count=8,
            tf_retry_elapsed_sec=0.16,
            retry_exhausted=True,
            load_error=None,
        )
        transient_after_candidate_frames = type(local_status)(
            state="tf_pending_exact_time",
            reason=None,
            consensus_sample_count=0,
            consensus_required_sample_count=7,
            tf_retry_count=2,
            tf_retry_elapsed_sec=0.03,
            retry_exhausted=False,
            load_error=None,
            accepted_frame_count=340,
            tf_retry_attempted_tuple_count=342,
            tf_retry_exhausted_tuple_count=104,
        )
        exhausted_after_candidate_frames = type(local_status)(
            state="tf_retry_exhausted",
            reason=None,
            consensus_sample_count=0,
            consensus_required_sample_count=7,
            tf_retry_count=8,
            tf_retry_elapsed_sec=0.16,
            retry_exhausted=True,
            load_error=None,
            accepted_frame_count=1,
            tf_retry_attempted_tuple_count=2,
            tf_retry_exhausted_tuple_count=1,
        )
        transient_without_candidate_frames = type(local_status)(
            state="tf_pending_exact_time",
            reason=None,
            consensus_sample_count=0,
            consensus_required_sample_count=7,
            tf_retry_count=4,
            tf_retry_elapsed_sec=0.08,
            retry_exhausted=False,
            load_error=None,
            accepted_frame_count=0,
        )

        self.assertTrue(
            is_candidate_local_observer_timeout(
                process=process,
                status=local_status,
            )
        )
        self.assertFalse(
            is_candidate_local_observer_timeout(
                process=process,
                status=systemic_status,
            )
        )
        self.assertTrue(
            is_candidate_local_observer_timeout(
                process=process,
                status=transient_after_candidate_frames,
            )
        )
        self.assertTrue(
            is_candidate_local_observer_timeout(
                process=process,
                status=exhausted_after_candidate_frames,
            )
        )
        self.assertFalse(
            is_candidate_local_observer_timeout(
                process=process,
                status=transient_without_candidate_frames,
            )
        )

    def test_lidar_rejected_frames_survive_trailing_tf_deadline_state(self):
        # Final candidate_0001 snapshot from the 2026-09-07 run.  All 272
        # LiDAR rejections followed successful exact-time transform lookup;
        # accepted frames and consensus samples both stayed at zero.
        for state in (
            "tf_pending_exact_time",
            "tf_retry_exhausted",
            "metric_model_measurement_unavailable",
        ):
            with self.subTest(state=state):
                status = self._load_payload(
                    {
                        "state": state,
                        "axis_consensus": {
                            "sample_count": 0,
                            "required_sample_count": 7,
                        },
                        "tf_retry_attempt_summary": {
                            "attempted_tuple_count": 127,
                            "exhausted_tuple_count": 2,
                        },
                        "observation_evidence": {
                            "accepted_frame_count": 0,
                            "lidar_rejection_count": 272,
                            "soft_miss_count": 346,
                            "last_soft_miss_reason": "lidar_target_not_associated",
                            "poisoned": False,
                            "poison_reason": None,
                        },
                    }
                )

                self.assertTrue(
                    is_candidate_local_observer_timeout(
                        process=self._process(), status=status
                    )
                )
                self.assertEqual(status.accepted_frame_count, 0)
                self.assertEqual(status.consensus_sample_count, 0)
                self.assertEqual(status.lidar_rejection_count, 272)
                if state.startswith("tf_"):
                    self.assertEqual(
                        candidate_local_observer_timeout_basis(status),
                        "accumulated_transform_ready_candidate_frames",
                    )

    def test_tf_activity_and_soft_misses_alone_do_not_allow_deferral(self):
        for state in (
            "tf_pending_exact_time", "tf_retry_exhausted", "obsolete_detector_result",
            "stale_sensor_tuple",
        ):
            for evidence in (
                {},
                {
                    "accepted_frame_count": 0,
                    "lidar_rejection_count": 0,
                    "soft_miss_count": 346,
                    "last_soft_miss_reason": "camera_lidar_skew",
                },
            ):
                with self.subTest(state=state, evidence=evidence):
                    status = self._load_payload(
                        {
                            "state": state,
                            "axis_consensus": {"sample_count": 3},
                            "tf_retry_attempt_summary": {
                                "attempted_tuple_count": 127,
                                "exhausted_tuple_count": 2,
                            },
                            "observation_evidence": evidence,
                        }
                    )

                    self.assertFalse(
                        is_candidate_local_observer_timeout(
                            process=self._process(), status=status
                        )
                    )

    def test_accumulated_frames_do_not_hide_child_exit_or_other_status(self):
        for completion, state in (
            ("child_exit", "tf_pending_exact_time"),
            ("artifact", "tf_pending_exact_time"),
            ("child_exit", "metric_model_measurement_unavailable"),
            ("deadline", "unknown_status"),
            ("deadline", "waiting_for_synchronized_sensors"),
        ):
            with self.subTest(completion=completion, state=state):
                status = self._load_payload(
                    {
                        "state": state,
                        "observation_evidence": {
                            "accepted_frame_count": 1,
                            "lidar_rejection_count": 272,
                        },
                    }
                )

                self.assertFalse(
                    is_candidate_local_observer_timeout(
                        process=self._process(completion), status=status
                    )
                )

    def test_poisoned_identity_evidence_remains_terminal_at_deadline(self):
        for state in (
            "tf_pending_exact_time", "evidence_not_committable", "obsolete_detector_result",
            "stale_sensor_tuple",
        ):
            for poison in (
                {"poisoned": True},
                {
                    "poisoned": False,
                    "poison_reason": "conflicting_qr_ids_in_motion_epoch",
                },
            ):
                with self.subTest(state=state, poison=poison):
                    status = self._load_payload(
                        {
                            "state": state,
                            "observation_evidence": {
                                "accepted_frame_count": 1,
                                "lidar_rejection_count": 272,
                                **poison,
                            },
                        }
                    )

                    self.assertFalse(
                        is_candidate_local_observer_timeout(
                            process=self._process(), status=status
                        )
                    )
                    self.assertIsNone(candidate_local_observer_timeout_basis(status))
                    self.assertEqual(
                        status.to_dict()["observation_evidence_poisoned"],
                        poison["poisoned"],
                    )
                    message = format_passive_observer_failure(
                        candidate_uid="candidate",
                        process=self._process(),
                        status=status,
                        process_evidence_path=Path("process.json"),
                    )
                    self.assertIn("observation_evidence_poisoned=", message)
                    if "poison_reason" in poison:
                        self.assertIn(poison["poison_reason"], message)

    def test_malformed_supplied_decision_evidence_cannot_allow_deferral(self):
        malformed = [[], "bad evidence", {"poisoned": "false"}, {"poison_reason": 3}]
        for field in ("accepted_frame_count", "lidar_rejection_count"):
            for value in (-1, 1.5, True, "272", None, float("inf"), float("nan")):
                malformed.append(
                    {
                        "accepted_frame_count": 1,
                        "lidar_rejection_count": 272,
                        field: value,
                    }
                )
        for state in (
            "tf_pending_exact_time", "lidar_target_mismatch", "obsolete_detector_result",
            "stale_sensor_tuple",
        ):
            for evidence in malformed:
                with self.subTest(state=state, evidence=evidence):
                    status = self._load_payload(
                        {"state": state, "observation_evidence": evidence}
                    )

                    self.assertIsNotNone(status.load_error)
                    self.assertIsNone(candidate_local_observer_timeout_basis(status))
                    self.assertFalse(
                        is_candidate_local_observer_timeout(
                            process=self._process(), status=status
                        )
                    )

    def test_missing_or_unreadable_status_remains_terminal(self):
        missing = load_passive_observer_status(Path("missing-status.json"))
        for status in (missing, self._load_payload([]), self._load_payload({})):
            with self.subTest(state=status.state):
                self.assertFalse(
                    is_candidate_local_observer_timeout(
                        process=self._process(), status=status
                    )
                )

    def test_recorded_obsolete_result_uses_prior_processing_without_admitting_axis(self):
        status = self._load_payload(recorded_backside_timeout_status())

        self.assertTrue(
            is_candidate_local_observer_timeout(process=self._process(), status=status)
        )
        self.assertEqual(
            candidate_local_observer_timeout_basis(status),
            "accumulated_transform_ready_candidate_frames",
        )
        self.assertEqual(status.state, "obsolete_detector_result")
        self.assertEqual(status.accepted_frame_count, 0)
        self.assertEqual(status.consensus_sample_count, 0)
        self.assertEqual(status.consensus_required_sample_count, 7)
        self.assertEqual(status.lidar_rejection_count, 21)

    def test_obsolete_result_requires_explicit_unpoisoned_processing_evidence(self):
        status = self._load_payload(recorded_backside_timeout_status())
        for changed in (
            replace(status, observation_evidence_poisoned=None),
            replace(status, accepted_frame_count=0, lidar_rejection_count=0),
            replace(status, accepted_frame_count=None, lidar_rejection_count=None),
        ):
            with self.subTest(status=changed):
                self.assertFalse(
                    is_candidate_local_observer_timeout(
                        process=self._process(), status=changed
                    )
                )

    def test_recorded_front_deadline_is_independent_of_last_stale_frame(self):
        for event_line in (407, 408):
            with self.subTest(event_line=event_line):
                status = self._load_payload(recorded_front_timeout_status(event_line))
                self.assertTrue(is_candidate_local_observer_timeout(
                    process=self._process(), status=status,
                ))
                self.assertEqual(candidate_local_observer_timeout_basis(status),
                                 "accumulated_transform_ready_candidate_frames")
                self.assertIsNone(status.load_error)
                self.assertEqual(status.accepted_frame_count, 0)
                self.assertEqual(status.lidar_rejection_count, 23)
                self.assertEqual(status.consensus_sample_count, 0)
                self.assertEqual(status.peak_consensus_sample_count, 0)
                self.assertEqual(status.consensus_required_sample_count, 7)
                self.assertIs(status.observation_evidence_poisoned, False)

    def test_stale_input_requires_explicit_unpoisoned_prior_processing(self):
        status = self._load_payload(recorded_front_timeout_status())
        for changed in (
            replace(status, observation_evidence_poisoned=None),
            replace(status, observation_evidence_poisoned=True),
            replace(status, observation_evidence_poison_reason="conflicting_qr_ids"),
            replace(status, accepted_frame_count=0, lidar_rejection_count=0),
            replace(status, accepted_frame_count=None, lidar_rejection_count=None),
            replace(status, load_error="invalid observation evidence"),
        ):
            with self.subTest(status=changed):
                self.assertIsNone(candidate_local_observer_timeout_basis(changed))
                self.assertFalse(is_candidate_local_observer_timeout(
                    process=self._process(), status=changed,
                ))
        accepted_frame = replace(status, accepted_frame_count=1, lidar_rejection_count=0)
        self.assertTrue(is_candidate_local_observer_timeout(
            process=self._process(), status=accepted_frame,
        ))

    def test_stale_input_does_not_hide_child_crash_or_unsolicited_signal(self):
        status = self._load_payload(recorded_front_timeout_status())
        for returncode, signals in (
            (1, ("SIGINT",)), (-11, ("SIGINT",)), (139, ("SIGINT",)),
            (130, ()), (-2, ()), (137, ("SIGINT",)), (True, ("SIGINT",)),
        ):
            with self.subTest(returncode=returncode, signals=signals):
                self.assertFalse(is_candidate_local_observer_timeout(
                    process=replace(self._process(), returncode=returncode, signals_sent=signals),
                    status=status,
                ))
        for completion in ("child_exit", "artifact"):
            with self.subTest(completion=completion):
                self.assertFalse(is_candidate_local_observer_timeout(
                    process=self._process(completion), status=status,
                ))

    def test_crash_racing_deadline_cannot_hide_behind_prior_candidate_processing(self):
        status = self._load_payload(recorded_backside_timeout_status())
        for returncode, signals in (
            (1, ("SIGINT",)),
            (-11, ("SIGINT",)),
            (139, ("SIGINT",)),
            (130, ()),
            (-2, ()),
            (137, ("SIGINT",)),
            (True, ("SIGINT",)),
        ):
            with self.subTest(returncode=returncode, signals=signals):
                process = replace(
                    self._process(), returncode=returncode, signals_sent=signals
                )
                self.assertFalse(
                    is_candidate_local_observer_timeout(process=process, status=status)
                )

    def test_only_deadline_with_expected_cleanup_exit_can_defer_recorded_status(self):
        status = self._load_payload(recorded_backside_timeout_status())
        for returncode, signals in (
            (0, ()), (130, ("SIGINT",)), (-2, ("SIGINT",)),
            (-15, ("SIGINT", "SIGTERM")),
            (-9, ("SIGINT", "SIGTERM", "SIGKILL")),
        ):
            with self.subTest(returncode=returncode, signals=signals):
                self.assertTrue(
                    is_candidate_local_observer_timeout(
                        process=replace(
                            self._process(), returncode=returncode, signals_sent=signals
                        ),
                        status=status,
                    )
                )
        for completion in ("child_exit", "artifact"):
            with self.subTest(completion=completion):
                self.assertFalse(
                    is_candidate_local_observer_timeout(
                        process=self._process(completion), status=status
                    )
                )


if __name__ == "__main__":
    unittest.main()
