import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.real_robot.observer.diagnostics import (
    format_passive_observer_failure,
    is_candidate_local_observer_timeout,
    load_passive_observer_status,
)
from scripts.aufgabe04.real_robot.observer.process import (
    PassiveObserverProcessEvidence,
)


class PassiveObserverDiagnosticsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
