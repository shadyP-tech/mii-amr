from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.aufgabe04.real_robot.autonomous_initial_readiness import (
    INITIAL_READINESS_PHASE,
    InitialReadinessContractError,
    InitialReadinessResult,
    InitialReadinessRejected,
    SealedRoutePaths,
    run_initial_readiness,
)


def _retryable_reason() -> str:
    return (
        "odom execution admission failed: route uncertainty budget exhausted: "
        "limiting_segment=0 remaining_margin=-0.010000 m"
    )


def _retryable_details() -> dict[str, object]:
    return {
        "fault_code": "odom_execution_admission_failed",
        "execution_pose_owner": "odom",
        "global_consistency_monitor": "amcl",
        "fail_closed": True,
        "motion_published": False,
        "margin": {"remaining_m": -0.01},
    }


def _outcome(
    request,
    *,
    status: str = "dry_run_ok",
    reason: str = "",
    details: object | None = None,
    motion_published: object = False,
    returncode: int | None = None,
    run_id: str | None = None,
    semantic_log_path: Path | None = None,
    certificate_path: Path | None = None,
    preflight_path: Path | None = None,
    budget_path: Path | None = None,
    materialize_success_evidence: bool = True,
):
    if details is None:
        details = {}
    if returncode is None:
        returncode = 0 if status == "dry_run_ok" else 1
    if status == "dry_run_ok" and materialize_success_evidence:
        request.semantic_log_path.parent.mkdir(parents=True, exist_ok=True)
        request.semantic_log_path.write_text(
            json.dumps(
                {
                    "event": "dry_run_completed",
                    "run_id": request.run_id,
                    "status": "dry_run_ok",
                    "motion_published": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        for path in (
            request.dry_preflight_path,
            request.dry_odom_certificate_path,
            request.dry_uncertainty_budget_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        run_id=request.run_id if run_id is None else run_id,
        status=status,
        stop_reason=reason,
        stop_details=details,
        motion_published=motion_published,
        returncode=returncode,
        semantic_log_path=(
            request.semantic_log_path
            if semantic_log_path is None
            else semantic_log_path
        ),
        dry_preflight_path=(
            request.dry_preflight_path
            if preflight_path is None
            else preflight_path
        ),
        odom_execution_certificate_path=(
            request.dry_odom_certificate_path
            if certificate_path is None and status == "dry_run_ok"
            else certificate_path
        ),
        dry_uncertainty_budget_path=(
            request.dry_uncertainty_budget_path
            if budget_path is None
            else budget_path
        ),
    )


class AutonomousInitialReadinessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.session_root = self.root / "session"
        self.sealed = SealedRoutePaths(
            route_csv=self.root / "route.csv",
            diagnostics_json=self.root / "diagnostics.json",
            route_certificate_json=self.root / "certificate.json",
        )

    def run_readiness(self, runner, *, retries: int = 2):
        return run_initial_readiness(
            sealed_route=self.sealed,
            session_root=self.session_root,
            run_id_prefix="mission_initial_readiness",
            maximum_retries=retries,
            dry_runner=runner,
        )

    def test_immediate_success_is_frozen_and_persist_ready(self):
        requests = []

        def runner(request):
            requests.append(request)
            return _outcome(request)

        result = self.run_readiness(runner)

        self.assertTrue(result.ready)
        self.assertEqual(result.reason, "sealed_route_dry_readiness_passed")
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(requests[0].run_id, "mission_initial_readiness_000")
        self.assertIs(requests[0].sealed_route, self.sealed)
        self.assertEqual(
            requests[0].dry_uncertainty_budget_path,
            self.session_root
            / "odom_execution"
            / "mission_initial_readiness_000_dry_uncertainty_budget.json",
        )
        with self.assertRaises(FrozenInstanceError):
            result.ready = False  # type: ignore[misc]

        evidence = result.to_evidence()
        self.assertEqual(evidence["run_id"], "mission_initial_readiness_000")
        self.assertEqual(evidence["status"], "dry_run_ok")
        self.assertEqual(evidence["reason"], "")
        self.assertEqual(evidence["details"], {})
        self.assertFalse(evidence["retry_decision"]["retryable"])
        self.assertFalse(evidence["typed_run_requested"])
        self.assertFalse(evidence["operator_input_requested"])
        self.assertFalse(evidence["motion_authorized"])
        self.assertFalse(evidence["motion_published"])
        self.assertFalse(evidence["permit_issued"])
        self.assertFalse(evidence["reusable_as_motion_permit"])
        self.assertTrue(evidence["route_limits_unchanged"])
        for field in (
            "semantic_log_sha256",
            "dry_preflight_sha256",
            "dry_odom_certificate_sha256",
            "dry_uncertainty_budget_sha256",
        ):
            self.assertRegex(evidence[field], r"^[0-9a-f]{64}$")
        with self.assertRaises(ValueError):
            result.to_failure_fields()

    def test_only_existing_retry_classifier_can_schedule_bounded_retry(self):
        requests = []

        def runner(request):
            requests.append(request)
            if request.attempt_index == 0:
                return _outcome(
                    request,
                    status="preflight_failed",
                    reason=_retryable_reason(),
                    details=_retryable_details(),
                )
            return _outcome(request)

        result = self.run_readiness(runner, retries=1)

        self.assertTrue(result.ready)
        self.assertEqual(
            [request.run_id for request in requests],
            [
                "mission_initial_readiness_000",
                "mission_initial_readiness_001",
            ],
        )
        first, second = result.attempts
        self.assertTrue(first.retry_decision.retryable)
        self.assertEqual(
            first.retry_decision.reason,
            "fresh_no_motion_admission_allowed",
        )
        self.assertTrue(first.retry_scheduled)
        self.assertFalse(second.retry_scheduled)
        events = result.to_events()
        self.assertEqual(
            [event["event"] for event in events],
            [
                "preauthorization_initial_readiness_retry_scheduled",
                "preauthorization_initial_readiness_passed",
            ],
        )
        required = {
            "run_id",
            "status",
            "reason",
            "details",
            "semantic_log_path",
            "dry_odom_certificate_path",
            "dry_uncertainty_budget_path",
            "retry_decision",
            "motion_authorized",
            "motion_published",
            "route_limits_unchanged",
        }
        for event in events:
            self.assertTrue(required.issubset(event))
            self.assertFalse(event["motion_authorized"])
            self.assertFalse(event["motion_published"])
            self.assertTrue(event["route_limits_unchanged"])

    def test_retry_budget_exhaustion_is_a_structured_no_motion_failure(self):
        requests = []

        def runner(request):
            requests.append(request)
            return _outcome(
                request,
                status="preflight_failed",
                reason=_retryable_reason(),
                details=_retryable_details(),
            )

        result = self.run_readiness(runner, retries=1)

        self.assertFalse(result.ready)
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            result.reason,
            "localization_readiness_retry_budget_exhausted",
        )
        self.assertFalse(result.attempts[-1].retry_scheduled)
        fields = result.to_failure_fields()
        self.assertEqual(fields["failure_phase"], INITIAL_READINESS_PHASE)
        self.assertFalse(fields["typed_run_requested"])
        self.assertEqual(fields["initial_readiness_attempt_count"], 2)
        self.assertEqual(
            fields["initial_readiness_last_details"]["margin"],
            {"remaining_m": -0.01},
        )
        self.assertFalse(fields["motion_authorized"])
        self.assertFalse(fields["motion_published"])

    def test_nonretryable_failure_stops_after_one_attempt(self):
        calls = 0

        def runner(request):
            nonlocal calls
            calls += 1
            return _outcome(
                request,
                status="preflight_failed",
                reason="camera topic missing",
                details={"fail_closed": True},
            )

        result = self.run_readiness(runner, retries=8)

        self.assertFalse(result.ready)
        self.assertEqual(calls, 1)
        self.assertEqual(
            result.reason,
            "nonretryable_dry_outcome:fault_code_not_retryable",
        )
        self.assertFalse(result.attempts[0].retry_decision.retryable)

        rejection = InitialReadinessRejected(
            result,
            evidence_path=self.root / "readiness.json",
            evidence_sha256="a" * 64,
        )
        self.assertIn(result.reason, str(rejection))
        rejection_fields = rejection.to_failure_fields()
        self.assertEqual(
            rejection_fields["initial_readiness_json"],
            str(self.root / "readiness.json"),
        )
        self.assertEqual(rejection_fields["initial_readiness_sha256"], "a" * 64)

    def test_motion_report_is_a_contract_violation_not_no_motion_evidence(self):
        def runner(request):
            return _outcome(request, motion_published=True)

        with self.assertRaises(InitialReadinessContractError) as caught:
            self.run_readiness(runner)

        self.assertEqual(
            caught.exception.reason_code,
            "dry_runner_reported_motion_published",
        )
        fields = caught.exception.to_failure_fields()
        self.assertFalse(fields["typed_run_requested"])
        self.assertTrue(fields["motion_published"])
        self.assertFalse(fields["motion_authorized"])

    def test_malformed_runner_outputs_fail_closed_without_retry(self):
        cases = {
            "wrong_run_id": lambda request: _outcome(
                request,
                run_id="another_run",
            ),
            "wrong_semantic_log": lambda request: _outcome(
                request,
                semantic_log_path=self.root / "other.jsonl",
            ),
            "non_mapping_details": lambda request: _outcome(
                request,
                status="preflight_failed",
                reason="bad",
                details=[],
            ),
            "status_exit_mismatch": lambda request: _outcome(
                request,
                returncode=1,
            ),
            "missing_fields": lambda _request: object(),
        }
        for label, runner in cases.items():
            with self.subTest(label=label), self.assertRaises(
                InitialReadinessContractError
            ):
                self.run_readiness(runner, retries=9)

    def test_synthetic_zero_exit_without_artifacts_cannot_be_ready(self):
        def runner(request):
            return _outcome(
                request,
                materialize_success_evidence=False,
            )

        with self.assertRaises(InitialReadinessContractError) as caught:
            self.run_readiness(runner)

        self.assertEqual(
            caught.exception.reason_code,
            "dry_success_evidence_invalid",
        )

    def test_success_requires_exact_artifact_paths_and_normal_files(self):
        cases = ("semantic_log", "preflight", "certificate", "budget")
        for label in cases:
            with self.subTest(label=label):
                def runner(request, label=label):
                    outcome = _outcome(request)
                    target = {
                        "semantic_log": request.semantic_log_path,
                        "preflight": request.dry_preflight_path,
                        "certificate": request.dry_odom_certificate_path,
                        "budget": request.dry_uncertainty_budget_path,
                    }[label]
                    target.unlink()
                    target.mkdir()
                    return outcome

                with self.assertRaises(InitialReadinessContractError) as caught:
                    run_initial_readiness(
                        sealed_route=self.sealed,
                        session_root=self.session_root / label,
                        run_id_prefix="mission_initial_readiness",
                        maximum_retries=2,
                        dry_runner=runner,
                    )
                self.assertEqual(
                    caught.exception.reason_code,
                    "dry_success_evidence_invalid",
                )

        def wrong_certificate(request):
            return _outcome(
                request,
                certificate_path=self.root / "other_certificate.json",
            )

        with self.assertRaises(InitialReadinessContractError) as caught:
            self.run_readiness(wrong_certificate)
        self.assertEqual(
            caught.exception.reason_code,
            "outcome_certificate_path_mismatch",
        )

        def symlink_budget(request):
            outcome = _outcome(request)
            real_budget = self.root / "real_budget.json"
            real_budget.write_text("{}\n", encoding="utf-8")
            request.dry_uncertainty_budget_path.unlink()
            request.dry_uncertainty_budget_path.symlink_to(real_budget)
            return outcome

        with self.assertRaises(InitialReadinessContractError) as caught:
            run_initial_readiness(
                sealed_route=self.sealed,
                session_root=self.session_root / "symlink",
                run_id_prefix="mission_initial_readiness",
                maximum_retries=2,
                dry_runner=symlink_budget,
            )
        self.assertEqual(
            caught.exception.reason_code,
            "dry_success_evidence_invalid",
        )

    def test_result_public_invariants_reject_inconsistent_construction(self):
        result = self.run_readiness(_outcome)
        invalid_builders = (
            lambda: replace(result, attempts=()),
            lambda: replace(result, ready=False),
            lambda: replace(
                result,
                maximum_retry_count=0,
                attempts=(
                    replace(result.final_attempt, maximum_retry_count=1),
                ),
            ),
            lambda: replace(
                result,
                attempts=(replace(result.final_attempt, attempt_index=1),),
            ),
            lambda: replace(
                result,
                attempts=(
                    replace(result.final_attempt, retry_scheduled=True),
                ),
            ),
        )
        for build in invalid_builders:
            with self.assertRaises(ValueError):
                build()
        self.assertIsInstance(result, InitialReadinessResult)

    def test_runner_exception_is_wrapped_with_current_expected_paths(self):
        def runner(_request):
            raise OSError("child unavailable")

        with self.assertRaises(InitialReadinessContractError) as caught:
            self.run_readiness(runner)

        self.assertEqual(caught.exception.reason_code, "dry_runner_raised")
        fields = caught.exception.to_failure_fields()
        self.assertEqual(
            fields["initial_readiness_last_run_id"],
            "mission_initial_readiness_000",
        )
        self.assertIsNone(fields["motion_published"])

    def test_input_counts_prefix_and_paths_are_validated_before_runner(self):
        calls = 0

        def runner(request):
            nonlocal calls
            calls += 1
            return _outcome(request)

        invalid_calls = (
            {"maximum_retries": -1},
            {"maximum_retries": True},
            {"run_id_prefix": "bad/prefix"},
            {"session_root": Path(".")},
        )
        base = {
            "sealed_route": self.sealed,
            "session_root": self.session_root,
            "run_id_prefix": "valid_prefix",
            "maximum_retries": 0,
            "dry_runner": runner,
        }
        for mutation in invalid_calls:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                run_initial_readiness(**{**base, **mutation})
        self.assertEqual(calls, 0)

        with self.assertRaises(ValueError):
            SealedRoutePaths(
                route_csv=self.root / "route.json",
                diagnostics_json=self.root / "diagnostics.json",
                route_certificate_json=self.root / "certificate.json",
            )
        with self.assertRaises(ValueError):
            SealedRoutePaths.from_mapping({"route_csv": self.root / "r.csv"})
        with self.assertRaises(ValueError):
            SealedRoutePaths.from_mapping(
                {
                    **self.sealed.to_mapping(),
                    "unexpected_authority": "RUN",
                }
            )


if __name__ == "__main__":
    unittest.main()
