import ast
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.aufgabe04.real_robot.readiness.initial import (
    InitialReadinessRejected,
)
from scripts.aufgabe04.real_robot.readiness.preauthorization import (
    PreauthorizationReadinessConfig,
    PreauthorizationReadinessEffects,
    admit_preauthorization_readiness,
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
    }


def _outcome(
    request,
    *,
    status: str = "dry_run_ok",
    reason: str = "",
    details: dict[str, object] | None = None,
):
    if status == "dry_run_ok":
        request.semantic_log_path.parent.mkdir(parents=True, exist_ok=True)
        request.semantic_log_path.write_text(
            json.dumps(
                {
                    "event": "dry_run_completed",
                    "run_id": request.run_id,
                    "status": status,
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
        run_id=request.run_id,
        status=status,
        stop_reason=reason,
        stop_details={} if details is None else details,
        motion_published=False,
        returncode=0 if status == "dry_run_ok" else 1,
        semantic_log_path=request.semantic_log_path,
        dry_preflight_path=request.dry_preflight_path,
        odom_execution_certificate_path=(
            request.dry_odom_certificate_path
            if status == "dry_run_ok"
            else None
        ),
        dry_uncertainty_budget_path=request.dry_uncertainty_budget_path,
    )


class AutonomousPreauthorizationReadinessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.session_root = self.root / "session_1"
        self.survey_root = self.session_root / "coverage"
        self.observation_tf = (
            self.session_root / "preflight/lidar_scan_tf_before_authorization.json"
        )
        self.sensor_timing = (
            self.session_root
            / "preflight/camera_lidar_timing_before_authorization.json"
        )
        self.config = PreauthorizationReadinessConfig(
            session_root=self.session_root,
            survey_root=self.survey_root,
            coverage_plan_path=self.survey_root / "coverage_plan.json",
            session_id="session_1",
            initial_leg_index=2,
            maximum_localization_readiness_retries=1,
            observation_tf_evidence_path=self.observation_tf,
            observation_tf_evidence_sha256="f" * 64,
            sensor_timing_evidence_path=self.sensor_timing,
            sensor_timing_evidence_sha256="a" * 64,
        )

    @staticmethod
    def _sealer(calls):
        def seal(**kwargs):
            calls.append(kwargs)
            output = kwargs["output_dir"]
            return {
                "route_csv": str(output / "route.csv"),
                "diagnostics_json": str(output / "route_diagnostics.json"),
                "route_certificate_json": str(output / "route_certificate.json"),
            }

        return seal

    def _effects(
        self,
        dry_runner,
        *,
        seals=None,
        persisted=None,
        published=None,
        order=None,
        prepare=None,
        notify=None,
    ):
        seals = [] if seals is None else seals
        persisted = [] if persisted is None else persisted
        published = [] if published is None else published
        order = [] if order is None else order
        ticks = iter((10.0, 11.0, 12.0))

        def append(path, payload):
            order.append("event")
            persisted.append((path, payload))

        def publish(path, payload, *, hash_field):
            order.append("evidence")
            published.append((path, payload, hash_field))
            return "e" * 64

        return PreauthorizationReadinessEffects(
            seal_route=self._sealer(seals),
            run_dry_motion_leg=dry_runner,
            append_event=append,
            publish_hashed_json=publish,
            wall_clock=lambda: next(ticks),
            notify=(notify or (lambda _message: None)),
            prepare_localization_attempt=prepare,
        )

    def test_pass_seals_once_persists_event_then_hash_bound_evidence(self):
        seals = []
        persisted = []
        published = []
        order = []
        requests = []

        def dry_runner(request):
            requests.append(request)
            return _outcome(request)

        outcome = admit_preauthorization_readiness(
            self.config,
            self._effects(
                dry_runner,
                seals=seals,
                persisted=persisted,
                published=published,
                order=order,
            ),
        )

        self.assertTrue(outcome.result.ready)
        self.assertEqual(len(seals), 1)
        self.assertEqual(len(requests), 1)
        paths = self.config.paths
        self.assertEqual(seals[0]["source_route_csv"], paths.source_route_csv)
        self.assertEqual(seals[0]["output_dir"], paths.sealed_output_dir)
        self.assertEqual(order, ["event", "evidence"])
        self.assertEqual(persisted[0][0], paths.event_log_jsonl)
        self.assertEqual(persisted[0][1]["timestamp"], 10.0)
        evidence_path, evidence, hash_field = published[0]
        self.assertEqual(evidence_path, paths.readiness_evidence_json)
        self.assertEqual(hash_field, "initial_readiness_sha256")
        self.assertEqual(
            evidence["observation_tf_readiness_json"],
            str(self.observation_tf),
        )
        self.assertEqual(evidence["observation_tf_readiness_sha256"], "f" * 64)
        self.assertEqual(
            evidence["sensor_timing_readiness_json"],
            str(self.sensor_timing),
        )
        self.assertEqual(evidence["sensor_timing_readiness_sha256"], "a" * 64)
        self.assertFalse(evidence["motion_authorized"])
        self.assertEqual(outcome.evidence_path, paths.readiness_evidence_json)
        with self.assertRaises(FrozenInstanceError):
            outcome.evidence_sha256 = "0" * 64  # type: ignore[misc]

    def test_camera_lidar_timing_receipt_is_bound_into_hashed_readiness_evidence(
        self,
    ):
        published = []
        sensor_timing = (
            self.session_root
            / "preflight/camera_lidar_timing_before_authorization.json"
        )
        config = PreauthorizationReadinessConfig(
            **{
                **self.config.__dict__,
                "sensor_timing_evidence_path": sensor_timing,
                "sensor_timing_evidence_sha256": "a" * 64,
            }
        )

        outcome = admit_preauthorization_readiness(
            config,
            self._effects(
                lambda request: _outcome(request),
                published=published,
            ),
        )

        self.assertTrue(outcome.result.ready)
        self.assertEqual(len(published), 1)
        evidence_path, evidence, hash_field = published[0]
        self.assertEqual(evidence_path, config.paths.readiness_evidence_json)
        self.assertEqual(hash_field, "initial_readiness_sha256")
        self.assertEqual(
            evidence["sensor_timing_readiness_json"],
            str(sensor_timing),
        )
        self.assertEqual(evidence["sensor_timing_readiness_sha256"], "a" * 64)

    def test_camera_lidar_timing_binding_requires_exact_path_and_sha_pair(self):
        exact_path = (
            self.session_root
            / "preflight/camera_lidar_timing_before_authorization.json"
        )
        invalid_overrides = (
            {
                "sensor_timing_evidence_path": exact_path,
                "sensor_timing_evidence_sha256": None,
            },
            {
                "sensor_timing_evidence_path": None,
                "sensor_timing_evidence_sha256": "a" * 64,
            },
            {
                "sensor_timing_evidence_path": (
                    self.session_root / "preflight/camera_lidar_timing.json"
                ),
                "sensor_timing_evidence_sha256": "a" * 64,
            },
        )

        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                PreauthorizationReadinessConfig(
                    **{
                        **self.config.__dict__,
                        **overrides,
                    }
                )

    def test_retry_reuses_one_seal_and_persists_attempt_events_in_order(self):
        seals = []
        persisted = []
        requests = []

        def dry_runner(request):
            requests.append(request)
            if request.attempt_index == 0:
                return _outcome(
                    request,
                    status="preflight_failed",
                    reason=_retryable_reason(),
                    details=_retryable_details(),
                )
            return _outcome(request)

        outcome = admit_preauthorization_readiness(
            self.config,
            self._effects(dry_runner, seals=seals, persisted=persisted),
        )

        self.assertTrue(outcome.result.ready)
        self.assertEqual(len(seals), 1)
        self.assertEqual(len(requests), 2)
        self.assertIs(requests[0].sealed_route, requests[1].sealed_route)
        self.assertEqual(
            [payload["event"] for _path, payload in persisted],
            [
                "preauthorization_initial_readiness_retry_scheduled",
                "preauthorization_initial_readiness_passed",
            ],
        )
        self.assertEqual(
            [payload["timestamp"] for _path, payload in persisted],
            [10.0, 11.0],
        )

    def test_readiness_retry_does_not_instruct_post_seal_initialpose_click(self):
        notices: list[str] = []

        def dry_runner(request):
            if request.attempt_index == 0:
                return _outcome(
                    request,
                    status="preflight_failed",
                    reason=_retryable_reason(),
                    details=_retryable_details(),
                )
            return _outcome(request)

        outcome = admit_preauthorization_readiness(
            self.config,
            self._effects(dry_runner, notify=notices.append),
        )

        self.assertTrue(outcome.result.ready)
        retry_notice = "\n".join(notices)
        self.assertIn("do not click RViz 2D Pose Estimate", retry_notice)
        self.assertIn("sealed-route retry", retry_notice)

    def test_prepare_localization_attempt_runs_before_each_dry_attempt(self):
        order = []

        def dry_runner(request):
            order.append(("dry", request.attempt_index))
            if request.attempt_index == 0:
                return _outcome(
                    request,
                    status="preflight_failed",
                    reason=_retryable_reason(),
                    details=_retryable_details(),
                )
            return _outcome(request)

        def prepare(request):
            order.append(("prepare", request.attempt_index))

        outcome = admit_preauthorization_readiness(
            self.config,
            self._effects(dry_runner, prepare=prepare),
        )

        self.assertTrue(outcome.result.ready)
        self.assertEqual(
            order,
            [
                ("prepare", 0),
                ("dry", 0),
                ("prepare", 1),
                ("dry", 1),
            ],
        )

    def test_rejection_is_structured_only_after_event_and_evidence_persist(self):
        persisted = []
        published = []
        order = []

        def rejected(request):
            return _outcome(
                request,
                status="preflight_failed",
                reason="static route certificate mismatch",
                details={"fail_closed": True},
            )

        with self.assertRaises(InitialReadinessRejected) as caught:
            admit_preauthorization_readiness(
                self.config,
                self._effects(
                    rejected,
                    persisted=persisted,
                    published=published,
                    order=order,
                ),
            )

        self.assertEqual(order, ["event", "evidence"])
        self.assertEqual(len(persisted), 1)
        self.assertEqual(len(published), 1)
        fields = caught.exception.to_failure_fields()
        self.assertFalse(fields["motion_authorized"])
        self.assertFalse(fields["motion_published"])
        self.assertFalse(fields["permit_issued"])
        self.assertEqual(
            fields["initial_readiness_json"],
            str(self.config.paths.readiness_evidence_json),
        )

    def test_paths_keep_source_sealed_events_and_evidence_separate(self):
        paths = self.config.paths
        source = {paths.source_route_csv, paths.source_diagnostics_json}
        sealed = {
            paths.sealed_route_csv,
            paths.sealed_diagnostics_json,
            paths.sealed_route_certificate_json,
        }
        self.assertTrue(source.isdisjoint(sealed))
        self.assertNotIn(paths.event_log_jsonl, source | sealed)
        self.assertNotIn(paths.readiness_evidence_json, source | sealed)
        self.assertEqual(
            paths.readiness_root,
            self.session_root / "authorization_readiness/coverage_leg_002",
        )
        self.assertEqual(
            paths.source_route_csv,
            self.survey_root / "legs/leg_002_route.csv",
        )
        with self.assertRaises(ValueError):
            PreauthorizationReadinessConfig(
                **{
                    **self.config.__dict__,
                    "observation_tf_evidence_path": (
                        paths.readiness_root / "scan_tf.json"
                    ),
                }
            )
        with self.assertRaises(ValueError):
            PreauthorizationReadinessConfig(
                **{
                    **self.config.__dict__,
                    "sensor_timing_evidence_path": (
                        paths.readiness_root / "sensor_timing.json"
                    ),
                }
            )

    def test_module_has_no_parent_runner_ros_subprocess_or_prompt_dependency(self):
        module_path = (
            Path(__file__).resolve().parents[2]
            / "scripts/aufgabe04/real_robot/readiness/preauthorization.py"
        )
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertTrue(
            {"rclpy", "subprocess", "tf2_ros", "geometry_msgs"}.isdisjoint(
                imported
            )
        )
        self.assertNotIn(
            "scripts.aufgabe04.real_robot.autonomous_runner.runtime",
            imported,
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "input"
                for node in ast.walk(tree)
            )
        )


if __name__ == "__main__":
    unittest.main()
