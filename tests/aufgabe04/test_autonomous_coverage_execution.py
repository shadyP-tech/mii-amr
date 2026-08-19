import ast
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.aufgabe04.navigation.localization_ownership import (
    evaluate_global_consistency_monitor,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.navigation.odom_route_adapter import (
    OdomExecutionContext,
    evaluate_map_odom_continuity,
)
from scripts.aufgabe04.navigation.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH,
    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome
from scripts.aufgabe04.real_robot.autonomous_coverage_execution import (
    CoverageLegConfig,
    CoverageLegEffects,
    MissionLegPermitContext,
    RuntimeLocalizationPermitContext,
    execute_coverage_leg_with_replans,
)
from scripts.aufgabe04.real_robot.autonomous_startup_reseal import (
    StartupResealPermitContext,
)


def _runtime_localization_stop_details() -> dict[str, object]:
    return {
        "reason": "global localization consistency requires zero and reseal",
        "fault_code": "localization_reseal_required",
        "source": "global_consistency_monitor",
        "execution_pose_owner": "odom",
        "global_consistency_monitor": "amcl",
        "monitor_action": "FORCE_ZERO_RESEAL",
        "fail_closed": True,
        "continuity": {
            "accepted": False,
            "requires_zero_cycle": True,
            "requires_reseal": True,
            "decision": "force_zero_reseal",
            "reason": "map_from_odom_yaw_drift",
            "fail_closed": True,
        },
    }


def _prestart_localization_stop_details(
    *,
    tf_warning: str = "",
) -> dict[str, object]:
    context = OdomExecutionContext(
        map_frame="map",
        odom_frame="odom",
        base_frame="base_footprint",
        frozen_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
        certificate_sha256="a" * 64,
        max_map_from_odom_translation_drift_m=0.03,
        max_map_from_odom_yaw_drift_rad=0.04,
    )
    live_map_from_odom = (
        None
        if tf_warning
        else PlanarTransform2D(0.08, 0.0, 0.0)
    )
    continuity = evaluate_map_odom_continuity(
        context,
        live_map_from_odom,
    )
    monitor = evaluate_global_consistency_monitor(
        reseal_required=True,
        diagnostic_warning=tf_warning,
    )
    reason = "global localization consistency requires zero and reseal"
    return {
        "reason": reason,
        "fault_code": "localization_reseal_required",
        "source": "global_consistency_monitor",
        "execution_phase": "before_motion",
        "phase": "initial_runtime_input_wait",
        "execution_pose_owner": "odom",
        "global_consistency_monitor": "amcl",
        "monitor_action": monitor.action,
        "monitor_reason": monitor.reason,
        "monitor_warning": monitor.diagnostic_warning,
        "motion_published": False,
        "continuity": continuity.to_evidence(),
        "fail_closed": True,
    }


def _outcome(
    root: Path,
    *,
    run_id: str,
    status: str,
    stop_reason: str = "",
    stop_details: dict[str, object] | None = None,
    motion_published: bool = False,
) -> MotionLegOutcome:
    semantic_log = root / f"{run_id}.jsonl"
    semantic_log.write_text("{}\n")
    return MotionLegOutcome(
        run_id=run_id,
        status=status,
        stop_reason=stop_reason,
        stop_details={} if stop_details is None else stop_details,
        motion_published=motion_published,
        returncode=0 if status == "completed" else 1,
        semantic_log_path=semantic_log,
    )


def _startup_mismatch_outcome(
    root: Path,
    *,
    run_id: str,
    route_pose: object,
) -> MotionLegOutcome:
    return _outcome(
        root,
        run_id=run_id,
        status="stopped",
        stop_reason="pose outside certified startup segment",
        stop_details={
            "source": "execution_route_certificate",
            "phase": "before_motion_confirmation",
            "route_pose": route_pose,
        },
        motion_published=False,
    )


def _prestart_localization_outcome(
    root: Path,
    *,
    run_id: str,
    tf_warning: str = "",
    motion_published: bool = False,
) -> MotionLegOutcome:
    details = _prestart_localization_stop_details(tf_warning=tf_warning)
    return _outcome(
        root,
        run_id=run_id,
        status="stopped",
        stop_reason=str(details["reason"]),
        stop_details=details,
        motion_published=motion_published,
    )


def _localization_readiness_rejection(
    root: Path,
    *,
    run_id: str,
) -> MotionLegOutcome:
    return _outcome(
        root,
        run_id=run_id,
        status="preflight_failed",
        stop_reason=(
            "odom execution admission failed: route uncertainty budget "
            "exhausted: temporary AMCL spread"
        ),
        stop_details={
            "fault_code": "odom_execution_admission_failed",
            "fail_closed": True,
            "execution_pose_owner": "odom",
            "global_consistency_monitor": "amcl",
        },
        motion_published=False,
    )


class AutonomousCoverageExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = SimpleNamespace(robot_radius_m=0.105)
        self.runtime = object()

    def _config(self, **overrides) -> CoverageLegConfig:
        values = {
            "session_id": "mission",
            "map_yaml": Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml"),
            "semantic_map_id": "arena_1p898x3p9_auto",
            "runtime": self.runtime,
            "robot_radius_m": 0.105,
            "max_blockage_replans_per_leg": 2,
            "max_startup_reseals_per_leg": 2,
            "max_runtime_localization_reseals_per_leg": 1,
            "max_localization_readiness_retries_per_leg": 2,
            "localization_branch_proof_id": "known_start_marker_20260807",
            "uncertainty_sigma_multiplier": 2.0,
        }
        values.update(overrides)
        return CoverageLegConfig(**values)

    def test_config_rejects_invalid_safety_budgets_and_geometry(self):
        invalid = (
            {"max_startup_reseals_per_leg": -1},
            {"max_runtime_localization_reseals_per_leg": True},
            {"robot_radius_m": 0.0},
            {"uncertainty_sigma_multiplier": float("nan")},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                self._config(**override)

    @staticmethod
    def _paths(root: Path) -> dict[str, Path]:
        source_route = root / "route.csv"
        source_diagnostics = root / "diagnostics.json"
        source_route.write_text("source\n")
        source_diagnostics.write_text("{}\n")
        return {
            "session_root": root / "session",
            "survey_root": root / "survey",
            "plan_path": root / "coverage_plan.json",
            "source_route": source_route,
            "source_diagnostics": source_diagnostics,
        }

    @staticmethod
    def _sealed(root: Path) -> dict[str, str]:
        paths = {
            "route_csv": root / "sealed_route.csv",
            "diagnostics_json": root / "sealed_diagnostics.json",
            "route_certificate_json": root / "route_certificate.json",
        }
        for path in paths.values():
            path.write_text("{}\n", encoding="utf-8")
        return {name: str(path) for name, path in paths.items()}

    def test_module_has_no_parent_ros_process_or_prompt_import(self):
        import scripts.aufgabe04.real_robot.autonomous_coverage_execution as module

        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertFalse(
            any("run_autonomous_stand_exploration" in name for name in imported)
        )
        self.assertNotIn("subprocess", imported)
        self.assertNotIn("rclpy", imported)
        self.assertNotIn("rospy", imported)
        loaded_names = {
            name.id for name in ast.walk(tree) if isinstance(name, ast.Name)
        }
        self.assertNotIn("input", loaded_names)

    def test_completed_leg_uses_one_injected_child_and_routine_permit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            completed = _outcome(
                root,
                run_id="mission_coverage_000",
                status="completed",
                motion_published=True,
            )
            run = Mock(return_value=completed)
            seal = Mock(return_value=self._sealed(root))
            admit = Mock()

            outcome = execute_coverage_leg_with_replans(
                profile=self.profile,
                config=self._config(),
                effects=CoverageLegEffects(
                    run_motion_leg=run,
                    admit_preplanning_localization=admit,
                    seal_route=seal,
                ),
                **paths,
                leg_index=0,
                target_viewpoint_id="survey_vp_001",
                mission_leg_motion_authorization_json=(
                    root / "mission_leg_authorization.json"
                ),
            )

            self.assertIs(outcome, completed)
            run.assert_called_once()
            seal.assert_called_once()
            admit.assert_not_called()
            call = run.call_args.kwargs
            self.assertEqual(call["run_id"], "mission_coverage_000")
            self.assertFalse(call["require_fresh_confirmation"])
            self.assertIsInstance(
                call["mission_leg_permit_context"],
                MissionLegPermitContext,
            )
            self.assertEqual(
                call["coverage_transient_replan"]["target_viewpoint_id"],
                "survey_vp_001",
            )

    def test_no_motion_readiness_retry_reseals_same_route_with_same_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            rejected = _outcome(
                root,
                run_id="mission_coverage_000",
                status="preflight_failed",
                stop_reason=(
                    "odom execution admission failed: route uncertainty budget "
                    "exhausted: temporary AMCL spread"
                ),
                stop_details={
                    "fault_code": "odom_execution_admission_failed",
                    "fail_closed": True,
                    "execution_pose_owner": "odom",
                    "global_consistency_monitor": "amcl",
                },
                motion_published=False,
            )
            completed = _outcome(
                root,
                run_id="mission_coverage_000_localization_readiness_001",
                status="completed",
                motion_published=True,
            )
            run = Mock(side_effect=(rejected, completed))
            seal = Mock(return_value=self._sealed(root))
            admit = Mock()

            outcome = execute_coverage_leg_with_replans(
                profile=self.profile,
                config=self._config(),
                effects=CoverageLegEffects(
                    run_motion_leg=run,
                    admit_preplanning_localization=admit,
                    seal_route=seal,
                ),
                **paths,
                leg_index=0,
                target_viewpoint_id="survey_vp_001",
            )

            self.assertIs(outcome, completed)
            self.assertEqual(seal.call_count, 2)
            self.assertEqual(
                seal.call_args_list[0].kwargs["source_route_csv"],
                seal.call_args_list[1].kwargs["source_route_csv"],
            )
            self.assertEqual(
                run.call_args_list[1].kwargs["run_id"],
                "mission_coverage_000_localization_readiness_001",
            )
            self.assertEqual(
                run.call_args_list[0].kwargs["coverage_transient_replan"][
                    "max_replans"
                ],
                run.call_args_list[1].kwargs["coverage_transient_replan"][
                    "max_replans"
                ],
            )
            events = [
                json.loads(line)
                for line in (
                    paths["session_root"] / "adaptive_replans.jsonl"
                ).read_text().splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                ["localization_readiness_retry_scheduled"],
            )
            self.assertTrue(events[0]["fresh_nomotion_amcl_preflight_required"])
            self.assertTrue(events[0]["route_limits_unchanged"])

    def test_startup_mismatch_replans_and_requests_fresh_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            replacement_route = root / "replacement_route.csv"
            replacement_diagnostics = root / "replacement_diagnostics.json"
            rejected = _outcome(
                root,
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason="pose outside certified startup segment",
                stop_details={
                    "source": "execution_route_certificate",
                    "phase": "before_motion_confirmation",
                    "route_pose": {
                        "x_m": -0.50,
                        "y_m": -0.62,
                        "yaw_rad": 1.70,
                    },
                },
                motion_published=False,
            )
            completed = _outcome(
                root,
                run_id="mission_coverage_000_startup_reseal_001",
                status="completed",
                motion_published=True,
            )
            run = Mock(side_effect=(rejected, completed))
            seal = Mock(return_value=self._sealed(root))
            admitted_pose = Pose2D(-0.48, -0.60, 1.69)
            admit = Mock(return_value=admitted_pose)
            replan = Mock(
                return_value={
                    "route_csv": str(replacement_route),
                    "diagnostics_json": str(replacement_diagnostics),
                    "summary_json": str(root / "startup_reseal_summary.json"),
                }
            )

            execute_coverage_leg_with_replans(
                profile=self.profile,
                config=self._config(),
                effects=CoverageLegEffects(
                    run_motion_leg=run,
                    admit_preplanning_localization=admit,
                    seal_route=seal,
                    replan_startup_source=replan,
                ),
                **paths,
                leg_index=0,
                target_viewpoint_id="survey_vp_001",
            )

            self.assertEqual(run.call_count, 2)
            self.assertFalse(
                run.call_args_list[0].kwargs["require_fresh_confirmation"]
            )
            self.assertTrue(
                run.call_args_list[1].kwargs["require_fresh_confirmation"]
            )
            self.assertEqual(
                run.call_args_list[1].kwargs["fresh_confirmation_reason"],
                "startup",
            )
            self.assertEqual(
                seal.call_args_list[1].kwargs["source_route_csv"],
                replacement_route,
            )
            self.assertEqual(
                replan.call_args.kwargs["expected_target_viewpoint_id"],
                "survey_vp_001",
            )
            self.assertEqual(
                replan.call_args.kwargs["current_pose"], admitted_pose
            )
            admit.assert_called_once()
            events = [
                json.loads(line)
                for line in (
                    paths["session_root"] / "adaptive_replans.jsonl"
                ).read_text().splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "startup_reseal_started",
                    "startup_localization_admitted",
                    "startup_pose_route_resealed",
                    "startup_reseal_route_sealed",
                ],
            )
            self.assertTrue(events[2]["fresh_confirmation_required"])

    def test_prestart_localization_drift_replans_same_target_with_exact_permit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            rejected = _prestart_localization_outcome(
                root,
                run_id="mission_coverage_000",
            )
            completed = _outcome(
                root,
                run_id="mission_coverage_000_startup_reseal_001",
                status="completed",
                motion_published=True,
            )
            run = Mock(side_effect=(rejected, completed))
            seal = Mock(return_value=self._sealed(root))
            admitted_pose = Pose2D(-0.48, -0.60, 1.69)
            admit = Mock(return_value=admitted_pose)
            replacement_route = root / "prestart_replacement.csv"
            replacement_diagnostics = root / "prestart_replacement.json"
            replan = Mock(
                return_value={
                    "route_csv": str(replacement_route),
                    "diagnostics_json": str(replacement_diagnostics),
                    "summary_json": str(root / "prestart_source_summary.json"),
                }
            )

            outcome = execute_coverage_leg_with_replans(
                profile=self.profile,
                config=self._config(),
                effects=CoverageLegEffects(
                    run_motion_leg=run,
                    admit_preplanning_localization=admit,
                    seal_route=seal,
                    replan_startup_source=replan,
                ),
                **paths,
                leg_index=0,
                target_viewpoint_id="survey_vp_001",
                startup_reseal_motion_authorization_json=(
                    root / "startup_master.json"
                ),
            )

            self.assertIs(outcome, completed)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(seal.call_count, 2)
            admit.assert_called_once()
            self.assertEqual(
                replan.call_args.kwargs["expected_target_viewpoint_id"],
                "survey_vp_001",
            )
            self.assertEqual(
                replan.call_args.kwargs["current_pose"],
                admitted_pose,
            )
            self.assertEqual(
                seal.call_args_list[1].kwargs["source_route_csv"],
                replacement_route,
            )
            retry = run.call_args_list[1].kwargs
            context = retry["startup_reseal_permit_context"]
            self.assertIsInstance(context, StartupResealPermitContext)
            self.assertEqual(
                context.recovery_source_kind,
                STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
            )
            self.assertEqual(context.target_viewpoint_id, "survey_vp_001")
            self.assertEqual(context.rejected_run_id, rejected.run_id)
            self.assertTrue(retry["require_fresh_confirmation"])
            self.assertEqual(retry["fresh_confirmation_reason"], "startup")
            summary = json.loads(
                context.startup_reseal_summary_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["recovery_source_kind"],
                STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
            )
            events = [
                json.loads(line)
                for line in (
                    paths["session_root"] / "adaptive_replans.jsonl"
                ).read_text().splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "startup_reseal_started",
                    "startup_localization_admitted",
                    "prestart_localization_route_resealed",
                    "startup_reseal_route_sealed",
                ],
            )
            decision = events[0]["prestart_localization_reseal_decision"]
            self.assertEqual(decision["recovery_action"], "fresh_localization_reseal")
            self.assertTrue(decision["requires_fresh_localization"])
            self.assertTrue(decision["requires_new_route_certificate"])
            self.assertFalse(decision["automatic_motion_authorized"])

    def test_prestart_missing_stale_tf_uses_bounded_warmup_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            rejected = _prestart_localization_outcome(
                root,
                run_id="mission_coverage_000",
                tf_warning="stale_map_from_odom",
            )
            completed = _outcome(
                root,
                run_id="mission_coverage_000_startup_reseal_001",
                status="completed",
                motion_published=True,
            )
            run = Mock(side_effect=(rejected, completed))
            replan = Mock(
                return_value={
                    "route_csv": str(root / "tf_replacement.csv"),
                    "diagnostics_json": str(root / "tf_replacement.json"),
                    "summary_json": str(root / "tf_source_summary.json"),
                }
            )
            events: list[dict[str, object]] = []

            execute_coverage_leg_with_replans(
                profile=self.profile,
                config=self._config(),
                effects=CoverageLegEffects(
                    run_motion_leg=run,
                    admit_preplanning_localization=Mock(
                        return_value=Pose2D(-0.48, -0.60, 1.69)
                    ),
                    seal_route=Mock(return_value=self._sealed(root)),
                    event_sink=lambda _path, payload: events.append(payload),
                    replan_startup_source=replan,
                ),
                **paths,
                leg_index=0,
                target_viewpoint_id="survey_vp_001",
                startup_reseal_motion_authorization_json=(
                    root / "startup_master.json"
                ),
            )

            decision = events[0]["prestart_localization_reseal_decision"]
            self.assertEqual(decision["recovery_action"], "tf_warmup_retry")
            self.assertEqual(decision["continuity_reason"], "map_from_odom_missing")
            self.assertEqual(decision["monitor_warning"], "stale_map_from_odom")
            retry_context = run.call_args_list[1].kwargs[
                "startup_reseal_permit_context"
            ]
            self.assertEqual(
                retry_context.recovery_source_kind,
                STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
            )
            self.assertEqual(
                replan.call_args.kwargs["expected_target_viewpoint_id"],
                "survey_vp_001",
            )

    def test_runtime_reseal_uses_injected_resolved_runtime_and_exact_permit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            stopped = _outcome(
                root,
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason=(
                    "global localization consistency requires zero and reseal"
                ),
                stop_details=_runtime_localization_stop_details(),
                motion_published=True,
            )
            completed = _outcome(
                root,
                run_id=(
                    "mission_coverage_000_runtime_localization_reseal_001"
                ),
                status="completed",
                motion_published=True,
            )
            run = Mock(side_effect=(stopped, completed))
            seal = Mock(return_value=self._sealed(root))
            admitted_pose = Pose2D(-0.31, -0.47, 0.12)
            admit = Mock(return_value=admitted_pose)
            replacement_route = root / "runtime_replacement_route.csv"
            replacement_diagnostics = root / "runtime_replacement_diagnostics.json"
            replan = Mock(
                return_value={
                    "route_csv": str(replacement_route),
                    "diagnostics_json": str(replacement_diagnostics),
                    "summary_json": str(root / "runtime_reseal_summary.json"),
                }
            )

            outcome = execute_coverage_leg_with_replans(
                profile=self.profile,
                config=self._config(),
                effects=CoverageLegEffects(
                    run_motion_leg=run,
                    admit_preplanning_localization=admit,
                    seal_route=seal,
                    advance_transient_overlay_resume_state=Mock(
                        return_value=None
                    ),
                    replan_runtime_localization_source=replan,
                ),
                **paths,
                leg_index=0,
                target_viewpoint_id="survey_vp_001",
                mission_motion_authorization_json=(
                    root / "mission_authorization.json"
                ),
            )

            self.assertIs(outcome, completed)
            self.assertIs(admit.call_args.args[0], self.runtime)
            self.assertEqual(run.call_count, 2)
            permit = run.call_args_list[1].kwargs[
                "runtime_localization_permit_context"
            ]
            self.assertIsInstance(permit, RuntimeLocalizationPermitContext)
            self.assertEqual(permit.reseal_index, 1)
            self.assertEqual(permit.target_viewpoint_id, "survey_vp_001")
            self.assertEqual(
                run.call_args_list[1].kwargs["fresh_confirmation_reason"],
                "runtime_localization",
            )
            self.assertEqual(
                seal.call_args_list[1].kwargs["source_route_csv"],
                replacement_route,
            )
            events = [
                json.loads(line)
                for line in (
                    paths["session_root"] / "adaptive_replans.jsonl"
                ).read_text().splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "runtime_localization_reseal_started",
                    "runtime_localization_admitted",
                    "runtime_localization_route_replanned",
                    "runtime_localization_route_sealed",
                ],
            )
            self.assertFalse(events[-1]["fresh_typed_run_required"])
            self.assertFalse(events[-1]["motion_continues_authorized"])

    def test_startup_reseal_master_yields_only_dedicated_recovery_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            rejected = _startup_mismatch_outcome(
                root,
                run_id="mission_coverage_000",
                route_pose={"x_m": -0.50, "y_m": -0.62, "yaw_rad": 1.70},
            )
            completed = _outcome(
                root,
                run_id="mission_coverage_000_startup_reseal_001",
                status="completed",
                motion_published=True,
            )
            run = Mock(side_effect=(rejected, completed))
            admit = Mock(return_value=Pose2D(-0.48, -0.60, 1.69))
            replan = Mock(
                return_value={
                    "route_csv": str(root / "replacement.csv"),
                    "diagnostics_json": str(root / "replacement.json"),
                    "summary_json": str(root / "source_summary.json"),
                }
            )

            execute_coverage_leg_with_replans(
                profile=self.profile,
                config=self._config(),
                effects=CoverageLegEffects(
                    run_motion_leg=run,
                    admit_preplanning_localization=admit,
                    seal_route=Mock(return_value=self._sealed(root)),
                    replan_startup_source=replan,
                ),
                **paths,
                leg_index=0,
                target_viewpoint_id="survey_vp_001",
                mission_leg_motion_authorization_json=(
                    root / "routine_master.json"
                ),
                startup_reseal_motion_authorization_json=(
                    root / "startup_master.json"
                ),
            )

            retry = run.call_args_list[1].kwargs
            self.assertTrue(retry["require_fresh_confirmation"])
            self.assertEqual(retry["fresh_confirmation_reason"], "startup")
            self.assertIsNone(retry["mission_leg_permit_context"])
            context = retry["startup_reseal_permit_context"]
            self.assertIsInstance(context, StartupResealPermitContext)
            self.assertEqual(context.leg_index, 0)
            self.assertEqual(context.target_viewpoint_id, "survey_vp_001")
            self.assertEqual(context.reseal_index, 1)
            self.assertEqual(context.rejected_run_id, rejected.run_id)
            self.assertEqual(
                context.recovery_source_kind,
                STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH,
            )

    def test_prestart_permit_context_survives_one_readiness_retry_without_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            recovery_run_id = "mission_coverage_000_startup_reseal_001"
            outcomes = (
                _prestart_localization_outcome(
                    root,
                    run_id="mission_coverage_000",
                ),
                _localization_readiness_rejection(
                    root,
                    run_id=recovery_run_id,
                ),
                _outcome(
                    root,
                    run_id=recovery_run_id + "_localization_readiness_001",
                    status="completed",
                    motion_published=True,
                ),
            )
            run = Mock(side_effect=outcomes)
            admit = Mock(return_value=Pose2D(-0.48, -0.60, 1.69))
            replan = Mock(
                return_value={
                    "route_csv": str(root / "prestart_replacement.csv"),
                    "diagnostics_json": str(root / "prestart_replacement.json"),
                    "summary_json": str(root / "prestart_source_summary.json"),
                }
            )
            events: list[dict[str, object]] = []
            startup_master = root / "startup_master.json"

            outcome = execute_coverage_leg_with_replans(
                profile=self.profile,
                config=self._config(),
                effects=CoverageLegEffects(
                    run_motion_leg=run,
                    admit_preplanning_localization=admit,
                    seal_route=Mock(return_value=self._sealed(root)),
                    event_sink=lambda _path, payload: events.append(payload),
                    replan_startup_source=replan,
                ),
                **paths,
                leg_index=0,
                target_viewpoint_id="survey_vp_001",
                startup_reseal_motion_authorization_json=startup_master,
            )

            self.assertIs(outcome, outcomes[-1])
            self.assertEqual(run.call_count, 3)
            admit.assert_called_once()
            replan.assert_called_once()
            first_context = run.call_args_list[1].kwargs[
                "startup_reseal_permit_context"
            ]
            retried_context = run.call_args_list[2].kwargs[
                "startup_reseal_permit_context"
            ]
            for context in (first_context, retried_context):
                self.assertIsInstance(context, StartupResealPermitContext)
                self.assertEqual(
                    context.recovery_source_kind,
                    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
                )
                self.assertEqual(
                    context.mission_authorization_json,
                    startup_master.absolute(),
                )
            self.assertNotEqual(
                first_context.permit_json_path,
                retried_context.permit_json_path,
            )
            self.assertEqual(
                run.call_args_list[2].kwargs["run_id"],
                recovery_run_id + "_localization_readiness_001",
            )
            route_seals = [
                event
                for event in events
                if event["event"] == "startup_reseal_route_sealed"
            ]
            self.assertEqual(len(route_seals), 2)
            self.assertTrue(
                all(event["covered_by_initial_mission_run"] for event in route_seals)
            )
            self.assertTrue(
                all(not event["additional_typed_run_required"] for event in route_seals)
            )

    def test_prestart_localization_budget_exhaustion_is_terminal_before_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            run = Mock(
                return_value=_prestart_localization_outcome(
                    root,
                    run_id="mission_coverage_000",
                )
            )
            admit = Mock()
            replan = Mock()

            with self.assertRaisesRegex(
                RuntimeError,
                "startup reseal budget exhausted",
            ):
                execute_coverage_leg_with_replans(
                    profile=self.profile,
                    config=self._config(max_startup_reseals_per_leg=0),
                    effects=CoverageLegEffects(
                        run_motion_leg=run,
                        admit_preplanning_localization=admit,
                        seal_route=Mock(return_value=self._sealed(root)),
                        replan_startup_source=replan,
                    ),
                    **paths,
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                )

            run.assert_called_once()
            admit.assert_not_called()
            replan.assert_not_called()

    def test_malformed_or_motion_published_prestart_evidence_is_terminal(self):
        for case in ("malformed", "motion_published"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = self._paths(root)
                rejected = _prestart_localization_outcome(
                    root,
                    run_id="mission_coverage_000",
                    motion_published=(case == "motion_published"),
                )
                if case == "malformed":
                    rejected.stop_details["continuity"] = {"accepted": False}
                run = Mock(return_value=rejected)
                admit = Mock()
                runtime_replan = Mock()
                events: list[dict[str, object]] = []

                with self.assertRaisesRegex(RuntimeError, "physical route failed"):
                    execute_coverage_leg_with_replans(
                        profile=self.profile,
                        config=self._config(),
                        effects=CoverageLegEffects(
                            run_motion_leg=run,
                            admit_preplanning_localization=admit,
                            seal_route=Mock(return_value=self._sealed(root)),
                            event_sink=(
                                lambda _path, payload: events.append(payload)
                            ),
                            replan_runtime_localization_source=runtime_replan,
                        ),
                        **paths,
                        leg_index=0,
                        target_viewpoint_id="survey_vp_001",
                    )

                run.assert_called_once()
                admit.assert_not_called()
                runtime_replan.assert_not_called()
                self.assertEqual(
                    [event["event"] for event in events],
                    ["prestart_localization_reseal_rejected"],
                )
                self.assertFalse(events[0]["motion_continues_authorized"])
                self.assertTrue(events[0]["fail_closed"])

    def test_runtime_reseal_budget_exhaustion_is_terminal_before_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            stopped = _outcome(
                root,
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason=(
                    "global localization consistency requires zero and reseal"
                ),
                stop_details=_runtime_localization_stop_details(),
                motion_published=True,
            )
            admit = Mock()
            with self.assertRaisesRegex(
                RuntimeError,
                "runtime localization reseal budget exhausted",
            ):
                execute_coverage_leg_with_replans(
                    profile=self.profile,
                    config=self._config(
                        max_runtime_localization_reseals_per_leg=0
                    ),
                    effects=CoverageLegEffects(
                        run_motion_leg=Mock(return_value=stopped),
                        admit_preplanning_localization=admit,
                        seal_route=Mock(return_value=self._sealed(root)),
                    ),
                    **paths,
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                )
            admit.assert_not_called()
            event = json.loads(
                (paths["session_root"] / "adaptive_replans.jsonl")
                .read_text()
                .splitlines()[0]
            )
            self.assertEqual(event["event"], "runtime_localization_reseal_rejected")
            self.assertTrue(event["fail_closed"])
            self.assertFalse(event["motion_continues_authorized"])

    def test_startup_reseal_budget_exhaustion_is_terminal_without_third_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            pose = {"x_m": -0.50, "y_m": -0.62, "yaw_rad": 1.70}
            run = Mock(
                side_effect=(
                    _startup_mismatch_outcome(
                        root,
                        run_id="mission_coverage_000",
                        route_pose=pose,
                    ),
                    _startup_mismatch_outcome(
                        root,
                        run_id="mission_coverage_000_startup_reseal_001",
                        route_pose=pose,
                    ),
                )
            )
            seal = Mock(return_value=self._sealed(root))
            replan = Mock(
                return_value={
                    "route_csv": str(root / "startup_replacement.csv"),
                    "diagnostics_json": str(
                        root / "startup_replacement_diagnostics.json"
                    ),
                    "summary_json": str(root / "startup_replacement_summary.json"),
                }
            )
            events: list[dict[str, object]] = []

            with self.assertRaisesRegex(
                RuntimeError,
                "startup reseal budget exhausted",
            ):
                execute_coverage_leg_with_replans(
                    profile=self.profile,
                    config=self._config(max_startup_reseals_per_leg=1),
                    effects=CoverageLegEffects(
                        run_motion_leg=run,
                        admit_preplanning_localization=Mock(
                            return_value=Pose2D(-0.48, -0.60, 1.69)
                        ),
                        seal_route=seal,
                        event_sink=lambda _path, payload: events.append(payload),
                        clock=lambda: 123.0,
                        replan_startup_source=replan,
                    ),
                    **paths,
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                )

            self.assertEqual(run.call_count, 2)
            self.assertEqual(seal.call_count, 2)
            replan.assert_called_once()
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "startup_reseal_started",
                    "startup_localization_admitted",
                    "startup_pose_route_resealed",
                    "startup_reseal_route_sealed",
                ],
            )
            self.assertEqual(events[0]["timestamp"], 123.0)

    def test_malformed_or_nonfinite_startup_pose_is_terminal(self):
        malformed_poses = (
            {"x_m": -0.50, "y_m": -0.62},
            {"x_m": float("inf"), "y_m": -0.62, "yaw_rad": 1.70},
        )
        for route_pose in malformed_poses:
            with self.subTest(route_pose=route_pose):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    paths = self._paths(root)
                    run = Mock(
                        return_value=_startup_mismatch_outcome(
                            root,
                            run_id="mission_coverage_000",
                            route_pose=route_pose,
                        )
                    )
                    seal = Mock(return_value=self._sealed(root))
                    replan = Mock()
                    events: list[dict[str, object]] = []

                    with self.assertRaises((KeyError, ValueError)):
                        execute_coverage_leg_with_replans(
                            profile=self.profile,
                            config=self._config(),
                            effects=CoverageLegEffects(
                                run_motion_leg=run,
                                admit_preplanning_localization=Mock(),
                                seal_route=seal,
                                event_sink=(
                                    lambda _path, payload: events.append(payload)
                                ),
                                clock=lambda: 124.0,
                                replan_startup_source=replan,
                            ),
                            **paths,
                            leg_index=0,
                            target_viewpoint_id="survey_vp_001",
                        )

                    run.assert_called_once()
                    seal.assert_called_once()
                    replan.assert_not_called()
                    self.assertEqual(events, [])

    def test_runtime_same_target_replan_failure_emits_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            run = Mock(
                return_value=_outcome(
                    root,
                    run_id="mission_coverage_000",
                    status="stopped",
                    stop_reason=(
                        "global localization consistency requires zero and reseal"
                    ),
                    stop_details=_runtime_localization_stop_details(),
                    motion_published=True,
                )
            )
            replan = Mock(side_effect=RuntimeError("same-target planner failed"))
            events: list[dict[str, object]] = []

            with self.assertRaisesRegex(RuntimeError, "same-target planner failed"):
                execute_coverage_leg_with_replans(
                    profile=self.profile,
                    config=self._config(),
                    effects=CoverageLegEffects(
                        run_motion_leg=run,
                        admit_preplanning_localization=Mock(
                            return_value=Pose2D(-0.31, -0.47, 0.12)
                        ),
                        seal_route=Mock(return_value=self._sealed(root)),
                        event_sink=lambda _path, payload: events.append(payload),
                        clock=lambda: 125.0,
                        advance_transient_overlay_resume_state=Mock(
                            return_value=None
                        ),
                        replan_runtime_localization_source=replan,
                    ),
                    **paths,
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                )

            run.assert_called_once()
            replan.assert_called_once()
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "runtime_localization_reseal_started",
                    "runtime_localization_admitted",
                    "runtime_localization_reseal_failed",
                ],
            )
            failure = events[-1]
            self.assertEqual(failure["phase"], "same_target_route_replan")
            self.assertEqual(failure["failure"], "same-target planner failed")
            self.assertEqual(failure["timestamp"], 125.0)
            self.assertFalse(failure["motion_continues_authorized"])

    def test_runtime_replacement_route_seal_failure_emits_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            stopped = _outcome(
                root,
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason=(
                    "global localization consistency requires zero and reseal"
                ),
                stop_details=_runtime_localization_stop_details(),
                motion_published=True,
            )
            run = Mock(return_value=stopped)
            seal = Mock(
                side_effect=(
                    self._sealed(root),
                    RuntimeError("replacement route seal failed"),
                )
            )
            replan = Mock(
                return_value={
                    "route_csv": str(root / "runtime_replacement.csv"),
                    "diagnostics_json": str(
                        root / "runtime_replacement_diagnostics.json"
                    ),
                    "summary_json": str(root / "runtime_replacement_summary.json"),
                }
            )
            events: list[dict[str, object]] = []

            with self.assertRaisesRegex(
                RuntimeError,
                "replacement route seal failed",
            ):
                execute_coverage_leg_with_replans(
                    profile=self.profile,
                    config=self._config(),
                    effects=CoverageLegEffects(
                        run_motion_leg=run,
                        admit_preplanning_localization=Mock(
                            return_value=Pose2D(-0.31, -0.47, 0.12)
                        ),
                        seal_route=seal,
                        event_sink=lambda _path, payload: events.append(payload),
                        clock=lambda: 126.0,
                        advance_transient_overlay_resume_state=Mock(
                            return_value=None
                        ),
                        replan_runtime_localization_source=replan,
                    ),
                    **paths,
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                )

            run.assert_called_once()
            self.assertEqual(seal.call_count, 2)
            replan.assert_called_once()
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "runtime_localization_reseal_started",
                    "runtime_localization_admitted",
                    "runtime_localization_route_replanned",
                    "runtime_localization_reseal_failed",
                ],
            )
            failure = events[-1]
            self.assertEqual(failure["phase"], "route_seal")
            self.assertEqual(failure["failure"], "replacement route seal failed")
            self.assertEqual(failure["timestamp"], 126.0)
            self.assertFalse(failure["motion_continues_authorized"])

    def test_runtime_permit_context_is_preserved_across_readiness_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._paths(root)
            runtime_run_id = (
                "mission_coverage_000_runtime_localization_reseal_001"
            )
            outcomes = (
                _outcome(
                    root,
                    run_id="mission_coverage_000",
                    status="stopped",
                    stop_reason=(
                        "global localization consistency requires zero and reseal"
                    ),
                    stop_details=_runtime_localization_stop_details(),
                    motion_published=True,
                ),
                _localization_readiness_rejection(
                    root,
                    run_id=runtime_run_id,
                ),
                _outcome(
                    root,
                    run_id=(
                        runtime_run_id + "_localization_readiness_001"
                    ),
                    status="completed",
                    motion_published=True,
                ),
            )
            run = Mock(side_effect=outcomes)
            seal = Mock(return_value=self._sealed(root))
            replan = Mock(
                return_value={
                    "route_csv": str(root / "runtime_replacement.csv"),
                    "diagnostics_json": str(
                        root / "runtime_replacement_diagnostics.json"
                    ),
                    "summary_json": str(root / "runtime_replacement_summary.json"),
                }
            )
            events: list[dict[str, object]] = []

            outcome = execute_coverage_leg_with_replans(
                profile=self.profile,
                config=self._config(),
                effects=CoverageLegEffects(
                    run_motion_leg=run,
                    admit_preplanning_localization=Mock(
                        return_value=Pose2D(-0.31, -0.47, 0.12)
                    ),
                    seal_route=seal,
                    event_sink=lambda _path, payload: events.append(payload),
                    clock=lambda: 127.0,
                    advance_transient_overlay_resume_state=Mock(return_value=None),
                    replan_runtime_localization_source=replan,
                ),
                **paths,
                leg_index=0,
                target_viewpoint_id="survey_vp_001",
                mission_motion_authorization_json=(
                    root / "mission_motion_authorization.json"
                ),
            )

            self.assertIs(outcome, outcomes[-1])
            self.assertEqual(run.call_count, 3)
            self.assertEqual(seal.call_count, 3)
            first_permit = run.call_args_list[1].kwargs[
                "runtime_localization_permit_context"
            ]
            retried_permit = run.call_args_list[2].kwargs[
                "runtime_localization_permit_context"
            ]
            self.assertIsInstance(first_permit, RuntimeLocalizationPermitContext)
            self.assertIs(retried_permit, first_permit)
            self.assertEqual(first_permit.reseal_index, 1)
            self.assertEqual(first_permit.target_viewpoint_id, "survey_vp_001")
            self.assertEqual(
                run.call_args_list[2].kwargs["run_id"],
                runtime_run_id + "_localization_readiness_001",
            )
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "runtime_localization_reseal_started",
                    "runtime_localization_admitted",
                    "runtime_localization_route_replanned",
                    "runtime_localization_route_sealed",
                    "localization_readiness_retry_scheduled",
                ],
            )
            self.assertTrue(events[-1]["route_limits_unchanged"])
            self.assertEqual(events[-1]["timestamp"], 127.0)


if __name__ == "__main__":
    unittest.main()
