from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.autonomous_runner import runtime as runner


class _AuthorizationBoundary(RuntimeError):
    """Stop a main-path test immediately after the RUN boundary."""


class AutonomousReadinessIntegrationTest(unittest.TestCase):
    @staticmethod
    def _profile() -> SimpleNamespace:
        runtime = SimpleNamespace(
            namespace="",
            cmd_vel_topic="/cmd_vel",
            scan_topic="/scan",
            odom_frame="odom",
        )
        return SimpleNamespace(
            robot_id="turtlebot1",
            map_frame="map",
            robot_radius_m=0.105,
            scan_origin_to_base_offset_m=0.05,
            scan_frame="base_scan",
            camera_optical_frame="camera_optical_frame",
            resolved_compressed_image_topic=(
                "/camera/image_raw/compressed"
            ),
            resolved_camera_info_topic="/camera/camera_info",
            resolved_runtime=lambda: runtime,
        )

    @staticmethod
    def _main_argv(root: Path) -> list[str]:
        return [
            "--robot-profile",
            str(root / "robot.json"),
            "--camera-calibration",
            str(root / "camera.json"),
            "--stand-model-profile",
            str(root / "stand_model.json"),
            "--physical-site",
            str(root / "site.json"),
            "--map",
            str(root / "map.yaml"),
            "--output-root",
            str(root / "runs"),
            "--session-id",
            "readiness_execute_full",
            "--localization-branch-proof-id",
            "known_start_marker_20260817",
            "--run-mode",
            "execute-full",
        ]

    def _patch_main_before_authorization(
        self,
        stack: ExitStack,
        root: Path,
        *,
        initial_readiness,
    ):
        digest = "d" * 64
        profile = self._profile()
        plan = SimpleNamespace(
            viewpoints=(SimpleNamespace(viewpoint_id="survey_vp_001"),),
            map_bundle_sha256=digest,
        )
        stack.enter_context(
            patch.object(runner, "load_real_robot_profile", return_value=profile)
        )
        stack.enter_context(
            patch.object(
                runner,
                "load_camera_calibration",
                return_value=SimpleNamespace(),
            )
        )
        stack.enter_context(
            patch.object(
                runner,
                "validate_physical_site_contract",
                return_value=SimpleNamespace(
                    expected_stand_count=5,
                    physical_site_path=(root / "site.json"),
                    map_yaml_path=(root / "map.yaml"),
                    map_bundle=SimpleNamespace(bundle_sha256=digest),
                ),
            )
        )
        stack.enter_context(patch.object(runner, "_validate_inputs"))
        stack.enter_context(
            patch.object(
                runner,
                "load_measured_physical_stand_model",
                return_value=SimpleNamespace(
                    navigation_footprint_radius_m=0.110,
                    head_center_height_m=0.171,
                ),
            )
        )
        stack.enter_context(
            patch.object(
                runner,
                "_physical_clearance",
                return_value={
                    "minimum_active_standoff_m": 0.20,
                    "minimum_static_inflation_m": 0.25,
                    "minimum_candidate_transit_radius_m": 0.31,
                },
            )
        )
        stack.enter_context(
            patch.object(
                runner,
                "_admit_sensor_timing_readiness",
                return_value=(
                    root
                    / (
                        "runs/readiness_execute_full/preflight/"
                        "camera_lidar_timing_before_authorization.json"
                    ),
                    "c" * 64,
                ),
            )
        )
        stack.enter_context(
            patch.object(
                runner,
                "_admit_observation_tf_readiness",
                return_value=(
                    root
                    / (
                        "runs/readiness_execute_full/preflight/"
                        "lidar_scan_tf_before_authorization.json"
                    ),
                    "f" * 64,
                ),
            )
        )
        stack.enter_context(
            patch.object(
                runner,
                "_admit_preplanning_localization",
                return_value=Pose2D(0.0, 0.0, 0.0),
            )
        )
        stack.enter_context(
            patch.object(runner, "plan_stand_coverage_survey", return_value=0)
        )
        stack.enter_context(
            patch.object(runner, "load_coverage_survey_plan", return_value=plan)
        )
        return stack.enter_context(
            patch.object(
                runner,
                "admit_preauthorization_readiness",
                side_effect=initial_readiness,
            )
        )

    def test_motion_leg_tf_rejection_after_dry_starts_no_live_child_or_permit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            canonical = SimpleNamespace(
                session_root=root,
                route_csv=root / "route.csv",
                diagnostics_json=root / "diagnostics.json",
                route_certificate_json=root / "certificate.json",
            )

            def dry_process(_command, **_kwargs):
                semantic_log = (
                    root / "run_events" / "mission_coverage_000.jsonl"
                )
                semantic_log.parent.mkdir(parents=True, exist_ok=True)
                semantic_log.write_text(
                    json.dumps(
                        {
                            "event": "dry_run_completed",
                            "run_id": "mission_coverage_000",
                            "status": "dry_run_ok",
                            "motion_published": False,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                for artifact in (
                    root / "preflight/mission_coverage_000_dry.json",
                    root
                    / "odom_execution/mission_coverage_000_dry_certificate.json",
                    root
                    / "odom_execution/mission_coverage_000_dry_uncertainty_budget.json",
                ):
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_text("{}\n", encoding="utf-8")
                return SimpleNamespace(returncode=0)

            with (
                patch.object(
                    runner,
                    "_runner_command",
                    return_value=["child", "--dry-run"],
                ),
                patch.object(
                    runner.subprocess,
                    "run",
                    side_effect=dry_process,
                ) as run_process,
                patch.object(
                    runner,
                    "resolve_child_artifact_paths",
                    return_value=canonical,
                ),
                patch.object(
                    runner,
                    "_admit_observation_tf_readiness",
                    side_effect=RuntimeError("exact-time scan transform missing"),
                ) as admit_tf,
                patch.object(
                    runner,
                    "_issue_mission_leg_motion_permit",
                ) as issue_permit,
                patch.object(runner, "_bundle_command") as bundle_command,
                self.assertRaisesRegex(
                    RuntimeError,
                    "exact-time scan transform missing",
                ),
            ):
                runner._run_motion_leg(
                    profile=SimpleNamespace(),
                    sealed={
                        "route_csv": str(root / "route.csv"),
                        "diagnostics_json": str(root / "diagnostics.json"),
                        "route_certificate_json": str(root / "certificate.json"),
                    },
                    run_id="mission_coverage_000",
                    session_root=root,
                    execute=True,
                    uncertainty_map_yaml=root / "map.yaml",
                    mission_leg_permit_context=runner.MissionLegPermitContext(
                        mission_authorization_json=(
                            root / "mission_leg_authorization.json"
                        ),
                        session_id="mission",
                        semantic_map_id="arena",
                        mission_leg_kind=MissionLegKind.COVERAGE,
                        mission_leg_index=0,
                        target_id="survey_vp_001",
                        permit_json_path=root / "permit.json",
                    ),
                    observation_tf_evidence_path=root / "scan_tf.json",
                )

            self.assertEqual(run_process.call_count, 1)
            self.assertEqual(run_process.call_args.args[0], ["child", "--dry-run"])
            admit_tf.assert_called_once_with(
                SimpleNamespace(),
                root / "scan_tf.json",
                phase="coverage_leg_before_motion",
                typed_run_already_issued=True,
            )
            issue_permit.assert_not_called()
            bundle_command.assert_not_called()

    def test_motion_leg_timing_rejection_runs_after_dry_and_tf_before_permit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run_id = "mission_candidate_000"
            call_order: list[str] = []
            canonical = SimpleNamespace(
                session_root=root,
                route_csv=root / "route.csv",
                diagnostics_json=root / "diagnostics.json",
                route_certificate_json=root / "certificate.json",
            )

            def dry_process(_command, **_kwargs):
                call_order.append("dry")
                semantic_log = root / "run_events" / f"{run_id}.jsonl"
                semantic_log.parent.mkdir(parents=True, exist_ok=True)
                semantic_log.write_text(
                    json.dumps(
                        {
                            "event": "dry_run_completed",
                            "run_id": run_id,
                            "status": "dry_run_ok",
                            "motion_published": False,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                for artifact in (
                    root / "preflight" / f"{run_id}_dry.json",
                    root
                    / "odom_execution"
                    / f"{run_id}_dry_certificate.json",
                    root
                    / "odom_execution"
                    / f"{run_id}_dry_uncertainty_budget.json",
                ):
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_text("{}\n", encoding="utf-8")
                return SimpleNamespace(returncode=0)

            def resolve_canonical(*_args, **_kwargs):
                call_order.append("canonicalize")
                return canonical

            def pass_tf(*_args, **_kwargs):
                call_order.append("scan_tf")
                return root / "scan_tf.json", "f" * 64

            def reject_timing(*_args, **_kwargs):
                call_order.append("sensor_timing")
                raise RuntimeError("camera/LiDAR tuple became stale")

            with (
                patch.object(
                    runner,
                    "_runner_command",
                    return_value=["child", "--dry-run"],
                ),
                patch.object(
                    runner.subprocess,
                    "run",
                    side_effect=dry_process,
                ) as run_process,
                patch.object(
                    runner,
                    "resolve_child_artifact_paths",
                    side_effect=resolve_canonical,
                ),
                patch.object(
                    runner,
                    "_admit_observation_tf_readiness",
                    side_effect=pass_tf,
                ),
                patch.object(
                    runner,
                    "_admit_sensor_timing_readiness",
                    side_effect=reject_timing,
                ) as admit_timing,
                patch.object(
                    runner,
                    "_issue_mission_leg_motion_permit",
                ) as issue_permit,
                patch.object(runner, "_bundle_command") as bundle_command,
                self.assertRaisesRegex(
                    RuntimeError,
                    "camera/LiDAR tuple became stale",
                ),
            ):
                runner._run_motion_leg(
                    profile=SimpleNamespace(),
                    sealed={
                        "route_csv": str(root / "route.csv"),
                        "diagnostics_json": str(root / "diagnostics.json"),
                        "route_certificate_json": str(
                            root / "certificate.json"
                        ),
                    },
                    run_id=run_id,
                    session_root=root,
                    execute=True,
                    uncertainty_map_yaml=root / "map.yaml",
                    mission_leg_permit_context=(
                        runner.MissionLegPermitContext(
                            mission_authorization_json=(
                                root / "mission_leg_authorization.json"
                            ),
                            session_id="mission",
                            semantic_map_id="arena",
                            mission_leg_kind=(
                                MissionLegKind.CANDIDATE_PREAPPROACH
                            ),
                            mission_leg_index=0,
                            target_id="candidate_000",
                            permit_json_path=root / "permit.json",
                        )
                    ),
                    observation_tf_evidence_path=root / "scan_tf.json",
                    sensor_timing_readiness_phase=(
                        runner.CANDIDATE_ROUTE_SENSOR_TIMING_PHASE
                    ),
                )

            self.assertEqual(
                call_order,
                ["dry", "canonicalize", "scan_tf", "sensor_timing"],
            )
            self.assertEqual(run_process.call_count, 1)
            admit_timing.assert_called_once_with(
                SimpleNamespace(),
                root
                / "preflight"
                / f"{run_id}_camera_lidar_timing_before_motion.json",
                phase=runner.CANDIDATE_ROUTE_SENSOR_TIMING_PHASE,
                typed_run_already_issued=True,
            )
            issue_permit.assert_not_called()
            bundle_command.assert_not_called()

    def test_zero_exit_without_dry_semantics_fails_before_tf_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with (
                patch.object(
                    runner,
                    "_runner_command",
                    return_value=["child", "--dry-run"],
                ),
                patch.object(
                    runner.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ),
                patch.object(
                    runner,
                    "_admit_observation_tf_readiness",
                ) as admit_tf,
                self.assertRaisesRegex(
                    RuntimeError,
                    "dry-run success evidence is invalid",
                ),
            ):
                runner._run_motion_leg(
                    profile=SimpleNamespace(),
                    sealed={
                        "route_csv": str(root / "route.csv"),
                        "diagnostics_json": str(root / "diagnostics.json"),
                        "route_certificate_json": str(root / "certificate.json"),
                    },
                    run_id="synthetic_zero",
                    session_root=root,
                    execute=False,
                    observation_tf_evidence_path=root / "scan_tf.json",
                )

            admit_tf.assert_not_called()

    def test_valid_dry_terminal_without_preflight_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()

            def dry_process(_command, **_kwargs):
                semantic_log = root / "run_events/artifact_missing.jsonl"
                semantic_log.parent.mkdir(parents=True, exist_ok=True)
                semantic_log.write_text(
                    json.dumps(
                        {
                            "event": "dry_run_completed",
                            "run_id": "artifact_missing",
                            "status": "dry_run_ok",
                            "motion_published": False,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with (
                patch.object(
                    runner,
                    "_runner_command",
                    return_value=["child", "--dry-run"],
                ),
                patch.object(
                    runner.subprocess,
                    "run",
                    side_effect=dry_process,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "dry-run success artifact is invalid",
                ),
            ):
                runner._run_motion_leg(
                    profile=SimpleNamespace(),
                    sealed={
                        "route_csv": str(root / "route.csv"),
                        "diagnostics_json": str(root / "diagnostics.json"),
                        "route_certificate_json": str(root / "certificate.json"),
                    },
                    run_id="artifact_missing",
                    session_root=root,
                    execute=False,
                )

    def test_lidar_epoch_tf_rejection_does_not_start_observer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            evidence_path = root / "scan_tf.json"
            with (
                patch.object(
                    runner,
                    "_admit_observation_tf_readiness",
                    side_effect=RuntimeError("scan transform gate rejected"),
                ) as admit_tf,
                patch.object(runner.subprocess, "run") as run_process,
                self.assertRaisesRegex(
                    RuntimeError,
                    "scan transform gate rejected",
                ),
            ):
                runner._capture_lidar_epoch(
                    profile=SimpleNamespace(),
                    args=SimpleNamespace(),
                    survey_root=root / "coverage",
                    viewpoint_id="survey_vp_001",
                    observation_tf_evidence_path=evidence_path,
                )

            admit_tf.assert_called_once_with(
                SimpleNamespace(),
                evidence_path,
                phase="coverage_lidar_epoch_before_observer",
                typed_run_already_issued=True,
            )
            run_process.assert_not_called()

    def test_initial_readiness_rejection_precedes_prompt_and_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with ExitStack() as stack:
                readiness = self._patch_main_before_authorization(
                    stack,
                    root,
                    initial_readiness=RuntimeError(
                        "initial route readiness rejected"
                    ),
                )
                run_prompt = stack.enter_context(patch("builtins.input"))
                mission_leg_writer = stack.enter_context(
                    patch.object(runner, "write_mission_leg_motion_authorization")
                )
                mission_writer = stack.enter_context(
                    patch.object(runner, "write_mission_motion_authorization")
                )
                startup_writer = stack.enter_context(
                    patch.object(
                        runner,
                        "write_startup_reseal_motion_authorization",
                    )
                )
                coverage = stack.enter_context(
                    patch.object(runner, "execute_coverage_mission")
                )
                stack.enter_context(redirect_stdout(StringIO()))
                stack.enter_context(redirect_stderr(StringIO()))

                with self.assertRaises(SystemExit) as raised:
                    runner.main(self._main_argv(root))

            self.assertEqual(raised.exception.code, 2)
            readiness.assert_called_once()
            run_prompt.assert_not_called()
            mission_leg_writer.assert_not_called()
            mission_writer.assert_not_called()
            startup_writer.assert_not_called()
            coverage.assert_not_called()

    def test_sensor_timing_rejection_is_first_post_session_motion_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            call_order: list[str] = []
            session_root = root / "runs/readiness_execute_full"

            def reject_sensor_timing(*_args, **_kwargs):
                self.assertTrue(
                    session_root.is_dir(),
                    "the evidence root must exist before the timing gate runs",
                )
                self.assertEqual(list(session_root.iterdir()), [])
                call_order.append("sensor_timing")
                raise RuntimeError("camera/LiDAR header clocks diverged")

            with ExitStack() as stack:
                self._patch_main_before_authorization(
                    stack,
                    root,
                    initial_readiness=lambda *_args, **_kwargs: None,
                )
                timing = stack.enter_context(
                    patch.object(
                        runner,
                        "_admit_sensor_timing_readiness",
                        side_effect=reject_sensor_timing,
                    )
                )
                scan_tf = stack.enter_context(
                    patch.object(
                        runner,
                        "_admit_observation_tf_readiness",
                        side_effect=lambda *_args, **_kwargs: call_order.append(
                            "scan_tf"
                        ),
                    )
                )
                initialpose = stack.enter_context(
                    patch.object(
                        runner,
                        "_prompt_for_preplanning_initialpose_if_requested",
                        side_effect=lambda *_args, **_kwargs: call_order.append(
                            "initialpose"
                        ),
                    )
                )
                localization = stack.enter_context(
                    patch.object(
                        runner,
                        "_admit_preplanning_localization",
                        side_effect=lambda *_args, **_kwargs: call_order.append(
                            "localization"
                        ),
                    )
                )
                planning = stack.enter_context(
                    patch.object(
                        runner,
                        (
                            "_plan_initial_coverage_with_optional_"
                            "startup_active_localization"
                        ),
                        side_effect=lambda **_kwargs: call_order.append(
                            "planning"
                        ),
                    )
                )
                startup_localize = stack.enter_context(
                    patch.object(
                        runner,
                        "run_startup_active_localization_child",
                        side_effect=lambda *_args, **_kwargs: call_order.append(
                            "startup_LOCALIZE"
                        ),
                    )
                )
                run_prompt = stack.enter_context(
                    patch(
                        "builtins.input",
                        side_effect=lambda _prompt: call_order.append("RUN"),
                    )
                )
                mission_leg_writer = stack.enter_context(
                    patch.object(runner, "write_mission_leg_motion_authorization")
                )
                mission_writer = stack.enter_context(
                    patch.object(runner, "write_mission_motion_authorization")
                )
                startup_writer = stack.enter_context(
                    patch.object(
                        runner,
                        "write_startup_reseal_motion_authorization",
                    )
                )
                coverage = stack.enter_context(
                    patch.object(runner, "execute_coverage_mission")
                )
                motion_leg = stack.enter_context(
                    patch.object(runner, "_run_motion_leg")
                )
                stack.enter_context(redirect_stdout(StringIO()))
                stack.enter_context(redirect_stderr(StringIO()))

                with self.assertRaises(SystemExit) as raised:
                    runner.main(self._main_argv(root))

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(call_order, ["sensor_timing"])
            timing.assert_called_once()
            timing_args, timing_kwargs = timing.call_args
            self.assertEqual(
                timing_args[1],
                session_root
                / "preflight/camera_lidar_timing_before_authorization.json",
            )
            self.assertEqual(
                timing_args[0].resolved_compressed_image_topic,
                "/camera/image_raw/compressed",
            )
            self.assertEqual(
                timing_kwargs,
                {
                    "phase": "preauthorization_sensor_timing_readiness",
                    "typed_run_already_issued": False,
                },
            )
            scan_tf.assert_not_called()
            initialpose.assert_not_called()
            localization.assert_not_called()
            planning.assert_not_called()
            startup_localize.assert_not_called()
            run_prompt.assert_not_called()
            mission_leg_writer.assert_not_called()
            mission_writer.assert_not_called()
            startup_writer.assert_not_called()
            coverage.assert_not_called()
            motion_leg.assert_not_called()

    def test_successful_readiness_precedes_the_single_run_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            call_order: list[str] = []

            def pass_sensor_timing(*_args, **_kwargs):
                call_order.append("sensor_timing")
                return (
                    root
                    / (
                        "runs/readiness_execute_full/preflight/"
                        "camera_lidar_timing_before_authorization.json"
                    ),
                    "c" * 64,
                )

            def pass_scan_tf(*_args, **_kwargs):
                call_order.append("scan_tf")
                return (
                    root
                    / (
                        "runs/readiness_execute_full/preflight/"
                        "lidar_scan_tf_before_authorization.json"
                    ),
                    "f" * 64,
                )

            def pass_readiness(*_args, **_kwargs):
                effects = _args[1]
                self.assertIsNone(effects.prepare_localization_attempt)
                call_order.append("readiness")
                return SimpleNamespace(
                    result=SimpleNamespace(attempts=(object(),)),
                    evidence_path=root / "initial_readiness.json",
                    evidence_sha256="e" * 64,
                )

            def confirm_run(_prompt):
                call_order.append("RUN")
                return "RUN"

            def stop_at_authorization(*_args, **_kwargs):
                call_order.append("authorization")
                raise _AuthorizationBoundary("authorization boundary reached")

            with ExitStack() as stack:
                readiness = self._patch_main_before_authorization(
                    stack,
                    root,
                    initial_readiness=pass_readiness,
                )
                sensor_timing = stack.enter_context(
                    patch.object(
                        runner,
                        "_admit_sensor_timing_readiness",
                        side_effect=pass_sensor_timing,
                    )
                )
                scan_tf = stack.enter_context(
                    patch.object(
                        runner,
                        "_admit_observation_tf_readiness",
                        side_effect=pass_scan_tf,
                    )
                )
                run_prompt = stack.enter_context(
                    patch("builtins.input", side_effect=confirm_run)
                )
                stack.enter_context(
                    patch.object(runner, "_file_sha256", return_value="a" * 64)
                )
                stack.enter_context(
                    patch.object(
                        runner,
                        "_checkpoint_config_sha256",
                        return_value="b" * 64,
                    )
                )
                mission_leg_writer = stack.enter_context(
                    patch.object(
                        runner,
                        "write_mission_leg_motion_authorization",
                        side_effect=stop_at_authorization,
                    )
                )
                stack.enter_context(redirect_stdout(StringIO()))
                stack.enter_context(redirect_stderr(StringIO()))

                with self.assertRaises(SystemExit) as raised:
                    runner.main(self._main_argv(root))

            self.assertEqual(raised.exception.code, 2)
            readiness.assert_called_once()
            sensor_timing.assert_called_once()
            scan_tf.assert_called_once()
            run_prompt.assert_called_once()
            mission_leg_writer.assert_called_once()
            self.assertEqual(
                call_order,
                [
                    "sensor_timing",
                    "scan_tf",
                    "readiness",
                    "RUN",
                    "authorization",
                ],
            )

    def test_initialpose_bootstrap_precedes_planning_and_is_not_owned_by_retries(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            call_order: list[str] = []
            readiness_effects = []

            def bootstrap_initialpose(**kwargs):
                self.assertTrue(kwargs["enabled"])
                self.assertEqual(kwargs["config"].maximum_retry_count, 0)
                call_order.append("initialpose")
                return True

            def admit_localization(*_args, **_kwargs):
                call_order.append("localization")
                return Pose2D(0.12, -0.34, 0.56)

            def plan_coverage(argv):
                call_order.append("planning")
                self.assertEqual(argv[argv.index("--start-x") + 1], "0.12")
                self.assertEqual(argv[argv.index("--start-y") + 1], "-0.34")
                self.assertEqual(argv[argv.index("--start-yaw") + 1], "0.56")
                return 0

            def pass_readiness(_config, effects):
                call_order.append("readiness")
                readiness_effects.append(effects)
                return SimpleNamespace(
                    result=SimpleNamespace(attempts=(object(), object())),
                    evidence_path=root / "initial_readiness.json",
                    evidence_sha256="e" * 64,
                )

            def confirm_run(_prompt):
                call_order.append("RUN")
                return "RUN"

            def stop_at_authorization(*_args, **_kwargs):
                call_order.append("authorization")
                raise _AuthorizationBoundary("authorization boundary reached")

            with ExitStack() as stack:
                self._patch_main_before_authorization(
                    stack,
                    root,
                    initial_readiness=pass_readiness,
                )
                bootstrap = stack.enter_context(
                    patch.object(
                        runner,
                        "prepare_preplanning_initialpose",
                        side_effect=bootstrap_initialpose,
                    )
                )
                stack.enter_context(
                    patch.object(
                        runner,
                        "_admit_preplanning_localization",
                        side_effect=admit_localization,
                    )
                )
                stack.enter_context(
                    patch.object(
                        runner,
                        "plan_stand_coverage_survey",
                        side_effect=plan_coverage,
                    )
                )
                stack.enter_context(
                    patch.object(
                        runner,
                        "admit_preauthorization_readiness",
                        side_effect=pass_readiness,
                    )
                )
                run_prompt = stack.enter_context(
                    patch("builtins.input", side_effect=confirm_run)
                )
                stack.enter_context(
                    patch.object(runner, "_file_sha256", return_value="a" * 64)
                )
                stack.enter_context(
                    patch.object(
                        runner,
                        "_checkpoint_config_sha256",
                        return_value="b" * 64,
                    )
                )
                mission_leg_writer = stack.enter_context(
                    patch.object(
                        runner,
                        "write_mission_leg_motion_authorization",
                        side_effect=stop_at_authorization,
                    )
                )
                coverage = stack.enter_context(
                    patch.object(runner, "execute_coverage_mission")
                )
                stack.enter_context(redirect_stdout(StringIO()))
                stack.enter_context(redirect_stderr(StringIO()))

                with self.assertRaises(SystemExit) as raised:
                    runner.main(
                        [
                            *self._main_argv(root),
                            "--prompt-for-initialpose",
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            bootstrap.assert_called_once()
            self.assertEqual(len(readiness_effects), 1)
            self.assertIsNone(
                readiness_effects[0].prepare_localization_attempt,
                "preauthorization retries must not request another pose click",
            )
            run_prompt.assert_called_once()
            mission_leg_writer.assert_called_once()
            coverage.assert_not_called()
            self.assertEqual(
                call_order,
                [
                    "initialpose",
                    "localization",
                    "planning",
                    "readiness",
                    "RUN",
                    "authorization",
                ],
            )


if __name__ == "__main__":
    unittest.main()
