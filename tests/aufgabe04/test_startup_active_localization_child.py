from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.localization.startup_active_localization import (
    STARTUP_ACTIVE_LOCALIZATION_AUTHORIZATION_HASH_FIELD,
    STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION,
    STARTUP_ACTIVE_LOCALIZATION_PREFLIGHT_HASH_FIELD,
    StartupActiveLocalizationConfig,
    StartupActiveLocalizationMotionResult,
    startup_active_localization_attempt_dir,
    startup_active_localization_result_payload,
    write_startup_active_localization_result,
)
from scripts.aufgabe04.navigation.missions.startup_route_uncertainty_selection import (
    StartupRouteUncertaintySelectionRejected,
)
from scripts.aufgabe04.real_robot.execution.startup_active_localization import (
    StartupActiveLocalizationChildRequest,
    build_startup_active_localization_child_command,
    run_startup_active_localization_child,
)


def _config():
    return StartupActiveLocalizationConfig(
        enabled=True,
        max_attempts=1,
        rotation_rad=0.35,
        angular_speed_radps=0.12,
        timeout_sec=8.0,
    )


def _profile():
    return SimpleNamespace(
        namespace="",
        scan_topic="scan",
        odom_topic="odom",
        cmd_vel_topic="cmd_vel",
        amcl_topic="amcl_pose",
        map_frame="map",
        odom_frame="odom",
        base_frame="base_footprint",
        max_angular_speed_radps=0.20,
    )


def _request(session_root: Path):
    selection_path = (
        startup_active_localization_attempt_dir(
            session_root,
            attempt_index=0,
        )
        / "startup_route_uncertainty_selection.json"
    )
    return StartupActiveLocalizationChildRequest(
        session_id="stand_explore_exact2_camera_test",
        session_root=session_root,
        profile=_profile(),
        config=_config(),
        attempt_index=0,
        rejected_selection=StartupRouteUncertaintySelectionRejected(
            evidence_path=selection_path,
            evidence_sha256="a" * 64,
            reason="no_accepted_route_options",
        ),
    )


class StartupActiveLocalizationChildTest(unittest.TestCase):
    def test_child_command_has_bounded_localize_scope_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            command, result_path, _, trace_path = (
                build_startup_active_localization_child_command(
                    _request(Path(tmp) / "session")
                )
            )

        self.assertIn("--rotation-rad", command)
        self.assertIn("--maximum-angular-speed-radps", command)
        self.assertEqual(
            command[command.index("--maximum-angular-speed-radps") + 1],
            "0.2",
        )
        self.assertNotIn("--execute", command)
        self.assertNotIn("RUN", command)
        self.assertEqual(
            result_path.name,
            "startup_active_localization_result.json",
        )
        self.assertEqual(trace_path.name, "controller_trace.jsonl")

    def test_parent_accepts_only_bound_result_with_stopped_odom_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = _request(Path(tmp) / "session")
            command, result_path, semantic_log_path, trace_path = (
                build_startup_active_localization_child_command(request)
            )

            def run_process(actual_command, *, check):
                self.assertEqual(actual_command, command)
                self.assertFalse(check)
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(
                    '{"event":"startup_active_localization_cycle"}\n',
                    encoding="utf-8",
                )
                semantic_log_path.write_text(
                    '{"event":"startup_active_localization_finished"}\n',
                    encoding="utf-8",
                )
                preflight_path = (
                    result_path.parent
                    / "startup_active_localization_preflight.json"
                )
                preflight_sha256 = write_content_hashed_json(
                    preflight_path,
                    {"ok": True, "failures": []},
                    hash_field=STARTUP_ACTIVE_LOCALIZATION_PREFLIGHT_HASH_FIELD,
                )
                authorization_path = (
                    result_path.parent
                    / "startup_active_localization_authorization.json"
                )
                write_content_hashed_json(
                    authorization_path,
                    {
                        "schema_version": 1,
                        "phase": "startup_active_localization",
                        "run_id": (
                            f"{request.session_id}_"
                            "startup_active_localization_000"
                        ),
                        "attempt_index": 0,
                        "operator_confirmation": (
                            STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION
                        ),
                        "scope": (
                            "one bounded in-place startup localization rotation"
                        ),
                        "config": request.config.to_evidence_dict(),
                        "runtime_config": {"cmd_vel_topic": "/cmd_vel"},
                        "source_route_selection_json": str(
                            request.rejected_selection.evidence_path
                        ),
                        "source_route_selection_sha256": (
                            request.rejected_selection.evidence_sha256
                        ),
                        "preflight_json": str(preflight_path),
                        "preflight_sha256": preflight_sha256,
                        "route_authorized": False,
                        "mission_run_authorized": False,
                    },
                    hash_field=(
                        STARTUP_ACTIVE_LOCALIZATION_AUTHORIZATION_HASH_FIELD
                    ),
                )
                result = StartupActiveLocalizationMotionResult(
                    status="completed",
                    stop_reason="",
                    duration_sec=3.0,
                    requested_rotation_rad=request.config.rotation_rad,
                    accumulated_progress_rad=(
                        request.config.target_progress_rad + 0.01
                    ),
                    accumulated_reverse_rad=0.0,
                    maximum_translation_m=0.002,
                    motion_published=True,
                    zero_command_count=request.config.stop_command_count,
                    stop_details={
                        "stationary_odom": {"accepted": True},
                    },
                )
                payload = startup_active_localization_result_payload(
                    run_id=(
                        f"{request.session_id}_startup_active_localization_000"
                    ),
                    attempt_index=0,
                    result=result,
                    config=request.config,
                    runtime_config={"cmd_vel_topic": "/cmd_vel"},
                    source_route_selection_json=(
                        request.rejected_selection.evidence_path
                    ),
                    source_route_selection_sha256=(
                        request.rejected_selection.evidence_sha256
                    ),
                    preflight_json=preflight_path,
                    preflight_sha256=preflight_sha256,
                    controller_trace_jsonl=trace_path,
                )
                write_startup_active_localization_result(result_path, payload)
                return SimpleNamespace(returncode=0)

            outcome = run_startup_active_localization_child(
                request,
                run_process=run_process,
            )

        self.assertEqual(outcome.returncode, 0)
        self.assertEqual(outcome.result["status"], "completed")
        self.assertFalse(outcome.result["mission_run_authorized"])

    def test_existing_attempt_artifact_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = _request(Path(tmp) / "session")
            _, result_path, _, _ = build_startup_active_localization_child_command(
                request
            )
            result_path.parent.mkdir(parents=True)
            result_path.write_text("stale\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "refusing to reuse"):
                run_startup_active_localization_child(
                    request,
                    run_process=lambda *_args, **_kwargs: self.fail(
                        "existing evidence must stop before subprocess launch"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
