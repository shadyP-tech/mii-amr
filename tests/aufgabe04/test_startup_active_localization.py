from pathlib import Path
import math
import tempfile
import unittest

from scripts.aufgabe04.navigation.localization.startup_active_localization import (
    STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION,
    STARTUP_ACTIVE_LOCALIZATION_SCHEMA_VERSION,
    RotationProgress,
    StartupActiveLocalizationConfig,
    StartupActiveLocalizationMotionResult,
    advance_rotation_progress,
    load_startup_active_localization_result,
    startup_active_localization_attempt_dir,
    startup_active_localization_result_payload,
    startup_active_localization_signed_turn,
    translation_from_start_m,
    write_startup_active_localization_result,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.missions.startup_route_uncertainty_selection import (
    StartupRouteUncertaintySelectionRejected,
)
from scripts.aufgabe04.real_robot.execution.startup_active_localization import (
    StartupActiveLocalizationChildRequest,
    build_startup_active_localization_child_command,
)
from scripts.aufgabe04.real_robot.readiness.active_localization import (
    StartupActiveLocalizationPlanningConfig,
    StartupActiveLocalizationPlanningEffects,
    plan_with_optional_startup_active_localization,
)


def _config(**overrides) -> StartupActiveLocalizationConfig:
    values = {
        "enabled": True,
        "max_attempts": 3,
        "rotation_rad": 0.35,
        "angular_speed_radps": 0.12,
        "timeout_sec": 8.0,
    }
    values.update(overrides)
    return StartupActiveLocalizationConfig(**values)


def _motion_result(config: StartupActiveLocalizationConfig):
    return StartupActiveLocalizationMotionResult(
        status="completed",
        stop_reason="",
        duration_sec=3.0,
        requested_rotation_rad=config.rotation_rad,
        accumulated_progress_rad=config.target_progress_rad + 0.01,
        accumulated_reverse_rad=0.0,
        maximum_translation_m=0.004,
        motion_published=True,
        zero_command_count=config.stop_command_count,
        stop_details={"stationary_odom": {"accepted": True}},
    )


class StartupActiveLocalizationContractTest(unittest.TestCase):
    def test_attempt_directions_alternate_with_same_bound(self):
        config = _config()

        self.assertAlmostEqual(
            startup_active_localization_signed_turn(config, attempt_index=0),
            0.35,
        )
        self.assertAlmostEqual(
            startup_active_localization_signed_turn(config, attempt_index=1),
            -0.35,
        )
        self.assertAlmostEqual(
            startup_active_localization_signed_turn(config, attempt_index=2),
            0.35,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            config.direction_for_attempt(3)

    def test_config_rejects_unbounded_or_unfinishable_rotation(self):
        with self.assertRaisesRegex(ValueError, r"2\*pi"):
            _config(rotation_rad=2.0 * math.pi + 0.01, timeout_sec=60.0)
        with self.assertRaisesRegex(ValueError, "must cover"):
            _config(rotation_rad=1.0, timeout_sec=1.0)
        with self.assertRaisesRegex(ValueError, "angular_speed_radps"):
            _config(angular_speed_radps=0.0)

    def test_progress_unwraps_yaw_and_counts_only_authorized_direction(self):
        progress = RotationProgress(previous_yaw_rad=math.pi - 0.05)
        progress = advance_rotation_progress(
            progress,
            current_yaw_rad=-math.pi + 0.05,
            direction=1,
        )
        progress = advance_rotation_progress(
            progress,
            current_yaw_rad=-math.pi,
            direction=1,
        )

        self.assertAlmostEqual(progress.accumulated_progress_rad, 0.10)
        self.assertAlmostEqual(progress.accumulated_reverse_rad, 0.05)

    def test_translation_guard_is_euclidean_from_start(self):
        self.assertAlmostEqual(
            translation_from_start_m((1.0, -2.0), (1.03, -1.96)),
            0.05,
        )

    def test_result_is_content_hashed_and_never_authorizes_mission(self):
        config = _config()
        payload = startup_active_localization_result_payload(
            run_id="run",
            attempt_index=0,
            result=_motion_result(config),
            config=config,
            runtime_config={"cmd_vel_topic": "/cmd_vel"},
            source_route_selection_json=Path("selection.json"),
            source_route_selection_sha256="a" * 64,
            preflight_json=Path("preflight.json"),
            preflight_sha256="b" * 64,
            controller_trace_jsonl=Path("controller_trace.jsonl"),
        )

        self.assertEqual(
            payload["schema_version"],
            STARTUP_ACTIVE_LOCALIZATION_SCHEMA_VERSION,
        )
        self.assertEqual(
            payload["operator_confirmation"],
            STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION,
        )
        self.assertFalse(payload["route_authorized"])
        self.assertFalse(payload["mission_run_authorized"])
        self.assertTrue(payload["requires_fresh_stationary_localization"])
        self.assertTrue(payload["requires_separate_mission_run"])
        self.assertFalse(payload["translation_commanded"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            digest = write_startup_active_localization_result(path, payload)
            loaded = load_startup_active_localization_result(path)

        self.assertEqual(len(digest), 64)
        self.assertEqual(loaded["phase"], "startup_active_localization")
        self.assertEqual(loaded["status"], "completed")

    def test_attempt_directory_is_stable_and_namespaced(self):
        self.assertEqual(
            startup_active_localization_attempt_dir(
                Path("session"),
                attempt_index=3,
            ),
            Path("session/startup_active_localization/attempt_003"),
        )

    def test_planning_state_machine_retries_only_bound_rejection(self):
        session_root = Path("session")
        events: list[dict[str, object]] = []
        attempts = []
        admitted_pose = Pose2D(-1.55, -0.50, 0.25)

        def plan_initial_route(attempt):
            attempts.append(attempt)
            if len(attempts) == 1:
                raise StartupRouteUncertaintySelectionRejected(
                    evidence_path=attempt.selection_evidence_path,
                    evidence_sha256="a" * 64,
                    reason="no_accepted_route_options",
                )
            return 0

        def run_active(attempt_index, rejection):
            self.assertEqual(attempt_index, 0)
            self.assertEqual(
                rejection.evidence_path,
                Path(
                    "session/startup_active_localization/attempt_000/"
                    "startup_route_uncertainty_selection.json"
                ),
            )
            return {
                "status": "completed",
                "motion_published": True,
                "route_authorized": False,
                "mission_run_authorized": False,
                "requires_fresh_stationary_localization": True,
                "requires_separate_mission_run": True,
            }

        def admit_stationary(path):
            self.assertEqual(
                path,
                Path(
                    "session/startup_active_localization/attempt_000/"
                    "post_motion_preplanning_localization.json"
                ),
            )
            return admitted_pose

        outcome = plan_with_optional_startup_active_localization(
            StartupActiveLocalizationPlanningConfig(
                session_root=session_root,
                motion=_config(max_attempts=1),
            ),
            StartupActiveLocalizationPlanningEffects(
                plan_initial_route=plan_initial_route,
                run_active_localization=run_active,
                admit_stationary_localization=admit_stationary,
                append_event=lambda _path, event: events.append(event),
                wall_clock=lambda: 1.0,
            ),
            initial_start=Pose2D(-1.61, -0.51, 0.01),
        )

        self.assertEqual(outcome.planning_status, 0)
        self.assertEqual(outcome.start, admitted_pose)
        self.assertEqual(outcome.planning_attempt_count, 2)
        self.assertEqual(outcome.active_localization_attempt_count, 1)
        self.assertEqual(
            [event["event"] for event in events],
            [
                "startup_active_localization_enabled",
                "startup_active_localization_scheduled",
                "startup_active_localization_completed",
            ],
        )
        self.assertEqual(events[1]["attempt_index"], 0)
        self.assertFalse(events[2]["mission_run_authorized"])

    def test_parent_child_command_keeps_zero_based_attempt_directory(self):
        profile = type(
            "Profile",
            (),
            {
                "namespace": "",
                "scan_topic": "scan",
                "odom_topic": "odom",
                "cmd_vel_topic": "cmd_vel",
                "amcl_topic": "amcl_pose",
                "map_frame": "map",
                "odom_frame": "odom",
                "base_frame": "base_footprint",
                "max_angular_speed_radps": 0.18,
            },
        )()
        request = StartupActiveLocalizationChildRequest(
            session_id="run",
            session_root=Path("session"),
            profile=profile,
            config=StartupActiveLocalizationConfig(enabled=True),
            attempt_index=0,
            rejected_selection=StartupRouteUncertaintySelectionRejected(
                evidence_path=Path("selection.json"),
                evidence_sha256="a" * 64,
                reason="no_accepted_route_options",
            ),
        )

        command, result_path, semantic_log_path, controller_trace_path = (
            build_startup_active_localization_child_command(request)
        )

        self.assertEqual(command[command.index("--attempt-index") + 1], "0")
        self.assertEqual(
            result_path,
            Path(
                "session/startup_active_localization/attempt_000/"
                "startup_active_localization_result.json"
            ),
        )
        self.assertEqual(
            semantic_log_path.name,
            "startup_active_localization_events.jsonl",
        )
        self.assertEqual(controller_trace_path.name, "controller_trace.jsonl")


if __name__ == "__main__":
    unittest.main()
