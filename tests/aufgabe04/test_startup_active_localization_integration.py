from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.missions.startup_route_uncertainty_selection import (
    StartupRouteUncertaintySelectionRejected,
)
from scripts.aufgabe04.real_robot.autonomous_runner.initial_coverage import (
    InitialCoveragePlanningStatusError,
    build_initial_coverage_planning_command,
    plan_initial_coverage,
)


def _args(*, enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        map=Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml"),
        semantic_map_id="arena_1p898x3p9_auto",
        session_id="stand_explore_exact2_camera_test",
        inspection_stop_spacing_m=0.70,
        exact_inspection_point_count=2,
        expected_stand_count=5,
        uncertainty_sigma_multiplier=2.0,
        enable_startup_active_localization=enabled,
        max_startup_active_localization_attempts=1,
        startup_active_localization_rotation_rad=0.35,
        startup_active_localization_angular_speed_radps=0.12,
        startup_active_localization_timeout_sec=8.0,
    )


def _profile() -> SimpleNamespace:
    return SimpleNamespace(
        map_frame="map",
        robot_radius_m=0.105,
        namespace="",
        scan_topic="scan",
        odom_topic="odom",
        cmd_vel_topic="cmd_vel",
        amcl_topic="amcl_pose",
        odom_frame="odom",
        base_frame="base_footprint",
        max_angular_speed_radps=0.20,
    )


def _option(command, name):
    return command[command.index(name) + 1]


class StartupActiveLocalizationIntegrationTest(unittest.TestCase):
    def test_adapter_preserves_certified_route_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_root = Path(tmp) / "session"
            command = build_initial_coverage_planning_command(
                args=_args(enabled=True),
                profile=_profile(),
                session_root=session_root,
                survey_root=session_root / "coverage",
                start=Pose2D(-1.6, -0.5, 0.01),
                inflation_radius_m=0.25,
                candidate_keepout_radius_m=0.31,
                route_selection_preflight_path=(
                    session_root / "preflight/preplanning_localization.json"
                ),
                route_selection_evidence_path=(
                    session_root / "attempt_selection.json"
                ),
            )

        self.assertEqual(
            _option(
                command,
                "--startup-route-selection-tracking-tube-radius-m",
            ),
            "0.03",
        )
        self.assertEqual(
            _option(command, "--startup-route-selection-sigma-multiplier"),
            "2.0",
        )
        self.assertEqual(
            _option(command, "--startup-route-selection-collision-margin-m"),
            "0.02",
        )

    def test_typed_rejection_runs_child_rechecks_amcl_and_replans(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_root = Path(tmp) / "session"
            survey_root = session_root / "coverage"
            commands = []
            child_requests = []
            admitted_paths = []
            events = []
            frozen_plan = object()

            def planner(command, **kwargs):
                commands.append((command, kwargs))
                if len(commands) == 1:
                    selection = Path(
                        _option(
                            command,
                            "--startup-route-selection-evidence-json",
                        )
                    )
                    raise StartupRouteUncertaintySelectionRejected(
                        evidence_path=selection,
                        evidence_sha256="a" * 64,
                        reason="no_accepted_route_options",
                    )
                return 0

            def child(request):
                child_requests.append(request)
                return SimpleNamespace(
                    result={
                        "status": "completed",
                        "motion_published": True,
                        "route_authorized": False,
                        "mission_run_authorized": False,
                        "requires_fresh_stationary_localization": True,
                        "requires_separate_mission_run": True,
                    }
                )

            def admit(path):
                admitted_paths.append(path)
                return Pose2D(-1.55, -0.48, 0.2)

            result = plan_initial_coverage(
                args=_args(enabled=True),
                profile=_profile(),
                session_root=session_root,
                survey_root=survey_root,
                start=Pose2D(-1.6, -0.5, 0.01),
                inflation_radius_m=0.25,
                candidate_keepout_radius_m=0.31,
                admit_stationary_localization=admit,
                append_event=lambda path, event: events.append((path, event)),
                planner=planner,
                active_localization_child=child,
                load_plan=lambda path: frozen_plan,
                wall_clock=lambda: 1.0,
            )

        plan_path, plan, leg_index, start = result
        self.assertEqual(plan_path, survey_root / "coverage_plan.json")
        self.assertIs(plan, frozen_plan)
        self.assertEqual(leg_index, 0)
        self.assertEqual(start, Pose2D(-1.55, -0.48, 0.2))
        self.assertEqual(len(commands), 2)
        self.assertEqual(
            commands[0][1],
            {"propagate_startup_route_selection_rejection": True},
        )
        self.assertEqual(
            commands[1][1],
            {"propagate_startup_route_selection_rejection": True},
        )
        self.assertEqual(child_requests[0].attempt_index, 0)
        self.assertEqual(
            admitted_paths,
            [
                session_root
                / "startup_active_localization/attempt_000/"
                "post_motion_preplanning_localization.json"
            ],
        )
        second_command = commands[1][0]
        self.assertEqual(
            _option(
                second_command,
                "--startup-route-selection-preflight-json",
            ),
            str(admitted_paths[0]),
        )
        self.assertTrue(
            _option(
                second_command,
                "--startup-route-selection-evidence-json",
            ).endswith("attempt_001/startup_route_uncertainty_selection.json")
        )
        self.assertFalse(events[-1][1]["mission_run_authorized"])

    def test_disabled_adapter_calls_legacy_planner_signature_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_root = Path(tmp) / "session"
            calls = []

            def legacy_planner(command):
                calls.append(command)
                return 0

            result = plan_initial_coverage(
                args=_args(enabled=False),
                profile=_profile(),
                session_root=session_root,
                survey_root=session_root / "coverage",
                start=Pose2D(-1.6, -0.5, 0.01),
                inflation_radius_m=0.25,
                candidate_keepout_radius_m=0.31,
                admit_stationary_localization=lambda _: self.fail(
                    "disabled path must not read another AMCL pose"
                ),
                append_event=lambda *_: self.fail(
                    "disabled path must not append active events"
                ),
                planner=legacy_planner,
                active_localization_child=lambda *_: self.fail(
                    "disabled path must not start a child"
                ),
                load_plan=lambda _: object(),
            )

        self.assertEqual(result[2], 0)
        self.assertEqual(len(calls), 1)
        self.assertNotIn(
            "--startup-route-selection-evidence-json",
            calls[0],
        )

    def test_disabled_adapter_preserves_nonzero_planner_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_root = Path(tmp) / "session"
            with self.assertRaises(
                InitialCoveragePlanningStatusError
            ) as raised:
                plan_initial_coverage(
                    args=_args(enabled=False),
                    profile=_profile(),
                    session_root=session_root,
                    survey_root=session_root / "coverage",
                    start=Pose2D(-1.6, -0.5, 0.01),
                    inflation_radius_m=0.25,
                    candidate_keepout_radius_m=0.31,
                    admit_stationary_localization=lambda _: self.fail(
                        "disabled path must not read another AMCL pose"
                    ),
                    append_event=lambda *_: self.fail(
                        "disabled path must not append active events"
                    ),
                    planner=lambda command: 2,
                    active_localization_child=lambda *_: self.fail(
                        "disabled path must not start a child"
                    ),
                    load_plan=lambda _: self.fail(
                        "failed planning must not load a plan"
                    ),
                )

        self.assertEqual(raised.exception.status, 2)


if __name__ == "__main__":
    unittest.main()
