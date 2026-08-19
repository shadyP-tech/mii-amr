from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.navigation.mission_leg_motion_permit import (
    MissionLegKind,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import (
    build_child_runner_command,
)


class AutonomousChildRunnerRouteIdentityTest(unittest.TestCase):
    @staticmethod
    def _profile():
        return SimpleNamespace(
            robot_id="turtlebot1",
            namespace="",
            scan_topic="scan",
            odom_topic="odom",
            cmd_vel_topic="cmd_vel",
            amcl_topic="amcl_pose",
            map_frame="map",
            odom_frame="odom",
            base_frame="base_footprint",
            localization_source="amcl",
            max_linear_speed_mps=0.055,
            max_angular_speed_radps=0.18,
            robot_radius_m=0.105,
        )

    def _base_arguments(self) -> dict[str, object]:
        return {
            "profile": self._profile(),
            "route_csv": Path("sealed/route.csv"),
            "diagnostics_json": Path("sealed/diagnostics.json"),
            "certificate_json": Path("sealed/certificate.json"),
            "run_id": "mission_leg",
            "session_root": Path("session"),
        }

    @staticmethod
    def _coverage_replan(leg_index: int) -> dict[str, object]:
        return {
            "survey_root": Path("survey"),
            "session_root": Path("session"),
            "map_yaml": Path("maps/arena.yaml"),
            "semantic_map_id": "arena",
            "target_viewpoint_id": f"survey_vp_{leg_index:03d}",
            "robot_radius_m": 0.105,
            "max_replans": 3,
            "leg_index": leg_index,
        }

    @staticmethod
    def _mission_leg_arguments(
        *, kind: MissionLegKind, mission_leg_index: int
    ) -> dict[str, object]:
        return {
            "mission_leg_motion_authorization_json": Path(
                "session/mission_authorization.json"
            ),
            "mission_leg_motion_permit_json": Path(
                "session/permits/leg.json"
            ),
            "mission_leg_kind": kind,
            "mission_leg_index": mission_leg_index,
            "mission_leg_target_id": "target",
            "mission_leg_semantic_map_id": "arena",
            "mission_leg_dry_preflight_json": Path(
                "session/preflight/leg_dry.json"
            ),
            "mission_leg_dry_odom_certificate_json": Path(
                "session/odom/leg_dry_certificate.json"
            ),
            "mission_leg_dry_uncertainty_budget_json": Path(
                "session/odom/leg_dry_budget.json"
            ),
            "mission_session_id": "mission",
        }

    @staticmethod
    def _option(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def test_coverage_mission_index_does_not_select_route_artifact_leg(self):
        command = build_child_runner_command(
            **self._base_arguments(),
            coverage_transient_replan=self._coverage_replan(1),
            dry_run=False,
            **self._mission_leg_arguments(
                kind=MissionLegKind.COVERAGE,
                mission_leg_index=1,
            ),
        )

        self.assertEqual(self._option(command, "--leg-index"), "0")
        self.assertEqual(self._option(command, "--mission-leg-index"), "1")
        self.assertEqual(
            self._option(command, "--coverage-transient-replan-leg-index"),
            "1",
        )

    def test_coverage_transient_index_does_not_select_route_artifact_leg(self):
        command = build_child_runner_command(
            **self._base_arguments(),
            coverage_transient_replan=self._coverage_replan(2),
            dry_run=True,
        )

        self.assertEqual(self._option(command, "--leg-index"), "0")
        self.assertEqual(
            self._option(command, "--coverage-transient-replan-leg-index"),
            "2",
        )

    def test_candidate_mission_index_does_not_select_route_artifact_leg(self):
        command = build_child_runner_command(
            **self._base_arguments(),
            candidate_snapshot=Path("session/candidate_snapshot.json"),
            dry_run=False,
            **self._mission_leg_arguments(
                kind=MissionLegKind.CANDIDATE_PREAPPROACH,
                mission_leg_index=1,
            ),
        )

        self.assertEqual(self._option(command, "--leg-index"), "0")
        self.assertEqual(self._option(command, "--mission-leg-index"), "1")
        self.assertEqual(
            self._option(command, "--mission-leg-kind"),
            MissionLegKind.CANDIDATE_PREAPPROACH.value,
        )

    def test_explicit_nonzero_route_artifact_index_is_preserved(self):
        command = build_child_runner_command(
            **self._base_arguments(),
            route_artifact_leg_index=4,
            coverage_transient_replan=self._coverage_replan(2),
            dry_run=True,
        )

        self.assertEqual(self._option(command, "--leg-index"), "4")
        self.assertEqual(
            self._option(command, "--coverage-transient-replan-leg-index"),
            "2",
        )

    def test_route_artifact_index_validation_fails_closed(self):
        for invalid in (-1, True, 1.5, "1"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "route_artifact_leg_index must be a non-negative integer",
            ):
                build_child_runner_command(
                    **self._base_arguments(),
                    route_artifact_leg_index=invalid,
                    dry_run=True,
                )

    def test_new_and_legacy_route_index_keywords_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_child_runner_command(
                **self._base_arguments(),
                route_artifact_leg_index=0,
                leg_index=0,
                dry_run=True,
            )

    def test_legacy_route_index_keyword_remains_an_explicit_alias(self):
        command = build_child_runner_command(
            **self._base_arguments(),
            leg_index=3,
            dry_run=True,
        )

        self.assertEqual(self._option(command, "--leg-index"), "3")


if __name__ == "__main__":
    unittest.main()
