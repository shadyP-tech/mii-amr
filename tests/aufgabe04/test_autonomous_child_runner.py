import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MissionLegKind,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import (
    build_child_runner_command,
    parse_dry_run_outcome,
    parse_motion_leg_outcome,
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

    def test_dry_candidate_identity_is_emitted_without_authorizing_motion(self):
        command = build_child_runner_command(
            **self._base_arguments(),
            candidate_snapshot=Path("session/candidate_snapshot.json"),
            mission_leg_evidence_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            mission_leg_evidence_index=4,
            mission_leg_evidence_target_id="survey_candidate_0005",
            dry_run=True,
        )

        self.assertEqual(
            self._option(command, "--mission-leg-evidence-kind"),
            MissionLegKind.CANDIDATE_PREAPPROACH.value,
        )
        self.assertEqual(
            self._option(command, "--mission-leg-evidence-index"), "4"
        )
        self.assertEqual(
            self._option(command, "--mission-leg-evidence-target-id"),
            "survey_candidate_0005",
        )
        self.assertNotIn("--startup-reseal-motion-permit-json", command)
        self.assertNotIn("--mission-leg-motion-permit-json", command)

    def test_candidate_startup_permit_does_not_require_coverage_replanner(self):
        command = build_child_runner_command(
            **self._base_arguments(),
            candidate_snapshot=Path("session/candidate_snapshot.json"),
            mission_leg_evidence_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            mission_leg_evidence_index=4,
            mission_leg_evidence_target_id="survey_candidate_0005",
            startup_reseal_motion_authorization_json=Path(
                "session/startup_authorization.json"
            ),
            startup_reseal_motion_permit_json=Path(
                "session/startup_permits/candidate.json"
            ),
            startup_reseal_mission_leg_kind=(
                MissionLegKind.CANDIDATE_PREAPPROACH
            ),
            startup_reseal_mission_leg_index=4,
            startup_reseal_target_id="survey_candidate_0005",
            startup_reseal_semantic_map_id="arena",
            mission_session_id="mission",
            dry_run=False,
        )

        self.assertEqual(
            self._option(command, "--startup-reseal-mission-leg-kind"),
            MissionLegKind.CANDIDATE_PREAPPROACH.value,
        )
        self.assertEqual(
            self._option(command, "--startup-reseal-mission-leg-index"), "4"
        )
        self.assertEqual(
            self._option(command, "--startup-reseal-target-id"),
            "survey_candidate_0005",
        )
        self.assertNotIn("--coverage-transient-replan-leg-index", command)

    def test_candidate_startup_identity_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "identities mismatch"):
            build_child_runner_command(
                **self._base_arguments(),
                mission_leg_evidence_kind=(
                    MissionLegKind.CANDIDATE_PREAPPROACH
                ),
                mission_leg_evidence_index=4,
                mission_leg_evidence_target_id="survey_candidate_0005",
                startup_reseal_motion_authorization_json=Path(
                    "session/startup_authorization.json"
                ),
                startup_reseal_motion_permit_json=Path(
                    "session/startup_permits/candidate.json"
                ),
                startup_reseal_mission_leg_kind=(
                    MissionLegKind.OPPOSITE_FACE
                ),
                startup_reseal_mission_leg_index=4,
                startup_reseal_target_id="survey_candidate_0005",
                startup_reseal_semantic_map_id="arena",
                mission_session_id="mission",
                dry_run=False,
            )

    def test_legacy_coverage_startup_aliases_resolve_to_generic_identity(self):
        command = build_child_runner_command(
            **self._base_arguments(),
            coverage_transient_replan=self._coverage_replan(2),
            startup_reseal_motion_authorization_json=Path(
                "session/startup_authorization.json"
            ),
            startup_reseal_motion_permit_json=Path(
                "session/startup_permits/coverage.json"
            ),
            startup_reseal_target_viewpoint_id="survey_vp_002",
            startup_reseal_semantic_map_id="arena",
            mission_session_id="mission",
            dry_run=False,
        )

        self.assertEqual(
            self._option(command, "--startup-reseal-mission-leg-kind"),
            MissionLegKind.COVERAGE.value,
        )
        self.assertEqual(
            self._option(command, "--startup-reseal-mission-leg-index"), "2"
        )
        self.assertEqual(
            self._option(command, "--startup-reseal-target-id"),
            "survey_vp_002",
        )


class AutonomousChildDryOutcomeTest(unittest.TestCase):
    @staticmethod
    def _event(run_id: str = "dry_leg", **updates) -> dict[str, object]:
        event = {
            "event": "dry_run_completed",
            "run_id": run_id,
            "status": "dry_run_ok",
            "motion_published": False,
        }
        event.update(updates)
        return event

    @staticmethod
    def _write(path: Path, *events: dict[str, object]) -> None:
        path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

    def test_exact_no_motion_dry_terminal_is_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "events.jsonl"
            self._write(log, self._event())

            outcome = parse_dry_run_outcome(
                log,
                run_id="dry_leg",
                returncode=0,
            )

        self.assertEqual(outcome.status, "dry_run_ok")
        self.assertFalse(outcome.motion_published)

    def test_missing_duplicate_or_malformed_dry_terminal_fails_closed(self):
        cases = {
            "missing": ({"event": "run_finished", "run_id": "dry_leg"},),
            "duplicate": (self._event(), self._event()),
            "wrong_status": (self._event(status="noop"),),
            "motion_true": (self._event(motion_published=True),),
            "motion_string": (self._event(motion_published="false"),),
        }
        for label, events in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                log = Path(tmp) / "events.jsonl"
                self._write(log, *events)
                with self.assertRaises(RuntimeError):
                    parse_dry_run_outcome(
                        log,
                        run_id="dry_leg",
                        returncode=0,
                    )

    def test_conflicting_same_run_terminal_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "events.jsonl"
            self._write(
                log,
                self._event(),
                {
                    "event": "safety_stop",
                    "run_id": "dry_leg",
                    "status": "stopped",
                    "motion_published": False,
                },
            )

            with self.assertRaisesRegex(RuntimeError, "ambiguous terminal"):
                parse_dry_run_outcome(
                    log,
                    run_id="dry_leg",
                    returncode=0,
                )

    def test_zero_returncode_is_required_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "events.jsonl"
            self._write(log, self._event())
            for invalid in (1, False):
                with self.subTest(returncode=invalid), self.assertRaises(
                    RuntimeError
                ):
                    parse_dry_run_outcome(
                        log,
                        run_id="dry_leg",
                        returncode=invalid,
                    )


class AutonomousChildMotionOutcomeTest(unittest.TestCase):
    def test_conservative_exception_safety_stop_is_a_terminal_failure(self):
        event = {
            "event": "safety_stop",
            "run_id": "failed_leg",
            "status": "stopped",
            "stop_reason": "unexpected follower exception: NameError",
            "motion_published": True,
            "stop_details": {
                "fault_code": "unexpected_follower_exception",
                "fail_closed": True,
                "motion_history_uncertain": True,
                "continuation_allowed": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "events.jsonl"
            log.write_text(json.dumps(event) + "\n", encoding="utf-8")

            outcome = parse_motion_leg_outcome(
                log,
                run_id="failed_leg",
                returncode=1,
            )

        self.assertEqual(outcome.status, "stopped")
        self.assertTrue(outcome.motion_published)
        self.assertTrue(outcome.stop_details["fail_closed"])
        self.assertTrue(outcome.stop_details["motion_history_uncertain"])
        self.assertFalse(outcome.stop_details["continuation_allowed"])


if __name__ == "__main__":
    unittest.main()
