import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.real_robot.run_autonomous_stand_exploration import (
    MotionLegOutcome,
    _execute_coverage_leg_with_replans,
    _run_motion_leg,
)


MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")


def uncertainty_failure(run_id: str, root: Path) -> MotionLegOutcome:
    reason = (
        "odom execution admission failed: route uncertainty budget exhausted: "
        "limiting_segment=3 remaining_margin=-0.026910 m"
    )
    return MotionLegOutcome(
        run_id=run_id,
        status="preflight_failed",
        stop_reason=reason,
        stop_details={
            "reason": reason,
            "fault_code": "odom_execution_admission_failed",
            "execution_pose_owner": "odom",
            "global_consistency_monitor": "amcl",
            "motion_published": False,
            "fail_closed": True,
        },
        motion_published=False,
        returncode=1,
        semantic_log_path=root / f"{run_id}.jsonl",
    )


class AutonomousLocalizationRetryIntegrationTest(unittest.TestCase):
    def args(self, *, maximum: int) -> SimpleNamespace:
        return SimpleNamespace(
            session_id="mission",
            max_blockage_replans_per_leg=2,
            max_startup_reseals_per_leg=2,
            max_runtime_localization_reseals_per_leg=1,
            max_localization_readiness_retries_per_leg=maximum,
            uncertainty_sigma_multiplier=2.0,
            localization_branch_proof_id="known_start_marker_20260807",
            map=MAP,
            semantic_map_id="arena_1p898x3p9_auto",
        )

    def test_retries_fresh_dry_admission_with_distinct_child_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_route = root / "route.csv"
            source_diagnostics = root / "diagnostics.json"
            source_route.write_text("source\n", encoding="utf-8")
            source_diagnostics.write_text("{}\n", encoding="utf-8")
            rejected = uncertainty_failure("mission_coverage_000", root)
            completed = MotionLegOutcome(
                run_id="mission_coverage_000_localization_readiness_001",
                status="completed",
                stop_reason="",
                stop_details={},
                motion_published=True,
                returncode=0,
                semantic_log_path=root / "completed.jsonl",
            )
            sealed = {
                "route_csv": str(source_route),
                "diagnostics_json": str(source_diagnostics),
                "route_certificate_json": str(root / "certificate.json"),
            }
            with patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "seal_stand_discovery_route",
                return_value=sealed,
            ) as seal, patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "_run_motion_leg",
                side_effect=(rejected, completed),
            ) as run:
                outcome = _execute_coverage_leg_with_replans(
                    profile=SimpleNamespace(robot_radius_m=0.105),
                    args=self.args(maximum=2),
                    session_root=root / "session",
                    survey_root=root / "survey",
                    plan_path=root / "coverage_plan.json",
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                    source_route=source_route,
                    source_diagnostics=source_diagnostics,
                )

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(run.call_count, 2)
            self.assertEqual(
                run.call_args_list[1].kwargs["run_id"],
                "mission_coverage_000_localization_readiness_001",
            )
            self.assertEqual(seal.call_count, 2)
            self.assertIn(
                "coverage_leg_000_localization_readiness_001",
                str(seal.call_args_list[1].kwargs["output_dir"]),
            )
            events = [
                json.loads(line)
                for line in (root / "session/adaptive_replans.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(events[0]["event"], "localization_readiness_retry_scheduled")
            self.assertFalse(events[0]["motion_published"])
            self.assertTrue(events[0]["route_limits_unchanged"])

    def test_child_dry_failure_is_returned_only_for_bounded_retry_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "uncertain_dry"
            reason = (
                "odom execution admission failed: route uncertainty budget "
                "exhausted: limiting_segment=2 remaining_margin=-0.01 m"
            )
            event = {
                "event": "safety_stop",
                "run_id": run_id,
                "status": "preflight_failed",
                "stop_reason": reason,
                "stop_details": {
                    "reason": reason,
                    "fault_code": "odom_execution_admission_failed",
                    "execution_pose_owner": "odom",
                    "global_consistency_monitor": "amcl",
                    "motion_published": False,
                    "fail_closed": True,
                },
                "motion_published": False,
            }
            sealed = {
                "route_csv": str(root / "route.csv"),
                "diagnostics_json": str(root / "diagnostics.json"),
                "route_certificate_json": str(root / "certificate.json"),
            }

            def reject_dry(_command, **_kwargs):
                log = root / "run_events" / f"{run_id}.jsonl"
                log.parent.mkdir(parents=True)
                log.write_text(json.dumps(event) + "\n", encoding="utf-8")
                return SimpleNamespace(returncode=1)

            with patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "subprocess.run",
                side_effect=reject_dry,
            ) as run:
                outcome = _run_motion_leg(
                    profile=SimpleNamespace(
                        robot_id="tb3_1",
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
                    ),
                    sealed=sealed,
                    run_id=run_id,
                    session_root=root,
                    execute=True,
                    uncertainty_map_yaml=MAP,
                    localization_branch_proof_id="known_start",
                )

            self.assertEqual(outcome.status, "preflight_failed")
            self.assertFalse(outcome.motion_published)
            self.assertEqual(run.call_count, 1)

    def test_exhaustion_fails_closed_without_third_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_route = root / "route.csv"
            source_diagnostics = root / "diagnostics.json"
            source_route.write_text("source\n", encoding="utf-8")
            source_diagnostics.write_text("{}\n", encoding="utf-8")
            failures = (
                uncertainty_failure("mission_coverage_000", root),
                uncertainty_failure(
                    "mission_coverage_000_localization_readiness_001",
                    root,
                ),
            )
            sealed = {
                "route_csv": str(source_route),
                "diagnostics_json": str(source_diagnostics),
                "route_certificate_json": str(root / "certificate.json"),
            }
            with patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "seal_stand_discovery_route",
                return_value=sealed,
            ), patch(
                "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration."
                "_run_motion_leg",
                side_effect=failures,
            ) as run, self.assertRaisesRegex(
                RuntimeError,
                "localization readiness retry budget exhausted",
            ):
                _execute_coverage_leg_with_replans(
                    profile=SimpleNamespace(robot_radius_m=0.105),
                    args=self.args(maximum=1),
                    session_root=root / "session",
                    survey_root=root / "survey",
                    plan_path=root / "coverage_plan.json",
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                    source_route=source_route,
                    source_diagnostics=source_diagnostics,
                )

            self.assertEqual(run.call_count, 2)
            events = [
                json.loads(line)
                for line in (root / "session/adaptive_replans.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "localization_readiness_retry_scheduled",
                    "localization_readiness_retry_exhausted",
                ],
            )
            self.assertFalse(events[-1]["motion_continues_authorized"])


if __name__ == "__main__":
    unittest.main()
