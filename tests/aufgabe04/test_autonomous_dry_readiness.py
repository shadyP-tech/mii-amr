import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.autonomous_runner import runtime as runner
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome


class AutonomousDryReadinessTest(unittest.TestCase):
    def test_dry_mode_retries_only_no_motion_uncertainty_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            output_root = root / "runs"
            reason = (
                "odom execution admission failed: route uncertainty budget "
                "exhausted: limiting_segment=0 remaining_margin=-0.01 m"
            )
            rejected = MotionLegOutcome(
                run_id="dry_session_preauthorization_coverage_000_000",
                status="preflight_failed",
                stop_reason=reason,
                stop_details={
                    "fault_code": "odom_execution_admission_failed",
                    "execution_pose_owner": "odom",
                    "global_consistency_monitor": "amcl",
                    "fail_closed": True,
                },
                motion_published=False,
                returncode=1,
                semantic_log_path=(
                    output_root
                    / "dry_session"
                    / "run_events"
                    / "dry_session_preauthorization_coverage_000_000.jsonl"
                ),
            )
            passed = MotionLegOutcome(
                run_id="dry_session_preauthorization_coverage_000_001",
                status="dry_run_ok",
                stop_reason="",
                stop_details={},
                motion_published=False,
                returncode=0,
                semantic_log_path=(
                    output_root
                    / "dry_session"
                    / "run_events"
                    / "dry_session_preauthorization_coverage_000_001.jsonl"
                ),
                dry_preflight_path=(
                    output_root
                    / "dry_session"
                    / "preflight"
                    / "dry_session_preauthorization_coverage_000_001_dry.json"
                ),
                odom_execution_certificate_path=(
                    output_root
                    / "dry_session"
                    / "odom_execution"
                    / (
                        "dry_session_preauthorization_coverage_000_001_"
                        "dry_certificate.json"
                    )
                ),
                dry_uncertainty_budget_path=(
                    output_root
                    / "dry_session"
                    / "odom_execution"
                    / (
                        "dry_session_preauthorization_coverage_000_001_"
                        "dry_uncertainty_budget.json"
                    )
                ),
            )
            outcomes = iter((rejected, passed))

            def run_motion_leg(**_kwargs):
                outcome = next(outcomes)
                if outcome.status == "dry_run_ok":
                    outcome.semantic_log_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    outcome.semantic_log_path.write_text(
                        json.dumps(
                            {
                                "event": "dry_run_completed",
                                "run_id": outcome.run_id,
                                "status": "dry_run_ok",
                                "motion_published": False,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    for artifact in (
                        outcome.dry_preflight_path,
                        outcome.odom_execution_certificate_path,
                        outcome.dry_uncertainty_budget_path,
                    ):
                        artifact.parent.mkdir(parents=True, exist_ok=True)
                        artifact.write_text("{}\n", encoding="utf-8")
                return outcome
            profile = SimpleNamespace(
                map_frame="map",
                robot_radius_m=0.105,
                scan_origin_to_base_offset_m=0.05,
                resolved_runtime=lambda: SimpleNamespace(namespace=""),
            )
            with (
                patch.object(runner, "load_real_robot_profile", return_value=profile),
                patch.object(
                    runner,
                    "load_camera_calibration",
                    return_value=SimpleNamespace(),
                ),
                patch.object(
                    runner,
                    "validate_physical_site_contract",
                    return_value=SimpleNamespace(
                        expected_stand_count=5,
                        physical_site_path=(root / "site.json"),
                        map_yaml_path=(root / "map.yaml"),
                        map_bundle=SimpleNamespace(bundle_sha256="a" * 64),
                    ),
                ) as validate_site,
                patch.object(runner, "_validate_inputs"),
                patch.object(
                    runner,
                    "_physical_clearance",
                    return_value={
                        "minimum_active_standoff_m": 0.20,
                        "minimum_static_inflation_m": 0.25,
                        "minimum_candidate_transit_radius_m": 0.31,
                    },
                ),
                patch.object(
                    runner,
                    "_admit_preplanning_localization",
                    return_value=Pose2D(0.0, 0.0, 0.0),
                ),
                patch.object(
                    runner,
                    "_admit_observation_tf_readiness",
                    return_value=(
                        output_root
                        / (
                            "dry_session/preflight/"
                            "lidar_scan_tf_before_authorization.json"
                        ),
                        "f" * 64,
                    ),
                ),
                patch.object(runner, "plan_stand_coverage_survey", return_value=0),
                patch.object(
                    runner,
                    "load_coverage_survey_plan",
                    return_value=SimpleNamespace(viewpoints=(object(),)),
                ),
                patch.object(
                    runner,
                    "seal_stand_discovery_route",
                    side_effect=lambda **kwargs: {
                        "route_csv": str(kwargs["output_dir"] / "route.csv"),
                        "diagnostics_json": str(
                            kwargs["output_dir"] / "route_diagnostics.json"
                        ),
                        "route_certificate_json": str(
                            kwargs["output_dir"] / "route_certificate.json"
                        ),
                    },
                ),
                patch.object(
                    runner,
                    "_run_motion_leg",
                    side_effect=run_motion_leg,
                ) as run_leg,
                redirect_stdout(StringIO()),
            ):
                status = runner.main(
                    [
                        "--robot-profile",
                        str(root / "robot.json"),
                        "--camera-calibration",
                        str(root / "camera.json"),
                        "--physical-site",
                        str(root / "site.json"),
                        "--output-root",
                        str(output_root),
                        "--session-id",
                        "dry_session",
                        "--run-mode",
                        "dry-first-leg",
                    ]
                )

            self.assertEqual(status, 0)
            validate_site.assert_called_once()
            self.assertEqual(
                validate_site.call_args.kwargs[
                    "requested_expected_stand_count"
                ],
                None,
            )
            self.assertEqual(run_leg.call_count, 2)
            self.assertEqual(
                run_leg.call_args_list[1].kwargs["run_id"],
                "dry_session_preauthorization_coverage_000_001",
            )
            summary = json.loads(
                (output_root / "dry_session/mission_summary.json").read_text()
            )
            self.assertEqual(summary["localization_readiness_retry_count"], 1)
            events = [
                json.loads(line)
                for line in (
                    output_root / "dry_session/adaptive_replans.jsonl"
                ).read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(events), 2)
            self.assertTrue(all(not event["motion_authorized"] for event in events))
            self.assertTrue(
                all(event["route_limits_unchanged"] for event in events)
            )


if __name__ == "__main__":
    unittest.main()
