import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.real_robot import (
    run_autonomous_stand_exploration as runner,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome


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
                run_id="dry_session_coverage_000_dry",
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
                semantic_log_path=root / "rejected.jsonl",
            )
            passed = MotionLegOutcome(
                run_id=(
                    "dry_session_coverage_000_dry_"
                    "localization_readiness_001"
                ),
                status="dry_run_ok",
                stop_reason="",
                stop_details={},
                motion_published=False,
                returncode=0,
                semantic_log_path=root / "passed.jsonl",
            )
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
                patch.object(runner, "plan_stand_coverage_survey", return_value=0),
                patch.object(
                    runner,
                    "load_coverage_survey_plan",
                    return_value=SimpleNamespace(viewpoints=(object(),)),
                ),
                patch.object(
                    runner,
                    "seal_stand_discovery_route",
                    return_value={
                        "route_csv": "route.csv",
                        "diagnostics_json": "diagnostics.json",
                        "route_certificate_json": "certificate.json",
                    },
                ),
                patch.object(
                    runner,
                    "_run_motion_leg",
                    side_effect=(rejected, passed),
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
                "dry_session_coverage_000_dry_localization_readiness_001",
            )
            summary = json.loads(
                (output_root / "dry_session/mission_summary.json").read_text()
            )
            self.assertEqual(summary["localization_readiness_retry_count"], 1)
            event = json.loads(
                (output_root / "dry_session/adaptive_replans.jsonl").read_text()
            )
            self.assertFalse(event["motion_authorized"])
            self.assertTrue(event["route_limits_unchanged"])


if __name__ == "__main__":
    unittest.main()
