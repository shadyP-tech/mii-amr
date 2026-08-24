from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot import (
    run_autonomous_stand_exploration as runner,
)


class AutonomousRunnerResumeTest(unittest.TestCase):
    def test_resume_checkpoint_and_mode_are_mutually_required(self):
        base = [
            "--robot-profile",
            "robot.json",
            "--camera-calibration",
            "camera.json",
            "--physical-site",
            "site.json",
        ]
        cases = (
            [*base, "--run-mode", "resume-next-coverage-leg"],
            [*base, "--resume-checkpoint", "checkpoint.json"],
        )
        for argv in cases:
            with self.subTest(argv=argv), patch(
                "sys.stderr", new=StringIO()
            ), self.assertRaises(SystemExit) as raised:
                runner.main(argv)
            self.assertEqual(raised.exception.code, 2)

    def test_runner_restores_checkpoint_before_fresh_motion_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for name in ("robot.json", "camera.json", "site.json"):
                (root / name).write_text("{}\n", encoding="utf-8")
            output_root = root / "runs"
            survey_root = output_root / "resume_session" / "coverage"
            plan = SimpleNamespace(
                viewpoints=(SimpleNamespace(), SimpleNamespace()),
                map_bundle_sha256="d" * 64,
            )
            fresh_pose = Pose2D(0.1, -0.2, 0.3)
            call_order = []
            profile = SimpleNamespace(
                robot_id="tb3_1",
                map_frame="map",
                scan_origin_to_base_offset_m=0.05,
                robot_radius_m=0.105,
                resolved_runtime=lambda: SimpleNamespace(
                    namespace="",
                    cmd_vel_topic="/cmd_vel",
                ),
            )

            def restore(_admitted, **kwargs):
                destination = Path(kwargs["survey_root"])
                (destination / "legs").mkdir(parents=True)
                (destination / "survey_summary.json").write_text(
                    '{"next_viewpoint_id":"survey_vp_002"}\n',
                    encoding="utf-8",
                )
                route = destination / "legs/leg_001_route.csv"
                diagnostics = destination / "legs/leg_001_diagnostics.json"
                route.write_text("route\n", encoding="utf-8")
                diagnostics.write_text("{}\n", encoding="utf-8")
                return SimpleNamespace(
                    plan_path=destination / "coverage_plan.json",
                    plan=plan,
                    leg_index=1,
                    parent_checkpoint_path=root / "checkpoint.json",
                    route_csv=route,
                    diagnostics_json=diagnostics,
                )

            with (
                patch.object(runner, "load_real_robot_profile", return_value=profile),
                patch.object(
                    runner,
                    "load_camera_calibration",
                    return_value=SimpleNamespace(),
                ),
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
                    side_effect=lambda *_args, **_kwargs: (
                        call_order.append("fresh_localization") or fresh_pose
                    ),
                ) as admit_localization,
                patch.object(
                    runner,
                    "_admit_observation_tf_readiness",
                    return_value=(
                        output_root
                        / (
                            "resume_session/preflight/"
                            "lidar_scan_tf_before_authorization.json"
                        ),
                        "f" * 64,
                    ),
                ),
                patch.object(
                    runner,
                    "admit_preauthorization_readiness",
                    side_effect=lambda *_args, **_kwargs: SimpleNamespace(
                        result=SimpleNamespace(attempts=(object(),)),
                        evidence_path=root / "initial_readiness.json",
                        evidence_sha256="e" * 64,
                    ),
                ),
                patch.object(
                    runner,
                    "validate_physical_site_contract",
                    return_value=SimpleNamespace(
                        expected_stand_count=5,
                        physical_site_path=(root / "site.json").resolve(),
                        map_yaml_path=(root / "map.yaml").resolve(),
                        map_bundle=SimpleNamespace(bundle_sha256="d" * 64),
                    ),
                ),
                patch.object(
                    runner,
                    "admit_coverage_resume",
                    side_effect=lambda *_args, **_kwargs: (
                        call_order.append("checkpoint_admitted") or object()
                    ),
                ) as admit_resume,
                patch.object(
                    runner,
                    "restore_and_replan_coverage_resume",
                    side_effect=restore,
                ) as restore_resume,
                patch.object(runner, "plan_stand_coverage_survey") as fresh_plan,
                patch.object(
                    runner,
                    "write_mission_leg_motion_authorization",
                    return_value="a" * 64,
                ),
                patch.object(
                    runner,
                    "write_mission_motion_authorization",
                    return_value="b" * 64,
                ),
                patch.object(
                    runner,
                    "_execute_coverage_leg_with_replans",
                    side_effect=RuntimeError("fresh motion attempt sentinel"),
                ) as execute_leg,
                patch("builtins.input", return_value="RUN"),
                patch("sys.stderr", new=StringIO()),
                redirect_stdout(StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                runner.main(
                    [
                        "--robot-profile",
                        str(root / "robot.json"),
                        "--camera-calibration",
                        str(root / "camera.json"),
                        "--physical-site",
                        str(root / "site.json"),
                        "--map",
                        str(root / "map.yaml"),
                        "--session-id",
                        "resume_session",
                        "--output-root",
                        str(output_root),
                        "--run-mode",
                        "resume-next-coverage-leg",
                        "--resume-checkpoint",
                        str(root / "checkpoint.json"),
                        "--localization-branch-proof-id",
                        "fresh_known_start",
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            admit_localization.assert_called_once()
            admit_resume.assert_called_once()
            self.assertEqual(
                call_order[:2],
                ["checkpoint_admitted", "fresh_localization"],
            )
            self.assertEqual(
                admit_resume.call_args.kwargs["new_session_id"],
                "resume_session",
            )
            self.assertEqual(
                restore_resume.call_args.kwargs["current_pose"],
                fresh_pose,
            )
            fresh_plan.assert_not_called()
            self.assertEqual(execute_leg.call_args.kwargs["leg_index"], 1)
            self.assertEqual(
                execute_leg.call_args.kwargs["source_route"],
                survey_root / "legs/leg_001_route.csv",
            )
            failure = (
                output_root / "resume_session/mission_failure.json"
            ).read_text(encoding="utf-8")
            self.assertIn("fresh motion attempt sentinel", failure)


if __name__ == "__main__":
    unittest.main()
