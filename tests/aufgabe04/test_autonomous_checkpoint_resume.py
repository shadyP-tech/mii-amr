import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.plan_stand_coverage_survey import (
    main as plan_coverage,
)
from scripts.aufgabe04.navigation.mission_leg_motion_permit import (
    load_mission_leg_motion_permit,
)
from scripts.aufgabe04.navigation.runtime_motion_authorization import (
    load_runtime_localization_motion_permit,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    load_coverage_survey_plan,
    load_stand_survey_registry,
    load_survey_progress,
    mark_viewpoint_visited,
    write_stand_survey_registry,
    write_survey_progress,
)
from scripts.aufgabe04.real_robot.autonomous_checkpoint_resume import (
    admit_coverage_resume,
    restore_and_replan_coverage_resume,
)
from scripts.aufgabe04.real_robot.autonomous_session_manifest import (
    COVERAGE_SURVEY_TERMINAL_CHECKPOINT,
    AutonomousSessionManifestError,
    publish_coverage_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "maps/aufgabe03/arena_1p898x3p9_auto.yaml"


class AutonomousCheckpointResumeTest(unittest.TestCase):
    def _checkpoint(self, root: Path):
        live = root / "live_session"
        survey = live / "coverage"
        with redirect_stdout(StringIO()):
            status = plan_coverage(
                [
                    "--map",
                    str(MAP),
                    "--semantic-map-id",
                    "arena_1p898x3p9_auto",
                    "--planning-frame",
                    "map",
                    "--start-x",
                    "0",
                    "--start-y",
                    "0",
                    "--start-yaw",
                    "0",
                    "--survey-id",
                    "parent_session",
                    "--output-dir",
                    str(survey),
                    "--lane-count",
                    "1",
                    "--exact-inspection-point-count",
                    "2",
                    "--expected-stand-count",
                    "1",
                ]
            )
        self.assertEqual(status, 0)
        plan = load_coverage_survey_plan(survey / "coverage_plan.json")
        progress = load_survey_progress(survey / "coverage_progress.json", plan)
        registry = load_stand_survey_registry(survey / "stand_registry.json", plan)
        progress = mark_viewpoint_visited(
            plan,
            progress,
            plan.viewpoints[0].viewpoint_id,
        )
        write_survey_progress(survey / "coverage_progress.json", progress, plan)
        write_stand_survey_registry(survey / "stand_registry.json", registry, plan)
        (survey / "survey_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "next_viewpoint_id": plan.viewpoints[1].viewpoint_id,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        observer = root / "observer_summary.json"
        observer.write_text('{"motion_published":false}\n', encoding="utf-8")
        published = publish_coverage_checkpoint(
            session_root=live,
            session_id="parent_session",
            run_mode="execute-coverage-checkpoint",
            robot_id="tb3_1",
            robot_profile_sha256="a" * 64,
            calibration_profile_sha256="b" * 64,
            physical_site_sha256="c" * 64,
            map_bundle_sha256=plan.map_bundle_sha256,
            config_sha256="e" * 64,
            completed_coverage_legs=1,
            next_viewpoint_id=plan.viewpoints[1].viewpoint_id,
            coverage_plan_path=survey / "coverage_plan.json",
            coverage_progress_path=survey / "coverage_progress.json",
            survey_summary_path=survey / "survey_summary.json",
            stand_registry_path=survey / "stand_registry.json",
            lidar_observer_summary_path=observer,
        )
        return plan, published

    def test_resume_rehashes_identity_and_replans_from_fresh_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            plan, published = self._checkpoint(root)
            admitted = admit_coverage_resume(
                published.manifest_path,
                new_session_id="resume_session",
                robot_id="tb3_1",
                robot_profile_sha256="a" * 64,
                calibration_profile_sha256="b" * 64,
                physical_site_sha256="c" * 64,
                map_bundle_sha256=plan.map_bundle_sha256,
                config_sha256="e" * 64,
            )
            fresh_pose = plan.viewpoints[0].pose
            restored = restore_and_replan_coverage_resume(
                admitted,
                survey_root=root / "resume_session/coverage",
                map_yaml=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                current_pose=fresh_pose,
            )

            self.assertEqual(restored.leg_index, 1)
            self.assertEqual(
                restored.target_viewpoint_id,
                plan.viewpoints[1].viewpoint_id,
            )
            self.assertTrue(restored.route_csv.is_file())
            diagnostics = json.loads(restored.diagnostics_json.read_text())
            metadata = diagnostics["metadata"]
            self.assertFalse(metadata["motion_authorized"])
            self.assertFalse(metadata["resume_motion_authorized"])
            self.assertEqual(
                metadata["resume_checkpoint_manifest_sha256"],
                published.manifest_sha256,
            )
            receipt = json.loads(
                (restored.survey_root / "resume_admission.json").read_text()
            )
            self.assertFalse(receipt["old_motion_permits_reused"])

    def test_resume_rejects_same_session_or_changed_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            plan, published = self._checkpoint(root)
            common = dict(
                checkpoint_path=published.manifest_path,
                robot_id="tb3_1",
                robot_profile_sha256="a" * 64,
                calibration_profile_sha256="b" * 64,
                physical_site_sha256="c" * 64,
                map_bundle_sha256=plan.map_bundle_sha256,
                config_sha256="e" * 64,
            )
            with self.assertRaises(AutonomousSessionManifestError):
                admit_coverage_resume(
                    new_session_id="parent_session",
                    **common,
                )
            with self.assertRaisesRegex(
                AutonomousSessionManifestError,
                "config_sha256 differs",
            ):
                admit_coverage_resume(
                    new_session_id="resume_session",
                    **{**common, "config_sha256": "f" * 64},
                )

            checkpoint_link = root / "checkpoint_link.json"
            checkpoint_link.symlink_to(published.manifest_path)
            with self.assertRaisesRegex(
                AutonomousSessionManifestError,
                "must not be a symlink",
            ):
                admit_coverage_resume(
                    new_session_id="resume_session",
                    **{**common, "checkpoint_path": checkpoint_link},
                )

    def test_checkpoint_cannot_be_consumed_as_a_motion_permit(self):
        with tempfile.TemporaryDirectory() as tmp:
            _plan, published = self._checkpoint(Path(tmp).resolve())

            for loader in (
                load_mission_leg_motion_permit,
                load_runtime_localization_motion_permit,
            ):
                with self.subTest(loader=loader.__name__):
                    with self.assertRaisesRegex(ValueError, "artifact is missing"):
                        loader(published.manifest_path)

    def test_terminal_checkpoint_is_explicitly_non_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            plan, resumable = self._checkpoint(root)
            terminal_root = root / "terminal_session"
            terminal_root.mkdir()
            manifest = resumable.manifest
            terminal = publish_coverage_checkpoint(
                session_root=terminal_root,
                session_id="terminal_session",
                run_mode="execute-coverage-checkpoint",
                robot_id=manifest.robot_id,
                robot_profile_sha256=manifest.robot_profile_sha256,
                calibration_profile_sha256=(
                    manifest.calibration_profile_sha256
                ),
                physical_site_sha256=manifest.physical_site_sha256,
                map_bundle_sha256=manifest.map_bundle_sha256,
                config_sha256=manifest.config_sha256,
                completed_coverage_legs=2,
                next_viewpoint_id=None,
                coverage_plan_path=Path(manifest.coverage_plan.path),
                coverage_progress_path=Path(manifest.coverage_progress.path),
                survey_summary_path=Path(manifest.survey_summary.path),
                stand_registry_path=Path(manifest.stand_registry.path),
                lidar_observer_summary_path=Path(
                    manifest.lidar_observer_summary.path
                ),
                status=COVERAGE_SURVEY_TERMINAL_CHECKPOINT,
            )

            with self.assertRaisesRegex(
                AutonomousSessionManifestError,
                "evidence-only and cannot be resumed",
            ) as raised:
                admit_coverage_resume(
                    terminal.manifest_path,
                    new_session_id="resume_terminal_session",
                    robot_id="tb3_1",
                    robot_profile_sha256="a" * 64,
                    calibration_profile_sha256="b" * 64,
                    physical_site_sha256="c" * 64,
                    map_bundle_sha256=plan.map_bundle_sha256,
                    config_sha256="e" * 64,
                )

            self.assertEqual(raised.exception.code, "invalid_cursor")


if __name__ == "__main__":
    unittest.main()
