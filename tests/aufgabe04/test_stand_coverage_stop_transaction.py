from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.planning.map_io import (
    load_occupancy_grid_with_bundle,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.coverage.record_stand_coverage_stop import (
    commit_stand_coverage_stop,
    plan_next_stand_coverage_leg,
    record_stand_coverage_stop,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    build_coverage_survey_plan,
    load_coverage_survey_plan,
    load_survey_progress,
    new_stand_survey_registry,
    new_survey_progress,
    write_coverage_survey_plan,
    write_stand_survey_registry,
    write_survey_progress,
)
from scripts.aufgabe04.real_robot.autonomous_session_manifest import (
    publish_coverage_checkpoint,
)


def _write_free_map(root: Path) -> Path:
    width = 50
    height = 30
    (root / "map.pgm").write_text(
        f"P2\n{width} {height}\n255\n"
        + " ".join(["255"] * width * height)
        + "\n"
    )
    (root / "map.yaml").write_text(
        "\n".join(
            [
                "image: map.pgm",
                "resolution: 0.1",
                "origin: [-2.5, -1.5, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.20",
                "mode: trinary",
            ]
        )
        + "\n"
    )
    return root / "map.yaml"


def _write_negative_observer_summary(
    path: Path,
    *,
    observations_path: Path,
    map_bundle_sha256: str,
    viewpoint_pose: Pose2D,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "motion_published": False,
                "processed_scan_count": 5,
                "accepted_observation_count": 0,
                "map_bundle_sha256": map_bundle_sha256,
                "planning_frame": "map",
                "output_jsonl": str(observations_path),
                "scan_frame_pose_in_planning_frame": {
                    "x_m": viewpoint_pose.x_m,
                    "y_m": viewpoint_pose.y_m,
                    "yaw_rad": viewpoint_pose.yaw_rad,
                },
            }
        )
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StandCoverageStopTransactionTest(unittest.TestCase):
    def _survey_fixture(self, root: Path):
        root = root.resolve(strict=True)
        map_yaml = _write_free_map(root)
        grid, bundle = load_occupancy_grid_with_bundle(
            map_yaml,
            semantic_map_id=map_yaml.stem,
            planning_frame="map",
        )
        survey = build_coverage_survey_plan(
            grid,
            map_bundle_sha256=bundle.bundle_sha256,
            start=Pose2D(-1.5, 0.0, 0.0),
            survey_id="survey_transaction",
            arena_bounds=ArenaBounds(length_m=4.0, width_m=2.0),
        )
        survey_root = root / "survey"
        write_coverage_survey_plan(survey_root / "coverage_plan.json", survey)
        write_survey_progress(
            survey_root / "coverage_progress.json",
            new_survey_progress(survey),
            survey,
        )
        write_stand_survey_registry(
            survey_root / "stand_registry.json",
            new_stand_survey_registry(survey),
            survey,
        )
        viewpoint = survey.viewpoints[0]
        observer_summary = root / "observer_summary.json"
        observations_path = root / "negative_observations.jsonl"
        _write_negative_observer_summary(
            observer_summary,
            observations_path=observations_path,
            map_bundle_sha256=bundle.bundle_sha256,
            viewpoint_pose=viewpoint.pose,
        )
        return map_yaml, survey_root, survey, viewpoint, observer_summary

    def _publish_checkpoint(self, root: Path, survey_root: Path, status: dict[str, object]):
        root = root.resolve(strict=True)
        survey_root = survey_root.resolve(strict=True)
        plan = load_coverage_survey_plan(survey_root / "coverage_plan.json")
        observer = root / "checkpoint_lidar_observer_summary.json"
        observer.write_text('{"motion_published": false}\n')
        session_root = root / "session"
        session_root.mkdir(exist_ok=True)
        return publish_coverage_checkpoint(
            session_root=session_root,
            session_id="session_transaction",
            run_mode="execute-full",
            robot_id="tb3_1",
            robot_profile_sha256="a" * 64,
            calibration_profile_sha256="b" * 64,
            physical_site_sha256="c" * 64,
            map_bundle_sha256=plan.map_bundle_sha256,
            config_sha256="d" * 64,
            completed_coverage_legs=int(status["visited_viewpoint_count"]),
            next_viewpoint_id=str(status["next_viewpoint_id"]),
            coverage_plan_path=survey_root / "coverage_plan.json",
            coverage_progress_path=survey_root / "coverage_progress.json",
            survey_summary_path=survey_root / "survey_summary.json",
            stand_registry_path=survey_root / "stand_registry.json",
            lidar_observer_summary_path=observer,
        )

    def _prepare_after_commit(
        self,
        root: Path,
        map_yaml: Path,
        survey_root: Path,
        viewpoint,
        observer_summary: Path,
    ):
        committed = commit_stand_coverage_stop(
            survey_root=survey_root,
            map_yaml=map_yaml,
            viewpoint_id=viewpoint.viewpoint_id,
            observer_summary_json=observer_summary,
        )
        published = self._publish_checkpoint(root, survey_root, committed)
        evidence = root / "fresh_localization.json"
        evidence.write_text("{}\n")
        return committed, published, evidence

    def test_commit_writes_state_but_no_next_route_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve(strict=True)
            map_yaml, survey_root, survey, viewpoint, observer_summary = (
                self._survey_fixture(root)
            )

            status = commit_stand_coverage_stop(
                survey_root=survey_root,
                map_yaml=map_yaml,
                viewpoint_id=viewpoint.viewpoint_id,
                observer_summary_json=observer_summary,
            )
            progress = load_survey_progress(
                survey_root / "coverage_progress.json",
                survey,
            )

            self.assertEqual(status["status"], "coverage_stop_recorded")
            self.assertEqual(status["next_route_csv"], None)
            self.assertEqual(status["next_diagnostics_json"], None)
            self.assertIsNotNone(status["next_viewpoint_id"])
            self.assertEqual(progress.visited_viewpoint_ids, (viewpoint.viewpoint_id,))
            self.assertFalse((survey_root / "legs").exists())
            self.assertTrue(Path(status["epoch_json"]).is_file())

    def test_planner_requires_fresh_localization_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml, survey_root, _, viewpoint, observer_summary = (
                self._survey_fixture(root)
            )
            status = commit_stand_coverage_stop(
                survey_root=survey_root,
                map_yaml=map_yaml,
                viewpoint_id=viewpoint.viewpoint_id,
                observer_summary_json=observer_summary,
            )
            published = self._publish_checkpoint(root, survey_root, status)

            with self.assertRaisesRegex(ValueError, "localization evidence"):
                plan_next_stand_coverage_leg(
                    survey_root=survey_root,
                    map_yaml=map_yaml,
                    expected_next_viewpoint_id=str(status["next_viewpoint_id"]),
                    current_pose=viewpoint.pose,
                    localization_evidence_json=root / "missing_evidence.json",
                    localization_evidence_sha256="0" * 64,
                    checkpoint_manifest_json=published.manifest_path,
                    checkpoint_manifest_sha256=published.manifest_sha256,
                )

            self.assertFalse((survey_root / "legs").exists())

    def test_planner_preserves_committed_target_identity_before_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml, survey_root, _, viewpoint, observer_summary = (
                self._survey_fixture(root)
            )
            _, published, evidence = self._prepare_after_commit(
                root,
                map_yaml,
                survey_root,
                viewpoint,
                observer_summary,
            )

            with self.assertRaisesRegex(ValueError, "differs from committed"):
                plan_next_stand_coverage_leg(
                    survey_root=survey_root,
                    map_yaml=map_yaml,
                    expected_next_viewpoint_id="wrong_viewpoint",
                    current_pose=viewpoint.pose,
                    localization_evidence_json=evidence,
                    localization_evidence_sha256=_sha256(evidence),
                    checkpoint_manifest_json=published.manifest_path,
                    checkpoint_manifest_sha256=published.manifest_sha256,
                )

            self.assertFalse((survey_root / "legs").exists())

    def test_planner_rejects_checkpoint_hash_mismatch_before_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml, survey_root, _, viewpoint, observer_summary = (
                self._survey_fixture(root)
            )
            committed, published, evidence = self._prepare_after_commit(
                root,
                map_yaml,
                survey_root,
                viewpoint,
                observer_summary,
            )

            with self.assertRaisesRegex(ValueError, "checkpoint manifest SHA-256"):
                plan_next_stand_coverage_leg(
                    survey_root=survey_root,
                    map_yaml=map_yaml,
                    expected_next_viewpoint_id=str(committed["next_viewpoint_id"]),
                    current_pose=viewpoint.pose,
                    localization_evidence_json=evidence,
                    localization_evidence_sha256=_sha256(evidence),
                    checkpoint_manifest_json=published.manifest_path,
                    checkpoint_manifest_sha256="0" * 64,
                )

            self.assertFalse((survey_root / "legs").exists())

    def test_planner_rejects_fake_checkpoint_even_when_raw_file_hash_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml, survey_root, _, viewpoint, observer_summary = (
                self._survey_fixture(root)
            )
            committed = commit_stand_coverage_stop(
                survey_root=survey_root,
                map_yaml=map_yaml,
                viewpoint_id=viewpoint.viewpoint_id,
                observer_summary_json=observer_summary,
            )
            evidence = root / "fresh_localization.json"
            evidence.write_text("{}\n")
            fake_checkpoint = root / "fake_checkpoint.json"
            fake_checkpoint.write_text('{"not": "a checkpoint"}\n')

            with self.assertRaises(ValueError):
                plan_next_stand_coverage_leg(
                    survey_root=survey_root,
                    map_yaml=map_yaml,
                    expected_next_viewpoint_id=str(committed["next_viewpoint_id"]),
                    current_pose=viewpoint.pose,
                    localization_evidence_json=evidence,
                    localization_evidence_sha256=_sha256(evidence),
                    checkpoint_manifest_json=fake_checkpoint,
                    checkpoint_manifest_sha256=_sha256(fake_checkpoint),
                )

            self.assertFalse((survey_root / "legs").exists())

    def test_planner_rejects_checkpoint_next_target_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve(strict=True)
            map_yaml, survey_root, survey, viewpoint, observer_summary = (
                self._survey_fixture(root)
            )
            committed = commit_stand_coverage_stop(
                survey_root=survey_root,
                map_yaml=map_yaml,
                viewpoint_id=viewpoint.viewpoint_id,
                observer_summary_json=observer_summary,
            )
            wrong_target = next(
                item.viewpoint_id
                for item in survey.viewpoints
                if item.viewpoint_id != committed["next_viewpoint_id"]
                and item.viewpoint_id != viewpoint.viewpoint_id
            )
            observer = root / "checkpoint_lidar_observer_summary.json"
            observer.write_text('{"motion_published": false}\n')
            session_root = root / "session"
            session_root.mkdir()
            published = publish_coverage_checkpoint(
                session_root=session_root,
                session_id="session_wrong_target",
                run_mode="execute-full",
                robot_id="tb3_1",
                robot_profile_sha256="a" * 64,
                calibration_profile_sha256="b" * 64,
                physical_site_sha256="c" * 64,
                map_bundle_sha256=survey.map_bundle_sha256,
                config_sha256="d" * 64,
                completed_coverage_legs=int(committed["visited_viewpoint_count"]),
                next_viewpoint_id=wrong_target,
                coverage_plan_path=survey_root / "coverage_plan.json",
                coverage_progress_path=survey_root / "coverage_progress.json",
                survey_summary_path=survey_root / "survey_summary.json",
                stand_registry_path=survey_root / "stand_registry.json",
                lidar_observer_summary_path=observer,
            )
            evidence = root / "fresh_localization.json"
            evidence.write_text("{}\n")

            with self.assertRaisesRegex(ValueError, "checkpoint next viewpoint"):
                plan_next_stand_coverage_leg(
                    survey_root=survey_root,
                    map_yaml=map_yaml,
                    expected_next_viewpoint_id=str(committed["next_viewpoint_id"]),
                    current_pose=viewpoint.pose,
                    localization_evidence_json=evidence,
                    localization_evidence_sha256=_sha256(evidence),
                    checkpoint_manifest_json=published.manifest_path,
                    checkpoint_manifest_sha256=published.manifest_sha256,
                )

            self.assertFalse((survey_root / "legs").exists())

    def test_planner_rejects_summary_progress_and_registry_drift(self):
        cases = ("survey_summary", "coverage_progress", "stand_registry")
        for artifact_name in cases:
            with self.subTest(artifact_name=artifact_name):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    map_yaml, survey_root, survey, viewpoint, observer_summary = (
                        self._survey_fixture(root)
                    )
                    committed, published, evidence = self._prepare_after_commit(
                        root,
                        map_yaml,
                        survey_root,
                        viewpoint,
                        observer_summary,
                    )
                    if artifact_name == "survey_summary":
                        path = survey_root / "survey_summary.json"
                    elif artifact_name == "coverage_progress":
                        path = survey_root / "coverage_progress.json"
                    else:
                        path = survey_root / "stand_registry.json"
                    path.write_text(path.read_text() + "\n")

                    with self.assertRaisesRegex(
                        ValueError,
                        "checkpoint does not bind the committed survey state",
                    ):
                        plan_next_stand_coverage_leg(
                            survey_root=survey_root,
                            map_yaml=map_yaml,
                            expected_next_viewpoint_id=str(
                                committed["next_viewpoint_id"]
                            ),
                            current_pose=viewpoint.pose,
                            localization_evidence_json=evidence,
                            localization_evidence_sha256=_sha256(evidence),
                            checkpoint_manifest_json=published.manifest_path,
                            checkpoint_manifest_sha256=published.manifest_sha256,
                        )

                    self.assertFalse((survey_root / "legs").exists())

    def test_planner_writes_receipt_without_rewriting_committed_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml, survey_root, _, viewpoint, observer_summary = (
                self._survey_fixture(root)
            )
            committed, published, evidence = self._prepare_after_commit(
                root,
                map_yaml,
                survey_root,
                viewpoint,
                observer_summary,
            )
            summary_path = survey_root / "survey_summary.json"
            summary_before = summary_path.read_bytes()
            summary_sha256_before = hashlib.sha256(summary_before).hexdigest()

            planned = plan_next_stand_coverage_leg(
                survey_root=survey_root,
                map_yaml=map_yaml,
                expected_next_viewpoint_id=str(committed["next_viewpoint_id"]),
                current_pose=viewpoint.pose,
                localization_evidence_json=evidence,
                localization_evidence_sha256=_sha256(evidence),
                checkpoint_manifest_json=published.manifest_path,
                checkpoint_manifest_sha256=published.manifest_sha256,
            )

            self.assertTrue(Path(str(planned["next_route_csv"])).is_file())
            self.assertTrue(Path(str(planned["next_diagnostics_json"])).is_file())
            diagnostics = json.loads(
                Path(str(planned["next_diagnostics_json"])).read_text()
            )
            metadata = diagnostics["metadata"]
            self.assertEqual(
                planned["next_leg_localization_evidence_json"],
                str(evidence),
            )
            self.assertEqual(
                planned["next_leg_localization_evidence_sha256"],
                _sha256(evidence),
            )
            self.assertEqual(
                planned["checkpoint_manifest_sha256"],
                published.manifest_sha256,
            )
            self.assertEqual(
                planned["committed_survey_summary_sha256"],
                summary_sha256_before,
            )
            self.assertEqual(
                metadata["next_leg_localization_evidence_sha256"],
                _sha256(evidence),
            )
            self.assertEqual(
                metadata["checkpoint_manifest_sha256"],
                published.manifest_sha256,
            )
            self.assertEqual(
                metadata["committed_survey_summary_sha256"],
                summary_sha256_before,
            )
            self.assertEqual(summary_path.read_bytes(), summary_before)
            self.assertEqual(
                hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                summary_sha256_before,
            )

    def test_legacy_api_still_records_stop_and_writes_next_route(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml, survey_root, survey, viewpoint, observer_summary = (
                self._survey_fixture(root)
            )

            status = record_stand_coverage_stop(
                survey_root=survey_root,
                map_yaml=map_yaml,
                viewpoint_id=viewpoint.viewpoint_id,
                observer_summary_json=observer_summary,
            )
            progress = load_survey_progress(
                survey_root / "coverage_progress.json",
                survey,
            )

            self.assertEqual(progress.visited_viewpoint_ids, (viewpoint.viewpoint_id,))
            self.assertTrue(Path(str(status["next_route_csv"])).is_file())
            self.assertTrue(Path(str(status["next_diagnostics_json"])).is_file())


if __name__ == "__main__":
    unittest.main()
