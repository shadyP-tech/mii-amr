import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.real_robot.autonomous_session_manifest import (
    AUTONOMOUS_SESSION_MANIFEST_KIND,
    AUTONOMOUS_SESSION_MANIFEST_SCHEMA_VERSION,
    COVERAGE_LEG_CHECKPOINT_COMPLETE,
    AutonomousSessionManifest,
    AutonomousSessionManifestError,
    ParentCheckpointReference,
    admit_autonomous_session_manifest,
    artifact_file_reference,
    autonomous_session_manifest_sha256,
    load_autonomous_session_manifest,
    parent_checkpoint_reference,
    publish_coverage_checkpoint,
    write_autonomous_session_manifest,
)


class AutonomousSessionManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def _artifact_set(self, prefix: str):
        paths = {}
        for name in (
            "coverage_plan",
            "coverage_progress",
            "survey_summary",
            "stand_registry",
            "lidar_observer_summary",
        ):
            path = self.root / f"{prefix}_{name}.json"
            path.write_text(
                json.dumps({"name": name, "prefix": prefix}) + "\n",
                encoding="utf-8",
            )
            paths[name] = path
        return paths

    def _manifest(
        self,
        prefix: str = "first",
        *,
        session_id: str = "stand_explore_001",
        completed: int = 1,
        parent: ParentCheckpointReference | None = None,
    ) -> AutonomousSessionManifest:
        artifacts = self._artifact_set(prefix)
        return AutonomousSessionManifest(
            schema_version=AUTONOMOUS_SESSION_MANIFEST_SCHEMA_VERSION,
            manifest_kind=AUTONOMOUS_SESSION_MANIFEST_KIND,
            session_id=session_id,
            run_mode="execute-coverage-checkpoint",
            status=COVERAGE_LEG_CHECKPOINT_COMPLETE,
            robot_id="tb3_1",
            robot_profile_sha256="a" * 64,
            calibration_profile_sha256="b" * 64,
            physical_site_sha256="c" * 64,
            map_bundle_sha256="d" * 64,
            config_sha256="e" * 64,
            completed_coverage_legs=completed,
            next_viewpoint_id=f"viewpoint_{completed + 1:03d}",
            coverage_plan=artifact_file_reference(artifacts["coverage_plan"]),
            coverage_progress=artifact_file_reference(
                artifacts["coverage_progress"]
            ),
            survey_summary=artifact_file_reference(artifacts["survey_summary"]),
            stand_registry=artifact_file_reference(artifacts["stand_registry"]),
            lidar_observer_summary=artifact_file_reference(
                artifacts["lidar_observer_summary"]
            ),
            parent_checkpoint=parent,
        )

    def test_round_trip_and_live_admission(self):
        manifest = self._manifest()
        path = self.root / "checkpoint.json"

        digest = write_autonomous_session_manifest(path, manifest)

        self.assertEqual(digest, autonomous_session_manifest_sha256(manifest))
        self.assertEqual(load_autonomous_session_manifest(path), manifest)
        self.assertEqual(admit_autonomous_session_manifest(path), manifest)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(payload["motion_authorized"])

    def test_manifest_tampering_fails_content_hash_check(self):
        path = self.root / "checkpoint.json"
        write_autonomous_session_manifest(path, self._manifest())
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["completed_coverage_legs"] = 7
        path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaises(AutonomousSessionManifestError) as raised:
            load_autonomous_session_manifest(path)

        self.assertEqual(raised.exception.code, "hash_mismatch")

    def test_artifact_mutation_is_rejected_during_admission(self):
        manifest = self._manifest()
        path = self.root / "checkpoint.json"
        write_autonomous_session_manifest(path, manifest)
        Path(manifest.coverage_progress.path).write_text(
            '{"mutated":true}\n', encoding="utf-8"
        )

        self.assertEqual(load_autonomous_session_manifest(path), manifest)
        with self.assertRaises(AutonomousSessionManifestError) as raised:
            admit_autonomous_session_manifest(path)

        self.assertEqual(raised.exception.code, "hash_mismatch")

    def test_missing_artifact_is_rejected_during_admission(self):
        manifest = self._manifest()
        path = self.root / "checkpoint.json"
        write_autonomous_session_manifest(path, manifest)
        Path(manifest.survey_summary.path).unlink()

        with self.assertRaises(AutonomousSessionManifestError) as raised:
            admit_autonomous_session_manifest(path)

        self.assertEqual(raised.exception.code, "artifact_unavailable")

    def test_symlink_artifacts_are_rejected_at_creation_and_admission(self):
        target = self.root / "target.json"
        target.write_text("{}\n", encoding="utf-8")
        link = self.root / "artifact_link.json"
        link.symlink_to(target.name)

        with self.assertRaises(AutonomousSessionManifestError):
            artifact_file_reference(link)

        manifest = self._manifest()
        path = self.root / "checkpoint.json"
        write_autonomous_session_manifest(path, manifest)
        registry = Path(manifest.stand_registry.path)
        replacement = self.root / "registry_replacement.json"
        replacement.write_bytes(registry.read_bytes())
        registry.unlink()
        registry.symlink_to(replacement.name)

        with self.assertRaises(AutonomousSessionManifestError) as raised:
            admit_autonomous_session_manifest(path)

        self.assertIn(
            raised.exception.code,
            {"invalid_manifest", "artifact_unavailable"},
        )

    def test_checkpoint_cursor_and_authority_are_fail_closed(self):
        manifest = self._manifest()
        invalid = (
            replace(manifest, completed_coverage_legs=0),
            replace(manifest, next_viewpoint_id=""),
            replace(manifest, motion_authorized=True),
            replace(manifest, run_mode="dry-first-leg"),
            replace(manifest, map_bundle_sha256="A" * 64),
        )

        for item in invalid:
            with self.subTest(item=item):
                with self.assertRaises(AutonomousSessionManifestError):
                    autonomous_session_manifest_sha256(item)

    def test_parent_checkpoint_hash_and_parent_artifacts_are_validated(self):
        parent = self._manifest("parent", session_id="parent_session")
        parent_path = self.root / "parent_checkpoint.json"
        parent_hash = write_autonomous_session_manifest(parent_path, parent)
        reference = parent_checkpoint_reference(parent_path)
        self.assertEqual(reference.sha256, parent_hash)

        child = self._manifest(
            "child",
            session_id="child_session",
            completed=2,
            parent=reference,
        )
        child_path = self.root / "child_checkpoint.json"
        write_autonomous_session_manifest(child_path, child)
        self.assertEqual(admit_autonomous_session_manifest(child_path), child)

        wrong_parent = replace(
            child,
            parent_checkpoint=replace(reference, sha256="f" * 64),
        )
        with self.assertRaises(AutonomousSessionManifestError) as wrong_hash:
            write_autonomous_session_manifest(
                self.root / "wrong_parent_checkpoint.json", wrong_parent
            )
        self.assertEqual(wrong_hash.exception.code, "provenance_mismatch")

        payload = json.loads(parent_path.read_text(encoding="utf-8"))
        payload.pop("manifest_sha256")
        payload["session_id"] = "substituted_parent"
        payload["manifest_sha256"] = payload_sha256(payload)
        parent_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(AutonomousSessionManifestError) as substituted:
            admit_autonomous_session_manifest(child_path)
        self.assertEqual(substituted.exception.code, "provenance_mismatch")

    def test_parent_chain_rehashes_parent_artifacts(self):
        parent = self._manifest("parent", session_id="parent_session")
        parent_path = self.root / "parent_checkpoint.json"
        write_autonomous_session_manifest(parent_path, parent)
        child = self._manifest(
            "child",
            session_id="child_session",
            completed=2,
            parent=parent_checkpoint_reference(parent_path),
        )
        child_path = self.root / "child_checkpoint.json"
        write_autonomous_session_manifest(child_path, child)
        Path(parent.coverage_plan.path).write_text("changed\n", encoding="utf-8")

        with self.assertRaises(AutonomousSessionManifestError) as raised:
            admit_autonomous_session_manifest(child_path)

        self.assertEqual(raised.exception.code, "hash_mismatch")

    def test_publisher_snapshots_mutable_survey_files_and_chains_checkpoints(self):
        sources = self._artifact_set("live")
        session_root = self.root / "session"
        session_root.mkdir()
        first = publish_coverage_checkpoint(
            session_root=session_root,
            session_id="session_001",
            run_mode="execute-coverage-checkpoint",
            robot_id="tb3_1",
            robot_profile_sha256="a" * 64,
            calibration_profile_sha256="b" * 64,
            physical_site_sha256="c" * 64,
            map_bundle_sha256="d" * 64,
            config_sha256="e" * 64,
            completed_coverage_legs=1,
            next_viewpoint_id="viewpoint_002",
            coverage_plan_path=sources["coverage_plan"],
            coverage_progress_path=sources["coverage_progress"],
            survey_summary_path=sources["survey_summary"],
            stand_registry_path=sources["stand_registry"],
            lidar_observer_summary_path=sources["lidar_observer_summary"],
        )
        for source in sources.values():
            source.write_text('{"new":"live state"}\n', encoding="utf-8")

        self.assertEqual(
            admit_autonomous_session_manifest(first.manifest_path),
            first.manifest,
        )
        second = publish_coverage_checkpoint(
            session_root=session_root,
            session_id="session_001",
            run_mode="execute-full",
            robot_id="tb3_1",
            robot_profile_sha256="a" * 64,
            calibration_profile_sha256="b" * 64,
            physical_site_sha256="c" * 64,
            map_bundle_sha256="d" * 64,
            config_sha256="e" * 64,
            completed_coverage_legs=2,
            next_viewpoint_id="viewpoint_003",
            coverage_plan_path=sources["coverage_plan"],
            coverage_progress_path=sources["coverage_progress"],
            survey_summary_path=sources["survey_summary"],
            stand_registry_path=sources["stand_registry"],
            lidar_observer_summary_path=sources["lidar_observer_summary"],
            parent_checkpoint_path=first.manifest_path,
        )

        admitted = admit_autonomous_session_manifest(second.manifest_path)
        self.assertEqual(admitted.parent_checkpoint.path, str(first.manifest_path))
        self.assertFalse(admitted.motion_authorized)
        self.assertNotEqual(
            admitted.coverage_progress.path,
            str(sources["coverage_progress"]),
        )

    def test_publisher_refuses_checkpoint_directory_reuse(self):
        sources = self._artifact_set("reuse")
        session_root = self.root / "session"
        session_root.mkdir()
        kwargs = dict(
            session_root=session_root,
            session_id="session_001",
            run_mode="execute-coverage-checkpoint",
            robot_id="tb3_1",
            robot_profile_sha256="a" * 64,
            calibration_profile_sha256="b" * 64,
            physical_site_sha256="c" * 64,
            map_bundle_sha256="d" * 64,
            config_sha256="e" * 64,
            completed_coverage_legs=1,
            next_viewpoint_id="viewpoint_002",
            coverage_plan_path=sources["coverage_plan"],
            coverage_progress_path=sources["coverage_progress"],
            survey_summary_path=sources["survey_summary"],
            stand_registry_path=sources["stand_registry"],
            lidar_observer_summary_path=sources["lidar_observer_summary"],
        )
        publish_coverage_checkpoint(**kwargs)

        with self.assertRaises(AutonomousSessionManifestError) as raised:
            publish_coverage_checkpoint(**kwargs)

        self.assertEqual(raised.exception.code, "immutable_conflict")


if __name__ == "__main__":
    unittest.main()
