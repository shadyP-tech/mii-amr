from __future__ import annotations

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.real_robot.autonomous_artifact_paths import (
    AutonomousArtifactPathError,
    CanonicalChildArtifactPaths,
    resolve_child_artifact_paths,
)


class AutonomousArtifactPathsTest(unittest.TestCase):
    def _sealed_files(self, root: Path) -> dict[str, str]:
        artifacts = root / "coverage" / "sealed"
        artifacts.mkdir(parents=True)
        paths = {
            "route_csv": artifacts / "route.csv",
            "diagnostics_json": artifacts / "route_diagnostics.json",
            "route_certificate_json": artifacts / "route_certificate.json",
        }
        for name, path in paths.items():
            path.write_text(f"{name}\n", encoding="utf-8")
        return {name: str(path) for name, path in paths.items()}

    def test_returns_frozen_canonical_paths_ready_for_child_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            session_root = Path(directory) / "session"
            session_root.mkdir()
            sealed = self._sealed_files(session_root)

            admitted = resolve_child_artifact_paths(
                session_root=session_root,
                sealed=sealed,
            )

            self.assertIsInstance(admitted, CanonicalChildArtifactPaths)
            self.assertEqual(admitted.session_root, session_root.resolve())
            self.assertEqual(
                admitted.route_csv,
                Path(sealed["route_csv"]).resolve(),
            )
            self.assertEqual(
                admitted.diagnostics_json,
                Path(sealed["diagnostics_json"]).resolve(),
            )
            self.assertEqual(
                admitted.route_certificate_json,
                Path(sealed["route_certificate_json"]).resolve(),
            )
            for path in (
                admitted.session_root,
                admitted.route_csv,
                admitted.diagnostics_json,
                admitted.route_certificate_json,
            ):
                self.assertTrue(path.is_absolute())
            with self.assertRaises(FrozenInstanceError):
                admitted.route_csv = Path("replacement.csv")  # type: ignore[misc]

    def test_relative_output_root_and_artifacts_resolve_against_working_directory(self):
        repository_root = Path.cwd().resolve()
        with tempfile.TemporaryDirectory(dir=repository_root) as directory:
            session_root = Path(directory) / "relative-session"
            session_root.mkdir()
            sealed = self._sealed_files(session_root)
            relative_root = session_root.relative_to(repository_root)
            relative_sealed = {
                name: str(Path(path).relative_to(repository_root))
                for name, path in sealed.items()
            }

            admitted = resolve_child_artifact_paths(
                session_root=relative_root,
                sealed=relative_sealed,
            )

            self.assertEqual(admitted.session_root, session_root.resolve())
            self.assertEqual(
                admitted.route_csv,
                Path(sealed["route_csv"]).resolve(),
            )

    def test_parent_path_alias_converges_on_canonical_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            actual_parent = temporary_root / "private"
            actual_parent.mkdir()
            alias_parent = temporary_root / "var"
            alias_parent.symlink_to(actual_parent, target_is_directory=True)
            canonical_session = actual_parent / "session"
            canonical_session.mkdir()
            self._sealed_files(canonical_session)
            alias_session = alias_parent / "session"
            sealed = {
                "route_csv": str(
                    alias_session / "coverage" / "sealed" / "route.csv"
                ),
                "diagnostics_json": str(
                    alias_session
                    / "coverage"
                    / "sealed"
                    / "route_diagnostics.json"
                ),
                "route_certificate_json": str(
                    alias_session
                    / "coverage"
                    / "sealed"
                    / "route_certificate.json"
                ),
            }

            admitted = resolve_child_artifact_paths(
                session_root=alias_session,
                sealed=sealed,
            )

            self.assertEqual(admitted.session_root, canonical_session.resolve())
            self.assertEqual(
                admitted.route_csv,
                (
                    canonical_session
                    / "coverage"
                    / "sealed"
                    / "route.csv"
                ).resolve(),
            )

    def test_rejects_missing_or_non_directory_session_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal_file = root / "not-a-directory"
            normal_file.write_text("file\n", encoding="utf-8")
            for session_root, message in (
                (root / "missing", "session_root is unavailable"),
                (normal_file, "session_root must resolve to a directory"),
            ):
                with self.subTest(session_root=session_root):
                    with self.assertRaisesRegex(
                        AutonomousArtifactPathError,
                        message,
                    ):
                        resolve_child_artifact_paths(
                            session_root=session_root,
                            sealed={},
                        )

    def test_rejects_incomplete_or_unknown_sealed_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            session_root = Path(directory) / "session"
            session_root.mkdir()
            sealed = self._sealed_files(session_root)
            missing = dict(sealed)
            del missing["diagnostics_json"]
            unknown = dict(sealed)
            unknown["unexpected"] = str(session_root / "unexpected")

            with self.assertRaisesRegex(
                AutonomousArtifactPathError,
                "missing=diagnostics_json",
            ):
                resolve_child_artifact_paths(
                    session_root=session_root,
                    sealed=missing,
                )
            with self.assertRaisesRegex(
                AutonomousArtifactPathError,
                "unexpected=unexpected",
            ):
                resolve_child_artifact_paths(
                    session_root=session_root,
                    sealed=unknown,
                )

    def test_rejects_missing_directory_and_non_text_artifact_values(self):
        with tempfile.TemporaryDirectory() as directory:
            session_root = Path(directory) / "session"
            session_root.mkdir()
            sealed = self._sealed_files(session_root)

            missing = dict(sealed)
            missing["route_csv"] = str(session_root / "missing.csv")
            with self.assertRaisesRegex(
                AutonomousArtifactPathError,
                "route_csv is unavailable",
            ):
                resolve_child_artifact_paths(
                    session_root=session_root,
                    sealed=missing,
                )

            directory_value = dict(sealed)
            directory_value["diagnostics_json"] = str(session_root)
            with self.assertRaisesRegex(
                AutonomousArtifactPathError,
                "diagnostics_json must resolve to a normal file",
            ):
                resolve_child_artifact_paths(
                    session_root=session_root,
                    sealed=directory_value,
                )

            non_text = dict(sealed)
            non_text["route_certificate_json"] = os.fsencode(
                sealed["route_certificate_json"]
            )
            with self.assertRaisesRegex(
                AutonomousArtifactPathError,
                "must be a non-empty text filesystem path",
            ):
                resolve_child_artifact_paths(
                    session_root=session_root,
                    sealed=non_text,  # type: ignore[arg-type]
                )

            invalid = dict(sealed)
            invalid["route_csv"] = "invalid\0route.csv"
            with self.assertRaisesRegex(
                AutonomousArtifactPathError,
                "route_csv is unavailable",
            ):
                resolve_child_artifact_paths(
                    session_root=session_root,
                    sealed=invalid,
                )

    def test_rejects_symlink_artifact_even_when_target_is_inside_session(self):
        with tempfile.TemporaryDirectory() as directory:
            session_root = Path(directory) / "session"
            session_root.mkdir()
            sealed = self._sealed_files(session_root)
            route = Path(sealed["route_csv"])
            route_link = route.with_name("route-link.csv")
            route_link.symlink_to(route)
            sealed["route_csv"] = str(route_link)

            with self.assertRaisesRegex(
                AutonomousArtifactPathError,
                "route_csv must not be a symlink",
            ):
                resolve_child_artifact_paths(
                    session_root=session_root,
                    sealed=sealed,
                )

    def test_admits_resolved_artifact_outside_session_by_exact_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_root = root / "session"
            session_root.mkdir()
            sealed = self._sealed_files(session_root)
            outside_route = root / "outside.csv"
            outside_route.write_text("outside\n", encoding="utf-8")
            sealed["route_csv"] = str(outside_route)

            admitted = resolve_child_artifact_paths(
                session_root=session_root,
                sealed=sealed,
            )

            self.assertEqual(admitted.route_csv, outside_route.resolve())


if __name__ == "__main__":
    unittest.main()
