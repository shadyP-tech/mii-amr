from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.aufgabe04.navigation.route_revision_store as revision_module
from scripts.aufgabe04.navigation.route_revision_store import (
    RouteRevisionError,
    RouteRevisionStore,
    read_committed_revision,
    read_route_revision,
)


ROUTE_CSV = """leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,yaw_rad,segment_length_m,cumulative_length_m
0,0,0,0,0.0,0.0,0.0,0.0,0.0
0,1,1,0,0.1,0.0,0.0,0.1,0.1
0,2,2,0,0.2,0.0,0.0,0.1,0.2
"""


def _publish(
    store: RouteRevisionStore,
    *,
    target_revision: int = 1,
    observation_unix_sec: float = 99.0,
    length: float = 0.2,
    takeover: bool = False,
):
    return store.publish_active(
        ROUTE_CSV,
        {"planner": "test", "ok": True},
        target_revision=target_revision,
        observation_unix_sec=observation_unix_sec,
        source_robot_pose={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        target={"x_m": 0.2, "y_m": 0.0, "yaw_rad": 0.0},
        evidence={"axis_rad": 0.0, "confidence": 0.9},
        previous_route_length_m=0.0,
        new_route_length_m=length,
        safety_diagnostics={"keepout_clear": True, "corridor_clear": True},
        takeover=takeover,
    )


class TestRouteRevisionStore(unittest.TestCase):
    def test_active_publish_writes_immutable_artifacts_before_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            manifest = tmp_path / "route_manifest.json"
            store = RouteRevisionStore(
                manifest,
                stream_id="sim-run-1",
                writer_id="planner-a",
                now_fn=lambda: 100.0,
            )
            real_atomic_replace = revision_module._atomic_replace
            observed: dict[str, bool] = {}

            def assert_artifacts_first(path: Path, data: bytes) -> None:
                payload = json.loads(data)
                route = path.parent / payload["route"]["relative_path"]
                diagnostics = path.parent / payload["diagnostics"]["relative_path"]
                observed["route_exists"] = route.is_file()
                observed["diagnostics_exists"] = diagnostics.is_file()
                real_atomic_replace(path, data)

            with mock.patch.object(
                revision_module, "_atomic_replace", new=assert_artifacts_first
            ):
                loaded = _publish(store)

            self.assertEqual(
                observed, {"route_exists": True, "diagnostics_exists": True}
            )
            self.assertEqual(loaded.status, "active")
            self.assertEqual(loaded.route_revision, 1)
            self.assertEqual(
                loaded.route_path, store.revision_dir / "route_000001.csv"
            )
            self.assertEqual(
                loaded.diagnostics_path,
                store.revision_dir / "diagnostics_000001.json",
            )
            self.assertEqual(loaded.route_hash, loaded.manifest["route"]["sha256"])
            self.assertIs(loaded.manifest["simulation_only"], True)
            self.assertIs(
                loaded.manifest["safety_diagnostics"]["corridor_clear"], True
            )

    def test_withdrawal_is_manifest_only_and_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            manifest = tmp_path / "route_manifest.json"
            store = RouteRevisionStore(
                manifest,
                stream_id="sim",
                writer_id="planner",
                now_fn=lambda: 100.0,
            )
            _publish(store)
            artifacts_before = sorted(path.name for path in store.revision_dir.iterdir())

            withdrawn = store.withdraw("camera evidence expired")

            self.assertEqual(withdrawn.status, "withdrawn")
            self.assertEqual(withdrawn.route_revision, 2)
            self.assertIsNone(withdrawn.route_path)
            self.assertEqual(withdrawn.reason, "camera evidence expired")
            self.assertEqual(
                sorted(path.name for path in store.revision_dir.iterdir()),
                artifacts_before,
            )
            committed = read_committed_revision(
                manifest, expected_stream_id="sim", now_unix_sec=100.0
            )
            self.assertEqual(committed.status, "withdrawn")
            self.assertEqual(committed.reason, "camera evidence expired")

    def test_survey_completion_is_success_terminal_and_preserves_route_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "route_manifest.json"
            store = RouteRevisionStore(
                manifest,
                stream_id="sim",
                writer_id="planner",
                now_fn=lambda: 100.0,
            )
            active = _publish(store)

            completed = store.complete_survey(
                "arrival pose recorded",
                completion={
                    "candidate_uid": "candidate-a",
                    "catalog_revision": 1,
                    "catalog_sha256": "a" * 64,
                },
            )

            self.assertEqual(completed.status, "survey_complete")
            self.assertEqual(completed.reason, "arrival pose recorded")
            self.assertEqual(completed.route_revision, active.route_revision + 1)
            self.assertEqual(completed.route_hash, active.route_hash)
            self.assertIsNone(completed.route_path)
            self.assertFalse(completed.manifest["completion"].get("fail_closed", False))
            self.assertEqual(
                store.complete_survey(
                    "arrival pose recorded",
                    completion={
                        "candidate_uid": "candidate-a",
                        "catalog_revision": 1,
                        "catalog_sha256": "a" * 64,
                    },
                ).route_revision,
                completed.route_revision,
            )

    def test_initial_withdrawal_commits_fail_closed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            manifest = tmp_path / "route_manifest.json"
            store = RouteRevisionStore(
                manifest,
                stream_id="sim",
                writer_id="planner",
                now_fn=lambda: 100.0,
            )

            withdrawn = store.withdraw(
                "recommendation unavailable",
                target_revision=0,
                observation_unix_sec=100.0,
            )

            self.assertEqual(withdrawn.status, "withdrawn")
            self.assertEqual(withdrawn.route_revision, 1)
            self.assertIsNone(withdrawn.route_path)
            self.assertEqual(withdrawn.reason, "recommendation unavailable")

    def test_hash_corruption_is_rejected(self) -> None:
        for artifact_key in ("route", "diagnostics"):
            with self.subTest(artifact_key=artifact_key):
                with tempfile.TemporaryDirectory() as temp_dir:
                    tmp_path = Path(temp_dir)
                    manifest = tmp_path / "route_manifest.json"
                    store = RouteRevisionStore(
                        manifest,
                        stream_id="sim",
                        writer_id="planner",
                        now_fn=lambda: 100.0,
                    )
                    loaded = _publish(store)
                    artifact = (
                        loaded.route_path
                        if artifact_key == "route"
                        else loaded.diagnostics_path
                    )
                    self.assertIsNotNone(artifact)
                    assert artifact is not None
                    artifact.write_bytes(artifact.read_bytes() + b"corruption")

                    with self.assertRaisesRegex(
                        RouteRevisionError, "hash mismatch"
                    ) as raised:
                        read_committed_revision(
                            manifest,
                            expected_stream_id="sim",
                            now_unix_sec=100.0,
                        )
                    self.assertEqual(
                        raised.exception.code, "artifact_hash_mismatch"
                    )

    def test_restart_resumes_revision_and_explicit_takeover_increments_generation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            manifest = tmp_path / "route_manifest.json"
            first = RouteRevisionStore(
                manifest,
                stream_id="sim",
                writer_id="planner-a",
                now_fn=lambda: 100.0,
            )
            self.assertEqual(_publish(first, target_revision=1).route_revision, 1)

            restarted = RouteRevisionStore(
                manifest,
                stream_id="sim",
                writer_id="planner-a",
                now_fn=lambda: 101.0,
            )
            resumed = _publish(
                restarted, target_revision=2, observation_unix_sec=101.0
            )
            self.assertEqual(resumed.route_revision, 2)
            self.assertEqual(resumed.writer_generation, 1)

            other = RouteRevisionStore(
                manifest,
                stream_id="sim",
                writer_id="planner-b",
                now_fn=lambda: 102.0,
            )
            with self.assertRaises(RouteRevisionError) as raised:
                _publish(other, target_revision=3, observation_unix_sec=102.0)
            self.assertEqual(raised.exception.code, "writer_conflict")

            takeover = _publish(
                other,
                target_revision=3,
                observation_unix_sec=102.0,
                takeover=True,
            )
            self.assertEqual(takeover.route_revision, 3)
            self.assertEqual(takeover.writer_generation, 2)
            self.assertEqual(
                takeover.manifest["writer_takeover"]["previous_writer_id"],
                "planner-a",
            )
            self.assertEqual(
                takeover.manifest["writer_takeover"][
                    "previous_writer_generation"
                ],
                1,
            )

    def test_unsafe_stream_and_writer_components_are_rejected(self) -> None:
        for bad_id in ("../escape", "a/b", ".", "..", "two words", ""):
            with self.subTest(bad_id=bad_id):
                with tempfile.TemporaryDirectory() as temp_dir:
                    tmp_path = Path(temp_dir)
                    with self.assertRaises(RouteRevisionError) as stream_error:
                        RouteRevisionStore(
                            tmp_path / "m.json", stream_id=bad_id, writer_id="ok"
                        )
                    self.assertEqual(stream_error.exception.code, "unsafe_component")
                    with self.assertRaises(RouteRevisionError) as writer_error:
                        RouteRevisionStore(
                            tmp_path / "m.json", stream_id="ok", writer_id=bad_id
                        )
                    self.assertEqual(writer_error.exception.code, "unsafe_component")

    def test_manifest_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            manifest = tmp_path / "route_manifest.json"
            store = RouteRevisionStore(
                manifest,
                stream_id="sim",
                writer_id="planner",
                now_fn=lambda: 100.0,
            )
            _publish(store)
            payload = json.loads(manifest.read_text())
            payload["route"]["relative_path"] = "../outside.csv"
            manifest.write_text(json.dumps(payload))

            with self.assertRaises(RouteRevisionError) as raised:
                read_committed_revision(
                    manifest,
                    expected_stream_id="sim",
                    now_unix_sec=100.0,
                )
            self.assertEqual(raised.exception.code, "unsafe_path")

    def test_manifest_and_observation_age_limits_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            manifest = tmp_path / "route_manifest.json"
            store = RouteRevisionStore(
                manifest,
                stream_id="sim",
                writer_id="planner",
                now_fn=lambda: 100.0,
            )
            _publish(store, observation_unix_sec=90.0)

            # Fresh manifest, stale observation.
            with self.assertRaises(RouteRevisionError) as observation_error:
                read_committed_revision(
                    manifest,
                    expected_stream_id="sim",
                    now_unix_sec=100.5,
                    max_manifest_age_sec=1.0,
                    max_observation_age_sec=5.0,
                )
            self.assertEqual(observation_error.exception.code, "observation_stale")

            # Observation allowance does not mask planner/manifest death.
            with self.assertRaises(RouteRevisionError) as manifest_error:
                read_committed_revision(
                    manifest,
                    expected_stream_id="sim",
                    now_unix_sec=102.0,
                    max_manifest_age_sec=1.0,
                    max_observation_age_sec=20.0,
                )
            self.assertEqual(manifest_error.exception.code, "manifest_stale")

    def test_stream_writer_duplicate_rollback_conflict_and_gap_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            manifest = tmp_path / "route_manifest.json"
            store = RouteRevisionStore(
                manifest,
                stream_id="sim",
                writer_id="planner",
                now_fn=lambda: 100.0,
            )
            first = _publish(store)
            first_bytes = manifest.read_bytes()

            with self.assertRaises(RouteRevisionError) as stream_error:
                read_route_revision(
                    manifest,
                    expected_stream_id="other",
                    now_unix_sec=100.0,
                )
            self.assertEqual(stream_error.exception.code, "wrong_stream")
            with self.assertRaises(RouteRevisionError) as writer_error:
                read_route_revision(
                    manifest,
                    expected_writer_id="other",
                    now_unix_sec=100.0,
                )
            self.assertEqual(writer_error.exception.code, "wrong_writer")

            duplicate = read_route_revision(
                manifest,
                last_route_revision=first.route_revision,
                last_manifest_sha256=first.manifest_sha256,
                now_unix_sec=100.0,
            )
            self.assertIs(duplicate.duplicate, True)

            changed = json.loads(first_bytes)
            changed["test_only_extra"] = True
            manifest.write_text(json.dumps(changed))
            with self.assertRaises(RouteRevisionError) as conflict:
                read_route_revision(
                    manifest,
                    last_route_revision=1,
                    last_manifest_sha256=first.manifest_sha256,
                    now_unix_sec=100.0,
                )
            self.assertEqual(conflict.exception.code, "duplicate_revision_conflict")

            manifest.write_bytes(first_bytes)
            second = _publish(store, target_revision=2, observation_unix_sec=100.0)
            manifest.write_bytes(first_bytes)
            with self.assertRaises(RouteRevisionError) as rollback:
                read_route_revision(
                    manifest,
                    last_route_revision=second.route_revision,
                    last_manifest_sha256=second.manifest_sha256,
                    now_unix_sec=100.0,
                )
            self.assertEqual(rollback.exception.code, "revision_rollback")

            gap_payload = json.loads(first_bytes)
            gap_payload["route_revision"] = 3
            manifest.write_text(json.dumps(gap_payload))
            with self.assertRaises(RouteRevisionError) as gap:
                read_route_revision(
                    manifest,
                    last_route_revision=1,
                    last_manifest_sha256=first.manifest_sha256,
                    require_contiguous_revision=True,
                    now_unix_sec=100.0,
                )
            self.assertEqual(gap.exception.code, "revision_gap")


if __name__ == "__main__":
    unittest.main()
