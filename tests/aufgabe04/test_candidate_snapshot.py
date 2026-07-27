import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSnapshotError,
    CandidateSource,
    FrozenCandidate,
    candidate_geometry_sha256,
    candidate_snapshot_payload,
    candidate_snapshot_sha256,
    candidate_source_sha256,
    load_candidate_snapshot,
    new_candidate_snapshot,
    write_candidate_snapshot,
)


MAP_HASH = "a" * 64


def _candidate(uid="candidate_a", x_m=1.0, observation_id="obs_001"):
    return FrozenCandidate(
        candidate_uid=uid,
        geometry=CandidateGeometry(
            x_m=x_m,
            y_m=2.0,
            radius_m=0.06,
            uncertainty_m=0.02,
            keepout_radius_m=0.26,
        ),
        source=CandidateSource(
            source_kind="lidar/stand_confirmation",
            source_artifact_sha256="b" * 64,
            detector_config_sha256="c" * 64,
            observation_ids=(observation_id,),
        ),
        confidence=0.91,
        hit_count=4,
        first_seen_sec=10.0,
        last_seen_sec=12.0,
    )


def _snapshot(*candidates):
    return new_candidate_snapshot(
        snapshot_id="candidate_snapshot_001",
        created_unix_sec=100.0,
        planning_frame="map",
        map_bundle_sha256=MAP_HASH,
        candidates=candidates or (_candidate(),),
    )


class CandidateSnapshotTest(unittest.TestCase):
    def test_round_trip_has_root_geometry_and_source_hashes(self):
        snapshot = _snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "candidates.json"
            written = write_candidate_snapshot(path, snapshot)
            loaded = load_candidate_snapshot(
                path, required_map_bundle_sha256=MAP_HASH
            )
            payload = json.loads(path.read_text())

        self.assertEqual(loaded, snapshot)
        self.assertEqual(written, candidate_snapshot_sha256(snapshot))
        self.assertEqual(payload["candidate_snapshot_sha256"], written)
        self.assertEqual(
            payload["candidates"][0]["geometry"]["geometry_sha256"],
            candidate_geometry_sha256(snapshot.candidates[0].geometry),
        )
        self.assertEqual(
            payload["candidates"][0]["source"]["source_sha256"],
            candidate_source_sha256(snapshot.candidates[0].source),
        )

    def test_geometry_and_source_changes_invalidate_independent_hashes(self):
        candidate = _candidate()
        moved = replace(
            candidate.geometry, x_m=candidate.geometry.x_m + 0.01
        )
        new_source = replace(candidate.source, source_artifact_sha256="d" * 64)

        self.assertNotEqual(
            candidate_geometry_sha256(candidate.geometry),
            candidate_geometry_sha256(moved),
        )
        self.assertNotEqual(
            candidate_source_sha256(candidate.source),
            candidate_source_sha256(new_source),
        )

    def test_candidates_are_canonicalized_but_observation_ancestry_is_exclusive(self):
        snapshot = _snapshot(
            _candidate("candidate_b", 2.0, "obs_002"),
            _candidate("candidate_a", 1.0, "obs_001"),
        )
        self.assertEqual(snapshot.candidate_uids, ("candidate_a", "candidate_b"))
        with self.assertRaises(CandidateSnapshotError) as raised:
            _snapshot(
                _candidate("candidate_a", 1.0, "same_obs"),
                _candidate("candidate_b", 2.0, "same_obs"),
            )
        self.assertEqual(raised.exception.code, "observation_identity_conflict")

    def test_map_provenance_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.json"
            write_candidate_snapshot(path, _snapshot())
            with self.assertRaises(CandidateSnapshotError) as raised:
                load_candidate_snapshot(
                    path, required_map_bundle_sha256="d" * 64
                )
        self.assertEqual(raised.exception.code, "provenance_mismatch")

    def test_nested_hash_detects_rehashed_root_tampering(self):
        payload = candidate_snapshot_payload(_snapshot())
        del payload["candidate_snapshot_sha256"]
        payload["candidates"][0]["geometry"]["x_m"] = 99.0
        payload["candidate_snapshot_sha256"] = payload_sha256(payload)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(CandidateSnapshotError) as raised:
                load_candidate_snapshot(path)
        self.assertEqual(raised.exception.code, "hash_mismatch")

    def test_immutable_path_allows_idempotent_retry_only(self):
        snapshot = _snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.json"
            first = write_candidate_snapshot(path, snapshot)
            retry = write_candidate_snapshot(path, snapshot)
            with self.assertRaises(CandidateSnapshotError) as raised:
                write_candidate_snapshot(
                    path, replace(snapshot, snapshot_id="candidate_snapshot_002")
                )
        self.assertEqual(first, retry)
        self.assertEqual(raised.exception.code, "immutable_conflict")

    def test_nonfinite_or_undersized_keepout_is_rejected(self):
        with self.assertRaises(CandidateSnapshotError):
            _snapshot(
                replace(
                    _candidate(),
                    geometry=replace(_candidate().geometry, keepout_radius_m=0.01),
                )
            )
        with self.assertRaises(CandidateSnapshotError):
            _snapshot(replace(_candidate(), confidence=float("nan")))


if __name__ == "__main__":
    unittest.main()
