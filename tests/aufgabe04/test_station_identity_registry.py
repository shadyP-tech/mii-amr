import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    candidate_snapshot_sha256,
    new_candidate_snapshot,
)
from scripts.aufgabe04.stations.station_identity_registry import (
    StationIdentity,
    StationIdentityRegistryError,
    candidate_order_for_server_order,
    load_station_identity_registry,
    new_station_identity_registry,
    station_identity_registry_sha256,
    write_station_identity_registry,
)


def _candidate(uid, x_m, observation_id):
    return FrozenCandidate(
        candidate_uid=uid,
        geometry=CandidateGeometry(x_m, 0.0, 0.06, 0.02, 0.26),
        source=CandidateSource(
            "lidar/stand_confirmation",
            "a" * 64,
            "b" * 64,
            (observation_id,),
        ),
        confidence=0.9,
        hit_count=3,
        first_seen_sec=1.0,
        last_seen_sec=2.0,
    )


def _snapshot():
    return new_candidate_snapshot(
        snapshot_id="snapshot_001",
        created_unix_sec=10.0,
        planning_frame="map",
        map_bundle_sha256="c" * 64,
        candidates=(
            _candidate("candidate_a", 0.0, "obs_a"),
            _candidate("candidate_b", 1.0, "obs_b"),
            _candidate("candidate_c", 2.0, "obs_c"),
        ),
    )


def _registry(snapshot=None, mappings=None):
    selected = snapshot or _snapshot()
    return new_station_identity_registry(
        registry_id="identity_registry_001",
        created_unix_sec=20.0,
        candidate_snapshot_sha256=candidate_snapshot_sha256(selected),
        source_artifact_sha256="d" * 64,
        expected_candidate_uids=selected.candidate_uids,
        mappings=mappings
        or (
            StationIdentity("candidate_a", "A", "station_A"),
            StationIdentity("candidate_b", "B", "station_B"),
            StationIdentity("candidate_c", "C", "station_C"),
        ),
    )


class StationIdentityRegistryTest(unittest.TestCase):
    def test_round_trip_is_bound_to_candidate_snapshot(self):
        snapshot = _snapshot()
        registry = _registry(snapshot)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "identity.json"
            written = write_station_identity_registry(path, registry)
            loaded = load_station_identity_registry(
                path, candidate_snapshot=snapshot
            )

        self.assertEqual(loaded, registry)
        self.assertEqual(written, station_identity_registry_sha256(registry))
        self.assertEqual(loaded.for_qr("A").candidate_uid, "candidate_a")
        self.assertEqual(
            loaded.for_server_station("station_B").qr_id, "B"
        )

    def test_server_order_is_preserved_exactly(self):
        order = candidate_order_for_server_order(
            _registry(), ("station_C", "station_A", "station_B", "station_A")
        )
        self.assertEqual(
            order,
            ("candidate_c", "candidate_a", "candidate_b", "candidate_a"),
        )

    def test_unknown_server_station_fails_closed(self):
        with self.assertRaises(StationIdentityRegistryError) as raised:
            candidate_order_for_server_order(_registry(), ("station_X",))
        self.assertEqual(raised.exception.code, "unknown_server_station")

    def test_candidate_qr_and_server_ids_are_each_one_to_one(self):
        base = (
            StationIdentity("candidate_a", "A", "station_A"),
            StationIdentity("candidate_b", "A", "station_B"),
            StationIdentity("candidate_c", "C", "station_C"),
        )
        with self.assertRaises(StationIdentityRegistryError) as qr_conflict:
            _registry(mappings=base)
        self.assertEqual(qr_conflict.exception.code, "identity_conflict")

        duplicate_server = (
            StationIdentity("candidate_a", "A", "station_A"),
            StationIdentity("candidate_b", "B", "station_A"),
            StationIdentity("candidate_c", "C", "station_C"),
        )
        with self.assertRaises(StationIdentityRegistryError) as server_conflict:
            _registry(mappings=duplicate_server)
        self.assertEqual(server_conflict.exception.code, "identity_conflict")

    def test_incomplete_or_wrong_snapshot_registry_is_rejected(self):
        incomplete = (
            StationIdentity("candidate_a", "A", "station_A"),
            StationIdentity("candidate_b", "B", "station_B"),
        )
        with self.assertRaises(StationIdentityRegistryError) as missing:
            _registry(mappings=incomplete)
        self.assertEqual(missing.exception.code, "incomplete_registry")

        snapshot = _snapshot()
        registry = replace(_registry(snapshot), candidate_snapshot_sha256="e" * 64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "identity.json"
            write_station_identity_registry(path, registry)
            with self.assertRaises(StationIdentityRegistryError) as mismatch:
                load_station_identity_registry(path, candidate_snapshot=snapshot)
        self.assertEqual(mismatch.exception.code, "provenance_mismatch")

    def test_immutable_registry_cannot_be_replaced(self):
        registry = _registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "identity.json"
            write_station_identity_registry(path, registry)
            with self.assertRaises(StationIdentityRegistryError) as raised:
                write_station_identity_registry(
                    path, replace(registry, registry_id="identity_registry_002")
                )
        self.assertEqual(raised.exception.code, "immutable_conflict")


if __name__ == "__main__":
    unittest.main()
