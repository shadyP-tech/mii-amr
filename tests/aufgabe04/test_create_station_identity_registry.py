import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    candidate_snapshot_sha256,
    new_candidate_snapshot,
    write_candidate_snapshot,
)
from scripts.aufgabe04.stations.create_station_identity_registry import main
from scripts.aufgabe04.stations.station_identity_registry import (
    load_station_identity_registry,
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


def _write_snapshot(directory):
    snapshot = new_candidate_snapshot(
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
    path = Path(directory) / "candidate_snapshot.json"
    write_candidate_snapshot(path, snapshot)
    return snapshot, path


class CreateStationIdentityRegistryCliTest(unittest.TestCase):
    def _arguments(self, snapshot_path, mappings, output_path=None):
        arguments = [
            "--candidate-snapshot",
            str(snapshot_path),
            "--created-unix-sec",
            "20",
        ]
        for mapping in mappings:
            arguments.extend(("--mapping", mapping))
        if output_path is not None:
            arguments.extend(("--output-json", str(output_path)))
        return arguments

    def _run_success(self, arguments):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = main(arguments)
        self.assertEqual(status, 0)
        return json.loads(stdout.getvalue())

    def _run_failure(self, arguments):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(arguments)
        self.assertEqual(raised.exception.code, 2)
        return stderr.getvalue()

    def test_writes_complete_registry_to_content_derived_default_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot, snapshot_path = _write_snapshot(tmpdir)
            result = self._run_success(
                self._arguments(
                    snapshot_path,
                    (
                        "candidate_a=A=station_A",
                        "candidate_b=B=station_B",
                        "candidate_c=C=station_C",
                    ),
                )
            )
            output_path = Path(result["output_json"])
            registry = load_station_identity_registry(
                output_path, candidate_snapshot=snapshot
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["mapping_count"], 3)
            self.assertEqual(
                result["candidate_snapshot_sha256"],
                candidate_snapshot_sha256(snapshot),
            )
            self.assertEqual(
                output_path.name,
                "station_identity_registry_"
                f"{result['station_identity_registry_sha256']}.json",
            )
            self.assertEqual(registry.for_candidate("candidate_b").qr_id, "B")
            self.assertEqual(
                registry.for_qr("C").server_station_id, "station_C"
            )

    def test_mapping_argument_order_has_identical_hash_and_retry_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, snapshot_path = _write_snapshot(tmpdir)
            mappings = (
                "candidate_a=A=station_A",
                "candidate_b=B=station_B",
                "candidate_c=C=station_C",
            )
            first = self._run_success(self._arguments(snapshot_path, mappings))
            second = self._run_success(
                self._arguments(snapshot_path, tuple(reversed(mappings)))
            )

            self.assertEqual(first["mapping_source_sha256"], second["mapping_source_sha256"])
            self.assertEqual(
                first["station_identity_registry_sha256"],
                second["station_identity_registry_sha256"],
            )
            self.assertEqual(first["output_json"], second["output_json"])

    def test_rejects_incomplete_unknown_and_non_unique_mappings(self):
        cases = (
            (
                "incomplete",
                (
                    "candidate_a=A=station_A",
                    "candidate_b=B=station_B",
                ),
                "missing=['candidate_c']",
            ),
            (
                "unknown candidate",
                (
                    "candidate_a=A=station_A",
                    "candidate_b=B=station_B",
                    "candidate_x=C=station_C",
                ),
                "unknown=['candidate_x']",
            ),
            (
                "duplicate QR",
                (
                    "candidate_a=A=station_A",
                    "candidate_b=A=station_B",
                    "candidate_c=C=station_C",
                ),
                "duplicate qr_id",
            ),
            (
                "duplicate server station",
                (
                    "candidate_a=A=station_A",
                    "candidate_b=B=station_A",
                    "candidate_c=C=station_C",
                ),
                "duplicate server_station_id",
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            _, snapshot_path = _write_snapshot(tmpdir)
            for label, mappings, expected_error in cases:
                with self.subTest(label):
                    error = self._run_failure(
                        self._arguments(snapshot_path, mappings)
                    )
                    self.assertIn(expected_error, error)

    def test_rejects_malformed_mapping_argument(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, snapshot_path = _write_snapshot(tmpdir)
            error = self._run_failure(
                self._arguments(snapshot_path, ("candidate_a=A",))
            )
        self.assertIn(
            "mapping must be CANDIDATE_UID=QR_ID=SERVER_STATION_ID", error
        )

    def test_explicit_path_remains_immutable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, snapshot_path = _write_snapshot(tmpdir)
            output_path = Path(tmpdir) / "registry.json"
            first_mappings = (
                "candidate_a=A=station_A",
                "candidate_b=B=station_B",
                "candidate_c=C=station_C",
            )
            changed_mappings = (
                "candidate_a=X=station_A",
                "candidate_b=B=station_B",
                "candidate_c=C=station_C",
            )
            self._run_success(
                self._arguments(snapshot_path, first_mappings, output_path)
            )
            error = self._run_failure(
                self._arguments(snapshot_path, changed_mappings, output_path)
            )

            self.assertIn("immutable", error)
            registry = load_station_identity_registry(output_path)
            self.assertEqual(registry.for_candidate("candidate_a").qr_id, "A")


if __name__ == "__main__":
    unittest.main()
