"""A successful QR discovery is not automatically a logistics arrival pose."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, write_content_hashed_json
from scripts.aufgabe04.stations.autonomous_arrival_catalog import promote_autonomous_arrival_catalog
from tests.aufgabe04.test_autonomous_arrival_catalog import fixture


class AutonomousCatalogQrObservationBoundaryTests(unittest.TestCase):
    def test_partial_or_qr_only_evidence_cannot_be_promoted(self):
        mutations = (
            lambda value: value.update(facing_complete=False),
            lambda value: value.update(facing_complete=True, qr_only_stand_count=1),
            lambda value: value.update(qr_only_candidate_uids=["survey_candidate_0003"]),
            lambda value: value["records"][0].update(evidence_kind="qr_verified_observation_pose"),
            lambda value: value["records"][0].update(facing_ready=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            inputs, now = fixture(Path(directory))
            original = load_content_hashed_json(inputs.facing_catalog, hash_field="stand_facing_catalog_sha256")
            for index, mutate in enumerate(mutations):
                with self.subTest(index=index):
                    payload = deepcopy(original)
                    mutate(payload)
                    changed_inputs = replace(inputs, facing_catalog=Path(directory) / f"changed_{index}.json")
                    write_content_hashed_json(changed_inputs.facing_catalog, payload, hash_field="stand_facing_catalog_sha256")
                    with self.assertRaisesRegex(ValueError, "QR|facing"):
                        promote_autonomous_arrival_catalog(changed_inputs, now_sec=now)
                    self.assertFalse(inputs.output_dir.exists())

    def test_explicit_complete_geometry_catalog_remains_promotable(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, now = fixture(Path(directory))
            payload = load_content_hashed_json(inputs.facing_catalog, hash_field="stand_facing_catalog_sha256")
            payload.update(facing_complete=True, qr_only_stand_count=0, qr_only_candidate_uids=[])
            inputs = replace(inputs, facing_catalog=Path(directory) / "complete_facing.json")
            write_content_hashed_json(inputs.facing_catalog, payload, hash_field="stand_facing_catalog_sha256")
            result = promote_autonomous_arrival_catalog(inputs, now_sec=now)
            self.assertEqual(result["stand_count"], 1)


if __name__ == "__main__":
    unittest.main()
