"""Catalog promotion preserves immediate front evidence and its limits."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json, write_content_hashed_json,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    load_recommendation, recommendation_uses_current_head_front,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.stations.arrival_pose_catalog import load_arrival_pose_catalog
from scripts.aufgabe04.stations.autonomous_arrival_catalog import (
    _project_recommendation, promote_autonomous_arrival_catalog,
)
from tests.aufgabe04.current_head_front_fixture import current_head_front_evidence
from tests.aufgabe04.test_autonomous_arrival_catalog import fixture
from tests.aufgabe04.test_head_model_admission import quality


def current_head_payload(payload):
    """Create synthetic validated evidence; no recording is certified here."""
    payload = deepcopy(payload)
    payload["axis_measurement"] = current_head_front_evidence(
        stamp=payload["sensor_stamp_sec"], qr_id="qr_Mixed_a", stand_axis_rad=0.,
        target_key=f"{payload['stream_id']}:{payload['stand_id']}:2.0:2.0",
        head_model_quality=quality(centered_neck_supported=False, neck_junction_verified=False),
    )
    payload["schema_version"] = 3
    payload["axis"]["sample_count"] = 1
    payload["axis"]["confidence"] = 0.
    payload["side_evidence"].update(
        kind="qr_observation", provenance="real/onboard_camera_qr_observation",
    )
    return payload


def replace_recommendation(inputs, payload):
    """Reseal the real source chain so tests reach recommendation admission."""
    path = inputs.facing_catalog.parent / "recommendation.json"
    path.write_text(json.dumps(payload))
    facing = load_content_hashed_json(
        inputs.facing_catalog, hash_field="stand_facing_catalog_sha256",
    )
    facing["records"][0]["camera_recommendation_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    inputs.facing_catalog.unlink()
    write_content_hashed_json(
        inputs.facing_catalog, facing, hash_field="stand_facing_catalog_sha256",
    )


class CurrentHeadCatalogTests(unittest.TestCase):
    def test_validated_single_fit_promotes_and_preserves_real_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, now = fixture(root)
            payload = current_head_payload(json.loads((root / "recommendation.json").read_text()))
            replace_recommendation(inputs, payload)

            result = promote_autonomous_arrival_catalog(inputs, now_sec=now)

            catalog = load_arrival_pose_catalog(inputs.output_dir / "arrival_pose_catalog.json")
            self.assertTrue(catalog.frozen)
            self.assertEqual(result["stand_count"], 1)
            self.assertEqual(result["obstacle_count"], 2)
            self.assertEqual(catalog.records[0].axis.sample_count, 1)
            self.assertEqual(catalog.records[0].face.evidence_kind, "qr_observation")
            self.assertEqual(catalog.records[0].face.evidence_provenance,
                             "real/onboard_camera_qr_observation")
            config_path = next(inputs.output_dir.glob("survey_config_*.json"))
            config = load_content_hashed_json(config_path, hash_field="survey_config_sha256")
            self.assertNotIn("axis_sample_count", config)
            self.assertEqual(config["axis_sample_counts_by_candidate"],
                             {"survey_candidate_0003": 1})

    def test_projection_rotates_receipt_angles_and_preserves_sensor_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            payload = current_head_payload(json.loads((root / "recommendation.json").read_text()))
            original = load_recommendation(payload)
            source = CandidatePlanningFrame(Pose2D(2., 3.4), PlanarTransform2D(0., 0., 0.))
            target = CandidatePlanningFrame(Pose2D(2.1, 3.4), PlanarTransform2D(.1, 0., .2))

            projected = _project_recommendation(original, source, target)

            self.assertTrue(recommendation_uses_current_head_front(projected))
            self.assertAlmostEqual(projected.axis_measurement["stand_axis_rad"], .2)
            self.assertAlmostEqual(projected.axis_measurement["camera_heading_rad"],
                                   original.axis_measurement["camera_heading_rad"] + .2)
            expected = deepcopy(original.axis_measurement)
            expected.update(stand_axis_rad=projected.axis_measurement["stand_axis_rad"],
                            camera_heading_rad=projected.axis_measurement["camera_heading_rad"])
            self.assertEqual(projected.axis_measurement, expected)
            self.assertEqual(original.axis_measurement, payload["axis_measurement"])

    def test_validated_receipt_cannot_relabel_observed_station_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, now = fixture(root)
            payload = current_head_payload(json.loads((root / "recommendation.json").read_text()))
            payload["axis_measurement"]["qr_id"] = "different_qr"
            payload["axis_measurement"]["qr_binding"]["qr_texts_for_evidence"] = ["different_qr"]
            replace_recommendation(inputs, payload)

            with self.assertRaisesRegex(ValueError, "QR identity differs"):
                promote_autonomous_arrival_catalog(inputs, now_sec=now)
            self.assertFalse(inputs.output_dir.exists())

    def test_single_fit_still_requires_collision_free_target_and_corridor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, now = fixture(root, obstacle=(2.0, 2.8))
            payload = current_head_payload(json.loads((root / "recommendation.json").read_text()))
            replace_recommendation(inputs, payload)

            with self.assertRaisesRegex(ValueError, "fixed target/corridor"):
                promote_autonomous_arrival_catalog(inputs, now_sec=now)
            self.assertFalse(inputs.output_dir.exists())

    def test_legacy_single_sample_does_not_gain_immediate_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, now = fixture(root)
            payload = json.loads((root / "recommendation.json").read_text())
            payload["axis"]["sample_count"] = 1
            replace_recommendation(inputs, payload)

            with self.assertRaisesRegex(ValueError, "committed onboard QR"):
                promote_autonomous_arrival_catalog(inputs, now_sec=now)
            self.assertFalse(inputs.output_dir.exists())

    def test_immediate_labels_on_legacy_schema_are_not_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, now = fixture(root)
            payload = json.loads((root / "recommendation.json").read_text())
            payload["side_evidence"].update(
                kind="qr_observation", provenance="real/onboard_camera_qr_observation",
            )
            replace_recommendation(inputs, payload)

            with self.assertRaises(ValueError):
                promote_autonomous_arrival_catalog(inputs, now_sec=now)
            self.assertFalse(inputs.output_dir.exists())

    def test_schema3_without_validated_current_head_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, now = fixture(root)
            payload = json.loads((root / "recommendation.json").read_text())
            payload["schema_version"] = 3
            payload["axis"]["sample_count"] = 1
            payload["side_evidence"].update(
                kind="qr_observation", provenance="real/onboard_camera_qr_observation",
            )
            replace_recommendation(inputs, payload)

            with self.assertRaises(ValueError):
                promote_autonomous_arrival_catalog(inputs, now_sec=now)
            self.assertFalse(inputs.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
