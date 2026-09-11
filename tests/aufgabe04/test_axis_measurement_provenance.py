"""Axis receipts survive offline handoffs without granting measurement authority."""

from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    load_recommendation,
    recommendation_axis_estimator,
    recommendation_to_dict,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.missions.plan_synchronized_viewpoint import main as plan_viewpoint
from scripts.aufgabe04.navigation.planning.map_io import freeze_map_bundle
from scripts.aufgabe04.stations.arrival_pose_catalog import load_arrival_pose_catalog
from scripts.aufgabe04.stations.autonomous_arrival_catalog import (
    _project_recommendation,
    promote_autonomous_arrival_catalog,
)
from tests.aufgabe04.test_autonomous_arrival_catalog import fixture
from tests.aufgabe04.test_viewpoint_recommendation import recommendation


def measurement(stamp=12.5):
    return {
        "source": "model_current_measured_head",
        "model_profile_sha256": "a" * 64,
        "model_measurement_status": "measured",
        "head_model_quality": {
            "accepted": True,
            "face_semantics": "undirected_plane",
            "head_size_m": [0.078, 0.078],
            "yaw_std_deg": 1.463,
        },
        "sample_admission": {"accepted": True, "reason": "quality_fixture"},
        "sensor_stamp_sec": stamp,
    }


class AxisMeasurementProvenanceTests(unittest.TestCase):
    def test_legacy_real_artifact_retains_unknown_source_and_original_shape(self):
        original = replace(recommendation(), simulation_only=False)
        payload = recommendation_to_dict(original)
        self.assertNotIn("axis_measurement", payload)
        loaded = load_recommendation(payload)
        self.assertIsNone(loaded.axis_measurement)
        self.assertEqual(recommendation_to_dict(loaded), payload)
        self.assertEqual(recommendation_axis_estimator(loaded), "real/legacy_axis_source_unrecorded")
        self.assertEqual(
            recommendation_axis_estimator(recommendation()),
            "simulation/silhouette_head_rectangle",
        )

    def test_current_receipt_round_trip_copies_all_nested_diagnostics(self):
        payload = recommendation_to_dict(replace(recommendation(), simulation_only=False))
        payload["axis_measurement"] = measurement()
        expected = deepcopy(payload["axis_measurement"])
        loaded = load_recommendation(payload)
        self.assertEqual(loaded.axis_measurement, expected)
        self.assertEqual(recommendation_axis_estimator(loaded), "real/model_current_measured_head")
        payload["axis_measurement"]["head_model_quality"]["head_size_m"][0] = 999.0
        serialized = recommendation_to_dict(loaded)
        self.assertEqual(serialized["axis_measurement"], expected)
        serialized["axis_measurement"]["sample_admission"]["accepted"] = False
        self.assertEqual(loaded.axis_measurement, expected)

    def test_labels_use_explicit_source_without_diagnostic_inference_or_admission(self):
        for source, label in (
            ("model_metric_joint", "real/model_metric_joint"),
            (None, "real/legacy_axis_source_unrecorded"),
            ("", "real/legacy_axis_source_unrecorded"),
            ("unsafe label", "real/legacy_axis_source_unrecorded"),
        ):
            with self.subTest(source=source):
                payload = recommendation_to_dict(replace(recommendation(), simulation_only=False))
                payload["axis_measurement"] = {**measurement(), "source": source}
                # This is audit metadata, not a second measurement-admission gate.
                payload["axis_measurement"]["sample_admission"]["accepted"] = False
                loaded = load_recommendation(payload)
                self.assertEqual(recommendation_axis_estimator(loaded), label)
                self.assertEqual(loaded.axis_state, "resolved")

    def test_nonobject_receipt_is_rejected_as_malformed_structure(self):
        payload = recommendation_to_dict(recommendation())
        payload["axis_measurement"] = "model_current_measured_head"
        with self.assertRaisesRegex(ValueError, "axis_measurement must be an object"):
            load_recommendation(payload)

    def test_frame_projection_preserves_original_sensor_receipt(self):
        original = replace(recommendation(), axis_measurement=measurement())
        source = CandidatePlanningFrame(Pose2D(1.0, 0.0), PlanarTransform2D(0.0, 0.0, 0.0))
        target = CandidatePlanningFrame(Pose2D(1.1, 0.0), PlanarTransform2D(0.1, 0.0, 0.2))
        projected = _project_recommendation(original, source, target)
        self.assertNotEqual(projected.stand.center, original.stand.center)
        self.assertEqual(projected.axis_measurement, original.axis_measurement)
        self.assertEqual(projected.sensor_stamp_sec, original.sensor_stamp_sec)

    def test_both_real_catalog_paths_report_actual_source_and_load_legacy(self):
        for include_receipt in (False, True):
            with self.subTest(include_receipt=include_receipt), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs, now = fixture(root)
                rec_path = root / "recommendation.json"
                payload = json.loads(rec_path.read_text())
                if include_receipt:
                    payload["axis_measurement"] = measurement(payload["sensor_stamp_sec"])
                    rec_path.write_text(json.dumps(payload))
                    facing = load_content_hashed_json(inputs.facing_catalog, hash_field="stand_facing_catalog_sha256")
                    facing["records"][0]["camera_recommendation_sha256"] = hashlib.sha256(rec_path.read_bytes()).hexdigest()
                    inputs = replace(inputs, facing_catalog=root / "facing_with_axis_measurement.json")
                    write_content_hashed_json(inputs.facing_catalog, facing, hash_field="stand_facing_catalog_sha256")
                expected = "real/model_current_measured_head" if include_receipt else "real/legacy_axis_source_unrecorded"
                promote_autonomous_arrival_catalog(inputs, now_sec=now)
                autonomous = load_arrival_pose_catalog(inputs.output_dir / "arrival_pose_catalog.json")
                self.assertEqual(autonomous.records[0].axis.estimator, expected)
                self.assertTrue(autonomous.records[0].face.evidence_hard)

                # Exercise the independent synchronized-viewpoint promotion path.
                catalog_path = root / "synchronized_catalog.json"
                bundle = freeze_map_bundle(inputs.map_yaml, semantic_map_id="arena", planning_frame="map")
                with redirect_stdout(io.StringIO()), patch("scripts.aufgabe04.navigation.missions.plan_synchronized_viewpoint.time.time", return_value=now):
                    status = plan_viewpoint([
                        "--environment", "real", "--workflow-mode", "survey-only",
                        "--start-from-recommendation", "--start-x", "2", "--start-y", "3.4",
                        "--map", str(inputs.map_yaml), "--map-frame", "map", "--semantic-map-id", "arena",
                        "--recommended-pose-json", str(rec_path), "--stream-id", payload["stream_id"],
                        "--route-csv", str(root / "route.csv"), "--diagnostics-json", str(root / "diagnostics.json"),
                        "--arrival-pose-catalog", str(catalog_path), "--candidate-uid", payload["stand_id"],
                        "--expected-candidate-uid", payload["stand_id"], "--world-id", "arena", "--world-sha256", "b" * 64,
                        "--session-id", "provenance_test", "--candidate-snapshot-sha256", "c" * 64,
                        "--expected-map-bundle-sha256", bundle.bundle_sha256, "--max-recommendation-age-sec", "300",
                        "--arena-length-m", "6", "--arena-width-m", "6", "--arena-center-x-m", "3", "--arena-center-y-m", "3",
                    ])
                self.assertEqual(status, 0)
                synchronized = load_arrival_pose_catalog(catalog_path)
                self.assertEqual(synchronized.records[0].axis.estimator, expected)
                self.assertTrue(synchronized.records[0].face.evidence_hard)


if __name__ == "__main__":
    unittest.main()
