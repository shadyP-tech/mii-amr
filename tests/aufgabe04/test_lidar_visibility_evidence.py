import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.perception.lidar_visibility_frames import LidarVisibilityFrameProvenance
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    append_lidar_visibility_receipts,
    lidar_visibility_receipt_from_scan,
    load_lidar_visibility_receipt_snapshot,
    load_lidar_visibility_receipts,
    sanitized_scan_ranges,
    visibility_receipts_sha256,
    validate_lidar_visibility_receipt,
)


MAP_SHA256 = "a" * 64
CONFIG_SHA256 = "b" * 64


def _receipt(
    receipt_id: str,
    *,
    stamp: float = 1.0,
    ranges=(1.0, 2.0, 3.0),
    frame_provenance=None,
):
    return lidar_visibility_receipt_from_scan(
        receipt_id=receipt_id,
        survey_id="survey_01",
        viewpoint_id="viewpoint_02",
        planning_frame="map",
        scan_frame="base_scan",
        scan_topic="/scan",
        map_bundle_sha256=MAP_SHA256,
        observer_config_sha256=CONFIG_SHA256,
        scan_stamp_sec=stamp,
        pose_stamp_sec=stamp,
        observer_clock_sec=stamp + 0.01,
        scan_pose_map=Pose2D(0.05, 0.05, 0.0),
        angle_min_rad=-1.0,
        angle_increment_rad=1.0,
        range_min_m=0.08,
        range_max_m=3.5,
        ranges_m=ranges,
        frame_provenance=frame_provenance,
    )


class LidarVisibilityEvidenceTest(unittest.TestCase):
    def _provenance(self, *, source_evidence_id="c" * 64):
        return LidarVisibilityFrameProvenance(
            map_frame="map",
            odom_frame="odom",
            map_from_odom=PlanarTransform2D(1.05, 2.05, math.pi / 2.0),
            canonical_scan_pose_odom=Pose2D(-2.0, 1.0, -math.pi / 2.0),
            source_evidence_id=source_evidence_id,
        )

    def test_schema_two_binds_nonzero_yaw_transform_and_certificate(self):
        receipt = _receipt("receipt_01", frame_provenance=self._provenance())
        self.assertEqual(receipt.schema_version, 2)
        self.assertIsNotNone(receipt.to_evidence_dict()["frame_provenance"])
        changed_identity = replace(
            receipt, frame_provenance=self._provenance(source_evidence_id="d" * 64)
        )
        self.assertNotEqual(receipt.receipt_sha256, changed_identity.receipt_sha256)
        changed_transform = replace(receipt.frame_provenance,
            map_from_odom=PlanarTransform2D(1.05, 2.15, math.pi / 2.0),
            canonical_scan_pose_odom=Pose2D(-2.1, 1.0, -math.pi / 2.0),
        )
        self.assertNotEqual(
            receipt.receipt_sha256,
            replace(receipt, frame_provenance=changed_transform).receipt_sha256,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility.jsonl"
            append_lidar_visibility_receipts(path, (receipt,))
            loaded = load_lidar_visibility_receipts(path)
        self.assertEqual(loaded, (receipt,))
        self.assertEqual(loaded[0].receipt_sha256, receipt.receipt_sha256)

    def test_schema_one_loads_with_original_hash_and_no_provenance(self):
        historical = replace(_receipt("receipt_legacy"), schema_version=1)
        payload = historical.to_evidence_dict()
        self.assertNotIn("frame_provenance", payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility.jsonl"
            path.write_text(json.dumps(payload) + "\n")
            loaded = load_lidar_visibility_receipts(path)[0]
        self.assertIsNone(loaded.frame_provenance)
        self.assertEqual(loaded.receipt_sha256, payload["receipt_sha256"])
        with self.assertRaisesRegex(ValueError, "legacy"):
            validate_lidar_visibility_receipt(replace(
                historical, frame_provenance=self._provenance()
            ))

    def test_inconsistent_pose_or_frame_cannot_claim_provenance(self):
        receipt = _receipt("receipt_01", frame_provenance=self._provenance())
        for pose in (Pose2D(0.06, 0.05, 0.0), Pose2D(0.05, 0.05, 0.01)):
            with self.subTest(pose=pose):
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    validate_lidar_visibility_receipt(replace(receipt, scan_pose_map=pose))
        with self.assertRaisesRegex(ValueError, "planning frame mismatch"):
            validate_lidar_visibility_receipt(replace(receipt, planning_frame="other_map"))
        # Equivalent wrapped yaw remains the same pose.
        validate_lidar_visibility_receipt(replace(
            receipt, scan_pose_map=Pose2D(0.05, 0.05, math.tau)
        ))

    def test_frame_provenance_hash_and_strict_nested_schema_are_checked(self):
        receipt = _receipt("receipt_01", frame_provenance=self._provenance())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility.jsonl"
            payload = receipt.to_evidence_dict()
            payload["frame_provenance"]["source_evidence_id"] = "d" * 64
            path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_lidar_visibility_receipts(path)
            for mutation in ("bool_pose", "extra_field", "missing_field", "bad_hash"):
                payload = receipt.to_evidence_dict()
                provenance = payload["frame_provenance"]
                if mutation == "bool_pose":
                    provenance["canonical_scan_pose_odom"]["x_m"] = True
                elif mutation == "extra_field":
                    provenance["extra"] = 1
                elif mutation == "missing_field":
                    del provenance["odom_frame"]
                else:
                    provenance["source_evidence_id"] = "not-a-certificate-hash"
                del payload["receipt_sha256"]
                payload["receipt_sha256"] = payload_sha256(payload)
                path.write_text(json.dumps(payload) + "\n")
                with self.subTest(mutation=mutation):
                    with self.assertRaises(ValueError):
                        load_lidar_visibility_receipts(path)
            line = json.dumps(receipt.to_evidence_dict())
            line = line.replace('"odom_frame": "odom"', '"odom_frame": "odom", "odom_frame": "odom"')
            path.write_text(line + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                load_lidar_visibility_receipts(path)

    def test_nonfinite_and_sensor_invalid_ranges_become_null(self):
        sanitized = sanitized_scan_ranges(
            (0.08, 1.25, math.inf, math.nan, -1.0, 3.6, True),
            range_min_m=0.08,
            range_max_m=3.5,
        )

        self.assertEqual(
            sanitized,
            (0.08, 1.25, None, None, None, None, None),
        )
        receipt = _receipt("receipt_01", ranges=sanitized)
        payload = receipt.to_evidence_dict()
        self.assertEqual(payload["ranges_m"], list(sanitized))
        self.assertEqual(receipt.finite_range_count, 2)
        json.dumps(payload, allow_nan=False)

    def test_exact_time_pose_is_required(self):
        with self.assertRaisesRegex(ValueError, "exact scan timestamp"):
            lidar_visibility_receipt_from_scan(
                receipt_id="receipt_01",
                survey_id="survey_01",
                viewpoint_id="viewpoint_02",
                planning_frame="map",
                scan_frame="base_scan",
                scan_topic="/scan",
                map_bundle_sha256=MAP_SHA256,
                observer_config_sha256=CONFIG_SHA256,
                scan_stamp_sec=1.0,
                pose_stamp_sec=1.01,
                observer_clock_sec=1.02,
                scan_pose_map=Pose2D(0.0, 0.0, 0.0),
                angle_min_rad=-math.pi,
                angle_increment_rad=math.radians(1.0),
                range_min_m=0.08,
                range_max_m=3.5,
                ranges_m=(1.0,),
            )

    def test_boolean_or_nonfinite_pose_and_boolean_stamp_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "numeric"):
            _receipt("receipt_01", stamp=True)
        for value in (True, math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    replace(self._provenance(),
                        canonical_scan_pose_odom=Pose2D(value, 1.0, 0.0)
                    )
        with self.assertRaisesRegex(ValueError, "schema_version"):
            validate_lidar_visibility_receipt(replace(
                _receipt("receipt_01"), schema_version=True
            ))

    def test_compact_append_load_and_snapshot_hash_round_trip(self):
        first = _receipt("receipt_01", stamp=1.0)
        second = _receipt("receipt_02", stamp=1.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility.jsonl"

            append_lidar_visibility_receipts(path, (first,))
            append_lidar_visibility_receipts(path, (second,))
            loaded, raw_sha256 = load_lidar_visibility_receipt_snapshot(path)

            self.assertEqual(loaded, (first, second))
            self.assertEqual(len(raw_sha256), 64)
            self.assertEqual(
                visibility_receipts_sha256(loaded),
                visibility_receipts_sha256((first, second)),
            )
            self.assertNotIn(" ", path.read_text().splitlines()[0])
            with self.assertRaisesRegex(ValueError, "already exists"):
                append_lidar_visibility_receipts(path, (second,))

    def test_tampered_hashed_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility.jsonl"
            append_lidar_visibility_receipts(path, (_receipt("receipt_01"),))
            payload = json.loads(path.read_text())
            payload["ranges_m"][1] = 3.25
            path.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_lidar_visibility_receipts(path)

    def test_duplicate_ids_and_nonfinite_json_fail_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility.jsonl"
            receipt = _receipt("receipt_01")
            line = json.dumps(receipt.to_evidence_dict(), separators=(",", ":"))
            path.write_text(line + "\n" + line + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate receipt_id"):
                load_lidar_visibility_receipts(path)

            path.write_text(line.replace("1.0", "NaN", 1) + "\n")
            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                load_lidar_visibility_receipts(path)


if __name__ == "__main__":
    unittest.main()
