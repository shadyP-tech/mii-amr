import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    append_lidar_visibility_receipts,
    lidar_visibility_receipt_from_scan,
    load_lidar_visibility_receipt_snapshot,
    load_lidar_visibility_receipts,
    sanitized_scan_ranges,
    visibility_receipts_sha256,
)


MAP_SHA256 = "a" * 64
CONFIG_SHA256 = "b" * 64


def _receipt(
    receipt_id: str,
    *,
    stamp: float = 1.0,
    ranges=(1.0, 2.0, 3.0),
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
    )


class LidarVisibilityEvidenceTest(unittest.TestCase):
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
