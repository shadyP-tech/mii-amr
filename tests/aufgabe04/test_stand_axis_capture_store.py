import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover
    cv2 = None
    numpy = None


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.debug.stand_axis_capture_store import (  # noqa: E402
    save_structural_capture,
    sensor_frame_status,
)


class StandAxisCaptureStoreTest(unittest.TestCase):
    def test_sensor_status_distinguishes_missing_future_stale_and_unverified(self):
        self.assertEqual(
            sensor_frame_status(
                source_stamp_sec=None,
                received_wall_sec=10.0,
                max_frame_age_sec=0.25,
            ),
            "no_header",
        )
        self.assertEqual(
            sensor_frame_status(
                source_stamp_sec=11.0,
                received_wall_sec=10.0,
                max_frame_age_sec=0.25,
            ),
            "header_future",
        )
        self.assertEqual(
            sensor_frame_status(
                source_stamp_sec=9.0,
                received_wall_sec=10.0,
                max_frame_age_sec=0.25,
            ),
            "header_stale",
        )
        self.assertEqual(
            sensor_frame_status(
                source_stamp_sec=9.99,
                received_wall_sec=10.0,
                max_frame_age_sec=0.0,
            ),
            "clock_unverified",
        )

    def test_sensor_status_accepts_verified_header_age(self):
        self.assertEqual(
            sensor_frame_status(
                source_stamp_sec=9.9,
                received_wall_sec=10.0,
                max_frame_age_sec=0.25,
            ),
            "header_age_ok",
        )

    @unittest.skipIf(cv2 is None or numpy is None, "OpenCV and numpy are required")
    def test_structural_capture_separates_raw_artifacts_and_marks_observe_only(self):
        image = numpy.full((8, 10, 3), 17, dtype=numpy.uint8)
        mask = numpy.full((8, 10), 255, dtype=numpy.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = save_structural_capture(
                cv2,
                Path(tmpdir),
                original_compressed=b"exact-compressed-payload",
                compressed_format="jpeg",
                decoded_frame=image,
                candidate_roi_frame=image,
                raw_edges=mask,
                localization_edges=mask,
                side_evidence=mask,
                rectangle_mask=mask,
                annotated_frame=image,
                metadata={
                    "measurement_status": "fresh",
                    "sensor_status": "clock_unverified",
                },
            )

            record = json.loads(metadata_path.read_text())
            self.assertTrue(record["observe_only"])
            self.assertFalse(record["authoritative"])
            self.assertEqual(record["measurement_status"], "fresh")
            self.assertEqual(
                (Path(tmpdir) / record["files"]["compressed"]).read_bytes(),
                b"exact-compressed-payload",
            )
            for key in (
                "decoded_frame",
                "candidate_roi",
                "raw_edges",
                "localization_edges",
                "side_evidence",
                "rectangle",
                "annotated",
            ):
                self.assertTrue((Path(tmpdir) / record["files"][key]).is_file())


if __name__ == "__main__":
    unittest.main()
