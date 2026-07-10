import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.qr_scanning.scan_processor import (  # noqa: E402
    QRScanProcessor,
    ScanProcessorConfig,
)


class QRScanProcessorTest(unittest.TestCase):
    def make_processor(self):
        return QRScanProcessor(
            ScanProcessorConfig(
                robot_id="Robot_Test_01",
                run_id="run-001",
                min_repeat_sec=2.0,
                max_frame_age_sec=1.0,
            )
        )

    def test_accepts_valid_qr_id_and_builds_row(self):
        outcome = self.make_processor().process_texts(
            (" qr_001 ",),
            source="/camera/image_raw/compressed",
            receipt_time_sec=10.0,
            stamp_sec=9.5,
        )[0]

        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.qr_id, "QR_001")
        self.assertEqual(outcome.row["status"], "accepted")
        self.assertEqual(outcome.row["timestamp"], 9.5)
        self.assertEqual(outcome.row["robot_id"], "Robot_Test_01")
        self.assertEqual(outcome.row["run_id"], "run-001")

    def test_rejects_invalid_qr_id(self):
        outcome = self.make_processor().process_texts(
            ("route: A -> B",),
            source="/camera/image_raw/compressed",
            receipt_time_sec=10.0,
        )[0]

        self.assertEqual(outcome.status, "rejected")
        self.assertEqual(outcome.row["status"], "rejected")
        self.assertEqual(outcome.row["qr_id"], "")
        self.assertIn("single station or QR identifier", outcome.reason)

    def test_rejects_stale_frame(self):
        outcome = self.make_processor().process_texts(
            ("QR_001",),
            source="/camera/image_raw/compressed",
            receipt_time_sec=10.0,
            stamp_sec=8.5,
        )[0]

        self.assertEqual(outcome.status, "rejected")
        self.assertEqual(outcome.reason, "stale_frame")
        self.assertEqual(outcome.row["reason"], "stale_frame")

    def test_debounces_repeated_accepted_qr_id(self):
        processor = self.make_processor()
        first = processor.process_texts(
            ("QR_001",),
            source="/camera/image_raw/compressed",
            receipt_time_sec=10.0,
        )[0]
        second = processor.process_texts(
            ("QR_001",),
            source="/camera/image_raw/compressed",
            receipt_time_sec=11.0,
        )[0]

        self.assertEqual(first.status, "accepted")
        self.assertEqual(second.status, "debounced")
        self.assertIsNone(second.row)

    def test_throttles_repeated_rejected_rows(self):
        processor = self.make_processor()
        first = processor.process_texts(
            ("route: A -> B",),
            source="/camera/image_raw/compressed",
            receipt_time_sec=10.0,
        )[0]
        second = processor.process_texts(
            ("route: A -> B",),
            source="/camera/image_raw/compressed",
            receipt_time_sec=11.0,
        )[0]

        self.assertEqual(first.status, "rejected")
        self.assertEqual(second.status, "debounced")
        self.assertIsNone(second.row)

    def test_negative_stamp_age_is_not_rejected_as_stale(self):
        outcome = self.make_processor().process_texts(
            ("QR_001",),
            source="/camera/image_raw/compressed",
            receipt_time_sec=10.0,
            stamp_sec=12.0,
        )[0]

        self.assertEqual(outcome.status, "accepted")


if __name__ == "__main__":
    unittest.main()
