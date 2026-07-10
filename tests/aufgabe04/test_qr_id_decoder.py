import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.qr_scanning.qr_id_decoder import decode_qr_id  # noqa: E402


class QRIdDecoderTest(unittest.TestCase):
    def test_decodes_single_qr_id(self):
        scanned = decode_qr_id(" qr_001 ")

        self.assertEqual(scanned.qr_id, "QR_001")
        self.assertEqual(scanned.raw_text, " qr_001 ")

    def test_rejects_route_payload(self):
        with self.assertRaisesRegex(ValueError, "single station or QR identifier"):
            decode_qr_id("route: A -> B")

    def test_rejects_empty_payload(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            decode_qr_id(" ")


if __name__ == "__main__":
    unittest.main()

