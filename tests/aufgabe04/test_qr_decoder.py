import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.qr_scanning.qr_decoder import parse_station_payload  # noqa: E402


class QRDecoderTest(unittest.TestCase):
    def test_parses_route_payload(self):
        detection = parse_station_payload("route: A -> B -> C", known_stations=["A", "B", "C"])

        self.assertEqual(detection.station_ids, ("A", "B", "C"))

    def test_rejects_unknown_station(self):
        with self.assertRaisesRegex(ValueError, "unknown station"):
            parse_station_payload("A, Z", known_stations=["A"])


if __name__ == "__main__":
    unittest.main()

