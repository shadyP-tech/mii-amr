import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.qr_scanning.station_order import station_order_from_payload  # noqa: E402


class StationOrderTest(unittest.TestCase):
    def test_station_order_preserves_qr_order(self):
        order = station_order_from_payload("stations: C;A;B", known_stations=["A", "B", "C"])

        self.assertEqual(order.station_ids, ("C", "A", "B"))


if __name__ == "__main__":
    unittest.main()

