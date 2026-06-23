import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.stations.models import Station, StationPose  # noqa: E402
from scripts.aufgabe04.stations.station_map import build_station_map, get_station  # noqa: E402


class StationMapTest(unittest.TestCase):
    def test_build_station_map_normalizes_ids(self):
        station_map = build_station_map([Station(" a ", StationPose(1.0, 2.0))])

        self.assertEqual(get_station(station_map, "A").station_id, "A")


if __name__ == "__main__":
    unittest.main()

