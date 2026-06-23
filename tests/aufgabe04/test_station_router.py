import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.stations.station_map import DEFAULT_STATIONS  # noqa: E402
from scripts.aufgabe04.stations.station_router import build_station_visits  # noqa: E402


class StationRouterTest(unittest.TestCase):
    def test_builds_visit_for_each_station_id(self):
        visits = build_station_visits(["A", "B"], DEFAULT_STATIONS)

        self.assertEqual([visit.station_id for visit in visits], ["A", "B"])


if __name__ == "__main__":
    unittest.main()

