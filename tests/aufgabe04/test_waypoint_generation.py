import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.planning.waypoint_generation import station_visits_to_waypoint_rows  # noqa: E402
from scripts.aufgabe04.stations.station_map import DEFAULT_STATIONS  # noqa: E402
from scripts.aufgabe04.stations.station_router import build_station_visits  # noqa: E402


class WaypointGenerationTest(unittest.TestCase):
    def test_generates_one_waypoint_row_per_visit(self):
        visits = build_station_visits(["A", "B"], DEFAULT_STATIONS)

        rows = station_visits_to_waypoint_rows(visits)

        self.assertEqual([row[3] for row in rows], ["A", "B"])


if __name__ == "__main__":
    unittest.main()

