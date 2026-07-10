import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.stations.models import Station, StationPose  # noqa: E402
from scripts.aufgabe04.stations.station_positioning import approach_target_for_station  # noqa: E402


class StationPositioningTest(unittest.TestCase):
    def test_approach_target_offsets_against_station_yaw(self):
        target = approach_target_for_station(Station("A", StationPose(1.0, 0.0, 0.0), 0.25))

        self.assertAlmostEqual(target.pose.x_m, 0.75)
        self.assertAlmostEqual(target.pose.y_m, 0.0)


if __name__ == "__main__":
    unittest.main()

