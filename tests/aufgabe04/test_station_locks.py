import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.fleet.station_locks import StationLockTable, acquire_station, release_station  # noqa: E402


class StationLocksTest(unittest.TestCase):
    def test_acquire_and_release_station(self):
        table = acquire_station(StationLockTable.empty(), "A", "robot_1")
        released = release_station(table, "A", "robot_1")

        self.assertEqual(released.leases, {})


if __name__ == "__main__":
    unittest.main()

