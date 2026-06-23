import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.fleet.coordination_policy import station_entry_decision  # noqa: E402
from scripts.aufgabe04.fleet.models import PriorityDecision  # noqa: E402
from scripts.aufgabe04.fleet.station_locks import StationLockTable, acquire_station  # noqa: E402


class CoordinationPolicyTest(unittest.TestCase):
    def test_robot_waits_for_leased_station(self):
        table = acquire_station(StationLockTable.empty(), "A", "robot_1")

        self.assertEqual(station_entry_decision(table, "A", "robot_2"), PriorityDecision.WAIT)


if __name__ == "__main__":
    unittest.main()

