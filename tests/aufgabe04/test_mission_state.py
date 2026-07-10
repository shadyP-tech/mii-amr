import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.logistics.mission_state import mark_visit_complete, start_from_station_order  # noqa: E402
from scripts.aufgabe04.logistics.models import MissionState  # noqa: E402


class MissionStateTest(unittest.TestCase):
    def test_mission_completes_after_last_visit(self):
        snapshot = start_from_station_order(("A",))

        completed = mark_visit_complete(snapshot)

        self.assertEqual(completed.state, MissionState.COMPLETED)


if __name__ == "__main__":
    unittest.main()

