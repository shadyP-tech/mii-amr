import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.fleet.coordination_policy import (  # noqa: E402
    geometry_right_before_left_decision,
    station_entry_decision,
)
from scripts.aufgabe04.fleet.models import PriorityDecision  # noqa: E402
from scripts.aufgabe04.fleet.station_locks import StationLockTable, acquire_station  # noqa: E402


class CoordinationPolicyTest(unittest.TestCase):
    def test_robot_waits_for_leased_station(self):
        table = acquire_station(StationLockTable.empty(), "A", "robot_1")

        self.assertEqual(station_entry_decision(table, "A", "robot_2"), PriorityDecision.WAIT)

    def test_expired_station_no_longer_blocks_entry(self):
        table = acquire_station(
            StationLockTable.empty(),
            "A",
            "robot_1",
            now_sec=1.0,
            expires_at_sec=2.0,
        )

        self.assertEqual(
            station_entry_decision(table, "A", "robot_2", now_sec=2.0),
            PriorityDecision.PROCEED,
        )

    def test_geometry_yields_to_robot_on_right(self):
        decision = geometry_right_before_left_decision(
            robot_id="eastbound",
            robot_x_m=-1.0,
            robot_y_m=0.0,
            robot_yaw_rad=0.0,
            robot_requested_at_sec=1.0,
            peer_id="northbound",
            peer_x_m=0.0,
            peer_y_m=-1.0,
            peer_yaw_rad=1.5707963267948966,
            peer_requested_at_sec=1.0,
        )

        self.assertEqual(decision, PriorityDecision.YIELD)

    def test_ambiguous_geometry_has_deterministic_robot_id_tie_break(self):
        kwargs = dict(
            robot_x_m=0.0,
            robot_y_m=0.0,
            robot_yaw_rad=0.0,
            robot_requested_at_sec=1.0,
            peer_x_m=1.0,
            peer_y_m=0.0,
            peer_yaw_rad=0.0,
            peer_requested_at_sec=1.0,
        )
        first = geometry_right_before_left_decision(
            robot_id="robot-a", peer_id="robot-b", **kwargs
        )
        second = geometry_right_before_left_decision(
            robot_id="robot-b",
            peer_id="robot-a",
            robot_x_m=kwargs["peer_x_m"],
            robot_y_m=kwargs["peer_y_m"],
            robot_yaw_rad=kwargs["peer_yaw_rad"],
            robot_requested_at_sec=1.0,
            peer_x_m=kwargs["robot_x_m"],
            peer_y_m=kwargs["robot_y_m"],
            peer_yaw_rad=kwargs["robot_yaw_rad"],
            peer_requested_at_sec=1.0,
        )

        self.assertEqual(first, PriorityDecision.PROCEED)
        self.assertEqual(second, PriorityDecision.YIELD)


if __name__ == "__main__":
    unittest.main()
