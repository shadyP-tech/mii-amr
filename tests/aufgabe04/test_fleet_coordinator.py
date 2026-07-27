import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.fleet.coordinator import (  # noqa: E402
    ConflictZoneRequest,
    FleetCoordinatorState,
    coordinate_conflict_zone,
    release_station_permit,
    renew_station_permit,
    request_station_permit,
)
from scripts.aufgabe04.fleet.models import PriorityDecision  # noqa: E402
from scripts.aufgabe04.fleet.robot_status import RobotStatus  # noqa: E402


def request(robot_id, x_m, y_m, yaw_rad, *, requested_at=9.0, stamp=10.0):
    return ConflictZoneRequest(
        zone_id="crossing-1",
        status=RobotStatus(
            robot_id,
            x_m=x_m,
            y_m=y_m,
            yaw_rad=yaw_rad,
            timestamp_sec=stamp,
        ),
        requested_at_sec=requested_at,
    )


class FleetCoordinatorTest(unittest.TestCase):
    def coordinate(self, requests, state=None):
        return coordinate_conflict_zone(
            state or FleetCoordinatorState.empty(),
            "crossing-1",
            requests,
            now_sec=10.0,
            permit_ttl_sec=2.0,
            max_status_age_sec=0.5,
        )

    def test_right_before_left_winner_receives_only_permit(self):
        eastbound = request("eastbound", -1.0, 0.0, 0.0)
        northbound = request("northbound", 0.0, -1.0, math.pi / 2.0)

        outcome = self.coordinate((eastbound, northbound))
        decisions = {item.robot_id: item for item in outcome.decisions}

        self.assertEqual(outcome.winner_robot_id, "northbound")
        self.assertEqual(decisions["northbound"].decision, PriorityDecision.PROCEED)
        self.assertEqual(decisions["eastbound"].decision, PriorityDecision.YIELD)
        self.assertGreater(decisions["northbound"].fencing_token, 0)

    def test_stale_peer_causes_everyone_to_wait_without_permit(self):
        fresh = request("fresh", -1.0, 0.0, 0.0)
        stale = request("stale", 0.0, -1.0, math.pi / 2.0, stamp=8.0)

        outcome = self.coordinate((fresh, stale))

        self.assertEqual(outcome.winner_robot_id, "")
        self.assertTrue(
            all(item.decision == PriorityDecision.WAIT for item in outcome.decisions)
        )
        self.assertEqual(outcome.state.reservations.permits, {})

    def test_ambiguous_cycle_uses_deterministic_tie_break(self):
        first = request("robot-b", -1.0, 0.0, 0.0, requested_at=9.0)
        second = request("robot-a", 1.0, 0.0, math.pi, requested_at=9.0)

        outcome = self.coordinate((first, second))

        self.assertEqual(outcome.winner_robot_id, "robot-a")
        self.assertTrue(outcome.tie_break_applied)

    def test_active_permit_is_not_stolen(self):
        owner = request("owner", -1.0, 0.0, 0.0)
        initial = self.coordinate((owner,))
        contender = request("contender", 0.0, -1.0, math.pi / 2.0)
        repeated = self.coordinate((owner, contender), state=initial.state)
        decisions = {item.robot_id: item.decision for item in repeated.decisions}

        self.assertEqual(repeated.winner_robot_id, "owner")
        self.assertEqual(decisions["owner"], PriorityDecision.PROCEED)
        self.assertEqual(decisions["contender"], PriorityDecision.WAIT)

    def test_absent_permit_owner_causes_wait(self):
        initial = self.coordinate((request("owner", -1.0, 0.0, 0.0),))
        contender = request("contender", 0.0, -1.0, math.pi / 2.0)

        repeated = self.coordinate((contender,), state=initial.state)

        self.assertEqual(repeated.winner_robot_id, "")
        self.assertEqual(repeated.decisions[0].decision, PriorityDecision.WAIT)

    def test_station_permit_waits_and_expiry_reacquisition_advances_fence(self):
        first = request_station_permit(
            FleetCoordinatorState.empty(),
            "station-A",
            "robot-1",
            now_sec=1.0,
            lease_ttl_sec=1.0,
        )
        waiting = request_station_permit(
            first.state,
            "station-A",
            "robot-2",
            now_sec=1.5,
            lease_ttl_sec=1.0,
        )
        second = request_station_permit(
            waiting.state,
            "station-A",
            "robot-2",
            now_sec=2.0,
            lease_ttl_sec=1.0,
        )

        self.assertEqual(waiting.decision, PriorityDecision.WAIT)
        self.assertEqual(second.decision, PriorityDecision.PROCEED)
        self.assertGreater(second.fencing_token, first.fencing_token)

    def test_station_renew_and_release_require_fence(self):
        outcome = request_station_permit(
            FleetCoordinatorState.empty(),
            "station-A",
            "robot-1",
            now_sec=1.0,
            lease_ttl_sec=1.0,
        )
        renewed = renew_station_permit(
            outcome.state,
            "station-A",
            "robot-1",
            outcome.fencing_token,
            now_sec=1.5,
            lease_ttl_sec=2.0,
        )
        released = release_station_permit(
            renewed,
            "station-A",
            "robot-1",
            outcome.fencing_token,
            now_sec=2.0,
        )

        self.assertEqual(released.station_locks.leases, {})

    def test_station_stale_fence_cannot_release(self):
        outcome = request_station_permit(
            FleetCoordinatorState.empty(),
            "station-A",
            "robot-1",
            now_sec=1.0,
            lease_ttl_sec=2.0,
        )

        with self.assertRaisesRegex(ValueError, "stale"):
            release_station_permit(
                outcome.state,
                "station-A",
                "robot-1",
                outcome.fencing_token + 1,
                now_sec=1.5,
            )

    def test_duplicate_station_request_does_not_disclose_or_bypass_fence(self):
        outcome = request_station_permit(
            FleetCoordinatorState.empty(),
            "station-A",
            "robot-1",
            now_sec=1.0,
            lease_ttl_sec=2.0,
        )
        duplicate = request_station_permit(
            outcome.state,
            "station-A",
            "robot-1",
            now_sec=1.5,
            lease_ttl_sec=2.0,
        )

        self.assertEqual(duplicate.decision, PriorityDecision.WAIT)
        self.assertEqual(duplicate.fencing_token, 0)


if __name__ == "__main__":
    unittest.main()
