import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.logistics.puck_custody import (  # noqa: E402
    CustodyState,
    PuckCustodyLedger,
    claim_puck,
    confirm_puck_delivered,
    confirm_puck_loaded,
    register_puck,
    release_puck_claim,
    report_puck_lost,
)


class PuckCustodyTest(unittest.TestCase):
    def setUp(self):
        self.ledger = register_puck(
            PuckCustodyLedger.empty(),
            "puck-1",
            source_station_id="A",
            target_station_id="B",
            now_sec=1.0,
        )

    def test_claim_load_deliver_requires_same_owner_and_fence(self):
        claimed = claim_puck(self.ledger, "puck-1", "robot-1", now_sec=2.0)
        token = claimed.records["puck-1"].fencing_token
        loaded = confirm_puck_loaded(
            claimed, "puck-1", "robot-1", token, now_sec=3.0
        )
        delivered = confirm_puck_delivered(
            loaded, "puck-1", "robot-1", token, now_sec=4.0
        )

        self.assertEqual(
            delivered.records["puck-1"].state, CustodyState.DELIVERED
        )

    def test_other_robot_cannot_steal_claim(self):
        claimed = claim_puck(self.ledger, "puck-1", "robot-1", now_sec=2.0)

        with self.assertRaisesRegex(ValueError, "claimed"):
            claim_puck(claimed, "puck-1", "robot-2", now_sec=3.0)

    def test_release_and_reclaim_advances_fence(self):
        claimed = claim_puck(self.ledger, "puck-1", "robot-1", now_sec=2.0)
        first_token = claimed.records["puck-1"].fencing_token
        released = release_puck_claim(
            claimed, "puck-1", "robot-1", first_token, now_sec=3.0
        )
        reclaimed = claim_puck(released, "puck-1", "robot-1", now_sec=4.0)
        second_token = reclaimed.records["puck-1"].fencing_token

        self.assertGreater(second_token, first_token)
        with self.assertRaisesRegex(ValueError, "stale"):
            confirm_puck_loaded(
                reclaimed, "puck-1", "robot-1", first_token, now_sec=5.0
            )

    def test_loaded_puck_can_be_reported_lost(self):
        claimed = claim_puck(self.ledger, "puck-1", "robot-1", now_sec=2.0)
        token = claimed.records["puck-1"].fencing_token
        loaded = confirm_puck_loaded(
            claimed, "puck-1", "robot-1", token, now_sec=3.0
        )
        lost = report_puck_lost(
            loaded,
            "puck-1",
            "robot-1",
            token,
            reason="retention sensor opened",
            now_sec=4.0,
        )

        self.assertEqual(lost.records["puck-1"].state, CustodyState.LOST)
        self.assertEqual(lost.records["puck-1"].detail, "retention sensor opened")

    def test_clock_rollback_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "backwards"):
            claim_puck(self.ledger, "puck-1", "robot-1", now_sec=0.5)


if __name__ == "__main__":
    unittest.main()
