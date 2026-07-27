import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.fleet.reservations import (  # noqa: E402
    ReservationTable,
    acquire_permit,
    release_permit,
    renew_permit,
    validate_permit,
)


class ReservationsTest(unittest.TestCase):
    def test_expiry_reacquisition_fences_old_holder(self):
        first = acquire_permit(
            ReservationTable.empty(), "crossing-1", "robot-1", now_sec=1.0, ttl_sec=1.0
        )
        token_1 = first.permits["crossing-1"].fencing_token
        second = acquire_permit(
            first, "crossing-1", "robot-2", now_sec=2.0, ttl_sec=1.0
        )
        token_2 = second.permits["crossing-1"].fencing_token

        self.assertGreater(token_2, token_1)
        with self.assertRaises(ValueError):
            release_permit(
                second, "crossing-1", "robot-1", token_1, now_sec=2.5
            )

    def test_renew_keeps_fence(self):
        table = acquire_permit(
            ReservationTable.empty(), "crossing-1", "robot-1", now_sec=1.0, ttl_sec=1.0
        )
        token = table.permits["crossing-1"].fencing_token
        renewed = renew_permit(
            table,
            "crossing-1",
            "robot-1",
            token,
            now_sec=1.5,
            ttl_sec=2.0,
        )

        permit = validate_permit(
            renewed, "crossing-1", "robot-1", token, now_sec=3.0
        )
        self.assertEqual(permit.expires_at_sec, 3.5)

    def test_same_robot_cannot_bypass_fenced_renewal(self):
        table = acquire_permit(
            ReservationTable.empty(), "crossing-1", "robot-1", now_sec=1.0, ttl_sec=1.0
        )

        with self.assertRaisesRegex(ValueError, "fencing token"):
            acquire_permit(
                table,
                "crossing-1",
                "robot-1",
                now_sec=1.5,
                ttl_sec=2.0,
            )

    def test_clock_rollback_is_rejected(self):
        table = acquire_permit(
            ReservationTable.empty(), "crossing-1", "robot-1", now_sec=2.0, ttl_sec=1.0
        )
        with self.assertRaisesRegex(ValueError, "backwards"):
            acquire_permit(
                table, "crossing-2", "robot-2", now_sec=1.0, ttl_sec=1.0
            )


if __name__ == "__main__":
    unittest.main()
