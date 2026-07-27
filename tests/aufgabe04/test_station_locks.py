import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.fleet.station_locks import (  # noqa: E402
    StationLockTable,
    acquire_station,
    expire_station_leases,
    release_station,
    renew_station,
    validate_station_lease,
)


class StationLocksTest(unittest.TestCase):
    def test_acquire_and_release_station(self):
        table = acquire_station(StationLockTable.empty(), "A", "robot_1")
        released = release_station(table, "A", "robot_1")

        self.assertEqual(released.leases, {})

    def test_expired_lease_can_be_reacquired_with_new_fence(self):
        first = acquire_station(
            StationLockTable.empty(),
            "A",
            "robot_1",
            now_sec=1.0,
            expires_at_sec=2.0,
        )
        first_token = first.leases["A"].fencing_token
        second = acquire_station(
            first,
            "A",
            "robot_2",
            now_sec=2.0,
            expires_at_sec=3.0,
        )

        self.assertEqual(second.leases["A"].robot_id, "robot_2")
        self.assertGreater(second.leases["A"].fencing_token, first_token)
        with self.assertRaisesRegex(ValueError, "leased by"):
            validate_station_lease(
                second, "A", "robot_1", first_token, now_sec=2.5
            )

    def test_renew_requires_current_fence_and_active_lease(self):
        table = acquire_station(
            StationLockTable.empty(),
            "A",
            "robot_1",
            now_sec=1.0,
            expires_at_sec=3.0,
        )
        token = table.leases["A"].fencing_token
        renewed = renew_station(
            table,
            "A",
            "robot_1",
            token,
            now_sec=2.0,
            expires_at_sec=5.0,
        )

        self.assertEqual(renewed.leases["A"].expires_at_sec, 5.0)
        with self.assertRaisesRegex(ValueError, "stale"):
            renew_station(
                renewed,
                "A",
                "robot_1",
                token + 1,
                now_sec=3.0,
                expires_at_sec=6.0,
            )

    def test_fenced_release_rejects_stale_token(self):
        table = acquire_station(
            StationLockTable.empty(),
            "A",
            "robot_1",
            now_sec=1.0,
            expires_at_sec=3.0,
        )

        with self.assertRaisesRegex(ValueError, "stale"):
            release_station(
                table,
                "A",
                "robot_1",
                fencing_token=99,
                now_sec=2.0,
            )

    def test_expire_and_clock_rollback(self):
        table = acquire_station(
            StationLockTable.empty(),
            "A",
            "robot_1",
            now_sec=1.0,
            expires_at_sec=2.0,
        )
        expired = expire_station_leases(table, now_sec=2.0)
        self.assertEqual(expired.leases, {})
        with self.assertRaisesRegex(ValueError, "backwards"):
            expire_station_leases(expired, now_sec=1.5)


if __name__ == "__main__":
    unittest.main()
