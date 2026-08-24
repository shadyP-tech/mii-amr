import unittest

from scripts.aufgabe04.navigation.control.cmd_vel_guard import (
    CommandLease,
    GuardedCommand,
    guard_command,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import VelocityCommand


class CmdVelGuardTest(unittest.TestCase):
    def setUp(self):
        self.lease = CommandLease("planner", 3, 10.0, 11.0)
        self.command = GuardedCommand(
            "planner", 3, 7, 10.4, VelocityCommand(0.05, 0.1)
        )

    def test_accepts_fresh_owned_monotonic_command(self):
        decision = guard_command(
            self.lease,
            self.command,
            now_monotonic_sec=10.5,
            last_accepted_sequence=6,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.command.linear_x_mps, 0.05)
        self.assertEqual(decision.next_sequence, 7)

    def test_zeroes_expired_replayed_or_wrong_owner_commands(self):
        expired = guard_command(
            self.lease, self.command, now_monotonic_sec=11.1
        )
        replayed = guard_command(
            self.lease,
            self.command,
            now_monotonic_sec=10.5,
            last_accepted_sequence=7,
        )
        wrong_owner = guard_command(
            self.lease,
            GuardedCommand(
                "teleop", 3, 8, 10.4, VelocityCommand(0.05, 0.0)
            ),
            now_monotonic_sec=10.5,
        )

        self.assertFalse(expired.accepted)
        self.assertFalse(replayed.accepted)
        self.assertFalse(wrong_owner.accepted)
        self.assertEqual(expired.command.linear_x_mps, 0.0)
        self.assertEqual(replayed.command.angular_z_radps, 0.0)
        self.assertEqual(wrong_owner.reason, "command owner or epoch mismatch")


if __name__ == "__main__":
    unittest.main()
