from __future__ import annotations

import math
import unittest

from scripts.aufgabe04.navigation.waypoint_follower.terminal_heading_budget import (
    DEFAULT_TERMINAL_HEADING_TIMEOUT_SEC,
    TERMINAL_HEADING_TIMEOUT,
    reset_terminal_heading_budget,
    terminal_heading_budget_decision,
)


class TerminalHeadingBudgetTest(unittest.TestCase):
    def test_default_covers_pi_rotation_with_bounded_margin(self):
        theoretical_rotation_sec = math.pi / 0.18

        self.assertGreaterEqual(
            DEFAULT_TERMINAL_HEADING_TIMEOUT_SEC,
            theoretical_rotation_sec + 5.0,
        )
        self.assertLessEqual(DEFAULT_TERMINAL_HEADING_TIMEOUT_SEC, 30.0)

    def test_only_final_terminal_heading_arms_the_budget(self):
        state = reset_terminal_heading_budget(target_index=1)

        intermediate = terminal_heading_budget_decision(
            state,
            target_index=1,
            final_target_index=2,
            progress_mode="terminal_heading",
            now_monotonic=40.0,
            timeout_sec=24.0,
        )
        tracking = terminal_heading_budget_decision(
            state,
            target_index=2,
            final_target_index=2,
            progress_mode="path_tracking",
            now_monotonic=40.0,
            timeout_sec=24.0,
        )

        self.assertFalse(intermediate.active)
        self.assertIsNone(intermediate.state.started_at)
        self.assertFalse(tracking.active)
        self.assertIsNone(tracking.state.started_at)

    def test_entry_arms_once_and_mode_chatter_cannot_extend_it(self):
        entered = terminal_heading_budget_decision(
            None,
            target_index=2,
            final_target_index=2,
            progress_mode="terminal_heading",
            now_monotonic=42.0,
            timeout_sec=24.0,
        )
        left_mode = terminal_heading_budget_decision(
            entered.state,
            target_index=2,
            final_target_index=2,
            progress_mode="path_tracking",
            now_monotonic=50.0,
            timeout_sec=24.0,
        )
        reentered = terminal_heading_budget_decision(
            left_mode.state,
            target_index=2,
            final_target_index=2,
            progress_mode="terminal_heading",
            now_monotonic=66.1,
            timeout_sec=24.0,
        )

        self.assertTrue(entered.active)
        self.assertEqual(entered.state.started_at, 42.0)
        self.assertFalse(left_mode.active)
        self.assertEqual(left_mode.state.started_at, 42.0)
        self.assertTrue(reentered.active)
        self.assertEqual(reentered.state.started_at, 42.0)
        self.assertEqual(reentered.failure, TERMINAL_HEADING_TIMEOUT)

    def test_expired_waypoint_cannot_acquire_a_fresh_phase_budget(self):
        decision = terminal_heading_budget_decision(
            None,
            target_index=2,
            final_target_index=2,
            progress_mode="terminal_heading",
            now_monotonic=45.1,
            timeout_sec=24.0,
            entry_allowed=False,
        )

        self.assertFalse(decision.active)
        self.assertIsNone(decision.state.started_at)

    def test_invalid_time_inputs_are_rejected(self):
        for value in (math.inf, math.nan):
            with self.subTest(field="now_monotonic", value=value):
                with self.assertRaisesRegex(ValueError, "now_monotonic"):
                    terminal_heading_budget_decision(
                        None,
                        target_index=2,
                        final_target_index=2,
                        progress_mode="terminal_heading",
                        now_monotonic=value,
                        timeout_sec=24.0,
                    )
        for value in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(field="timeout_sec", value=value):
                with self.assertRaisesRegex(ValueError, "timeout_sec"):
                    terminal_heading_budget_decision(
                        None,
                        target_index=2,
                        final_target_index=2,
                        progress_mode="terminal_heading",
                        now_monotonic=42.0,
                        timeout_sec=value,
                    )

    def test_target_change_discards_the_old_clock(self):
        entered = terminal_heading_budget_decision(
            None,
            target_index=1,
            final_target_index=1,
            progress_mode="terminal_heading",
            now_monotonic=10.0,
            timeout_sec=24.0,
        )
        changed = terminal_heading_budget_decision(
            entered.state,
            target_index=2,
            final_target_index=2,
            progress_mode="path_tracking",
            now_monotonic=20.0,
            timeout_sec=24.0,
        )

        self.assertFalse(changed.active)
        self.assertEqual(changed.state.target_index, 2)
        self.assertIsNone(changed.state.started_at)

    def test_material_route_reset_clears_clock_even_for_same_target_index(self):
        entered = terminal_heading_budget_decision(
            None,
            target_index=2,
            final_target_index=2,
            progress_mode="terminal_heading",
            now_monotonic=10.0,
            timeout_sec=24.0,
        )

        reset = reset_terminal_heading_budget(target_index=2)

        self.assertIsNotNone(entered.state.started_at)
        self.assertEqual(reset.target_index, 2)
        self.assertIsNone(reset.started_at)

    def test_timeout_boundary_is_admitted_and_greater_value_fails(self):
        entered = terminal_heading_budget_decision(
            None,
            target_index=2,
            final_target_index=2,
            progress_mode="terminal_heading",
            now_monotonic=10.0,
            timeout_sec=24.0,
        )
        at_boundary = terminal_heading_budget_decision(
            entered.state,
            target_index=2,
            final_target_index=2,
            progress_mode="terminal_heading",
            now_monotonic=34.0,
            timeout_sec=24.0,
        )
        over_boundary = terminal_heading_budget_decision(
            entered.state,
            target_index=2,
            final_target_index=2,
            progress_mode="terminal_heading",
            now_monotonic=34.001,
            timeout_sec=24.0,
        )

        self.assertEqual(at_boundary.elapsed_sec, 24.0)
        self.assertEqual(at_boundary.failure, "")
        self.assertEqual(over_boundary.failure, TERMINAL_HEADING_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
