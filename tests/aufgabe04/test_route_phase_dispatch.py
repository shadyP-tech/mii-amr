from __future__ import annotations

import unittest

from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    AcquisitionGoalDecision,
    RouteCommandPhase,
    ViewpointSamplingDeadlineDecision,
    acquisition_goal_decision,
    route_command_phase,
    viewpoint_sampling_deadline_decision,
)
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    AcquisitionGoalAction,
)


class RoutePhaseDispatchTest(unittest.TestCase):
    def test_dynamic_join_has_highest_precedence(self):
        phase = route_command_phase(
            dynamic_join_pending=True,
            start_egress_lock_index=1,
            start_egress_forward_alignment_index=2,
        )

        self.assertIs(phase, RouteCommandPhase.DYNAMIC_JOIN)

    def test_start_egress_precedes_forward_alignment(self):
        phase = route_command_phase(
            dynamic_join_pending=False,
            start_egress_lock_index=1,
            start_egress_forward_alignment_index=2,
        )

        self.assertIs(phase, RouteCommandPhase.START_EGRESS)

    def test_forward_alignment_is_selected_after_egress_lock_releases(self):
        phase = route_command_phase(
            dynamic_join_pending=False,
            start_egress_lock_index=None,
            start_egress_forward_alignment_index=2,
        )

        self.assertIs(
            phase,
            RouteCommandPhase.REVERSE_EGRESS_ALIGNMENT,
        )

    def test_certified_route_is_the_default_phase(self):
        phase = route_command_phase(
            dynamic_join_pending=False,
            start_egress_lock_index=None,
            start_egress_forward_alignment_index=None,
        )

        self.assertIs(phase, RouteCommandPhase.CERTIFIED_ROUTE)
        self.assertEqual(phase, "certified_route")

    def test_total_sampling_deadline_precedes_target_deadline(self):
        decision = viewpoint_sampling_deadline_decision(
            route_kind="viewpoint_sampling",
            phase_started_at=10.0,
            target_started_at=20.0,
            now_monotonic=40.0,
            phase_timeout_sec=30.0,
            target_timeout_sec=20.0,
        )

        self.assertIsInstance(decision, ViewpointSamplingDeadlineDecision)
        self.assertEqual(decision.failure, "viewpoint_sampling_timeout")
        self.assertEqual(decision.phase_elapsed_sec, 30.0)
        self.assertEqual(decision.target_elapsed_sec, 20.0)

    def test_target_deadline_applies_while_total_deadline_is_open(self):
        decision = viewpoint_sampling_deadline_decision(
            route_kind="viewpoint_sampling",
            phase_started_at=10.0,
            target_started_at=25.0,
            now_monotonic=40.0,
            phase_timeout_sec=31.0,
            target_timeout_sec=15.0,
        )

        self.assertEqual(
            decision.failure,
            "viewpoint_sampling_target_timeout",
        )
        self.assertEqual(decision.phase_elapsed_sec, 30.0)
        self.assertEqual(decision.target_elapsed_sec, 15.0)

    def test_missing_total_sampling_clock_fails_closed(self):
        decision = viewpoint_sampling_deadline_decision(
            route_kind="viewpoint_sampling",
            phase_started_at=None,
            target_started_at=37.0,
            now_monotonic=40.0,
            phase_timeout_sec=30.0,
            target_timeout_sec=15.0,
        )

        self.assertEqual(
            decision.failure,
            "viewpoint_sampling_clock_unavailable",
        )
        self.assertIsNone(decision.phase_elapsed_sec)
        self.assertEqual(decision.target_elapsed_sec, 3.0)

    def test_non_sampling_route_has_no_sampling_deadline(self):
        decision = viewpoint_sampling_deadline_decision(
            route_kind="stand_discovery_corridor",
            phase_started_at=10.0,
            target_started_at=20.0,
            now_monotonic=40.0,
            phase_timeout_sec=30.0,
            target_timeout_sec=15.0,
        )

        self.assertEqual(
            decision,
            ViewpointSamplingDeadlineDecision("", None, None),
        )

    def test_acquisition_goal_initializes_hold_clock(self):
        decision = acquisition_goal_decision(
            route_kind="axis_acquisition",
            provider_available=True,
            hold_started_at=None,
            now_monotonic=40.0,
            timeout_sec=5.0,
        )

        self.assertIsInstance(decision, AcquisitionGoalDecision)
        self.assertIs(
            decision.action,
            AcquisitionGoalAction.HOLD_FOR_PHYSICAL_FACE,
        )
        self.assertEqual(decision.hold_started_at, 40.0)
        self.assertEqual(decision.hold_elapsed_sec, 0.0)

    def test_acquisition_goal_reuses_clock_and_times_out_at_boundary(self):
        decision = acquisition_goal_decision(
            route_kind="axis_acquisition",
            provider_available=True,
            hold_started_at=35.0,
            now_monotonic=40.0,
            timeout_sec=5.0,
        )

        self.assertIs(
            decision.action,
            AcquisitionGoalAction.AXIS_ACQUISITION_TIMEOUT,
        )
        self.assertEqual(decision.hold_started_at, 35.0)
        self.assertEqual(decision.hold_elapsed_sec, 5.0)


if __name__ == "__main__":
    unittest.main()
