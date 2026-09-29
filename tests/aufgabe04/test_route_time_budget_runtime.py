"""Replay the failed run's deadline boundary through the production guard."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerStep, VelocityCommand,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    WaypointLifecycleAction,
)
from tests.aufgabe04.test_control_loop_ordering import _route_tube_stop_node


# Admitted odom route and observed poses from 20260929T124849Z candidate_002.
ROUTE = (
    Pose2D(3.3070012894, 1.1446570996, 1.4671965800),
    Pose2D(1.7777528241, 0.3462392961, math.nan),
)
START = Pose2D(3.3058867870, 1.1264675182, 1.4671965800)
STOP = Pose2D(1.8899342425, 0.4022654106, -2.6751786508)


def _node():
    node = _route_tube_stop_node([])
    node.waypoints = ROUTE
    node.current_route_kind = "detected_stand_preapproach"
    node.follower_config = replace(
        node.follower_config, route_time_budget_enabled=True,
    )
    node.controller_route_revision = 0
    node.budget_traces = []
    node._append_controller_trace = lambda **record: node.budget_traces.append(record) or ""
    node.progress_heading_modes_seen = set()
    node.progress_heading_error_by_mode = {}
    node._reset_progress_watchdog(0.0)
    return node


def _step(distance=0.125394, mode="path_tracking", target=1):
    return ControllerStep(
        command=VelocityCommand(0.055 if mode == "path_tracking" else 0.0, 0.18),
        target_index=target,
        reached_goal=False,
        distance_to_target_m=distance,
        pursuit_index=target,
        controlled_heading_error_rad=0.0,
        progress_mode=mode,
    )


def _decision(node, step, pose, *, monotonic_fn):
    with patch(
        "scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_loop.time.monotonic",
        side_effect=monotonic_fn,
    ):
        return node._waypoint_lifecycle_decision(step, pose)


class RouteTimeBudgetRuntimeTests(unittest.TestCase):
    def _enter(self, node):
        decision = _decision(node,
            _step(1.716, "exact_vertex_alignment"), START,
            monotonic_fn=lambda: 0.0,
        )
        self.assertIs(decision.action, WaypointLifecycleAction.PROCEED)

    def test_recorded_45_second_stop_proceeds_but_finite_deadline_still_stops(self):
        node = _node()
        self._enter(node)
        budget = node.waypoint_time_budget
        self.assertGreater(budget.timeout_sec, 59.0)
        self.assertLess(budget.timeout_sec, 61.0)
        for elapsed in (12.6, 30.0, 45.002241):
            decision = _decision(node,
                _step(), STOP, monotonic_fn=lambda: elapsed,
            )
            self.assertIs(decision.action, WaypointLifecycleAction.PROCEED)
            self.assertIs(node.waypoint_time_budget, budget)
        self.assertEqual(len(node.budget_traces), 1)
        self.assertEqual(node.target_started_at, 0.0)
        expired = _decision(node,
            _step(0.04), STOP, monotonic_fn=lambda: budget.timeout_sec + 0.01,
        )
        self.assertIs(expired.action, WaypointLifecycleAction.STOP)
        self.assertEqual(expired.stop_reason, "waypoint timeout")
        self.assertEqual(expired.stop_details["waypoint_time_budget"], budget.to_dict())
        self.assertEqual(expired.stop_details["timeout_sec"], budget.timeout_sec)

    def test_stalled_motion_still_fails_after_eight_seconds(self):
        node = _node()
        self._enter(node)
        step = _step(1.5)
        for now in (0.0, 8.01):
            lifecycle = _decision(node,
                step, START, monotonic_fn=lambda: now,
            )
            self.assertIs(lifecycle.action, WaypointLifecycleAction.PROCEED)
            progress = node._progress_watchdog_decision(
                step, now_monotonic=now, front_clearance_scale=1.0,
                effective_linear_x_mps=0.055,
            )
            self.assertEqual(bool(progress.failure), now > 8.0)
        self.assertEqual(progress.stop_details["max_without_progress_sec"], 8.0)

    def test_terminal_heading_keeps_separate_24_second_deadline(self):
        node = _node()
        self._enter(node)
        entry = node.waypoint_time_budget.timeout_sec - 0.5
        step = _step(0.02, "terminal_heading")
        for now in (entry, entry + 23.9):
            self.assertIs(_decision(node,
                step, STOP, monotonic_fn=lambda: now,
            ).action, WaypointLifecycleAction.PROCEED)
        expired = _decision(node,
            step, STOP, monotonic_fn=lambda: entry + 24.01,
        )
        self.assertEqual(expired.stop_reason, "terminal heading timeout")
        self.assertEqual(expired.stop_details["terminal_heading_timeout_sec"], 24.0)

    def test_terminal_heading_cannot_start_after_derived_deadline(self):
        node = _node()
        self._enter(node)
        expired = _decision(node,
            _step(0.02, "terminal_heading"), STOP,
            monotonic_fn=lambda: node.waypoint_time_budget.timeout_sec + 0.01,
        )
        self.assertEqual(expired.stop_reason, "waypoint timeout")
        self.assertIsNone(node.terminal_heading_budget_state.started_at)

    def test_target_change_and_admitted_route_revision_freeze_new_budget(self):
        node = _node()
        self._enter(node)
        node.waypoints = (*ROUTE, Pose2D(1.0, 0.3, 0.0))
        _decision(node,
            _step(0.7, target=2), STOP, monotonic_fn=lambda: 30.0,
        )
        second = node.waypoint_time_budget
        self.assertEqual(second.target_index, 2)
        self.assertEqual(node.target_started_at, 30.0)
        # Route adoption already owns the timer reset; budget calculation must
        # honor it even when the replacement retains the same target index.
        node.controller_route_revision = 1
        node.target_started_at = 40.0
        _decision(node,
            _step(0.7, target=2), STOP, monotonic_fn=lambda: 40.0,
        )
        self.assertIsNot(node.waypoint_time_budget, second)
        self.assertEqual(len(node.budget_traces), 3)

    def test_insufficient_explicit_limit_and_trace_failure_stop_before_motion(self):
        for trace_failure in (False, True):
            with self.subTest(trace_failure=trace_failure):
                node = _node()
                if trace_failure:
                    node._append_controller_trace = lambda **_: "trace write failed"
                else:
                    node.follower_config = replace(
                        node.follower_config, waypoint_timeout_limit_sec=45.0,
                    )
                decision = _decision(node,
                    _step(1.716), START, monotonic_fn=lambda: 0.0,
                )
                self.assertIs(decision.action, WaypointLifecycleAction.STOP)
                self.assertEqual(decision.stop_details["fault_code"], "route_time_budget_rejected")
                self.assertFalse(hasattr(node, "waypoint_time_budget_key"))

    def test_unaffected_route_keeps_legacy_fixed_timeout(self):
        node = _node()
        node.current_route_kind = "stand_discovery_corridor"
        decision = _decision(node,
            _step(), STOP, monotonic_fn=lambda: 45.01,
        )
        self.assertEqual(decision.stop_reason, "waypoint timeout")
        self.assertEqual(node.budget_traces, [])

    def test_invalid_budget_configuration_is_rejected(self):
        node = _node()
        for value in (0.0, -1.0, math.inf, math.nan, True, "45"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(node.follower_config, waypoint_timeout_limit_sec=value)
        with self.assertRaises(ValueError):
            replace(node.follower_config, route_time_budget_enabled=1)
