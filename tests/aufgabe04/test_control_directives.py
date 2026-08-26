from __future__ import annotations

import json
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    AcquisitionGoalAction,
    BlockageRecoveryAction,
    RouteRefreshAction,
    StartupJoinAction,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_admission import (
    certified_startup_join_action,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    acquisition_goal_action,
)


class ControlDirectivesTest(unittest.TestCase):
    def test_directives_remain_string_and_json_compatible(self):
        directives = (
            RouteRefreshAction.ADOPTED,
            BlockageRecoveryAction.CLEARED,
            StartupJoinAction.ZERO,
            AcquisitionGoalAction.HOLD_FOR_PHYSICAL_FACE,
        )

        self.assertEqual(
            directives,
            ("adopted", "cleared", "zero", "hold_for_physical_face"),
        )
        self.assertEqual(
            json.dumps({"actions": directives}),
            '{"actions": ["adopted", "cleared", "zero", '
            '"hold_for_physical_face"]}',
        )
        self.assertEqual(str(RouteRefreshAction.STOPPED), "stopped")

    def test_empty_directives_are_explicit_typed_noops(self):
        self.assertEqual(RouteRefreshAction.CONTINUE, "")
        self.assertEqual(BlockageRecoveryAction.NOT_ATTEMPTED, "")
        self.assertIs(RouteRefreshAction(""), RouteRefreshAction.CONTINUE)
        self.assertIs(
            BlockageRecoveryAction(""),
            BlockageRecoveryAction.NOT_ATTEMPTED,
        )

    def test_each_directive_domain_rejects_foreign_values(self):
        with self.assertRaises(ValueError):
            RouteRefreshAction("cleared")
        with self.assertRaises(ValueError):
            StartupJoinAction("adopted")
        with self.assertRaises(ValueError):
            AcquisitionGoalAction("stopped")

    def test_startup_join_producer_returns_typed_actions(self):
        anchor = Pose2D(0.0, 0.0)

        zero_action, zero_failure = certified_startup_join_action(
            Pose2D(0.01, 0.0, 0.0),
            anchor,
            effective_join_limit_m=0.1,
            join_tolerance_m=0.02,
        )
        anchor_action, anchor_failure = certified_startup_join_action(
            Pose2D(0.05, 0.0, 0.0),
            anchor,
            effective_join_limit_m=0.1,
            join_tolerance_m=0.02,
        )
        stop_action, stop_failure = certified_startup_join_action(
            Pose2D(0.11, 0.0, 0.0),
            anchor,
            effective_join_limit_m=0.1,
            join_tolerance_m=0.02,
        )

        self.assertIs(zero_action, StartupJoinAction.ZERO)
        self.assertIsNone(zero_failure)
        self.assertIs(anchor_action, StartupJoinAction.ANCHOR)
        self.assertIsNone(anchor_failure)
        self.assertIs(stop_action, StartupJoinAction.STOP)
        self.assertIsNotNone(stop_failure)

    def test_acquisition_goal_producer_returns_typed_actions(self):
        complete = acquisition_goal_action(
            route_kind="stand_discovery_corridor",
            provider_available=True,
            hold_elapsed_sec=0.0,
            timeout_sec=5.0,
        )
        hold = acquisition_goal_action(
            route_kind="axis_acquisition",
            provider_available=True,
            hold_elapsed_sec=0.0,
            timeout_sec=5.0,
        )
        missing_provider = acquisition_goal_action(
            route_kind="axis_acquisition",
            provider_available=False,
            hold_elapsed_sec=0.0,
            timeout_sec=5.0,
        )
        timed_out = acquisition_goal_action(
            route_kind="axis_acquisition",
            provider_available=True,
            hold_elapsed_sec=5.0,
            timeout_sec=5.0,
        )

        self.assertIs(complete, AcquisitionGoalAction.COMPLETE)
        self.assertIs(hold, AcquisitionGoalAction.HOLD_FOR_PHYSICAL_FACE)
        self.assertIs(
            missing_provider,
            AcquisitionGoalAction.MISSING_DYNAMIC_ROUTE_PROVIDER,
        )
        self.assertIs(
            timed_out,
            AcquisitionGoalAction.AXIS_ACQUISITION_TIMEOUT,
        )


if __name__ == "__main__":
    unittest.main()
