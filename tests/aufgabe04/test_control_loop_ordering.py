from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerConfig,
    ControllerStep,
    VelocityCommand,
)
from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    ExecutionRouteCheck,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.config import FollowerConfig
from scripts.aufgabe04.navigation.waypoint_follower.runtime import (
    SimpleWaypointFollowerNode,
)


def _route_tube_stop_node(events: list[str]) -> SimpleWaypointFollowerNode:
    node = object.__new__(SimpleWaypointFollowerNode)
    pose = Pose2D(0.05, 0.04, 0.0)
    step = ControllerStep(
        command=VelocityCommand(0.03, 0.0),
        target_index=1,
        reached_goal=False,
        distance_to_target_m=0.15,
        pursuit_index=1,
        controlled_heading_error_rad=0.0,
    )
    node.waypoints = (Pose2D(0.0, 0.0), Pose2D(0.2, 0.0))
    node.follower_config = FollowerConfig(controller=ControllerConfig())
    node.current_route_kind = "stand_discovery_corridor"
    node.target_index = 1
    node.target_started_at = 0.0
    node.distance_estimate_m = 0.0
    node.motion_published = True
    node.last_pose = pose
    node.latest_stop_details = None
    node.waypoint_provider = None
    node.blockage_recovery_provider = None
    node.dynamic_join_pending = False
    node.dynamic_join_limit_m = None
    node.start_egress_lock_index = None
    node.start_egress_forward_alignment_index = None
    node.reverse_staging = False
    node.certified_static_start_pending = False
    node.certified_corner_latch = None
    node.intermediate_terminal_heading_latch = None
    node.axis_acquisition_hold_started_at = None
    node.viewpoint_sampling_started_at = None
    node.viewpoint_sampling_target_started_at = None
    node.latest_front_clearance_details = None
    node._wait_for_initial_runtime_inputs = lambda _started_at: ""
    node._drain_runtime_callbacks = lambda: None
    node._safety_failure = lambda: ""
    node._global_consistency_monitor_failure = lambda: ""
    node._current_pose_lookup_with_stale_recovery = lambda: SimpleNamespace(
        pose=pose
    )
    node._refresh_dynamic_route = lambda _pose: ""
    node._certified_corner_decision = lambda *_args: SimpleNamespace(
        failure="",
        step=step,
    )
    node._log_certified_corner_phase = lambda _step: None
    node._execution_route_check = lambda *_args: ExecutionRouteCheck(
        ok=False,
        reason="pose left certified route tube",
        pose_distance_to_segment_m=0.04,
        maximum_chord_distance_to_segment_m=0.04,
        active_segment_start_index=0,
        active_segment_end_index=1,
        target_index=1,
        pursuit_index=1,
        tracking_tube_radius_m=0.03,
    )
    node.publish_zero = lambda: events.append("zero")
    node.publish_repeated_zero = lambda: events.append("repeated_zero")
    node._append_controller_trace = lambda **_kwargs: events.append("trace") or ""
    return node


class ControlLoopOrderingTest(unittest.TestCase):
    def test_route_tube_stop_zeroes_before_trace_and_return(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: True),
        ):
            result = node.run()
        events.append("return")

        self.assertEqual(result.status, "stopped")
        self.assertEqual(
            result.stop_reason,
            "pose left certified route tube",
        )
        # The first repeated zero is startup admission.  The route-tube stop
        # must add another one before trace I/O, with final cleanup after it.
        stop_zero_index = events.index("repeated_zero", 1)
        self.assertLess(stop_zero_index, events.index("trace"))
        self.assertLess(stop_zero_index, events.index("return"))


if __name__ == "__main__":
    unittest.main()
