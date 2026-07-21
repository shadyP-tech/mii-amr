from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.simple_waypoint_follower import (
    FollowerConfig,
    SimpleWaypointFollowerNode,
    acquisition_goal_action,
    certified_startup_join_action,
    certified_startup_route_state,
    controller_config_for_route_kind,
    dynamic_route_kind_transition_failure,
    dynamic_join_envelope_failure,
    viewpoint_sampling_timeout_failure,
)
from scripts.aufgabe04.navigation.waypoint_controller import (
    ControllerConfig,
    compute_join_anchor_command,
    compute_start_egress_vertex_command,
)


def bare_follower(update: RouteUpdate, callback):
    node = object.__new__(SimpleWaypointFollowerNode)
    node.waypoints = (Pose2D(0.0, 0.0), Pose2D(0.1, 0.0))
    node.follower_config = FollowerConfig(
        controller=ControllerConfig(enforce_heading_corridor=True),
        dynamic_route_refresh_sec=0.1,
    )
    node.waypoint_provider = lambda _pose: update
    node.route_update_callback = callback
    node.last_route_refresh_at = -1.0
    node.initial_route_refresh_pending = True
    node.target_index = 1
    node.target_started_at = 0.0
    node.last_progress_distance_m = 0.0
    node.last_progress_heading_error_rad = math.inf
    node.last_progress_target_index = None
    node.last_progress_pursuit_index = None
    node.last_progress_at = 0.0
    node.last_pose = None
    node.latest_stop_details = None
    node.dynamic_join_pending = False
    node.dynamic_join_limit_m = None
    node.start_egress_lock_index = None
    node.current_route_kind = "axis_acquisition"
    node.reverse_staging = False
    node.axis_acquisition_hold_started_at = None
    node.viewpoint_sampling_started_at = None
    return node


class DynamicFollowerHandoffTest(unittest.TestCase):
    def test_static_catalog_startup_is_anchor_zero_then_egress_vertex(self):
        waypoints = (
            Pose2D(-0.18491188596302915, -0.200843551718869, float("nan")),
            Pose2D(-0.19499999999999984, -0.11499999999999977, float("nan")),
            Pose2D(-0.6449999999999996, -0.11499999999999977, float("nan")),
        )
        join_limit_m = 0.03508711403697043
        join_tolerance_m = 0.02
        startup = certified_startup_route_state(
            FollowerConfig(
                controller=ControllerConfig(),
                initial_start_egress_waypoint_index=1,
                initial_start_join_clearance_m=join_limit_m,
            ),
            len(waypoints),
        )
        self.assertTrue(startup.join_pending)
        self.assertEqual(startup.join_limit_m, join_limit_m)
        self.assertEqual(startup.egress_lock_index, 1)

        outside_action, outside_failure = certified_startup_join_action(
            Pose2D(waypoints[0].x_m + 0.036, waypoints[0].y_m, 0.0),
            waypoints[0],
            join_limit_m,
            join_tolerance_m,
        )
        self.assertEqual(outside_action, "stop")
        self.assertEqual(outside_failure["fault_code"], "join_envelope_exceeded")

        live_pose = Pose2D(
            waypoints[0].x_m + 0.025,
            waypoints[0].y_m,
            math.pi,
        )
        action, failure = certified_startup_join_action(
            live_pose,
            waypoints[0],
            join_limit_m,
            join_tolerance_m,
        )
        self.assertEqual((action, failure), ("anchor", None))
        anchor_step = compute_join_anchor_command(
            live_pose,
            waypoints[0],
            ControllerConfig(),
            join_tolerance_m=join_tolerance_m,
        )
        self.assertEqual(anchor_step.pursuit_index, 0)

        anchored_pose = Pose2D(waypoints[0].x_m, waypoints[0].y_m, 0.0)
        action, failure = certified_startup_join_action(
            anchored_pose,
            waypoints[0],
            join_limit_m,
            join_tolerance_m,
        )
        self.assertEqual((action, failure), ("zero", None))
        egress_step = compute_start_egress_vertex_command(
            anchored_pose,
            waypoints,
            1,
            ControllerConfig(),
        )
        self.assertEqual(egress_step.pursuit_index, 1)
        self.assertNotEqual(egress_step.pursuit_index, 2)

    def test_heading_convergence_resets_progress_but_wrong_rotation_times_out(self):
        node = bare_follower(
            RouteUpdate(kind=RouteUpdateKind.UNCHANGED),
            None,
        )
        node.last_progress_distance_m = math.inf
        node.last_progress_heading_error_rad = math.inf

        self.assertEqual(
            node._progress_failure(0.365, 1.20, 2, 2, 0.0, True),
            "",
        )
        # Distance improved only 7 mm, but the necessary post-egress turn made
        # more than 0.10 rad of controlled-heading progress before 8 seconds.
        self.assertEqual(
            node._progress_failure(0.358, 1.08, 2, 2, 7.0, True),
            "",
        )
        self.assertEqual(
            node._progress_failure(0.351, 0.96, 2, 2, 14.0, True),
            "",
        )
        self.assertEqual(node.last_progress_at, 14.0)

        # Rotation that stalls or increases the error still fails closed.
        self.assertEqual(
            node._progress_failure(0.350, 1.02, 2, 2, 22.1, True),
            "stuck no progress",
        )

        node._reset_progress_watchdog(30.0)
        self.assertEqual(
            node._progress_failure(0.35, 0.80, 2, 2, 30.0, True),
            "",
        )
        self.assertEqual(
            node._progress_failure(0.35, 1.00, 2, 2, 38.1, True),
            "stuck no progress",
        )

    def test_adopted_start_egress_lock_targets_vertex_one_then_releases(self):
        waypoints = (
            Pose2D(-0.131011, -0.270103, float("nan")),
            Pose2D(-0.195, -0.115, float("nan")),
            Pose2D(-0.595, -0.115, float("nan")),
        )
        update = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=waypoints,
            target_index=0,
            event_fields={
                "effective_join_limit_m": 0.20,
                "route_kind": "axis_acquisition",
                "start_egress_vertex_lock": True,
                "start_egress_waypoint_index": 1,
                "start_egress_continuous_clearance_validated": True,
            },
        )
        node = bare_follower(update, None)
        node.publish_zero = lambda: None
        start = Pose2D(-0.131011, -0.270103, -2.702)

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=10.0,
        ):
            self.assertEqual(node._refresh_dynamic_route(start), "adopted")
            node.dynamic_join_pending = False
            step = node._start_egress_command(
                start,
                node.follower_config.controller,
            )

        self.assertEqual(node.start_egress_lock_index, 1)
        self.assertEqual(step.target_index, 1)
        self.assertEqual(step.pursuit_index, 1)
        self.assertEqual(step.command.linear_x_mps, 0.0)

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=11.0,
        ):
            released = node._start_egress_command(
                Pose2D(-0.195, -0.115, 1.9),
                node.follower_config.controller,
            )
        self.assertIsNone(released)
        self.assertIsNone(node.start_egress_lock_index)
        self.assertEqual(node.target_index, 1)

        node.start_egress_lock_index = 1
        ordinary = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(Pose2D(-0.195, -0.115), Pose2D(-0.8, -0.1)),
            target_index=0,
            event_fields={
                "effective_join_limit_m": 0.20,
                "route_kind": "axis_acquisition",
            },
        )
        node.waypoint_provider = lambda _pose: ordinary
        node.initial_route_refresh_pending = True
        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=12.0,
        ):
            self.assertEqual(
                node._refresh_dynamic_route(Pose2D(-0.195, -0.115, 1.9)),
                "adopted",
            )
        self.assertIsNone(node.start_egress_lock_index)

    def test_sensor_receipts_use_monotonic_time_not_uninitialized_ros_time(self):
        node = object.__new__(SimpleWaypointFollowerNode)
        scan = object()
        odom = object()

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            side_effect=(41.0, 42.0),
        ):
            node._scan_callback(scan)
            node._odom_callback(odom)

        self.assertIs(node.latest_scan, scan)
        self.assertEqual(node.latest_scan_receipt, 41.0)
        self.assertIs(node.latest_odom, odom)
        self.assertEqual(node.latest_odom_receipt, 42.0)

    def test_motion_clearance_uses_rear_sector_only_while_reversing(self):
        node = object.__new__(SimpleWaypointFollowerNode)
        node.follower_config = FollowerConfig(controller=ControllerConfig())
        node.latest_front_clearance_details = None
        node.latest_scan = SimpleNamespace(
            ranges=(0.50, 0.80, 0.25, 0.80, 0.50),
            angle_min=-math.pi,
            angle_increment=math.pi / 2.0,
            range_min=0.12,
            range_max=3.5,
        )

        forward_scale = node._motion_clearance_linear_scale(0.05)
        reverse_scale = node._motion_clearance_linear_scale(-0.05)

        self.assertLess(forward_scale, 1.0)
        self.assertEqual(reverse_scale, 1.0)
        self.assertEqual(node.latest_front_clearance_details["source"], "rear_sector")

        node.latest_scan.ranges = (0.19, 0.80, 0.50, 0.80, 0.19)
        self.assertEqual(node._motion_clearance_linear_scale(-0.05), 0.0)

    def test_heading_corridor_only_applies_to_physical_face_routes(self):
        configured = ControllerConfig(enforce_heading_corridor=True)

        self.assertFalse(
            controller_config_for_route_kind(
                configured, "axis_acquisition"
            ).enforce_heading_corridor
        )
        self.assertFalse(
            controller_config_for_route_kind(
                configured,
                "viewpoint_sampling",
                viewpoint_sampling_goal_tolerance_m=0.01,
            ).enforce_heading_corridor
        )
        sampling = controller_config_for_route_kind(
            configured,
            "viewpoint_sampling",
            viewpoint_sampling_goal_tolerance_m=0.01,
            viewpoint_sampling_heading_tolerance_rad=math.radians(5.0),
        )
        self.assertEqual(sampling.goal_tolerance_m, 0.01)
        self.assertAlmostEqual(sampling.heading_tolerance_rad, math.radians(5.0))
        self.assertFalse(sampling.reverse_staging)
        self.assertTrue(
            controller_config_for_route_kind(
                configured,
                "synchronized_face_approach",
                reverse_staging=True,
                physical_goal_tolerance_m=0.03,
            ).enforce_heading_corridor
        )
        physical = controller_config_for_route_kind(
            configured,
            "synchronized_face_approach",
            reverse_staging=True,
            physical_goal_tolerance_m=0.03,
        )
        self.assertTrue(physical.reverse_staging)
        self.assertEqual(physical.goal_tolerance_m, 0.03)
        self.assertTrue(
            controller_config_for_route_kind(
                configured, "synchronized_viewpoint"
            ).enforce_heading_corridor
        )
        catalog = controller_config_for_route_kind(
            configured,
            "catalog_face_approach",
            physical_goal_tolerance_m=0.03,
        )
        self.assertTrue(catalog.enforce_heading_corridor)
        self.assertEqual(catalog.goal_tolerance_m, 0.03)

    def test_non_dynamic_route_preserves_explicit_corridor_setting(self):
        configured = ControllerConfig(enforce_heading_corridor=True)

        self.assertIs(
            controller_config_for_route_kind(configured, "ordinary_route"),
            configured,
        )

    def test_axis_acquisition_goal_holds_until_physical_route_arrives(self):
        self.assertEqual(
            acquisition_goal_action(
                route_kind="axis_acquisition",
                provider_available=True,
                hold_elapsed_sec=4.0,
                timeout_sec=12.0,
            ),
            "hold_for_physical_face",
        )
        self.assertEqual(
            acquisition_goal_action(
                route_kind="viewpoint_sampling",
                provider_available=True,
                hold_elapsed_sec=4.0,
                timeout_sec=12.0,
            ),
            "hold_for_physical_face",
        )
        self.assertEqual(
            acquisition_goal_action(
                route_kind="synchronized_face_approach",
                provider_available=True,
                hold_elapsed_sec=0.0,
                timeout_sec=12.0,
            ),
            "complete",
        )

    def test_axis_acquisition_goal_fails_closed_without_revision(self):
        self.assertEqual(
            acquisition_goal_action(
                route_kind="axis_acquisition",
                provider_available=True,
                hold_elapsed_sec=12.0,
                timeout_sec=12.0,
            ),
            "axis_acquisition_timeout",
        )
        self.assertEqual(
            acquisition_goal_action(
                route_kind="axis_acquisition",
                provider_available=False,
                hold_elapsed_sec=0.0,
                timeout_sec=12.0,
            ),
            "missing_dynamic_route_provider",
        )

    def test_one_shot_provider_is_polled_exactly_once(self):
        update = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(Pose2D(0.02, 0.0), Pose2D(0.2, 0.0)),
            target_index=0,
            event_fields={
                "effective_join_limit_m": 0.2,
                "route_kind": "axis_acquisition",
            },
        )
        node = bare_follower(update, None)
        node.follower_config = FollowerConfig(
            controller=ControllerConfig(enforce_heading_corridor=True),
            dynamic_route_refresh_sec=0.0,
        )
        calls = []
        node.waypoint_provider = lambda pose: calls.append(pose) or update
        node.publish_zero = lambda: None

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=10.0,
        ):
            self.assertEqual(
                node._refresh_dynamic_route(Pose2D(0.02, 0.0)), "adopted"
            )
            self.assertEqual(node._refresh_dynamic_route(Pose2D(0.03, 0.0)), "")

        self.assertEqual(calls, [Pose2D(0.02, 0.0)])

    def test_survey_completion_stops_with_success(self):
        update = RouteUpdate(
            kind=RouteUpdateKind.COMPLETE,
            reason="arrival pose recorded",
            event_name="dynamic_survey_completed",
            event_fields={
                "candidate_uid": "candidate-a",
                "fail_closed": False,
            },
        )
        events = []
        node = bare_follower(update, events.append)
        zero_calls = []
        node.publish_zero = lambda: zero_calls.append(True)

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=10.0,
        ):
            result = node._refresh_dynamic_route(Pose2D(0.02, 0.0))

        self.assertEqual(result, "completed")
        self.assertEqual(len(zero_calls), 1)
        self.assertEqual(events, [update])
        self.assertFalse(node.latest_stop_details["fail_closed"])

    def test_sampling_phase_deadline_is_total(self):
        self.assertEqual(
            viewpoint_sampling_timeout_failure(
                route_kind="viewpoint_sampling",
                phase_started_at=10.0,
                now_monotonic=39.9,
                timeout_sec=30.0,
            ),
            "",
        )
        self.assertEqual(
            viewpoint_sampling_timeout_failure(
                route_kind="viewpoint_sampling",
                phase_started_at=10.0,
                now_monotonic=40.0,
                timeout_sec=30.0,
            ),
            "viewpoint_sampling_timeout",
        )
        self.assertEqual(
            viewpoint_sampling_timeout_failure(
                route_kind="synchronized_face_approach",
                phase_started_at=None,
                now_monotonic=100.0,
                timeout_sec=30.0,
            ),
            "",
        )

    def test_dynamic_route_kind_transitions_reject_backward_or_unknown_states(self):
        self.assertEqual(
            dynamic_route_kind_transition_failure(
                "axis_acquisition", "viewpoint_sampling"
            ),
            "",
        )
        self.assertEqual(
            dynamic_route_kind_transition_failure(
                "viewpoint_sampling", "synchronized_face_approach"
            ),
            "",
        )
        self.assertIn(
            "backward",
            dynamic_route_kind_transition_failure(
                "viewpoint_sampling", "axis_acquisition"
            ),
        )
        self.assertIn(
            "missing",
            dynamic_route_kind_transition_failure("axis_acquisition", ""),
        )
        self.assertIn(
            "unknown",
            dynamic_route_kind_transition_failure(
                "axis_acquisition", "mystery_route"
            ),
        )

    def test_route_phase_clocks_change_only_on_genuine_forward_transition(self):
        same_axis = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(Pose2D(0.02, 0.0), Pose2D(0.2, 0.0)),
            target_index=0,
            event_fields={
                "effective_join_limit_m": 0.2,
                "route_kind": "axis_acquisition",
            },
        )
        node = bare_follower(same_axis, None)
        node.axis_acquisition_hold_started_at = 4.0
        node.publish_zero = lambda: None
        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=10.0,
        ):
            self.assertEqual(
                node._refresh_dynamic_route(Pose2D(0.02, 0.0)), "adopted"
            )
        self.assertEqual(node.axis_acquisition_hold_started_at, 4.0)
        self.assertIsNone(node.viewpoint_sampling_started_at)

        sampling = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(Pose2D(0.02, 0.0), Pose2D(0.25, 0.1)),
            target_index=0,
            event_fields={
                "effective_join_limit_m": 0.2,
                "route_kind": "viewpoint_sampling",
            },
        )
        node.waypoint_provider = lambda _pose: sampling
        node.initial_route_refresh_pending = True
        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=12.0,
        ):
            self.assertEqual(
                node._refresh_dynamic_route(Pose2D(0.02, 0.0)), "adopted"
            )
        self.assertIsNone(node.axis_acquisition_hold_started_at)
        self.assertEqual(node.viewpoint_sampling_started_at, 12.0)

        node.initial_route_refresh_pending = True
        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=22.0,
        ):
            self.assertEqual(
                node._refresh_dynamic_route(Pose2D(0.02, 0.0)), "adopted"
            )
        self.assertEqual(node.viewpoint_sampling_started_at, 12.0)

        physical = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(Pose2D(0.02, 0.0), Pose2D(0.3, 0.0)),
            target_index=0,
            event_fields={
                "effective_join_limit_m": 0.2,
                "route_kind": "synchronized_face_approach",
            },
        )
        node.waypoint_provider = lambda _pose: physical
        node.initial_route_refresh_pending = True
        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=29.0,
        ):
            self.assertEqual(
                node._refresh_dynamic_route(Pose2D(0.02, 0.0)), "adopted"
            )
        self.assertEqual(node.current_route_kind, "synchronized_face_approach")
        self.assertIsNone(node.viewpoint_sampling_started_at)
        self.assertIsNone(node.axis_acquisition_hold_started_at)

    def test_physical_route_adoption_locks_reverse_staging_and_logs_it(self):
        physical = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(
                Pose2D(0.0, 0.0, float("nan")),
                Pose2D(0.5, 0.0, float("nan")),
                Pose2D(0.5, 0.0, math.pi),
                Pose2D(0.3, 0.0, math.pi),
            ),
            target_index=0,
            event_name="dynamic_route_adopted",
            event_fields={
                "effective_join_limit_m": 0.2,
                "route_kind": "synchronized_face_approach",
            },
        )
        emitted = []
        node = bare_follower(physical, emitted.append)
        node.publish_zero = lambda: None

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=10.0,
        ):
            self.assertEqual(
                node._refresh_dynamic_route(Pose2D(0.0, 0.0, math.pi)),
                "adopted",
            )

        self.assertTrue(node.reverse_staging)
        self.assertEqual(emitted[0].event_fields["staging_motion"], "reverse")
        self.assertEqual(
            emitted[0].event_fields["physical_goal_tolerance_m"], 0.03
        )

    def test_backward_sampling_transition_stops_before_route_mutation(self):
        backward = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(Pose2D(0.02, 0.0), Pose2D(0.8, 0.0)),
            target_index=0,
            event_fields={
                "effective_join_limit_m": 0.2,
                "route_kind": "axis_acquisition",
            },
        )
        node = bare_follower(backward, None)
        node.current_route_kind = "viewpoint_sampling"
        node.viewpoint_sampling_started_at = 2.0
        original = node.waypoints
        zeros = []
        node.publish_zero = lambda: zeros.append(True)

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=10.0,
        ):
            outcome = node._refresh_dynamic_route(Pose2D(0.02, 0.0))

        self.assertEqual(outcome, "stopped")
        self.assertEqual(node.waypoints, original)
        self.assertTrue(zeros)
        self.assertEqual(node.latest_stop_details["fault_code"], "invalid_route_phase")

    def test_join_envelope_rejects_excess_and_nonfinite_live_pose(self):
        exceeded = dynamic_join_envelope_failure(
            Pose2D(0.21, 0.0, 0.0), Pose2D(0.0, 0.0, 0.0), 0.2
        )
        nonfinite = dynamic_join_envelope_failure(
            Pose2D(float("nan"), 0.0, 0.0), Pose2D(0.0, 0.0, 0.0), 0.2
        )

        self.assertEqual(exceeded["fault_code"], "join_envelope_exceeded")
        self.assertTrue(exceeded["fail_closed"])
        self.assertEqual(nonfinite["fault_code"], "invalid_current_pose")
        self.assertTrue(nonfinite["fail_closed"])

    def test_adoption_zeros_installs_and_only_then_emits_reload(self):
        order = []
        update = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(Pose2D(0.02, 0.0), Pose2D(0.2, 0.0)),
            target_index=0,
            event_name="dynamic_route_adopted",
            event_fields={
                "effective_join_limit_m": 0.2,
                "route_kind": "axis_acquisition",
            },
        )
        node = None

        def callback(_update):
            order.append(("callback", node.waypoints, node.dynamic_join_pending))

        node = bare_follower(update, callback)
        node.publish_zero = lambda: order.append(("zero",))

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=10.0,
        ):
            outcome = node._refresh_dynamic_route(Pose2D(0.02, 0.0))

        self.assertEqual(outcome, "adopted")
        self.assertEqual(order[0], ("zero",))
        self.assertEqual(order[1][0], "callback")
        self.assertEqual(order[1][1], update.waypoints)
        self.assertTrue(order[1][2])
        self.assertEqual(node.target_index, 0)

    def test_stop_zeros_before_callback_and_callback_failure_is_fail_closed(self):
        order = []
        update = RouteUpdate(
            kind=RouteUpdateKind.STOP,
            reason="route withdrawn",
            event_name="dynamic_route_withdrawn",
        )

        def callback(_update):
            order.append("callback")
            raise RuntimeError("log sink unavailable")

        node = bare_follower(update, callback)
        node.publish_zero = lambda: order.append("zero")

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=10.0,
        ):
            outcome = node._refresh_dynamic_route(Pose2D(0.0, 0.0))

        self.assertEqual(outcome, "stopped")
        self.assertEqual(order, ["zero", "callback"])
        self.assertEqual(
            node.latest_stop_details["fault_code"],
            "route_event_callback_exception",
        )
        self.assertTrue(node.latest_stop_details["fail_closed"])


if __name__ == "__main__":
    unittest.main()
