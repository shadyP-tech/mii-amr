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
    viewpoint_sampling_target_timeout_failure,
)
from scripts.aufgabe04.navigation.transient_blockage_policy import (
    CLEARANCE_LIMITED_MOTION_FLOOR,
    classify_linear_command,
)
from scripts.aufgabe04.navigation.waypoint_controller import (
    CertifiedCornerTransitionLatch,
    ControllerConfig,
    VelocityCommand,
    compute_join_anchor_command,
    compute_start_egress_vertex_command,
    compute_waypoint_command,
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
    node.last_progress_mode = None
    node.progress_heading_modes_seen = set()
    node.progress_heading_error_by_mode = {}
    node.last_progress_at = 0.0
    node.last_pose = None
    node.latest_stop_details = None
    node.dynamic_join_pending = False
    node.dynamic_join_limit_m = None
    node.start_egress_lock_index = None
    node.start_egress_reverse = False
    node.start_egress_reverse_until_index = None
    node.start_egress_forward_alignment_index = None
    node.certified_corner_latch = None
    node._last_certified_corner_phase = None
    node.current_route_kind = "axis_acquisition"
    node.reverse_staging = False
    node.axis_acquisition_hold_started_at = None
    node.axis_acquisition_target_revision = None
    node.viewpoint_sampling_started_at = None
    node.viewpoint_sampling_target_started_at = None
    node.viewpoint_sampling_target_revision = None
    return node


class DynamicFollowerHandoffTest(unittest.TestCase):
    def test_certified_corner_stop_zeroes_before_diagnostics_and_traces_pose(self):
        events = []
        records = []
        pose = Pose2D(0.09, 0.0, 0.0)
        odom_pose = Pose2D(0.05, 0.0, 0.0)
        node = bare_follower(RouteUpdate(kind=RouteUpdateKind.UNCHANGED), None)
        node.waypoints = (
            Pose2D(0.0, 0.0, math.nan),
            Pose2D(0.05, 0.0, math.nan),
            Pose2D(0.05, 0.05, math.nan),
        )
        node.current_route_kind = "stand_discovery_corridor"
        node.target_index = 1
        node.certified_corner_latch = CertifiedCornerTransitionLatch(1)
        node.certified_static_start_pending = False
        node.distance_estimate_m = 0.0
        node.motion_published = True
        node.last_pose = pose
        node._wait_for_initial_runtime_inputs = lambda _started_at: ""
        node._drain_runtime_callbacks = lambda: None
        node._safety_failure = lambda: ""
        node._current_pose_lookup_with_stale_recovery = lambda: SimpleNamespace(
            pose=pose
        )
        node._refresh_dynamic_route = lambda _pose: ""
        node._latest_odom_pose = lambda: odom_pose
        node.publish_zero = lambda: events.append("zero")
        node.publish_repeated_zero = lambda: events.append("repeated_zero")
        node._log_certified_corner_phase = lambda _step: events.append("log")
        route_check = node._execution_route_check

        def checked_route(live_pose, step):
            events.append("route_check")
            return route_check(live_pose, step)

        class CaptureWriter:
            def append(self, record):
                events.append("trace")
                records.append(record)

        node._execution_route_check = checked_route
        node.controller_trace_writer = CaptureWriter()

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.rclpy",
            SimpleNamespace(ok=lambda: True),
        ):
            result = node.run()

        self.assertEqual(result.status, "stopped")
        self.assertEqual(
            result.stop_reason,
            "certified corner hard tolerance exceeded",
        )
        zero_index = events.index("zero")
        self.assertLess(zero_index, events.index("log"))
        self.assertLess(zero_index, events.index("route_check"))
        self.assertLess(zero_index, events.index("trace"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].event, "certified_corner_stop")
        self.assertEqual(records[0].map_pose, pose)
        self.assertEqual(records[0].odom_pose, odom_pose)
        self.assertTrue(records[0].fail_closed)

    def test_discovery_replan_adopts_reverse_egress_without_phase_restart(self):
        update = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(
                Pose2D(-0.86, -0.46, math.nan),
                Pose2D(-0.74, -0.46, math.nan),
                Pose2D(-0.69, -0.46, math.nan),
                Pose2D(-0.69, -0.41, 0.0),
            ),
            target_index=0,
            event_fields={
                "effective_join_limit_m": 0.03,
                "route_kind": "stand_discovery_corridor",
                "start_egress_vertex_lock": True,
                "start_egress_waypoint_index": 1,
                "start_egress_continuous_clearance_validated": True,
                "start_egress_motion": "reverse",
                "start_egress_reverse_until_waypoint_index": 2,
                "start_egress_forward_alignment_waypoint_index": 3,
            },
        )
        node = bare_follower(update, None)
        node.current_route_kind = "stand_discovery_corridor"
        node.publish_zero = lambda: None

        result = node._refresh_dynamic_route(
            Pose2D(-0.86, -0.46, math.pi)
        )

        self.assertEqual(result, "adopted")
        self.assertEqual(node.current_route_kind, "stand_discovery_corridor")
        self.assertEqual(node.start_egress_lock_index, 1)
        self.assertTrue(node.start_egress_reverse)
        self.assertEqual(node.start_egress_reverse_until_index, 2)
        self.assertEqual(node.start_egress_forward_alignment_index, 3)
        self.assertFalse(node.reverse_staging)

    def test_reverse_egress_without_forward_handoff_certificate_is_rejected(self):
        update = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(
                Pose2D(-0.86, -0.46, math.nan),
                Pose2D(-0.74, -0.46, math.nan),
                Pose2D(-0.69, -0.46, math.nan),
                Pose2D(-0.69, -0.41, 0.0),
            ),
            target_index=0,
            event_fields={
                "effective_join_limit_m": 0.03,
                "route_kind": "stand_discovery_corridor",
                "start_egress_vertex_lock": True,
                "start_egress_waypoint_index": 1,
                "start_egress_continuous_clearance_validated": True,
                "start_egress_motion": "reverse",
            },
        )
        node = bare_follower(update, None)
        node.current_route_kind = "stand_discovery_corridor"
        node.publish_zero = lambda: None

        self.assertEqual(
            node._refresh_dynamic_route(Pose2D(-0.86, -0.46, math.pi)),
            "stopped",
        )
        self.assertEqual(
            node.latest_stop_details["reason"],
            "dynamic route reverse-egress handoff certificate is malformed",
        )
        self.assertTrue(node.latest_stop_details["fail_closed"])

    def test_reverse_egress_uses_rear_sector_not_front_blocker(self):
        node = bare_follower(
            RouteUpdate(kind=RouteUpdateKind.UNCHANGED),
            None,
        )
        ranges = [1.0] * 360
        ranges[180] = 0.19
        node.latest_scan = SimpleNamespace(
            ranges=ranges,
            angle_min=-math.pi,
            angle_increment=2.0 * math.pi / 360.0,
            range_min=0.02,
            range_max=12.0,
        )
        node.blockage_recovery_provider = lambda *_args: None
        node.start_egress_reverse = True

        self.assertEqual(node._obstacle_failure(), "")

        node.start_egress_reverse = False
        self.assertEqual(node._obstacle_failure(), "obstacle too close")
        self.assertEqual(
            node.latest_stop_details["front_clearance"]["source"],
            "front_sector",
        )

    def test_reverse_egress_crosses_later_anchor_before_forward_handoff(self):
        """Regression for the 20260805T120749Z recovery-mode discontinuity."""

        waypoints = (
            Pose2D(-0.8215194236642129, -0.515331012111416, math.nan),
            Pose2D(-0.7449999999999997, -0.565, math.nan),
            Pose2D(-0.6949999999999998, -0.565, math.nan),
            Pose2D(-0.6949999999999998, -0.5149999999999999, math.nan),
        )
        update = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=waypoints,
            target_index=0,
            event_fields={
                "effective_join_limit_m": 0.03,
                "route_kind": "stand_discovery_corridor",
                "start_egress_vertex_lock": True,
                "start_egress_waypoint_index": 1,
                "start_egress_continuous_clearance_validated": True,
                "start_egress_motion": "reverse",
                "start_egress_reverse_until_waypoint_index": 2,
                "start_egress_forward_alignment_waypoint_index": 3,
            },
        )
        node = bare_follower(update, None)
        node.current_route_kind = "stand_discovery_corridor"
        node.publish_zero = lambda: None
        self.assertEqual(
            node._refresh_dynamic_route(
                Pose2D(waypoints[0].x_m, waypoints[0].y_m, math.pi)
            ),
            "adopted",
        )
        node.dynamic_join_pending = False

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            side_effect=(10.0, 10.1, 11.0, 11.1),
        ):
            first_vertex = node._start_egress_command(
                Pose2D(waypoints[1].x_m, waypoints[1].y_m, 2.565848),
                node.follower_config.controller,
            )
            second_vertex = node._start_egress_command(
                Pose2D(waypoints[2].x_m, waypoints[2].y_m, math.pi),
                node.follower_config.controller,
            )

        self.assertIsNone(first_vertex)
        self.assertIsNone(second_vertex)
        self.assertIsNone(node.start_egress_lock_index)
        self.assertFalse(node.start_egress_reverse)
        self.assertEqual(node.target_index, 2)
        self.assertEqual(node.start_egress_forward_alignment_index, 3)

        aligning = node._reverse_egress_forward_alignment_command(
            Pose2D(waypoints[2].x_m, waypoints[2].y_m, math.pi),
            node.follower_config.controller,
        )
        self.assertEqual(aligning.target_index, 3)
        self.assertEqual(
            aligning.progress_mode,
            "reverse_egress_forward_alignment",
        )
        self.assertEqual(node.start_egress_forward_alignment_index, 3)

        handoff = node._reverse_egress_forward_alignment_command(
            Pose2D(waypoints[2].x_m, waypoints[2].y_m, math.pi / 2.0),
            node.follower_config.controller,
        )
        self.assertEqual(
            handoff.progress_mode,
            "reverse_egress_forward_handoff",
        )
        self.assertIsNone(node.start_egress_forward_alignment_index)

    def test_discovery_route_node_uses_corner_contract_only_for_material_bend(self):
        node = bare_follower(RouteUpdate(kind=RouteUpdateKind.UNCHANGED), None)
        node.current_route_kind = "stand_discovery_corridor"
        node.waypoints = (
            Pose2D(-0.4405799284256718, -0.6511976219870571, math.nan),
            Pose2D(-0.44499999999999984, -0.615, math.nan),
            Pose2D(-0.49499999999999966, -0.565, math.nan),
        )
        node.target_index = 1
        incoming_heading = math.atan2(
            node.waypoints[1].y_m - node.waypoints[0].y_m,
            node.waypoints[1].x_m - node.waypoints[0].x_m,
        )

        decision = node._certified_corner_decision(
            Pose2D(node.waypoints[1].x_m, node.waypoints[1].y_m, incoming_heading),
            node.follower_config.controller,
        )

        self.assertEqual(decision.failure, "")
        self.assertEqual(decision.step.target_index, 1)
        self.assertEqual(decision.step.command.linear_x_mps, 0.0)
        self.assertEqual(
            decision.step.progress_mode,
            "certified_corner_alignment",
        )
        self.assertIsNotNone(node.certified_corner_latch)

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

    def test_terminal_heading_handoff_resets_path_progress_baseline_once(self):
        node = bare_follower(RouteUpdate(kind=RouteUpdateKind.UNCHANGED), None)

        self.assertEqual(
            node._progress_failure(
                0.105,
                0.021,
                1,
                1,
                0.0,
                True,
                "path_tracking",
            ),
            "",
        )
        self.assertEqual(
            node._progress_failure(
                0.076,
                0.726,
                1,
                1,
                8.01,
                True,
                "terminal_heading",
            ),
            "",
        )
        self.assertEqual(node.last_progress_at, 8.01)
        self.assertEqual(node.last_progress_mode, "terminal_heading")
        self.assertEqual(
            node._progress_failure(
                0.076,
                0.60,
                1,
                1,
                12.0,
                True,
                "terminal_heading",
            ),
            "",
        )

    def test_terminal_heading_mode_chatter_cannot_reset_watchdog_forever(self):
        node = bare_follower(RouteUpdate(kind=RouteUpdateKind.UNCHANGED), None)

        samples = (
            (0.0, "path_tracking", 0.02, ""),
            (4.0, "terminal_heading", 0.72, ""),
            (8.0, "path_tracking", 0.02, ""),
            (12.0, "terminal_heading", 0.72, ""),
            (16.1, "path_tracking", 0.02, "stuck no progress"),
        )
        for now, mode, heading_error, expected in samples:
            with self.subTest(now=now, mode=mode):
                self.assertEqual(
                    node._progress_failure(
                        0.076,
                        heading_error,
                        1,
                        1,
                        now,
                        True,
                        mode,
                    ),
                    expected,
                )

    def test_terminal_heading_reentry_uses_its_own_progress_baseline(self):
        node = bare_follower(RouteUpdate(kind=RouteUpdateKind.UNCHANGED), None)

        # Live retry 06 crossed the 3 cm position tolerance once, briefly
        # returned to point pursuit, and then re-entered terminal yaw.  The
        # small point-bearing error must not replace the terminal-yaw baseline.
        samples = (
            (0.0, "path_tracking", 0.56, ""),
            (1.0, "terminal_heading", 2.33, ""),
            (2.0, "path_tracking", 0.44, ""),
            (5.0, "terminal_heading", 2.10, ""),
            (10.0, "terminal_heading", 1.85, ""),
        )
        for now, mode, heading_error, expected in samples:
            with self.subTest(now=now, mode=mode):
                self.assertEqual(
                    node._progress_failure(
                        0.029,
                        heading_error,
                        1,
                        1,
                        now,
                        True,
                        mode,
                    ),
                    expected,
                )

        self.assertAlmostEqual(
            node.progress_heading_error_by_mode["path_tracking"], 0.44
        )
        self.assertAlmostEqual(
            node.progress_heading_error_by_mode["terminal_heading"], 1.85
        )
        self.assertEqual(node.last_progress_at, 10.0)

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

    def test_recorded_single_ray_low_speed_is_zero_hold_not_creep(self):
        node = object.__new__(SimpleWaypointFollowerNode)
        node.follower_config = FollowerConfig(controller=ControllerConfig())
        node.latest_front_clearance_details = None
        node.latest_scan = SimpleNamespace(
            ranges=(0.23400000417232513,),
            angle_min=0.0,
            angle_increment=0.01,
            range_min=0.12,
            range_max=3.5,
        )
        nominal_linear_mps = 0.04303463189017459

        scale = node._motion_clearance_linear_scale(nominal_linear_mps)
        effective_linear_mps = nominal_linear_mps * scale
        decision = classify_linear_command(
            nominal_linear_mps,
            effective_linear_mps,
            linear_motion_floor_mps=node.follower_config.linear_motion_floor_mps,
        )

        self.assertAlmostEqual(effective_linear_mps, 0.00812876317446179)
        self.assertEqual(
            node.latest_front_clearance_details["valid_sample_count"],
            1,
        )
        self.assertEqual(decision.reasons, (CLEARANCE_LIMITED_MOTION_FLOOR,))
        self.assertTrue(decision.stationary_confirmation_required)
        self.assertEqual(decision.output_linear_x_mps, 0.0)

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
        discovery = controller_config_for_route_kind(
            configured,
            "stand_discovery_corridor",
            physical_waypoint_tolerance_m=0.02,
            physical_goal_tolerance_m=0.03,
        )
        self.assertFalse(discovery.enforce_heading_corridor)
        self.assertTrue(discovery.exact_vertex_pursuit)
        self.assertEqual(discovery.goal_tolerance_m, 0.02)
        self.assertEqual(discovery.terminal_goal_tolerance_m, 0.03)
        detected_preapproach = controller_config_for_route_kind(
            configured,
            "detected_stand_preapproach",
            physical_waypoint_tolerance_m=0.02,
            physical_goal_tolerance_m=0.03,
        )
        self.assertFalse(detected_preapproach.enforce_heading_corridor)
        self.assertTrue(detected_preapproach.exact_vertex_pursuit)
        self.assertEqual(detected_preapproach.goal_tolerance_m, 0.02)
        self.assertEqual(
            detected_preapproach.terminal_goal_tolerance_m,
            0.03,
        )
        acquisition = controller_config_for_route_kind(
            configured,
            "axis_acquisition",
            viewpoint_sampling_goal_tolerance_m=0.01,
            viewpoint_sampling_heading_tolerance_rad=math.radians(5.0),
        )
        # Retry 08 reached its first acquisition pose with 0.224 rad heading
        # error.  Position was valid, but that heading was outside the camera
        # centering gate and must not start the stationary wait timer.
        self.assertEqual(acquisition.goal_tolerance_m, 0.01)
        self.assertAlmostEqual(
            acquisition.heading_tolerance_rad,
            math.radians(5.0),
        )
        self.assertTrue(
            controller_config_for_route_kind(
                configured,
                "synchronized_face_approach",
                reverse_staging=True,
                physical_waypoint_tolerance_m=0.02,
                physical_goal_tolerance_m=0.03,
            ).enforce_heading_corridor
        )
        physical = controller_config_for_route_kind(
            configured,
            "synchronized_face_approach",
            reverse_staging=True,
            physical_waypoint_tolerance_m=0.02,
            physical_goal_tolerance_m=0.03,
        )
        self.assertTrue(physical.reverse_staging)
        self.assertEqual(physical.goal_tolerance_m, 0.02)
        self.assertEqual(physical.terminal_goal_tolerance_m, 0.03)
        self.assertTrue(
            controller_config_for_route_kind(
                configured, "synchronized_viewpoint"
            ).enforce_heading_corridor
        )
        catalog = controller_config_for_route_kind(
            configured,
            "catalog_face_approach",
            physical_waypoint_tolerance_m=0.02,
            physical_goal_tolerance_m=0.03,
        )
        self.assertTrue(catalog.enforce_heading_corridor)
        self.assertEqual(catalog.goal_tolerance_m, 0.02)
        self.assertEqual(catalog.terminal_goal_tolerance_m, 0.03)

    def test_discovery_inspection_yaw_is_terminal_only_after_replan(self):
        configured = controller_config_for_route_kind(
            ControllerConfig(
                goal_tolerance_m=0.02,
                terminal_goal_tolerance_m=0.03,
                heading_tolerance_rad=0.25,
                stop_heading_error_rad=1.25,
                enforce_heading_corridor=True,
                exact_vertex_pursuit=True,
            ),
            "stand_discovery_corridor",
            physical_waypoint_tolerance_m=0.02,
            physical_goal_tolerance_m=0.03,
        )
        waypoints = (
            Pose2D(-1.195, -0.065, math.nan),
            Pose2D(-1.595, -0.015, 0.0),
        )
        path_heading = math.atan2(
            waypoints[1].y_m - waypoints[0].y_m,
            waypoints[1].x_m - waypoints[0].x_m,
        )

        align_to_path = compute_waypoint_command(
            Pose2D(-1.205, -0.062, 0.24),
            waypoints,
            1,
            configured,
        )
        moderate_path_error = compute_waypoint_command(
            Pose2D(
                waypoints[0].x_m,
                waypoints[0].y_m,
                path_heading - 0.50,
            ),
            waypoints,
            1,
            configured,
        )
        inside_path_alignment_tolerance = compute_waypoint_command(
            Pose2D(
                waypoints[0].x_m,
                waypoints[0].y_m,
                path_heading - 0.24,
            ),
            waypoints,
            1,
            configured,
        )
        translate = compute_waypoint_command(
            Pose2D(waypoints[0].x_m, waypoints[0].y_m, path_heading),
            waypoints,
            1,
            configured,
        )
        terminal_turn = compute_waypoint_command(
            Pose2D(waypoints[1].x_m, waypoints[1].y_m, path_heading),
            waypoints,
            1,
            configured,
        )
        completed = compute_waypoint_command(
            Pose2D(waypoints[1].x_m, waypoints[1].y_m, 0.0),
            waypoints,
            1,
            configured,
        )

        self.assertEqual(
            align_to_path.progress_mode,
            "exact_vertex_alignment",
        )
        self.assertEqual(align_to_path.command.linear_x_mps, 0.0)
        self.assertGreater(align_to_path.command.angular_z_radps, 0.0)
        self.assertEqual(
            moderate_path_error.progress_mode,
            "exact_vertex_alignment",
        )
        self.assertEqual(moderate_path_error.command.linear_x_mps, 0.0)
        self.assertGreater(moderate_path_error.command.angular_z_radps, 0.0)
        self.assertEqual(
            inside_path_alignment_tolerance.progress_mode,
            "path_tracking",
        )
        self.assertGreater(
            inside_path_alignment_tolerance.command.linear_x_mps,
            0.0,
        )
        self.assertEqual(translate.progress_mode, "path_tracking")
        self.assertGreater(translate.command.linear_x_mps, 0.0)
        self.assertEqual(terminal_turn.progress_mode, "terminal_heading")
        self.assertEqual(terminal_turn.command.linear_x_mps, 0.0)
        self.assertFalse(terminal_turn.reached_goal)
        self.assertTrue(completed.reached_goal)

    def test_recorded_detected_stand_preapproach_yaw_is_terminal_only(self):
        start = Pose2D(
            0.951971715233173,
            -0.04319279069223947,
            0.1717540796057199,
        )
        target = Pose2D(0.755, 0.035, -0.3583366136065122)
        waypoints = (
            Pose2D(start.x_m, start.y_m, math.nan),
            target,
        )
        configured = controller_config_for_route_kind(
            ControllerConfig(
                goal_tolerance_m=0.02,
                terminal_goal_tolerance_m=0.03,
                heading_tolerance_rad=0.25,
                stop_heading_error_rad=1.25,
                enforce_heading_corridor=True,
            ),
            "detected_stand_preapproach",
            physical_waypoint_tolerance_m=0.02,
            physical_goal_tolerance_m=0.03,
        )
        travel_heading = math.atan2(
            target.y_m - start.y_m,
            target.x_m - start.x_m,
        )

        align_to_vertex = compute_waypoint_command(
            start,
            waypoints,
            1,
            configured,
        )
        forward_transit = compute_waypoint_command(
            Pose2D(start.x_m, start.y_m, travel_heading),
            waypoints,
            1,
            configured,
        )
        terminal_turn = compute_waypoint_command(
            Pose2D(target.x_m, target.y_m, travel_heading),
            waypoints,
            1,
            configured,
        )
        completed = compute_waypoint_command(
            target,
            waypoints,
            1,
            configured,
        )

        self.assertFalse(configured.enforce_heading_corridor)
        self.assertTrue(configured.exact_vertex_pursuit)
        self.assertEqual(
            align_to_vertex.progress_mode,
            "exact_vertex_alignment",
        )
        self.assertNotEqual(align_to_vertex.progress_mode, "heading_corridor")
        self.assertEqual(align_to_vertex.command.linear_x_mps, 0.0)
        self.assertEqual(forward_transit.progress_mode, "path_tracking")
        self.assertNotEqual(forward_transit.progress_mode, "heading_corridor")
        self.assertGreater(forward_transit.command.linear_x_mps, 0.0)
        self.assertEqual(terminal_turn.progress_mode, "terminal_heading")
        self.assertEqual(terminal_turn.command.linear_x_mps, 0.0)
        self.assertFalse(terminal_turn.reached_goal)
        self.assertTrue(completed.reached_goal)
        self.assertEqual(completed.command, VelocityCommand(0.0, 0.0))

    def test_exact_vertex_alignment_has_its_own_progress_baseline(self):
        node = bare_follower(RouteUpdate(kind=RouteUpdateKind.UNCHANGED), None)

        self.assertEqual(
            node._progress_failure(
                0.04,
                0.02,
                2,
                2,
                0.0,
                True,
                "path_tracking",
            ),
            "",
        )
        self.assertEqual(
            node._progress_failure(
                0.04,
                0.46,
                2,
                2,
                7.9,
                True,
                "exact_vertex_alignment",
            ),
            "",
        )
        self.assertEqual(node.last_progress_at, 7.9)

    def test_retry08_acquisition_pose_keeps_correcting_before_hold(self):
        pose = Pose2D(
            -1.0190934638169666,
            -0.4629104883592975,
            0.27879300289087416,
        )
        target = Pose2D(
            -0.9434690364310175,
            -0.4415727854644216,
            0.0552268781046199,
        )
        waypoints = (Pose2D(-1.3974097814674362, -0.46666806006977657), target)

        old_step = compute_waypoint_command(
            pose,
            waypoints,
            1,
            ControllerConfig(),
        )
        corrected_step = compute_waypoint_command(
            pose,
            waypoints,
            1,
            controller_config_for_route_kind(
                ControllerConfig(),
                "axis_acquisition",
                viewpoint_sampling_goal_tolerance_m=0.03,
                viewpoint_sampling_heading_tolerance_rad=math.radians(5.0),
            ),
        )

        self.assertTrue(old_step.reached_goal)
        self.assertFalse(corrected_step.reached_goal)
        self.assertEqual(corrected_step.progress_mode, "path_tracking")
        self.assertGreater(corrected_step.distance_to_target_m, 0.03)
        self.assertNotEqual(corrected_step.command.linear_x_mps, 0.0)

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

    def test_fresh_admission_pose_outside_replacement_join_is_rejected(self):
        update = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(Pose2D(0.0, 0.0), Pose2D(0.2, 0.0)),
            target_index=0,
            event_fields={
                "effective_join_limit_m": 0.03,
                "route_kind": "axis_acquisition",
            },
        )
        node = bare_follower(update, None)
        node.publish_zero = lambda: None
        original_waypoints = node.waypoints

        result = node._refresh_dynamic_route(Pose2D(0.03154134331426062, 0.0))

        self.assertEqual(result, "stopped")
        self.assertEqual(node.waypoints, original_waypoints)
        self.assertEqual(
            node.latest_stop_details["fault_code"],
            "join_envelope_exceeded",
        )
        self.assertEqual(
            node.latest_stop_details["source"],
            "dynamic_route_admission",
        )

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
            viewpoint_sampling_target_timeout_failure(
                route_kind="viewpoint_sampling",
                target_started_at=20.0,
                now_monotonic=50.0,
                timeout_sec=30.0,
            ),
            "viewpoint_sampling_target_timeout",
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
            target_revision=1,
            event_fields={
                "effective_join_limit_m": 0.2,
                "route_kind": "axis_acquisition",
            },
        )
        node = bare_follower(same_axis, None)
        node.axis_acquisition_hold_started_at = 4.0
        node.axis_acquisition_target_revision = 1
        node.publish_zero = lambda: None
        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=10.0,
        ):
            self.assertEqual(
                node._refresh_dynamic_route(Pose2D(0.02, 0.0)), "adopted"
            )
        self.assertEqual(node.axis_acquisition_hold_started_at, 4.0)
        self.assertEqual(node.axis_acquisition_target_revision, 1)
        self.assertIsNone(node.viewpoint_sampling_started_at)

        next_acquisition_ray = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(Pose2D(0.02, 0.0), Pose2D(0.25, 0.15)),
            target_index=0,
            target_revision=2,
            event_fields={
                "effective_join_limit_m": 0.2,
                "route_kind": "axis_acquisition",
            },
        )
        node.waypoint_provider = lambda _pose: next_acquisition_ray
        node.initial_route_refresh_pending = True
        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=11.0,
        ):
            self.assertEqual(
                node._refresh_dynamic_route(Pose2D(0.02, 0.0)), "adopted"
            )
        self.assertIsNone(node.axis_acquisition_hold_started_at)
        self.assertEqual(node.axis_acquisition_target_revision, 2)

        sampling = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(Pose2D(0.02, 0.0), Pose2D(0.25, 0.1)),
            target_index=0,
            target_revision=2,
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
        self.assertEqual(node.viewpoint_sampling_target_started_at, 12.0)
        self.assertEqual(node.viewpoint_sampling_target_revision, 2)

        node.initial_route_refresh_pending = True
        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=22.0,
        ):
            self.assertEqual(
                node._refresh_dynamic_route(Pose2D(0.02, 0.0)), "adopted"
            )
        self.assertEqual(node.viewpoint_sampling_started_at, 12.0)
        self.assertEqual(node.viewpoint_sampling_target_started_at, 12.0)

        newer_sampling_target = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(Pose2D(0.02, 0.0), Pose2D(0.27, 0.12)),
            target_index=0,
            target_revision=3,
            event_fields={
                "effective_join_limit_m": 0.2,
                "route_kind": "viewpoint_sampling",
            },
        )
        node.waypoint_provider = lambda _pose: newer_sampling_target
        node.initial_route_refresh_pending = True
        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.time.monotonic",
            return_value=25.0,
        ):
            self.assertEqual(
                node._refresh_dynamic_route(Pose2D(0.02, 0.0)), "adopted"
            )
        self.assertEqual(node.viewpoint_sampling_started_at, 12.0)
        self.assertEqual(node.viewpoint_sampling_target_started_at, 25.0)
        self.assertEqual(node.viewpoint_sampling_target_revision, 3)

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
        self.assertIsNone(node.viewpoint_sampling_target_started_at)
        self.assertIsNone(node.viewpoint_sampling_target_revision)
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
