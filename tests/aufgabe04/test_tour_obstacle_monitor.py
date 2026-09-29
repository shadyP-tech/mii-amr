"""The tour monitor stops early without steering or weakening existing gates."""

from dataclasses import replace
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.control.waypoint_controller import ControllerConfig
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.config import FollowerConfig
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components import tour_obstacle_monitor as monitor
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.motion_cycle_guard import (
    MotionCycleGuardRuntimeMixin, MotionCycleGuardAction,
)


class TourMonitorGeometryTest(unittest.TestCase):
    def hit(self, points, *, waypoints=None):
        return monitor.forward_corridor_hit(points, pose=Pose2D(0., 0., 0.),
            waypoints=waypoints or [Pose2D(0., 0., 0.), Pose2D(2., 0., 0.)],
            target_index=1, corridor_radius_m=.225)

    def test_early_obstacle_within_swept_radius_is_seen_before_lidar_stop(self):
        self.assertEqual(self.hit([(.6, .2)]), {"x_m": .6, "y_m": .2})
        self.assertIsNone(self.hit([(.6, .226)]))
        self.assertIsNone(self.hit([(.81, 0.)]))
        self.assertIsNone(self.hit([(-.05, 0.)]))

    def test_obstacle_beyond_goal_is_not_part_of_route(self):
        route = [Pose2D(0., 0., 0.), Pose2D(.4, 0., 0.)]
        self.assertIsNone(self.hit([(.41, 0.)], waypoints=route))
        self.assertIsNotNone(self.hit([(.39, .1)], waypoints=route))

    def test_corridor_follows_remaining_polyline_turn(self):
        route = [Pose2D(0., 0., 0.), Pose2D(.3, 0., 0.), Pose2D(.3, 1., 0.)]
        self.assertIsNotNone(self.hit([(.3, .4)], waypoints=route))
        self.assertIsNone(self.hit([(.3, .51)], waypoints=route))

    def test_projection_uses_scanner_pose_and_discards_unknown_rays(self):
        points = list(monitor.scan_points_odom({"ranges": [1., None], "angle_min": 0., "angle_increment": .1},
            {"x_m": .04, "y_m": .02, "yaw_rad": math.pi/2}))
        self.assertAlmostEqual(points[0][0], .04)
        self.assertAlmostEqual(points[0][1], 1.02)
        self.assertEqual(len(points), 1)

    def test_first_scan_holds_duplicate_never_confirms_second_distinct_scan_stops(self):
        kwargs = {"blocked_point": {"x_m": .5, "y_m": 0.}, "evidence": {"scan_stamp_sec": 10.}}
        first, pending = monitor.classify_monitor_scan(previous_blocked_stamp=None, stamp_sec=10., **kwargs)
        self.assertEqual(first.action, "hold")
        same, pending = monitor.classify_monitor_scan(previous_blocked_stamp=pending, stamp_sec=10., **kwargs)
        self.assertEqual(same.action, "hold")
        second, _ = monitor.classify_monitor_scan(previous_blocked_stamp=pending, stamp_sec=10.1, **kwargs)
        self.assertEqual(second.action, "stop")
        self.assertEqual(second.details["reason"], monitor.TOUR_ROUTE_BLOCKED)
        self.assertEqual(second.details["confirmed_distinct_scans"], 2)

    def test_clear_scan_resets_pending_and_late_scan_starts_new_confirmation(self):
        clear, pending = monitor.classify_monitor_scan(previous_blocked_stamp=10., stamp_sec=10.1,
            blocked_point=None, evidence={})
        self.assertEqual(clear.action, "clear")
        self.assertIsNone(pending)
        late, pending = monitor.classify_monitor_scan(previous_blocked_stamp=10., stamp_sec=10.6,
            blocked_point={"x_m": .5, "y_m": 0.}, evidence={})
        self.assertEqual(late.action, "hold")
        self.assertEqual(pending, 10.6)


class TourMonitorRuntimeTest(unittest.TestCase):
    def test_monitor_config_is_opt_in_and_keeps_existing_stop_distances(self):
        base = FollowerConfig(controller=ControllerConfig())
        self.assertFalse(base.stored_pose_tour_obstacle_monitor)
        configured = replace(base, stored_pose_tour_obstacle_monitor=True,
            stored_pose_tour_robot_radius_m=.1, stored_pose_tour_scan_frame="laser")
        self.assertEqual(configured.min_obstacle_distance_m, .20)
        self.assertEqual(configured.front_obstacle_slow_distance_m, .38)
        with self.assertRaises(ValueError):
            replace(base, stored_pose_tour_obstacle_monitor=True)

    def test_zero_hold_or_stop_precedes_any_command_admission(self):
        for action in ("hold", "stop"):
            with self.subTest(action=action):
                node = NS(follower_config=NS(stored_pose_tour_obstacle_monitor=True),
                    _tour_obstacle_monitor_decision=Mock(return_value=monitor.TourMonitorDecision(action,
                        {"reason": monitor.TOUR_ROUTE_BLOCKED})),
                    publish_repeated_zero=Mock(), _hold_zero_control_period=Mock(),
                    _motion_command_admission_decision=Mock(side_effect=AssertionError("must stay stopped")))
                result = MotionCycleGuardRuntimeMixin._motion_cycle_guard_decision(node,
                    Pose2D(0., 0., 0.), NS(), None, .1)
                self.assertEqual(result.action, MotionCycleGuardAction.RETRY if action == "hold" else MotionCycleGuardAction.STOP)
                node.publish_repeated_zero.assert_called_once()
                node._motion_command_admission_decision.assert_not_called()

    def test_missing_exact_transform_stops_with_nonreplannable_sensor_fault(self):
        node = NS(follower_config=NS(stored_pose_tour_obstacle_monitor=True,
            stored_pose_tour_scan_frame="laser", stored_pose_tour_robot_radius_m=.1),
            current_route_kind="admitted_candidate_pose", odom_execution_context=object(),
            latest_scan=NS(header=NS(frame_id="laser", stamp=NS(sec=100, nanosec=0))),
            get_clock=lambda: NS(now=lambda: NS(nanoseconds=100_010_000_000)),
            runtime_config=NS(odom_frame="odom", base_frame="base"),
            tf_buffer=NS(lookup_transform=Mock(side_effect=RuntimeError("no exact TF"))))
        with patch.object(monitor, "Time", NS(from_msg=lambda value: value)), \
             patch.object(monitor, "Duration", lambda **kwargs: kwargs):
            result = monitor.TourObstacleMonitorRuntimeMixin._tour_obstacle_monitor_decision(
                node, Pose2D(0., 0., 0.), NS(command=NS(linear_x_mps=.15)))
        self.assertEqual(result.action, "stop")
        self.assertEqual(result.details["reason"], monitor.TOUR_MONITOR_INVALID)
        self.assertEqual(result.details["fault_code"], "stored_pose_tour_monitor_invalid")


if __name__ == "__main__":
    unittest.main()
