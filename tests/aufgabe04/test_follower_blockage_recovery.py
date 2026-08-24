from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.runtime import (
    FollowerConfig,
    PoseLookupResult,
    SimpleWaypointFollowerNode,
)
from scripts.aufgabe04.navigation.coverage.transient_blockage_admission import (
    StationaryBlockageAdmission,
)
from scripts.aufgabe04.navigation.coverage.transient_blockage_policy import (
    PersistentObstacleConfig,
    StationaryFrontSectorSample,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import ControllerConfig


def _bare_node() -> SimpleWaypointFollowerNode:
    node = object.__new__(SimpleWaypointFollowerNode)
    node.follower_config = FollowerConfig(controller=ControllerConfig())
    node.latest_stop_details = None
    node.latest_front_clearance_details = None
    node.controller_trace_writer = None
    node.controller_route_revision = 0
    node.current_route_kind = "stand_discovery_corridor"
    node.dynamic_join_pending = False
    node.start_egress_reverse = False
    node.start_egress_forward_alignment_index = None
    node.start_egress_lock_index = None
    node.last_progress_distance_m = math.inf
    node.last_progress_heading_error_rad = math.inf
    node.last_progress_target_index = None
    node.last_progress_pursuit_index = None
    node.last_progress_mode = None
    node.progress_heading_modes_seen = set()
    node.progress_heading_error_by_mode = {}
    node.last_progress_at = 0.0
    node.publish_repeated_zero = Mock()
    node.publish_zero = Mock()
    node._cmd_vel_ownership_failure = Mock(return_value="")
    return node


class FollowerBlockageRecoveryTest(unittest.TestCase):
    def test_replan_uses_confirmed_stop_pose_and_fresh_post_plan_pose(self):
        node = _bare_node()
        trigger_pose = Pose2D(-0.86, -0.46, math.pi)
        confirmed_pose = Pose2D(-0.858, -0.461, math.pi)
        post_plan_pose = Pose2D(-0.856, -0.462, math.pi)
        confirmation = {
            "status": "persistent_obstacle_confirmed",
            "stationary_obstacle_confirmation": {
                "confirmed": True,
                "fail_closed": False,
                "distinct_sample_count": 3,
                "thresholds": {"min_distinct_samples": 3},
            },
        }
        node._confirm_stationary_blockage = Mock(
            return_value=StationaryBlockageAdmission(
                "confirmed",
                confirmed_pose,
                {
                    "source": "front_sector",
                    "nearest_valid_range_m": 0.234,
                    "nearest_valid_bearing_rad": 0.08,
                },
                confirmation,
            )
        )
        provider = Mock(
            return_value=RouteUpdate(
                kind=RouteUpdateKind.ADOPT,
                waypoints=(confirmed_pose, Pose2D(-0.70, -0.46)),
                target_index=0,
                event_fields={"effective_join_limit_m": 0.03},
            )
        )
        node.blockage_recovery_provider = provider
        node._post_replan_admission_pose = Mock(
            return_value=PoseLookupResult(post_plan_pose)
        )
        node._refresh_dynamic_route = Mock(return_value="adopted")

        result = node._attempt_blockage_recovery(
            trigger_pose,
            "clearance-limited motion floor",
            {"front_clearance": {"source": "front_sector"}},
        )

        self.assertEqual(result, "adopted")
        self.assertEqual(provider.call_args.args[0], confirmed_pose)
        self.assertEqual(node._refresh_dynamic_route.call_args.args[0], post_plan_pose)
        update = node.queued_route_update
        self.assertEqual(
            update.event_fields["planning_stop_pose"]["x_m"],
            confirmed_pose.x_m,
        )
        self.assertEqual(
            update.event_fields["post_plan_admission_pose"]["x_m"],
            post_plan_pose.x_m,
        )
        self.assertTrue(update.event_fields["post_plan_runtime_revalidated"])

    def test_unconfirmed_single_scan_never_calls_planner(self):
        node = _bare_node()
        node.blockage_recovery_provider = Mock()
        node._confirm_stationary_blockage = Mock(
            return_value=StationaryBlockageAdmission(
                "failed",
                None,
                None,
                {
                    "status": "stationary_blockage_unconfirmed",
                    "stationary_obstacle_confirmation": {
                        "confirmed": False,
                        "distinct_sample_count": 1,
                    },
                },
            )
        )

        result = node._attempt_blockage_recovery(
            Pose2D(0.0, 0.0, 0.0),
            "obstacle too close",
            {},
        )

        self.assertEqual(result, "stopped")
        node.blockage_recovery_provider.assert_not_called()
        self.assertEqual(
            node.latest_stop_details["fault_code"],
            "stationary_blockage_unconfirmed",
        )

    def test_ambiguous_cmd_vel_owner_blocks_confirmation_and_planning(self):
        node = _bare_node()
        node._cmd_vel_ownership_failure = Mock(
            return_value="ambiguous cmd_vel ownership"
        )
        node._confirm_stationary_blockage = Mock()
        node.blockage_recovery_provider = Mock()

        result = node._attempt_blockage_recovery(
            Pose2D(0.0, 0.0, 0.0),
            "obstacle too close",
            {"front_clearance": {"source": "front_sector"}},
        )

        self.assertEqual(result, "stopped")
        node._confirm_stationary_blockage.assert_not_called()
        node.blockage_recovery_provider.assert_not_called()
        self.assertEqual(
            node.latest_stop_details["fault_code"],
            "cmd_vel_ownership_ambiguous_before_replan",
        )
        self.assertTrue(node.latest_stop_details["fail_closed"])

    def test_post_plan_admission_rechecks_cmd_vel_ownership(self):
        node = _bare_node()
        node._cmd_vel_ownership_failure = Mock(
            return_value="ambiguous cmd_vel ownership"
        )
        node._drain_runtime_callbacks = Mock()
        node._freshness_failure = Mock()
        node._current_pose_lookup_with_stale_recovery = Mock()

        admission = node._post_replan_admission_pose()

        self.assertIsNone(admission.pose)
        self.assertEqual(
            admission.details["fault_code"],
            "cmd_vel_ownership_ambiguous_after_replan",
        )
        node._freshness_failure.assert_not_called()
        node._current_pose_lookup_with_stale_recovery.assert_not_called()

    def test_stationary_confirmation_collects_three_post_stop_scans(self):
        node = _bare_node()
        node.follower_config = FollowerConfig(
            controller=ControllerConfig(),
            persistent_obstacle_config=PersistentObstacleConfig(
                min_distinct_samples=3,
                min_front_range_m=0.12,
                max_front_range_m=0.38,
            ),
        )
        node.latest_scan_receipt = 99.0

        class Clock:
            value = 100.0

            def tick(self):
                result = self.value
                self.value += 0.05
                return result

        clock = Clock()

        def service(_timeout_sec):
            node.latest_scan_receipt = clock.value

        def sample():
            pose = Pose2D(-0.86, -0.46, math.pi)
            return (
                StationaryFrontSectorSample(
                    timestamp_sec=node.latest_scan_receipt,
                    front_range_m=0.234,
                    front_bearing_rad=0.08,
                    map_pose=pose,
                    odom_pose=Pose2D(0.0, 0.0, 0.0),
                ),
                {
                    "source": "front_sector",
                    "nearest_valid_range_m": 0.234,
                    "nearest_valid_bearing_rad": 0.08,
                },
            )

        node._service_or_wait_for_callbacks = service
        node._stationary_front_sample = sample
        fake_rclpy = SimpleNamespace(ok=lambda: True)
        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            fake_rclpy,
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
            side_effect=clock.tick,
        ):
            admission = node._confirm_stationary_blockage()

        self.assertEqual(admission.status, "confirmed")
        evidence = admission.evidence["stationary_obstacle_confirmation"]
        self.assertEqual(evidence["distinct_sample_count"], 3)
        self.assertAlmostEqual(evidence["median_front_range_m"], 0.234)

    def test_stationary_sample_binds_front_ray_to_map_and_odom(self):
        node = _bare_node()
        node.latest_scan_receipt = 42.0
        node.latest_odom_receipt = 42.0
        node.latest_scan = SimpleNamespace(
            ranges=(0.8, 0.234, 0.8),
            angle_min=-0.1,
            angle_increment=0.1,
            range_min=0.12,
            range_max=3.5,
        )
        node.latest_odom = SimpleNamespace(
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=1.0, y=2.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            )
        )
        node._freshness_failure = Mock(return_value="")
        node._current_pose_lookup_with_stale_recovery = Mock(
            return_value=PoseLookupResult(Pose2D(-0.86, -0.46, math.pi))
        )

        sample, details = node._stationary_front_sample()

        self.assertIsNotNone(sample)
        self.assertEqual(sample.timestamp_sec, 42.0)
        self.assertEqual(sample.map_pose, Pose2D(-0.86, -0.46, math.pi))
        self.assertEqual(sample.odom_pose, Pose2D(1.0, 2.0, 0.0))
        self.assertEqual(details["source"], "front_sector")


if __name__ == "__main__":
    unittest.main()
