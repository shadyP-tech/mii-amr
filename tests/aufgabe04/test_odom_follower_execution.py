from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.navigation.odom_route_adapter import (
    OdomExecutionContext,
)
from scripts.aufgabe04.navigation.simple_waypoint_follower import (
    FollowerConfig,
    PoseLookupResult,
    SimpleWaypointFollowerNode,
)


def _context(
    *,
    transform: PlanarTransform2D = PlanarTransform2D(1.0, 2.0, math.pi / 2.0),
    translation_limit_m: float = 0.05,
    yaw_limit_rad: float = 0.10,
) -> OdomExecutionContext:
    return OdomExecutionContext(
        map_frame="map",
        odom_frame="odom",
        base_frame="base_footprint",
        frozen_map_from_odom=transform,
        certificate_sha256="a" * 64,
        max_map_from_odom_translation_drift_m=translation_limit_m,
        max_map_from_odom_yaw_drift_rad=yaw_limit_rad,
    )


class _FakeTime:
    def __init__(self, nanoseconds: int = 0):
        self.nanoseconds = nanoseconds

    @classmethod
    def from_msg(cls, stamp):
        return cls(
            int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        )

    def __sub__(self, other):
        return SimpleNamespace(
            nanoseconds=self.nanoseconds - other.nanoseconds
        )


def _transform(
    *,
    x_m: float,
    y_m: float,
    yaw_rad: float,
    stamp_sec: int = 100,
    parent_frame: str = "odom",
    child_frame: str = "base_footprint",
):
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=parent_frame,
            stamp=SimpleNamespace(sec=stamp_sec, nanosec=0),
        ),
        child_frame_id=child_frame,
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=x_m, y=y_m, z=0.0),
            rotation=SimpleNamespace(
                x=0.0,
                y=0.0,
                z=math.sin(yaw_rad / 2.0),
                w=math.cos(yaw_rad / 2.0),
            ),
        ),
    )


class OdomFollowerExecutionTest(unittest.TestCase):
    def test_control_pose_lookup_targets_odom_not_map(self):
        calls = []
        transform = _transform(x_m=0.4, y_m=-0.2, yaw_rad=0.3)

        class Buffer:
            def lookup_transform(self, target, source, _time, *, timeout):
                calls.append((target, source, timeout.seconds))
                return transform

        node = object.__new__(SimpleWaypointFollowerNode)
        node.odom_execution_context = _context()
        node.runtime_config = SimpleNamespace(
            map_frame="map",
            odom_frame="odom",
            base_frame="base_footprint",
        )
        node.follower_config = SimpleNamespace(
            max_tf_age_sec=1.0,
            max_future_timestamp_sec=0.25,
        )
        node.tf_buffer = Buffer()
        node.get_clock = lambda: SimpleNamespace(
            now=lambda: _FakeTime(100_000_000_000)
        )

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.Time",
            _FakeTime,
        ), patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.Duration",
            lambda *, seconds: SimpleNamespace(seconds=seconds),
        ):
            result = node._current_pose_lookup()

        self.assertEqual(calls, [("odom", "base_footprint", 0.1)])
        self.assertEqual(result.pose, Pose2D(0.4, -0.2, 0.3))

    def test_stationary_blockage_sample_reconstructs_map_pose_from_frozen_frame(self):
        node = object.__new__(SimpleWaypointFollowerNode)
        node.odom_execution_context = _context()
        node.follower_config = SimpleNamespace(
            max_scan_age_sec=1.0,
            max_odom_age_sec=1.0,
            front_obstacle_sector_rad=0.35,
            min_obstacle_distance_m=0.20,
        )
        node.latest_scan = SimpleNamespace(
            ranges=(0.18,),
            angle_min=0.0,
            angle_increment=0.1,
        )
        node.latest_scan_receipt = 12.5
        node.latest_odom = object()
        node.latest_odom_receipt = 12.5
        node.latest_stop_details = None
        node._freshness_failure = Mock(return_value="")
        node._scan_range_min = Mock(return_value=0.02)
        node._scan_range_max = Mock(return_value=12.0)
        node._current_pose_lookup_with_stale_recovery = Mock(
            return_value=PoseLookupResult(Pose2D(3.0, 4.0, -math.pi / 4.0))
        )
        node._latest_odom_pose = Mock(
            return_value=Pose2D(3.0, 4.0, -math.pi / 4.0)
        )

        sample, _details = node._stationary_front_sample()

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertAlmostEqual(sample.map_pose.x_m, -3.0)
        self.assertAlmostEqual(sample.map_pose.y_m, 5.0)
        self.assertAlmostEqual(sample.map_pose.yaw_rad, math.pi / 4.0)
        self.assertEqual(sample.odom_pose, Pose2D(3.0, 4.0, -math.pi / 4.0))

    def test_live_map_odom_drift_forces_zero_reseal_decision(self):
        node = object.__new__(SimpleWaypointFollowerNode)
        node.odom_execution_context = _context(
            transform=PlanarTransform2D(0.0, 0.0, 0.0),
            translation_limit_m=0.05,
        )
        node.follower_config = SimpleNamespace(
            amcl_edge_future_tolerance_sec=1.1,
            max_tf_age_sec=1.0,
        )
        node.tf_buffer = SimpleNamespace(
            lookup_transform=lambda *_args, **_kwargs: _transform(
                x_m=0.051,
                y_m=0.0,
                yaw_rad=0.0,
                parent_frame="map",
                child_frame="odom",
            )
        )
        node.get_clock = lambda: SimpleNamespace(
            now=lambda: _FakeTime(100_000_000_000)
        )
        node.latest_stop_details = None

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.Time",
            _FakeTime,
        ), patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.Duration",
            lambda *, seconds: SimpleNamespace(seconds=seconds),
        ):
            reason = node._global_consistency_monitor_failure()

        self.assertIn("requires zero and reseal", reason)
        self.assertEqual(
            node.latest_stop_details["fault_code"],
            "localization_reseal_required",
        )
        self.assertFalse(
            node.latest_stop_details["continuity"]["accepted"]
        )

    def test_nonfinite_or_nonunit_control_tf_is_rejected(self):
        cases = (
            _transform(x_m=math.nan, y_m=0.0, yaw_rad=0.0),
            SimpleNamespace(
                **{
                    **_transform(
                        x_m=0.0,
                        y_m=0.0,
                        yaw_rad=0.0,
                    ).__dict__,
                    "transform": SimpleNamespace(
                        translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                        rotation=SimpleNamespace(
                            x=0.0, y=0.0, z=0.0, w=0.5
                        ),
                    ),
                }
            ),
        )
        for transform in cases:
            with self.subTest(transform=transform):
                node = object.__new__(SimpleWaypointFollowerNode)
                node.odom_execution_context = _context()
                node.runtime_config = SimpleNamespace(
                    map_frame="map",
                    odom_frame="odom",
                    base_frame="base_footprint",
                )
                node.follower_config = SimpleNamespace(
                    max_tf_age_sec=1.0,
                    max_future_timestamp_sec=0.25,
                )
                node.tf_buffer = SimpleNamespace(
                    lookup_transform=lambda *_args, **_kwargs: transform
                )
                node.get_clock = lambda: SimpleNamespace(
                    now=lambda: _FakeTime(100_000_000_000)
                )

                with patch(
                    "scripts.aufgabe04.navigation.simple_waypoint_follower.Time",
                    _FakeTime,
                ), patch(
                    "scripts.aufgabe04.navigation.simple_waypoint_follower.Duration",
                    lambda *, seconds: SimpleNamespace(seconds=seconds),
                ):
                    result = node._current_pose_lookup()

                self.assertIsNone(result.pose)
                self.assertEqual(
                    result.details["reason"], "malformed_transform_pose"
                )

    def test_lidar_safety_failure_precedes_global_monitor(self):
        node = object.__new__(SimpleWaypointFollowerNode)
        node.waypoints = (Pose2D(0.0, 0.0), Pose2D(0.1, 0.0))
        node.current_route_kind = "stand_discovery_corridor"
        node.blockage_recovery_provider = None
        node.follower_config = SimpleNamespace(control_rate_hz=10.0)
        node.distance_estimate_m = 0.0
        node.motion_published = False
        node.latest_stop_details = {"source": "front_sector"}
        node._wait_for_initial_runtime_inputs = Mock(return_value="")
        node._drain_runtime_callbacks = Mock()
        node._safety_failure = Mock(return_value="obstacle too close")
        node._global_consistency_monitor_failure = Mock(
            side_effect=AssertionError("monitor ran before LiDAR stop")
        )
        node.publish_repeated_zero = Mock()

        with patch(
            "scripts.aufgabe04.navigation.simple_waypoint_follower.rclpy",
            SimpleNamespace(ok=lambda: True),
        ):
            result = node.run()

        self.assertEqual(result.stop_reason, "obstacle too close")
        node._global_consistency_monitor_failure.assert_not_called()
        self.assertGreaterEqual(node.publish_repeated_zero.call_count, 2)


if __name__ == "__main__":
    unittest.main()
