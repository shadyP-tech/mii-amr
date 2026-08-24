from __future__ import annotations

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.control.controller_trace import (
    ControllerTraceWriter,
    load_controller_traces,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.navigation.localization.odom_route_adapter import (
    OdomExecutionContext,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime import (
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
    def _initial_wait_node(self):
        node = object.__new__(SimpleWaypointFollowerNode)
        node.odom_execution_context = _context()
        node.follower_config = SimpleNamespace(
            initial_sensor_wait_sec=1.0,
            max_scan_age_sec=1.0,
            max_odom_age_sec=1.0,
        )
        node.latest_scan = object()
        node.latest_scan_receipt = 0.0
        node.latest_odom = object()
        node.latest_odom_receipt = 0.0
        node.latest_stop_details = None
        node._service_or_wait_for_callbacks = Mock()
        node._freshness_failure = Mock(return_value="")
        node._current_pose_lookup = Mock(
            return_value=PoseLookupResult(Pose2D(0.0, 0.0, 0.0))
        )
        node.publish_zero = Mock()
        return node

    def _startup_failure_run_node(self, stop_details):
        node = object.__new__(SimpleWaypointFollowerNode)
        node.waypoints = (Pose2D(0.0, 0.0), Pose2D(0.1, 0.0))
        node.current_route_kind = "stand_discovery_corridor"
        node.distance_estimate_m = 0.0
        node.motion_published = False
        node.latest_stop_details = stop_details
        node.latest_front_clearance_details = None
        node.latest_odom = None
        node.target_index = 0
        node.controller_route_revision = 0
        node.controller_trace_writer = None
        node._wait_for_initial_runtime_inputs = Mock(
            return_value=(
                "global localization consistency requires zero and reseal"
            )
        )
        node.publish_repeated_zero = Mock()
        return node

    @staticmethod
    def _global_consistency_stop_details():
        return {
            "reason": (
                "global localization consistency requires zero and reseal"
            ),
            "fault_code": "localization_reseal_required",
            "source": "global_consistency_monitor",
            "execution_pose_owner": "odom",
            "global_consistency_monitor": "amcl",
            "monitor_action": "FORCE_ZERO_RESEAL",
            "monitor_reason": "reseal_required",
            "monitor_warning": "",
            "continuity": {
                "schema_version": 1,
                "accepted": False,
                "decision": "force_zero_reseal",
                "reason": "map_from_odom_translation_and_yaw_drift",
                "fail_closed": True,
                "requires_zero_cycle": True,
                "requires_reseal": True,
                "threshold_semantics": (
                    "accept_if_observed_less_than_or_equal_to_limit"
                ),
                "certificate_sha256": "a" * 64,
                "map_frame": "map",
                "odom_frame": "odom",
                "base_frame": "base_footprint",
                "frozen_map_from_odom": {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "yaw_rad": 0.0,
                },
                "live_map_from_odom": {
                    "x_m": 0.102556,
                    "y_m": 0.0,
                    "yaw_rad": 0.067884,
                },
                "relative_translation_x_m": 0.102556,
                "relative_translation_y_m": 0.0,
                "translation_drift_m": 0.102556,
                "relative_yaw_rad": 0.067884,
                "absolute_yaw_drift_rad": 0.067884,
                "max_translation_drift_m": 0.03,
                "max_yaw_drift_rad": 0.03,
                "validation_error": None,
            },
            "fail_closed": True,
        }

    def test_initial_wait_warms_global_consistency_tf_before_motion(self):
        node = self._initial_wait_node()
        node._global_consistency_monitor_failure = Mock(
            side_effect=[
                "global localization consistency requires zero and reseal",
                "",
            ]
        )
        node.latest_stop_details = {"fault_code": "localization_reseal_required"}

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: True),
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
            return_value=0.5,
        ):
            result = node._wait_for_initial_runtime_inputs(0.0)

        self.assertEqual(result, "")
        self.assertEqual(node._global_consistency_monitor_failure.call_count, 2)
        node.publish_zero.assert_called_once_with()
        self.assertIsNone(node.latest_stop_details)

    def test_initial_wait_keeps_persistent_global_tf_failure_terminal(self):
        node = self._initial_wait_node()
        failure = "global localization consistency requires zero and reseal"
        node._global_consistency_monitor_failure = Mock(return_value=failure)
        node.latest_stop_details = {"fault_code": "localization_reseal_required"}

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: True),
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
            side_effect=[0.5, 1.0],
        ):
            result = node._wait_for_initial_runtime_inputs(0.0)

        self.assertEqual(result, failure)
        self.assertEqual(node._global_consistency_monitor_failure.call_count, 2)
        node.publish_zero.assert_called_once_with()
        self.assertEqual(
            node.latest_stop_details["fault_code"],
            "localization_reseal_required",
        )

    def test_run_preserves_startup_global_consistency_evidence(self):
        details = self._global_consistency_stop_details()
        expected = {
            **details,
            "execution_phase": "before_motion",
            "phase": "initial_runtime_input_wait",
            "motion_published": False,
        }
        node = self._startup_failure_run_node(details)

        result = node.run()

        self.assertEqual(result.status, "stopped")
        self.assertFalse(result.motion_published)
        self.assertEqual(result.stop_details, expected)
        self.assertEqual(
            result.stop_details["continuity"]["translation_drift_m"],
            0.102556,
        )
        self.assertEqual(node.latest_stop_details, expected)
        self.assertEqual(node.publish_repeated_zero.call_count, 2)

    def test_startup_phase_marker_does_not_rewrite_conflicting_evidence(self):
        details = {
            **self._global_consistency_stop_details(),
            "execution_phase": "conflicting_phase",
            "phase": "conflicting_source_phase",
        }
        node = self._startup_failure_run_node(details)

        result = node.run()

        self.assertEqual(
            result.stop_details["execution_phase"],
            "conflicting_phase",
        )
        self.assertEqual(
            result.stop_details["phase"],
            "conflicting_source_phase",
        )

    def test_startup_global_consistency_stop_is_persisted_in_trace(self):
        details = self._global_consistency_stop_details()
        expected = {
            **details,
            "execution_phase": "before_motion",
            "phase": "initial_runtime_input_wait",
            "motion_published": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "controller_trace.jsonl"
            node = self._startup_failure_run_node(details)
            node.controller_trace_writer = ControllerTraceWriter(trace_path)

            result = node.run()
            records = load_controller_traces(trace_path)

        self.assertEqual(result.stop_details, expected)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.event, "initial_runtime_input_stop")
        self.assertTrue(record.fail_closed)
        self.assertEqual(record.reason, details["reason"])
        self.assertEqual(record.effective_command.linear_x_mps, 0.0)
        self.assertEqual(record.effective_command.angular_z_radps, 0.0)
        self.assertEqual(
            record.diagnostics["fault_code"],
            "localization_reseal_required",
        )
        self.assertEqual(
            record.diagnostics["continuity"]["absolute_yaw_drift_rad"],
            0.067884,
        )
        self.assertEqual(record.diagnostics["execution_phase"], "before_motion")
        self.assertEqual(
            record.diagnostics["phase"],
            "initial_runtime_input_wait",
        )

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
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.Time",
            _FakeTime,
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.Duration",
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
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.Time",
            _FakeTime,
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.Duration",
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
                    "scripts.aufgabe04.navigation.waypoint_follower.runtime.Time",
                    _FakeTime,
                ), patch(
                    "scripts.aufgabe04.navigation.waypoint_follower.runtime.Duration",
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
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: True),
        ):
            result = node.run()

        self.assertEqual(result.stop_reason, "obstacle too close")
        node._global_consistency_monitor_failure.assert_not_called()
        self.assertGreaterEqual(node.publish_repeated_zero.call_count, 2)

    def test_runtime_global_tf_failure_preserves_prior_motion_evidence(self):
        node = object.__new__(SimpleWaypointFollowerNode)
        node.waypoints = (Pose2D(0.0, 0.0), Pose2D(0.1, 0.0))
        node.current_route_kind = "stand_discovery_corridor"
        node.follower_config = SimpleNamespace(control_rate_hz=10.0)
        node.distance_estimate_m = 0.04
        node.motion_published = True
        node.latest_stop_details = {"fault_code": "localization_reseal_required"}
        node._wait_for_initial_runtime_inputs = Mock(return_value="")
        node._drain_runtime_callbacks = Mock()
        node._safety_failure = Mock(return_value="")
        failure = "global localization consistency requires zero and reseal"
        node._global_consistency_monitor_failure = Mock(return_value=failure)
        node.publish_repeated_zero = Mock()

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: True),
        ):
            result = node.run()

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, failure)
        self.assertTrue(result.motion_published)
        node._global_consistency_monitor_failure.assert_called_once_with()
        self.assertGreaterEqual(node.publish_repeated_zero.call_count, 2)


if __name__ == "__main__":
    unittest.main()
