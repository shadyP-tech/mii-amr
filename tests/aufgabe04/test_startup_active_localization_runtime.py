from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerConfig,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.startup_active_localization import (
    StartupActiveLocalizationConfig,
)
from scripts.aufgabe04.navigation.waypoint_follower.config import FollowerConfig
from scripts.aufgabe04.navigation.waypoint_follower.runtime import (
    SimpleWaypointFollowerNode,
)


def _config() -> StartupActiveLocalizationConfig:
    return StartupActiveLocalizationConfig(
        enabled=True,
        max_attempts=1,
        rotation_rad=0.35,
        angular_speed_radps=0.12,
        timeout_sec=8.0,
        control_rate_hz=10.0,
        maximum_translation_m=0.03,
        minimum_clearance_m=0.20,
        stop_command_count=4,
    )


def _node(events: list[object]) -> SimpleWaypointFollowerNode:
    node = object.__new__(SimpleWaypointFollowerNode)
    node.follower_config = FollowerConfig(
        controller=ControllerConfig(max_angular_radps=0.20),
        min_obstacle_distance_m=0.20,
        initial_sensor_wait_sec=2.0,
    )
    node.motion_published = False
    node.zero_command_publish_count = 0
    node.latest_stop_details = None
    node.command_smoother = SimpleNamespace(
        apply=lambda command, dt_sec: command,
    )
    node.publish_repeated_zero = lambda count=5: (
        events.append(("zero", count)),
        setattr(
            node,
            "zero_command_publish_count",
            node.zero_command_publish_count + count,
        ),
    )[-1]
    node._service_or_wait_for_callbacks = lambda _: None
    node._safety_failure = lambda: ""
    node._append_controller_trace = (
        lambda **kwargs: events.append(("trace", kwargs)) or ""
    )

    def publish(command):
        events.append(("publish", command))
        node.motion_published = True

    node._publish_velocity_command = publish
    node._wait_for_stationary_odom_pair = lambda **_: (
        object(),
        {"accepted": True},
    )
    return node


class StartupActiveLocalizationRuntimeTest(unittest.TestCase):
    def test_unsafe_startup_inputs_stop_before_nonzero_publication(self):
        events = []
        node = _node(events)
        node._wait_for_active_localization_inputs = (
            lambda **_: "obstacle too close"
        )

        result = node.run_startup_active_localization(
            _config(),
            attempt_index=0,
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "obstacle too close")
        self.assertFalse(result.motion_published)
        self.assertFalse(any(event[0] == "publish" for event in events))

    def test_rotation_is_odom_measured_linear_zero_and_stopped_afterward(self):
        events = []
        node = _node(events)
        node._wait_for_active_localization_inputs = lambda **_: ""
        poses = iter(
            (
                Pose2D(0.0, 0.0, 0.0),
                Pose2D(0.0, 0.0, 0.15),
                Pose2D(0.0, 0.0, 0.31),
            )
        )
        node._latest_odom_pose = lambda: next(poses)

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower."
            "runtime_components.startup_active_localization.rclpy",
            SimpleNamespace(ok=lambda: True),
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower."
            "runtime_components.startup_active_localization.time.sleep",
            return_value=None,
        ):
            result = node.run_startup_active_localization(
                _config(),
                attempt_index=0,
            )

        published = [event[1] for event in events if event[0] == "publish"]
        self.assertTrue(result.completed)
        self.assertTrue(result.motion_published)
        self.assertGreaterEqual(result.accumulated_progress_rad, 0.30)
        self.assertTrue(published)
        self.assertTrue(
            all(abs(command.linear_x_mps) <= 1.0e-12 for command in published)
        )
        self.assertTrue(
            all(0.0 < command.angular_z_radps <= 0.12 for command in published)
        )
        self.assertTrue(result.stop_details["stationary_odom"]["accepted"])
        self.assertGreaterEqual(result.zero_command_count, 8)

    def test_translation_drift_stops_before_first_rotation_command(self):
        events = []
        node = _node(events)
        node._wait_for_active_localization_inputs = lambda **_: ""
        poses = iter(
            (
                Pose2D(0.0, 0.0, 0.0),
                Pose2D(0.031, 0.0, 0.10),
            )
        )
        node._latest_odom_pose = lambda: next(poses)

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower."
            "runtime_components.startup_active_localization.rclpy",
            SimpleNamespace(ok=lambda: True),
        ):
            result = node.run_startup_active_localization(
                _config(),
                attempt_index=0,
            )

        self.assertEqual(result.status, "stopped")
        self.assertIn("translation bound", result.stop_reason)
        self.assertFalse(result.motion_published)
        self.assertFalse(any(event[0] == "publish" for event in events))


if __name__ == "__main__":
    unittest.main()
