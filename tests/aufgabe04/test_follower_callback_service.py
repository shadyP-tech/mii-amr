from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from scripts.aufgabe04.navigation import simple_waypoint_follower as follower


class FollowerCallbackServiceTest(unittest.TestCase):
    def bare_node(self):
        return object.__new__(follower.SimpleWaypointFollowerNode)

    def test_background_mode_waits_without_caller_spin(self):
        node = self.bare_node()
        node.enable_background_callback_service()
        fake_rclpy = SimpleNamespace(spin_once=Mock())

        with patch.object(follower, "rclpy", fake_rclpy), patch.object(
            follower.time,
            "sleep",
        ) as sleep:
            node._service_or_wait_for_callbacks(0.05)

        self.assertEqual(
            node.callback_service_mode,
            follower.CALLBACK_SERVICE_BACKGROUND_EXECUTOR,
        )
        sleep.assert_called_once_with(0.05)
        fake_rclpy.spin_once.assert_not_called()

    def test_repeated_zero_uses_background_wait_instead_of_second_executor(self):
        node = self.bare_node()
        node.enable_background_callback_service()
        node.publish_zero = Mock()
        fake_rclpy = SimpleNamespace(spin_once=Mock())

        with patch.object(follower, "rclpy", fake_rclpy), patch.object(
            follower.time,
            "sleep",
        ) as sleep:
            node.publish_repeated_zero(count=3)

        self.assertEqual(node.publish_zero.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(0.02)] * 3)
        fake_rclpy.spin_once.assert_not_called()

    def test_background_drain_is_immediate_except_for_bounded_recovery(self):
        node = self.bare_node()
        node.enable_background_callback_service()
        fake_rclpy = SimpleNamespace(spin_once=Mock())

        with patch.object(follower, "rclpy", fake_rclpy), patch.object(
            follower.time,
            "monotonic",
            side_effect=(10.0, 10.0),
        ), patch.object(follower.time, "sleep") as sleep:
            ordinary = node._drain_runtime_callbacks()

        sleep.assert_not_called()
        fake_rclpy.spin_once.assert_not_called()
        self.assertEqual(
            ordinary["callback_service_mode"],
            follower.CALLBACK_SERVICE_BACKGROUND_EXECUTOR,
        )
        self.assertEqual(ordinary["spin_count"], 0)
        self.assertFalse(ordinary["deadline_reached"])
        self.assertEqual(ordinary["background_wait_requested_sec"], 0.0)

        with patch.object(follower, "rclpy", fake_rclpy), patch.object(
            follower.time,
            "monotonic",
            side_effect=(20.0, 20.18),
        ), patch.object(follower.time, "sleep") as sleep:
            recovery = node._drain_runtime_callbacks(
                max_callbacks=follower.STALE_TF_RECOVERY_MAX_CALLBACKS,
                max_duration_sec=follower.STALE_TF_RECOVERY_MAX_DURATION_SEC,
                spin_timeout_sec=follower.STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC,
            )

        sleep.assert_called_once_with(
            follower.STALE_TF_RECOVERY_MAX_DURATION_SEC
        )
        fake_rclpy.spin_once.assert_not_called()
        self.assertEqual(recovery["spin_count"], 0)
        self.assertTrue(recovery["deadline_reached"])
        self.assertEqual(
            recovery["background_wait_requested_sec"],
            follower.STALE_TF_RECOVERY_MAX_DURATION_SEC,
        )

    def test_pose_lookup_uses_the_buffer_injected_on_the_follower(self):
        node = self.bare_node()
        injected_buffer = Mock()
        injected_buffer.lookup_transform.side_effect = RuntimeError(
            "injected buffer sentinel"
        )
        node.tf_buffer = injected_buffer
        node.runtime_config = SimpleNamespace(
            map_frame="odom",
            base_frame="base_footprint",
        )
        node.follower_config = SimpleNamespace(max_tf_age_sec=1.0)
        lookup_time = object()
        lookup_timeout = object()

        with patch.object(
            follower,
            "TransformException",
            RuntimeError,
        ), patch.object(follower, "Time", return_value=lookup_time), patch.object(
            follower,
            "Duration",
            return_value=lookup_timeout,
        ):
            result = node._current_pose_lookup()

        injected_buffer.lookup_transform.assert_called_once_with(
            "odom",
            "base_footprint",
            lookup_time,
            timeout=lookup_timeout,
        )
        self.assertIsNone(result.pose)
        self.assertEqual(result.details["reason"], "lookup_exception")
        self.assertEqual(
            result.details["exception"],
            "injected buffer sentinel",
        )

    def test_runner_isolates_tf_listener_and_orders_both_executor_teardowns(self):
        events = []
        created = {}
        expected_result = object()

        class FakeRclpy:
            def init(self, *, args):
                self.asserted_args = args
                events.append("rclpy.init")

            def ok(self):
                events.append("rclpy.ok")
                return True

            def shutdown(self):
                events.append("rclpy.shutdown")

        class FakeListenerNode:
            def __init__(self, name, *, namespace):
                created["listener_node"] = self
                created["listener_node_name"] = name
                created["listener_node_namespace"] = namespace
                events.append("listener_node.init")

            def destroy_node(self):
                events.append("listener_node.destroy")

        class FakeFollowerNode:
            def __init__(self, *args, tf_buffer):
                created["node"] = self
                created["node_args"] = args
                created["injected_tf_buffer"] = tf_buffer
                events.append("node.init")

            def enable_background_callback_service(self):
                events.append("node.enable_background")

            def disable_background_callback_service(self):
                events.append("node.disable_background")

            def run(self):
                events.append("node.run")
                return expected_result

            def destroy_node(self):
                events.append("node.destroy")

        class FakeBuffer:
            def __init__(self, *, node):
                created["buffer"] = self
                created["buffer_node"] = node
                events.append("buffer.init")

        class FakeTransformListener:
            def __init__(self, buffer, node, *, spin_thread):
                created["tf_listener"] = self
                created["tf_listener_buffer"] = buffer
                created["tf_listener_node"] = node
                created["tf_listener_spin_thread"] = spin_thread
                events.append("tf_listener.init")

            def unregister(self):
                events.append("tf_listener.unregister")

        class FakeExecutor:
            def __init__(self, *, num_threads):
                created["follower_executor"] = self
                created["follower_executor_num_threads"] = num_threads
                events.append("follower_executor.init")

            def add_node(self, node):
                self.node = node
                events.append("follower_executor.add_node")
                return True

            def spin(self):
                events.append("follower_executor.spin")

            def shutdown(self):
                events.append("follower_executor.shutdown")

            def remove_node(self, node):
                self.removed_node = node
                events.append("follower_executor.remove_node")

        class FakeSingleThreadedExecutor:
            def __init__(self):
                created["tf_executor"] = self
                events.append("tf_executor.init")

            def add_node(self, node):
                self.node = node
                events.append("tf_executor.add_node")
                return True

            def spin(self):
                events.append("tf_executor.spin")

            def shutdown(self):
                events.append("tf_executor.shutdown")

            def remove_node(self, node):
                self.removed_node = node
                events.append("tf_executor.remove_node")

        class FakeThread:
            def __init__(self, *, target, name, daemon):
                self.target = target
                self.name = name
                self.daemon = daemon
                created.setdefault("threads", []).append(self)
                self.ident = None
                events.append(f"thread.init:{name}")

            def start(self):
                self.ident = 1234
                events.append(f"thread.start:{self.name}")

            def join(self):
                events.append(f"thread.join:{self.name}")

        fake_rclpy = FakeRclpy()
        with patch.object(follower, "rclpy", fake_rclpy), patch.object(
            follower,
            "SimpleWaypointFollowerNode",
            FakeFollowerNode,
        ), patch.object(
            follower,
            "Node",
            FakeListenerNode,
        ), patch.object(
            follower,
            "Buffer",
            FakeBuffer,
        ), patch.object(
            follower,
            "TransformListener",
            FakeTransformListener,
        ), patch.object(
            follower,
            "MultiThreadedExecutor",
            FakeExecutor,
        ), patch.object(
            follower,
            "SingleThreadedExecutor",
            FakeSingleThreadedExecutor,
        ), patch.object(
            follower.threading,
            "Thread",
            FakeThread,
        ):
            result = follower.run_simple_waypoint_follower(
                SimpleNamespace(
                    namespace="/robot1",
                    use_sim_time=True,
                ),
                (),
                object(),
            )

        self.assertIs(result, expected_result)
        self.assertEqual(
            created["follower_executor_num_threads"],
            follower.FOLLOWER_EXECUTOR_NUM_THREADS,
        )
        self.assertIsNot(
            created["follower_executor"],
            created["tf_executor"],
        )
        self.assertIsInstance(
            created["follower_executor"],
            FakeExecutor,
        )
        self.assertIsInstance(
            created["tf_executor"],
            FakeSingleThreadedExecutor,
        )
        self.assertIs(created["follower_executor"].node, created["node"])
        self.assertIs(
            created["tf_executor"].node,
            created["listener_node"],
        )
        self.assertIsNot(created["node"], created["listener_node"])
        self.assertEqual(
            created["listener_node_name"],
            follower.TF_LISTENER_NODE_NAME,
        )
        self.assertEqual(created["listener_node_namespace"], "/robot1")
        self.assertIs(created["buffer_node"], created["listener_node"])
        self.assertIs(created["tf_listener_buffer"], created["buffer"])
        self.assertIs(created["tf_listener_node"], created["listener_node"])
        self.assertFalse(created["tf_listener_spin_thread"])
        self.assertIs(created["injected_tf_buffer"], created["buffer"])
        threads_by_name = {thread.name: thread for thread in created["threads"]}
        self.assertEqual(
            set(threads_by_name),
            {
                "aufgabe04-follower-callbacks",
                "aufgabe04-follower-tf-listener",
            },
        )
        self.assertIs(
            threads_by_name["aufgabe04-follower-callbacks"].target.__self__,
            created["follower_executor"],
        )
        self.assertIs(
            threads_by_name["aufgabe04-follower-tf-listener"].target.__self__,
            created["tf_executor"],
        )
        self.assertTrue(
            all(not thread.daemon for thread in created["threads"])
        )
        self.assertEqual(
            events,
            [
                "rclpy.init",
                "listener_node.init",
                "buffer.init",
                "tf_listener.init",
                "node.init",
                "follower_executor.init",
                "tf_executor.init",
                "follower_executor.add_node",
                "tf_executor.add_node",
                "node.enable_background",
                "thread.init:aufgabe04-follower-tf-listener",
                "thread.init:aufgabe04-follower-callbacks",
                "thread.start:aufgabe04-follower-tf-listener",
                "thread.start:aufgabe04-follower-callbacks",
                "node.run",
                "follower_executor.shutdown",
                "tf_executor.shutdown",
                "thread.join:aufgabe04-follower-callbacks",
                "thread.join:aufgabe04-follower-tf-listener",
                "follower_executor.remove_node",
                "tf_executor.remove_node",
                "node.disable_background",
                "node.destroy",
                "tf_listener.unregister",
                "listener_node.destroy",
                "rclpy.ok",
                "rclpy.shutdown",
            ],
        )


if __name__ == "__main__":
    unittest.main()
