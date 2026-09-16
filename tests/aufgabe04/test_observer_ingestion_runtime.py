"""Slow camera work cannot block sensor receipts or create detector backlog."""

from collections import deque
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.real_robot.observer.ingestion_runtime import (
    BoundedSensorIngress, ObserverIngestionLoop, ObserverWorkSchedule,
)
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer import node as observer_node


class BoundedSensorIngressTests(unittest.TestCase):
    def test_retains_newest_bounded_sources_and_accounts_for_invalid_and_overwritten(self):
        ingress = BoundedSensorIngress()
        for channel, limit in ingress.LIMITS.items():
            for index in range(limit + 5):
                ingress.offer(channel, index)
            ingress.offer(channel, None)
        batch = ingress.drain()
        for channel, limit in ingress.LIMITS.items():
            self.assertEqual(getattr(batch, channel), tuple(range(5, limit + 5)))
            self.assertEqual(batch.counts[f"received_{channel}"], limit + 6)
            self.assertEqual(batch.counts[f"ingress_overwritten_{channel}"], 5)
            self.assertEqual(batch.counts[ingress.INVALID[channel]], 1)
        self.assertEqual(ingress.drain().counts, {})
        self.assertEqual(ingress.drain().images, ())

    def test_callbacks_continue_during_blocked_detector_without_mutating_owner_state(self):
        adapter = PassiveRealViewpointNode.__new__(PassiveRealViewpointNode)
        adapter._sensor_ingress = BoundedSensorIngress()
        adapter.images, adapter.scans, adapter.camera_infos = deque(), deque(), deque()
        adapter._camera_pipeline_counters = {}
        adapter.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=50_000_000_000)))
        detector_started, receipts_done = threading.Event(), threading.Event()
        stop_spin = threading.Event()
        transform_receipts = []

        def ingest():
            self.assertTrue(detector_started.wait(1.))
            for index in range(40):
                message = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(
                    sec=index + 1, nanosec=0)))
                adapter._on_image(message)
                adapter._on_scan(message)
                transform_receipts.append(index)  # Represents TF's independent callback.
            receipts_done.set()
            stop_spin.wait(1.)

        loop = ObserverIngestionLoop(spin_once=ingest, wake=stop_spin.set, ok=lambda: True)
        loop.start()
        try:
            schedule = ObserverWorkSchedule(process_rate_hz=5., tf_retry_rate_hz=50.)

            def process():
                detector_started.set()
                self.assertTrue(receipts_done.wait(1.))
                self.assertEqual(len(transform_receipts), 40)
                self.assertEqual(len(adapter.images), 0)
                self.assertEqual(len(adapter.scans), 0)

            schedule.run_due(drain=adapter._drain_received_sensors,
                             collect_witnesses=lambda: None, process=process, retry=lambda: None)
            adapter._drain_received_sensors()
            self.assertEqual(tuple(sample.stamp_sec for sample in adapter.images),
                             tuple(range(33, 41)))
            self.assertEqual(tuple(sample.stamp_sec for sample in adapter.scans),
                             tuple(range(21, 41)))
            self.assertEqual(adapter.images[-1].received_ros_sec, 50.)
            self.assertEqual(adapter._camera_pipeline_counters["received_images"], 40)
        finally:
            loop.close()

    def test_missing_scan_in_ingress_resets_consecutive_witness_history(self):
        adapter = PassiveRealViewpointNode.__new__(PassiveRealViewpointNode)
        adapter._sensor_ingress = BoundedSensorIngress()
        adapter.images, adapter.scans, adapter.camera_infos = deque(), deque(), deque()
        adapter._camera_pipeline_counters = {}
        adapter._scan_target_persistence = Mock()
        for index in range(21):
            adapter._sensor_ingress.offer("scans", index)
        adapter._drain_received_sensors()
        adapter._scan_target_persistence.reset.assert_called_once()
        self.assertEqual(adapter._camera_pipeline_counters["scan_witness_ingress_gap"], 1)
        self.assertEqual(tuple(adapter._pending_scan_witnesses), tuple(range(1, 21)))


class ObserverWorkScheduleTests(unittest.TestCase):
    def test_slow_detector_skips_missed_ticks_without_work_backlog(self):
        clock = [10.]
        schedule = ObserverWorkSchedule(process_rate_hz=5., tf_retry_rate_hz=50.,
                                        monotonic=lambda: clock[0])
        callbacks = {name: Mock() for name in ("drain", "collect_witnesses", "process", "retry")}
        schedule.run_due(**callbacks)
        clock[0] = 15.  # A long detector pause does not enqueue the 25 missed jobs.
        schedule.run_due(**callbacks)
        schedule.run_due(**callbacks)
        self.assertEqual(callbacks["process"].call_count, 2)
        callbacks["retry"].assert_not_called()
        clock[0] = 15.03
        schedule.run_due(**callbacks)
        callbacks["retry"].assert_called_once()
        self.assertEqual(callbacks["drain"].call_count, 4)

    def test_detector_and_retry_cannot_reenter_owner(self):
        schedule = ObserverWorkSchedule(process_rate_hz=5., tf_retry_rate_hz=50.)
        callbacks = {name: Mock() for name in ("drain", "collect_witnesses", "process", "retry")}
        callbacks["process"].side_effect = lambda: schedule.run_due(**callbacks)
        with self.assertRaisesRegex(RuntimeError, "non-reentrant"):
            schedule.run_due(**callbacks)
        self.assertEqual(callbacks["process"].call_count, 1)

    def test_other_thread_cannot_mutate_camera_owner_state(self):
        schedule = ObserverWorkSchedule(process_rate_hz=5., tf_retry_rate_hz=50.)
        callbacks = {name: Mock() for name in ("drain", "collect_witnesses", "process", "retry")}
        failures = []

        def invoke():
            try:
                schedule.run_due(**callbacks)
            except RuntimeError as exc:
                failures.append(exc)

        thread = threading.Thread(target=invoke)
        thread.start()
        thread.join(1.)
        self.assertEqual(len(failures), 1)
        for callback in callbacks.values():
            callback.assert_not_called()


class ObserverIngestionLoopTests(unittest.TestCase):
    def test_executor_failure_reaches_owner_with_original_cause(self):
        failure = ValueError("bad callback")
        loop = ObserverIngestionLoop(spin_once=Mock(side_effect=failure),
                                     wake=lambda: None, ok=lambda: True)
        loop.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "ROS ingestion failed") as caught:
                loop.wait(1.)
            self.assertIs(caught.exception.__cause__, failure)
        finally:
            loop.close()

    def test_shutdown_wakes_and_joins_executor_before_returning(self):
        entered, release, exited = threading.Event(), threading.Event(), threading.Event()

        def spin():
            entered.set()
            release.wait(1.)
            exited.set()

        loop = ObserverIngestionLoop(spin_once=spin, wake=release.set, ok=lambda: True)
        loop.start()
        self.assertTrue(entered.wait(1.))
        loop.close()
        self.assertTrue(exited.is_set())
        self.assertFalse(loop._thread.is_alive())

    def test_unresponsive_executor_shutdown_is_reported(self):
        entered, release = threading.Event(), threading.Event()

        def spin():
            entered.set()
            release.wait(1.)

        loop = ObserverIngestionLoop(spin_once=spin, wake=lambda: None, ok=lambda: True)
        loop.start()
        self.assertTrue(entered.wait(1.))
        try:
            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                loop.close(timeout_sec=.01)
        finally:
            release.set()
            loop.close()


class ObserverMainIngestionTests(unittest.TestCase):
    def test_main_services_callbacks_on_background_thread_and_stops_before_destroy(self):
        owner = threading.get_ident()
        entered, wake = threading.Event(), threading.Event()
        callback_threads, teardown = [], []
        adapter = SimpleNamespace(completed=False, axis_observation_committed=False,
                                  capture_history=None, node=Mock())

        def process():
            self.assertEqual(threading.get_ident(), owner)
            self.assertTrue(entered.wait(1.))
            adapter.completed = True

        adapter.process_pending_work = process
        adapter.node.destroy_node.side_effect = lambda: teardown.append("node")

        class Executor:
            def add_node(self, node):
                self.node = node

            def spin_once(self, timeout_sec):
                callback_threads.append(threading.get_ident())
                entered.set()
                wake.wait(timeout_sec)

            def wake(self):
                wake.set()

            def shutdown(self, timeout_sec):
                # Joining the ingestion loop precedes executor/node teardown.
                self_test.assertFalse(any(thread.name == "observer-ros-ingestion"
                                          for thread in threading.enumerate()))
                teardown.append("executor")

        self_test = self
        ros = ModuleType("rclpy")
        ros.init, ros.ok, ros.shutdown = Mock(), lambda: True, Mock()
        executors = ModuleType("rclpy.executors")
        executors.SingleThreadedExecutor = Executor
        parser = Mock()
        parser.parse_args.return_value = SimpleNamespace(once=True)
        with patch.dict("sys.modules", {"rclpy": ros, "rclpy.executors": executors}), \
                patch.object(observer_node, "build_parser", return_value=parser), \
                patch.object(observer_node, "_validate_args"), \
                patch.object(observer_node, "PassiveRealViewpointNode", return_value=adapter):
            self.assertEqual(observer_node.main([]), 0)
        self.assertTrue(all(ident != owner for ident in callback_threads))
        self.assertEqual(teardown, ["executor", "node"])
        ros.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
