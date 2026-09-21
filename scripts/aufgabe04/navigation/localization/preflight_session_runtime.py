"""ROS ownership for the mission's persistent preflight TF buffer."""

import threading
import os

from . import ros_preflight as preflight
from ..foundation.ros_runtime_config import ResolvedRuntimeConfig, RuntimeConfig


class PreflightSessionRuntime:
    def __init__(self):
        preflight._require_ros()
        from rclpy.executors import SingleThreadedExecutor

        preflight.rclpy.init(args=None)
        self.config = None
        self.listener_node = None
        self.listener = None
        self.buffer = None
        self.thread = None
        self.executor = SingleThreadedExecutor()
        self.executor_error = None
        self.clock_epoch = 0
        self.clock_jump_handle = None
        self.capture_count = 0

    def _clock_jumped(self, _jump):
        self.clock_epoch += 1
        self.buffer.clear()

    def _spin(self):
        try:
            self.executor.spin()
        except Exception as exc:
            self.executor_error = exc

    def _ensure_listener(self, config):
        if self.config is not None:
            if config != self.config:
                raise RuntimeError("preflight session runtime configuration changed")
            if self.executor_error is not None or self.thread is None or not self.thread.is_alive():
                raise RuntimeError("preflight session TF executor stopped")
            return
        self.config = config
        self.listener_node = preflight.Node(
            "aufgabe04_preflight_tf_session",
            parameter_overrides=preflight._node_parameter_overrides(config.use_sim_time),
        )
        self.buffer = preflight.Buffer()
        # Humble Buffer does not install a clock-jump callback itself.
        from rclpy.clock import JumpThreshold
        self.clock_jump_handle = self.listener_node.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None,
                          min_backward=preflight.Duration(nanoseconds=-1),
                          on_clock_change=True),
            post_callback=self._clock_jumped,
        )
        self.listener = preflight.TransformListener(
            self.buffer, self.listener_node, spin_thread=False,
        )
        if not self.executor.add_node(self.listener_node):
            raise RuntimeError("cannot attach persistent preflight TF listener")
        self.thread = threading.Thread(target=self._spin, daemon=True,
                                       name="aufgabe04-preflight-tf")
        self.thread.start()

    def collect(self, request):
        config_data = dict(request["config"])
        config_data["configured"] = RuntimeConfig(**config_data["configured"])
        config = ResolvedRuntimeConfig(**config_data)
        self._ensure_listener(config)
        clock_epoch = self.clock_epoch
        self.capture_count += 1
        options = dict(request["options"])
        options["preflight_requirements"] = preflight.RosPreflightRequirements(
            **options["preflight_requirements"])
        # Fresh node means no prior leg's AMCL samples, direct TF epoch, sensor
        # receipts, rejection history, or nomotion service futures are reused.
        node = preflight.RosPreflightNode(config, tf_buffer=self.buffer, **options)
        try:
            result = node.collect()
            if self.clock_epoch != clock_epoch:
                raise RuntimeError("preflight session clock changed during capture")
            if self.executor_error is not None or not self.thread.is_alive():
                raise RuntimeError("preflight session TF executor stopped during capture")
            result.observations.append(preflight.RosObservation(
                "preflight TF session", True, "fresh capture with session-owned TF buffer",
                {"process_id": os.getpid(), "capture_index": self.capture_count,
                 "tf_buffer_reused": self.capture_count > 1, "clock_epoch": self.clock_epoch},
            ))
            return result.to_json_dict()
        finally:
            node.destroy_node()

    def close(self):
        try:
            self.executor.shutdown(timeout_sec=2.0)
            if self.thread is not None:
                self.thread.join(timeout=2.0)
            if self.listener is not None:
                self.listener.unregister()
            if self.clock_jump_handle is not None:
                self.clock_jump_handle.unregister()
            if self.listener_node is not None:
                self.listener_node.destroy_node()
        finally:
            preflight.rclpy.try_shutdown()
