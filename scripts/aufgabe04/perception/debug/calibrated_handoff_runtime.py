"""ROS adapter for live CameraInfo and camera-to-scan TF diagnostics."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from scripts.aufgabe04.perception.camera_calibration import (
    CameraCalibration,
    camera_calibration_from_info,
)
from scripts.aufgabe04.perception.stand_axis_handoff.models import RigidTransform


@dataclass(frozen=True)
class CalibrationRuntimeSnapshot:
    ready: bool
    reason: str
    calibration: CameraCalibration | None = None
    scan_from_camera: RigidTransform | None = None
    camera_info_age_sec: float | None = None


class RosCameraCalibrationTfSource:
    """Read-only source for calibrated CameraInfo and a sealed TF edge."""

    def __init__(
        self,
        *,
        camera_info_topic: str,
        scan_frame: str,
        camera_frame: str,
        max_camera_info_age_sec: float,
        tf_timeout_sec: float,
    ) -> None:
        self.camera_info_topic = camera_info_topic
        self.scan_frame = scan_frame
        self.camera_frame = camera_frame
        self.max_camera_info_age_sec = max_camera_info_age_sec
        self.tf_timeout_sec = tf_timeout_sec
        self._lock = threading.Lock()
        self._calibration: CameraCalibration | None = None
        self._camera_info_received_sec: float | None = None
        self._calibration_error: str | None = None
        self._running = False
        self._spin_thread = None

        try:
            import rclpy
            from rclpy.duration import Duration
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, qos_profile_sensor_data
            from rclpy.time import Time
            from sensor_msgs.msg import CameraInfo
            from tf2_ros import Buffer, TransformException, TransformListener
        except ImportError as exc:
            raise SystemExit(
                "calibrated handoff requires rclpy, sensor_msgs, and tf2_ros"
            ) from exc

        self.rclpy = rclpy
        self.Duration = Duration
        self.Time = Time
        self.TransformException = TransformException
        self.owns_rclpy = not rclpy.ok()
        if self.owns_rclpy:
            rclpy.init(args=None)
        self.executor = SingleThreadedExecutor()

        class CalibrationNode(Node):
            pass

        self.node = CalibrationNode("aufgabe04_stand_axis_calibrated_handoff")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self.node, spin_thread=False
        )
        qos_profile = QoSProfile(
            reliability=qos_profile_sensor_data.reliability,
            durability=qos_profile_sensor_data.durability,
            history=qos_profile_sensor_data.history,
            depth=1,
        )
        self.subscription = self.node.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._on_camera_info,
            qos_profile,
        )
        self.executor.add_node(self.node)

    def _on_camera_info(self, message) -> None:
        try:
            calibration = camera_calibration_from_info(message)
        except ValueError as exc:
            with self._lock:
                self._calibration_error = str(exc)
            return
        with self._lock:
            self._calibration = calibration
            self._camera_info_received_sec = time.time()
            self._calibration_error = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _spin_loop(self) -> None:
        while self._running and self.rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except Exception:
                if not self._running or not self.rclpy.ok():
                    break
                raise

    def snapshot(self, *, now_sec: float | None = None) -> CalibrationRuntimeSnapshot:
        now_sec = time.time() if now_sec is None else float(now_sec)
        with self._lock:
            calibration = self._calibration
            received_sec = self._camera_info_received_sec
            calibration_error = self._calibration_error
        if calibration is None:
            return CalibrationRuntimeSnapshot(
                False,
                calibration_error or "camera_info_unavailable",
            )
        age = None if received_sec is None else max(0.0, now_sec - received_sec)
        if (
            age is not None
            and self.max_camera_info_age_sec > 0.0
            and age > self.max_camera_info_age_sec
        ):
            return CalibrationRuntimeSnapshot(
                False,
                "camera_info_stale",
                calibration=calibration,
                camera_info_age_sec=age,
            )
        if (
            calibration.frame_id
            and calibration.frame_id.lstrip("/") != self.camera_frame.lstrip("/")
        ):
            return CalibrationRuntimeSnapshot(
                False,
                "camera_info_frame_mismatch",
                calibration=calibration,
                camera_info_age_sec=age,
            )
        try:
            transform = self.tf_buffer.lookup_transform(
                self.scan_frame,
                self.camera_frame,
                self.Time(),
                timeout=self.Duration(seconds=self.tf_timeout_sec),
            )
        except self.TransformException:
            return CalibrationRuntimeSnapshot(
                False,
                "camera_to_scan_tf_unavailable",
                calibration=calibration,
                camera_info_age_sec=age,
            )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        scan_from_camera = RigidTransform(
            parent_frame=self.scan_frame,
            child_frame=self.camera_frame,
            translation_xyz_m=(
                float(translation.x),
                float(translation.y),
                float(translation.z),
            ),
            rotation_xyzw=(
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
                float(rotation.w),
            ),
        )
        return CalibrationRuntimeSnapshot(
            True,
            "calibrated",
            calibration=calibration,
            scan_from_camera=scan_from_camera,
            camera_info_age_sec=age,
        )

    def release(self) -> None:
        self._running = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        self.executor.shutdown()
        if self.owns_rclpy and self.rclpy.ok():
            self.rclpy.shutdown()
