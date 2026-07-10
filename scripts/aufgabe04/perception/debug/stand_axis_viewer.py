from __future__ import annotations

import argparse
import math
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.debug.color_mask_viewer import (
    FrameRead,
    RosCompressedImageTopicFrameSource,
    create_trackbars,
    current_track_range,
    palette_labels,
    print_palette,
    ranges_for_label,
)
from scripts.aufgabe04.perception.mask_processing import apply_morphology, build_mask_for_ranges
from scripts.aufgabe04.perception.camera_stand_observation import (
    CameraStandObservation,
    stand_axis_from_camera_yaw,
    write_camera_observation,
)
from scripts.aufgabe04.perception.ros_image_adapter import (
    compressed_msg_stamp_sec,
    compressed_msg_to_bgr_frame,
    raw_msg_to_bgr_frame,
)
from scripts.aufgabe04.perception.stand_side_classification import (
    StandSideClassification,
    classify_stand_side,
    color_confidence_for_estimate,
    _qr_scan_frames_for_estimate,
)
from scripts.aufgabe04.perception.stand_axis_lidar_roi import (
    DEFAULT_ROI_OBSERVATION_JSONL,
    ROI_OBSERVATION_SCHEMA_VERSION,
    ROI_OBSERVER_VERSION,
    PlainLaserScan,
    ScanConeRangeQuery,
    StandAxisLidarRoiObservation,
    camera_bearing_rad,
    image_center_x_to_bearing_rad,
    median_range_in_scan_cone,
    write_observation_jsonl,
)
from scripts.aufgabe04.perception.stand_axis_image import (
    ImagePoint,
    StandAxisImageEstimate,
    estimate_stand_axis_from_edges,
    estimate_stand_axis_from_mask,
)
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_texts_bgr
from scripts.aufgabe04.simulation.sim_qr_detector import detect_simulated_station_qr_bgr


WINDOW_FRAME = "aufgabe04/stand-axis"
WINDOW_MASK = "aufgabe04/stand-axis-mask"
WINDOW_EDGES = "aufgabe04/stand-axis-edges"
WINDOW_FACE_MASK = "aufgabe04/stand-axis-face-mask"
WINDOW_RECTANGLE_MASK = "aufgabe04/stand-axis-rectangle"


def build_parser() -> argparse.ArgumentParser:
    labels = palette_labels()
    parser = argparse.ArgumentParser(
        description=(
            "Debug-only live stand-axis viewer for a square stand face. "
            "Subscribes to compressed camera frames and does not move the robot."
        )
    )
    image_topics = parser.add_mutually_exclusive_group(required=True)
    image_topics.add_argument(
        "--compressed-image-topic",
        help="ROS 2 sensor_msgs/CompressedImage topic, e.g. /camera/image_raw/compressed.",
    )
    image_topics.add_argument(
        "--sim-raw-image-topic",
        help="Simulation only: Gazebo sensor_msgs/Image topic, e.g. /camera/image_raw.",
    )
    parser.add_argument("--resize", type=float, default=1.0)
    parser.add_argument("--color", choices=labels, default="green")
    parser.add_argument(
        "--axis-source",
        choices=("edges", "color-mask"),
        default="edges",
        help="edges is color/QR agnostic and uses the filled outer silhouette; color-mask keeps the HSV contour mode.",
    )
    parser.add_argument("--tune", action="store_true", help="Show HSV trackbars for color-mask axis debugging.")
    parser.add_argument("--print-palette", action="store_true")
    parser.add_argument("--print-every", type=int, default=15)
    parser.add_argument("--save-snapshot", type=Path)
    parser.add_argument("--no-morph", action="store_true")
    parser.add_argument("--morph-kernel", type=int, default=5)
    parser.add_argument("--close-iterations", type=int, default=2)
    parser.add_argument("--open-iterations", type=int, default=1)
    parser.add_argument("--min-area-px", type=float, default=250.0)
    parser.add_argument("--min-edge-height-px", type=float, default=8.0)
    parser.add_argument(
        "--edge-preprocess",
        choices=("outer-border", "gray"),
        default="outer-border",
        help="outer-border smooths QR/internal texture before Canny; gray uses the raw grayscale edge path.",
    )
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--edge-blur-kernel", type=int, default=5)
    parser.add_argument("--edge-dilate-iterations", type=int, default=1)
    parser.add_argument("--edge-close-kernel", type=int, default=5)
    parser.add_argument("--edge-close-iterations", type=int, default=1)
    parser.add_argument("--hough-threshold", type=int, default=20)
    parser.add_argument("--hough-min-line-length-px", type=int, default=12)
    parser.add_argument("--hough-max-line-gap-px", type=int, default=8)
    parser.add_argument("--min-boundary-line-length-px", type=float, default=35.0)
    parser.add_argument(
        "--face-width-fraction",
        type=float,
        default=0.60,
        help="Silhouette fallback: rows at this fraction of max width are treated as the broad square face.",
    )
    parser.add_argument(
        "--min-face-area-fraction",
        type=float,
        default=0.25,
        help="Reject face candidates smaller than this fraction of the largest external silhouette bounding area.",
    )
    parser.add_argument("--min-aspect-ratio", type=float, default=0.45)
    parser.add_argument("--max-aspect-ratio", type=float, default=1.80)
    parser.add_argument(
        "--side-color-confidence",
        type=float,
        default=0.20,
        help="Minimum selected-color fraction inside the detected square for classifying the plain color side.",
    )
    parser.add_argument(
        "--qr-crop-margin-px",
        type=int,
        default=8,
        help="Pixel margin around the detected square crop used for QR side detection.",
    )
    parser.add_argument(
        "--qr-decode-fps",
        type=float,
        default=2.0,
        help="Maximum background QR decode attempts per second. Use 0 for unlimited submissions.",
    )
    parser.add_argument(
        "--qr-result-ttl-sec",
        type=float,
        default=1.0,
        help="How long the viewer may reuse the last background QR result.",
    )
    parser.add_argument(
        "--no-qr-decode",
        action="store_true",
        help="Disable QR decoding in this debug viewer; color side classification still runs.",
    )
    parser.add_argument(
        "--front-face-to-qr-width-ratio",
        type=float,
        default=None,
        help="Known physical holder/front-face width divided by detected QR-code width. Enables QR-plane face expansion.",
    )
    parser.add_argument(
        "--median-window",
        type=int,
        default=7,
        help="Number of usable frames for median filtering ratio/yaw-proxy display. Use 1 to disable.",
    )
    parser.add_argument(
        "--stand-width-m",
        type=float,
        help="Physical square width in meters. Used with distance and optional camera intrinsics for yaw degrees.",
    )
    parser.add_argument(
        "--stand-face-size-m",
        type=float,
        help="Alias for --stand-width-m for a square stand face, e.g. 0.078 for a 7.8 cm x 7.8 cm frame.",
    )
    parser.add_argument(
        "--stand-distance-m",
        type=float,
        help="Approximate camera-to-stand center distance. Used as fallback when LiDAR distance is unavailable.",
    )
    parser.add_argument(
        "--camera-fx-px",
        type=float,
        help="Camera focal length fx in pixels for the processed image. If --resize is used, provide the resized fx or let the viewer scale this value.",
    )
    parser.add_argument(
        "--camera-fy-px",
        type=float,
        help="Camera focal length fy in pixels for the processed image. Defaults to --camera-fx-px.",
    )
    parser.add_argument(
        "--camera-cx-px",
        type=float,
        help="Camera principal point cx in pixels for the processed image. Defaults to the image center.",
    )
    parser.add_argument(
        "--camera-cy-px",
        type=float,
        help="Camera principal point cy in pixels for the processed image. Defaults to the image center.",
    )
    parser.add_argument(
        "--camera-fx-is-full-resolution",
        action="store_true",
        help="Treat --camera-fx-px/--camera-fy-px/--camera-cx-px/--camera-cy-px as original camera values before --resize and scale them by --resize internally.",
    )
    parser.add_argument(
        "--stand-head-depth-m",
        type=float,
        default=None,
        help="Physical stand head thickness/depth in meters, e.g. 0.007 for 0.7 cm. Kept with the model for cuboid diagnostics.",
    )
    parser.add_argument(
        "--stand-head-bottom-height-m",
        type=float,
        default=None,
        help="Height of the bottom of the square head above the floor in meters, e.g. 0.135 for 13.5 cm.",
    )
    parser.add_argument(
        "--scan-topic",
        default=None,
        help="Optional ROS 2 sensor_msgs/LaserScan topic used to estimate stand distance.",
    )
    parser.add_argument(
        "--use-lidar-distance",
        action="store_true",
        help="Use the latest LaserScan range as --stand-distance-m when available.",
    )
    parser.add_argument(
        "--lidar-bearing-rad",
        type=float,
        default=0.0,
        help="LaserScan bearing used for the distance estimate. 0 is straight ahead in the scan frame.",
    )
    parser.add_argument(
        "--lidar-bearing-source",
        choices=("fixed", "image-center"),
        default="fixed",
        help=(
            "fixed keeps the --lidar-bearing-rad cone. image-center is an opt-in debug mode that maps "
            "the detected stand rectangle center through camera intrinsics into an approximate LiDAR bearing."
        ),
    )
    parser.add_argument(
        "--camera-to-lidar-yaw-offset-rad",
        type=float,
        default=0.0,
        help="Measured yaw offset added to the image-derived camera bearing before sampling the LiDAR scan.",
    )
    parser.add_argument(
        "--lidar-cone-deg",
        type=float,
        default=10.0,
        help="Half-width cone around --lidar-bearing-rad for median range selection.",
    )
    parser.add_argument(
        "--lidar-min-samples",
        type=int,
        default=1,
        help="Minimum valid LaserScan samples required inside the selected cone.",
    )
    parser.add_argument(
        "--lidar-roi-log-jsonl",
        type=Path,
        default=DEFAULT_ROI_OBSERVATION_JSONL,
        help="Debug-only JSONL artifact for LiDAR ROI provenance.",
    )
    parser.add_argument(
        "--no-lidar-roi-log",
        action="store_true",
        help="Disable writing the debug LiDAR ROI JSONL artifact.",
    )
    parser.add_argument("--max-scan-age-sec", type=float, default=0.5)
    parser.add_argument(
        "--max-display-fps",
        type=float,
        default=20.0,
        help="Limit display/render rate while keeping only the newest ROS frame. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Process and write observations without OpenCV windows (useful in headless simulation).",
    )
    parser.add_argument(
        "--max-frame-age-sec",
        type=float,
        default=0.25,
        help="Drop incoming ROS image messages older than this. Use 0 to disable.",
    )
    parser.add_argument(
        "--observation-output-json",
        type=Path,
        help="Continuously replace this JSON with the latest valid camera stand observation.",
    )
    parser.add_argument("--map-frame", default="odom")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--robot-x", type=float, help="Map-frame robot x at the stationary pre-approach pose.")
    parser.add_argument("--robot-y", type=float, help="Map-frame robot y at the stationary pre-approach pose.")
    parser.add_argument("--stand-x", type=float, help="Detected map-frame stand center x.")
    parser.add_argument("--stand-y", type=float, help="Detected map-frame stand center y.")
    parser.add_argument(
        "--observation-write-hz",
        type=float,
        default=2.0,
        help="Maximum rate for replacing the latest valid observation JSON.",
    )
    parser.add_argument(
        "--display-mask",
        action="store_true",
        help="Also show the HSV color mask window. Mainly useful with --axis-source color-mask.",
    )
    parser.add_argument(
        "--display-edges",
        action="store_true",
        help="Also show the Canny/morphology edge image used by --axis-source edges.",
    )
    parser.add_argument(
        "--display-face-mask",
        action="store_true",
        help="Also show the filled upper-face geometry mask produced by the edge/silhouette path.",
    )
    return parser


@dataclass(frozen=True)
class _QrDecodeTask:
    frame: object
    estimate: StandAxisImageEstimate
    sequence: int
    submitted_sec: float


@dataclass(frozen=True)
class _QrDecodeResult:
    qr_texts: tuple[str, ...]
    sequence: int
    completed_sec: float


class RosSimulationRawImageTopicFrameSource:
    """Latest-frame source deliberately restricted to the explicit simulation CLI."""

    def __init__(self, topic: str, max_frame_age_sec: float):
        self.topic = topic
        self.max_frame_age_sec = max_frame_age_sec
        self.latest_message = None
        self.latest_stamp_sec = None
        self.latest_sequence = 0
        self._lock = threading.Lock()
        self._running = False
        self._spin_thread = None
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, qos_profile_sensor_data
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise SystemExit("Simulation raw image mode requires ROS 2 sensor_msgs.") from exc
        self.rclpy = rclpy
        self.owns_rclpy = not rclpy.ok()
        if self.owns_rclpy:
            rclpy.init(args=None)
        self.node = Node("aufgabe04_sim_raw_stand_axis_viewer")
        qos = QoSProfile(
            reliability=qos_profile_sensor_data.reliability,
            durability=qos_profile_sensor_data.durability,
            history=qos_profile_sensor_data.history,
            depth=1,
        )
        self.subscription = self.node.create_subscription(Image, topic, self._on_image, qos)

    def _on_image(self, msg) -> None:
        stamp_sec = compressed_msg_stamp_sec(msg)
        with self._lock:
            self.latest_message = msg
            self.latest_stamp_sec = stamp_sec
            self.latest_sequence += 1

    def start(self) -> None:
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _spin_loop(self) -> None:
        while self._running and self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def read(self) -> FrameRead:
        with self._lock:
            message = self.latest_message
            stamp_sec = self.latest_stamp_sec
            sequence = self.latest_sequence
        if message is None:
            return FrameRead(False, message=f"waiting for first frame on {self.topic}", waiting=True)
        return FrameRead(True, frame=message, stamp_sec=stamp_sec, sequence=sequence)

    def release(self) -> None:
        self._running = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
        self.node.destroy_node()
        if self.owns_rclpy and self.rclpy.ok():
            self.rclpy.shutdown()


class BackgroundQrDecoder:
    def __init__(
        self,
        *,
        cv2,
        numpy,
        detector,
        crop_margin_px: int,
        max_decode_fps: float,
        result_ttl_sec: float,
    ) -> None:
        self._cv2 = cv2
        self._numpy = numpy
        self._detector = detector
        self._crop_margin_px = crop_margin_px
        self._min_submit_period_sec = 0.0 if max_decode_fps <= 0.0 else 1.0 / max_decode_fps
        self._result_ttl_sec = max(0.0, result_ttl_sec)
        self._condition = threading.Condition()
        self._task: _QrDecodeTask | None = None
        self._result = _QrDecodeResult((), 0, 0.0)
        self._busy = False
        self._stopped = False
        self._last_submit_sec = 0.0
        self._thread = threading.Thread(target=self._run, name="stand-axis-qr-decoder", daemon=True)
        self._thread.start()

    def submit_latest(self, frame, estimate: StandAxisImageEstimate, sequence: int, now_sec: float) -> None:
        if now_sec - self._last_submit_sec < self._min_submit_period_sec:
            return
        with self._condition:
            if self._stopped or self._busy or self._task is not None:
                return
            self._task = _QrDecodeTask(frame.copy(), estimate, sequence, now_sec)
            self._busy = True
            self._last_submit_sec = now_sec
            self._condition.notify()

    def latest_texts(self, now_sec: float) -> tuple[str, ...]:
        with self._condition:
            result = self._result
        if self._result_ttl_sec > 0.0 and now_sec - result.completed_sec > self._result_ttl_sec:
            return ()
        return result.qr_texts

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._task is None and not self._stopped:
                    self._condition.wait()
                if self._stopped:
                    return
                task = self._task
                self._task = None

            qr_texts: tuple[str, ...] = ()
            if task is not None:
                for qr_frame in _qr_scan_frames_for_estimate(
                    self._cv2,
                    self._numpy,
                    task.frame,
                    task.estimate,
                    margin_px=self._crop_margin_px,
                ):
                    qr_texts = self._detector(qr_frame, self._cv2)
                    if qr_texts:
                        break

            with self._condition:
                if task is not None:
                    self._result = _QrDecodeResult(qr_texts, task.sequence, time.time())
                self._busy = False
                self._condition.notify()


class RosLaserScanRangeSource:
    def __init__(
        self,
        *,
        topic: str,
        max_scan_age_sec: float,
    ) -> None:
        self.topic = topic
        self.max_scan_age_sec = max_scan_age_sec
        self._lock = threading.Lock()
        self._latest_scan: PlainLaserScan | None = None
        self._running = False
        self._spin_thread = None

        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, qos_profile_sensor_data
            from sensor_msgs.msg import LaserScan
        except ImportError as exc:
            raise SystemExit(
                "LaserScan distance mode requires rclpy and sensor_msgs. "
                "Source the ROS 2 Humble and TurtleBot workspaces first."
            ) from exc

        self.rclpy = rclpy
        self.owns_rclpy = not rclpy.ok()
        if self.owns_rclpy:
            rclpy.init(args=None)
        self.executor = SingleThreadedExecutor()

        class StandAxisScanNode(Node):
            pass

        self.node = StandAxisScanNode("aufgabe04_stand_axis_lidar_range")
        qos_profile = QoSProfile(
            reliability=qos_profile_sensor_data.reliability,
            durability=qos_profile_sensor_data.durability,
            history=qos_profile_sensor_data.history,
            depth=1,
        )
        self.subscription = self.node.create_subscription(
            LaserScan,
            topic,
            self._on_scan,
            qos_profile,
        )
        self.executor.add_node(self.node)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _spin_loop(self) -> None:
        while self._running and self.rclpy.ok():
            self.executor.spin_once(timeout_sec=0.05)

    def _on_scan(self, msg) -> None:
        header = getattr(msg, "header", None)
        frame_id = str(getattr(header, "frame_id", "") or "")
        stamp = getattr(header, "stamp", None)
        scan_stamp_sec = _stamp_to_sec(stamp)
        scan = PlainLaserScan(
            ranges=tuple(float(value) for value in msg.ranges),
            angle_min=float(msg.angle_min),
            angle_increment=float(msg.angle_increment),
            range_min=float(msg.range_min),
            range_max=float(msg.range_max),
            scan_frame_id=frame_id,
            scan_stamp_sec=scan_stamp_sec,
            receipt_sec=time.time(),
        )
        with self._lock:
            self._latest_scan = scan

    def latest_scan(self) -> PlainLaserScan | None:
        with self._lock:
            return self._latest_scan

    def release(self) -> None:
        self._running = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        self.executor.shutdown()
        if self.owns_rclpy and self.rclpy.ok():
            self.rclpy.shutdown()


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _stamp_to_sec(stamp) -> float | None:
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return float(sec) + float(nanosec) / 1_000_000_000.0


def _rectangle_center_x_px(corners: Sequence[ImagePoint] | None) -> float | None:
    if not corners:
        return None
    return sum(point.u_px for point in corners) / len(corners)


def _empty_scan_query(
    *,
    bearing_rad: float,
    cone_half_angle_rad: float,
    reason: str,
) -> ScanConeRangeQuery:
    return ScanConeRangeQuery(
        distance_m=None,
        selected_sample_count=0,
        rejection_reason=reason,
        bearing_rad=bearing_rad,
        cone_half_angle_rad=cone_half_angle_rad,
        scan_frame_id="",
        scan_stamp_sec=None,
        scan_age_sec=None,
    )


def _query_lidar_distance(
    *,
    lidar_source: RosLaserScanRangeSource | None,
    bearing_source: str,
    fixed_bearing_rad: float,
    cone_half_angle_rad: float,
    max_scan_age_sec: float,
    min_sample_count: int,
    estimate: StandAxisImageEstimate | None,
    camera_fx_px: float | None,
    camera_cx_px: float | None,
    camera_to_lidar_yaw_offset_rad: float,
    now_sec: float,
) -> tuple[ScanConeRangeQuery | None, float | None, float | None, str]:
    if lidar_source is None:
        return None, None, None, "no_lidar_source"

    scan = lidar_source.latest_scan()
    camera_bearing = None
    rect_center_x = None
    query_bearing = fixed_bearing_rad
    fallback_source = "fixed_bearing"

    if bearing_source == "image-center":
        rect_center_x = _rectangle_center_x_px(estimate.corners if estimate is not None else None)
        if estimate is None or not estimate.usable:
            return (
                _empty_scan_query(
                    bearing_rad=fixed_bearing_rad,
                    cone_half_angle_rad=cone_half_angle_rad,
                    reason="unusable_estimate",
                ),
                None,
                rect_center_x,
                "unusable_estimate",
            )
        if rect_center_x is None:
            return (
                _empty_scan_query(
                    bearing_rad=fixed_bearing_rad,
                    cone_half_angle_rad=cone_half_angle_rad,
                    reason="missing_rectangle_center",
                ),
                None,
                rect_center_x,
                "missing_rectangle_center",
            )
        if camera_fx_px is None or camera_cx_px is None:
            return (
                _empty_scan_query(
                    bearing_rad=fixed_bearing_rad,
                    cone_half_angle_rad=cone_half_angle_rad,
                    reason="missing_camera_intrinsics",
                ),
                None,
                rect_center_x,
                "missing_camera_intrinsics",
            )
        try:
            camera_bearing = camera_bearing_rad(
                rect_center_x,
                camera_fx_px=camera_fx_px,
                camera_cx_px=camera_cx_px,
            )
            query_bearing = image_center_x_to_bearing_rad(
                rect_center_x,
                camera_fx_px=camera_fx_px,
                camera_cx_px=camera_cx_px,
                camera_to_lidar_yaw_offset_rad=camera_to_lidar_yaw_offset_rad,
            )
        except ValueError as exc:
            return (
                _empty_scan_query(
                    bearing_rad=fixed_bearing_rad,
                    cone_half_angle_rad=cone_half_angle_rad,
                    reason=str(exc),
                ),
                None,
                rect_center_x,
                "invalid_camera_intrinsics",
            )
        fallback_source = "image_center"

    query = median_range_in_scan_cone(
        scan,
        bearing_rad=query_bearing,
        cone_half_angle_rad=cone_half_angle_rad,
        now_sec=now_sec,
        max_scan_age_sec=max_scan_age_sec,
        min_sample_count=min_sample_count,
    )
    return query, camera_bearing, rect_center_x, fallback_source


def _write_lidar_roi_observation(
    *,
    path: Path,
    image_topic: str,
    image_stamp_sec: float | None,
    scan_topic: str,
    query: ScanConeRangeQuery,
    rect_center_x_px: float | None,
    camera_fx_px: float | None,
    camera_cx_px: float | None,
    camera_bearing_rad_value: float | None,
    bearing_source: str,
    fallback_source: str,
    estimate: StandAxisImageEstimate,
    observed_at_sec: float,
) -> None:
    observation = StandAxisLidarRoiObservation(
        schema_version=ROI_OBSERVATION_SCHEMA_VERSION,
        observer_version=ROI_OBSERVER_VERSION,
        observed_at_sec=observed_at_sec,
        image_topic=image_topic,
        image_stamp_sec=image_stamp_sec,
        scan_topic=scan_topic,
        scan_frame_id=query.scan_frame_id,
        scan_stamp_sec=query.scan_stamp_sec,
        scan_age_sec=query.scan_age_sec,
        rect_center_x_px=rect_center_x_px,
        camera_fx_px=camera_fx_px,
        camera_cx_px=camera_cx_px,
        camera_bearing_rad=camera_bearing_rad_value,
        lidar_bearing_rad=query.bearing_rad,
        bearing_source=bearing_source,
        cone_half_angle_rad=query.cone_half_angle_rad,
        selected_sample_count=query.selected_sample_count,
        used_distance_m=query.distance_m,
        fallback_source=fallback_source,
        rejection_reason=query.rejection_reason,
        estimate_source=estimate.source,
        estimate_usable=bool(estimate.usable),
    )
    write_observation_jsonl(path, (observation,))


def display_side_label(side: StandSideClassification, estimate: StandAxisImageEstimate) -> str:
    if side.side == "qr_code_side" or estimate.source == "edge_qr_scaled_front":
        return "front"
    if side.side == "basic_color_side" or estimate.source == "edge_plain_face_stem_anchor":
        return "back"
    return "unknown"


def format_optional_float(value: float | None, *, precision: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


def print_status_line(
    estimate: StandAxisImageEstimate,
    side: StandSideClassification,
    *,
    lidar_distance_m: float | None,
    stand_distance_m: float | None,
    lidar_query: ScanConeRangeQuery | None,
    lidar_bearing_source: str,
    qr_texts: tuple[str, ...],
) -> None:
    side_label = display_side_label(side, estimate)
    angle_text = format_optional_float(estimate.yaw_deg, precision=1, suffix="deg")
    proxy_text = format_optional_float(estimate.yaw_proxy, precision=3)
    lidar_text = format_optional_float(lidar_distance_m, precision=3, suffix="m")
    distance_text = format_optional_float(stand_distance_m, precision=3, suffix="m")
    if lidar_query is None:
        lidar_roi_text = "n/a"
    else:
        reason = lidar_query.rejection_reason or "ok"
        lidar_roi_text = (
            f"{lidar_bearing_source}:bearing={lidar_query.bearing_rad:.3f}rad "
            f"samples={lidar_query.selected_sample_count} reason={reason}"
        )
    ratio_text = format_optional_float(estimate.height_ratio, precision=3)
    print(
        f"stand_axis source={estimate.source} mode={estimate.mode} usable={estimate.usable} "
        f"angle={angle_text} proxy={proxy_text} ratio={ratio_text} "
        f"side={side_label} raw_side={side.side} reason={side.reason} "
        f"lidar_distance={lidar_text} used_distance={distance_text} "
        f"lidar_roi={lidar_roi_text} "
        f"left_px={estimate.left_height_px:.1f} right_px={estimate.right_height_px:.1f} "
        f"qr_texts={list(qr_texts)}"
    )


def annotate_frame(
    cv2,
    frame,
    estimate: StandAxisImageEstimate,
    side: StandSideClassification,
    filtered_ratio,
    filtered_proxy,
    age_ms,
) -> None:
    if estimate.corners is not None:
        corners = estimate.corners
        int_points = [(int(round(point.u_px)), int(round(point.v_px))) for point in corners]
        for start, end in zip(int_points, int_points[1:] + int_points[:1]):
            cv2.line(frame, start, end, (0, 255, 255), 2)
        cv2.line(frame, int_points[0], int_points[3], (0, 180, 255), 3)
        cv2.line(frame, int_points[1], int_points[2], (255, 180, 0), 3)
        for label, point in zip(("TL", "TR", "BR", "BL"), int_points):
            cv2.putText(frame, label, point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if estimate.axis_line is not None:
        start, end = estimate.axis_line
        start_point = (int(round(start.u_px)), int(round(start.v_px)))
        end_point = (int(round(end.u_px)), int(round(end.v_px)))
        cv2.line(frame, start_point, end_point, (0, 0, 255), 3)
        cv2.putText(frame, "EDGE-ON", start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if estimate.mode == "face_visible" and estimate.usable:
        yaw_text = f" camera_yaw={estimate.yaw_deg:.1f}deg" if estimate.yaw_deg is not None else ""
        line1 = (
            f"camera axis rot proxy={estimate.yaw_proxy:+.3f}{yaw_text}"
        )
        line2 = (
            f"L={estimate.left_height_px:.1f}px R={estimate.right_height_px:.1f}px "
            f"ratio={estimate.height_ratio:.3f}"
        )
        line3 = f"closer={estimate.closer_side} stand_side={side.side}"
        if filtered_ratio is not None and filtered_proxy is not None:
            line3 += f" med_ratio={filtered_ratio:.3f} med_proxy={filtered_proxy:+.3f}"
        color = (0, 255, 0)
    elif estimate.mode == "edge_on" and estimate.usable:
        line1 = "camera axis approx side-on / 90deg"
        line2 = f"edge-on line height={estimate.left_height_px:.1f}px"
        line3 = f"ratio unavailable stand_side={side.side}"
        color = (0, 0, 255)
    else:
        line1 = f"no usable stand axis: {estimate.reason}"
        line2 = f"source={estimate.source} area={estimate.contour_area_px:.0f}px"
        line3 = f"camera axis rotation unavailable stand_side={side.side}"
        color = (0, 200, 255)

    cv2.putText(frame, line1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    cv2.putText(frame, line2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2)
    cv2.putText(frame, line3, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
    if age_ms is not None:
        cv2.putText(
            frame,
            f"age={age_ms:.0f}ms",
            (12, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


def save_snapshot(cv2, directory: Path, frame, mask) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    frame_path = directory / f"{stamp}_stand_axis_frame.png"
    mask_path = directory / f"{stamp}_stand_axis_mask.png"
    cv2.imwrite(str(frame_path), frame)
    cv2.imwrite(str(mask_path), mask)
    print(f"saved snapshot: {frame_path} {mask_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.observation_output_json is not None:
        required_geometry = (args.robot_x, args.robot_y, args.stand_x, args.stand_y)
        if any(value is None for value in required_geometry):
            raise SystemExit(
                "--observation-output-json requires --robot-x, --robot-y, --stand-x, and --stand-y"
            )
        if args.observation_write_hz <= 0.0:
            raise SystemExit("--observation-write-hz must be positive")
    stand_width_m = args.stand_face_size_m if args.stand_face_size_m is not None else args.stand_width_m
    camera_fx_px = args.camera_fx_px
    camera_fy_px = args.camera_fy_px
    camera_cx_px = args.camera_cx_px
    camera_cy_px = args.camera_cy_px
    if camera_fx_px is not None and args.camera_fx_is_full_resolution:
        camera_fx_px *= args.resize
    if camera_fy_px is not None and args.camera_fx_is_full_resolution:
        camera_fy_px *= args.resize
    if camera_cx_px is not None and args.camera_fx_is_full_resolution:
        camera_cx_px *= args.resize
    if camera_cy_px is not None and args.camera_fx_is_full_resolution:
        camera_cy_px *= args.resize

    try:
        import cv2
        import numpy
    except ImportError as exc:
        raise SystemExit("OpenCV and numpy are required for the stand-axis viewer.") from exc

    selected_ranges = list(ranges_for_label(args.color))
    if args.print_palette:
        print_palette(selected_ranges)

    if args.sim_raw_image_topic:
        frame_source = RosSimulationRawImageTopicFrameSource(
            args.sim_raw_image_topic, args.max_frame_age_sec
        )
        image_topic = args.sim_raw_image_topic
    else:
        frame_source = RosCompressedImageTopicFrameSource(
            args.compressed_image_topic, args.max_frame_age_sec
        )
        image_topic = args.compressed_image_topic
    frame_source.start()
    lidar_source = None
    if args.use_lidar_distance:
        if not args.scan_topic:
            raise SystemExit("--use-lidar-distance requires --scan-topic")
        lidar_source = RosLaserScanRangeSource(
            topic=args.scan_topic,
            max_scan_age_sec=args.max_scan_age_sec,
        )
        lidar_source.start()
    qr_decoder = None if args.no_qr_decode or args.sim_raw_image_topic else BackgroundQrDecoder(
        cv2=cv2,
        numpy=numpy,
        detector=detect_qr_texts_bgr,
        crop_margin_px=args.qr_crop_margin_px,
        max_decode_fps=args.qr_decode_fps,
        result_ttl_sec=args.qr_result_ttl_sec,
    )
    if args.tune:
        create_trackbars(cv2, selected_ranges[0])

    print("Aufgabe 04 stand-axis viewer: debug-only, no robot motion, no /cmd_vel.")
    print("Face-visible mode: + proxy means left image edge is closer/taller; - means right edge is closer/taller.")
    print("Edge-on mode: reports approximate side-on / 90deg and does not compute a ratio.")
    print("Keys: ESC/q quit, p print ColorRange, s save snapshot.")

    ratio_window = deque(maxlen=max(1, args.median_window))
    proxy_window = deque(maxlen=max(1, args.median_window))
    frame_count = 0
    last_processed_sequence = 0
    last_display_sec = 0.0
    last_waiting_message_sec = 0.0
    last_observation_write_sec = 0.0

    try:
        while True:
            now = time.time()
            if args.max_display_fps > 0.0:
                min_period = 1.0 / args.max_display_fps
                sleep_sec = min_period - (now - last_display_sec)
                if sleep_sec > 0.0:
                    time.sleep(min(sleep_sec, 0.05))

            read = frame_source.read()
            if not read.ok:
                if read.waiting:
                    now = time.time()
                    if now - last_waiting_message_sec >= 1.0:
                        print(f"WARNING: {read.message}")
                        last_waiting_message_sec = now
                    key = 0 if args.headless else cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    continue
                print(f"WARNING: {read.message}")
                break
            if read.sequence == last_processed_sequence:
                key = 0 if args.headless else cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue
            last_processed_sequence = read.sequence

            try:
                frame = (
                    raw_msg_to_bgr_frame(read.frame, cv2, numpy)
                    if args.sim_raw_image_topic
                    else compressed_msg_to_bgr_frame(read.frame, cv2, numpy)
                )
            except ValueError as exc:
                print(f"WARNING: {exc}")
                continue
            last_display_sec = time.time()
            age_ms = (last_display_sec - read.stamp_sec) * 1000.0 if read.stamp_sec is not None else None

            if args.resize != 1.0:
                frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize)
            query_camera_cx_px = camera_cx_px if camera_cx_px is not None else float(frame.shape[1]) / 2.0

            mask = None
            edges = None
            face_mask = None
            rectangle_mask = None
            lidar_query = None
            lidar_camera_bearing_rad = None
            lidar_rect_center_x_px = None
            lidar_fallback_source = "none"
            if args.lidar_bearing_source == "fixed":
                lidar_query, lidar_camera_bearing_rad, lidar_rect_center_x_px, lidar_fallback_source = (
                    _query_lidar_distance(
                        lidar_source=lidar_source,
                        bearing_source=args.lidar_bearing_source,
                        fixed_bearing_rad=args.lidar_bearing_rad,
                        cone_half_angle_rad=math.radians(args.lidar_cone_deg),
                        max_scan_age_sec=args.max_scan_age_sec,
                        min_sample_count=args.lidar_min_samples,
                        estimate=None,
                        camera_fx_px=camera_fx_px,
                        camera_cx_px=query_camera_cx_px,
                        camera_to_lidar_yaw_offset_rad=args.camera_to_lidar_yaw_offset_rad,
                        now_sec=last_display_sec,
                    )
                )
            lidar_distance_m = lidar_query.distance_m if lidar_query is not None else None
            stand_distance_m = lidar_distance_m if lidar_distance_m is not None else args.stand_distance_m
            active_ranges = selected_ranges
            if args.axis_source == "color-mask" or args.display_mask or args.tune:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                active_ranges = [current_track_range(cv2, args.color)] if args.tune else selected_ranges
                mask = build_mask_for_ranges(cv2, numpy, hsv, active_ranges)
                if not args.no_morph:
                    mask = apply_morphology(
                        cv2,
                        mask,
                        kernel_size=args.morph_kernel,
                        close_iterations=args.close_iterations,
                        open_iterations=args.open_iterations,
                    )

            if args.axis_source == "edges":
                estimate, edge_artifacts = estimate_stand_axis_from_edges(
                    cv2,
                    frame,
                    edge_preprocess=args.edge_preprocess.replace("-", "_"),
                    blur_kernel=args.edge_blur_kernel,
                    canny_low=args.canny_low,
                    canny_high=args.canny_high,
                    dilate_iterations=args.edge_dilate_iterations,
                    close_kernel=args.edge_close_kernel,
                    close_iterations=args.edge_close_iterations,
                    hough_threshold=args.hough_threshold,
                    hough_min_line_length_px=args.hough_min_line_length_px,
                    hough_max_line_gap_px=args.hough_max_line_gap_px,
                    min_boundary_line_length_px=args.min_boundary_line_length_px,
                    face_width_fraction=args.face_width_fraction,
                    min_face_area_fraction=args.min_face_area_fraction,
                    min_area_px=args.min_area_px,
                    min_edge_height_px=args.min_edge_height_px,
                    min_aspect_ratio=args.min_aspect_ratio,
                    max_aspect_ratio=args.max_aspect_ratio,
                    front_face_to_qr_width_ratio=args.front_face_to_qr_width_ratio,
                    stand_width_m=stand_width_m,
                    stand_distance_m=stand_distance_m,
                    camera_fx_px=camera_fx_px,
                    camera_fy_px=camera_fy_px,
                    camera_cx_px=camera_cx_px,
                    camera_cy_px=camera_cy_px,
                    stand_depth_m=args.stand_head_depth_m,
                    stand_head_bottom_height_m=args.stand_head_bottom_height_m,
                )
                edges = edge_artifacts.edges
                face_mask = edge_artifacts.face_mask
                rectangle_mask = edge_artifacts.rectangle_mask
            else:
                estimate = estimate_stand_axis_from_mask(
                    cv2,
                    mask,
                    min_area_px=args.min_area_px,
                    min_edge_height_px=args.min_edge_height_px,
                    stand_width_m=stand_width_m,
                    stand_distance_m=stand_distance_m,
                    camera_fx_px=camera_fx_px,
                    camera_fy_px=camera_fy_px,
                    camera_cx_px=camera_cx_px,
                    camera_cy_px=camera_cy_px,
                    stand_depth_m=args.stand_head_depth_m,
                    stand_head_bottom_height_m=args.stand_head_bottom_height_m,
                )
            if mask is None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                active_ranges = [current_track_range(cv2, args.color)] if args.tune else selected_ranges
                mask = build_mask_for_ranges(cv2, numpy, hsv, active_ranges)
                if not args.no_morph:
                    mask = apply_morphology(
                        cv2,
                        mask,
                        kernel_size=args.morph_kernel,
                        close_iterations=args.close_iterations,
                        open_iterations=args.open_iterations,
                    )
            if args.lidar_bearing_source == "image-center":
                lidar_query, lidar_camera_bearing_rad, lidar_rect_center_x_px, lidar_fallback_source = (
                    _query_lidar_distance(
                        lidar_source=lidar_source,
                        bearing_source=args.lidar_bearing_source,
                        fixed_bearing_rad=args.lidar_bearing_rad,
                        cone_half_angle_rad=math.radians(args.lidar_cone_deg),
                        max_scan_age_sec=args.max_scan_age_sec,
                        min_sample_count=args.lidar_min_samples,
                        estimate=estimate,
                        camera_fx_px=camera_fx_px,
                        camera_cx_px=query_camera_cx_px,
                        camera_to_lidar_yaw_offset_rad=args.camera_to_lidar_yaw_offset_rad,
                        now_sec=last_display_sec,
                    )
                )
                lidar_distance_m = lidar_query.distance_m if lidar_query is not None else None
                stand_distance_m = lidar_distance_m if lidar_distance_m is not None else args.stand_distance_m
            if (
                lidar_query is not None
                and args.use_lidar_distance
                and not args.no_lidar_roi_log
                and args.lidar_roi_log_jsonl is not None
            ):
                _write_lidar_roi_observation(
                    path=args.lidar_roi_log_jsonl,
                    image_topic=args.compressed_image_topic,
                    image_stamp_sec=read.stamp_sec,
                    scan_topic=args.scan_topic or "",
                    query=lidar_query,
                    rect_center_x_px=lidar_rect_center_x_px,
                    camera_fx_px=camera_fx_px,
                    camera_cx_px=query_camera_cx_px,
                    camera_bearing_rad_value=lidar_camera_bearing_rad,
                    bearing_source=args.lidar_bearing_source,
                    fallback_source=lidar_fallback_source,
                    estimate=estimate,
                    observed_at_sec=last_display_sec,
                )
            color_confidence = color_confidence_for_estimate(cv2, numpy, mask, estimate)
            sim_qr_detection = None
            if args.sim_raw_image_topic and not args.no_qr_decode:
                sim_qr_detection = detect_simulated_station_qr_bgr(
                    frame,
                    cv2,
                    camera_fx_px=camera_fx_px,
                    camera_fy_px=camera_fy_px,
                    camera_cx_px=camera_cx_px,
                    camera_cy_px=camera_cy_px,
                )
                qr_texts = () if sim_qr_detection is None else (sim_qr_detection.station_id,)
            elif qr_decoder is not None:
                qr_decoder.submit_latest(frame, estimate, read.sequence, time.time())
                qr_texts = qr_decoder.latest_texts(time.time())
            else:
                qr_texts = ()
            observation_distance_m = None
            if args.observation_output_json is not None:
                observation_distance_m = math.hypot(
                    args.stand_x - args.robot_x,
                    args.stand_y - args.robot_y,
                )
            allow_color_only = (
                not args.sim_raw_image_topic
                or (
                    observation_distance_m is not None
                    and observation_distance_m <= 0.33
                )
            )
            side = classify_stand_side(
                qr_texts=qr_texts,
                color_confidence=color_confidence,
                min_color_confidence=args.side_color_confidence,
                allow_color_only=allow_color_only,
            )
            qr_face_yaw_rad = (
                None if sim_qr_detection is None else sim_qr_detection.face_yaw_rad
            )
            orientation_yaw_rad = (
                qr_face_yaw_rad
                if side.side == "qr_code_side" and qr_face_yaw_rad is not None
                else (None if estimate.yaw_deg is None else math.radians(estimate.yaw_deg))
            )
            if (
                args.observation_output_json is not None
                and estimate.usable
                and estimate.mode == "face_visible"
                and orientation_yaw_rad is not None
                and side.side in ("qr_code_side", "basic_color_side")
                and side.confidence >= 0.60
                and time.time() - last_observation_write_sec >= 1.0 / args.observation_write_hz
            ):
                pnp_calibrated = all(value is not None for value in (
                    stand_width_m, camera_fx_px, camera_fy_px,
                ))
                metric_fallback = stand_width_m is not None and stand_distance_m is not None
                if pnp_calibrated or metric_fallback:
                    # Staleness checks use the workstation wall clock, not Gazebo's /clock epoch.
                    observed_at_sec = time.time()
                    observation = CameraStandObservation(
                        schema_version=1,
                        observed_at_sec=observed_at_sec,
                        image_topic=image_topic,
                        camera_frame=args.camera_frame,
                        map_frame=args.map_frame,
                        robot_x_m=args.robot_x,
                        robot_y_m=args.robot_y,
                        stand_x_m=args.stand_x,
                        stand_y_m=args.stand_y,
                        stand_axis_rad=stand_axis_from_camera_yaw(
                            robot_x_m=args.robot_x,
                            robot_y_m=args.robot_y,
                            stand_x_m=args.stand_x,
                            stand_y_m=args.stand_y,
                            camera_yaw_rad=orientation_yaw_rad,
                        ),
                        axis_confidence=(
                            max(0.85, 1.0 - sim_qr_detection.mismatch_fraction)
                            if sim_qr_detection is not None and qr_face_yaw_rad is not None
                            else (0.85 if pnp_calibrated else 0.65)
                        ),
                        side=side.side,
                        side_confidence=side.confidence,
                        qr_texts=tuple(qr_texts),
                    )
                    write_camera_observation(args.observation_output_json, observation)
                    last_observation_write_sec = time.time()
            if estimate.mode == "face_visible" and estimate.usable:
                ratio_window.append(float(estimate.height_ratio))
                proxy_window.append(float(estimate.yaw_proxy))
            filtered_ratio = statistics.median(ratio_window) if ratio_window else None
            filtered_proxy = statistics.median(proxy_window) if proxy_window else None

            annotated = frame.copy()
            annotate_frame(cv2, annotated, estimate, side, filtered_ratio, filtered_proxy, age_ms)

            frame_count += 1
            if args.headless and args.save_snapshot is not None and frame_count == 1:
                debug_image = edges if args.axis_source == "edges" and edges is not None else mask
                save_snapshot(cv2, args.save_snapshot, frame, debug_image)
            if args.print_every > 0 and frame_count % args.print_every == 0:
                print_status_line(
                    estimate,
                    side,
                    lidar_distance_m=lidar_distance_m,
                    stand_distance_m=stand_distance_m,
                    lidar_query=lidar_query,
                    lidar_bearing_source=args.lidar_bearing_source,
                    qr_texts=qr_texts,
                )

            if not args.headless:
                cv2.imshow(WINDOW_FRAME, annotated)
                if args.display_mask and mask is not None:
                    cv2.imshow(WINDOW_MASK, mask)
                if args.display_edges and edges is not None:
                    cv2.imshow(WINDOW_EDGES, edges)
                if args.display_face_mask and face_mask is not None:
                    cv2.imshow(WINDOW_FACE_MASK, face_mask)
                if args.display_face_mask and rectangle_mask is not None:
                    cv2.imshow(WINDOW_RECTANGLE_MASK, rectangle_mask)

            key = 0 if args.headless else cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("p"):
                print_palette(active_ranges)
            if key == ord("s") and args.save_snapshot is not None:
                if args.display_face_mask and rectangle_mask is not None:
                    debug_image = rectangle_mask
                elif args.display_face_mask and face_mask is not None:
                    debug_image = face_mask
                else:
                    debug_image = edges if args.axis_source == "edges" and edges is not None else mask
                save_snapshot(cv2, args.save_snapshot, annotated, debug_image)
    finally:
        if qr_decoder is not None:
            qr_decoder.stop()
        if lidar_source is not None:
            lidar_source.release()
        frame_source.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
