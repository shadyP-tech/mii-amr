"""Onboard ROS compressed-image QR scanner for Aufgabe 04.

This node is intentionally passive: it subscribes to a camera image topic,
detects QR text, validates QR identifiers, and prints/appends scan evidence.
It does not start missions or publish robot motion.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_texts_bgr
from scripts.aufgabe04.qr_scanning.scan_logger import append_qr_scan
from scripts.aufgabe04.qr_scanning.scan_processor import QRScanProcessor, ScanProcessorConfig
from scripts.aufgabe04.qr_scanning.topic_resolution import (
    DEFAULT_COMPRESSED_IMAGE_TOPIC,
    resolve_topic,
)


DEFAULT_QR_SCAN_LOG = Path("results/aufgabe04/qr_scans.csv")


class WarningThrottle:
    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = interval_sec
        self._last_warning_sec_by_key: dict[str, float] = {}

    def should_emit(self, key: str, now_sec: float) -> bool:
        last_warning_sec = self._last_warning_sec_by_key.get(key)
        if last_warning_sec is not None and now_sec - last_warning_sec < self.interval_sec:
            return False
        self._last_warning_sec_by_key[key] = now_sec
        return True


class OnboardQRScanner:
    def __init__(
        self,
        *,
        source: str,
        processor: QRScanProcessor,
        log_path: Path,
        cv2,
        numpy,
        frame_converter: Callable,
        stamp_reader: Callable,
        detector: Callable = detect_qr_texts_bgr,
        time_source: Callable[[], float] = time.time,
        printer: Callable[[str], None] = print,
        row_appender: Callable[[Path, dict], None] = append_qr_scan,
        once: bool = False,
    ):
        self.source = source
        self.processor = processor
        self.log_path = log_path
        self.cv2 = cv2
        self.numpy = numpy
        self.frame_converter = frame_converter
        self.stamp_reader = stamp_reader
        self.detector = detector
        self.time_source = time_source
        self.printer = printer
        self.row_appender = row_appender
        self.once = once
        self.stop_requested = False
        self.received_frame_count = 0
        self._warnings = WarningThrottle()

    def handle_compressed_image(self, msg) -> None:
        receipt_time_sec = self.time_source()
        self.received_frame_count += 1
        try:
            frame = self.frame_converter(msg, self.cv2, self.numpy)
        except ValueError as exc:
            self._warn("conversion", f"WARNING: {exc}", receipt_time_sec)
            return

        stamp_sec = self.stamp_reader(msg)
        raw_texts = self.detector(frame, self.cv2)
        outcomes = self.processor.process_texts(
            raw_texts,
            source=self.source,
            receipt_time_sec=receipt_time_sec,
            stamp_sec=stamp_sec,
        )
        for outcome in outcomes:
            if outcome.row is not None:
                self.row_appender(self.log_path, dict(outcome.row))
            if outcome.accepted:
                self.printer(f"QR scan: qr_id={outcome.qr_id} source={self.source}")
                if self.once:
                    self.stop_requested = True

    def warn_if_waiting_for_first_frame(self) -> None:
        now_sec = self.time_source()
        self._warn(
            "waiting_first_frame",
            f"WARNING: waiting for first frame on {self.source}",
            now_sec,
        )

    def _warn(self, key: str, message: str, now_sec: float) -> None:
        if self._warnings.should_emit(key, now_sec):
            self.printer(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Passive Aufgabe 04 ROS compressed-image QR scanner. Does not move the robot."
    )
    parser.add_argument(
        "--compressed-image-topic",
        default=DEFAULT_COMPRESSED_IMAGE_TOPIC,
        help=(
            "sensor_msgs/CompressedImage topic. Relative topics are resolved under "
            "--namespace; default resolves to /camera/image_raw/compressed."
        ),
    )
    parser.add_argument(
        "--namespace",
        default="",
        help="Optional robot namespace used only for relative image topics.",
    )
    parser.add_argument("--robot-id", default="Robot_Test_01")
    parser.add_argument("--qr-scan-log", type=Path, default=DEFAULT_QR_SCAN_LOG)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--min-repeat-sec", type=float, default=2.0)
    parser.add_argument("--max-frame-age-sec", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    return parser


def _load_runtime_dependencies():
    try:
        import cv2
        import numpy
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage

        from scripts.aufgabe04.perception.ros_image_adapter import (
            compressed_msg_stamp_sec,
            compressed_msg_to_bgr_frame,
        )
    except ImportError as exc:
        raise SystemExit(
            "ROS 2, OpenCV, numpy, and sensor_msgs are required to run the onboard QR scanner."
        ) from exc

    return {
        "cv2": cv2,
        "numpy": numpy,
        "rclpy": rclpy,
        "DurabilityPolicy": DurabilityPolicy,
        "HistoryPolicy": HistoryPolicy,
        "QoSProfile": QoSProfile,
        "ReliabilityPolicy": ReliabilityPolicy,
        "CompressedImage": CompressedImage,
        "compressed_msg_stamp_sec": compressed_msg_stamp_sec,
        "compressed_msg_to_bgr_frame": compressed_msg_to_bgr_frame,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_topic = resolve_topic(args.compressed_image_topic, args.namespace)
    deps = _load_runtime_dependencies()
    rclpy = deps["rclpy"]

    rclpy.init(args=None)
    node = rclpy.create_node("aufgabe04_onboard_qr_scanner")
    processor = QRScanProcessor(
        ScanProcessorConfig(
            robot_id=args.robot_id,
            run_id=args.run_id,
            min_repeat_sec=args.min_repeat_sec,
            max_frame_age_sec=args.max_frame_age_sec,
        )
    )
    scanner = OnboardQRScanner(
        source=resolved_topic,
        processor=processor,
        log_path=args.qr_scan_log,
        cv2=deps["cv2"],
        numpy=deps["numpy"],
        frame_converter=deps["compressed_msg_to_bgr_frame"],
        stamp_reader=deps["compressed_msg_stamp_sec"],
        once=args.once,
    )

    qos_profile = deps["QoSProfile"](
        depth=1,
        history=deps["HistoryPolicy"].KEEP_LAST,
        reliability=deps["ReliabilityPolicy"].BEST_EFFORT,
        durability=deps["DurabilityPolicy"].VOLATILE,
    )

    node.create_subscription(
        deps["CompressedImage"],
        resolved_topic,
        scanner.handle_compressed_image,
        qos_profile,
    )
    print("Aufgabe 04 onboard QR scanner: passive image-to-log mode.")
    print(f"Resolved image topic: {resolved_topic}")
    print(f"Robot namespace: {args.namespace.strip().strip('/') or '(none)'}")
    print(f"ROS_DOMAIN_ID: {os.environ.get('ROS_DOMAIN_ID', '(unset)')}")
    print(f"QR scan log: {args.qr_scan_log}")

    last_wait_warning_sec = 0.0
    try:
        while rclpy.ok() and not scanner.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
            now_sec = time.time()
            if scanner.received_frame_count == 0 and now_sec - last_wait_warning_sec >= 1.0:
                scanner.warn_if_waiting_for_first_frame()
                last_wait_warning_sec = now_sec
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
