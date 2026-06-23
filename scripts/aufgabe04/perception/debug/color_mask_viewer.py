from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from scripts.aufgabe04.perception.color_classifier import (
    DEFAULT_STAND_PALETTE,
)
from scripts.aufgabe04.perception.models import (
    ColorRange,
)


WINDOW_FRAME = "aufgabe04/frame"
WINDOW_MASK = "aufgabe04/mask"
WINDOW_PREVIEW = "aufgabe04/preview"
WINDOW_CONTROLS = "aufgabe04/controls"


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class FrameRead:
    ok: bool
    frame: object | None = None
    message: str | None = None
    waiting: bool = False
    stamp_sec: float | None = None


@dataclass(frozen=True)
class LiveMaskClassification:
    label: str
    confidence: float
    matched_pixels: int
    total_pixels: int


def parse_roi(value: str) -> Rect:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must use x,y,w,h")
    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--roi values must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--roi width and height must be positive")
    return Rect(x, y, width, height)


def clamp_roi(roi: Rect, frame_shape: Sequence[int]) -> Rect:
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    x = max(0, min(roi.x, frame_w))
    y = max(0, min(roi.y, frame_h))
    width = max(0, min(roi.width, frame_w - x))
    height = max(0, min(roi.height, frame_h - y))
    return Rect(x, y, width, height)


def palette_labels(palette: Sequence[ColorRange] = DEFAULT_STAND_PALETTE) -> Tuple[str, ...]:
    labels: List[str] = []
    for color_range in palette:
        if color_range.label not in labels:
            labels.append(color_range.label)
    return tuple(labels)


def ranges_for_label(label: str, palette: Sequence[ColorRange] = DEFAULT_STAND_PALETTE) -> Tuple[ColorRange, ...]:
    selected = tuple(color_range for color_range in palette if color_range.label == label)
    if not selected:
        raise ValueError(f"unknown color label: {label}")
    return selected


def image_msg_to_bgr_frame(msg, numpy):
    encoding = str(getattr(msg, "encoding", "")).strip().lower()
    if encoding not in {"bgr8", "bgr888", "rgb8", "rgb888", "8uc3"}:
        raise ValueError(f"unsupported ROS image encoding: {getattr(msg, 'encoding', '')!r}")

    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    channels = 3
    min_step = width * channels
    if height <= 0 or width <= 0:
        raise ValueError("ROS image dimensions must be positive")
    if step < min_step:
        raise ValueError("ROS image step is smaller than width * channels")

    data = numpy.frombuffer(msg.data, dtype=numpy.uint8)
    expected_values = height * step
    if data.size < expected_values:
        raise ValueError("ROS image data is shorter than height * step")

    rows = data[:expected_values].reshape((height, step))
    frame = rows[:, :min_step].reshape((height, width, channels))
    if encoding in {"rgb8", "rgb888"}:
        frame = frame[:, :, ::-1]
    return frame.copy()


def image_msg_stamp_sec(msg) -> float | None:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return None
    try:
        return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
    except (AttributeError, TypeError, ValueError):
        return None


def build_mask_for_ranges(cv2, numpy, hsv_frame, color_ranges: Sequence[ColorRange]):
    mask = numpy.zeros(hsv_frame.shape[:2], dtype=numpy.uint8)
    for color_range in color_ranges:
        lower = numpy.array(color_range.lower_hsv, dtype=numpy.uint8)
        upper = numpy.array(color_range.upper_hsv, dtype=numpy.uint8)
        range_mask = cv2.inRange(hsv_frame, lower, upper)
        mask = cv2.bitwise_or(mask, range_mask)
    return mask


def apply_morphology(cv2, mask, *, kernel_size: int, close_iterations: int, open_iterations: int):
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    return cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=open_iterations)


def classify_mask_roi(cv2, mask, roi: Rect, label: str) -> LiveMaskClassification:
    clipped = clamp_roi(roi, mask.shape)
    total = clipped.width * clipped.height
    if total <= 0:
        return LiveMaskClassification("unknown", 0.0, 0, 0)
    mask_roi = mask[clipped.y : clipped.y + clipped.height, clipped.x : clipped.x + clipped.width]
    matched = int(cv2.countNonZero(mask_roi))
    return LiveMaskClassification(label, matched / total, matched, total)


def current_track_range(cv2, label: str) -> ColorRange:
    lower = (
        cv2.getTrackbarPos("HMin", WINDOW_CONTROLS),
        cv2.getTrackbarPos("SMin", WINDOW_CONTROLS),
        cv2.getTrackbarPos("VMin", WINDOW_CONTROLS),
    )
    upper = (
        cv2.getTrackbarPos("HMax", WINDOW_CONTROLS),
        cv2.getTrackbarPos("SMax", WINDOW_CONTROLS),
        cv2.getTrackbarPos("VMax", WINDOW_CONTROLS),
    )
    return ColorRange(label, lower, upper)


def create_trackbars(cv2, color_range: ColorRange) -> None:
    def nothing(_value):
        return None

    cv2.namedWindow(WINDOW_CONTROLS)
    cv2.createTrackbar("HMin", WINDOW_CONTROLS, color_range.lower_hsv[0], 179, nothing)
    cv2.createTrackbar("SMin", WINDOW_CONTROLS, color_range.lower_hsv[1], 255, nothing)
    cv2.createTrackbar("VMin", WINDOW_CONTROLS, color_range.lower_hsv[2], 255, nothing)
    cv2.createTrackbar("HMax", WINDOW_CONTROLS, color_range.upper_hsv[0], 179, nothing)
    cv2.createTrackbar("SMax", WINDOW_CONTROLS, color_range.upper_hsv[1], 255, nothing)
    cv2.createTrackbar("VMax", WINDOW_CONTROLS, color_range.upper_hsv[2], 255, nothing)


def print_palette(color_ranges: Iterable[ColorRange]) -> None:
    for color_range in color_ranges:
        print(
            "ColorRange("
            f"{color_range.label!r}, "
            f"{color_range.lower_hsv!r}, "
            f"{color_range.upper_hsv!r}"
            ")"
        )


def annotate_frame(
    cv2,
    frame,
    roi: Rect | None,
    result: LiveMaskClassification,
    warning: str | None,
    *,
    receive_fps: float | None = None,
    display_fps: float | None = None,
    age_ms: float | None = None,
) -> None:
    if roi is not None:
        clipped = clamp_roi(roi, frame.shape)
        if clipped.width > 0 and clipped.height > 0:
            cv2.rectangle(
                frame,
                (clipped.x, clipped.y),
                (clipped.x + clipped.width, clipped.y + clipped.height),
                (0, 255, 255),
                2,
            )
    text = (
        f"{result.label} conf={result.confidence:.2f} "
        f"matched={result.matched_pixels}/{result.total_pixels}"
    )
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if warning:
        cv2.putText(frame, warning, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    fps_parts = []
    if receive_fps is not None:
        fps_parts.append(f"rx={receive_fps:.1f}")
    if display_fps is not None:
        fps_parts.append(f"disp={display_fps:.1f}")
    if age_ms is not None:
        fps_parts.append(f"age={age_ms:.0f}ms")
    if fps_parts:
        cv2.putText(
            frame,
            " ".join(fps_parts),
            (12, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


def save_snapshot(cv2, directory: Path, frame, mask, preview, color_label: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = directory / f"{stamp}_{color_label}"
    cv2.imwrite(str(prefix.with_name(prefix.name + "_frame.png")), frame)
    cv2.imwrite(str(prefix.with_name(prefix.name + "_mask.png")), mask)
    cv2.imwrite(str(prefix.with_name(prefix.name + "_preview.png")), preview)
    print(f"saved snapshot: {prefix}_*.png")


class RosImageTopicFrameSource:
    def __init__(self, numpy, topic: str, qos: str):
        self.numpy = numpy
        self.topic = topic
        self.description = f"ROS image topic {topic}"
        self.latest_frame = None
        self.latest_stamp_sec = None
        self.latest_sequence = 0
        self.latest_error = None
        self.received_count = 0
        self.last_received_sec = None
        self._last_fps_count = 0
        self._last_fps_sec = time.time()
        self.receive_fps = None
        self._lock = threading.Lock()
        self._running = False
        self._spin_thread = None

        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise SystemExit(
                "ROS image topic mode requires rclpy and sensor_msgs. "
                "Source the ROS 2 Humble and TurtleBot workspaces first."
            ) from exc

        self.rclpy = rclpy
        self.owns_rclpy = not rclpy.ok()
        if self.owns_rclpy:
            rclpy.init(args=None)

        class ColorMaskViewerNode(Node):
            pass

        self.node = ColorMaskViewerNode("aufgabe04_color_mask_viewer")
        qos_profile = QoSProfile(
            reliability=(
                ReliabilityPolicy.BEST_EFFORT
                if qos == "best-effort"
                else ReliabilityPolicy.RELIABLE
            ),
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.node.create_subscription(
            Image,
            topic,
            self._on_image,
            qos_profile,
        )

    def _on_image(self, msg) -> None:
        try:
            frame = image_msg_to_bgr_frame(msg, self.numpy)
            stamp_sec = image_msg_stamp_sec(msg)
            now = time.time()
            with self._lock:
                self.latest_frame = frame
                self.latest_stamp_sec = stamp_sec
                self.latest_sequence += 1
                self.latest_error = None
                self.received_count += 1
                self.last_received_sec = now
                elapsed = now - self._last_fps_sec
                if elapsed >= 1.0:
                    self.receive_fps = (self.received_count - self._last_fps_count) / elapsed
                    self._last_fps_count = self.received_count
                    self._last_fps_sec = now
        except ValueError as exc:
            with self._lock:
                self.latest_error = str(exc)

    def is_opened(self) -> bool:
        return True

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _spin_loop(self) -> None:
        while self._running and self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def read(self) -> FrameRead:
        with self._lock:
            error = self.latest_error
            frame = self.latest_frame
            stamp_sec = self.latest_stamp_sec
        if error:
            return FrameRead(False, message=error, waiting=True)
        if frame is None:
            return FrameRead(
                False,
                message=f"waiting for first frame on {self.topic}",
                waiting=True,
            )
        return FrameRead(True, frame=frame, stamp_sec=stamp_sec)

    def release(self) -> None:
        self._running = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
        self.node.destroy_node()
        if self.owns_rclpy and self.rclpy.ok():
            self.rclpy.shutdown()


def create_frame_source(numpy, args):
    return RosImageTopicFrameSource(numpy, args.ros_image_topic, args.qos)


def build_parser() -> argparse.ArgumentParser:
    labels = palette_labels()
    parser = argparse.ArgumentParser(
        description="Debug-only Aufgabe 04 stand color HSV mask viewer. Does not move the robot."
    )
    parser.add_argument(
        "--ros-image-topic",
        required=True,
        help="Read frames from a ROS 2 sensor_msgs/Image topic, e.g. /camera/image_raw.",
    )
    parser.add_argument(
        "--qos",
        choices=("best-effort", "reliable"),
        default="best-effort",
        help="ROS image subscription reliability. best-effort is lower latency for live video.",
    )
    parser.add_argument("--resize", type=float, default=1.0)
    parser.add_argument("--color", choices=labels, default="green")
    parser.add_argument("--roi", type=parse_roi, help="ROI as x,y,w,h in resized frame pixels.")
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--tune", action="store_true", help="Show HSV trackbars for the selected color.")
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--print-palette", action="store_true")
    parser.add_argument("--save-snapshot", type=Path)
    parser.add_argument("--no-morph", action="store_true")
    parser.add_argument("--morph-kernel", type=int, default=5)
    parser.add_argument("--close-iterations", type=int, default=2)
    parser.add_argument("--open-iterations", type=int, default=1)
    parser.add_argument("--detect-duplicates", action="store_true")
    parser.add_argument(
        "--max-display-fps",
        type=float,
        default=30.0,
        help="Limit display/render rate while keeping the latest received ROS frame. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--display-mode",
        choices=("frame", "frame-mask", "all"),
        default="frame",
        help="frame is lowest latency; frame-mask also shows mask; all also shows masked preview.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import cv2
        import numpy
    except ImportError as exc:
        raise SystemExit("OpenCV and numpy are required for the debug viewer.") from exc

    selected_ranges = list(ranges_for_label(args.color))
    if args.print_palette:
        print_palette(selected_ranges)

    frame_source = create_frame_source(numpy, args)
    if not frame_source.is_opened():
        raise SystemExit(f"failed to open frame source: {frame_source.description}")
    frame_source.start()

    if args.tune:
        create_trackbars(cv2, selected_ranges[0])

    print("Aufgabe 04 color mask viewer: debug-only, no robot motion, no /cmd_vel.")
    print("Keys: ESC/q quit, p print ColorRange, s save snapshot.")

    frame_count = 0
    last_frame_id = None
    duplicate_count = 0
    last_waiting_message_sec = 0.0
    last_display_sec = 0.0
    last_display_fps_sec = time.time()
    last_display_fps_count = 0
    display_fps = None
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
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    continue
                print(f"WARNING: {read.message}")
                break
            frame = read.frame
            now = time.time()
            last_display_sec = now
            age_ms = (now - read.stamp_sec) * 1000.0 if read.stamp_sec is not None else None

            if args.resize != 1.0:
                frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize)

            frame_id = id(frame)
            warning = None
            if args.detect_duplicates:
                if frame_id == last_frame_id:
                    duplicate_count += 1
                else:
                    duplicate_count = 0
                last_frame_id = frame_id
                if duplicate_count >= 15:
                    warning = "WARNING: repeated camera frame"

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

            roi = args.roi or Rect(0, 0, frame.shape[1], frame.shape[0])
            result = classify_mask_roi(cv2, mask, roi, args.color)
            annotated = frame.copy()
            annotate_frame(
                cv2,
                annotated,
                roi,
                result,
                warning,
                receive_fps=frame_source.receive_fps,
                display_fps=display_fps,
                age_ms=age_ms,
            )
            preview = None
            if args.display_mode == "all" or args.save_snapshot is not None:
                preview = cv2.bitwise_and(frame, frame, mask=mask)

            frame_count += 1
            last_display_fps_count += 1
            fps_elapsed = time.time() - last_display_fps_sec
            if fps_elapsed >= 1.0:
                display_fps = last_display_fps_count / fps_elapsed
                last_display_fps_count = 0
                last_display_fps_sec = time.time()
            if args.print_every > 0 and frame_count % args.print_every == 0:
                line = (
                    f"{result.label} confidence={result.confidence:.3f} "
                    f"matched={result.matched_pixels}/{result.total_pixels} "
                )
                if age_ms is not None:
                    line += f"age_ms={age_ms:.0f}"
                print(line)

            cv2.imshow(WINDOW_FRAME, annotated)
            if args.display_mode in {"frame-mask", "all"}:
                cv2.imshow(WINDOW_MASK, mask)
            if args.display_mode == "all":
                cv2.imshow(WINDOW_PREVIEW, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("p"):
                print_palette(active_ranges)
            if key == ord("s") and args.save_snapshot is not None:
                if preview is None:
                    preview = cv2.bitwise_and(frame, frame, mask=mask)
                save_snapshot(cv2, args.save_snapshot, annotated, mask, preview, args.color)
    finally:
        frame_source.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
