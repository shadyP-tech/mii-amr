from __future__ import annotations

import argparse
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Sequence

from scripts.aufgabe04.perception.debug.color_mask_viewer import (
    RosCompressedImageTopicFrameSource,
    create_trackbars,
    current_track_range,
    palette_labels,
    print_palette,
    ranges_for_label,
)
from scripts.aufgabe04.perception.mask_processing import apply_morphology, build_mask_for_ranges
from scripts.aufgabe04.perception.ros_image_adapter import compressed_msg_to_bgr_frame
from scripts.aufgabe04.perception.stand_side_classification import (
    StandSideClassification,
    classify_stand_side_from_frame,
)
from scripts.aufgabe04.perception.stand_axis_image import (
    StandAxisImageEstimate,
    estimate_stand_axis_from_edges,
    estimate_stand_axis_from_mask,
)
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_texts_bgr


WINDOW_FRAME = "aufgabe04/stand-axis"
WINDOW_MASK = "aufgabe04/stand-axis-mask"
WINDOW_EDGES = "aufgabe04/stand-axis-edges"


def build_parser() -> argparse.ArgumentParser:
    labels = palette_labels()
    parser = argparse.ArgumentParser(
        description=(
            "Debug-only live stand-axis viewer for a square stand face. "
            "Subscribes to compressed camera frames and does not move the robot."
        )
    )
    parser.add_argument(
        "--compressed-image-topic",
        required=True,
        help="ROS 2 sensor_msgs/CompressedImage topic, e.g. /camera/image_raw/compressed.",
    )
    parser.add_argument("--resize", type=float, default=1.0)
    parser.add_argument("--color", choices=labels, default="green")
    parser.add_argument(
        "--axis-source",
        choices=("edges", "color-mask"),
        default="edges",
        help="edges is color/QR agnostic and uses the outer square shape; color-mask keeps the old HSV contour mode.",
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
        "--median-window",
        type=int,
        default=7,
        help="Number of usable frames for median filtering ratio/yaw-proxy display. Use 1 to disable.",
    )
    parser.add_argument(
        "--stand-width-m",
        type=float,
        help="Physical square width. Required together with --stand-distance-m for approximate yaw degrees.",
    )
    parser.add_argument(
        "--stand-distance-m",
        type=float,
        help="Approximate camera-to-stand center distance. Required with --stand-width-m for yaw degrees.",
    )
    parser.add_argument(
        "--max-display-fps",
        type=float,
        default=20.0,
        help="Limit display/render rate while keeping only the newest ROS frame. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--max-frame-age-sec",
        type=float,
        default=0.25,
        help="Drop incoming ROS image messages older than this. Use 0 to disable.",
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
    return parser


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

    try:
        import cv2
        import numpy
    except ImportError as exc:
        raise SystemExit("OpenCV and numpy are required for the stand-axis viewer.") from exc

    selected_ranges = list(ranges_for_label(args.color))
    if args.print_palette:
        print_palette(selected_ranges)

    frame_source = RosCompressedImageTopicFrameSource(args.compressed_image_topic, args.max_frame_age_sec)
    frame_source.start()
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
            if read.sequence == last_processed_sequence:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue
            last_processed_sequence = read.sequence

            try:
                frame = compressed_msg_to_bgr_frame(read.frame, cv2, numpy)
            except ValueError as exc:
                print(f"WARNING: {exc}")
                continue
            last_display_sec = time.time()
            age_ms = (last_display_sec - read.stamp_sec) * 1000.0 if read.stamp_sec is not None else None

            if args.resize != 1.0:
                frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize)

            mask = None
            edges = None
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
                estimate, edges = estimate_stand_axis_from_edges(
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
                    min_area_px=args.min_area_px,
                    min_edge_height_px=args.min_edge_height_px,
                    min_aspect_ratio=args.min_aspect_ratio,
                    max_aspect_ratio=args.max_aspect_ratio,
                    stand_width_m=args.stand_width_m,
                    stand_distance_m=args.stand_distance_m,
                )
            else:
                estimate = estimate_stand_axis_from_mask(
                    cv2,
                    mask,
                    min_area_px=args.min_area_px,
                    min_edge_height_px=args.min_edge_height_px,
                    stand_width_m=args.stand_width_m,
                    stand_distance_m=args.stand_distance_m,
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
            side = classify_stand_side_from_frame(
                cv2,
                numpy,
                frame,
                mask,
                estimate,
                detect_qr_texts_bgr=detect_qr_texts_bgr,
                min_color_confidence=args.side_color_confidence,
                qr_crop_margin_px=args.qr_crop_margin_px,
            )
            if estimate.mode == "face_visible" and estimate.usable:
                ratio_window.append(float(estimate.height_ratio))
                proxy_window.append(float(estimate.yaw_proxy))
            filtered_ratio = statistics.median(ratio_window) if ratio_window else None
            filtered_proxy = statistics.median(proxy_window) if proxy_window else None

            annotated = frame.copy()
            annotate_frame(cv2, annotated, estimate, side, filtered_ratio, filtered_proxy, age_ms)

            frame_count += 1
            if args.print_every > 0 and frame_count % args.print_every == 0:
                if estimate.mode == "face_visible" and estimate.usable:
                    yaw_part = f" camera_yaw_deg={estimate.yaw_deg:.1f}" if estimate.yaw_deg is not None else ""
                    print(
                        f"camera_axis_rotation_proxy={estimate.yaw_proxy:+.3f}{yaw_part} "
                        f"ratio={estimate.height_ratio:.3f} closer={estimate.closer_side} "
                        f"left_px={estimate.left_height_px:.1f} right_px={estimate.right_height_px:.1f} "
                        f"source={estimate.source} stand_side={side.side} side_reason={side.reason} "
                        f"color_confidence={side.color_confidence:.3f} qr_texts={list(side.qr_texts)}"
                    )
                elif estimate.mode == "edge_on" and estimate.usable:
                    print(
                        f"camera_axis_edge_on_approx_90deg=true "
                        f"line_height_px={estimate.left_height_px:.1f} "
                        f"ratio=unavailable source={estimate.source} stand_side={side.side} "
                        f"side_reason={side.reason} color_confidence={side.color_confidence:.3f} "
                        f"qr_texts={list(side.qr_texts)}"
                    )
                else:
                    print(
                        f"camera_axis_rotation_unavailable reason={estimate.reason} "
                        f"area_px={estimate.contour_area_px:.0f} source={estimate.source} "
                        f"stand_side={side.side} side_reason={side.reason} "
                        f"color_confidence={side.color_confidence:.3f} qr_texts={list(side.qr_texts)}"
                    )

            cv2.imshow(WINDOW_FRAME, annotated)
            if args.display_mask and mask is not None:
                cv2.imshow(WINDOW_MASK, mask)
            if args.display_edges and edges is not None:
                cv2.imshow(WINDOW_EDGES, edges)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("p"):
                print_palette(active_ranges)
            if key == ord("s") and args.save_snapshot is not None:
                debug_image = edges if args.axis_source == "edges" and edges is not None else mask
                save_snapshot(cv2, args.save_snapshot, annotated, debug_image)
    finally:
        frame_source.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
