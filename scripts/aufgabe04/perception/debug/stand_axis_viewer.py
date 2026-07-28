from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
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
from scripts.aufgabe04.perception.debug.stand_axis_capture_store import (
    save_structural_capture,
    sensor_frame_status,
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
    nearest_scan_to_stamp,
    write_observation_jsonl,
)
from scripts.aufgabe04.perception.sim_wall_edge_mask import (
    WallEdgeMaskResult,
    build_confirmed_wall_exclusion_mask,
)
from scripts.aufgabe04.perception.stand_axis_image import (
    ImagePoint,
    StandAxisEdgeDebugArtifacts,
    StandAxisImageEstimate,
    estimate_stand_axis_from_edges,
    estimate_stand_axis_from_mask,
)
from scripts.aufgabe04.perception.stand_axis.radiator_rib_mask import (
    repeated_vertical_rib_exclusion_mask,
)
from scripts.aufgabe04.perception.stand_axis_tracking import (
    HeadCandidateTemporalGate,
    HeadTemporalSelection,
    _head_candidate_signature,
)
from scripts.aufgabe04.perception.stand_axis_consensus import (
    AxisConsensusAccumulator,
    axis_conditioning,
)
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_texts_bgr
from scripts.aufgabe04.simulation.sim_head_roi import (
    CameraTargetProjection,
    HeadRoi,
    project_target_to_camera,
    qr_corners_inside_roi,
    stand_head_roi,
)
from scripts.aufgabe04.simulation.sim_qr_detector import detect_simulated_station_qr_bgr


WINDOW_FRAME = "aufgabe04/stand-axis"
WINDOW_MASK = "aufgabe04/stand-axis-mask"
WINDOW_EDGES = "aufgabe04/stand-axis-edges"
# Keep the historical --display-face-mask flag/API name for compatibility, but
# label the window by what it now contains: sparse, untouched Canny evidence
# for the four independently fitted sides, not a connected face segmentation.
WINDOW_FACE_MASK = "aufgabe04/stand-axis-side-evidence"
WINDOW_RECTANGLE_MASK = "aufgabe04/stand-axis-rectangle"
NATIVE_PIXEL_DIAGNOSTIC_WINDOWS = frozenset(
    (WINDOW_FACE_MASK, WINDOW_RECTANGLE_MASK)
)
RECORDING_FILENAMES = {
    WINDOW_FRAME: "annotated.avi",
    WINDOW_MASK: "color_mask.avi",
    WINDOW_EDGES: "edges.avi",
    WINDOW_FACE_MASK: "side_evidence.avi",
    WINDOW_RECTANGLE_MASK: "rectangle.avi",
}


class DebugWindowRecorder:
    """Write one diagnostic video per currently displayed OpenCV window."""

    def __init__(self, cv2, output_directory: Path, fps: float) -> None:
        self._cv2 = cv2
        self._output_directory = output_directory
        self._fps = fps
        self._writers = {}
        self._sizes = {}
        self._session_directory: Path | None = None

    @property
    def active(self) -> bool:
        return bool(self._writers)

    def _bgr_frame(self, image):
        if image is None or len(image.shape) not in (2, 3):
            raise ValueError("recording requires a non-empty grayscale or BGR image")
        if len(image.shape) == 2:
            return self._cv2.cvtColor(image, self._cv2.COLOR_GRAY2BGR)
        if image.shape[2] != 3:
            raise ValueError("recording requires grayscale or BGR images")
        return image

    def start(self, images: dict[str, object]) -> None:
        if self.active:
            return
        if not images:
            raise ValueError("no displayed windows are available to record")

        session_name = (
            "recording_"
            + time.strftime("%Y%m%d_%H%M%S")
            + f"_{time.time_ns() % 1_000_000_000:09d}"
        )
        session_directory = self._output_directory / session_name
        session_directory.mkdir(parents=True, exist_ok=False)
        codec = self._cv2.VideoWriter_fourcc(*"MJPG")
        writers = {}
        sizes = {}
        try:
            for window_name, image in images.items():
                frame = self._bgr_frame(image)
                height, width = frame.shape[:2]
                if height <= 0 or width <= 0:
                    raise ValueError("recording requires non-empty window images")
                filename = RECORDING_FILENAMES.get(
                    window_name,
                    window_name.replace("/", "_") + ".avi",
                )
                writer = self._cv2.VideoWriter(
                    str(session_directory / filename),
                    codec,
                    self._fps,
                    (width, height),
                )
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"could not open video writer for {window_name}")
                writers[window_name] = writer
                sizes[window_name] = (width, height)
        except Exception:
            for writer in writers.values():
                writer.release()
            raise

        self._writers = writers
        self._sizes = sizes
        self._session_directory = session_directory
        self.write(images)

    def write(self, images: dict[str, object]) -> None:
        for window_name, writer in self._writers.items():
            image = images.get(window_name)
            if image is None:
                continue
            frame = self._bgr_frame(image)
            width, height = self._sizes[window_name]
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = self._cv2.resize(
                    frame,
                    (width, height),
                    interpolation=self._cv2.INTER_NEAREST,
                )
            writer.write(frame)

    def stop(self) -> None:
        if not self.active:
            return
        for writer in self._writers.values():
            writer.release()
        self._writers = {}
        self._sizes = {}
        self._session_directory = None


@dataclass(frozen=True)
class SimulationRobotPose:
    """Stamped read-only robot pose used by the simulation debug viewer."""

    stamp_sec: float | None
    frame_id: str
    child_frame_id: str
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float


@dataclass(frozen=True)
class SimulationMapTargetProjection:
    """One synchronized map-target projection into scan and camera domains."""

    scan_bearing_rad: float
    camera: CameraTargetProjection
    roi: HeadRoi | None


@dataclass(frozen=True)
class HeadDisplaySnapshot:
    """Pixel data captured from one accepted simulation camera frame.

    A temporal hold must reuse this complete bundle. Reusing only the previous
    corners or masks on top of the newest camera image makes a valid held
    detection appear spatially wrong whenever the image changes between the
    two frames.
    """

    frame: object
    mask: object | None
    edges: object | None
    face_mask: object | None
    rectangle_mask: object | None
    rectangle_overlay: object | None
    detected_head_roi: HeadRoi | None
    diagnostic_head_roi: HeadRoi | None


def _copy_display_image(image):
    return None if image is None else image.copy()


def _capture_head_display_snapshot(
    *,
    frame,
    mask,
    edges,
    face_mask,
    rectangle_mask,
    rectangle_overlay,
    detected_head_roi: HeadRoi | None,
    diagnostic_head_roi: HeadRoi | None,
) -> HeadDisplaySnapshot:
    """Copy all viewer pixels that must remain on the same source frame."""

    return HeadDisplaySnapshot(
        frame=_copy_display_image(frame),
        mask=_copy_display_image(mask),
        edges=_copy_display_image(edges),
        face_mask=_copy_display_image(face_mask),
        rectangle_mask=_copy_display_image(rectangle_mask),
        rectangle_overlay=_copy_display_image(rectangle_overlay),
        detected_head_roi=detected_head_roi,
        diagnostic_head_roi=diagnostic_head_roi,
    )


def _head_display_snapshot_for_selection(
    selection: HeadTemporalSelection,
    *,
    current: HeadDisplaySnapshot,
    last_accepted: HeadDisplaySnapshot | None,
) -> HeadDisplaySnapshot | None:
    """Select current pixels for fresh results and accepted pixels for holds."""

    if selection.current_accepted:
        return current
    if selection.held:
        return last_accepted
    return current


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
    parser.add_argument(
        "--candidate-center-width-fraction",
        type=float,
        default=1.0,
        help=(
            "Real compressed-camera debug only: restrict all head-candidate "
            "estimation to this centered fraction of the image width. "
            "Use 1.0 to keep the full width."
        ),
    )
    parser.add_argument(
        "--candidate-center-height-fraction",
        type=float,
        default=1.0,
        help=(
            "Real compressed-camera debug only: restrict all head-candidate "
            "estimation to this centered fraction of the image height. "
            "Use 1.0 to keep the full height."
        ),
    )
    parser.add_argument(
        "--candidate-center-y-fraction",
        type=float,
        default=0.5,
        help=(
            "Real compressed-camera debug only: vertical location of the "
            "candidate ROI center as a fraction of image height, where 0.0 "
            "is the top and 1.0 is the bottom. Default: 0.5."
        ),
    )
    parser.add_argument("--color", choices=labels, default="green")
    parser.add_argument(
        "--axis-source",
        choices=("edges", "color-mask"),
        default="edges",
        help="edges is color/QR agnostic and uses the filled outer silhouette; color-mask keeps the HSV contour mode.",
    )
    parser.add_argument(
        "--structural-diagnostic",
        action="store_true",
        help=(
            "Observe-only real-camera mode: require a raw-edge head over a "
            "paired stem and bounded base. Generic fallbacks cannot accept, "
            "and operational observation outputs are disabled."
        ),
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
        choices=("outer-border", "gray", "channel-union"),
        default="outer-border",
        help=(
            "outer-border smooths QR/internal texture; gray uses luminance; "
            "channel-union applies identical Canny extraction to B/G/R and "
            "unions the result without selecting a color. Standalone simulation "
            "uses channel-union automatically."
        ),
    )
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--edge-blur-kernel", type=int, default=5)
    parser.add_argument("--edge-dilate-iterations", type=int, default=1)
    parser.add_argument("--edge-close-kernel", type=int, default=5)
    parser.add_argument("--edge-close-iterations", type=int, default=1)
    parser.add_argument(
        "--suppress-repeated-vertical-ribs",
        dest="suppress_repeated_vertical_ribs",
        action="store_true",
        default=True,
        help=(
            "Suppress a real-camera family of at least four regular, near-vertical "
            "Hough rails (for example a heater) before silhouette fitting. "
            "Enabled by default; it never applies to simulation."
        ),
    )
    parser.add_argument(
        "--no-suppress-repeated-vertical-ribs",
        dest="suppress_repeated_vertical_ribs",
        action="store_false",
        help="Keep all repeated vertical edge families for a diagnostic A/B comparison.",
    )
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
        default=1.30,
        help=(
            "Known physical holder/front-face width divided by detected "
            "QR-code width. The Aufgabe 04 real stand default is 1.30; "
            "use 1.0 to disable QR-anchored head geometry."
        ),
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
        "--camera-height-m",
        type=float,
        default=0.093,
        help="Simulation camera optical-centre height above the floor (default: 0.093 m).",
    )
    parser.add_argument(
        "--camera-forward-offset-m",
        type=float,
        default=0.076,
        help="Simulation camera x offset from base_link (default: 0.076 m).",
    )
    parser.add_argument(
        "--camera-lateral-offset-m",
        type=float,
        default=0.0,
        help="Simulation camera y-left offset from base_link (default: 0 m).",
    )
    parser.add_argument(
        "--camera-yaw-offset-rad",
        type=float,
        default=0.0,
        help="Simulation camera yaw relative to base_link (default: 0 rad).",
    )
    parser.add_argument(
        "--stand-head-center-height-m",
        type=float,
        default=0.165035,
        help="Simulation stand-head centre height above the floor (default: 0.165035 m).",
    )
    parser.add_argument(
        "--head-roi-padding-scale",
        type=float,
        default=1.6,
        help="Simulation-only projected head ROI size relative to the expected head width.",
    )
    parser.add_argument(
        "--head-hold-sec",
        type=float,
        default=0.35,
        help=(
            "Retain the last validated edge/QR head during brief detector or "
            "outlier gaps (default: 0.35 s; 0 disables)."
        ),
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
        "--odom-topic",
        default="/odom",
        help="Simulation-only nav_msgs/Odometry topic used by map-target projection.",
    )
    parser.add_argument(
        "--sim-ground-truth-topic",
        default="/gazebo_ground_truth",
        help="Simulation-only stamped Gazebo world-pose topic used for wall projection.",
    )
    parser.add_argument(
        "--sim-sync-tolerance-sec",
        type=float,
        default=0.30,
        help=(
            "Maximum image/odometry/LaserScan timestamp difference in simulation. "
            "The default spans one 5 Hz LaserScan period plus scheduling jitter."
        ),
    )
    parser.add_argument(
        "--sim-wall-edge-suppression",
        dest="sim_wall_edge_suppression",
        action="store_true",
        default=True,
        help=(
            "In standalone simulation edge mode, remove arena-wall edges only when "
            "the synchronized LaserScan confirms the projected map wall (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-sim-wall-edge-suppression",
        dest="sim_wall_edge_suppression",
        action="store_false",
        help="Disable synchronized map/LiDAR wall-edge suppression.",
    )
    parser.add_argument(
        "--sim-wall-range-tolerance-m",
        type=float,
        default=0.08,
        help="Allowed LaserScan error when confirming a projected simulation wall.",
    )
    parser.add_argument(
        "--sim-wall-mask-line-width-px",
        type=int,
        default=7,
        help="Horizontal thickness of the projected simulation wall exclusion mask.",
    )
    parser.add_argument(
        "--sim-lidar-forward-offset-m",
        type=float,
        default=-0.032,
        help="Simulation base_link-to-base_scan forward offset (Burger default: -0.032 m).",
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
        choices=("fixed", "image-center", "map-target"),
        default="fixed",
        help=(
            "fixed keeps the --lidar-bearing-rad cone except in the standalone raw-simulation edge "
            "viewer, which detects on the full frame and then samples LiDAR at the detected head; "
            "image-center maps a detected rectangle into a LiDAR bearing; map-target is simulation-only "
            "and derives synchronized scan/camera bearings from /odom plus --stand-x/--stand-y."
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
    parser.add_argument(
        "--observation-status-json",
        type=Path,
        help="Write accepted/rejected conditioning status for pre-approach resampling.",
    )
    parser.add_argument("--map-frame", default="odom")
    parser.add_argument("--base-frame", default="base_footprint")
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
    parser.add_argument("--axis-consensus-frames", type=int, default=5)
    parser.add_argument("--axis-consensus-max-deviation-deg", type=float, default=8.0)
    parser.add_argument(
        "--max-observation-obliqueness-deg",
        type=float,
        default=30.0,
        help=(
            "Reject otherwise stable silhouette observations whose absolute "
            "camera-relative face angle exceeds this threshold."
        ),
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
        help="Also show untouched Canny evidence selected for the four fitted head sides.",
    )
    parser.add_argument(
        "--display-rectangle-mask",
        action="store_true",
        help="Also show the fitted quadrilateral outline as a separate diagnostic window.",
    )
    parser.add_argument(
        "--diagnostic-window-size-px",
        type=int,
        default=320,
        help=(
            "Maximum on-screen dimension for mask/edge diagnostic windows (default: 320). "
            "The side-evidence cutout and derived rectangle always use native-pixel "
            "AUTOSIZE windows. This never resamples processed pixels."
        ),
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=Path("results/aufgabe04/stand_axis_debug_recordings"),
        help=(
            "Directory where pressing r creates a timestamped set of diagnostic "
            "AVI recordings (default: results/aufgabe04/stand_axis_debug_recordings)."
        ),
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=15.0,
        help="Frames per second for keyboard-started diagnostic recordings (default: 15).",
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


def _quaternion_yaw(orientation) -> float:
    x = float(getattr(orientation, "x", 0.0))
    y = float(getattr(orientation, "y", 0.0))
    z = float(getattr(orientation, "z", 0.0))
    w = float(getattr(orientation, "w", 1.0))
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _nearest_simulation_pose(
    poses: Sequence[SimulationRobotPose],
    *,
    image_stamp_sec: float | None,
    tolerance_sec: float,
) -> SimulationRobotPose | None:
    """Select the odometry sample synchronized to one simulation image."""

    if not poses or not math.isfinite(tolerance_sec) or tolerance_sec <= 0.0:
        return None
    if image_stamp_sec is None or not math.isfinite(image_stamp_sec):
        return poses[-1]
    stamped = [
        pose
        for pose in poses
        if pose.stamp_sec is not None and math.isfinite(pose.stamp_sec)
    ]
    if not stamped:
        return None
    nearest = min(stamped, key=lambda pose: abs(pose.stamp_sec - image_stamp_sec))
    if abs(nearest.stamp_sec - image_stamp_sec) > tolerance_sec:
        return None
    return nearest


def _simulation_pose_frame_error(
    pose: SimulationRobotPose,
    *,
    map_frame: str,
    base_frame: str,
) -> str | None:
    if pose.frame_id and pose.frame_id.lstrip("/") != map_frame.lstrip("/"):
        return "odom_frame_mismatch"
    if (
        pose.child_frame_id
        and pose.child_frame_id.lstrip("/") != base_frame.lstrip("/")
    ):
        return "odom_child_frame_mismatch"
    return None


def _project_simulation_map_target(
    *,
    robot_pose: SimulationRobotPose,
    stand_x_m: float,
    stand_y_m: float,
    stand_head_center_height_m: float,
    camera_forward_offset_m: float,
    camera_lateral_offset_m: float,
    camera_height_m: float,
    camera_yaw_offset_rad: float,
    frame_width: int,
    frame_height: int,
    camera_fx_px: float,
    camera_fy_px: float | None,
    camera_cx_px: float,
    camera_cy_px: float,
    stand_face_size_m: float,
    head_roi_padding_scale: float,
) -> SimulationMapTargetProjection:
    """Project a known stand without confusing scan and image bearings."""

    scan_delta = (
        math.atan2(stand_y_m - robot_pose.y_m, stand_x_m - robot_pose.x_m)
        - robot_pose.yaw_rad
    )
    scan_bearing = math.atan2(math.sin(scan_delta), math.cos(scan_delta))
    camera_projection = project_target_to_camera(
        robot_x_m=robot_pose.x_m,
        robot_y_m=robot_pose.y_m,
        robot_z_m=robot_pose.z_m,
        robot_yaw_rad=robot_pose.yaw_rad,
        target_x_m=stand_x_m,
        target_y_m=stand_y_m,
        target_height_m=stand_head_center_height_m,
        camera_forward_offset_m=camera_forward_offset_m,
        camera_lateral_offset_m=camera_lateral_offset_m,
        camera_height_m=camera_height_m,
        camera_yaw_offset_rad=camera_yaw_offset_rad,
    )
    roi = stand_head_roi(
        frame_width=frame_width,
        frame_height=frame_height,
        bearing_rad=camera_projection.bearing_rad,
        distance_m=None,
        camera_fx_px=camera_fx_px,
        camera_fy_px=camera_fy_px,
        camera_cx_px=camera_cx_px,
        camera_cy_px=camera_cy_px,
        stand_face_size_m=stand_face_size_m,
        camera_depth_m=camera_projection.depth_m,
        target_height_delta_m=camera_projection.height_delta_m,
        padding_scale=head_roi_padding_scale,
    )
    return SimulationMapTargetProjection(
        scan_bearing_rad=scan_bearing,
        camera=camera_projection,
        roi=roi,
    )


class RosSimulationRawImageTopicFrameSource:
    """Latest-frame source deliberately restricted to the explicit simulation CLI."""

    def __init__(
        self,
        topic: str,
        max_frame_age_sec: float,
        *,
        odom_topic: str | None = None,
        ground_truth_topic: str | None = None,
    ):
        self.topic = topic
        self.odom_topic = odom_topic
        self.ground_truth_topic = ground_truth_topic
        self.max_frame_age_sec = max_frame_age_sec
        self.latest_message = None
        self.latest_stamp_sec = None
        self.latest_sequence = 0
        self._robot_poses: deque[SimulationRobotPose] = deque(maxlen=40)
        self._ground_truth_poses: deque[SimulationRobotPose] = deque(maxlen=40)
        self._lock = threading.Lock()
        self._running = False
        self._spin_thread = None
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, qos_profile_sensor_data
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise SystemExit(
                "Simulation raw image mode requires ROS 2 sensor_msgs and nav_msgs."
            ) from exc
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
        self.odom_subscription = None
        if odom_topic is not None:
            self.odom_subscription = self.node.create_subscription(
                Odometry,
                odom_topic,
                self._on_odom,
                qos,
            )
        self.ground_truth_subscription = None
        if ground_truth_topic is not None:
            self.ground_truth_subscription = self.node.create_subscription(
                Odometry,
                ground_truth_topic,
                self._on_ground_truth,
                qos,
            )

    def _on_image(self, msg) -> None:
        stamp_sec = compressed_msg_stamp_sec(msg)
        with self._lock:
            self.latest_message = msg
            self.latest_stamp_sec = stamp_sec
            self.latest_sequence += 1

    def _on_odom(self, msg) -> None:
        header = getattr(msg, "header", None)
        pose = msg.pose.pose
        position = pose.position
        sample = SimulationRobotPose(
            stamp_sec=_stamp_to_sec(getattr(header, "stamp", None)),
            frame_id=str(getattr(header, "frame_id", "") or ""),
            child_frame_id=str(getattr(msg, "child_frame_id", "") or ""),
            x_m=float(position.x),
            y_m=float(position.y),
            z_m=float(position.z),
            yaw_rad=_quaternion_yaw(pose.orientation),
        )
        with self._lock:
            self._robot_poses.append(sample)

    def _on_ground_truth(self, msg) -> None:
        header = getattr(msg, "header", None)
        pose = msg.pose.pose
        position = pose.position
        sample = SimulationRobotPose(
            stamp_sec=_stamp_to_sec(getattr(header, "stamp", None)),
            frame_id=str(getattr(header, "frame_id", "") or "world"),
            child_frame_id=str(getattr(msg, "child_frame_id", "") or "base_footprint"),
            x_m=float(position.x),
            y_m=float(position.y),
            z_m=float(position.z),
            yaw_rad=_quaternion_yaw(pose.orientation),
        )
        with self._lock:
            self._ground_truth_poses.append(sample)

    def start(self) -> None:
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _spin_loop(self) -> None:
        while self._running and self.rclpy.ok():
            try:
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
            except Exception:
                if not self._running or not self.rclpy.ok():
                    break
                raise

    def read(self) -> FrameRead:
        with self._lock:
            message = self.latest_message
            stamp_sec = self.latest_stamp_sec
            sequence = self.latest_sequence
        if message is None:
            return FrameRead(False, message=f"waiting for first frame on {self.topic}", waiting=True)
        return FrameRead(True, frame=message, stamp_sec=stamp_sec, sequence=sequence)

    def nearest_robot_pose(
        self,
        *,
        image_stamp_sec: float | None,
        tolerance_sec: float,
    ) -> SimulationRobotPose | None:
        with self._lock:
            poses = tuple(self._robot_poses)
        return _nearest_simulation_pose(
            poses,
            image_stamp_sec=image_stamp_sec,
            tolerance_sec=tolerance_sec,
        )

    def nearest_ground_truth_pose(
        self,
        *,
        image_stamp_sec: float | None,
        tolerance_sec: float,
    ) -> SimulationRobotPose | None:
        with self._lock:
            poses = tuple(self._ground_truth_poses)
        return _nearest_simulation_pose(
            poses,
            image_stamp_sec=image_stamp_sec,
            tolerance_sec=tolerance_sec,
        )

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
        self._scans: deque[PlainLaserScan] = deque(maxlen=80)
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
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except Exception:
                if not self._running or not self.rclpy.ok():
                    break
                raise

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
            self._scans.append(scan)

    def latest_scan(self) -> PlainLaserScan | None:
        with self._lock:
            return self._latest_scan

    def nearest_scan(
        self,
        *,
        image_stamp_sec: float | None,
        tolerance_sec: float,
    ) -> PlainLaserScan | None:
        with self._lock:
            scans = tuple(self._scans)
        return nearest_scan_to_stamp(
            scans,
            image_stamp_sec=image_stamp_sec,
            tolerance_sec=tolerance_sec,
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
    fallback_source = (
        "map_target_projection"
        if bearing_source == "map-target"
        else "fixed_bearing"
    )

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
    wall_edge_mask_result: WallEdgeMaskResult | None = None,
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
    wall_mask_text = "n/a"
    if wall_edge_mask_result is not None:
        wall_mask_text = (
            f"{wall_edge_mask_result.reason}:wall="
            f"{wall_edge_mask_result.confirmed_wall_samples},foreground="
            f"{wall_edge_mask_result.protected_foreground_samples}"
        )
    print(
        f"stand_axis source={estimate.source} mode={estimate.mode} usable={estimate.usable} "
        f"axis_reason={estimate.reason} "
        f"angle={angle_text} proxy={proxy_text} ratio={ratio_text} "
        f"side={side_label} raw_side={side.side} reason={side.reason} "
        f"lidar_distance={lidar_text} used_distance={distance_text} "
        f"lidar_roi={lidar_roi_text} "
        f"wall_mask={wall_mask_text} "
        f"left_px={estimate.left_height_px:.1f} right_px={estimate.right_height_px:.1f} "
        f"qr_texts={list(qr_texts)}"
    )


def annotate_simulation_target_roi(
    cv2,
    frame,
    *,
    target_roi: HeadRoi | None,
    camera_bearing_rad: float | None,
    scan_bearing_rad: float | None,
    camera_depth_m: float | None,
    failure_reason: str | None,
    label: str = "target ROI",
) -> None:
    """Expose a projected target ROI or post-detection head ROI."""

    color = (255, 255, 0)
    if target_roi is None:
        if failure_reason:
            cv2.putText(
                frame,
                f"{label} unavailable: {failure_reason}",
                (12, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2,
            )
        return

    top_left = (target_roi.x0, target_roi.y0)
    bottom_right = (target_roi.x1 - 1, target_roi.y1 - 1)
    cv2.rectangle(frame, top_left, bottom_right, color, 2)
    camera_text = format_optional_float(
        None if camera_bearing_rad is None else math.degrees(camera_bearing_rad),
        precision=1,
        suffix="deg",
    )
    scan_text = format_optional_float(
        None if scan_bearing_rad is None else math.degrees(scan_bearing_rad),
        precision=1,
        suffix="deg",
    )
    depth_text = format_optional_float(camera_depth_m, precision=3, suffix="m")
    cv2.putText(
        frame,
        f"{label} camera={camera_text} scan={scan_text} depth={depth_text}",
        (target_roi.x0, max(14, target_roi.y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
    )


def annotate_candidate_search_roi(cv2, frame, target_roi: HeadRoi | None) -> None:
    """Show the real-camera pixel domain allowed to produce head candidates."""

    if target_roi is None:
        return
    color = (255, 0, 255)
    cv2.rectangle(
        frame,
        (target_roi.x0, target_roi.y0),
        (target_roi.x1 - 1, target_roi.y1 - 1),
        color,
        2,
    )
    cv2.putText(
        frame,
        "candidate search ROI",
        (target_roi.x0, max(14, target_roi.y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
    )


def annotate_frame(
    cv2,
    frame,
    estimate: StandAxisImageEstimate,
    side: StandSideClassification,
    filtered_ratio,
    filtered_proxy,
    age_ms,
    detector_duration_sec,
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
    timing_parts = []
    if age_ms is not None:
        timing_parts.append(f"age={age_ms:.0f}ms")
    if detector_duration_sec is not None:
        timing_parts.append(f"detect={detector_duration_sec * 1000.0:.0f}ms")
    if timing_parts:
        cv2.putText(
            frame,
            " ".join(timing_parts),
            (12, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


def annotate_recording_indicator(cv2, frame, recording_active: bool) -> None:
    """Draw a compact red dot only while diagnostic recording is active."""

    if not recording_active:
        return
    height, width = frame.shape[:2]
    radius = max(6, min(12, min(height, width) // 30))
    margin = radius + 8
    cv2.circle(
        frame,
        (max(radius, width - margin), min(height - radius, margin)),
        radius,
        (0, 0, 255),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


def save_snapshot(cv2, directory: Path, frame, mask) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    frame_path = directory / f"{stamp}_stand_axis_frame.png"
    mask_path = directory / f"{stamp}_stand_axis_mask.png"
    cv2.imwrite(str(frame_path), frame)
    cv2.imwrite(str(mask_path), mask)
    print(f"saved snapshot: {frame_path} {mask_path}")


def _save_structural_viewer_capture(
    cv2,
    *,
    args,
    read: FrameRead,
    decoded_source_frame,
    axis_frame,
    edge_artifacts: StandAxisEdgeDebugArtifacts,
    annotated,
    detector_estimate: StandAxisImageEstimate,
    temporal_selection: HeadTemporalSelection | None,
    target_roi: HeadRoi | None,
    decode_duration_sec: float,
    detector_duration_sec: float,
    processed_monotonic_sec: float,
) -> None:
    compressed = getattr(read.frame, "data", None)
    compressed_format = getattr(read.frame, "format", "")
    structure = edge_artifacts.structure_evidence
    if temporal_selection is None:
        measurement_status = "fresh" if detector_estimate.usable else "unavailable"
        held_age_sec = None
    else:
        measurement_status = temporal_selection.measurement_status
        held_age_sec = temporal_selection.held_age_sec
    roi_payload = (
        None
        if target_roi is None
        else {
            "x0": target_roi.x0,
            "y0": target_roi.y0,
            "x1": target_roi.x1,
            "y1": target_roi.y1,
        }
    )
    save_structural_capture(
        cv2,
        args.save_snapshot,
        original_compressed=compressed,
        compressed_format=compressed_format,
        decoded_frame=decoded_source_frame,
        candidate_roi_frame=axis_frame,
        raw_edges=edge_artifacts.raw_edges,
        localization_edges=edge_artifacts.edges,
        side_evidence=edge_artifacts.face_mask,
        rectangle_mask=edge_artifacts.rectangle_mask,
        annotated_frame=annotated,
        metadata={
            "topic": args.compressed_image_topic or "",
            "frame_id": read.frame_id,
            "source_frame_stamp_sec": read.stamp_sec,
            "received_wall_sec": read.received_wall_sec,
            "received_monotonic_sec": read.received_monotonic_sec,
            "processed_monotonic_sec": processed_monotonic_sec,
            "timings_sec": {
                "decode": decode_duration_sec,
                "detector": detector_duration_sec,
                "receipt_to_processed": (
                    None
                    if read.received_monotonic_sec is None
                    else processed_monotonic_sec - read.received_monotonic_sec
                ),
            },
            "sensor_status": sensor_frame_status(
                source_stamp_sec=read.stamp_sec,
                received_wall_sec=read.received_wall_sec,
                max_frame_age_sec=args.max_frame_age_sec,
            ),
            "measurement_status": measurement_status,
            "held_age_sec": held_age_sec,
            "detector_reason": detector_estimate.reason,
            "detector_source": detector_estimate.source,
            "detector_usable": detector_estimate.usable,
            "candidate_roi": roi_payload,
            "decoded_image_size": {
                "width": int(decoded_source_frame.shape[1]),
                "height": int(decoded_source_frame.shape[0]),
            },
            "structure": None if structure is None else structure.status_dict(),
        },
    )


def _initialize_display_windows(cv2, args) -> None:
    """Create windows; diagnostic content sizing follows the live ROI later."""
    if args.headless:
        return

    cv2.namedWindow(WINDOW_FRAME, cv2.WINDOW_AUTOSIZE)
    diagnostic_windows = []
    if args.display_mask:
        diagnostic_windows.append(WINDOW_MASK)
    if args.display_edges:
        diagnostic_windows.append(WINDOW_EDGES)
    if args.display_face_mask:
        diagnostic_windows.append(WINDOW_FACE_MASK)
    if args.display_rectangle_mask:
        diagnostic_windows.append(WINDOW_RECTANGLE_MASK)

    for window_name in diagnostic_windows:
        window_mode = (
            cv2.WINDOW_AUTOSIZE
            if window_name in NATIVE_PIXEL_DIAGNOSTIC_WINDOWS
            else cv2.WINDOW_NORMAL
        )
        cv2.namedWindow(window_name, window_mode)


def _resize_diagnostic_windows(cv2, args, image_shape) -> tuple[int, int]:
    """Size only the resizable mask/edge viewports; cutouts stay native."""
    height_px, width_px = (int(image_shape[0]), int(image_shape[1]))
    if height_px <= 0 or width_px <= 0:
        raise ValueError("diagnostic image dimensions must be positive")
    scale = min(
        1.0,
        float(args.diagnostic_window_size_px) / float(max(width_px, height_px)),
    )
    display_width = max(1, int(round(width_px * scale)))
    display_height = max(1, int(round(height_px * scale)))
    windows = []
    if args.display_mask:
        windows.append(WINDOW_MASK)
    if args.display_edges:
        windows.append(WINDOW_EDGES)
    for window_name in windows:
        cv2.resizeWindow(window_name, display_width, display_height)
    return display_width, display_height


def _resize_diagnostic_window(cv2, window_name: str, image_shape, maximum_px: int) -> tuple[int, int]:
    """Size one diagnostic window to its own native aspect ratio."""

    height_px, width_px = (int(image_shape[0]), int(image_shape[1]))
    if height_px <= 0 or width_px <= 0:
        raise ValueError("diagnostic image dimensions must be positive")
    scale = min(1.0, float(maximum_px) / float(max(width_px, height_px)))
    display_width = max(1, int(round(width_px * scale)))
    display_height = max(1, int(round(height_px * scale)))
    cv2.resizeWindow(window_name, display_width, display_height)
    return display_width, display_height


def _unavailable_target_estimate(reason: str) -> StandAxisImageEstimate:
    return StandAxisImageEstimate(
        usable=False,
        reason=reason,
        mode="unavailable",
        corners=None,
        axis_line=None,
        left_height_px=0.0,
        right_height_px=0.0,
        height_ratio=None,
        yaw_proxy=None,
        yaw_deg=None,
        closer_side=None,
        contour_area_px=0.0,
        source="target_roi",
    )


def _diagnostic_roi_image(image, target_roi):
    """Return the target pixel domain shared by mask/edge diagnostics."""
    if image is None or target_roi is None:
        return image
    return image[
        target_roi.y0 : target_roi.y1,
        target_roi.x0 : target_roi.x1,
    ]


def _centered_candidate_roi(
    *,
    frame_width: int,
    frame_height: int,
    width_fraction: float,
    height_fraction: float,
    center_y_fraction: float = 0.5,
) -> HeadRoi | None:
    """Return a horizontally centred real-camera search crop."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("candidate ROI frame dimensions must be positive")
    if (
        not math.isfinite(width_fraction)
        or not 0.0 < width_fraction <= 1.0
        or not math.isfinite(height_fraction)
        or not 0.0 < height_fraction <= 1.0
    ):
        raise ValueError("candidate ROI fractions must be finite and in (0, 1]")
    if not math.isfinite(center_y_fraction) or not 0.0 <= center_y_fraction <= 1.0:
        raise ValueError("candidate ROI vertical center must be finite and in [0, 1]")
    if (
        width_fraction == 1.0
        and height_fraction == 1.0
        and center_y_fraction == 0.5
    ):
        return None

    roi_width = min(
        frame_width,
        max(16, int(round(frame_width * width_fraction))),
    )
    roi_height = min(
        frame_height,
        max(16, int(round(frame_height * height_fraction))),
    )
    x0 = (frame_width - roi_width) // 2
    desired_center_y = int(round((frame_height - 1) * center_y_fraction))
    y0 = max(
        0,
        min(
            frame_height - roi_height,
            desired_center_y - roi_height // 2,
        ),
    )
    return HeadRoi(
        x0=x0,
        y0=y0,
        x1=x0 + roi_width,
        y1=y0 + roi_height,
        source="candidate_center",
        expected_head_px=float(min(roi_width, roi_height)),
    )


def _simulation_full_frame_edge_mode(args) -> bool:
    """Use edge-first discovery for the standalone raw simulation viewer."""

    return bool(
        args.sim_raw_image_topic
        and args.axis_source == "edges"
        and args.lidar_bearing_source == "fixed"
    )


def _simulation_wall_suppression_mode(args) -> bool:
    return bool(
        _simulation_full_frame_edge_mode(args)
        and args.sim_wall_edge_suppression
        and args.use_lidar_distance
        and args.scan_topic
    )


def _detected_head_roi(
    estimate: StandAxisImageEstimate,
    *,
    frame_width: int,
    frame_height: int,
    padding_scale: float,
) -> HeadRoi | None:
    """Derive a display/cutout ROI only after full-frame silhouette detection."""

    if estimate.corners is None or frame_width <= 0 or frame_height <= 0:
        return None
    if not math.isfinite(padding_scale) or padding_scale <= 0.0:
        return None
    xs = [point.u_px for point in estimate.corners]
    ys = [point.v_px for point in estimate.corners]
    if not all(math.isfinite(value) for value in (*xs, *ys)):
        return None
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    expected_head_px = max(width, height)
    if expected_head_px <= 0.0:
        return None
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    half_extent = max(12.0, expected_head_px * padding_scale / 2.0)
    x0 = max(0, int(math.floor(center_x - half_extent)))
    y0 = max(0, int(math.floor(center_y - half_extent)))
    x1 = min(frame_width, int(math.ceil(center_x + half_extent)))
    y1 = min(frame_height, int(math.ceil(center_y + half_extent)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return HeadRoi(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        source="edge_detected",
        expected_head_px=expected_head_px,
    )


def _standalone_head_geometry_reason(
    estimate: StandAxisImageEstimate,
    *,
    frame_width: int,
    frame_height: int,
    max_extent_fraction: float = 0.38,
) -> str | None:
    """Reject wall-sized candidates before they can seed the temporal gate."""

    signature = _head_candidate_signature(estimate)
    if signature is None:
        return estimate.reason
    if frame_width <= 0 or frame_height <= 0:
        return "invalid_camera_frame_size"
    if not math.isfinite(max_extent_fraction) or not 0.0 < max_extent_fraction < 1.0:
        return "invalid_head_extent_fraction"
    if signature.extent_px > max_extent_fraction * min(frame_width, frame_height):
        return "head_candidate_too_large"
    assert estimate.corners is not None
    border_margin_px = 1.0
    if any(
        point.u_px <= border_margin_px
        or point.v_px <= border_margin_px
        or point.u_px >= frame_width - 1 - border_margin_px
        or point.v_px >= frame_height - 1 - border_margin_px
        for point in estimate.corners
    ):
        return "head_candidate_touches_frame_border"
    return None


def _validate_runtime_args(args) -> None:
    if args.diagnostic_window_size_px <= 0:
        raise ValueError("--diagnostic-window-size-px must be positive")
    if not math.isfinite(args.record_fps) or args.record_fps <= 0.0:
        raise ValueError("--record-fps must be finite and positive")
    if not 0.0 < args.max_observation_obliqueness_deg < 90.0:
        raise ValueError("--max-observation-obliqueness-deg must be in (0, 90)")
    if args.sim_sync_tolerance_sec <= 0.0:
        raise ValueError("--sim-sync-tolerance-sec must be positive")
    if args.head_roi_padding_scale <= 0.0:
        raise ValueError("--head-roi-padding-scale must be positive")
    if not math.isfinite(args.head_hold_sec) or args.head_hold_sec < 0.0:
        raise ValueError("--head-hold-sec must be finite and non-negative")
    if args.sim_wall_range_tolerance_m <= 0.0:
        raise ValueError("--sim-wall-range-tolerance-m must be positive")
    if args.sim_wall_mask_line_width_px <= 0:
        raise ValueError("--sim-wall-mask-line-width-px must be positive")
    if (
        args.front_face_to_qr_width_ratio is not None
        and (
            not math.isfinite(args.front_face_to_qr_width_ratio)
            or args.front_face_to_qr_width_ratio < 1.0
        )
    ):
        raise ValueError(
            "--front-face-to-qr-width-ratio must be finite and at least 1.0"
        )
    candidate_roi_fractions = (
        args.candidate_center_width_fraction,
        args.candidate_center_height_fraction,
    )
    if not all(
        math.isfinite(value) and 0.0 < value <= 1.0
        for value in candidate_roi_fractions
    ):
        raise ValueError(
            "--candidate-center-width-fraction and "
            "--candidate-center-height-fraction must be finite and in (0, 1]"
        )
    if (
        not math.isfinite(args.candidate_center_y_fraction)
        or not 0.0 <= args.candidate_center_y_fraction <= 1.0
    ):
        raise ValueError(
            "--candidate-center-y-fraction must be finite and in [0, 1]"
        )
    candidate_roi_is_default = (
        candidate_roi_fractions == (1.0, 1.0)
        and args.candidate_center_y_fraction == 0.5
    )
    if args.sim_raw_image_topic and not candidate_roi_is_default:
        raise ValueError(
            "centered candidate ROI options are available only with "
            "--compressed-image-topic"
        )

    simulation_map_target = bool(
        args.sim_raw_image_topic and args.lidar_bearing_source == "map-target"
    )
    if args.lidar_bearing_source == "map-target" and not args.sim_raw_image_topic:
        raise ValueError("--lidar-bearing-source map-target is simulation-only")

    if args.observation_output_json is not None:
        required_geometry = (args.stand_x, args.stand_y)
        if not simulation_map_target:
            required_geometry += (args.robot_x, args.robot_y)
        if any(value is None for value in required_geometry):
            required = "--stand-x and --stand-y"
            if not simulation_map_target:
                required = "--robot-x, --robot-y, --stand-x, and --stand-y"
            raise ValueError(f"--observation-output-json requires {required}")
        if args.observation_write_hz <= 0.0:
            raise ValueError("--observation-write-hz must be positive")

    if args.structural_diagnostic:
        if args.sim_raw_image_topic:
            raise ValueError("--structural-diagnostic is real-camera-only")
        if args.axis_source != "edges":
            raise ValueError("--structural-diagnostic requires --axis-source edges")
        if (
            args.observation_output_json is not None
            or args.observation_status_json is not None
        ):
            raise ValueError(
                "--structural-diagnostic is observe-only and cannot be combined "
                "with operational observation output"
            )

    if not args.sim_raw_image_topic:
        return

    # The standalone simulation edge viewer discovers the silhouette in the
    # complete raw frame.  It must not require map coordinates, a projected
    # range, or camera geometry merely to expose the edge/head diagnostics.
    if _simulation_full_frame_edge_mode(args):
        return

    stand_width_m = (
        args.stand_face_size_m
        if args.stand_face_size_m is not None
        else args.stand_width_m
    )
    if stand_width_m is None or args.camera_fx_px is None:
        raise ValueError(
            "simulation stand detection requires --stand-face-size-m and --camera-fx-px"
        )
    if simulation_map_target:
        if args.stand_x is None or args.stand_y is None:
            raise ValueError(
                "simulation map-target projection requires --stand-x and --stand-y"
            )
        if not args.odom_topic:
            raise ValueError("simulation map-target projection requires --odom-topic")
        if not args.use_lidar_distance or not args.scan_topic:
            raise ValueError(
                "simulation map-target projection requires --scan-topic and --use-lidar-distance"
            )
        if not args.map_frame or not args.base_frame:
            raise ValueError(
                "simulation map-target projection requires --map-frame and --base-frame"
            )
        values = (
            args.stand_x,
            args.stand_y,
            args.camera_forward_offset_m,
            args.camera_lateral_offset_m,
            args.camera_height_m,
            args.camera_yaw_offset_rad,
            args.stand_head_center_height_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("simulation map-target geometry must be finite")
        return

    if args.stand_distance_m is None and not args.use_lidar_distance:
        raise ValueError(
            "simulation stand detection requires --stand-distance-m or --use-lidar-distance"
        )
    if args.lidar_bearing_source != "fixed":
        raise ValueError(
            "simulation target-first detection requires fixed or map-target LiDAR bearing"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_runtime_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.structural_diagnostic:
        print(
            "Structural diagnostic mode: observe_only=true authoritative=false "
            "metric_pose=false pnp=false"
        )
    simulation_full_frame_edges = _simulation_full_frame_edge_mode(args)
    simulation_wall_suppression = _simulation_wall_suppression_mode(args)
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
            args.sim_raw_image_topic,
            args.max_frame_age_sec,
            odom_topic=(
                args.odom_topic
                if args.lidar_bearing_source == "map-target"
                else None
            ),
            ground_truth_topic=(
                args.sim_ground_truth_topic
                if simulation_wall_suppression
                else None
            ),
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
    _initialize_display_windows(cv2, args)
    recorder = DebugWindowRecorder(cv2, args.record_dir, args.record_fps)

    print("Aufgabe 04 stand-axis viewer: debug-only, read-only ROS subscriptions.")
    if simulation_full_frame_edges:
        print(
            "Simulation standalone mode: full-frame edges -> detected silhouette "
            "-> dynamic head ROI -> head cutout/rectangle."
        )
        if simulation_wall_suppression:
            print(
                "Simulation wall filter: stamped Gazebo world pose + synchronized LaserScan-confirmed "
                "arena-wall projection; closer foreground corridors remain protected."
            )
    print("Face-visible mode: + proxy means left image edge is closer/taller; - means right edge is closer/taller.")
    print("Edge-on mode: reports approximate side-on / 90deg and does not compute a ratio.")
    print(
        "Keys: ESC/q quit, p print ColorRange, s save snapshot, "
        "r start/stop all displayed-window recordings."
    )

    ratio_window = deque(maxlen=max(1, args.median_window))
    proxy_window = deque(maxlen=max(1, args.median_window))
    frame_count = 0
    last_processed_sequence = 0
    last_display_sec = 0.0
    last_waiting_message_sec = 0.0
    last_observation_write_sec = 0.0
    last_diagnostic_shapes = {}
    record_start_pending = False
    axis_consensus = AxisConsensusAccumulator(
        required_samples=args.axis_consensus_frames,
        max_deviation_rad=math.radians(args.axis_consensus_max_deviation_deg),
    )
    head_candidate_temporal_gate = HeadCandidateTemporalGate(
        hold_sec=args.head_hold_sec,
        # Rail-owned real-head fits may seed the first rectangle immediately;
        # every later geometry change still needs the normal three-frame
        # reacquisition path. This avoids a long blank bootstrap on compressed
        # real-camera edges while preventing flickering refits.
        initial_acquire_frames=1,
        max_width_ratio=(1.25 if args.sim_raw_image_topic else 1.15),
        max_height_ratio=(1.25 if args.sim_raw_image_topic else 1.15),
        max_corner_jump_scale=(0.18 if args.sim_raw_image_topic else 0.12),
        max_side_direction_jump_deg=(8.0 if args.sim_raw_image_topic else 6.0),
    )
    last_accepted_head_display_snapshot = None

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
                    if key == ord("r"):
                        if recorder.active:
                            recorder.stop()
                        elif record_start_pending:
                            record_start_pending = False
                        else:
                            record_start_pending = True
                    continue
                print(f"WARNING: {read.message}")
                break
            if read.sequence == last_processed_sequence:
                key = 0 if args.headless else cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    if recorder.active:
                        recorder.stop()
                    elif record_start_pending:
                        record_start_pending = False
                    else:
                        record_start_pending = True
                continue
            last_processed_sequence = read.sequence

            decode_started_monotonic = time.monotonic()
            try:
                frame = (
                    raw_msg_to_bgr_frame(read.frame, cv2, numpy)
                    if args.sim_raw_image_topic
                    else compressed_msg_to_bgr_frame(read.frame, cv2, numpy)
                )
            except ValueError as exc:
                print(f"WARNING: {exc}")
                continue
            decode_duration_sec = time.monotonic() - decode_started_monotonic
            decoded_source_frame = frame.copy()
            last_display_sec = time.time()
            age_ms = (
                None
                if args.sim_raw_image_topic or read.stamp_sec is None
                else (last_display_sec - read.stamp_sec) * 1000.0
            )

            if args.resize != 1.0:
                frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize)
            query_camera_cx_px = camera_cx_px if camera_cx_px is not None else float(frame.shape[1]) / 2.0
            query_camera_cy_px = camera_cy_px if camera_cy_px is not None else float(frame.shape[0]) / 2.0
            sim_qr_detection = None
            axis_frame = frame
            axis_camera_cx_px = camera_cx_px
            axis_camera_cy_px = camera_cy_px
            target_roi = None
            candidate_search_roi = None
            detected_head_roi = None
            diagnostic_head_roi = None
            simulation_pose = None
            map_target_projection = None
            target_roi_failure_reason = None
            head_measurement_fresh = True
            head_estimate_held = False
            temporal_selection = None
            active_lidar_bearing_source = args.lidar_bearing_source
            roi_camera_bearing_rad = None
            roi_scan_bearing_rad = None
            roi_camera_depth_m = None

            mask = None
            edges = None
            face_mask = None
            rectangle_mask = None
            rectangle_overlay = None
            edge_artifacts = StandAxisEdgeDebugArtifacts(edges=None)
            detector_estimate = None
            wall_edge_mask = None
            wall_edge_mask_result = None
            lidar_query = None
            lidar_camera_bearing_rad = None
            lidar_rect_center_x_px = None
            lidar_fallback_source = "none"
            if args.lidar_bearing_source == "map-target":
                simulation_pose = frame_source.nearest_robot_pose(
                    image_stamp_sec=read.stamp_sec,
                    tolerance_sec=args.sim_sync_tolerance_sec,
                )
                if simulation_pose is None:
                    target_roi_failure_reason = "synchronized_odom_unavailable"
                else:
                    target_roi_failure_reason = _simulation_pose_frame_error(
                        simulation_pose,
                        map_frame=args.map_frame,
                        base_frame=args.base_frame,
                    )
                    if target_roi_failure_reason is not None:
                        simulation_pose = None
                if simulation_pose is not None and target_roi_failure_reason is None:
                    map_target_projection = _project_simulation_map_target(
                        robot_pose=simulation_pose,
                        stand_x_m=args.stand_x,
                        stand_y_m=args.stand_y,
                        stand_head_center_height_m=args.stand_head_center_height_m,
                        camera_forward_offset_m=args.camera_forward_offset_m,
                        camera_lateral_offset_m=args.camera_lateral_offset_m,
                        camera_height_m=args.camera_height_m,
                        camera_yaw_offset_rad=args.camera_yaw_offset_rad,
                        lidar_forward_offset_m=args.sim_lidar_forward_offset_m,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        camera_fx_px=camera_fx_px,
                        camera_fy_px=camera_fy_px,
                        camera_cx_px=query_camera_cx_px,
                        camera_cy_px=query_camera_cy_px,
                        stand_face_size_m=stand_width_m,
                        head_roi_padding_scale=args.head_roi_padding_scale,
                    )
                    target_roi = map_target_projection.roi
                    roi_camera_bearing_rad = map_target_projection.camera.bearing_rad
                    roi_scan_bearing_rad = map_target_projection.scan_bearing_rad
                    roi_camera_depth_m = map_target_projection.camera.depth_m
                    if target_roi is None:
                        target_roi_failure_reason = "target_outside_camera_fov"
                    lidar_query, _, _, lidar_fallback_source = _query_lidar_distance(
                        lidar_source=lidar_source,
                        bearing_source="map-target",
                        fixed_bearing_rad=map_target_projection.scan_bearing_rad,
                        cone_half_angle_rad=math.radians(args.lidar_cone_deg),
                        max_scan_age_sec=args.max_scan_age_sec,
                        min_sample_count=args.lidar_min_samples,
                        estimate=None,
                        camera_fx_px=camera_fx_px,
                        camera_cx_px=query_camera_cx_px,
                        camera_to_lidar_yaw_offset_rad=0.0,
                        now_sec=last_display_sec,
                    )
                    lidar_camera_bearing_rad = map_target_projection.camera.bearing_rad
            elif (
                args.lidar_bearing_source == "fixed"
                and not simulation_full_frame_edges
            ):
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
            stand_distance_m = (
                map_target_projection.camera.depth_m
                if map_target_projection is not None
                else (
                    lidar_distance_m
                    if lidar_distance_m is not None
                    else args.stand_distance_m
                )
            )
            detector_started_monotonic = time.monotonic()
            if (
                args.sim_raw_image_topic
                and args.lidar_bearing_source != "map-target"
                and not simulation_full_frame_edges
                and stand_distance_m is not None
            ):
                camera_target_bearing_rad = (
                    args.lidar_bearing_rad - args.camera_to_lidar_yaw_offset_rad
                )
                roi_camera_bearing_rad = camera_target_bearing_rad
                roi_scan_bearing_rad = args.lidar_bearing_rad
                roi_camera_depth_m = max(
                    1e-6,
                    stand_distance_m * math.cos(camera_target_bearing_rad),
                )
                target_roi = stand_head_roi(
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    bearing_rad=camera_target_bearing_rad,
                    distance_m=stand_distance_m,
                    camera_fx_px=camera_fx_px,
                    camera_fy_px=camera_fy_px,
                    camera_cx_px=query_camera_cx_px,
                    camera_cy_px=query_camera_cy_px,
                    stand_face_size_m=stand_width_m,
                    camera_depth_m=roi_camera_depth_m,
                    target_height_delta_m=(
                        args.stand_head_center_height_m - args.camera_height_m
                    ),
                    padding_scale=args.head_roi_padding_scale,
                )
                if target_roi is None:
                    target_roi_failure_reason = "target_outside_camera_fov"
            if args.sim_raw_image_topic and target_roi is not None:
                if not args.no_qr_decode:
                    sim_qr_detection = detect_simulated_station_qr_bgr(
                        frame,
                        cv2,
                        roi=(
                            target_roi.x0,
                            target_roi.y0,
                            target_roi.x1,
                            target_roi.y1,
                        ),
                    )
                    if sim_qr_detection is not None and not qr_corners_inside_roi(
                        sim_qr_detection.corners_px, target_roi
                    ):
                        sim_qr_detection = None
                axis_frame = frame[
                    target_roi.y0 : target_roi.y1,
                    target_roi.x0 : target_roi.x1,
                ]
                axis_camera_cx_px = query_camera_cx_px - target_roi.x0
                axis_camera_cy_px = query_camera_cy_px - target_roi.y0
            elif not args.sim_raw_image_topic:
                candidate_search_roi = _centered_candidate_roi(
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    width_fraction=args.candidate_center_width_fraction,
                    height_fraction=args.candidate_center_height_fraction,
                    center_y_fraction=args.candidate_center_y_fraction,
                )
                if candidate_search_roi is not None:
                    target_roi = candidate_search_roi
                    axis_frame = frame[
                        target_roi.y0 : target_roi.y1,
                        target_roi.x0 : target_roi.x1,
                    ]
                    axis_camera_cx_px = query_camera_cx_px - target_roi.x0
                    axis_camera_cy_px = query_camera_cy_px - target_roi.y0
            if simulation_wall_suppression:
                # The scan identifies which image columns contain a foreground
                # obstacle. Protect camera edges in those columns without
                # assuming any stand hue; this mask is used only to keep wall
                # suppression from erasing foreground evidence.
                simulation_foreground_support_mask = cv2.Canny(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    20,
                    60,
                )
                wall_pose = frame_source.nearest_ground_truth_pose(
                    image_stamp_sec=read.stamp_sec,
                    tolerance_sec=args.sim_sync_tolerance_sec,
                )
                wall_scan = (
                    None
                    if lidar_source is None
                    else lidar_source.nearest_scan(
                        image_stamp_sec=read.stamp_sec,
                        tolerance_sec=args.sim_sync_tolerance_sec,
                    )
                )
                wall_pose_error = (
                    "synchronized_world_pose_unavailable"
                    if wall_pose is None
                    else None
                )
                if wall_pose_error is not None:
                    wall_edge_mask_result = WallEdgeMaskResult(
                        None,
                        wall_pose_error,
                        0,
                        0,
                        None,
                    )
                elif camera_fx_px is None:
                    wall_edge_mask_result = WallEdgeMaskResult(
                        None,
                        "camera_intrinsics_unavailable",
                        0,
                        0,
                        None,
                    )
                else:
                    wall_edge_mask_result = build_confirmed_wall_exclusion_mask(
                        cv2,
                        numpy,
                        scan=wall_scan,
                        image_stamp_sec=read.stamp_sec,
                        sync_tolerance_sec=args.sim_sync_tolerance_sec,
                        robot_x_m=wall_pose.x_m,
                        robot_y_m=wall_pose.y_m,
                        robot_z_m=wall_pose.z_m,
                        robot_yaw_rad=wall_pose.yaw_rad,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        camera_fx_px=camera_fx_px,
                        camera_fy_px=(
                            camera_fx_px if camera_fy_px is None else camera_fy_px
                        ),
                        camera_cx_px=query_camera_cx_px,
                        camera_cy_px=query_camera_cy_px,
                        camera_forward_offset_m=args.camera_forward_offset_m,
                        camera_lateral_offset_m=args.camera_lateral_offset_m,
                        camera_height_m=args.camera_height_m,
                        camera_yaw_offset_rad=args.camera_yaw_offset_rad,
                        lidar_forward_offset_m=args.sim_lidar_forward_offset_m,
                        wall_range_tolerance_m=args.sim_wall_range_tolerance_m,
                        foreground_support_mask=simulation_foreground_support_mask,
                        mask_line_width_px=args.sim_wall_mask_line_width_px,
                    )
                    wall_edge_mask = wall_edge_mask_result.mask
            active_ranges = selected_ranges
            if args.axis_source == "color-mask" or args.display_mask or args.tune:
                # This optional HSV view is diagnostic (or the explicitly
                # selected legacy color-mask mode). Edge/silhouette mode does
                # not consume it for localization, fitting, or yaw.
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

            if (
                simulation_wall_suppression
                and (
                    wall_edge_mask_result is None
                    or wall_edge_mask_result.mask is None
                )
            ):
                # Never pass an unfiltered simulation frame to the silhouette
                # detector.  A temporarily missing synchronized scan otherwise
                # reconnects the head to the arena wall and alternates between
                # a valid head and head_candidate_too_large.  Dropping this
                # observation preserves the last temporal/consensus state and
                # lets the next synchronized frame recover immediately.
                wall_reason = (
                    "wall_filter_unavailable"
                    if wall_edge_mask_result is None
                    else wall_edge_mask_result.reason
                )
                estimate = _unavailable_target_estimate(wall_reason)
                edges = numpy.zeros(frame.shape[:2], dtype=numpy.uint8)
                edge_artifacts = StandAxisEdgeDebugArtifacts(edges=edges)
            elif (
                args.sim_raw_image_topic
                and target_roi is None
                and not simulation_full_frame_edges
            ):
                estimate = _unavailable_target_estimate(
                    target_roi_failure_reason
                    or (
                        "target_roi_unavailable"
                        if stand_distance_m is None
                        else "target_outside_camera_fov"
                    )
                )
                edges = numpy.zeros(frame.shape[:2], dtype=numpy.uint8)
                edge_artifacts = StandAxisEdgeDebugArtifacts(edges=edges)
            elif args.axis_source == "edges":
                edge_exclusion_mask = wall_edge_mask
                if (
                    not args.sim_raw_image_topic
                    and args.suppress_repeated_vertical_ribs
                ):
                    radiator_rib_mask = repeated_vertical_rib_exclusion_mask(
                        cv2,
                        cv2.Canny(
                            cv2.cvtColor(axis_frame, cv2.COLOR_BGR2GRAY),
                            args.canny_low,
                            args.canny_high,
                        ),
                    ).mask
                    edge_exclusion_mask = (
                        radiator_rib_mask
                        if edge_exclusion_mask is None
                        else cv2.bitwise_or(edge_exclusion_mask, radiator_rib_mask)
                    )
                edge_estimator_options = dict(
                    edge_preprocess=(
                        "channel_union"
                        if args.sim_raw_image_topic
                        else args.edge_preprocess.replace("-", "_")
                    ),
                    blur_kernel=args.edge_blur_kernel,
                    canny_low=(20 if args.sim_raw_image_topic else args.canny_low),
                    canny_high=(60 if args.sim_raw_image_topic else args.canny_high),
                    dilate_iterations=args.edge_dilate_iterations,
                    close_kernel=args.edge_close_kernel,
                    close_iterations=args.edge_close_iterations,
                    hough_threshold=args.hough_threshold,
                    hough_min_line_length_px=args.hough_min_line_length_px,
                    hough_max_line_gap_px=args.hough_max_line_gap_px,
                    min_boundary_line_length_px=args.min_boundary_line_length_px,
                    face_width_fraction=args.face_width_fraction,
                    min_face_area_fraction=(
                        0.0 if args.sim_raw_image_topic else args.min_face_area_fraction
                    ),
                    min_area_px=args.min_area_px,
                    min_edge_height_px=args.min_edge_height_px,
                    min_aspect_ratio=args.min_aspect_ratio,
                    max_aspect_ratio=args.max_aspect_ratio,
                    front_face_to_qr_width_ratio=args.front_face_to_qr_width_ratio,
                    stand_width_m=(
                        None if args.structural_diagnostic else stand_width_m
                    ),
                    stand_distance_m=(
                        None if args.structural_diagnostic else stand_distance_m
                    ),
                    camera_fx_px=(
                        None if args.structural_diagnostic else camera_fx_px
                    ),
                    camera_fy_px=(
                        None if args.structural_diagnostic else camera_fy_px
                    ),
                    camera_cx_px=axis_camera_cx_px,
                    camera_cy_px=axis_camera_cy_px,
                    stand_depth_m=args.stand_head_depth_m,
                    stand_head_bottom_height_m=args.stand_head_bottom_height_m,
                    # Real-camera structural diagnostics use the raw-side
                    # head geometry path; stem/base structure is metadata only.
                    silhouette_only=bool(
                        args.sim_raw_image_topic or args.structural_diagnostic
                    ),
                    structural_diagnostic=args.structural_diagnostic,
                    # A real camera may see a rotated head frame.  Its side
                    # rails must be parallel, but need not be image-vertical.
                    parallel_side_direction=(
                        (0.0, 1.0) if args.sim_raw_image_topic else None
                    ),
                )
                estimate, edge_artifacts = estimate_stand_axis_from_edges(
                    cv2,
                    axis_frame,
                    edge_exclusion_mask=edge_exclusion_mask,
                    **edge_estimator_options,
                )
                edges = edge_artifacts.edges
                face_mask = edge_artifacts.face_mask
                rectangle_mask = edge_artifacts.rectangle_mask
                rectangle_overlay = edge_artifacts.rectangle_overlay
            else:
                axis_mask = (
                    mask
                    if target_roi is None
                    else mask[target_roi.y0 : target_roi.y1, target_roi.x0 : target_roi.x1]
                )
                estimate = estimate_stand_axis_from_mask(
                    cv2,
                    axis_mask,
                    min_area_px=args.min_area_px,
                    min_edge_height_px=args.min_edge_height_px,
                    stand_width_m=stand_width_m,
                    stand_distance_m=stand_distance_m,
                    camera_fx_px=camera_fx_px,
                    camera_fy_px=camera_fy_px,
                    camera_cx_px=axis_camera_cx_px,
                    camera_cy_px=axis_camera_cy_px,
                    stand_depth_m=args.stand_head_depth_m,
                    stand_head_bottom_height_m=args.stand_head_bottom_height_m,
                )
            detector_duration_sec = time.monotonic() - detector_started_monotonic
            if estimate.corners is not None and target_roi is not None:
                estimate = replace(
                    estimate,
                    corners=tuple(
                        ImagePoint(
                            point.u_px + target_roi.x0,
                            point.v_px + target_roi.y0,
                        )
                        for point in estimate.corners
                    ),
                )
            detector_estimate = estimate
            if simulation_full_frame_edges:
                rejected_fit_diagnostic_roi = (
                    _detected_head_roi(
                        estimate,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        padding_scale=args.head_roi_padding_scale,
                    )
                    if estimate.reason == "head_rectangle_fit_unreliable"
                    and face_mask is not None
                    else None
                )
                geometry_reason = _standalone_head_geometry_reason(
                    estimate,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                )
                temporal_selection = head_candidate_temporal_gate.stabilize(
                    estimate,
                    now_sec=time.monotonic(),
                    rejection_reason=geometry_reason,
                )
                head_measurement_fresh = temporal_selection.current_accepted
                head_estimate_held = temporal_selection.held
                if (
                    not temporal_selection.current_accepted
                    and not temporal_selection.held
                    and rejected_fit_diagnostic_roi is not None
                ):
                    # Keep the current rejected cutout visible.  Replacing it
                    # with a synthetic rectangle would hide why this frame
                    # failed.
                    rectangle_mask = None
                    rectangle_overlay = None
                elif (
                    not temporal_selection.current_accepted
                    and not temporal_selection.held
                ):
                    face_mask = None
                    rectangle_mask = None
                    rectangle_overlay = None
                estimate = (
                    temporal_selection.estimate
                    if temporal_selection.estimate is not None
                    else _unavailable_target_estimate(temporal_selection.reason)
                )
                detected_head_roi = _detected_head_roi(
                    estimate,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    padding_scale=args.head_roi_padding_scale,
                )
                diagnostic_head_roi = (
                    detected_head_roi
                    if temporal_selection.held
                    else (
                        rejected_fit_diagnostic_roi
                        if rejected_fit_diagnostic_roi is not None
                        else detected_head_roi
                    )
                )
                target_roi_failure_reason = (
                    None
                    if detected_head_roi is not None
                    else estimate.reason
                )
                if detected_head_roi is not None:
                    (
                        lidar_query,
                        lidar_camera_bearing_rad,
                        lidar_rect_center_x_px,
                        lidar_fallback_source,
                    ) = _query_lidar_distance(
                        lidar_source=lidar_source,
                        bearing_source="image-center",
                        fixed_bearing_rad=args.lidar_bearing_rad,
                        cone_half_angle_rad=math.radians(args.lidar_cone_deg),
                        max_scan_age_sec=args.max_scan_age_sec,
                        min_sample_count=args.lidar_min_samples,
                        estimate=estimate,
                        camera_fx_px=camera_fx_px,
                        camera_cx_px=query_camera_cx_px,
                        camera_to_lidar_yaw_offset_rad=(
                            args.camera_to_lidar_yaw_offset_rad
                        ),
                        now_sec=last_display_sec,
                    )
                    active_lidar_bearing_source = "image-center"
                    lidar_distance_m = (
                        None if lidar_query is None else lidar_query.distance_m
                    )
                    if lidar_distance_m is not None:
                        stand_distance_m = lidar_distance_m
                    roi_camera_bearing_rad = lidar_camera_bearing_rad
                    roi_scan_bearing_rad = (
                        None if lidar_query is None else lidar_query.bearing_rad
                    )
                    roi_camera_depth_m = lidar_distance_m
                    if (
                        roi_camera_bearing_rad is None
                        and camera_fx_px is not None
                    ):
                        detected_center_x = (
                            detected_head_roi.x0 + detected_head_roi.x1
                        ) / 2.0
                        try:
                            roi_camera_bearing_rad = camera_bearing_rad(
                                detected_center_x,
                                camera_fx_px=camera_fx_px,
                                camera_cx_px=query_camera_cx_px,
                            )
                        except ValueError:
                            roi_camera_bearing_rad = None
                    if not args.no_qr_decode:
                        sim_qr_detection = detect_simulated_station_qr_bgr(
                            frame,
                            cv2,
                            roi=(
                                detected_head_roi.x0,
                                detected_head_roi.y0,
                                detected_head_roi.x1,
                                detected_head_roi.y1,
                            ),
                        )
                        if (
                            sim_qr_detection is not None
                            and not qr_corners_inside_roi(
                                sim_qr_detection.corners_px,
                                detected_head_roi,
                            )
                        ):
                            sim_qr_detection = None
            elif not args.sim_raw_image_topic and args.axis_source == "edges":
                temporal_selection = head_candidate_temporal_gate.stabilize(
                    estimate,
                    now_sec=time.monotonic(),
                )
                head_measurement_fresh = temporal_selection.current_accepted
                head_estimate_held = temporal_selection.held
                if (
                    not temporal_selection.current_accepted
                    and not temporal_selection.held
                ):
                    face_mask = None
                    rectangle_mask = None
                    rectangle_overlay = None
                estimate = (
                    temporal_selection.estimate
                    if temporal_selection.estimate is not None
                    else _unavailable_target_estimate(temporal_selection.reason)
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
            display_frame = frame
            display_mask = mask
            display_edges = edges
            display_face_mask = face_mask
            display_rectangle_mask = rectangle_mask
            display_rectangle_overlay = rectangle_overlay
            display_detected_head_roi = detected_head_roi
            display_diagnostic_head_roi = diagnostic_head_roi
            if temporal_selection is not None:
                current_head_display_snapshot = HeadDisplaySnapshot(
                    frame=frame,
                    mask=mask,
                    edges=edges,
                    face_mask=face_mask,
                    rectangle_mask=rectangle_mask,
                    rectangle_overlay=rectangle_overlay,
                    detected_head_roi=detected_head_roi,
                    diagnostic_head_roi=diagnostic_head_roi,
                )
                if temporal_selection.current_accepted:
                    # Copy only accepted frames, which may outlive this loop
                    # iteration. Current/rejected frames are displayed
                    # directly and need no additional full-frame copies.
                    last_accepted_head_display_snapshot = (
                        _capture_head_display_snapshot(
                            frame=frame,
                            mask=mask,
                            edges=edges,
                            face_mask=face_mask,
                            rectangle_mask=rectangle_mask,
                            rectangle_overlay=rectangle_overlay,
                            detected_head_roi=detected_head_roi,
                            diagnostic_head_roi=diagnostic_head_roi,
                        )
                    )
                selected_head_display_snapshot = (
                    _head_display_snapshot_for_selection(
                        temporal_selection,
                        current=current_head_display_snapshot,
                        last_accepted=last_accepted_head_display_snapshot,
                    )
                )
                if selected_head_display_snapshot is not None:
                    display_frame = selected_head_display_snapshot.frame
                    display_mask = selected_head_display_snapshot.mask
                    display_edges = selected_head_display_snapshot.edges
                    display_face_mask = selected_head_display_snapshot.face_mask
                    display_rectangle_mask = (
                        selected_head_display_snapshot.rectangle_mask
                    )
                    display_rectangle_overlay = (
                        selected_head_display_snapshot.rectangle_overlay
                    )
                    display_detected_head_roi = (
                        selected_head_display_snapshot.detected_head_roi
                    )
                    display_diagnostic_head_roi = (
                        selected_head_display_snapshot.diagnostic_head_roi
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
                    image_topic=image_topic,
                    image_stamp_sec=read.stamp_sec,
                    scan_topic=args.scan_topic or "",
                    query=lidar_query,
                    rect_center_x_px=lidar_rect_center_x_px,
                    camera_fx_px=camera_fx_px,
                    camera_cx_px=query_camera_cx_px,
                    camera_bearing_rad_value=lidar_camera_bearing_rad,
                    bearing_source=active_lidar_bearing_source,
                    fallback_source=lidar_fallback_source,
                    estimate=estimate,
                    observed_at_sec=last_display_sec,
                )
            color_confidence = color_confidence_for_estimate(cv2, numpy, mask, estimate)
            if args.sim_raw_image_topic and not args.no_qr_decode:
                qr_texts = () if sim_qr_detection is None else (sim_qr_detection.station_id,)
            elif qr_decoder is not None:
                qr_decoder.submit_latest(frame, estimate, read.sequence, time.time())
                qr_texts = qr_decoder.latest_texts(time.time())
            else:
                qr_texts = ()
            observation_robot_x = (
                simulation_pose.x_m
                if simulation_pose is not None
                else args.robot_x
            )
            observation_robot_y = (
                simulation_pose.y_m
                if simulation_pose is not None
                else args.robot_y
            )
            observation_distance_m = None
            if (
                args.observation_output_json is not None
                and observation_robot_x is not None
                and observation_robot_y is not None
            ):
                observation_distance_m = math.hypot(
                    args.stand_x - observation_robot_x,
                    args.stand_y - observation_robot_y,
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
            # QR evidence identifies the station/visible side only. Orientation
            # always comes from the same head-silhouette estimate used on the
            # real robot.
            orientation_yaw_rad = (
                None if estimate.yaw_deg is None else math.radians(estimate.yaw_deg)
            )
            consensus = None
            if (
                orientation_yaw_rad is not None
                and estimate.usable
                and estimate.mode == "face_visible"
                and head_measurement_fresh
            ):
                consensus = axis_consensus.add(
                    yaw_rad=orientation_yaw_rad,
                    source=estimate.source,
                    side=side.side,
                    qr_texts=tuple(qr_texts),
                )
            elif not head_estimate_held:
                axis_consensus.reset()
            conditioning = (
                None
                if consensus is None
                else axis_conditioning(
                    consensus.yaw_rad,
                    max_obliqueness_rad=math.radians(
                        args.max_observation_obliqueness_deg
                    ),
                )
            )
            if conditioning is not None and args.observation_status_json is not None:
                status_payload = {
                    "schema_version": 1,
                    "accepted": conditioning.accepted,
                    "reason": conditioning.reason,
                    "obliqueness_deg": math.degrees(conditioning.obliqueness_rad),
                    "max_obliqueness_deg": math.degrees(
                        conditioning.max_obliqueness_rad
                    ),
                    "consensus_samples": consensus.sample_count,
                    "consensus_max_deviation_deg": math.degrees(
                        consensus.max_deviation_rad
                    ),
                    "source": consensus.source,
                    "qr_texts": list(qr_texts),
                }
                args.observation_status_json.parent.mkdir(parents=True, exist_ok=True)
                status_tmp = args.observation_status_json.with_suffix(
                    args.observation_status_json.suffix + ".tmp"
                )
                status_tmp.write_text(
                    json.dumps(status_payload, indent=2, sort_keys=True) + "\n"
                )
                status_tmp.replace(args.observation_status_json)
            if (
                args.observation_output_json is not None
                and estimate.usable
                and estimate.mode == "face_visible"
                and consensus is not None
                and conditioning is not None
                and conditioning.accepted
                and side.side in ("qr_code_side", "basic_color_side")
                and side.confidence >= 0.60
                and observation_robot_x is not None
                and observation_robot_y is not None
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
                        robot_x_m=observation_robot_x,
                        robot_y_m=observation_robot_y,
                        stand_x_m=args.stand_x,
                        stand_y_m=args.stand_y,
                        stand_axis_rad=stand_axis_from_camera_yaw(
                            robot_x_m=observation_robot_x,
                            robot_y_m=observation_robot_y,
                            stand_x_m=args.stand_x,
                            stand_y_m=args.stand_y,
                            camera_yaw_rad=consensus.yaw_rad,
                        ),
                        axis_confidence=0.85 if pnp_calibrated else 0.65,
                        side=side.side,
                        side_confidence=side.confidence,
                        qr_texts=tuple(qr_texts),
                    )
                    write_camera_observation(args.observation_output_json, observation)
                    last_observation_write_sec = time.time()
            if (
                estimate.mode == "face_visible"
                and estimate.usable
                and head_measurement_fresh
            ):
                ratio_window.append(float(estimate.height_ratio))
                proxy_window.append(float(estimate.yaw_proxy))
            filtered_ratio = statistics.median(ratio_window) if ratio_window else None
            filtered_proxy = statistics.median(proxy_window) if proxy_window else None

            annotated = display_frame.copy()
            annotate_frame(
                cv2,
                annotated,
                estimate,
                side,
                filtered_ratio,
                filtered_proxy,
                age_ms,
                detector_duration_sec,
            )
            annotate_candidate_search_roi(cv2, annotated, candidate_search_roi)
            if args.sim_raw_image_topic:
                annotate_simulation_target_roi(
                    cv2,
                    annotated,
                    target_roi=(
                        display_detected_head_roi
                        if simulation_full_frame_edges
                        else target_roi
                    ),
                    camera_bearing_rad=roi_camera_bearing_rad,
                    scan_bearing_rad=roi_scan_bearing_rad,
                    camera_depth_m=roi_camera_depth_m,
                    failure_reason=target_roi_failure_reason,
                    label=(
                        (
                            "held head ROI"
                            if head_estimate_held
                            else "detected head ROI"
                        )
                        if simulation_full_frame_edges
                        else "target ROI"
                    ),
                )
            annotate_recording_indicator(cv2, annotated, recorder.active)

            frame_count += 1
            if args.headless and args.save_snapshot is not None and frame_count == 1:
                if args.structural_diagnostic:
                    _save_structural_viewer_capture(
                        cv2,
                        args=args,
                        read=read,
                        decoded_source_frame=decoded_source_frame,
                        axis_frame=axis_frame,
                        edge_artifacts=edge_artifacts,
                        annotated=annotated,
                        detector_estimate=detector_estimate,
                        temporal_selection=temporal_selection,
                        target_roi=target_roi,
                        decode_duration_sec=decode_duration_sec,
                        detector_duration_sec=detector_duration_sec,
                        processed_monotonic_sec=time.monotonic(),
                    )
                else:
                    debug_image = (
                        display_edges
                        if args.axis_source == "edges" and display_edges is not None
                        else display_mask
                    )
                    save_snapshot(cv2, args.save_snapshot, annotated, debug_image)
            if args.print_every > 0 and frame_count % args.print_every == 0:
                print_status_line(
                    estimate,
                    side,
                    lidar_distance_m=lidar_distance_m,
                    stand_distance_m=stand_distance_m,
                    lidar_query=lidar_query,
                    lidar_bearing_source=active_lidar_bearing_source,
                    qr_texts=qr_texts,
                    wall_edge_mask_result=wall_edge_mask_result,
                )
                if args.structural_diagnostic:
                    measurement_status = (
                        "fresh"
                        if temporal_selection is None and detector_estimate.usable
                        else (
                            "unavailable"
                            if temporal_selection is None
                            else temporal_selection.measurement_status
                        )
                    )
                    print(
                        "stand_structure "
                        "observe_only=true authoritative=false "
                        f"sensor_status={sensor_frame_status(source_stamp_sec=read.stamp_sec, received_wall_sec=read.received_wall_sec, max_frame_age_sec=args.max_frame_age_sec)} "
                        f"measurement_status={measurement_status} "
                        f"detector_reason={detector_estimate.reason}"
                    )

            if not args.headless:
                cv2.imshow(WINDOW_FRAME, annotated)
                diagnostic_mask = _diagnostic_roi_image(display_mask, target_roi)
                diagnostic_edges = display_edges
                diagnostic_face_mask = display_face_mask
                diagnostic_rectangle_mask = (
                    display_rectangle_overlay
                    if display_rectangle_overlay is not None
                    else display_rectangle_mask
                )
                if (
                    simulation_full_frame_edges
                    and display_diagnostic_head_roi is not None
                ):
                    diagnostic_face_mask = _diagnostic_roi_image(
                        display_face_mask,
                        display_diagnostic_head_roi,
                    )
                    diagnostic_rectangle_mask = _diagnostic_roi_image(
                        diagnostic_rectangle_mask,
                        display_diagnostic_head_roi,
                    )
                diagnostic_reference = (
                    diagnostic_edges
                    if diagnostic_edges is not None
                    else diagnostic_mask
                )
                face_reference = (
                    _diagnostic_roi_image(
                        diagnostic_reference,
                        display_diagnostic_head_roi,
                    )
                    if simulation_full_frame_edges
                    and display_diagnostic_head_roi is not None
                    else diagnostic_reference
                )
                if diagnostic_face_mask is None and face_reference is not None:
                    diagnostic_face_mask = numpy.zeros(
                        face_reference.shape[:2],
                        dtype=numpy.uint8,
                    )
                if diagnostic_rectangle_mask is None and face_reference is not None:
                    diagnostic_rectangle_mask = numpy.zeros(
                        face_reference.shape[:2],
                        dtype=numpy.uint8,
                    )
                diagnostic_images = (
                    (WINDOW_MASK, args.display_mask, diagnostic_mask),
                    (WINDOW_EDGES, args.display_edges, diagnostic_edges),
                    (
                        WINDOW_FACE_MASK,
                        args.display_face_mask,
                        diagnostic_face_mask,
                    ),
                    (
                        WINDOW_RECTANGLE_MASK,
                        args.display_rectangle_mask,
                        diagnostic_rectangle_mask,
                    ),
                )
                for window_name, enabled, diagnostic_image in diagnostic_images:
                    if not enabled or diagnostic_image is None:
                        continue
                    diagnostic_shape = tuple(diagnostic_image.shape[:2])
                    if window_name in NATIVE_PIXEL_DIAGNOSTIC_WINDOWS:
                        # WINDOW_AUTOSIZE presents these exact cutout pixels;
                        # never stretch the sparse side evidence or rectangle.
                        last_diagnostic_shapes[window_name] = diagnostic_shape
                        continue
                    if last_diagnostic_shapes.get(window_name) != diagnostic_shape:
                        _resize_diagnostic_window(
                            cv2,
                            window_name,
                            diagnostic_shape,
                            args.diagnostic_window_size_px,
                        )
                        last_diagnostic_shapes[window_name] = diagnostic_shape
                if args.display_mask and mask is not None:
                    cv2.imshow(WINDOW_MASK, diagnostic_mask)
                if args.display_edges and diagnostic_edges is not None:
                    cv2.imshow(WINDOW_EDGES, diagnostic_edges)
                if args.display_face_mask and diagnostic_face_mask is not None:
                    cv2.imshow(WINDOW_FACE_MASK, diagnostic_face_mask)
                if (
                    args.display_rectangle_mask
                    and diagnostic_rectangle_mask is not None
                ):
                    cv2.imshow(WINDOW_RECTANGLE_MASK, diagnostic_rectangle_mask)

            recording_images = {WINDOW_FRAME: annotated}
            if not args.headless:
                for window_name, enabled, diagnostic_image in diagnostic_images:
                    if enabled and diagnostic_image is not None:
                        recording_images[window_name] = diagnostic_image
            if record_start_pending:
                try:
                    recorder.start(recording_images)
                except (RuntimeError, ValueError) as exc:
                    print(f"WARNING: recording did not start: {exc}")
                record_start_pending = False
            elif recorder.active:
                recorder.write(recording_images)

            key = 0 if args.headless else cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("p"):
                print_palette(active_ranges)
            if key == ord("r"):
                if recorder.active:
                    recorder.stop()
                else:
                    try:
                        recorder.start(recording_images)
                    except (RuntimeError, ValueError) as exc:
                        print(f"WARNING: recording did not start: {exc}")
            if key == ord("s") and args.save_snapshot is not None:
                if args.structural_diagnostic:
                    _save_structural_viewer_capture(
                        cv2,
                        args=args,
                        read=read,
                        decoded_source_frame=decoded_source_frame,
                        axis_frame=axis_frame,
                        edge_artifacts=edge_artifacts,
                        annotated=annotated,
                        detector_estimate=detector_estimate,
                        temporal_selection=temporal_selection,
                        target_roi=target_roi,
                        decode_duration_sec=decode_duration_sec,
                        detector_duration_sec=detector_duration_sec,
                        processed_monotonic_sec=time.monotonic(),
                    )
                else:
                    if (
                        args.display_rectangle_mask
                        and display_rectangle_mask is not None
                    ):
                        debug_image = display_rectangle_mask
                    elif args.display_face_mask and display_face_mask is not None:
                        debug_image = display_face_mask
                    else:
                        debug_image = (
                            display_edges
                            if args.axis_source == "edges"
                            and display_edges is not None
                            else display_mask
                        )
                    save_snapshot(cv2, args.save_snapshot, annotated, debug_image)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        if qr_decoder is not None:
            qr_decoder.stop()
        if lidar_source is not None:
            lidar_source.release()
        frame_source.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
