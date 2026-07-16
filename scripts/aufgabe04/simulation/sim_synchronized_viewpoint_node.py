#!/usr/bin/env python3
"""Simulation-only synchronized LiDAR/camera viewpoint optimizer.

The node never publishes velocity. It observes the robot while another
planner/follower is driving, holds one initial axis-acquisition target, can
latch bounded alternate sampling viewpoints for an oblique silhouette, and
commits one face-normal target only after stable silhouette consensus.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.viewpoint_recommendation import (
    FaceCandidate,
    MaterialTarget,
    QrBindingObservation,
    QrFaceLatch,
    SideEvidence,
    StableFaceResolver,
    StandGeometry,
    SynchronizedViewpointRecommendation,
    angular_distance,
    recommendation_to_dict,
)
from scripts.aufgabe04.perception.camera_stand_observation import (
    CameraStandObservation,
    stand_axis_from_camera_yaw,
    write_camera_observation,
)
from scripts.aufgabe04.perception.ros_image_adapter import raw_msg_to_bgr_frame
from scripts.aufgabe04.perception.stand_axis_consensus import AxisConsensusAccumulator
from scripts.aufgabe04.perception.stand_axis_consensus import axis_conditioning
from scripts.aufgabe04.perception.stand_axis_image import (
    StandAxisEdgeDebugArtifacts,
    StandAxisImageEstimate,
    estimate_stand_axis_from_edges,
)
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan, median_range_in_scan_cone
from scripts.aufgabe04.simulation.sim_head_roi import (
    project_target_to_camera,
    qr_corners_inside_roi,
    silhouette_close_kernel,
    silhouette_min_edge_height_px,
    stand_head_roi,
)
from scripts.aufgabe04.simulation.sim_qr_detector import detect_simulated_station_qr_bgr
from scripts.aufgabe04.simulation.sim_viewpoint_optimization import (
    DynamicPreApproachTracker,
    DynamicTargetConfig,
    StationarySettleGate,
    TimedSample,
    ViewpointConfig,
    ViewpointMeasurement,
    ViewpointSamplingLatch,
    evaluate_viewpoint,
    face_normal_candidates,
    newest_synchronized_triplet,
    normalize_angle,
)


def _stamp_sec(message) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) / 1e9


def _yaw(quaternion) -> float:
    siny = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny, cosy)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def _nearest_face_id(face_candidates, pose: Pose2D) -> str:
    return min(
        face_candidates,
        key=lambda face: math.hypot(face.pose.x_m - pose.x_m, face.pose.y_m - pose.y_m),
    ).face_id


@dataclass(frozen=True)
class ConditionedAxisDecision:
    axis_rad: float | None
    confidence: float
    reason: str


def camera_yaw_from_target_line_of_sight(
    optical_axis_yaw_rad: float | None,
    target_center_error_rad: float | None,
) -> float | None:
    """Express PnP face yaw relative to the target line of sight.

    The image estimator reports face orientation from the camera optical axis.
    Obliqueness conditioning is defined relative to the stand line of sight,
    so an off-center target bearing must be removed first.  Map conversion is
    kept separate and uses the raw optical yaw plus synchronized camera heading.
    """

    if optical_axis_yaw_rad is None or target_center_error_rad is None:
        return None
    if not math.isfinite(optical_axis_yaw_rad) or not math.isfinite(
        target_center_error_rad
    ):
        return None
    # Image x increases to the right, which is a negative camera-bearing
    # rotation.  ``target_center_error_rad`` retains that image-space sign, so
    # adding it removes the optical-axis-to-line-of-sight offset.
    return normalize_angle(optical_axis_yaw_rad + target_center_error_rad)


def _conditioned_axis_decision(
    *,
    camera_yaw_rad: float | None,
    silhouette_usable: bool,
    estimate_mode: str,
    expected_head_px: float = math.inf,
    min_expected_head_px: float = 0.0,
    max_obliqueness_rad: float,
    robot_pose: Pose2D,
    stand_pose: Pose2D,
    camera_heading_rad: float | None = None,
    conditioning_yaw_rad: float | None = None,
) -> ConditionedAxisDecision:
    """Gate physical-axis conversion and expose why evidence was rejected."""

    if camera_yaw_rad is None:
        return ConditionedAxisDecision(None, 0.0, "camera_yaw_unavailable")
    if not silhouette_usable:
        return ConditionedAxisDecision(None, 0.0, "silhouette_unavailable")
    if estimate_mode != "face_visible":
        return ConditionedAxisDecision(None, 0.0, "silhouette_not_face_visible")
    if min_expected_head_px > 0.0 and (
        not math.isfinite(expected_head_px)
        or expected_head_px < min_expected_head_px
    ):
        return ConditionedAxisDecision(None, 0.0, "projected_head_too_small")
    conditioning_yaw = (
        camera_yaw_rad
        if conditioning_yaw_rad is None
        else conditioning_yaw_rad
    )
    conditioning = axis_conditioning(
        conditioning_yaw,
        max_obliqueness_rad=max_obliqueness_rad,
    )
    if not conditioning.accepted:
        return ConditionedAxisDecision(None, 0.0, conditioning.reason)
    axis_rad = stand_axis_from_camera_yaw(
        robot_x_m=robot_pose.x_m,
        robot_y_m=robot_pose.y_m,
        stand_x_m=stand_pose.x_m,
        stand_y_m=stand_pose.y_m,
        camera_yaw_rad=camera_yaw_rad,
        camera_heading_rad=camera_heading_rad,
    )
    confidence = max(
        0.0, min(1.0, 1.0 - abs(conditioning_yaw) / (math.pi / 2.0))
    )
    return ConditionedAxisDecision(axis_rad, confidence, "well_conditioned")


def _conditioned_axis_input(
    *,
    camera_yaw_rad: float | None,
    silhouette_usable: bool,
    estimate_mode: str,
    expected_head_px: float = math.inf,
    min_expected_head_px: float = 0.0,
    max_obliqueness_rad: float,
    robot_pose: Pose2D,
    stand_pose: Pose2D,
    camera_heading_rad: float | None = None,
    conditioning_yaw_rad: float | None = None,
) -> tuple[float | None, float]:
    """Convert only a well-conditioned silhouette into tracker evidence."""

    decision = _conditioned_axis_decision(
        camera_yaw_rad=camera_yaw_rad,
        silhouette_usable=silhouette_usable,
        estimate_mode=estimate_mode,
        expected_head_px=expected_head_px,
        min_expected_head_px=min_expected_head_px,
        max_obliqueness_rad=max_obliqueness_rad,
        robot_pose=robot_pose,
        stand_pose=stand_pose,
        camera_heading_rad=camera_heading_rad,
        conditioning_yaw_rad=conditioning_yaw_rad,
    )
    return decision.axis_rad, decision.confidence


def provisional_viewpoint_candidates(
    stand_pose: Pose2D,
    target_pose: Pose2D,
    *,
    near_id: str,
    far_id: str,
) -> tuple[FaceCandidate, FaceCandidate]:
    """Represent one nonphysical sampling ray as a valid antipodal pair."""

    radial_x = target_pose.x_m - stand_pose.x_m
    radial_y = target_pose.y_m - stand_pose.y_m
    radius = math.hypot(radial_x, radial_y)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("provisional viewpoint target must be outside the stand")
    normal = math.atan2(radial_y, radial_x)
    opposite = normalize_angle(normal + math.pi)
    near_pose = Pose2D(
        target_pose.x_m,
        target_pose.y_m,
        normalize_angle(normal + math.pi),
    )
    far_pose = Pose2D(
        stand_pose.x_m + radius * math.cos(opposite),
        stand_pose.y_m + radius * math.sin(opposite),
        normalize_angle(opposite + math.pi),
    )
    return (
        FaceCandidate(near_id, normal, near_pose, False),
        FaceCandidate(far_id, opposite, far_pose, False),
    )


def select_published_viewpoint_pose(
    decision_pose: Pose2D, dynamic_update
) -> Pose2D:
    """Compose camera sampling with the dynamic target hysteresis.

    An unavailable axis can specifically mean that the silhouette is too
    oblique.  In that case the camera evaluator's tangential sampling pose
    must win over a stale tracker pose so the robot can improve the view.
    """

    retained_target_reasons = {
        "target_change_below_threshold",
        "target_frozen_near_stand",
        "target_frozen_with_side_evidence",
        "near_stand_without_side_evidence",
        "linear_motion_too_fast",
        "angular_motion_too_fast",
        "insufficient_axis_samples",
        "axis_consensus_uncertain",
        "axis_samples_not_stable",
        "target_committed",
    }
    if dynamic_update.pose is not None and (
        dynamic_update.accepted or dynamic_update.reason in retained_target_reasons
    ):
        return dynamic_update.pose
    return decision_pose


def should_reseed_face_resolver(
    previous_mode: str, new_mode: str, *, hard_qr_latched: bool
) -> bool:
    return (
        new_mode == "measured"
        and previous_mode != "measured"
        and not hard_qr_latched
    )


def suspend_qr_binding_while_identity_unresolved(
    qr_binding, *, identity_resolved: bool, hard_qr_latched: bool
):
    if identity_resolved or not hard_qr_latched:
        return qr_binding
    return replace(
        qr_binding,
        evidence=SideEvidence(
            kind="none",
            confidence=0.0,
            hard=False,
            valid=False,
            face_id=None,
            provenance="face_identity_unresolved_latch_suspended",
        ),
        accepted=False,
        reason="face_identity_unresolved_latch_suspended",
    )


def should_defer_initial_physical_recommendation(
    *, acquiring_axis: bool, face_identity_resolved: bool, hard_qr_latched: bool
) -> bool:
    """Hold the last sampling route during face-ID resolver bootstrap."""

    return (
        not acquiring_axis
        and not face_identity_resolved
        and not hard_qr_latched
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-topic", default="/camera/image_raw")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--stand-x", required=True, type=float)
    parser.add_argument("--stand-y", required=True, type=float)
    parser.add_argument("--stand-id", default="A")
    parser.add_argument("--stream-id", default="sim-stand-viewpoint")
    parser.add_argument("--stand-radius-m", type=float, default=0.06)
    parser.add_argument("--stand-uncertainty-m", type=float, default=0.02)
    parser.add_argument("--map-frame", default="odom")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--scan-frame", default="base_scan")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--camera-fx-px", type=float, default=381.36246688)
    parser.add_argument("--camera-fy-px", type=float, default=381.36246688)
    parser.add_argument("--camera-cx-px", type=float, default=320.5)
    parser.add_argument("--camera-cy-px", type=float, default=240.5)
    parser.add_argument("--camera-forward-offset-m", type=float, default=0.076)
    parser.add_argument("--camera-lateral-offset-m", type=float, default=0.0)
    parser.add_argument("--camera-height-m", type=float, default=0.093)
    parser.add_argument("--camera-yaw-offset-rad", type=float, default=0.0)
    parser.add_argument("--stand-face-size-m", type=float, default=0.078)
    parser.add_argument("--stand-head-center-height-m", type=float, default=0.165035)
    parser.add_argument("--head-roi-padding-scale", type=float, default=1.6)
    parser.add_argument(
        "--min-silhouette-head-px",
        type=float,
        default=50.0,
        help=(
            "Reject silhouette-axis measurements while the projected stand head is "
            "smaller than this many pixels; LiDAR/fallback navigation remains active."
        ),
    )
    parser.add_argument("--sync-tolerance-sec", type=float, default=0.12)
    parser.add_argument("--target-distance-m", type=float, default=0.30)
    parser.add_argument(
        "--axis-acquisition-distance-m",
        type=float,
        default=0.55,
        help=(
            "Fixed stand distance on the initial robot-to-stand ray while no "
            "stable silhouette axis has been committed."
        ),
    )
    parser.add_argument(
        "--sampling-arrival-tolerance-m",
        type=float,
        default=0.10,
        help=(
            "Distance for considering a latched viewpoint-sampling target reached; "
            "another tangential sample cannot be selected before this gate."
        ),
    )
    parser.add_argument(
        "--axis-acquisition-arrival-tolerance-m",
        type=float,
        default=0.10,
        help=(
            "Distance for recognizing completion of the initial acquisition route. "
            "This is independent of the tighter camera-sampling tolerance."
        ),
    )
    parser.add_argument("--min-distance-m", type=float, default=0.28)
    parser.add_argument("--max-distance-m", type=float, default=0.35)
    parser.add_argument("--max-center-error-deg", type=float, default=12.0)
    parser.add_argument(
        "--max-obliqueness-deg",
        type=float,
        default=20.0,
        help=(
            "Simulation silhouette yaw bound for committing a physical axis. "
            "More oblique heads remain viewpoint-sampling evidence only."
        ),
    )
    parser.add_argument(
        "--max-tangential-step-deg",
        type=float,
        default=35.0,
        help=(
            "Simulation-only bound for one camera-derived sampling correction. "
            "This moves an oblique view but never turns it into accepted axis evidence."
        ),
    )
    parser.add_argument("--max-linear-speed-mps", type=float, default=0.01)
    parser.add_argument("--max-angular-speed-radps", type=float, default=0.05)
    parser.add_argument("--settle-time-sec", type=float, default=0.40)
    parser.add_argument("--consensus-frames", type=int, default=5)
    parser.add_argument("--consensus-max-deviation-deg", type=float, default=8.0)
    parser.add_argument("--qr-visibility-margin-deg", type=float, default=8.0)
    parser.add_argument("--dynamic-freeze-distance-m", type=float, default=0.42)
    parser.add_argument("--dynamic-min-axis-samples", type=int, default=7)
    parser.add_argument("--dynamic-min-target-change-m", type=float, default=0.06)
    parser.add_argument(
        "--process-rate-hz",
        type=float,
        default=3.0,
        help="Expensive silhouette processing rate; camera/scan subscriptions remain unthrottled.",
    )
    parser.add_argument(
        "--fallback-candidate-index",
        type=int,
        default=1,
        help="Deprecated compatibility option; axis acquisition no longer samples guessed tangential targets.",
    )
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--recommended-pose-json", type=Path, required=True)
    parser.add_argument("--observation-json", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--debug-write-hz", type=float, default=0.5)
    return parser


class SimSynchronizedViewpointNode:  # instantiated only with ROS available
    def __init__(self, args) -> None:
        import cv2
        import numpy
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, LaserScan

        class _Node(Node):
            pass

        self.node = _Node("aufgabe04_sim_synchronized_viewpoint")
        self.cv2 = cv2
        self.numpy = numpy
        self.args = args
        self.images = deque(maxlen=8)
        self.scans = deque(maxlen=20)
        self.odometry = deque(maxlen=40)
        self.last_image_stamp = -math.inf
        self.last_recommendation = None
        self.last_debug_write = -math.inf
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stand_axis")
        self.processing_future = None
        self.fallback_reference_pose = None
        self.config = ViewpointConfig(
            min_distance_m=args.min_distance_m,
            max_distance_m=args.max_distance_m,
            target_distance_m=args.target_distance_m,
            max_center_error_rad=math.radians(args.max_center_error_deg),
            max_obliqueness_rad=math.radians(args.max_obliqueness_deg),
            max_tangential_step_rad=math.radians(args.max_tangential_step_deg),
            max_linear_speed_mps=args.max_linear_speed_mps,
            max_angular_speed_radps=args.max_angular_speed_radps,
            settle_time_sec=args.settle_time_sec,
        )
        self.settle_gate = StationarySettleGate(args.settle_time_sec)
        self.sampling_settle_gate = StationarySettleGate(args.settle_time_sec)
        self.sampling_latch = ViewpointSamplingLatch(
            arrival_tolerance_m=args.sampling_arrival_tolerance_m
        )
        self.consensus = AxisConsensusAccumulator(
            required_samples=args.consensus_frames,
            max_deviation_rad=math.radians(args.consensus_max_deviation_deg),
        )
        self.dynamic_tracker = DynamicPreApproachTracker(
            Pose2D(args.stand_x, args.stand_y),
            DynamicTargetConfig(
                approach_offset_m=args.target_distance_m,
                freeze_distance_m=args.dynamic_freeze_distance_m,
                min_axis_samples=args.dynamic_min_axis_samples,
                min_target_translation_m=args.dynamic_min_target_change_m,
            ),
        )
        self.face_resolver = StableFaceResolver()
        self.axis_identity_mode = "uninitialized"
        self.qr_face_latch = QrFaceLatch(
            min_visibility_margin_rad=math.radians(args.qr_visibility_margin_deg)
        )
        self.node.create_subscription(Image, args.image_topic, self._on_image, qos_profile_sensor_data)
        self.node.create_subscription(LaserScan, args.scan_topic, self._on_scan, qos_profile_sensor_data)
        self.node.create_subscription(Odometry, args.odom_topic, self._on_odom, qos_profile_sensor_data)
        self.node.create_timer(1.0 / max(args.process_rate_hz, 1.0), self._schedule_processing)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _schedule_processing(self) -> None:
        if self.processing_future is not None and not self.processing_future.done():
            return
        if self.processing_future is not None:
            # Do not silently lose validation or image-processing exceptions
            # raised in the worker thread.
            self.processing_future.result()
        self.processing_future = self.executor.submit(self._process_latest)

    def _on_image(self, message) -> None:
        self.images.append(TimedSample(_stamp_sec(message), message))

    def _on_scan(self, message) -> None:
        self.scans.append(TimedSample(_stamp_sec(message), message))

    def _on_odom(self, message) -> None:
        self.odometry.append(TimedSample(_stamp_sec(message), message))

    def _invalidate_recommendation(
        self, *, axis_state: str, evidence_state: str, sensor_stamp_sec: float
    ) -> None:
        if self.last_recommendation is None:
            return
        invalid = replace(
            self.last_recommendation,
            observation_unix_sec=time.time(),
            sensor_stamp_sec=sensor_stamp_sec,
            axis_state=axis_state,
            material_target=replace(
                self.last_recommendation.material_target,
                evidence_state=evidence_state,
            ),
        )
        self.last_recommendation = invalid
        _atomic_json(
            self.args.recommended_pose_json,
            recommendation_to_dict(invalid),
        )

    def _process_latest(self) -> None:
        if not self.images:
            return
        newest_image = self.images[-1]
        if newest_image.stamp_sec + self.args.sync_tolerance_sec < self.last_image_stamp:
            self.last_image_stamp = newest_image.stamp_sec
            self.scans.clear()
            self.odometry.clear()
            self.consensus.reset()
            self.settle_gate.update(stamp_sec=newest_image.stamp_sec, ready=False)
            self.face_resolver.reset(self.args.stream_id)
            self.sampling_latch.reset()
            self.axis_identity_mode = "uninitialized"
            self.qr_face_latch.invalidate(
                stream_id=self.args.stream_id,
                reason="sensor_clock_reset",
            )
            _atomic_json(
                self.args.status_json,
                {
                    "schema_version": 1,
                    "state": "clock_reset",
                    "reason": "image_sensor_time_moved_backwards",
                    "image_stamp_sec": newest_image.stamp_sec,
                },
            )
            self._invalidate_recommendation(
                axis_state="invalid_sensor_clock_reset",
                evidence_state="invalid_clock_reset",
                sensor_stamp_sec=newest_image.stamp_sec,
            )
            return
        synchronized = newest_synchronized_triplet(
            tuple(self.images),
            tuple(self.scans),
            tuple(self.odometry),
            min_image_stamp_exclusive=self.last_image_stamp,
            max_delta_sec=self.args.sync_tolerance_sec,
        )
        if synchronized is None:
            _atomic_json(self.args.status_json, {
                "schema_version": 1,
                "state": "buffering_synchronized_tuple",
                "reason": "awaiting_scan_or_odom_partner",
                "image_stamp_sec": newest_image.stamp_sec,
                "sync_tolerance_sec": self.args.sync_tolerance_sec,
                "buffered_image_count": len(self.images),
                "buffered_scan_count": len(self.scans),
                "buffered_odom_count": len(self.odometry),
            })
            return
        image_sample, scan_sample, odom_sample = synchronized
        self.last_image_stamp = image_sample.stamp_sec
        odom = odom_sample.value
        scan = scan_sample.value
        observed_frames = {
            "odom_parent": str(getattr(odom.header, "frame_id", "")),
            "odom_child": str(getattr(odom, "child_frame_id", "")),
            "scan": str(getattr(scan.header, "frame_id", "")),
            "image": str(getattr(image_sample.value.header, "frame_id", "")),
        }
        expected_frames = {
            "odom_parent": self.args.map_frame,
            "odom_child": self.args.base_frame,
            "scan": self.args.scan_frame,
            "image": self.args.camera_frame,
        }
        frame_mismatches = {
            name: {"expected": expected_frames[name], "observed": observed}
            for name, observed in observed_frames.items()
            if observed != expected_frames[name]
        }
        if frame_mismatches:
            _atomic_json(
                self.args.status_json,
                {
                    "schema_version": 1,
                    "state": "frame_mismatch",
                    "reason": "synchronized_sensor_frames_do_not_match_simulation_contract",
                    "frame_mismatches": frame_mismatches,
                },
            )
            self.consensus.reset()
            self.settle_gate.update(stamp_sec=image_sample.stamp_sec, ready=False)
            self.face_resolver.reset(self.args.stream_id)
            self.sampling_latch.reset()
            self.axis_identity_mode = "uninitialized"
            self.qr_face_latch.invalidate(
                stream_id=self.args.stream_id,
                reason="sensor_frame_mismatch",
            )
            self._invalidate_recommendation(
                axis_state="invalid_sensor_frame_mismatch",
                evidence_state="invalid_frame_mismatch",
                sensor_stamp_sec=image_sample.stamp_sec,
            )
            return
        position = odom.pose.pose.position
        robot_yaw = _yaw(odom.pose.pose.orientation)
        robot_pose = Pose2D(position.x, position.y, robot_yaw)
        if self.fallback_reference_pose is None:
            self.fallback_reference_pose = robot_pose
        bearing = normalize_angle(
            math.atan2(self.args.stand_y - position.y, self.args.stand_x - position.x)
            - robot_yaw
        )
        plain_scan = PlainLaserScan(
            ranges=tuple(scan.ranges),
            angle_min=scan.angle_min,
            angle_increment=scan.angle_increment,
            range_min=scan.range_min,
            range_max=scan.range_max,
            scan_frame_id=scan.header.frame_id,
            scan_stamp_sec=scan_sample.stamp_sec,
            receipt_sec=scan_sample.stamp_sec,
        )
        range_query = median_range_in_scan_cone(
            plain_scan,
            bearing_rad=bearing,
            cone_half_angle_rad=math.radians(3.0),
            min_sample_count=1,
        )
        frame = raw_msg_to_bgr_frame(image_sample.value, self.cv2, self.numpy)
        camera_projection = project_target_to_camera(
            robot_x_m=position.x,
            robot_y_m=position.y,
            robot_z_m=position.z,
            robot_yaw_rad=robot_yaw,
            target_x_m=self.args.stand_x,
            target_y_m=self.args.stand_y,
            target_height_m=self.args.stand_head_center_height_m,
            camera_forward_offset_m=self.args.camera_forward_offset_m,
            camera_lateral_offset_m=self.args.camera_lateral_offset_m,
            camera_height_m=self.args.camera_height_m,
            camera_yaw_offset_rad=self.args.camera_yaw_offset_rad,
        )
        lidar_seeded_roi = stand_head_roi(
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            bearing_rad=camera_projection.bearing_rad,
            distance_m=range_query.distance_m,
            camera_fx_px=self.args.camera_fx_px,
            camera_fy_px=self.args.camera_fy_px,
            camera_cx_px=self.args.camera_cx_px,
            camera_cy_px=self.args.camera_cy_px,
            stand_face_size_m=self.args.stand_face_size_m,
            camera_depth_m=camera_projection.depth_m,
            target_height_delta_m=camera_projection.height_delta_m,
            padding_scale=self.args.head_roi_padding_scale,
        )
        qr_detection = None
        if lidar_seeded_roi is not None:
            qr_detection = detect_simulated_station_qr_bgr(
                frame,
                self.cv2,
                roi=(
                    lidar_seeded_roi.x0,
                    lidar_seeded_roi.y0,
                    lidar_seeded_roi.x1,
                    lidar_seeded_roi.y1,
                ),
            )
        qr_inside_target_roi = bool(
            qr_detection is not None
            and qr_corners_inside_roi(qr_detection.corners_px, lidar_seeded_roi)
        )
        # The real stand-axis pipeline derives orientation from the head
        # silhouette, not from QR geometry.  Keep the estimator crop strictly
        # LiDAR/bearing seeded; QR corners are used only for association above.
        roi = lidar_seeded_roi
        if roi is None:
            # A target outside the camera FOV must not fall back to an unrelated
            # full-frame stand or arena boundary.
            roi_frame = frame
            estimate = _unavailable_target_estimate("target_outside_camera_fov")
            debug = StandAxisEdgeDebugArtifacts(
                edges=self.numpy.zeros(frame.shape[:2], dtype=self.numpy.uint8)
            )
        else:
            roi_frame = frame[roi.y0:roi.y1, roi.x0:roi.x1]
            expected_head_px = roi.expected_head_px
            if expected_head_px < self.args.min_silhouette_head_px:
                estimate = _unavailable_target_estimate("projected_head_too_small")
                debug = StandAxisEdgeDebugArtifacts(
                    edges=self.numpy.zeros(roi_frame.shape[:2], dtype=self.numpy.uint8)
                )
            else:
                estimate, debug = estimate_stand_axis_from_edges(
                    self.cv2,
                    roi_frame,
                    edge_preprocess="gray",
                    canny_low=20,
                    canny_high=60,
                    silhouette_only=True,
                    min_area_px=max(40.0, 0.10 * expected_head_px * expected_head_px),
                    # This is already a target-specific projected ROI.  A wall
                    # branch inside it must not raise the minimum head area via
                    # the full-frame largest-contour heuristic.
                    min_face_area_fraction=0.0,
                    min_edge_height_px=silhouette_min_edge_height_px(expected_head_px),
                    close_kernel=silhouette_close_kernel(expected_head_px),
                    stand_width_m=self.args.stand_face_size_m,
                    stand_distance_m=camera_projection.depth_m,
                    camera_fx_px=self.args.camera_fx_px,
                    camera_fy_px=self.args.camera_fy_px,
                    camera_cx_px=self.args.camera_cx_px - roi.x0,
                    camera_cy_px=self.args.camera_cy_px - roi.y0,
                )
        if estimate.corners is not None and roi is not None:
            estimate = replace(
                estimate,
                corners=tuple(
                    type(point)(point.u_px + roi.x0, point.v_px + roi.y0)
                    for point in estimate.corners
                ),
            )
        camera_optical_yaw = (
            None if estimate.yaw_deg is None else math.radians(estimate.yaw_deg)
        )
        center_error = None
        if estimate.corners is not None:
            center_x = sum(point.u_px for point in estimate.corners) / len(estimate.corners)
            center_error = math.atan((center_x - self.args.camera_cx_px) / self.args.camera_fx_px)
        camera_yaw = camera_yaw_from_target_line_of_sight(
            camera_optical_yaw,
            center_error,
        )
        twist = odom.twist.twist
        measurement = ViewpointMeasurement(
            image_stamp_sec=image_sample.stamp_sec,
            scan_stamp_sec=scan_sample.stamp_sec,
            robot_pose=robot_pose,
            linear_speed_mps=twist.linear.x,
            angular_speed_radps=twist.angular.z,
            stand_x_m=self.args.stand_x,
            stand_y_m=self.args.stand_y,
            distance_m=range_query.distance_m,
            camera_center_error_rad=center_error,
            camera_yaw_rad=camera_yaw,
            silhouette_usable=bool(estimate.usable and estimate.mode == "face_visible"),
        )
        decision = evaluate_viewpoint(measurement, self.config)
        qr_texts = () if qr_detection is None else (qr_detection.station_id,)
        view_centered = (
            center_error is not None
            and abs(center_error) <= self.config.max_center_error_rad
        )
        sampling_view_settled = self.sampling_settle_gate.update(
            stamp_sec=image_sample.stamp_sec,
            ready=decision.stationary and view_centered,
        )
        settled = self.settle_gate.update(
            stamp_sec=image_sample.stamp_sec,
            ready=decision.geometrically_ready and decision.stationary,
        )
        camera_heading = normalize_angle(
            robot_pose.yaw_rad + self.args.camera_yaw_offset_rad
        )
        axis_input = _conditioned_axis_decision(
            camera_yaw_rad=camera_optical_yaw,
            silhouette_usable=estimate.usable,
            estimate_mode=estimate.mode,
            expected_head_px=(
                0.0 if lidar_seeded_roi is None else lidar_seeded_roi.expected_head_px
            ),
            min_expected_head_px=self.args.min_silhouette_head_px,
            max_obliqueness_rad=self.config.max_obliqueness_rad,
            robot_pose=robot_pose,
            stand_pose=Pose2D(self.args.stand_x, self.args.stand_y),
            camera_heading_rad=camera_heading,
            conditioning_yaw_rad=camera_yaw,
        )
        if not settled:
            self.dynamic_tracker.reset_uncommitted_samples()
        dynamic_update = self.dynamic_tracker.update(
            robot_pose=robot_pose,
            axis_rad=(axis_input.axis_rad if settled else None),
            measurement_confidence=(axis_input.confidence if settled else 0.0),
            linear_speed_mps=twist.linear.x,
            angular_speed_radps=twist.angular.z,
            freeze_allowed=self.qr_face_latch.latched_evidence is not None,
            allow_opposite_side_switch=False,
        )
        fallback_candidate_index = 0
        decision = replace(
            decision,
            recommended_pose=select_published_viewpoint_pose(
                decision.recommended_pose, dynamic_update
            ),
        )
        consensus = None
        if settled and camera_optical_yaw is not None and qr_texts:
            consensus = self.consensus.add(
                yaw_rad=camera_optical_yaw,
                source=estimate.source,
                side="qr_code_side",
                qr_texts=tuple(qr_texts),
            )
        else:
            self.consensus.reset()

        stand_pose = Pose2D(self.args.stand_x, self.args.stand_y)
        # A filtered angle is diagnostic until the tracker has passed the
        # complete sample/stability gate and committed one physical face.
        recommendation_axis = (
            dynamic_update.stand_axis_rad
            if dynamic_update.pose is not None
            else None
        )
        axis_state = dynamic_update.reason
        axis_identity_mode = "measured"
        acquiring_axis = recommendation_axis is None
        precommit_selected_face_id = ""
        precommit_evidence_state = ""
        if recommendation_axis is None:
            reference = self.fallback_reference_pose or robot_pose
            acquisition_normal = math.atan2(
                reference.y_m - self.args.stand_y,
                reference.x_m - self.args.stand_x,
            )
            acquisition_target = Pose2D(
                self.args.stand_x
                + self.args.axis_acquisition_distance_m * math.cos(acquisition_normal),
                self.args.stand_y
                + self.args.axis_acquisition_distance_m * math.sin(acquisition_normal),
                normalize_angle(acquisition_normal + math.pi),
            )
            previous = self.last_recommendation
            acquisition_reached = (
                previous is not None
                and previous.axis_state == "axis_acquisition"
                and math.hypot(
                    robot_pose.x_m - previous.material_target.pose.x_m,
                    robot_pose.y_m - previous.material_target.pose.y_m,
                )
                <= self.args.axis_acquisition_arrival_tolerance_m
            )
            sampling_update = self.sampling_latch.update(
                robot_pose=robot_pose,
                stationary=decision.stationary,
                axis_input_reason=axis_input.reason,
                candidate_pose=decision.recommended_pose,
                allow_start=acquisition_reached,
                view_centered=view_centered,
                view_settled=sampling_view_settled,
            )
            if sampling_update.active:
                assert sampling_update.target_pose is not None
                axis_state = "viewpoint_sampling"
                axis_identity_mode = "sampling"
                face_candidates = provisional_viewpoint_candidates(
                    stand_pose,
                    sampling_update.target_pose,
                    near_id="sampling_near",
                    far_id="sampling_far",
                )
                precommit_selected_face_id = "sampling_near"
                precommit_evidence_state = "viewpoint_sampling"
            else:
                axis_state = "axis_acquisition"
                axis_identity_mode = "acquisition"
                face_candidates = provisional_viewpoint_candidates(
                    stand_pose,
                    acquisition_target,
                    near_id="acquisition_near",
                    far_id="acquisition_far",
                )
                precommit_selected_face_id = "acquisition_near"
                precommit_evidence_state = "axis_acquisition"
            face_identity_resolved = False
        else:
            self.sampling_latch.reset()
            if should_reseed_face_resolver(
                self.axis_identity_mode,
                axis_identity_mode,
                hard_qr_latched=self.qr_face_latch.latched_evidence is not None,
            ):
                self.face_resolver.reset(self.args.stream_id)
            raw_face_poses = face_normal_candidates(
                stand_pose,
                recommendation_axis,
                self.config.target_distance_m,
            )
            raw_normals = tuple(
                math.atan2(pose.y_m - self.args.stand_y, pose.x_m - self.args.stand_x)
                for pose in raw_face_poses
            )
            resolved_faces = self.face_resolver.update(
                stream_id=self.args.stream_id,
                outward_normals_rad=raw_normals,
            )
            face_candidates = tuple(
                FaceCandidate(
                    face_id=face.face_id,
                    outward_normal_rad=face.outward_normal_rad,
                    pose=Pose2D(
                        self.args.stand_x
                        + self.config.target_distance_m * math.cos(face.outward_normal_rad),
                        self.args.stand_y
                        + self.config.target_distance_m * math.sin(face.outward_normal_rad),
                        normalize_angle(face.outward_normal_rad + math.pi),
                    ),
                    identity_resolved=face.identity_resolved,
                )
                for face in resolved_faces.faces
            )
            face_identity_resolved = resolved_faces.identity_resolved
        self.axis_identity_mode = axis_identity_mode
        if should_defer_initial_physical_recommendation(
            acquiring_axis=acquiring_axis,
            face_identity_resolved=face_identity_resolved,
            hard_qr_latched=self.qr_face_latch.latched_evidence is not None,
        ):
            self._write_debug(frame, roi_frame, debug, estimate, image_sample.stamp_sec)
            return

        qr_binding_observation = None
        if consensus is not None and qr_detection is not None:
            observer_normal = math.atan2(
                robot_pose.y_m - self.args.stand_y,
                robot_pose.x_m - self.args.stand_x,
            )
            ranked_faces = sorted(
                face_candidates,
                key=lambda face: angular_distance(face.outward_normal_rad, observer_normal),
            )
            visibility_margin = angular_distance(
                ranked_faces[1].outward_normal_rad, observer_normal
            ) - angular_distance(ranked_faces[0].outward_normal_rad, observer_normal)
            qr_binding_observation = QrBindingObservation(
                face_id=ranked_faces[0].face_id,
                confidence=max(0.0, min(1.0, 1.0 - qr_detection.mismatch_fraction)),
                provenance=f"sim_qr:{qr_detection.station_id}:lidar_head_roi",
                registry_match=qr_detection.station_id == self.args.stand_id,
                inside_target_roi=qr_inside_target_roi,
                distinct_fresh_frame_consensus=consensus.sample_count
                >= self.args.consensus_frames,
                visibility_margin_rad=visibility_margin,
                identity_resolved=face_identity_resolved,
                contradiction=(
                    self.qr_face_latch.latched_evidence is not None
                    and self.qr_face_latch.latched_evidence.face_id
                    != ranked_faces[0].face_id
                ),
            )
        qr_binding = self.qr_face_latch.update(
            stream_id=self.args.stream_id,
            observation=qr_binding_observation,
            known_face_ids={face.face_id for face in face_candidates},
        )
        suspended_binding = suspend_qr_binding_while_identity_unresolved(
            qr_binding,
            identity_resolved=face_identity_resolved,
            hard_qr_latched=self.qr_face_latch.latched_evidence is not None,
        )
        if suspended_binding is not qr_binding:
            # Suspend, but do not remap, a hard binding while physical face
            # continuity is unresolved.  It can resume only after the same
            # trusted IDs relock; the emitted recommendation remains valid
            # and explicitly fail-closed in the meantime.
            qr_binding = suspended_binding
            axis_state = "invalid_face_identity_unresolved_latch_suspended"
        if qr_binding.reason in (
            "contradicts_latch",
            "visibility_near_tangent",
            "face_identity_unresolved",
        ):
            axis_state = f"invalid_{qr_binding.reason}"

        if acquiring_axis:
            selected_face_id = precommit_selected_face_id
            evidence_state = precommit_evidence_state
        elif qr_binding.evidence.hard and qr_binding.evidence.valid:
            selected_face_id = str(qr_binding.evidence.face_id)
            evidence_state = "hard_qr"
        else:
            selected_face_id = _nearest_face_id(face_candidates, decision.recommended_pose)
            evidence_state = (
                "invalid_qr" if axis_state.startswith("invalid_") else "ambiguous_axis"
            )
        selected_face = next(
            face for face in face_candidates if face.face_id == selected_face_id
        )
        published_side_evidence = (
            SideEvidence(
                kind="none",
                confidence=0.0,
                hard=False,
                valid=False,
                face_id=None,
                provenance=f"{precommit_evidence_state}_axis_uncommitted",
            )
            if acquiring_axis
            else qr_binding.evidence
        )
        recommendation = SynchronizedViewpointRecommendation(
            schema_version=1,
            simulation_only=True,
            stream_id=self.args.stream_id,
            stand_id=self.args.stand_id,
            planning_frame=self.args.map_frame,
            source="synchronized_lidar_camera_viewpoint",
            observation_unix_sec=time.time(),
            sensor_stamp_sec=image_sample.stamp_sec,
            stand=StandGeometry(
                center=stand_pose,
                radius_m=self.args.stand_radius_m,
                uncertainty_m=self.args.stand_uncertainty_m,
                provenance="synchronized_lidar_cluster",
            ),
            robot_pose=robot_pose,
            axis_confidence=(0.0 if acquiring_axis else dynamic_update.axis_confidence),
            axis_state=axis_state,
            face_candidates=(face_candidates[0], face_candidates[1]),
            side_evidence=published_side_evidence,
            material_target=MaterialTarget(
                face_id=selected_face.face_id,
                pose=selected_face.pose,
                evidence_state=evidence_state,
            ),
        )
        recommendation_payload = recommendation_to_dict(recommendation)
        self.last_recommendation = recommendation
        status = {
            "schema_version": 1,
            **asdict(decision),
            "image_stamp_sec": image_sample.stamp_sec,
            "scan_stamp_sec": scan_sample.stamp_sec,
            "odom_stamp_sec": odom_sample.stamp_sec,
            "settled": settled,
            "sampling_view_settled": sampling_view_settled,
            "qr_texts": list(qr_texts),
            "camera_yaw_deg": None if camera_yaw is None else math.degrees(camera_yaw),
            "camera_optical_yaw_deg": (
                None
                if camera_optical_yaw is None
                else math.degrees(camera_optical_yaw)
            ),
            "center_error_deg": None if center_error is None else math.degrees(center_error),
            "distance_m": range_query.distance_m,
            "camera_projection": asdict(camera_projection),
            "fallback_candidate_index": fallback_candidate_index,
            "axis_estimator_reason": estimate.reason,
            "axis_estimator_source": estimate.source,
            "axis_contour_area_px": estimate.contour_area_px,
            "silhouette_close_kernel": silhouette_close_kernel(
                lidar_seeded_roi.expected_head_px
            ) if lidar_seeded_roi is not None else None,
            "head_roi": None if roi is None else asdict(roi),
            "lidar_seeded_head_roi": (
                None if lidar_seeded_roi is None else asdict(lidar_seeded_roi)
            ),
            "qr_inside_target_roi": qr_inside_target_roi,
            "qr_binding_reason": qr_binding.reason,
            "dynamic_target": asdict(dynamic_update),
            "robot_pose": asdict(robot_pose),
            "recommendation": recommendation_payload,
        }
        status["recommended_pose"] = asdict(decision.recommended_pose)
        _atomic_json(self.args.status_json, status)
        _atomic_json(self.args.recommended_pose_json, recommendation_payload)
        self._write_debug(frame, roi_frame, debug, estimate, image_sample.stamp_sec)
        if (
            consensus is None
            or not qr_binding.evidence.hard
            or not qr_binding.evidence.valid
        ):
            return
        observation = CameraStandObservation(
            schema_version=1,
            # Camera observation freshness elsewhere uses workstation wall time.
            observed_at_sec=time.time(),
            image_topic=self.args.image_topic,
            camera_frame=self.args.camera_frame,
            map_frame=self.args.map_frame,
            robot_x_m=position.x,
            robot_y_m=position.y,
            stand_x_m=self.args.stand_x,
            stand_y_m=self.args.stand_y,
            stand_axis_rad=stand_axis_from_camera_yaw(
                robot_x_m=position.x,
                robot_y_m=position.y,
                stand_x_m=self.args.stand_x,
                stand_y_m=self.args.stand_y,
                camera_yaw_rad=consensus.yaw_rad,
                camera_heading_rad=normalize_angle(
                    robot_pose.yaw_rad + self.args.camera_yaw_offset_rad
                ),
            ),
            axis_confidence=0.85,
            side="qr_code_side",
            side_confidence=1.0,
            qr_texts=tuple(qr_texts),
        )
        write_camera_observation(self.args.observation_json, observation)

    def _write_debug(self, frame, roi_frame, debug, estimate, stamp_sec: float) -> None:
        if self.args.debug_dir is None:
            return
        interval = 1.0 / max(self.args.debug_write_hz, 0.01)
        if stamp_sec - self.last_debug_write < interval:
            return
        self.last_debug_write = stamp_sec
        self.args.debug_dir.mkdir(parents=True, exist_ok=True)
        self.cv2.imwrite(str(self.args.debug_dir / "latest_frame.png"), frame)
        self.cv2.imwrite(str(self.args.debug_dir / "latest_head_roi.png"), roi_frame)
        self.cv2.imwrite(str(self.args.debug_dir / "latest_edges.png"), debug.edges)
        if debug.face_mask is not None:
            self.cv2.imwrite(str(self.args.debug_dir / "latest_face_mask.png"), debug.face_mask)
        if debug.rectangle_mask is not None:
            self.cv2.imwrite(str(self.args.debug_dir / "latest_rectangle.png"), debug.rectangle_mask)
        _atomic_json(self.args.debug_dir / "latest_estimate.json", {
            "stamp_sec": stamp_sec,
            "usable": estimate.usable,
            "reason": estimate.reason,
            "mode": estimate.mode,
            "source": estimate.source,
            "yaw_deg": estimate.yaw_deg,
            "contour_area_px": estimate.contour_area_px,
        })


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.fallback_candidate_index <= 7:
        raise SystemExit("--fallback-candidate-index must be in [1, 7]")
    if args.stand_radius_m <= 0.0 or args.stand_uncertainty_m < 0.0:
        raise SystemExit("stand radius must be positive and uncertainty non-negative")
    positive_values = {
        "--camera-fx-px": args.camera_fx_px,
        "--camera-fy-px": args.camera_fy_px,
        "--camera-height-m": args.camera_height_m,
        "--stand-face-size-m": args.stand_face_size_m,
        "--stand-head-center-height-m": args.stand_head_center_height_m,
        "--head-roi-padding-scale": args.head_roi_padding_scale,
        "--min-silhouette-head-px": args.min_silhouette_head_px,
        "--target-distance-m": args.target_distance_m,
        "--axis-acquisition-distance-m": args.axis_acquisition_distance_m,
        "--sampling-arrival-tolerance-m": args.sampling_arrival_tolerance_m,
        "--axis-acquisition-arrival-tolerance-m": (
            args.axis_acquisition_arrival_tolerance_m
        ),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_values.values()):
        raise SystemExit(
            "camera intrinsics, heights, stand size, and ROI padding must be finite and positive"
        )
    if args.axis_acquisition_distance_m <= args.max_distance_m:
        raise SystemExit(
            "--axis-acquisition-distance-m must exceed --max-distance-m so acquisition "
            "and the final observation band are distinct"
        )
    if args.dynamic_min_axis_samples < 2:
        raise SystemExit("--dynamic-min-axis-samples must be at least 2")
    if (
        not math.isfinite(args.max_tangential_step_deg)
        or not 0.0 < args.max_tangential_step_deg <= 45.0
    ):
        raise SystemExit("--max-tangential-step-deg must be in (0, 45]")
    for option, frame in (
        ("--map-frame", args.map_frame),
        ("--base-frame", args.base_frame),
        ("--scan-frame", args.scan_frame),
        ("--camera-frame", args.camera_frame),
    ):
        if not frame or frame.startswith("/") or frame.endswith("/") or "//" in frame:
            raise SystemExit(f"{option} must be a non-empty relative ROS frame")
    try:
        import rclpy
    except ImportError as exc:
        raise SystemExit("simulation synchronized viewpoint node requires ROS 2") from exc
    rclpy.init(args=None)
    wrapper = SimSynchronizedViewpointNode(args)
    try:
        rclpy.spin(wrapper.node)
    except KeyboardInterrupt:
        pass
    except Exception:
        # SIGTERM from timeout / launch shutdown may invalidate the context
        # before spin returns. Preserve genuine runtime exceptions.
        if rclpy.ok():
            raise
    finally:
        wrapper.close()
        try:
            wrapper.node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
