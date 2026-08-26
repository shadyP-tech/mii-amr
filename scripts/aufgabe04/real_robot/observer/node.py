#!/usr/bin/env python3
"""Passive real-robot camera/LiDAR viewpoint observer.

This node never creates a publisher and never commands motion.  It requires a
sealed real-robot profile, measured camera calibration, live ``CameraInfo``,
compressed onboard images, a synchronized LaserScan, and exact-time TF.  Once
stationary silhouette and QR evidence reach consensus it writes the same
environment-tagged recommendation contract used by the shared survey planner.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    recommendation_to_dict,
)
from scripts.aufgabe04.perception.camera_stand_observation import (
    stand_axis_from_camera_yaw,
)
from scripts.aufgabe04.perception.camera_calibration import (
    rectify_bgr_frame as _rectify_bgr_frame,
)
from scripts.aufgabe04.perception.ros_image_adapter import (
    compressed_msg_stamp_sec,
    compressed_msg_to_bgr_frame,
)
from scripts.aufgabe04.perception.candidate_lidar_association import (
    associate_candidate_lidar_target,
)
from scripts.aufgabe04.perception.stand_axis_consensus import axis_conditioning
from scripts.aufgabe04.perception.stand_axis.real_camera_profile import (
    RealCameraStandAxisProfile,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import (
    load_stand_model,
)
from scripts.aufgabe04.perception.stand_axis.pose_tracking import (
    MetricPoseTracker,
)
from scripts.aufgabe04.perception.stand_axis_image import (
    estimate_stand_axis_from_edges,
    estimate_stand_axis_from_metric_model,
)
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.perception.stand_axis_handoff import (
    RigidTransform,
    rectified_pixel_bearing_in_scan,
)
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_texts_bgr
from scripts.aufgabe04.real_robot.configuration.geometry import (
    intrinsics_from_camera_info,
    optical_heading_from_transform,
    pose2d_from_transform,
    project_optical_point,
    project_rectified_image_direction,
    roi_from_projection,
    transform_point,
)
from scripts.aufgabe04.real_robot.configuration.profile import (
    camera_calibration_sha256,
    camera_info_mismatches,
    load_camera_calibration,
    load_real_robot_profile,
    real_robot_profile_sha256,
    transform_mismatches,
)
from scripts.aufgabe04.real_robot.observer.contract import (
    PASSIVE_VIEWPOINT_OBSERVER_VERSION,
)
from scripts.aufgabe04.real_robot.observer.tf_retry import (
    PassiveObserverTfRetryScheduler,
)
from scripts.aufgabe04.real_robot.observer.evidence import (
    EvidencePose,
    PassiveObserverEvidence,
)
from scripts.aufgabe04.real_robot.configuration.recommendation import (
    build_real_viewpoint_recommendation,
)


OBSERVER_VERSION = PASSIVE_VIEWPOINT_OBSERVER_VERSION
AUTO_QR_ID = "auto"


def _head_scale_gate(
    *,
    expected_size_px: float,
    left_height_px: float,
    right_height_px: float,
) -> dict[str, object]:
    """Check that accepted head sides have the calibrated physical scale."""

    expected = float(expected_size_px)
    heights = (float(left_height_px), float(right_height_px))
    measured = sum(heights) / 2.0
    ratio = measured / max(expected, 1.0e-9)
    balance = min(heights) / max(max(heights), 1.0e-9)
    accepted = (
        all(math.isfinite(value) and value > 0.0 for value in (*heights, expected))
        and 0.60 <= ratio <= 1.35
        and balance >= 0.65
    )
    return {
        "accepted": accepted,
        "expected_size_px": expected,
        "measured_height_px": measured,
        "left_height_px": heights[0],
        "right_height_px": heights[1],
        "height_ratio": ratio,
        "side_balance": balance,
        "reason": "ok" if accepted else "head_size_projection_mismatch",
    }


@dataclass(frozen=True)
class _StampedMessage:
    stamp_sec: float
    value: object


@dataclass(frozen=True)
class _SynchronizedSensorTuple:
    """One immutable image/scan/CameraInfo tuple retried at exact TF time."""

    image: _StampedMessage
    scan: _StampedMessage
    camera_info: _StampedMessage


def _stamp_sec(message) -> float:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        raise ValueError("ROS message has no header stamp")
    value = float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("ROS message stamp must be finite and positive")
    return value


def _transform_values(transform) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return (
        (float(translation.x), float(translation.y), float(translation.z)),
        (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
    )


def _nearest(
    samples: tuple[_StampedMessage, ...],
    *,
    stamp_sec: float,
    tolerance_sec: float,
) -> _StampedMessage | None:
    if not samples:
        return None
    nearest = min(samples, key=lambda item: abs(item.stamp_sec - stamp_sec))
    return (
        nearest
        if abs(nearest.stamp_sec - stamp_sec) <= tolerance_sec
        else None
    )


def _pose_is_stationary(
    previous: Pose2D | None,
    current: Pose2D,
    *,
    max_translation_m: float,
    max_rotation_rad: float,
) -> bool:
    if previous is None:
        return True
    translation = math.hypot(current.x_m - previous.x_m, current.y_m - previous.y_m)
    rotation = abs(
        math.atan2(
            math.sin(current.yaw_rad - previous.yaw_rad),
            math.cos(current.yaw_rad - previous.yaw_rad),
        )
    )
    return translation <= max_translation_m and rotation <= max_rotation_rad


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    """Append one durable observer state without replacing prior evidence."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _stand_axis_profile_from_args(args) -> RealCameraStandAxisProfile:
    """Build the pure real-camera estimator profile from parsed CLI values."""

    return RealCameraStandAxisProfile.from_cli(
        edge_preprocess=args.edge_preprocess,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
    )


class PassiveRealViewpointNode:  # pragma: no cover - requires ROS runtime.
    def __init__(self, args) -> None:
        import cv2
        import numpy
        import rclpy
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import CameraInfo, CompressedImage, LaserScan
        from tf2_ros import Buffer, TransformException, TransformListener

        class _Node(Node):
            pass

        self.node = _Node("aufgabe04_real_passive_viewpoint")
        if bool(self.node.get_parameter("use_sim_time").value):
            raise RuntimeError("real passive viewpoint node requires use_sim_time=false")
        self.args = args
        self.cv2 = cv2
        self.numpy = numpy
        self.Duration = Duration
        self.Time = Time
        self.TransformException = TransformException
        self.stand_axis_profile = _stand_axis_profile_from_args(args)
        self.stand_model_profile = (
            None
            if args.stand_model_profile is None
            else load_stand_model(args.stand_model_profile)
        )
        if (
            self.stand_model_profile is not None
            and not self.stand_model_profile.committable
        ):
            raise ValueError(
                "passive observation requires a measured stand model profile"
            )
        self.model_pose_tracker = (
            None
            if self.stand_model_profile is None
            else MetricPoseTracker(prediction_ttl_sec=0.25)
        )
        self.profile = load_real_robot_profile(args.robot_profile)
        self.calibration = load_camera_calibration(args.camera_calibration)
        if camera_calibration_sha256(self.calibration) != (
            self.profile.calibration_profile_sha256
        ):
            raise ValueError(
                "robot profile references a different camera calibration"
            )
        if self.profile.camera_optical_frame != self.calibration.camera_optical_frame:
            raise ValueError("robot and calibration camera frames differ")
        if self.profile.base_frame != self.calibration.base_frame:
            raise ValueError("robot and calibration base frames differ")
        self.runtime = self.profile.resolved_runtime()
        self.images: deque[_StampedMessage] = deque(maxlen=8)
        self.scans: deque[_StampedMessage] = deque(maxlen=20)
        self.camera_infos: deque[_StampedMessage] = deque(maxlen=8)
        self.last_processed_image_stamp = -math.inf
        self.tf_retry_scheduler = (
            PassiveObserverTfRetryScheduler[_SynchronizedSensorTuple]()
        )
        self._active_tf_request: dict[str, object] | None = None
        self._tf_retry_tuple_count = 0
        self._tf_retry_exhausted_tuple_count = 0
        self._tf_retry_peak_count = 0
        self.last_pose: Pose2D | None = None
        self.observation_evidence: PassiveObserverEvidence | None = None
        self._last_observation_update = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.completed = False
        self.axis_observation_committed = False
        self.node.create_subscription(
            CompressedImage,
            self.profile.resolved_compressed_image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self.node.create_subscription(
            CameraInfo,
            self.profile.resolved_camera_info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self.node.create_subscription(
            LaserScan,
            self.runtime.scan_topic,
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.node.create_timer(
            1.0 / max(args.process_rate_hz, 0.5),
            self._process_latest,
        )
        self.node.create_timer(
            1.0 / max(args.tf_retry_rate_hz, 1.0),
            self._retry_pending_exact_tf,
        )
        self._write_status(
            "waiting_for_sensors",
            resolved_runtime=self.runtime.as_log_dict(),
        )

    def _on_image(self, message) -> None:
        stamp = compressed_msg_stamp_sec(message)
        if stamp is not None and stamp > 0.0:
            self.images.append(_StampedMessage(stamp, message))

    def _on_camera_info(self, message) -> None:
        try:
            self.camera_infos.append(_StampedMessage(_stamp_sec(message), message))
        except ValueError:
            return

    def _on_scan(self, message) -> None:
        try:
            self.scans.append(_StampedMessage(_stamp_sec(message), message))
        except ValueError:
            return

    def _retry_pending_exact_tf(self) -> None:
        """Yield-driven retry for one frozen tuple while TF callbacks advance."""

        if self.tf_retry_scheduler.pending_frame is not None:
            self._process_latest()

    @staticmethod
    def _evidence_pose(pose: Pose2D) -> EvidencePose:
        return EvidencePose(pose.x_m, pose.y_m, pose.yaw_rad)

    def _target_evidence_key(self) -> str:
        return (
            f"{self.args.stream_id}:{self.args.stand_id}:"
            f"{self.args.stand_x:.9f}:{self.args.stand_y:.9f}"
        )

    def _ensure_observation_evidence(
        self,
        pose: Pose2D,
    ) -> PassiveObserverEvidence:
        if self.observation_evidence is None:
            self.observation_evidence = PassiveObserverEvidence(
                target_key=self._target_evidence_key(),
                anchor_pose=self._evidence_pose(pose),
                required_axis_samples=self.args.consensus_frames,
                max_axis_deviation_rad=math.radians(
                    self.args.consensus_max_deviation_deg
                ),
                axis_ttl_sec=self.args.consensus_axis_ttl_sec,
                qr_ttl_sec=self.args.consensus_qr_ttl_sec,
                max_lidar_age_sec=self.args.max_sensor_age_sec,
                max_sensor_skew_sec=self.args.sync_tolerance_sec,
                max_future_stamp_sec=self.args.max_future_timestamp_sec,
                max_anchor_translation_m=self.args.stationary_translation_m,
                max_anchor_rotation_rad=math.radians(
                    self.args.stationary_rotation_deg
                ),
            )
        return self.observation_evidence

    def _note_observation_soft_miss(
        self,
        reason: str,
        *,
        stamp_sec: float,
        pose: Pose2D | None = None,
    ) -> None:
        evidence_pose = self.last_pose if pose is None else pose
        if evidence_pose is None or self.observation_evidence is None:
            return
        self._last_observation_update = self.observation_evidence.note_soft_miss(
            target_key=self._target_evidence_key(),
            pose=self._evidence_pose(evidence_pose),
            stamp_sec=max(0.0, float(stamp_sec)),
            reason=reason,
        )

    def _reset_observation_evidence(self) -> None:
        """Drop evidence after a sealed sensor/frame contract violation."""

        self.observation_evidence = None
        self._last_observation_update = None

    def _record_observation_frame(
        self,
        *,
        robot_pose: Pose2D,
        image_stamp_sec: float,
        scan_stamp_sec: float,
        observed_at_sec: float,
        lidar_associated: bool,
        axis_yaw_rad: float | None,
        axis_source: str | None,
        qr_texts: tuple[str, ...],
    ):
        evidence = self._ensure_observation_evidence(robot_pose)
        update = evidence.record_frame(
            target_key=self._target_evidence_key(),
            pose=self._evidence_pose(robot_pose),
            frame_stamp_sec=image_stamp_sec,
            lidar_stamp_sec=scan_stamp_sec,
            observed_at_sec=observed_at_sec,
            lidar_associated=lidar_associated,
            axis_yaw_rad=axis_yaw_rad,
            axis_source=axis_source,
            qr_texts=qr_texts,
        )
        self._last_observation_update = update
        return update

    def _lookup(self, target_frame: str, source_frame: str, stamp) -> object:
        query_time = self.Time.from_msg(stamp)
        query_stamp_sec = (
            float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
        )
        self._active_tf_request = {
            "target_frame": target_frame,
            "source_frame": source_frame,
            "query_kind": "exact_sensor_time",
            "query_stamp_sec": query_stamp_sec,
        }
        return self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            query_time,
            timeout=self.Duration(seconds=0.0),
        )

    def _lookup_static_transform(
        self,
        target_frame: str,
        source_frame: str,
    ) -> object:
        """Read the time-invariant, calibration-checked camera extrinsic."""

        self._active_tf_request = {
            "target_frame": target_frame,
            "source_frame": source_frame,
            "query_kind": "time_invariant_camera_extrinsic",
            "query_stamp_sec": None,
        }
        return self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            self.Time(),
            timeout=self.Duration(seconds=0.0),
        )

    def _next_sensor_tuple(self) -> _SynchronizedSensorTuple | None:
        pending = self.tf_retry_scheduler.pending_frame
        if pending is not None:
            return pending.frame
        if not self.images:
            return None
        image = self.images[-1]
        if image.stamp_sec <= self.last_processed_image_stamp:
            return None
        scan = _nearest(
            tuple(self.scans),
            stamp_sec=image.stamp_sec,
            tolerance_sec=self.args.sync_tolerance_sec,
        )
        camera_info = _nearest(
            tuple(self.camera_infos),
            stamp_sec=image.stamp_sec,
            tolerance_sec=self.args.camera_info_tolerance_sec,
        )
        if scan is None or camera_info is None:
            self.last_processed_image_stamp = image.stamp_sec
            self._note_observation_soft_miss(
                "awaiting_synchronized_sensors",
                stamp_sec=image.stamp_sec,
            )
            self._write_status(
                "awaiting_synchronized_sensors",
                image_stamp_sec=image.stamp_sec,
                scan_available=scan is not None,
                camera_info_available=camera_info is not None,
            )
            return None
        sensor_tuple = _SynchronizedSensorTuple(
            image=image,
            scan=scan,
            camera_info=camera_info,
        )
        accepted = self.tf_retry_scheduler.offer(
            sensor_tuple,
            stamp_sec=image.stamp_sec,
        )
        if not accepted:
            raise RuntimeError("new sensor tuple replaced pending TF retry")
        return sensor_tuple

    def _discard_sensor_tuple(
        self,
        sensor_tuple: _SynchronizedSensorTuple,
        *,
        reason: str,
    ) -> None:
        stamp_sec = sensor_tuple.image.stamp_sec
        self.tf_retry_scheduler.discard(
            stamp_sec=stamp_sec,
            reason=reason,
        )
        self.last_processed_image_stamp = stamp_sec

    def _consume_transform_ready_tuple(
        self,
        sensor_tuple: _SynchronizedSensorTuple,
    ) -> None:
        stamp_sec = sensor_tuple.image.stamp_sec
        self.tf_retry_scheduler.mark_transform_ready(stamp_sec=stamp_sec)
        self.tf_retry_scheduler.consume(stamp_sec=stamp_sec)
        self.last_processed_image_stamp = stamp_sec

    def _defer_for_exact_tf(
        self,
        sensor_tuple: _SynchronizedSensorTuple,
        *,
        reason: str,
    ) -> None:
        observed_sec = time.time()
        stamp_sec = sensor_tuple.image.stamp_sec
        evidence = self.tf_retry_scheduler.mark_transform_unavailable(
            stamp_sec=stamp_sec,
            observed_sec=observed_sec,
            reason=reason,
        )
        if evidence.retry_count == 1:
            self._tf_retry_tuple_count = (
                getattr(self, "_tf_retry_tuple_count", 0) + 1
            )
        self._tf_retry_peak_count = max(
            getattr(self, "_tf_retry_peak_count", 0),
            evidence.retry_count,
        )
        first_failure = evidence.first_failure_time_sec
        retry_elapsed_sec = (
            0.0
            if first_failure is None
            else max(0.0, observed_sec - first_failure)
        )
        retry_exhausted = retry_elapsed_sec >= self.args.tf_timeout_sec
        state = "tf_pending_exact_time"
        if retry_exhausted:
            self._tf_retry_exhausted_tuple_count = (
                getattr(self, "_tf_retry_exhausted_tuple_count", 0) + 1
            )
            self._discard_sensor_tuple(
                sensor_tuple,
                reason="exact-time TF retry budget exhausted",
            )
            state = "tf_retry_exhausted"
        # Persist the transition and its terminal result, not every 50 Hz
        # poll.  Repeated fsyncs inside this single-threaded ROS callback would
        # delay the TF subscription callbacks that the retry is yielding to.
        if evidence.retry_count == 1 or retry_exhausted:
            self._write_status(
                state,
                reason=reason,
                transform_request=self._active_tf_request,
                image_stamp_sec=sensor_tuple.image.stamp_sec,
                scan_stamp_sec=sensor_tuple.scan.stamp_sec,
                tf_retry_attempt=asdict(evidence),
                tf_retry_elapsed_sec=retry_elapsed_sec,
                tf_retry_timeout_sec=self.args.tf_timeout_sec,
                retry_exhausted=retry_exhausted,
            )

    def _process_latest(self) -> None:
        if self.completed:
            return
        sensor_tuple = self._next_sensor_tuple()
        if sensor_tuple is None:
            return
        image = sensor_tuple.image
        scan = sensor_tuple.scan
        camera_info = sensor_tuple.camera_info
        now_sec = self.node.get_clock().now().nanoseconds / 1_000_000_000.0
        image_age = now_sec - image.stamp_sec
        if (
            image_age < -self.args.max_future_timestamp_sec
            or image_age > self.args.max_sensor_age_sec
        ):
            had_transient_tf_retry = (
                self.tf_retry_scheduler.evidence.retry_count > 0
            )
            self._note_observation_soft_miss(
                "stale_sensor_tuple",
                stamp_sec=image.stamp_sec,
            )
            self._discard_sensor_tuple(
                sensor_tuple,
                reason="sensor tuple outside freshness window",
            )
            self._write_status(
                "stale_sensor_tuple",
                image_stamp_sec=image.stamp_sec,
                image_age_sec=image_age,
                transient_tf_retry=had_transient_tf_retry,
            )
            return
        info_mismatches = camera_info_mismatches(
            self.calibration,
            camera_info.value,
        )
        if info_mismatches:
            self._reset_observation_evidence()
            self._discard_sensor_tuple(
                sensor_tuple,
                reason="CameraInfo does not match sealed calibration",
            )
            self._write_status(
                "camera_info_mismatch",
                mismatches=list(info_mismatches),
            )
            return
        image_message = image.value
        scan_message = scan.value
        frame_mismatches = []
        image_frame = str(image_message.header.frame_id).strip("/")
        scan_frame = str(scan_message.header.frame_id).strip("/")
        if image_frame != self.profile.camera_optical_frame:
            frame_mismatches.append(
                "compressed image frame "
                f"{image_frame!r} != {self.profile.camera_optical_frame!r}"
            )
        if scan_frame != self.profile.scan_frame:
            frame_mismatches.append(
                f"LaserScan frame {scan_frame!r} != {self.profile.scan_frame!r}"
            )
        if frame_mismatches:
            self._reset_observation_evidence()
            self._discard_sensor_tuple(
                sensor_tuple,
                reason="sensor frame does not match sealed profile",
            )
            self._write_status(
                "sensor_frame_mismatch",
                mismatches=frame_mismatches,
            )
            return
        try:
            map_from_base = self._lookup(
                self.profile.map_frame,
                self.profile.base_frame,
                image_message.header.stamp,
            )
            map_from_camera = self._lookup(
                self.profile.map_frame,
                self.profile.camera_optical_frame,
                image_message.header.stamp,
            )
            camera_from_map = self._lookup(
                self.profile.camera_optical_frame,
                self.profile.map_frame,
                image_message.header.stamp,
            )
            scan_from_map = self._lookup(
                self.profile.scan_frame,
                self.profile.map_frame,
                scan_message.header.stamp,
            )
            base_from_camera = self._lookup_static_transform(
                self.profile.base_frame,
                self.profile.camera_optical_frame,
            )
            scan_from_camera_transform = self._lookup_static_transform(
                self.profile.scan_frame,
                self.profile.camera_optical_frame,
            )
        except self.TransformException as exc:
            self._defer_for_exact_tf(sensor_tuple, reason=str(exc))
            return
        self._consume_transform_ready_tuple(sensor_tuple)
        extrinsic_mismatches = transform_mismatches(
            self.calibration.base_to_camera,
            base_from_camera,
            translation_tolerance_m=self.args.extrinsic_translation_tolerance_m,
            rotation_tolerance_rad=math.radians(
                self.args.extrinsic_rotation_tolerance_deg
            ),
        )
        if extrinsic_mismatches:
            self._reset_observation_evidence()
            self._write_status(
                "camera_extrinsic_mismatch",
                mismatches=list(extrinsic_mismatches),
            )
            return
        robot_pose = pose2d_from_transform(map_from_base)
        if not _pose_is_stationary(
            self.last_pose,
            robot_pose,
            max_translation_m=self.args.stationary_translation_m,
            max_rotation_rad=math.radians(self.args.stationary_rotation_deg),
        ):
            self._note_observation_soft_miss(
                "robot_not_stationary",
                stamp_sec=image.stamp_sec,
                pose=robot_pose,
            )
            if self.model_pose_tracker is not None:
                self.model_pose_tracker.reset()
            self.last_pose = robot_pose
            self._write_status(
                "robot_not_stationary",
                robot_pose=asdict(robot_pose),
            )
            return
        self.last_pose = robot_pose
        intrinsics = intrinsics_from_camera_info(camera_info.value)
        camera_translation, camera_rotation = _transform_values(camera_from_map)
        head_half_height_m = 0.5 * self.args.stand_face_size_m
        try:
            camera_point = transform_point(
                (
                    self.args.stand_x,
                    self.args.stand_y,
                    self.args.stand_head_center_height_m,
                ),
                translation_xyz=camera_translation,
                rotation_xyzw=camera_rotation,
            )
            camera_top_point = transform_point(
                (
                    self.args.stand_x,
                    self.args.stand_y,
                    self.args.stand_head_center_height_m + head_half_height_m,
                ),
                translation_xyz=camera_translation,
                rotation_xyzw=camera_rotation,
            )
            camera_bottom_point = transform_point(
                (
                    self.args.stand_x,
                    self.args.stand_y,
                    self.args.stand_head_center_height_m - head_half_height_m,
                ),
                translation_xyz=camera_translation,
                rotation_xyzw=camera_rotation,
            )
            parallel_side_direction = project_rectified_image_direction(
                camera_top_point,
                camera_bottom_point,
                intrinsics,
            )
        except ValueError as exc:
            self._note_observation_soft_miss(
                "world_vertical_projection_failed",
                stamp_sec=image.stamp_sec,
                pose=robot_pose,
            )
            self._write_status(
                "world_vertical_projection_failed",
                reason=str(exc),
            )
            return
        projection = project_optical_point(
            camera_point,
            intrinsics,
            physical_size_m=self.args.stand_face_size_m,
        )
        roi = roi_from_projection(
            projection,
            intrinsics,
            padding_scale=self.args.head_roi_padding_scale,
        )
        if roi is None or projection.expected_size_px < self.args.min_head_size_px:
            self._note_observation_soft_miss(
                "target_outside_camera_gate",
                stamp_sec=image.stamp_sec,
                pose=robot_pose,
            )
            self._write_status(
                "target_outside_camera_gate",
                projection=asdict(projection),
            )
            return
        resolved_stand_axis_profile = self.stand_axis_profile.resolve(
            projection.expected_size_px
        )
        scan_translation, scan_rotation = _transform_values(scan_from_map)
        scan_point = transform_point(
            (self.args.stand_x, self.args.stand_y, 0.0),
            translation_xyz=scan_translation,
            rotation_xyzw=scan_rotation,
        )
        scan_bearing = math.atan2(scan_point[1], scan_point[0])
        plain_scan = PlainLaserScan(
            ranges=tuple(float(value) for value in scan_message.ranges),
            angle_min=float(scan_message.angle_min),
            angle_increment=float(scan_message.angle_increment),
            range_min=float(scan_message.range_min),
            range_max=float(scan_message.range_max),
            scan_frame_id=str(scan_message.header.frame_id),
            scan_stamp_sec=scan.stamp_sec,
            receipt_sec=now_sec,
        )
        center_distance = math.hypot(
            robot_pose.x_m - self.args.stand_x,
            robot_pose.y_m - self.args.stand_y,
        )
        lower_surface_bound = (
            center_distance
            - 2.0 * self.args.stand_radius_m
            - self.args.stand_uncertainty_m
            - self.args.lidar_range_tolerance_m
        )
        upper_surface_bound = center_distance + self.args.lidar_range_tolerance_m
        preliminary_lidar_association = associate_candidate_lidar_target(
            plain_scan,
            map_bearing_rad=scan_bearing,
            cone_half_angle_rad=math.radians(
                self.args.lidar_cone_half_angle_deg
            ),
            accepted_range_m=(lower_surface_bound, upper_surface_bound),
            now_sec=now_sec,
            max_scan_age_sec=self.args.max_sensor_age_sec,
            min_cluster_sample_count=self.args.lidar_min_samples,
        )
        scan_camera_translation, scan_camera_rotation = _transform_values(
            scan_from_camera_transform
        )
        scan_from_camera_geometry = RigidTransform(
            parent_frame=self.profile.scan_frame,
            child_frame=self.profile.camera_optical_frame,
            translation_xyz_m=scan_camera_translation,
            rotation_xyzw=scan_camera_rotation,
        )
        try:
            frame = compressed_msg_to_bgr_frame(
                image_message,
                self.cv2,
                self.numpy,
            )
            frame = _rectify_bgr_frame(
                frame,
                camera_info.value,
                self.cv2,
                self.numpy,
            )
        except (TypeError, ValueError) as exc:
            self._note_observation_soft_miss(
                "image_rectification_failed",
                stamp_sec=image.stamp_sec,
                pose=robot_pose,
            )
            self._write_status("image_rectification_failed", reason=str(exc))
            return
        roi_frame = frame[roi.y0 : roi.y1, roi.x0 : roi.x1]
        fallback_estimate, fallback_debug = estimate_stand_axis_from_edges(
            self.cv2,
            roi_frame,
            edge_preprocess=resolved_stand_axis_profile.edge_preprocess,
            canny_low=resolved_stand_axis_profile.canny_low,
            canny_high=resolved_stand_axis_profile.canny_high,
            silhouette_only=True,
            parallel_side_direction=parallel_side_direction,
            min_area_px=resolved_stand_axis_profile.min_area_px,
            min_face_area_fraction=0.0,
            min_edge_height_px=resolved_stand_axis_profile.min_edge_height_px,
            close_kernel=resolved_stand_axis_profile.close_kernel,
            min_aspect_ratio=resolved_stand_axis_profile.min_aspect_ratio,
            max_aspect_ratio=resolved_stand_axis_profile.max_aspect_ratio,
            stand_width_m=self.args.stand_face_size_m,
            stand_distance_m=projection.depth_m,
            camera_fx_px=intrinsics.fx_px,
            camera_fy_px=intrinsics.fy_px,
            camera_cx_px=intrinsics.cx_px - roi.x0,
            camera_cy_px=intrinsics.cy_px - roi.y0,
        )
        estimate, debug = fallback_estimate, fallback_debug
        model_metadata = None
        if self.stand_model_profile is not None:
            camera_signature = (
                intrinsics.fx_px,
                intrinsics.fy_px,
                intrinsics.cx_px - roi.x0,
                intrinsics.cy_px - roi.y0,
            )
            prediction = self.model_pose_tracker.prediction(
                now_sec=image.stamp_sec,
                profile_sha256=self.stand_model_profile.sha256,
                camera_signature=camera_signature,
            )
            model_estimate, model_debug = estimate_stand_axis_from_metric_model(
                self.cv2,
                roi_frame,
                model_profile=self.stand_model_profile,
                camera_fx_px=intrinsics.fx_px,
                camera_fy_px=intrinsics.fy_px,
                camera_cx_px=intrinsics.cx_px - roi.x0,
                camera_cy_px=intrinsics.cy_px - roi.y0,
                pose_hint=prediction.pose,
                edge_preprocess=resolved_stand_axis_profile.edge_preprocess,
                canny_low=resolved_stand_axis_profile.canny_low,
                canny_high=resolved_stand_axis_profile.canny_high,
                min_edge_height_px=resolved_stand_axis_profile.min_edge_height_px,
            )
            if (
                model_debug.model_pose is not None
                and (
                    model_estimate.evidence_state == "fresh_refined"
                    or model_debug.qr_detected
                )
            ):
                self.model_pose_tracker.accept(
                    model_debug.model_pose,
                    now_sec=image.stamp_sec,
                    profile_sha256=self.stand_model_profile.sha256,
                    camera_signature=camera_signature,
                )
            if (
                model_estimate.usable
                or model_estimate.evidence_state
                in ("predicted_only", "ambiguous")
            ):
                estimate, debug = model_estimate, model_debug
            model_metadata = {
                "profile_id": self.stand_model_profile.profile_id,
                "profile_sha256": self.stand_model_profile.sha256,
                "measurement_status": self.stand_model_profile.measurement_status,
                "evidence_state": model_estimate.evidence_state,
                "qr_detected": model_debug.qr_detected,
                "pose_reprojection_rmse_px": (
                    model_estimate.pose_reprojection_rmse_px
                ),
                "pose_ambiguity_gap_px": model_estimate.pose_ambiguity_gap_px,
                "refinement_support_mean": model_debug.refinement_support_mean,
            }
        axis_metadata = {
            "profile": asdict(resolved_stand_axis_profile),
            "parallel_side_direction": list(parallel_side_direction),
            "direction_source": "observer_rectified_tf_camera_info",
            "estimator_usable": estimate.usable,
            "estimator_reason": estimate.reason,
            "estimator_source": estimate.source,
            "metric_model": model_metadata,
            "preliminary_candidate_lidar_association": asdict(
                preliminary_lidar_association
            ),
        }
        qr_texts = tuple(
            sorted(set(detect_qr_texts_bgr(roi_frame, self.cv2)))
        )
        if (
            self.stand_model_profile is not None
            and estimate.model_profile_sha256
            != self.stand_model_profile.sha256
        ):
            self._reset_observation_evidence()
            self._write_debug(
                frame,
                roi_frame,
                debug,
                metadata=axis_metadata,
            )
            self._write_status(
                "metric_model_measurement_unavailable",
                estimator_reason=estimate.reason,
                estimator_source=estimate.source,
                stand_axis_debug=axis_metadata,
            )
            return
        if (
            self.args.expected_qr_id != AUTO_QR_ID
            and qr_texts
            and qr_texts != (self.args.expected_qr_id,)
        ):
            self._reset_observation_evidence()
            self._write_debug(frame, roi_frame, debug, metadata=axis_metadata)
            self._write_status(
                "evidence_not_committable",
                reason="observed QR identity differs from expected identity",
                qr_texts=list(qr_texts),
                expected_qr_id=self.args.expected_qr_id,
                stand_axis_debug=axis_metadata,
            )
            return
        if not estimate.usable or estimate.yaw_deg is None:
            update = self._record_observation_frame(
                robot_pose=robot_pose,
                image_stamp_sec=image.stamp_sec,
                scan_stamp_sec=scan.stamp_sec,
                observed_at_sec=now_sec,
                lidar_associated=preliminary_lidar_association.associated,
                axis_yaw_rad=None,
                axis_source=None,
                qr_texts=qr_texts,
            )
            self._write_debug(
                frame,
                roi_frame,
                debug,
                metadata=axis_metadata,
            )
            self._write_status(
                "silhouette_unavailable",
                estimator_reason=estimate.reason,
                estimator_source=estimate.source,
                qr_texts=list(qr_texts),
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
            )
            return
        scale_gate = _head_scale_gate(
            expected_size_px=projection.expected_size_px,
            left_height_px=estimate.left_height_px,
            right_height_px=estimate.right_height_px,
        )
        axis_metadata["head_scale_gate"] = scale_gate
        if not scale_gate["accepted"]:
            update = self._record_observation_frame(
                robot_pose=robot_pose,
                image_stamp_sec=image.stamp_sec,
                scan_stamp_sec=scan.stamp_sec,
                observed_at_sec=now_sec,
                lidar_associated=preliminary_lidar_association.associated,
                axis_yaw_rad=None,
                axis_source=None,
                qr_texts=qr_texts,
            )
            self._write_debug(
                frame,
                roi_frame,
                debug,
                metadata=axis_metadata,
            )
            self._write_status(
                "head_size_projection_mismatch",
                qr_texts=list(qr_texts),
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
            )
            return
        optical_yaw = math.radians(estimate.yaw_deg)
        conditioning = axis_conditioning(
            optical_yaw,
            max_obliqueness_rad=math.radians(self.args.max_obliqueness_deg),
        )
        if estimate.corners is None:
            update = self._record_observation_frame(
                robot_pose=robot_pose,
                image_stamp_sec=image.stamp_sec,
                scan_stamp_sec=scan.stamp_sec,
                observed_at_sec=now_sec,
                lidar_associated=preliminary_lidar_association.associated,
                axis_yaw_rad=None,
                axis_source=None,
                qr_texts=qr_texts,
            )
            self._write_debug(
                frame,
                roi_frame,
                debug,
                metadata=axis_metadata,
            )
            self._write_status(
                "evidence_not_committable",
                reason="usable axis estimate has no image corners",
                conditioning=asdict(conditioning),
                qr_texts=list(qr_texts),
                expected_qr_id=self.args.expected_qr_id,
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
            )
            return
        detected_center_u_px = (
            sum(point.u_px for point in estimate.corners)
            / len(estimate.corners)
            + roi.x0
        )
        detected_center_v_px = (
            sum(point.v_px for point in estimate.corners)
            / len(estimate.corners)
            + roi.y0
        )
        try:
            observed_camera_bearing = rectified_pixel_bearing_in_scan(
                u_px=detected_center_u_px,
                v_px=detected_center_v_px,
                fx_px=intrinsics.fx_px,
                fy_px=intrinsics.fy_px,
                cx_px=intrinsics.cx_px,
                cy_px=intrinsics.cy_px,
                scan_from_camera=scan_from_camera_geometry,
            )
        except ValueError as exc:
            self._note_observation_soft_miss(
                "camera_lidar_bearing_unavailable",
                stamp_sec=image.stamp_sec,
                pose=robot_pose,
            )
            self._write_debug(frame, roi_frame, debug, metadata=axis_metadata)
            self._write_status(
                "evidence_not_committable",
                reason=str(exc),
                conditioning=asdict(conditioning),
                qr_texts=list(qr_texts),
                expected_qr_id=self.args.expected_qr_id,
                stand_axis_debug=axis_metadata,
            )
            return
        lidar_association = associate_candidate_lidar_target(
            plain_scan,
            map_bearing_rad=scan_bearing,
            cone_half_angle_rad=math.radians(
                self.args.lidar_cone_half_angle_deg
            ),
            accepted_range_m=(lower_surface_bound, upper_surface_bound),
            now_sec=now_sec,
            max_scan_age_sec=self.args.max_sensor_age_sec,
            min_cluster_sample_count=self.args.lidar_min_samples,
            observed_camera_bearing_rad=observed_camera_bearing,
        )
        axis_metadata["candidate_lidar_association"] = asdict(
            lidar_association
        )
        if not lidar_association.associated:
            update = self._record_observation_frame(
                robot_pose=robot_pose,
                image_stamp_sec=image.stamp_sec,
                scan_stamp_sec=scan.stamp_sec,
                observed_at_sec=now_sec,
                lidar_associated=False,
                axis_yaw_rad=None,
                axis_source=None,
                qr_texts=qr_texts,
            )
            self._write_debug(frame, roi_frame, debug, metadata=axis_metadata)
            self._write_status(
                "lidar_target_mismatch",
                center_distance_m=center_distance,
                candidate_lidar_association=asdict(lidar_association),
                accepted_range_m=[lower_surface_bound, upper_surface_bound],
                observation_evidence=update.snapshot.as_dict(),
            )
            return
        if not conditioning.accepted:
            update = self._record_observation_frame(
                robot_pose=robot_pose,
                image_stamp_sec=image.stamp_sec,
                scan_stamp_sec=scan.stamp_sec,
                observed_at_sec=now_sec,
                lidar_associated=True,
                axis_yaw_rad=None,
                axis_source=None,
                qr_texts=qr_texts,
            )
            self._write_debug(frame, roi_frame, debug, metadata=axis_metadata)
            self._write_status(
                "evidence_not_committable",
                conditioning=asdict(conditioning),
                qr_texts=list(qr_texts),
                expected_qr_id=self.args.expected_qr_id,
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
            )
            return
        update = self._record_observation_frame(
            robot_pose=robot_pose,
            image_stamp_sec=image.stamp_sec,
            scan_stamp_sec=scan.stamp_sec,
            observed_at_sec=now_sec,
            lidar_associated=True,
            axis_yaw_rad=optical_yaw,
            axis_source=estimate.source,
            qr_texts=qr_texts,
        )
        consensus = update.axis_consensus
        resolved_qr_id = update.resolved_qr_id
        self._write_debug(
            frame,
            roi_frame,
            debug,
            metadata=axis_metadata,
        )
        if update.snapshot.poisoned:
            self._write_status(
                "evidence_not_committable",
                reason=update.snapshot.poison_reason,
                conditioning=asdict(conditioning),
                qr_texts=list(qr_texts),
                expected_qr_id=self.args.expected_qr_id,
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
            )
            return
        if consensus is None:
            self._write_status(
                "collecting_consensus",
                qr_texts=list(qr_texts),
                estimator_source=estimate.source,
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
            )
            return
        camera_heading = optical_heading_from_transform(map_from_camera)
        stand_axis = stand_axis_from_camera_yaw(
            robot_x_m=robot_pose.x_m,
            robot_y_m=robot_pose.y_m,
            stand_x_m=self.args.stand_x,
            stand_y_m=self.args.stand_y,
            camera_yaw_rad=consensus.yaw_rad,
            camera_heading_rad=camera_heading,
        )
        confidence = max(
            0.0,
            min(
                1.0,
                1.0
                - consensus.max_deviation_rad
                / max(math.radians(self.args.consensus_max_deviation_deg), 1.0e-9),
            ),
        )
        if resolved_qr_id is None:
            if self.args.axis_observation_json is not None:
                _atomic_json(
                    self.args.axis_observation_json,
                    {
                        "schema_version": 1,
                        "observation_kind": "real_stand_axis_without_qr",
                        "motion_capability": "none",
                        "stream_id": self.args.stream_id,
                        "stand_id": self.args.stand_id,
                        "planning_frame": self.profile.map_frame,
                        "stand_center": {
                            "x_m": self.args.stand_x,
                            "y_m": self.args.stand_y,
                        },
                        "robot_pose": {
                            "x_m": robot_pose.x_m,
                            "y_m": robot_pose.y_m,
                            "yaw_rad": robot_pose.yaw_rad,
                        },
                        "stand_axis_rad": stand_axis,
                        "axis_confidence": confidence,
                        "axis_sample_count": consensus.sample_count,
                        "sensor_stamp_sec": image.stamp_sec,
                        "observer_version": OBSERVER_VERSION,
                        "stand_model_profile_sha256": (
                            estimate.model_profile_sha256
                        ),
                        "model_evidence_state": estimate.evidence_state,
                        "pose_reprojection_rmse_px": (
                            estimate.pose_reprojection_rmse_px
                        ),
                        "pose_ambiguity_gap_px": (
                            estimate.pose_ambiguity_gap_px
                        ),
                        "robot_profile_sha256": real_robot_profile_sha256(
                            self.profile
                        ),
                        "calibration_profile_sha256": (
                            camera_calibration_sha256(self.calibration)
                        ),
                    },
                )
                self.axis_observation_committed = True
                self.completed = True
            self._write_status(
                "axis_committed_qr_unresolved",
                axis_observation=(
                    str(self.args.axis_observation_json)
                    if self.args.axis_observation_json is not None
                    else None
                ),
                axis_sample_count=consensus.sample_count,
                axis_confidence=confidence,
                qr_texts=[],
                stand_axis_debug=axis_metadata,
            )
            return
        recommendation = build_real_viewpoint_recommendation(
            stream_id=self.args.stream_id,
            stand_id=self.args.stand_id,
            planning_frame=self.profile.map_frame,
            stand_center=Pose2D(self.args.stand_x, self.args.stand_y),
            stand_radius_m=self.args.stand_radius_m,
            stand_uncertainty_m=self.args.stand_uncertainty_m,
            robot_pose=robot_pose,
            stand_axis_rad=stand_axis,
            axis_confidence=confidence,
            axis_sample_count=consensus.sample_count,
            sensor_stamp_sec=image.stamp_sec,
            expected_qr_id=resolved_qr_id,
            observed_qr_ids=(resolved_qr_id,),
            target_distance_m=self.args.target_distance_m,
        )
        _atomic_json(
            self.args.recommended_pose_json,
            recommendation_to_dict(recommendation),
        )
        self.completed = True
        self._write_status(
            "recommendation_committed",
            recommendation=str(self.args.recommended_pose_json),
            robot_profile_sha256=real_robot_profile_sha256(self.profile),
            calibration_profile_sha256=camera_calibration_sha256(
                self.calibration
            ),
            axis_sample_count=consensus.sample_count,
            axis_confidence=confidence,
            qr_texts=[resolved_qr_id],
            observation_evidence=update.snapshot.as_dict(),
            stand_axis_debug=axis_metadata,
        )
        self.node.get_logger().info(
            f"committed passive recommendation: {self.args.recommended_pose_json}"
        )

    def _write_debug(self, frame, roi_frame, debug, *, metadata) -> None:
        if self.args.debug_dir is None:
            return
        self.args.debug_dir.mkdir(parents=True, exist_ok=True)
        image_artifacts = (
            ("latest_frame.png", frame),
            ("latest_head_roi.png", roi_frame),
            ("latest_edges.png", debug.edges),
            ("latest_raw_edges.png", debug.raw_edges),
            ("latest_side_evidence.png", debug.face_mask),
            ("latest_rectangle_mask.png", debug.rectangle_mask),
            ("latest_rectangle_overlay.png", debug.rectangle_overlay),
        )
        written = []
        for filename, image in image_artifacts:
            artifact_path = self.args.debug_dir / filename
            if image is not None and self.cv2.imwrite(str(artifact_path), image):
                written.append(filename)
            else:
                artifact_path.unlink(missing_ok=True)
        structure = debug.structure_evidence
        _atomic_json(
            self.args.debug_dir / "latest_metadata.json",
            {
                "schema_version": 1,
                "observed_unix_sec": time.time(),
                "artifacts": written,
                "stand_axis": metadata,
                "structure_evidence_reason": (
                    structure.reason if structure is not None else None
                ),
            },
        )

    def _write_status(self, state: str, **details) -> None:
        observation_evidence = getattr(self, "observation_evidence", None)
        if observation_evidence is not None:
            evidence_snapshot = observation_evidence.snapshot()
            consensus_status = {
                "sample_count": evidence_snapshot.current_axis_sample_count,
                "peak_sample_count": evidence_snapshot.peak_axis_sample_count,
                "required_sample_count": (
                    evidence_snapshot.required_axis_sample_count
                ),
                "sample_count_by_source": (
                    evidence_snapshot.current_axis_sample_count_by_source
                ),
                "peak_sample_count_by_source": (
                    evidence_snapshot.peak_axis_sample_count_by_source
                ),
            }
            observation_status = evidence_snapshot.as_dict()
        else:
            legacy_consensus = getattr(self, "consensus", None)
            legacy_sample_count = getattr(legacy_consensus, "sample_count", 0)
            consensus_status = {
                "sample_count": legacy_sample_count,
                "peak_sample_count": legacy_sample_count,
                "required_sample_count": getattr(
                    legacy_consensus,
                    "required_samples",
                    getattr(self.args, "consensus_frames", 0),
                ),
            }
            observation_status = None
        payload = {
            "schema_version": 2,
            "observer_version": OBSERVER_VERSION,
            "state": state,
            "motion_capability": "none",
            "observed_unix_sec": time.time(),
            "stand_axis_profile": asdict(self.stand_axis_profile),
            "axis_consensus": consensus_status,
            "observation_evidence": observation_status,
            "tf_retry": asdict(self.tf_retry_scheduler.evidence),
            "tf_retry_attempt_summary": {
                "attempted_tuple_count": getattr(
                    self, "_tf_retry_tuple_count", 0
                ),
                "exhausted_tuple_count": getattr(
                    self, "_tf_retry_exhausted_tuple_count", 0
                ),
                "peak_retry_count": getattr(
                    self, "_tf_retry_peak_count", 0
                ),
            },
            **details,
        }
        _atomic_json(
            self.args.status_json,
            payload,
        )
        status_events_jsonl = getattr(self.args, "status_events_jsonl", None)
        if status_events_jsonl is not None:
            _append_jsonl(status_events_jsonl, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-profile", required=True, type=Path)
    parser.add_argument("--camera-calibration", required=True, type=Path)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--stand-id", required=True)
    parser.add_argument("--expected-qr-id", required=True)
    parser.add_argument("--stand-x", required=True, type=float)
    parser.add_argument("--stand-y", required=True, type=float)
    parser.add_argument("--stand-radius-m", type=float, default=0.06)
    parser.add_argument("--stand-uncertainty-m", type=float, default=0.02)
    parser.add_argument("--stand-face-size-m", type=float, default=0.078)
    parser.add_argument(
        "--stand-model-profile",
        type=Path,
        default=None,
        help=(
            "Optional content-hashed measured stand model. Provisional models "
            "are rejected by this operational observer."
        ),
    )
    parser.add_argument("--stand-head-center-height-m", type=float, default=0.165)
    parser.add_argument("--target-distance-m", type=float, default=0.33)
    parser.add_argument("--head-roi-padding-scale", type=float, default=1.8)
    parser.add_argument("--min-head-size-px", type=float, default=18.0)
    parser.add_argument("--sync-tolerance-sec", type=float, default=0.10)
    parser.add_argument("--camera-info-tolerance-sec", type=float, default=1.0)
    parser.add_argument("--max-sensor-age-sec", type=float, default=0.5)
    parser.add_argument("--max-future-timestamp-sec", type=float, default=0.05)
    parser.add_argument(
        "--tf-timeout-sec",
        type=float,
        default=0.15,
        help=(
            "Maximum accumulated nonblocking retry time for one exact sensor "
            "tuple; ROS callbacks continue running between polls."
        ),
    )
    parser.add_argument("--tf-retry-rate-hz", type=float, default=50.0)
    parser.add_argument("--process-rate-hz", type=float, default=5.0)
    parser.add_argument("--stationary-translation-m", type=float, default=0.01)
    parser.add_argument("--stationary-rotation-deg", type=float, default=2.0)
    parser.add_argument("--consensus-frames", type=int, default=7)
    parser.add_argument("--consensus-max-deviation-deg", type=float, default=8.0)
    parser.add_argument(
        "--consensus-axis-ttl-sec",
        type=float,
        default=5.0,
        help=(
            "Retain same-target, same-motion-epoch axis samples across brief "
            "perception misses; expired samples never count."
        ),
    )
    parser.add_argument(
        "--consensus-qr-ttl-sec",
        type=float,
        default=5.0,
        help="Bound the independent two-frame QR identity latch.",
    )
    parser.add_argument("--max-obliqueness-deg", type=float, default=30.0)
    parser.add_argument("--lidar-cone-half-angle-deg", type=float, default=3.0)
    parser.add_argument("--lidar-min-samples", type=int, default=1)
    parser.add_argument("--lidar-range-tolerance-m", type=float, default=0.04)
    parser.add_argument("--extrinsic-translation-tolerance-m", type=float, default=0.005)
    parser.add_argument("--extrinsic-rotation-tolerance-deg", type=float, default=1.0)
    parser.add_argument(
        "--edge-preprocess",
        choices=("gray", "channel-union"),
        default="channel-union",
    )
    parser.add_argument("--canny-low", type=int, default=20)
    parser.add_argument("--canny-high", type=int, default=60)
    parser.add_argument("--status-json", required=True, type=Path)
    parser.add_argument("--status-events-jsonl", type=Path, default=None)
    parser.add_argument("--recommended-pose-json", required=True, type=Path)
    parser.add_argument("--axis-observation-json", type=Path, default=None)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--once", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    positive = {
        "--stand-radius-m": args.stand_radius_m,
        "--stand-face-size-m": args.stand_face_size_m,
        "--stand-head-center-height-m": args.stand_head_center_height_m,
        "--target-distance-m": args.target_distance_m,
        "--head-roi-padding-scale": args.head_roi_padding_scale,
        "--min-head-size-px": args.min_head_size_px,
        "--sync-tolerance-sec": args.sync_tolerance_sec,
        "--camera-info-tolerance-sec": args.camera_info_tolerance_sec,
        "--max-sensor-age-sec": args.max_sensor_age_sec,
        "--tf-timeout-sec": args.tf_timeout_sec,
        "--tf-retry-rate-hz": args.tf_retry_rate_hz,
        "--process-rate-hz": args.process_rate_hz,
        "--consensus-max-deviation-deg": args.consensus_max_deviation_deg,
        "--consensus-axis-ttl-sec": args.consensus_axis_ttl_sec,
        "--consensus-qr-ttl-sec": args.consensus_qr_ttl_sec,
        "--max-obliqueness-deg": args.max_obliqueness_deg,
        "--lidar-cone-half-angle-deg": args.lidar_cone_half_angle_deg,
        "--lidar-range-tolerance-m": args.lidar_range_tolerance_m,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    if args.stand_uncertainty_m < 0.0:
        parser.error("--stand-uncertainty-m must be non-negative")
    if args.consensus_frames < 2:
        parser.error("--consensus-frames must be at least two")
    nominal_consensus_span_sec = (
        args.consensus_frames - 1
    ) / args.process_rate_hz
    if args.consensus_axis_ttl_sec < nominal_consensus_span_sec:
        parser.error(
            "--consensus-axis-ttl-sec must cover at least the nominal "
            "consensus collection span"
        )
    if args.lidar_min_samples < 1:
        parser.error("--lidar-min-samples must be positive")
    if not 0 <= args.canny_low < args.canny_high <= 255:
        parser.error("Canny thresholds must satisfy 0 <= low < high <= 255")
    try:
        _stand_axis_profile_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.stand_model_profile is not None:
        try:
            stand_model = load_stand_model(args.stand_model_profile)
        except ValueError as exc:
            parser.error(f"invalid stand model profile: {exc}")
        if not stand_model.committable:
            parser.error(
                "--stand-model-profile must have measurement_status=measured"
            )
        if (
            abs(args.stand_face_size_m - stand_model.head_width_m)
            > max(stand_model.tolerance_m, 1.0e-6)
        ):
            parser.error(
                "--stand-face-size-m disagrees with the stand model profile"
            )
    if args.status_json.resolve() == args.recommended_pose_json.resolve():
        parser.error("status and recommendation outputs must be distinct")
    output_paths = [args.status_json.resolve(), args.recommended_pose_json.resolve()]
    if args.status_events_jsonl is not None:
        output_paths.append(args.status_events_jsonl.resolve())
    if args.axis_observation_json is not None:
        output_paths.append(args.axis_observation_json.resolve())
    if len(set(output_paths)) != len(output_paths):
        parser.error(
            "status, status events, recommendation, and axis outputs must "
            "be distinct"
        )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        import rclpy
    except ImportError as exc:
        parser.exit(2, f"error: ROS 2 Python packages are required: {exc}\n")
    rclpy.init(args=None)
    adapter = None
    try:
        adapter = PassiveRealViewpointNode(args)
        while rclpy.ok() and not (args.once and adapter.completed):
            rclpy.spin_once(adapter.node, timeout_sec=0.1)
        return 0 if adapter.completed or not args.once else 2
    except KeyboardInterrupt:
        return (
            0
            if adapter is not None
            and (adapter.completed or adapter.axis_observation_committed)
            else 130
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    finally:
        if adapter is not None:
            adapter.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
