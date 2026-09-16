#!/usr/bin/env python3
"""Passive real-robot camera/LiDAR viewpoint observer.

This node never creates a publisher and never commands motion.  It requires a
sealed real-robot profile, measured camera calibration, a measured physical
stand model, live ``CameraInfo``, compressed onboard images, a synchronized
LaserScan, and exact-time TF.  Current-frame model, QR, and LiDAR consensus can
write a QR-bound recommendation. Repeated model-backed backside-candidate
evidence can write only the separate, motion-neutral axis receipt.
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
    MAX_CAMERA_MAP_BEARING_DELTA_DEG,
    associate_camera_registered_candidate_lidar_target,
    associate_candidate_lidar_target,
    normalize_certified_camera_map_bearing_limit,
)
from scripts.aufgabe04.perception.stand_axis_consensus import axis_conditioning
from scripts.aufgabe04.perception.stand_axis.real_camera_profile import (
    RealCameraStandAxisProfile,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import (
    load_measured_physical_stand_model,
    resolve_head_center_height_m,
)
from scripts.aufgabe04.perception.stand_axis.model_pipeline import (
    estimate_stand_axis_from_metric_model,
)
from scripts.aufgabe04.perception.stand_axis.model_input_cache import MetricModelInputCache
from scripts.aufgabe04.perception.stand_axis.current_image_head_fit import CurrentImageHeadFit
from scripts.aufgabe04.perception.stand_axis.pose_tracking import (
    MetricPoseTracker,
)
from scripts.aufgabe04.perception.stand_axis.observation_freshness import (
    observation_freshness,
)
from scripts.aufgabe04.perception.stand_axis.model_diagnostics import (
    metric_fit_diagnostics_payload,
)
from scripts.aufgabe04.perception.stand_axis_handoff import (
    RigidTransform,
    rectified_pixel_bearing_in_scan,
)
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import (
    detect_qr_observations_bgr,
    detect_qr_texts_bgr,
)
from scripts.aufgabe04.qr_scanning.native_qr_observations import (
    detect_native_qr_observations_bgr,
)
from scripts.aufgabe04.real_robot.configuration.geometry import (
    intrinsics_from_camera_info,
    optical_heading_from_transform,
    pose2d_from_transform,
    project_optical_point,
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
    BACKSIDE_AXIS_SAMPLE_SOURCE,
    PASSIVE_VIEWPOINT_OBSERVER_VERSION,
    REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE,
)
from scripts.aufgabe04.real_robot.observer.backside_axis_observation import (
    build_backside_axis_observation,
)
from scripts.aufgabe04.real_robot.observer.axis_sample_policy import (
    DEFAULT_QR_BOUND_MODEL_MAX_OBLIQUENESS_DEG,
    admit_axis_sample,
    normalize_qr_bound_model_obliqueness_limit,
)
from scripts.aufgabe04.real_robot.observer.tf_retry import (
    PassiveObserverTfRetryScheduler,
)
from scripts.aufgabe04.real_robot.observer.ingestion_runtime import (
    BoundedSensorIngress, ObserverIngestionLoop, ObserverWorkSchedule,
)
from scripts.aufgabe04.real_robot.observer.tf_delivery_trace import (
    ObserverTfDeliveryTrace, create_observer_traced_buffer, traced_observer_lookup,
)
from scripts.aufgabe04.real_robot.observer.evidence import (
    AxisWindowReview,
    EvidencePose,
    PassiveObserverEvidence,
)
from scripts.aufgabe04.real_robot.observer.camera_target_registration import (
    HeadRoiEvaluation,
)
from scripts.aufgabe04.real_robot.observer.candidate_head_tracking import (
    CandidateHeadContext, CandidateHeadTracking,
)
from scripts.aufgabe04.real_robot.observer.tracked_head_registration import (
    tracked_head_selection, register_current_tracked_head,
)
from scripts.aufgabe04.real_robot.observer.backside_proposal_reuse import (
    BacksideProposalContext,
    BacksideProposalReuse,
)
from scripts.aufgabe04.real_robot.observer.backside_head_crop import (
    gate_backside_head_crop, review_backside_head_crop, review_current_head_crop,
)
from scripts.aufgabe04.real_robot.observer.head_observation_confidence import (
    HeadConfidenceInput, HeadObservationConfidence,
)
from scripts.aufgabe04.real_robot.observer.head_observation_window import (
    MEASURED_HEAD_SOURCES, current_head_window_input, review_current_head_window,
)
from scripts.aufgabe04.real_robot.observer.head_temporal_consistency import StationaryHeadConsistency
from scripts.aufgabe04.perception.stand_axis.head_orientation_bounds import validated_current_head_orientation_bounds
from scripts.aufgabe04.real_robot.observer.bounded_head_observation import (
    prepare_bounded_head, record_bounded_head, commit_bounded_head,
)
from scripts.aufgabe04.artifacts.backside_axis_observation import MINIMUM_BACKSIDE_AXIS_CONFIDENCE
from scripts.aufgabe04.real_robot.observer.camera_publication import (
    CameraPublicationExpired,
    camera_source_freshness,
)
from scripts.aufgabe04.real_robot.observer.qr_decode_cache import RoiQrDecodeCache
from scripts.aufgabe04.real_robot.observer.qr_acquisition_policy import (
    QrAcquisitionPolicy, evaluate_roi_with_qr_acquisition,
)
from scripts.aufgabe04.real_robot.observer.roi_qr_evidence import summarize_roi_qr_evidence
from scripts.aufgabe04.real_robot.observer.head_proposal_registration import (
    acquire_registered_head_measurement, unresolved_front_framing_hint,
)
from scripts.aufgabe04.real_robot.observer.head_acquisition_schedule import (
    HeadProcessingDeadline, select_cold_candidate_head, unavailable_head_evaluation,
)
from scripts.aufgabe04.real_robot.observer.capture_history import (
    BoundedObserverCapture,
    sensor_capture_metadata,
)
from scripts.aufgabe04.real_robot.observer.qr_target_binding import (
    bind_qr_observations_to_target,
)
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    DEFAULT_BACKSIDE_REACQUISITION_PADDING_SCALE,
    DEFAULT_BACKSIDE_REGISTRATION_MAX_CENTER_OFFSET_RATIO,
    HeadRoiAttempt,
    MAX_BACKSIDE_REACQUISITION_PADDING_SCALE,
    MAX_BACKSIDE_REGISTRATION_CENTER_OFFSET_RATIO,
    is_camera_registered_head_roi_attempt,
    target_centered_head_roi_attempts,
)
from scripts.aufgabe04.real_robot.observer.registration_evidence import (
    build_backside_target_registration_evidence,
)
from scripts.aufgabe04.artifacts.candidate_inspection_observation import (
    build_candidate_inspection_observation,
)
from scripts.aufgabe04.real_robot.observer.inspection_progress import (
    INSPECTION_PROGRESS_WINDOW_SEC,
    InspectionProgress,
    classify_inspection_progress,
)
from scripts.aufgabe04.real_robot.observer.front_view_recovery import (
    FrontViewRecovery, front_view_failure_kind,
)
from scripts.aufgabe04.real_robot.observer.head_model_admission import (
    MEASURED_HEAD_AXIS_SOURCE, measured_head_front_is_current, requires_measured_head_admission,
    head_scale_gate as _head_scale_gate,
)
from scripts.aufgabe04.real_robot.observer.current_head_association import (
    associate_current_measured_head,
)
from scripts.aufgabe04.real_robot.observer.current_head_qr_binding import (
    bind_qr_to_current_head,
)
from scripts.aufgabe04.real_robot.observer.scan_target_geometry import (
    scan_target_geometry,
)
from scripts.aufgabe04.real_robot.observer.scan_witness_collection import (
    collect_pending_scan_witnesses, plain_scan_from_sample,
)
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import (
    ScanPersistenceContext, StoppedScanTargetPersistence, scan_pose_in_map,
    scan_pose_from_camera_extrinsics,
)
from scripts.aufgabe04.real_robot.observer.front_observation import (
    BACKSIDE_AXIS_SOURCES,
    front_observation_decision,
)
from scripts.aufgabe04.real_robot.readiness.sensor_timing_contract import (
    DEFAULT_MAX_CAMERA_INFO_IMAGE_SKEW_SEC,
    DEFAULT_MAX_FUTURE_TIMESTAMP_SEC,
    DEFAULT_MAX_IMAGE_SCAN_SKEW_SEC,
    DEFAULT_MAX_SENSOR_AGE_SEC,
)
from scripts.aufgabe04.real_robot.configuration.recommendation import (
    build_real_viewpoint_recommendation,
)


OBSERVER_VERSION = PASSIVE_VIEWPOINT_OBSERVER_VERSION
AUTO_QR_ID = "auto"


def _consensus_for_current_axis_source(update, axis_sample_source: str):
    """Return only a consensus authenticated by the current accepted frame."""

    consensus = update.axis_consensus
    if (
        consensus is None
        or not update.axis_sample_accepted
        or consensus.source != axis_sample_source
    ):
        return None
    return consensus


@dataclass(frozen=True)
class _StampedMessage:
    stamp_sec: float
    value: object
    received_ros_sec: float | None = None
    received_monotonic_sec: float | None = None


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


def _atomic_json(path: Path, payload: dict[str, object], *, before_commit=None) -> None:
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
        if before_commit is not None:
            before_commit()
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
        self.stand_model_profile = load_measured_physical_stand_model(
            args.stand_model_profile
        )
        self.stand_head_center_height_m = resolve_head_center_height_m(
            self.stand_model_profile,
            args.stand_head_center_height_m,
        )
        self.model_pose_tracker = MetricPoseTracker(prediction_ttl_sec=0.25)
        self.backside_proposal_reuse = BacksideProposalReuse(
            max_translation_m=args.stationary_translation_m,
            max_rotation_rad=math.radians(args.stationary_rotation_deg),
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
        self._sensor_ingress = BoundedSensorIngress()
        self._work_schedule = ObserverWorkSchedule(
            process_rate_hz=args.process_rate_hz,
            tf_retry_rate_hz=args.tf_retry_rate_hz)
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
        self._tf_delivery_trace = ObserverTfDeliveryTrace(
            ros_now=lambda: self.node.get_clock().now().nanoseconds / 1e9,
        )
        self.tf_buffer = create_observer_traced_buffer(Buffer, trace=self._tf_delivery_trace)
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.completed = False
        self.axis_observation_committed = False
        self._camera_pipeline_counters = {}
        self._last_camera_publication_freshness = None
        self.capture_history = None
        self._capture_pending = None
        self._capture_error = None
        if getattr(args, "capture_history_dir", None) is not None:
            try:
                self.capture_history = BoundedObserverCapture(
                    args.capture_history_dir, max_frames=args.capture_max_frames,
                    max_bytes=args.capture_max_bytes,
                )
            except Exception as exc:
                self._capture_error = f"{type(exc).__name__}: {exc}"[:256]
        # A visible QR marker proves that the stationary viewpoint is not a
        # geometric backside candidate even when OpenCV cannot decode it.
        # Keep that fact latched until the robot starts a new motion epoch.
        self._qr_marker_seen_in_stationary_epoch = False
        self._qr_marker_stationary_epoch_anchor: Pose2D | None = None
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
        # The ROS executor only ingests sensor/TF receipts. All detector,
        # evidence and scan-persistence work stays on the main owner thread.
        self._write_status(
            "waiting_for_sensors",
            resolved_runtime=self.runtime.as_log_dict(),
        )

    def _on_image(self, message) -> None:
        stamp = compressed_msg_stamp_sec(message)
        sample = (self._received_message(stamp, message)
                  if stamp is not None and stamp > 0.0 else None)
        self._sensor_ingress.offer("images", sample)

    def _on_camera_info(self, message) -> None:
        try:
            sample = self._received_message(_stamp_sec(message), message)
        except ValueError:
            sample = None
        self._sensor_ingress.offer("camera_infos", sample)

    def _on_scan(self, message) -> None:
        try:
            sample = self._received_message(_stamp_sec(message), message)
        except ValueError:
            sample = None
        self._sensor_ingress.offer("scans", sample)

    def _drain_received_sensors(self) -> None:
        """Move immutable callback receipts into this owner's sensor histories."""
        batch = self._sensor_ingress.drain()
        self.images.extend(batch.images)
        self.camera_infos.extend(batch.camera_infos)
        self.scans.extend(batch.scans)
        if getattr(self, "_pending_scan_witnesses", None) is None:
            self._pending_scan_witnesses = deque(maxlen=20)
        if len(self._pending_scan_witnesses) + len(batch.scans) > 20 or (
                batch.counts.get("ingress_overwritten_scans", 0)):
            # Missing intervening scans cannot preserve consecutive witnesses.
            persistence = getattr(self, "_scan_target_persistence", None)
            if persistence is not None:
                persistence.reset()
            self._camera_count("scan_witness_ingress_gap")
        self._pending_scan_witnesses.extend(batch.scans)
        for name, count in batch.counts.items():
            self._camera_pipeline_counters[name] = (
                self._camera_pipeline_counters.get(name, 0) + count)

    def process_pending_work(self) -> None:
        self._work_schedule.run_due(
            drain=self._drain_received_sensors,
            collect_witnesses=self._collect_scan_witnesses,
            process=self._process_latest,
            retry=self._retry_pending_exact_tf)

    def _reset_scan_witnesses(self):
        self._pending_scan_witnesses = deque(maxlen=20)
        persistence = getattr(self, "_scan_target_persistence", None)
        if persistence is not None:
            persistence.reset()

    def _lookup_scan_witness(self, target, source, stamp=None):
        # Independent exact-time witness reads must not replace the selected
        # camera tuple's TF diagnostics or evict its captured transforms.
        request = {"target_frame": target, "source_frame": source,
                   "query_kind": ("exact_scan_witness_time" if stamp is not None
                                  else "time_invariant_camera_extrinsic"),
                   "query_stamp_sec": (None if stamp is None else
                                       float(stamp.sec) + float(stamp.nanosec) / 1e9)}
        return traced_observer_lookup(
            getattr(self, "_tf_delivery_trace", None), request,
            lambda: self.tf_buffer.lookup_transform(
                target, source, self.Time() if stamp is None else self.Time.from_msg(stamp),
                timeout=self.Duration(seconds=0.0)))

    def _collect_scan_witnesses(self):
        pending = getattr(self, "_pending_scan_witnesses", None)
        if not pending:
            return
        if getattr(self, "_scan_target_persistence", None) is None:
            self._scan_target_persistence = StoppedScanTargetPersistence()
        valid = collect_pending_scan_witnesses(
            pending, self._scan_target_persistence,
            now_sec=lambda: self.node.get_clock().now().nanoseconds / 1e9,
            lookup=self._lookup_scan_witness, args=self.args, profile=self.profile,
            calibration=self.calibration, target_key=self._target_evidence_key(),
            epoch_key=str(0 if self.observation_evidence is None else
                          self.observation_evidence.snapshot().motion_epoch),
            transform_error=self.TransformException, count=self._camera_count)
        if not valid:
            self._reset_scan_witnesses()

    def _received_message(self, stamp, message):
        return _StampedMessage(
            stamp, message, self.node.get_clock().now().nanoseconds / 1e9,
            time.monotonic(),
        )

    def _camera_count(self, name):
        counters = getattr(self, "_camera_pipeline_counters", None)
        if counters is None:
            counters = self._camera_pipeline_counters = {}
        counters[name] = counters.get(name, 0) + 1

    def _stage_camera_capture(self, image, scan, camera_info):
        if getattr(self, "capture_history", None) is None:
            return
        self._capture_pending = {
            "image": image, "scan": scan, "camera_info": camera_info,
            "tf_samples": [], "selected_monotonic_sec": time.monotonic(),
        }

    def _capture_tf_sample(self, transform):
        pending = getattr(self, "_capture_pending", None)
        if pending is None:
            return
        try:
            translation, rotation = _transform_values(transform)
            stamp = transform.header.stamp
            returned_stamp = float(stamp.sec) + float(stamp.nanosec) / 1e9
            if not math.isfinite(returned_stamp) or returned_stamp < 0:
                raise ValueError("invalid captured TF timestamp")
            pending["tf_samples"].append({
                **self._active_tf_request,
                "returned_stamp_sec": returned_stamp,
                "returned_target_frame": str(transform.header.frame_id),
                "returned_source_frame": str(transform.child_frame_id),
                "translation_xyz_m": translation, "rotation_xyzw": rotation,
            })
            del pending["tf_samples"][:-16]
        except Exception as exc:
            pending["tf_metadata_error"] = f"{type(exc).__name__}: {exc}"[:256]

    def _capture_camera_outcome(self, state, details):
        capture = getattr(self, "capture_history", None)
        pending = getattr(self, "_capture_pending", None)
        if capture is None or pending is None or state == "tf_pending_exact_time":
            return
        self._capture_pending = None
        try:
            image, scan, info = (pending[name] for name in ("image", "scan", "camera_info"))
            metadata = {
                "observer_state": state, "outcome": details,
                "image_stamp_sec": image.stamp_sec,
                "image_received_ros_sec": image.received_ros_sec,
                "image_received_monotonic_sec": image.received_monotonic_sec,
                "scan_stamp_sec": None if scan is None else scan.stamp_sec,
                "scan_received_ros_sec": None if scan is None else scan.received_ros_sec,
                "selected_monotonic_sec": pending["selected_monotonic_sec"],
                "outcome_monotonic_sec": time.monotonic(),
                "outcome_ros_sec": self.node.get_clock().now().nanoseconds / 1e9,
                "tf_samples": pending["tf_samples"],
                "tf_sample_history_limit": 16,
                "tf_metadata_error": pending.get("tf_metadata_error"),
                "detector_metadata": pending.get("detector_metadata"),
                "robot_profile_sha256": real_robot_profile_sha256(self.profile),
                "calibration_profile_sha256": camera_calibration_sha256(self.calibration),
                "stand_model_profile_sha256": self.stand_model_profile.sha256,
                "publication_freshness": getattr(self, "_last_camera_publication_freshness", None),
            }
            try:
                metadata["sensors"] = sensor_capture_metadata(
                    image.value, None if info is None else info.value,
                    None if scan is None else scan.value,
                )
            except Exception as exc:
                # Unpaired/malformed sensor frames still retain their raw pixels.
                metadata["sensor_metadata_error"] = f"{type(exc).__name__}: {exc}"[:256]
            capture.submit(bytes(image.value.data), metadata=metadata,
                           compressed_format=str(getattr(image.value, "format", "")))
        except Exception as exc:
            self._capture_error = f"{type(exc).__name__}: {exc}"[:256]
            self._camera_count("capture_metadata_failures")

    def _capture_snapshot(self):
        capture = getattr(self, "capture_history", None)
        if capture is None:
            return None
        try:
            return capture.snapshot()
        except Exception as exc:
            return {"diagnostic_only": True, "snapshot_error": f"{type(exc).__name__}: {exc}"[:256]}

    def _source_freshness(self, image_stamp_sec, scan_stamp_sec):
        return camera_source_freshness(
            image_stamp_sec=image_stamp_sec, scan_stamp_sec=scan_stamp_sec,
            now_sec=self.node.get_clock().now().nanoseconds / 1e9,
            max_age_sec=self.args.max_sensor_age_sec,
            max_future_sec=self.args.max_future_timestamp_sec,
        )

    def _commit_sensor_artifact(self, path, payload, *, image_stamp_sec,
                                scan_stamp_sec, artifact_kind):
        """Check source ages after debug/association and again after durable serialization."""

        def check():
            freshness = self._source_freshness(image_stamp_sec, scan_stamp_sec)
            self._last_camera_publication_freshness = {
                "artifact_kind": artifact_kind, **freshness.metadata(),
            }
            if not freshness.accepted:
                raise CameraPublicationExpired()

        try:
            check()
            _atomic_json(path, payload, before_commit=check)
        except CameraPublicationExpired:
            self._camera_count("publication_rejections")
            return False
        self._camera_count("committed_artifacts")
        return True

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
        self._head_qr_tracking_stamp_sec = None
        if evidence_pose is None or self.observation_evidence is None:
            return
        self._last_observation_update = self.observation_evidence.note_soft_miss(
            target_key=self._target_evidence_key(),
            pose=self._evidence_pose(evidence_pose),
            stamp_sec=max(0.0, float(stamp_sec)),
            reason=reason,
        )

    def _reset_observation_evidence(self, *, reset_inspection: bool = True) -> None:
        """Drop evidence after a sealed sensor/frame contract violation."""

        self.observation_evidence = None
        self._head_observation_confidence = None
        self._head_confidence_metadata = None
        self._pending_head_confidence = None
        self._pending_head_window = None
        self._pending_head_window_associated = False
        self._head_window_consistency = None
        self._head_window_decision = None
        self._pending_bounded_head = None
        self._bounded_head_window = None
        self._bounded_head_ready = None
        self._scan_target_persistence = None
        self._reset_scan_witnesses()
        self._reset_candidate_search("observation_evidence_reset")
        self._camera_framing = None
        self._front_view_recovery = None
        self._head_qr_tracking_stamp_sec = None
        if getattr(self, "backside_proposal_reuse", None) is not None:
            self.backside_proposal_reuse.reset()
        self._last_observation_update = None
        self._inspection_frame = None
        if reset_inspection:
            self._inspection_progress = None

    def _candidate_search(self):
        if getattr(self, "_candidate_head_tracking", None) is None:
            self._candidate_head_tracking = CandidateHeadTracking(
                max_translation_m=self.args.stationary_translation_m,
                max_rotation_rad=math.radians(self.args.stationary_rotation_deg))
        return self._candidate_head_tracking

    def _reset_candidate_search(self, reason):
        tracking = getattr(self, "_candidate_head_tracking", None)
        if tracking is not None:
            tracking.reset(reason)

    def _reset_qr_marker_epoch(self) -> None:
        """Forget marker presence only after leaving its stationary epoch."""

        self._qr_marker_seen_in_stationary_epoch = False
        self._qr_marker_stationary_epoch_anchor = None
        self._camera_framing = None
        self._front_view_recovery = None
        self._head_qr_tracking_stamp_sec = None

    def _note_front_observation(self, decision, robot_pose: Pose2D) -> None:
        """Veto QR-free axes without resetting target identity evidence."""

        if decision.marker_observed_now:
            self._qr_marker_seen_in_stationary_epoch = True
            if getattr(self, "backside_proposal_reuse", None) is not None:
                self.backside_proposal_reuse.reset()
            if self._qr_marker_stationary_epoch_anchor is None:
                self._qr_marker_stationary_epoch_anchor = robot_pose
        if decision.marker_seen_in_stationary_epoch and self.observation_evidence is not None:
            self.observation_evidence.discard_axis_samples(
                sources=BACKSIDE_AXIS_SOURCES,
            )

    def _record_front_seen_axis_unresolved(
        self, *, decision, robot_pose, image_stamp_sec, scan_stamp_sec,
        observed_at_sec, lidar_associated, qr_texts,
    ):
        """Record the ordinary gated QR channel with no contradicted axis."""

        if not decision.withhold_backside_axis:
            raise ValueError("front unresolved recording requires a withheld axis")
        update = self._record_observation_frame(
            robot_pose=robot_pose,
            image_stamp_sec=image_stamp_sec,
            scan_stamp_sec=scan_stamp_sec,
            observed_at_sec=observed_at_sec,
            lidar_associated=lidar_associated,
            axis_yaw_rad=None,
            axis_source=None,
            qr_texts=qr_texts,
        )
        return update, decision.metadata(
            lidar_associated=lidar_associated,
            frame_accepted=update.frame_accepted,
        )

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
        qr_symbol_count: int | None = None,
    ):
        evidence = self._ensure_observation_evidence(robot_pose)
        # Association and diagnostic work may outlive detector freshness.
        # Admission uses a new clock sample, never the earlier detector time.
        source_freshness = self._source_freshness(image_stamp_sec, scan_stamp_sec)
        self._last_evidence_source_freshness = source_freshness.metadata()
        observed_at_sec = source_freshness.checked_at_sec
        window = getattr(self, "_pending_head_window", None)
        confidence = getattr(self, "_pending_head_confidence", None)
        review_axis = None
        self._head_window_decision = None
        if (window is not None and window.frame_stamp_sec == image_stamp_sec
                and (axis_source in MEASURED_HEAD_SOURCES
                     or getattr(self, "_pending_head_window_associated", False))):
            if getattr(self, "_head_window_consistency", None) is None:
                self._head_window_consistency = StationaryHeadConsistency(
                    required_samples=evidence.required_axis_samples,
                    max_axis_span_rad=evidence.max_axis_deviation_rad,
                    window_ttl_sec=evidence.axis_ttl_sec)

            def review_axis(snapshot):
                review, decision = review_current_head_window(
                    self._head_window_consistency, window, snapshot=snapshot)
                self._head_window_decision = decision
                if confidence is not None:
                    confidence[1]["head_temporal_consistency"] = decision.metadata()
                return review

        elif axis_source in MEASURED_HEAD_SOURCES:
            def review_axis(snapshot):
                return AxisWindowReview(False, MEASURED_HEAD_SOURCES,
                                        "current_head_window_input_unavailable")

        update = evidence.record_frame(
            target_key=self._target_evidence_key(),
            pose=self._evidence_pose(robot_pose),
            frame_stamp_sec=image_stamp_sec,
            lidar_stamp_sec=scan_stamp_sec,
            observed_at_sec=observed_at_sec,
            lidar_associated=lidar_associated and source_freshness.accepted,
            axis_yaw_rad=axis_yaw_rad,
            axis_source=axis_source,
            qr_texts=qr_texts,
            qr_symbol_count=qr_symbol_count,
            expected_qr_id=(
                None if getattr(self.args, "expected_qr_id", AUTO_QR_ID) == AUTO_QR_ID
                else self.args.expected_qr_id
            ),
            axis_window_review=review_axis,
        )
        pending = getattr(self, "_pending_head_confidence", None)
        if pending is not None and pending[0].frame_stamp_sec == image_stamp_sec:
            if getattr(self, "_head_observation_confidence", None) is None:
                self._head_observation_confidence = HeadObservationConfidence(
                    required_samples=evidence.required_axis_samples,
                    ttl_sec=evidence.axis_ttl_sec)
            self._head_confidence_metadata = self._head_observation_confidence.observe(
                pending[0], update=update, observed_at_sec=observed_at_sec,
                angle_temporally_consistent=(self._head_window_decision is not None
                                            and self._head_window_decision.current_sample_accepted))
            pending[1]["observation_confidence"] = self._head_confidence_metadata
        record_bounded_head(self, update=update, image_stamp_sec=image_stamp_sec,
                            observed_at_sec=observed_at_sec)
        self._last_observation_update = update
        self._head_qr_tracking_stamp_sec = (
            image_stamp_sec
            if (update.frame_accepted and update.qr_sample_accepted and update.resolved_qr_id
                and not update.snapshot.poisoned and not update.motion_epoch_reset)
            else None
        )
        if update.snapshot.poisoned or update.motion_epoch_reset:
            self._reset_candidate_search("observation_epoch_reset_or_poisoned")
            self._reset_scan_witnesses()
            self._camera_framing = None
            recovery = getattr(self, "_front_view_recovery", None)
            if recovery is not None:
                if update.motion_epoch_reset:
                    recovery.reset()
                if update.snapshot.poisoned:
                    recovery.poison()
        if update.frame_accepted:
            self._camera_count("associated_frames")
        if update.axis_sample_accepted:
            self._camera_count("axis_sample_frames")
        if update.qr_sample_accepted:
            self._camera_count("qr_sample_frames")
        self._inspection_frame = {
            "frame_stamp_sec": image_stamp_sec,
            "scan_stamp_sec": scan_stamp_sec,
            "robot_pose": asdict(robot_pose),
            "frame_accepted": update.frame_accepted,
            "poisoned": update.snapshot.poisoned,
            "current_qr_id": update.resolved_qr_id,
            "current_qr_sample_count": update.snapshot.current_qr_sample_count,
            "motion_epoch_reset": update.motion_epoch_reset,
            "axis_sample_accepted": update.axis_sample_accepted,
            "qr_sample_accepted": update.qr_sample_accepted,
        }
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
        transform = traced_observer_lookup(
            getattr(self, "_tf_delivery_trace", None), self._active_tf_request,
            lambda: self.tf_buffer.lookup_transform(
                target_frame, source_frame, query_time,
                timeout=self.Duration(seconds=0.0),
            ),
        )
        self._capture_tf_sample(transform)
        return transform

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
        transform = traced_observer_lookup(
            getattr(self, "_tf_delivery_trace", None), self._active_tf_request,
            lambda: self.tf_buffer.lookup_transform(
                target_frame, source_frame, self.Time(),
                timeout=self.Duration(seconds=0.0),
            ),
        )
        self._capture_tf_sample(transform)
        return transform

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
        self._stage_camera_capture(image, scan, camera_info)
        if scan is None or camera_info is None:
            self._camera_count("unpaired_images")
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
        self._camera_count("synchronized_tuples")
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
        self._camera_count("tf_ready_tuples")

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
        self._pending_bounded_head = None
        self._bounded_head_ready = None
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
            self._camera_count("stale_input_images")
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
        marker_anchor = self._qr_marker_stationary_epoch_anchor
        if marker_anchor is not None and not _pose_is_stationary(
            marker_anchor,
            robot_pose,
            max_translation_m=self.args.stationary_translation_m,
            max_rotation_rad=math.radians(self.args.stationary_rotation_deg),
        ):
            # Compare against the original marker pose as well as the previous
            # sample, so several tiny pose changes cannot carry a front-marker
            # latch indefinitely into a genuinely new viewpoint.
            self._reset_qr_marker_epoch()
        if not _pose_is_stationary(
            self.last_pose,
            robot_pose,
            max_translation_m=self.args.stationary_translation_m,
            max_rotation_rad=math.radians(self.args.stationary_rotation_deg),
        ):
            self._reset_qr_marker_epoch()
            self._note_observation_soft_miss(
                "robot_not_stationary",
                stamp_sec=image.stamp_sec,
                pose=robot_pose,
            )
            self.model_pose_tracker.reset()
            self.backside_proposal_reuse.reset()
            self._reset_candidate_search("robot_not_stationary")
            self._reset_scan_witnesses()
            self.last_pose = robot_pose
            self._write_status(
                "robot_not_stationary",
                robot_pose=asdict(robot_pose),
            )
            return
        self.last_pose = robot_pose
        intrinsics = intrinsics_from_camera_info(camera_info.value)
        camera_translation, camera_rotation = _transform_values(camera_from_map)
        try:
            camera_point = transform_point(
                (
                    self.args.stand_x,
                    self.args.stand_y,
                    self.stand_head_center_height_m,
                ),
                translation_xyz=camera_translation,
                rotation_xyzw=camera_rotation,
            )
        except ValueError as exc:
            self._note_observation_soft_miss(
                "stand_head_projection_failed",
                stamp_sec=image.stamp_sec,
                pose=robot_pose,
            )
            self._write_status(
                "stand_head_projection_failed",
                reason=str(exc),
            )
            return
        projection = project_optical_point(
            camera_point,
            intrinsics,
            physical_size_m=max(
                self.stand_model_profile.head_width_m,
                self.stand_model_profile.head_height_m,
            ),
        )
        expected_head_height_px = (
            0.0
            if projection.depth_m <= 0.0
            else (
                intrinsics.fy_px
                * self.stand_model_profile.head_height_m
                / projection.depth_m
            )
        )
        roi_attempts = target_centered_head_roi_attempts(
            projection,
            intrinsics,
            expected_head_height_px=expected_head_height_px,
            nominal_padding_scale=self.args.head_roi_padding_scale,
            backside_reacquisition_padding_scale=(
                self.args.backside_reacquisition_padding_scale
            ),
            enable_backside_reacquisition=(
                not self.args.disable_backside_reacquisition
            ),
        )
        if not roi_attempts or expected_head_height_px < self.args.min_head_size_px:
            self._note_observation_soft_miss(
                "target_outside_camera_gate",
                stamp_sec=image.stamp_sec,
                pose=robot_pose,
            )
            self._write_status(
                "target_outside_camera_gate",
                projection=asdict(projection),
                expected_head_height_px=expected_head_height_px,
            )
            return
        resolved_stand_axis_profile = self.stand_axis_profile.resolve(
            expected_head_height_px
        )
        scan_translation, scan_rotation = _transform_values(scan_from_map)
        scan_point = transform_point(
            (self.args.stand_x, self.args.stand_y, 0.0),
            translation_xyz=scan_translation,
            rotation_xyzw=scan_rotation,
        )
        scan_target = scan_target_geometry(
            scan_point,
            stand_radius_m=self.args.stand_radius_m,
            stand_uncertainty_m=self.args.stand_uncertainty_m,
            lidar_range_tolerance_m=self.args.lidar_range_tolerance_m,
        )
        scan_bearing = scan_target.bearing_rad
        plain_scan = plain_scan_from_sample(
            scan, topology_profile=getattr(self.args, "scan_topology_profile", "linear"))
        center_distance = math.hypot(
            robot_pose.x_m - self.args.stand_x,
            robot_pose.y_m - self.args.stand_y,
        )
        lower_surface_bound, upper_surface_bound = scan_target.accepted_range_m
        if getattr(self, "_scan_target_persistence", None) is None:
            self._scan_target_persistence = StoppedScanTargetPersistence()

        def resolve_lidar_association(association, current_scan, *, preview=False):
            # Use the same exact scan<-map transform that projected this
            # candidate. A witness never supplies a current beam or a pose.
            try:
                scan_pose = scan_pose_in_map(scan_translation, scan_rotation)
                static_scan_pose = scan_pose_from_camera_extrinsics(
                    *_transform_values(base_from_camera),
                    *_transform_values(scan_from_camera_transform))
                context = ScanPersistenceContext(
                    target_key=self._target_evidence_key(),
                    epoch_key=str(0 if self.observation_evidence is None else
                        self.observation_evidence.snapshot().motion_epoch),
                    robot_pose=robot_pose, scan_pose_map=scan_pose, image_stamp_sec=image.stamp_sec,
                    candidate_x_m=self.args.stand_x, candidate_y_m=self.args.stand_y,
                    stand_radius_m=self.args.stand_radius_m,
                    stand_uncertainty_m=self.args.stand_uncertainty_m,
                    lidar_range_tolerance_m=self.args.lidar_range_tolerance_m,
                    scan_pose_robot=static_scan_pose)
            except (TypeError, ValueError, ArithmeticError):
                if not preview:
                    self._scan_target_persistence.reset()
                return association
            resolver = (self._scan_target_persistence.preview if preview
                        else self._scan_target_persistence.resolve)
            return resolver(
                association, current_scan, context=context,
                now_sec=self.node.get_clock().now().nanoseconds / 1e9,
                max_scan_age_sec=self.args.max_sensor_age_sec)

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
        processing_started_monotonic = time.monotonic()
        processing_started_ros = self.node.get_clock().now().nanoseconds / 1e9
        head_budget = HeadProcessingDeadline(
            image_stamp_sec=image.stamp_sec, scan_stamp_sec=scan.stamp_sec,
            started_ros_sec=processing_started_ros,
            started_monotonic_sec=processing_started_monotonic,
            max_sensor_age_sec=self.args.max_sensor_age_sec)
        self._camera_count("processed_images")
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

        # A metric pose is expressed in camera coordinates, so its tracking
        # context is the immutable full-frame calibration rather than a
        # crop-local principal point.  Keeping this signature stable prevents
        # a bounded reacquisition crop from resetting an otherwise valid pose.
        camera_signature = (
            intrinsics.fx_px,
            intrinsics.fy_px,
            intrinsics.cx_px,
            intrinsics.cy_px,
        )
        prediction = self.model_pose_tracker.prediction(
            now_sec=image.stamp_sec,
            profile_sha256=self.stand_model_profile.sha256,
            camera_signature=camera_signature,
        )
        qr_decode_cache = RoiQrDecodeCache()
        model_input_cache = MetricModelInputCache(frame)
        if not hasattr(self, "_qr_acquisition_policy"):
            self._qr_acquisition_policy = QrAcquisitionPolicy()
        qr_acquisition_budget = self._qr_acquisition_policy.begin_frame(
            target_key=self._target_evidence_key(), image_stamp_sec=image.stamp_sec,
            started_ros_sec=processing_started_ros,
            started_monotonic_sec=processing_started_monotonic,
            max_sensor_age_sec=self.args.max_sensor_age_sec,
            work_deadline_monotonic_sec=head_budget.deadline_monotonic_sec,
        )

        def evaluate_roi_attempt(
            attempt: HeadRoiAttempt,
            pose_hint,
            current_head_proposal_corners=None,
        ) -> HeadRoiEvaluation:
            if not head_budget.allow("current_head_fit"):
                return unavailable_head_evaluation(attempt, frame, self.stand_model_profile,
                    "head_acquisition_deadline_exceeded", diagnostics=head_budget.metadata())
            attempt_roi = attempt.roi
            attempt_frame = frame[
                attempt_roi.y0 : attempt_roi.y1,
                attempt_roi.x0 : attempt_roi.x1,
            ]
            current_image_head_fit = CurrentImageHeadFit()
            def fit(qr_observations):
                return estimate_stand_axis_from_metric_model(
                    self.cv2,
                    attempt_frame,
                    model_profile=self.stand_model_profile,
                    camera_fx_px=intrinsics.fx_px,
                    camera_fy_px=intrinsics.fy_px,
                    camera_cx_px=intrinsics.cx_px - attempt_roi.x0,
                    camera_cy_px=intrinsics.cy_px - attempt_roi.y0,
                    pose_hint=pose_hint,
                    qr_observations=qr_observations,
                    edge_preprocess=resolved_stand_axis_profile.edge_preprocess,
                    canny_low=resolved_stand_axis_profile.canny_low,
                    canny_high=resolved_stand_axis_profile.canny_high,
                    min_edge_height_px=(
                        resolved_stand_axis_profile.min_edge_height_px
                    ),
                    expected_head_center_u_px=(
                        attempt.expected_center_u_px - attempt_roi.x0
                    ),
                    expected_head_center_v_px=(
                        attempt.expected_center_v_px - attempt_roi.y0
                    ),
                    expected_head_height_px=attempt.expected_head_height_px,
                    backside_target_crop_horizontal_half_width_ratio=(
                        attempt.backside_target_crop_half_width_ratio
                    ),
                    input_cache=model_input_cache,
                    input_cache_roi=(attempt_roi.x0, attempt_roi.y0, attempt_roi.x1, attempt_roi.y1),
                    current_head_proposal_corners=current_head_proposal_corners,
                    current_image_head_fit=current_image_head_fit,
                    deadline_monotonic_sec=head_budget.deadline_monotonic_sec,
                )

            attempt_estimate, attempt_debug, qr_observations, qr_metadata = evaluate_roi_with_qr_acquisition(
                frame=attempt_frame,
                roi=(attempt_roi.x0, attempt_roi.y0, attempt_roi.x1, attempt_roi.y1),
                roi_source=attempt.source, cache=qr_decode_cache, budget=qr_acquisition_budget,
                native_decoder=lambda crop: detect_native_qr_observations_bgr(crop, self.cv2),
                full_decoder=lambda crop, limit, provenance: detect_qr_observations_bgr(
                    crop, self.cv2, diagnostics=provenance, max_elapsed_sec=limit,
                    **({"prefer_native_geometry": True} if (
                        current_head_proposal_corners is not None
                        or attempt.source == "candidate_tracked_head_search") else {}),
                ),
                estimate=fit, now=time.monotonic, current_image_head_fit=current_image_head_fit,
            )
            return HeadRoiEvaluation(
                attempt=attempt,
                frame=attempt_frame,
                estimate=attempt_estimate,
                debug=attempt_debug,
                qr_observations=qr_observations,
                qr_decode_metadata={
                    **qr_metadata, "model_inputs": dict(model_input_cache.last_metadata),
                },
            )

        head_acquisition_metadata = {}

        def acquire_registered(attempt, primary):
            return acquire_registered_head_measurement(
                self.cv2, frame, attempt, intrinsics=intrinsics,
                scan_from_camera=scan_from_camera_geometry, scan=plain_scan,
                map_bearing_rad=scan_bearing,
                cone_half_angle_rad=math.radians(self.args.lidar_cone_half_angle_deg),
                accepted_range_m=(lower_surface_bound, upper_surface_bound),
                now_sec=self.node.get_clock().now().nanoseconds / 1e9,
                max_scan_age_sec=self.args.max_sensor_age_sec,
                min_cluster_sample_count=self.args.lidar_min_samples,
                max_camera_map_bearing_delta_rad=math.radians(
                    self.args.backside_registration_max_bearing_delta_deg),
                max_center_offset_ratio=self.args.backside_registration_max_center_offset_ratio,
                edge_preprocess=resolved_stand_axis_profile.edge_preprocess,
                canny_low=resolved_stand_axis_profile.canny_low,
                canny_high=resolved_stand_axis_profile.canny_high,
                evaluate=lambda selected, corners: evaluate_roi_attempt(selected, None, corners),
                diagnostics=head_acquisition_metadata,
                primary=primary,
                resolve_lidar_association=resolve_lidar_association,
                preview_lidar_association=lambda association, current_scan:
                    resolve_lidar_association(association, current_scan, preview=True),
                deadline_monotonic_sec=head_budget.deadline_monotonic_sec,
                current_ros_sec=lambda: self.node.get_clock().now().nanoseconds / 1e9,
            )

        candidate_context = CandidateHeadContext(
            target_key=self._target_evidence_key(),
            model_sha256=self.stand_model_profile.sha256,
            camera_signature=(self.profile.camera_optical_frame, *camera_signature,
                *(tuple(getattr(camera_info.value, field, ())) for field in ("k", "d", "r", "p")),
                str(getattr(camera_info.value, "distortion_model", "")),
                scan_camera_translation, scan_camera_rotation),
            image_shape=tuple(frame.shape),
            stationary_epoch=(0 if self.observation_evidence is None else
                              self.observation_evidence.snapshot().motion_epoch))
        candidate_search = self._candidate_search()
        search_hint = candidate_search.hint(
            roi_attempts, context=candidate_context,
            observed_at_sec=image.stamp_sec, robot_pose=robot_pose,
            max_center_offset_ratio=self.args.backside_registration_max_center_offset_ratio)
        search_metadata = dict(candidate_search.last_metadata)
        if search_hint is not None:
            # Prior pose/corners only locate pixels. This image gets one strict
            # current-border fit, with no alternate-border retry on ambiguity.
            registration = tracked_head_selection(
                evaluate_roi_attempt(search_hint.attempt, search_hint.pose_hint))
        elif (getattr(self.stand_model_profile, "committable", False)
                and self.stand_model_profile.environment == "physical"
                and not self.args.disable_backside_reacquisition):
            registration = select_cold_candidate_head(
                roi_attempts, frame=frame, model_profile=self.stand_model_profile,
                acquire_registered=acquire_registered,
                diagnostics=head_acquisition_metadata, budget=head_budget)
        else:
            registration = self.backside_proposal_reuse.select(
                roi_attempts,
                context=BacksideProposalContext(
                    target_key=candidate_context.target_key,
                    model_sha256=candidate_context.model_sha256,
                    camera_signature=candidate_context.camera_signature,
                    image_shape=candidate_context.image_shape),
                observed_at_sec=image.stamp_sec,
                robot_pose=robot_pose,
                marker_seen_in_stationary_epoch=self._qr_marker_seen_in_stationary_epoch,
                tracked_pose=None,
                evaluate=evaluate_roi_attempt,
                acquire_registered=acquire_registered,
                enable_reacquisition=not self.args.disable_backside_reacquisition,
                max_center_offset_ratio=(
                    self.args.backside_registration_max_center_offset_ratio
                ),
            )
        current_head_association = None
        current = registration.selected
        estimate, debug, selected_attempt = current.estimate, current.debug, current.attempt
        now_sec = self.node.get_clock().now().nanoseconds / 1e9
        if (requires_measured_head_admission(estimate, debug)
                and (estimate.usable or validated_current_head_orientation_bounds(
                    getattr(debug, "head_orientation_bounds", None), estimate=estimate, debug=debug))
                and self._source_freshness(image.stamp_sec, scan.stamp_sec).accepted):
            current_head_association = associate_current_measured_head(
                estimate=estimate, debug=debug, attempt=selected_attempt,
                projection=projection, expected_head_height_px=expected_head_height_px,
                profile_sha256=self.stand_model_profile.sha256,
                intrinsics=intrinsics, scan_from_camera=scan_from_camera_geometry,
                scan=plain_scan, map_bearing_rad=scan_bearing,
                cone_half_angle_rad=math.radians(self.args.lidar_cone_half_angle_deg),
                accepted_range_m=(lower_surface_bound, upper_surface_bound),
                now_sec=now_sec, max_scan_age_sec=self.args.max_sensor_age_sec,
                min_cluster_sample_count=self.args.lidar_min_samples,
                max_center_offset_ratio=self.args.backside_registration_max_center_offset_ratio,
                max_camera_map_bearing_delta_rad=math.radians(
                    self.args.backside_registration_max_bearing_delta_deg),
                resolve_lidar_association=resolve_lidar_association,
            )
        if search_hint is not None:
            registration = register_current_tracked_head(
                registration, association=current_head_association,
                observed_at_sec=image.stamp_sec, now_sec=now_sec,
                max_age_sec=self.args.max_sensor_age_sec,
                expected_model_sha256=self.stand_model_profile.sha256)
        registration, backside_crop_review = gate_backside_head_crop(registration)
        selected = registration.selected
        selected_attempt = selected.attempt
        roi_frame = selected.frame
        estimate = selected.estimate
        debug = selected.debug
        self._last_metric_axis_source = estimate.source
        roi = selected_attempt.roi
        # Processing can outlive the tuple's admission-time freshness check.
        now_sec = self.node.get_clock().now().nanoseconds / 1_000_000_000.0
        processing_completed_monotonic = time.monotonic()
        result_freshness = observation_freshness(
            observed_at_sec=image.stamp_sec,
            now_sec=now_sec,
            max_age_sec=self.args.max_sensor_age_sec,
            max_future_sec=self.args.max_future_timestamp_sec,
        )
        tracker_update = self.model_pose_tracker.update_from_observation(
            estimate,
            debug,
            observed_at_sec=image.stamp_sec,
            completed_at_sec=now_sec,
            profile_sha256=self.stand_model_profile.sha256,
            camera_signature=camera_signature,
            result_fresh=result_freshness.accepted,
        )
        if result_freshness.accepted:
            self._camera_count("fresh_detector_results")
        if estimate.usable and estimate.evidence_state in {"fresh_refined", "fresh_backside"}:
            self._camera_count("verified_geometry_results")
        model_metadata = {
            "mode": "metric_model_only",
            "profile_id": self.stand_model_profile.profile_id,
            "profile_sha256": self.stand_model_profile.sha256,
            "environment": self.stand_model_profile.environment,
            "measurement_status": self.stand_model_profile.measurement_status,
            "target_projection": asdict(projection),
            "head_roi": selected_attempt.metadata(),
            "head_roi_attempts": [
                evaluation.attempt.metadata()
                for evaluation in registration.evaluations
            ],
            "camera_target_registration": registration.metadata(
                enabled=not self.args.disable_backside_reacquisition
            ),
            "candidate_head_search": search_metadata,
            "backside_proposal_reuse": dict(self.backside_proposal_reuse.last_metadata),
            "backside_head_crop": (None if backside_crop_review is None
                                   else backside_crop_review.metadata()),
            "scan_target_persistence": dict(self._scan_target_persistence.last_metadata),
            "head_acquisition": head_acquisition_metadata,
            "evidence_state": estimate.evidence_state,
            "qr_detected": debug.qr_detected,
            "qr_marker_verified": debug.qr_marker_verified,
            "qr_marker_reason": debug.qr_marker_reason,
            "pose_reprojection_rmse_px": (
                estimate.pose_reprojection_rmse_px
            ),
            "pose_ambiguity_gap_px": estimate.pose_ambiguity_gap_px,
            "refinement_support_mean": debug.refinement_support_mean,
            "model_pose_fit_source": debug.model_pose_fit_source,
            **metric_fit_diagnostics_payload(debug),
            "tracker_prediction": {
                "state": prediction.state,
                "age_sec": prediction.age_sec,
                "reason": prediction.reason,
            },
            "tracker_update": asdict(tracker_update),
            "result_freshness": asdict(result_freshness),
            "processing_timing": {
                "image_stamp_sec": image.stamp_sec,
                "image_received_ros_sec": image.received_ros_sec,
                "image_received_monotonic_sec": image.received_monotonic_sec,
                "started_ros_sec": processing_started_ros,
                "started_monotonic_sec": processing_started_monotonic,
                "detector_completed_ros_sec": now_sec,
                "detector_completed_monotonic_sec": processing_completed_monotonic,
                "detector_elapsed_ms": (processing_completed_monotonic - processing_started_monotonic) * 1000,
                "qr_acquisition": qr_acquisition_budget.metadata(),
                "head_processing_budget": head_budget.metadata(),
                "attempts": [{
                    "roi": evaluation.attempt.metadata(),
                    "qr_decode": evaluation.qr_decode_metadata,
                    "stage_timings_ms": evaluation.debug.stage_timings_ms,
                } for evaluation in registration.evaluations],
            },
            "visible_face": getattr(estimate, "visible_face", None),
            "visible_face_confidence": getattr(
                estimate,
                "visible_face_confidence",
                None,
            ),
            "visible_face_reason": getattr(
                debug,
                "visible_face_reason",
                None,
            ),
            "head_scale_ratio": getattr(debug, "head_scale_ratio", None),
            "head_center_error_ratio": getattr(
                debug,
                "head_center_error_ratio",
                None,
            ),
        }
        axis_metadata = {
            "profile": asdict(resolved_stand_axis_profile),
            "estimator_mode": "metric_model_only",
            "estimator_usable": estimate.usable,
            "estimator_view_mode": estimate.mode,
            "estimator_reason": estimate.reason,
            "estimator_source": estimate.source,
            "advisory_camera_relative_yaw_rad": (
                None if estimate.yaw_deg is None else math.radians(estimate.yaw_deg)
            ),
            "metric_model": model_metadata,
            "preliminary_candidate_lidar_association": asdict(
                preliminary_lidar_association
            ),
        }
        if getattr(self, "_capture_pending", None) is not None:
            self._capture_pending["detector_metadata"] = axis_metadata
        if not result_freshness.accepted:
            self._camera_count("obsolete_detector_results")
            self._reset_candidate_search("obsolete_detector_result")
            self._note_observation_soft_miss(
                "obsolete_detector_result", stamp_sec=image.stamp_sec, pose=robot_pose
            )
            self._write_debug(frame, roi_frame, debug, metadata=axis_metadata)
            self._write_status(
                "obsolete_detector_result",
                image_stamp_sec=image.stamp_sec,
                image_age_sec=result_freshness.age_sec,
                stand_axis_debug=axis_metadata,
            )
            return
        if current_head_association is not None:
            axis_metadata["current_head_candidate_association"] = current_head_association.metadata()
            model_metadata["scan_target_persistence"] = dict(self._scan_target_persistence.last_metadata)
        selected_qr_observations = getattr(selected, "qr_observations", None)
        if selected_qr_observations is None and selected.qr_decode_metadata is None:
            # Preserve legacy injected evaluations; the operational evaluator
            # carries the same observations used by metric fitting, or an
            # explicit not-performed result. A spent head budget must not
            # trigger an unbudgeted legacy decode of the wider search image.
            qr_texts = tuple(sorted(set(detect_qr_texts_bgr(roi_frame, self.cv2))))
        else:
            qr_texts = tuple(sorted({
                observation.text for observation in (selected_qr_observations or ())
            }))
        # Recentring may remove a previously observed marker from the crop.
        # Preserve its veto/conflict evidence; target identity still comes
        # only from the selected symbol's independently bound image ray.
        roi_qr_evidence = summarize_roi_qr_evidence(registration)
        qr_texts = tuple(sorted(set(qr_texts) | set(roi_qr_evidence.qr_texts)))
        axis_metadata["roi_qr_evidence"] = roi_qr_evidence.metadata()
        qr_binding = bind_qr_observations_to_target(
            selected_qr_observations, roi=roi, intrinsics=intrinsics,
            scan_from_camera=scan_from_camera_geometry, scan=plain_scan,
            map_bearing_rad=scan_bearing,
            cone_half_angle_rad=math.radians(self.args.lidar_cone_half_angle_deg),
            accepted_range_m=(lower_surface_bound, upper_surface_bound),
            now_sec=now_sec, max_scan_age_sec=self.args.max_sensor_age_sec,
            min_cluster_sample_count=self.args.lidar_min_samples,
            camera_registration_accepted=(is_camera_registered_head_roi_attempt(selected_attempt)
                or (current_head_association is not None and current_head_association.accepted)),
            max_camera_map_bearing_delta_rad=math.radians(
                self.args.backside_registration_max_bearing_delta_deg
            ),
        )
        if current_head_association is not None and current_head_association.accepted:
            qr_binding = bind_qr_to_current_head(
                qr_binding, selected_qr_observations, head_corners=estimate.corners,
                head_association=current_head_association,
            )
        qr_evidence_texts = qr_binding.qr_texts_for_evidence
        axis_metadata["decoded_qr_target_binding"] = qr_binding.metadata()
        # The neutral head can be associated even before a QR ray or axis is
        # usable. Admit that frame's unresolved progress through the same
        # current scan gate, without granting its unbound text identity.
        frame_lidar_associated = (
            preliminary_lidar_association.associated or qr_binding.accepted
            or (current_head_association is not None and current_head_association.accepted)
            or (registration.registered
                and (registration.head_acquisition or {}).get("candidate_associated") is True)
        )
        front_decision = front_observation_decision(
            qr_texts=qr_texts,
            qr_marker_detected=roi_qr_evidence.marker_detected,
            qr_marker_verified=roi_qr_evidence.marker_verified,
            estimate_source=estimate.source,
            marker_seen_in_stationary_epoch=self._qr_marker_seen_in_stationary_epoch,
        )
        self._note_front_observation(front_decision, robot_pose)
        # Appearance is independent of angle observability. Complete current
        # pixels, association and the ordinary evidence update still gate each
        # repeated sample; this pending value belongs to this image only.
        appearance_crop = backside_crop_review or review_backside_head_crop(registration)
        self._pending_head_confidence = (
            HeadConfidenceInput(
                image.stamp_sec, getattr(debug, "head_backside_appearance", None),
                appearance_crop.accepted, front_decision.marker_observed_now,
                front_decision.marker_seen_in_stationary_epoch,
                estimate.usable and estimate.yaw_deg is not None, estimate.reason),
            model_metadata,
        )
        model_metadata["backside_head_crop"] = appearance_crop.metadata()
        self._pending_head_window = current_head_window_input(
            estimate, debug, frame_stamp_sec=image.stamp_sec,
            camera_signature=(self.profile.camera_optical_frame, *camera_signature,
                              intrinsics.width_px, intrinsics.height_px),
            roi=roi, projected_center_px=(projection.u_px, projection.v_px),
            expected_head_height_px=expected_head_height_px)
        current_crop = review_current_head_crop(registration)
        self._pending_head_window_associated = current_crop.accepted
        model_metadata["current_head_crop"] = current_crop.metadata()
        self._pending_bounded_head = prepare_bounded_head(
            estimate=estimate, debug=debug, association=current_head_association,
            crop=current_crop, appearance_crop=appearance_crop, qr_binding=qr_binding,
            marker_verified=roi_qr_evidence.marker_verified,
            marker_seen_in_epoch=self._qr_marker_seen_in_stationary_epoch,
            image_stamp_sec=image.stamp_sec, scan_stamp_sec=scan.stamp_sec,
            robot_pose=robot_pose, camera_heading_rad=optical_heading_from_transform(map_from_camera),
            stand_x_m=self.args.stand_x, stand_y_m=self.args.stand_y,
            camera_signature=candidate_context.camera_signature, roi=roi, metadata=model_metadata,
            projected_center_px=(projection.u_px, projection.v_px),
            expected_head_height_px=expected_head_height_px)
        if self._pending_bounded_head is not None:
            proof = self._pending_bounded_head.proof
            axis_metadata["bounded_head_view_hint"] = {
                "camera_relative_yaw_rad": proof.center_rad,
                "orientation_half_width_rad": proof.half_width_rad,
                "purpose": "orientation_disambiguation", "candidate_associated": True,
                "source_fresh": self._source_freshness(image.stamp_sec, scan.stamp_sec).accepted,
            }
        if estimate.model_profile_sha256 != self.stand_model_profile.sha256:
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
        if qr_binding.symbol_count > 1 or roi_qr_evidence.conflict_reason is not None:
            self._camera_framing = None
            update = self._record_observation_frame(
                robot_pose=robot_pose,
                image_stamp_sec=image.stamp_sec, scan_stamp_sec=scan.stamp_sec,
                observed_at_sec=now_sec,
                lidar_associated=frame_lidar_associated,
                axis_yaw_rad=None, axis_source=None, qr_texts=(),
                qr_symbol_count=max(qr_binding.symbol_count, roi_qr_evidence.symbol_count, 2),
            )
            self._write_debug(frame, roi_frame, debug, metadata=axis_metadata)
            self._write_status(
                "evidence_not_committable",
                reason="multiple_qr_symbols_in_candidate_frame",
                roi_conflict_reason=roi_qr_evidence.conflict_reason,
                qr_texts=list(qr_texts),
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
            )
            return
        if (
            self.args.expected_qr_id != AUTO_QR_ID
            and qr_texts
            and qr_texts != (self.args.expected_qr_id,)
        ):
            self._camera_framing = None
            update = self._record_observation_frame(
                robot_pose=robot_pose,
                image_stamp_sec=image.stamp_sec, scan_stamp_sec=scan.stamp_sec,
                observed_at_sec=now_sec, lidar_associated=qr_binding.accepted,
                axis_yaw_rad=None, axis_source=None,
                qr_texts=qr_evidence_texts,
            )
            self._write_debug(frame, roi_frame, debug, metadata=axis_metadata)
            self._write_status(
                "evidence_not_committable",
                reason="observed QR identity differs from expected identity",
                qr_texts=list(qr_texts),
                expected_qr_id=self.args.expected_qr_id,
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
            )
            return
        # Retain a locator only after the final current-source check and target
        # association. QR presence/classification never controls this hint.
        candidate_search.remember(
            selected, context=candidate_context,
            observed_at_sec=image.stamp_sec,
            now_sec=self.node.get_clock().now().nanoseconds / 1e9,
            max_age_sec=self.args.max_sensor_age_sec, robot_pose=robot_pose,
            candidate_associated=(current_head_association is not None
                and current_head_association.accepted
                and self._source_freshness(image.stamp_sec, scan.stamp_sec).accepted))
        model_metadata["candidate_head_tracking"] = dict(candidate_search.last_metadata)
        framing = unresolved_front_framing_hint(
            registration, target_key=self.args.stand_id,
            source_image_stamp_sec=image.stamp_sec,
            source_fresh=self._source_freshness(image.stamp_sec, scan.stamp_sec).accepted,
            range_m=center_distance, optical_depth_m=projection.depth_m,
            intrinsics=intrinsics,
        )
        if estimate.usable:
            self._camera_framing = None
        if front_decision.withhold_backside_axis:
            # Positive QR evidence contradicts this geometry mode, not the
            # identity channel. Retain its ordinary target/sensor gates and
            # poison history while withholding every QR-free axis sample.
            update, front_metadata = self._record_front_seen_axis_unresolved(
                decision=front_decision,
                robot_pose=robot_pose,
                image_stamp_sec=image.stamp_sec,
                scan_stamp_sec=scan.stamp_sec,
                observed_at_sec=now_sec,
                lidar_associated=frame_lidar_associated,
                qr_texts=qr_evidence_texts,
            )
            self._write_debug(frame, roi_frame, debug, metadata=axis_metadata)
            self._write_status(
                "evidence_not_committable",
                reason="front_seen_axis_unresolved",
                estimator_reason=estimate.reason,
                qr_texts=list(qr_texts),
                expected_qr_id=self.args.expected_qr_id,
                observation_evidence=update.snapshot.as_dict(),
                front_observation=front_metadata,
                stand_axis_debug=axis_metadata,
            )
            return
        if not estimate.usable or estimate.yaw_deg is None:
            update = self._record_observation_frame(
                robot_pose=robot_pose,
                image_stamp_sec=image.stamp_sec,
                scan_stamp_sec=scan.stamp_sec,
                observed_at_sec=now_sec,
                lidar_associated=frame_lidar_associated,
                axis_yaw_rad=None,
                axis_source=None,
                qr_texts=qr_evidence_texts,
            )
            if framing is not None and update.frame_accepted and not update.snapshot.poisoned:
                self._camera_framing = framing
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
                qr_texts=list(qr_texts),
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
            )
            return
        scale_gate = _head_scale_gate(
            expected_size_px=expected_head_height_px,
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
                lidar_associated=(preliminary_lidar_association.associated or qr_binding.accepted),
                axis_yaw_rad=None,
                axis_source=None,
                qr_texts=qr_evidence_texts,
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
                lidar_associated=(preliminary_lidar_association.associated or qr_binding.accepted),
                axis_yaw_rad=None,
                axis_source=None,
                qr_texts=qr_evidence_texts,
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
        registration_applied = is_camera_registered_head_roi_attempt(
            selected_attempt
        )
        registered_lidar_association = None
        if current_head_association is not None:
            registered_lidar_association = current_head_association.lidar_association
            lidar_association = (
                registered_lidar_association.search_association
                if registered_lidar_association is not None
                and registered_lidar_association.search_association is not None
                else preliminary_lidar_association
            )
            # The nested diagnostic fallback carries no acceptance authority.
            lidar_target_associated = current_head_association.accepted
            axis_metadata["measured_head_lidar_admission"] = current_head_association.metadata()
        elif registration_applied:
            registered_lidar_association = (
                associate_camera_registered_candidate_lidar_target(
                    plain_scan,
                    map_bearing_rad=scan_bearing,
                    observed_camera_bearing_rad=observed_camera_bearing,
                    cone_half_angle_rad=math.radians(
                        self.args.lidar_cone_half_angle_deg
                    ),
                    accepted_range_m=(
                        lower_surface_bound,
                        upper_surface_bound,
                    ),
                    now_sec=now_sec,
                    max_scan_age_sec=self.args.max_sensor_age_sec,
                    min_cluster_sample_count=self.args.lidar_min_samples,
                    max_camera_map_bearing_delta_rad=math.radians(
                        self.args.backside_registration_max_bearing_delta_deg
                    ),
                )
            )
            # Keep the stable diagnostics payload shaped like the legacy
            # association even when registration moves the narrow search
            # cone.  The wrapper remains a separate, explicit provenance
            # record and alone decides whether this exception path passed.
            lidar_association = (
                registered_lidar_association.search_association
                or preliminary_lidar_association
            )
            axis_metadata[
                "camera_registered_candidate_lidar_association"
            ] = asdict(registered_lidar_association)
            lidar_target_associated = registered_lidar_association.associated
        else:
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
            lidar_target_associated = lidar_association.associated
        lidar_status_details = {
            "candidate_lidar_association": asdict(lidar_association),
            "camera_registered_candidate_lidar_association": (
                None
                if registered_lidar_association is None
                else asdict(registered_lidar_association)
            ),
        }
        axis_metadata.update(lidar_status_details)
        if not lidar_target_associated:
            update = self._record_observation_frame(
                robot_pose=robot_pose,
                image_stamp_sec=image.stamp_sec,
                scan_stamp_sec=scan.stamp_sec,
                observed_at_sec=now_sec,
                lidar_associated=False,
                axis_yaw_rad=None,
                axis_source=None,
                qr_texts=qr_evidence_texts,
            )
            self._write_debug(frame, roi_frame, debug, metadata=axis_metadata)
            self._write_status(
                "lidar_target_mismatch",
                center_distance_m=center_distance,
                accepted_range_m=[lower_surface_bound, upper_surface_bound],
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
                **lidar_status_details,
            )
            return
        axis_sample_admission = admit_axis_sample(
            estimate=estimate,
            debug=debug,
            conditioning=conditioning,
            yaw_rad=optical_yaw,
            qr_texts=qr_evidence_texts,
            lidar_target_associated=lidar_target_associated,
            max_qr_bound_model_obliqueness_rad=math.radians(
                self.args.qr_bound_model_max_obliqueness_deg
            ),
        )
        axis_metadata["axis_sample_admission"] = (
            axis_sample_admission.metadata()
        )
        if not axis_sample_admission.accepted:
            update = self._record_observation_frame(
                robot_pose=robot_pose,
                image_stamp_sec=image.stamp_sec,
                scan_stamp_sec=scan.stamp_sec,
                observed_at_sec=now_sec,
                lidar_associated=True,
                axis_yaw_rad=None,
                axis_source=None,
                qr_texts=qr_evidence_texts,
            )
            self._write_debug(frame, roi_frame, debug, metadata=axis_metadata)
            self._write_status(
                "evidence_not_committable",
                conditioning=asdict(conditioning),
                qr_texts=list(qr_texts),
                expected_qr_id=self.args.expected_qr_id,
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
                **lidar_status_details,
            )
            return
        axis_sample_source = axis_sample_admission.source
        if (
            axis_sample_source is None
            or axis_sample_admission.yaw_rad is None
        ):
            raise AssertionError("accepted axis sample lacks yaw/source")
        target_registration = None
        if estimate.source == BACKSIDE_AXIS_SAMPLE_SOURCE:
            try:
                target_registration = (
                    build_backside_target_registration_evidence(
                        current_head_association=current_head_association,
                        final_head_center_error_ratio=(
                            debug.head_center_error_ratio
                        ),
                        candidate_lidar_association=lidar_association,
                        registration_decision=(
                            registration.decision
                            if registration_applied
                            else None
                        ),
                        registered_lidar_association=(
                            registered_lidar_association
                        ),
                    )
                )
            except (TypeError, ValueError) as exc:
                update = self._record_observation_frame(
                    robot_pose=robot_pose,
                    image_stamp_sec=image.stamp_sec,
                    scan_stamp_sec=scan.stamp_sec,
                    observed_at_sec=now_sec,
                    lidar_associated=True,
                    axis_yaw_rad=None,
                    axis_source=None,
                    qr_texts=qr_evidence_texts,
                )
                self._write_debug(
                    frame,
                    roi_frame,
                    debug,
                    metadata=axis_metadata,
                )
                self._write_status(
                    "evidence_not_committable",
                    reason=f"target registration evidence unavailable: {exc}",
                    qr_texts=list(qr_texts),
                    observation_evidence=update.snapshot.as_dict(),
                    stand_axis_debug=axis_metadata,
                    **lidar_status_details,
                )
                return
            if target_registration["mode"] == "bounded_camera_lidar_registration":
                axis_sample_source = (
                    REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE
                )
            axis_metadata["target_registration"] = target_registration
        update = self._record_observation_frame(
            robot_pose=robot_pose,
            image_stamp_sec=image.stamp_sec,
            scan_stamp_sec=scan.stamp_sec,
            observed_at_sec=now_sec,
            lidar_associated=True,
            axis_yaw_rad=axis_sample_admission.yaw_rad,
            axis_source=axis_sample_source,
            qr_texts=qr_evidence_texts,
        )
        # A completed bucket from a different acquisition mode cannot
        # authenticate the current frame's registration evidence.
        consensus = _consensus_for_current_axis_source(
            update,
            axis_sample_source,
        )
        resolved_qr_id = update.resolved_qr_id
        self._write_debug(
            frame,
            roi_frame,
            debug,
            metadata=axis_metadata,
        )
        if getattr(self, "_pending_bounded_head", None) is not None:
            # A measured interval must never be erased by the older precise-
            # axis writer. Its receipt is attempted first in _write_status.
            window = getattr(self, "_bounded_head_window", None)
            reason = "collecting_bounded_orientation" if window is None else window.metadata["reason"]
            state = ("collecting_consensus" if reason in {
                "collecting_bounded_orientation", "bounded_orientation_ready"
            } else "evidence_not_committable")
            self._write_status(state, reason=reason, qr_texts=list(qr_texts),
                observation_evidence=update.snapshot.as_dict(), stand_axis_debug=axis_metadata,
                **lidar_status_details)
            return
        if update.snapshot.poisoned:
            self._write_status(
                "evidence_not_committable",
                reason=update.snapshot.poison_reason,
                conditioning=asdict(conditioning),
                qr_texts=list(qr_texts),
                expected_qr_id=self.args.expected_qr_id,
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
                **lidar_status_details,
            )
            return
        if consensus is None:
            self._write_status(
                "collecting_consensus",
                qr_texts=list(qr_texts),
                estimator_source=estimate.source,
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
                **lidar_status_details,
            )
            return
        if (consensus.source in BACKSIDE_AXIS_SOURCES
                and requires_measured_head_admission(estimate, debug)
                and (getattr(self, "_head_confidence_metadata", None) or {}).get(
                    "backside", {}).get("state") != "backside_supported"):
            self._write_status(
                "collecting_consensus", reason="backside_appearance_not_yet_repeated",
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata, **lidar_status_details)
            return
        self._camera_count("consensus_frames")
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
        if consensus.source in BACKSIDE_AXIS_SOURCES and confidence < MINIMUM_BACKSIDE_AXIS_CONFIDENCE:
            # Readiness and receipt validation use the same threshold. The
            # separate bounded path can still supply an honest interval.
            self._write_status(
                "collecting_consensus", reason="backside_axis_confidence_below_receipt_minimum",
                axis_confidence=confidence, required_axis_confidence=MINIMUM_BACKSIDE_AXIS_CONFIDENCE,
                observation_evidence=update.snapshot.as_dict(), stand_axis_debug=axis_metadata,
                **lidar_status_details)
            return
        if (consensus.source == MEASURED_HEAD_AXIS_SOURCE
                and not measured_head_front_is_current(
                    qr_binding=qr_binding, marker_verified=roi_qr_evidence.marker_verified,
                    resolved_qr_id=resolved_qr_id)):
            # The current head plane is an angle measurement, not a face
            # classification. Missing QR cannot authorize an opposite-side
            # route, and an older identity latch cannot declare this face.
            self._write_status(
                "axis_observation_not_committable",
                reason="measured_head_front_identity_unresolved",
                axis_sample_count=consensus.sample_count,
                axis_sample_source=consensus.source,
                qr_texts=list(qr_texts),
                observation_evidence=update.snapshot.as_dict(),
                stand_axis_debug=axis_metadata,
                **lidar_status_details,
            )
            return
        if resolved_qr_id is None:
            snapshot = update.snapshot
            try:
                axis_observation = build_backside_axis_observation(
                    stream_id=self.args.stream_id,
                    stand_id=self.args.stand_id,
                    planning_frame=self.profile.map_frame,
                    stand_x_m=self.args.stand_x,
                    stand_y_m=self.args.stand_y,
                    robot_x_m=robot_pose.x_m,
                    robot_y_m=robot_pose.y_m,
                    robot_yaw_rad=robot_pose.yaw_rad,
                    stand_axis_rad=stand_axis,
                    axis_confidence=confidence,
                    axis_sample_count=consensus.sample_count,
                    consensus_source=consensus.source,
                    estimate_source=estimate.source,
                    estimate_evidence_state=estimate.evidence_state,
                    estimate_visible_face=getattr(
                        estimate,
                        "visible_face",
                        None,
                    ),
                    visible_face_confidence=getattr(
                        estimate,
                        "visible_face_confidence",
                        None,
                    ),
                    debug_qr_detected=debug.qr_detected,
                    qr_texts=qr_texts,
                    evidence_qr_sample_count=(
                        snapshot.current_qr_sample_count
                    ),
                    evidence_tentative_qr_id=snapshot.tentative_qr_id,
                    evidence_latched_qr_id=snapshot.latched_qr_id,
                    qr_marker_seen_in_stationary_epoch=(
                        self._qr_marker_seen_in_stationary_epoch
                    ),
                    # Axis samples enter PassiveObserverEvidence only below
                    # the stationary, synchronized-tuple, and final LiDAR
                    # association gates in this method.
                    all_samples_stationary=True,
                    all_samples_synchronized=True,
                    all_samples_lidar_associated=True,
                    sensor_stamp_sec=image.stamp_sec,
                    stand_model_profile_sha256=(
                        estimate.model_profile_sha256
                    ),
                    stand_model_measurement_status=(
                        self.stand_model_profile.measurement_status
                    ),
                    head_scale_ratio=getattr(
                        debug,
                        "head_scale_ratio",
                        None,
                    ),
                    head_center_error_ratio=getattr(
                        debug,
                        "head_center_error_ratio",
                        None,
                    ),
                    pose_reprojection_rmse_px=(
                        estimate.pose_reprojection_rmse_px
                    ),
                    pose_ambiguity_gap_px=estimate.pose_ambiguity_gap_px,
                    robot_profile_sha256=real_robot_profile_sha256(
                        self.profile
                    ),
                    calibration_profile_sha256=(
                        camera_calibration_sha256(self.calibration)
                    ),
                    target_registration=target_registration,
                )
            except (TypeError, ValueError) as exc:
                # Ordinary QR/tracked metric estimates without a resolved QR
                # must not silently become an opposite-face motion handoff.
                self._write_status(
                    "axis_observation_not_committable",
                    reason=str(exc),
                    axis_sample_count=consensus.sample_count,
                    axis_sample_source=consensus.source,
                    qr_texts=[],
                    observation_evidence=snapshot.as_dict(),
                    stand_axis_debug=axis_metadata,
                    **lidar_status_details,
                )
                return
            if self.args.axis_observation_json is None:
                self._write_status(
                    "backside_axis_output_unconfigured",
                    axis_sample_count=consensus.sample_count,
                    axis_confidence=confidence,
                    stand_axis_debug=axis_metadata,
                    **lidar_status_details,
                )
                return
            if not self._commit_sensor_artifact(
                self.args.axis_observation_json,
                axis_observation,
                image_stamp_sec=image.stamp_sec, scan_stamp_sec=scan.stamp_sec,
                artifact_kind="backside_axis_observation",
            ):
                self._write_status("obsolete_publication_evidence", stand_axis_debug=axis_metadata)
                return
            self.axis_observation_committed = True
            self.completed = True
            self._write_status(
                "backside_axis_committed_qr_unresolved",
                axis_observation=str(self.args.axis_observation_json),
                axis_sample_count=consensus.sample_count,
                axis_confidence=confidence,
                visible_face="backside_candidate",
                qr_texts=[],
                stand_axis_debug=axis_metadata,
                **lidar_status_details,
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
            observation_unix_sec=image.stamp_sec,
        )
        recommendation_payload = recommendation_to_dict(recommendation)
        recommendation_payload["axis_measurement"] = {
            "source": consensus.source,
            "model_profile_sha256": estimate.model_profile_sha256,
            "model_measurement_status": estimate.model_measurement_status,
            "head_model_quality": (
                None if getattr(debug, "head_model_quality", None) is None
                else asdict(debug.head_model_quality)
            ),
            "sample_admission": axis_sample_admission.metadata(),
            "sensor_stamp_sec": image.stamp_sec,
        }
        if not self._commit_sensor_artifact(
            self.args.recommended_pose_json,
            recommendation_payload,
            image_stamp_sec=image.stamp_sec, scan_stamp_sec=scan.stamp_sec,
            artifact_kind="recommendation",
        ):
            self._write_status("obsolete_publication_evidence", stand_axis_debug=axis_metadata)
            return
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
            **lidar_status_details,
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
        _atomic_json(
            self.args.debug_dir / "latest_metadata.json",
            {
                "schema_version": 1,
                "observed_unix_sec": time.time(),
                "artifacts": written,
                "stand_axis": metadata,
            },
        )

    def _maybe_commit_inspection_progress(self, state: str, details: dict):
        """Publish a third, advisory result only after stronger paths declined."""

        output = getattr(self.args, "inspection_observation_json", None)
        if output is None or getattr(self, "completed", False):
            return None
        current = getattr(self, "_inspection_frame", None)
        # Each processed tuple can be consumed by exactly its immediate
        # status publication. A later TF/sensor/motion status cannot reuse it.
        self._inspection_frame = None
        axis_sample_accepted = qr_sample_accepted = False
        if current is not None:
            current = dict(current)
            axis_sample_accepted = current.pop("axis_sample_accepted", False)
            qr_sample_accepted = current.pop("qr_sample_accepted", False)
            scan_stamp_sec = current.pop("scan_stamp_sec", None)
        if state not in {
            "metric_model_measurement_unavailable", "evidence_not_committable",
            "axis_observation_not_committable", "collecting_consensus",
        }:
            return None
        if current is None:
            return None
        progress = getattr(self, "_inspection_progress", None)
        if progress is None:
            progress = InspectionProgress(
                required_frames=getattr(self.args, "inspection_progress_frames", 7),
                minimum_span_sec=getattr(self.args, "inspection_progress_min_span_sec", 2.0),
                max_age_sec=INSPECTION_PROGRESS_WINDOW_SEC,
                max_translation_m=self.args.stationary_translation_m,
                max_rotation_rad=math.radians(self.args.stationary_rotation_deg),
            )
            self._inspection_progress = progress
        failure_kind = front_view_failure_kind(state, details)
        # An ambiguous neutral head cannot become a front-view advisory just
        # because the nominal projected cone contained some LiDAR returns.
        # This narrows advisory accumulation only; QR/axis admission is unchanged.
        unassociated_front = failure_kind is None and front_view_failure_kind(
            state, details, require_candidate_association=False,
        ) is not None
        fields = progress.record(
            **{**current, "frame_accepted": current["frame_accepted"] and not unassociated_front},
            classification=classify_inspection_progress(state, details),
        )
        recovery = getattr(self, "_front_view_recovery", None)
        if recovery is None and failure_kind is not None:
            recovery = FrontViewRecovery(
                max_translation_m=self.args.stationary_translation_m,
                max_rotation_rad=math.radians(self.args.stationary_rotation_deg),
            )
            self._front_view_recovery = recovery
        defer_front_advisory = False
        if recovery is not None:
            fresh = self._source_freshness(current["frame_stamp_sec"], scan_stamp_sec).accepted
            defer_front_advisory = recovery.observe(
                target_key=self._target_evidence_key(), now_sec=time.monotonic(),
                frame_stamp_sec=current["frame_stamp_sec"], robot_pose=current["robot_pose"],
                frame_accepted=current["frame_accepted"], source_fresh=fresh,
                poisoned=current["poisoned"] or progress.poisoned,
                motion_epoch_reset=current.get("motion_epoch_reset", False),
                failure_kind=failure_kind,
            )
        if (
            state == "collecting_consensus" and axis_sample_accepted
        ) or (
            qr_sample_accepted and current.get("current_qr_sample_count", 0) < 2
        ):
            # Give axis acquisition its full consensus window and a new QR
            # decode its second latch sample. Keep epoch poison/identity while
            # clearing earlier failure timing, so recovery cannot hide conflict.
            progress.restart_acquisition_window()
            return None
        if state == "collecting_consensus":
            return None
        if defer_front_advisory:
            return None
        if fields is None:
            return None
        payload = build_candidate_inspection_observation(
            candidate_uid=self.args.stand_id,
            stream_id=self.args.stream_id,
            planning_frame=self.profile.map_frame,
            stand_center={"x_m": self.args.stand_x, "y_m": self.args.stand_y},
            robot_profile_sha256=real_robot_profile_sha256(self.profile),
            calibration_profile_sha256=camera_calibration_sha256(self.calibration),
            stand_model_profile_sha256=self.stand_model_profile.sha256,
            camera_framing=getattr(self, "_camera_framing", None),
            front_view_recovery=(None if recovery is None else recovery.metadata(now_sec=time.monotonic())),
            **fields,
        )
        if not self._commit_sensor_artifact(
            output, payload, image_stamp_sec=current["frame_stamp_sec"],
            scan_stamp_sec=scan_stamp_sec, artifact_kind="inspection_observation",
        ):
            return None
        self.completed = True
        return payload

    def _write_status(self, state: str, **details) -> None:
        bounded = commit_bounded_head(self)
        if bounded is not None:
            state, bounded_details = bounded
            details = {**details, **bounded_details}
        elif not getattr(self, "completed", False):
            current = getattr(self, "_pending_bounded_head", None)
            rejection = None if current is None else current.metadata.get("bounded_orientation_rejection")
            if rejection is not None and state == "collecting_consensus":
                # A ready interval with no feasible viewing pose is not an
                # invitation to wait forever or publish a legacy point angle.
                state = "evidence_not_committable"
                details = {**details, "reason": rejection}
        progress = self._maybe_commit_inspection_progress(state, details)
        if progress is not None:
            details = {
                **details,
                "inspection_observation": str(self.args.inspection_observation_json),
                "inspection_classification": progress["classification"],
                "inspection_observation_sha256": progress["inspection_observation_sha256"],
                "preceding_state": state,
            }
            state = "inspection_progress_committed"
        self._capture_camera_outcome(state, details)
        observation_evidence = getattr(self, "observation_evidence", None)
        stand_model = getattr(self, "stand_model_profile", None)
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
            compatibility_consensus = getattr(self, "consensus", None)
            compatibility_sample_count = getattr(
                compatibility_consensus,
                "sample_count",
                0,
            )
            consensus_status = {
                "sample_count": compatibility_sample_count,
                "peak_sample_count": compatibility_sample_count,
                "required_sample_count": getattr(
                    compatibility_consensus,
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
            "stand_model": (
                None
                if stand_model is None
                else {
                    "mode": "metric_model_only",
                    "profile_id": stand_model.profile_id,
                    "profile_sha256": stand_model.sha256,
                    "environment": stand_model.environment,
                    "measurement_status": stand_model.measurement_status,
                }
            ),
            "axis_consensus": consensus_status,
            "observation_evidence": observation_status,
            "camera_pipeline_counts": dict(getattr(self, "_camera_pipeline_counters", {})),
            "camera_framing": getattr(self, "_camera_framing", None),
            "front_view_recovery": (
                None if getattr(self, "_front_view_recovery", None) is None
                else self._front_view_recovery.metadata(now_sec=time.monotonic())
            ),
            "last_evidence_source_freshness": getattr(self, "_last_evidence_source_freshness", None),
            "publication_freshness": getattr(self, "_last_camera_publication_freshness", None),
            "capture_history": self._capture_snapshot(),
            "capture_error": getattr(self, "_capture_error", None),
            "tf_retry": asdict(self.tf_retry_scheduler.evidence),
            "tf_delivery": (None if getattr(self, "_tf_delivery_trace", None) is None
                            else self._tf_delivery_trace.snapshot()),
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
    parser.add_argument(
        "--stand-model-profile",
        required=True,
        type=Path,
        help=(
            "Content-hashed measured physical stand model. The operational "
            "observer has no legacy image-detector fallback."
        ),
    )
    parser.add_argument(
        "--stand-head-center-height-m",
        type=float,
        default=None,
        help=(
            "Optional consistency assertion. Operational projection always "
            "uses the head centre derived from --stand-model-profile."
        ),
    )
    parser.add_argument("--target-distance-m", type=float, default=0.33)
    parser.add_argument("--head-roi-padding-scale", type=float, default=1.8)
    parser.add_argument(
        "--backside-reacquisition-padding-scale",
        type=float,
        default=DEFAULT_BACKSIDE_REACQUISITION_PADDING_SCALE,
        help=(
            "Target-centred expanded ROI padding used after the nominal "
            "QR/model or no-QR backside acquisition cannot produce strict "
            "evidence."
        ),
    )
    parser.add_argument(
        "--disable-backside-reacquisition",
        action="store_true",
        help=(
            "Disable the bounded QR/model and backside expanded-ROI retry and "
            "use only the nominal projected head crop."
        ),
    )
    parser.add_argument(
        "--backside-registration-max-center-offset-ratio",
        type=float,
        default=DEFAULT_BACKSIDE_REGISTRATION_MAX_CENTER_OFFSET_RATIO,
        help=(
            "Maximum projected-to-detected head-centre displacement, in "
            "expected head heights, for the proposal-only registration path. "
            "The final metric pass still uses the strict receipt gate."
        ),
    )
    parser.add_argument(
        "--backside-registration-max-bearing-delta-deg",
        type=float,
        default=MAX_CAMERA_MAP_BEARING_DELTA_DEG,
        help=(
            "Maximum map-to-camera bearing correction for shifting the same "
            "narrow LiDAR cone after bounded visual registration."
        ),
    )
    parser.add_argument(
        "--qr-bound-model-max-obliqueness-deg",
        type=float,
        default=DEFAULT_QR_BOUND_MODEL_MAX_OBLIQUENESS_DEG,
        help=(
            "Maximum obliqueness for accepting a QR-decoded, joint QR/head "
            "measured-model pose as an axis sample. This does not relax the "
            "generic silhouette gate."
        ),
    )
    parser.add_argument("--min-head-size-px", type=float, default=18.0)
    parser.add_argument(
        "--sync-tolerance-sec",
        type=float,
        default=DEFAULT_MAX_IMAGE_SCAN_SKEW_SEC,
    )
    parser.add_argument(
        "--camera-info-tolerance-sec",
        type=float,
        default=DEFAULT_MAX_CAMERA_INFO_IMAGE_SKEW_SEC,
    )
    parser.add_argument(
        "--max-sensor-age-sec",
        type=float,
        default=DEFAULT_MAX_SENSOR_AGE_SEC,
    )
    parser.add_argument(
        "--max-future-timestamp-sec",
        type=float,
        default=DEFAULT_MAX_FUTURE_TIMESTAMP_SEC,
    )
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
    parser.add_argument("--scan-topology-profile", choices=("linear", "full_rotation"), default="linear")
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
    parser.add_argument("--inspection-observation-json", type=Path, default=None)
    parser.add_argument("--inspection-progress-frames", type=int, default=7)
    parser.add_argument("--inspection-progress-min-span-sec", type=float, default=2.0)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--capture-history-dir", type=Path, default=None)
    parser.add_argument("--capture-max-frames", type=int, default=64)
    parser.add_argument("--capture-max-bytes", type=int, default=33554432)
    parser.add_argument("--once", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.capture_max_frames <= 0 or args.capture_max_bytes <= 0:
        parser.error("capture frame and byte limits must be positive")
    try:
        stand_model = load_measured_physical_stand_model(
            args.stand_model_profile
        )
        args.stand_head_center_height_m = resolve_head_center_height_m(
            stand_model,
            args.stand_head_center_height_m,
        )
    except (OSError, ValueError) as exc:
        parser.error(f"invalid stand model profile: {exc}")
    positive = {
        "--stand-radius-m": args.stand_radius_m,
        "--stand-head-center-height-m": args.stand_head_center_height_m,
        "--target-distance-m": args.target_distance_m,
        "--head-roi-padding-scale": args.head_roi_padding_scale,
        "--backside-reacquisition-padding-scale": (
            args.backside_reacquisition_padding_scale
        ),
        "--backside-registration-max-center-offset-ratio": (
            args.backside_registration_max_center_offset_ratio
        ),
        "--backside-registration-max-bearing-delta-deg": (
            args.backside_registration_max_bearing_delta_deg
        ),
        "--qr-bound-model-max-obliqueness-deg": (
            args.qr_bound_model_max_obliqueness_deg
        ),
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
    if (
        args.backside_reacquisition_padding_scale
        > MAX_BACKSIDE_REACQUISITION_PADDING_SCALE
    ):
        parser.error(
            "--backside-reacquisition-padding-scale must be no greater than "
            f"{MAX_BACKSIDE_REACQUISITION_PADDING_SCALE}"
        )
    if (
        args.backside_registration_max_center_offset_ratio
        > MAX_BACKSIDE_REGISTRATION_CENTER_OFFSET_RATIO
    ):
        parser.error(
            "--backside-registration-max-center-offset-ratio must be no "
            f"greater than {MAX_BACKSIDE_REGISTRATION_CENTER_OFFSET_RATIO}"
        )
    try:
        normalize_certified_camera_map_bearing_limit(
            math.radians(
                args.backside_registration_max_bearing_delta_deg
            )
        )
    except ValueError as exc:
        parser.error(
            "invalid --backside-registration-max-bearing-delta-deg: "
            f"{exc}"
        )
    try:
        normalize_qr_bound_model_obliqueness_limit(
            math.radians(args.qr_bound_model_max_obliqueness_deg),
            generic_max_obliqueness_rad=math.radians(
                args.max_obliqueness_deg
            ),
        )
    except ValueError as exc:
        parser.error(
            "invalid --qr-bound-model-max-obliqueness-deg: "
            f"{exc}"
        )
    if args.stand_uncertainty_m < 0.0:
        parser.error("--stand-uncertainty-m must be non-negative")
    if args.consensus_frames < 2:
        parser.error("--consensus-frames must be at least two")
    try:
        StationaryHeadConsistency(
            required_samples=args.consensus_frames,
            max_axis_span_rad=math.radians(args.consensus_max_deviation_deg),
            window_ttl_sec=args.consensus_axis_ttl_sec)
    except ValueError as exc:
        parser.error(f"invalid head consistency configuration: {exc}")
    if args.inspection_progress_frames < 7:
        parser.error("--inspection-progress-frames must be at least seven")
    if (not math.isfinite(args.inspection_progress_min_span_sec)
            or not 2.0 <= args.inspection_progress_min_span_sec <= INSPECTION_PROGRESS_WINDOW_SEC):
        parser.error("--inspection-progress-min-span-sec must be between two and 15 seconds")
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
    if args.status_json.resolve() == args.recommended_pose_json.resolve():
        parser.error("status and recommendation outputs must be distinct")
    output_paths = [args.status_json.resolve(), args.recommended_pose_json.resolve()]
    if args.status_events_jsonl is not None:
        output_paths.append(args.status_events_jsonl.resolve())
    if args.axis_observation_json is not None:
        output_paths.append(args.axis_observation_json.resolve())
    if args.inspection_observation_json is not None:
        output_paths.append(args.inspection_observation_json.resolve())
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
        from rclpy.executors import SingleThreadedExecutor
    except ImportError as exc:
        parser.exit(2, f"error: ROS 2 Python packages are required: {exc}\n")
    rclpy.init(args=None)
    adapter = None
    ingestion = None
    executor = None
    try:
        adapter = PassiveRealViewpointNode(args)
        executor = SingleThreadedExecutor()
        executor.add_node(adapter.node)
        ingestion = ObserverIngestionLoop(
            spin_once=lambda: executor.spin_once(timeout_sec=0.05),
            wake=executor.wake, ok=rclpy.ok)
        ingestion.start()
        while rclpy.ok() and not (args.once and adapter.completed):
            ingestion.raise_if_failed()
            adapter.process_pending_work()
            ingestion.wait()
        ingestion.raise_if_failed()
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
        # No callback may still reference the node or clock during teardown.
        if ingestion is not None:
            ingestion.close()
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if adapter is not None:
            if getattr(adapter, "capture_history", None) is not None:
                try:
                    adapter.capture_history.close(timeout_sec=2.0)
                except Exception:
                    pass  # Diagnostic cleanup cannot prevent ROS teardown.
            adapter.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
