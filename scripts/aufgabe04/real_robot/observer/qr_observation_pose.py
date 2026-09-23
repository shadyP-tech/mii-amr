"""Current QR-only discovery, optionally retaining a certified backside angle.

This path does not estimate a stand angle or authorize a facing approach. Its
optional same-stop grace requires a new, source-fresh associated decode after
that delay; by default one current decode suffices. Old identity latches cannot
substitute for a current decode. The certified opposite branch binds text by an
exclusive current crop and finishes immediately without corners or a new fit.
"""

from dataclasses import asdict, dataclass
import math
import time

from scripts.aufgabe04.artifacts.qr_verified_observation_pose import (
    SOURCE_GATES, build_qr_verified_observation_pose,
)
from scripts.aufgabe04.qr_scanning.qr_observation import validated_qr_corners
from scripts.aufgabe04.real_robot.configuration.profile import (
    camera_calibration_sha256, real_robot_profile_sha256,
)


@dataclass(frozen=True)
class QrObservationFrame:
    stamp_sec: float
    scan_stamp_sec: float
    robot_pose: object
    target_key: str
    camera_signature: tuple
    image_shape: tuple
    qr_binding: object
    qr_corners: tuple | None
    observed_qr_texts: tuple
    model_profile_sha256: str
    metadata: dict
    retained_backside_orientation: dict | None = None


def prepare_qr_observation_pose(*, qr_binding, qr_observations, observed_qr_texts,
        image_stamp_sec, scan_stamp_sec, robot_pose, target_key, camera_signature,
        image_shape, roi, model_profile_sha256, metadata, retained_backside_orientation=None):
    observations = tuple(qr_observations or ())
    corners = None
    if (qr_binding.accepted and qr_binding.reason == "decoded_qr_target_associated"
            and qr_binding.symbol_count == 1 and len(observations) == 1
            and qr_binding.qr_texts_for_evidence == (observations[0].text,)):
        local = validated_qr_corners(observations[0].corners,
            image_shape=(roi.y1 - roi.y0, roi.x1 - roi.x0))
        if local is not None:
            corners = tuple((x + roi.x0, y + roi.y0) for x, y in local)
    return QrObservationFrame(image_stamp_sec, scan_stamp_sec, robot_pose,
        target_key, tuple(camera_signature), tuple(image_shape[:2]), qr_binding,
        corners, tuple(observed_qr_texts), model_profile_sha256, metadata, retained_backside_orientation)


class QrObservationPoseFallback:
    def __init__(self, *, delay_sec=0.):
        if type(delay_sec) not in (int, float) or not math.isfinite(delay_sec) or not 0 <= delay_sec <= 10.:
            raise ValueError("QR observation fallback delay must be between zero and ten seconds")
        self.delay_sec = float(delay_sec)
        self.reset()

    def reset(self):
        self.context = None
        self.first_checked_sec = None
        self.first_monotonic_sec = None
        self.qr_id = None
        self.latest_stamp = -math.inf
        self.poisoned = False
        self.effective_delay_sec = self.delay_sec

    def grace_pending(self, *, now_monotonic_sec):
        return (not self.poisoned and self.first_monotonic_sec is not None
                and 0 <= now_monotonic_sec - self.first_monotonic_sec < self.effective_delay_sec)

    def observe(self, current, *, update, observed_at_sec, now_monotonic_sec):
        snapshot = update.snapshot
        context = (snapshot.target_key, snapshot.motion_epoch,
                   current.camera_signature, current.image_shape, current.model_profile_sha256,
                   None if current.retained_backside_orientation is None else
                   current.retained_backside_orientation.get('projection_sha256'))
        if context != self.context or update.motion_epoch_reset:
            self.reset()
            self.context = context
        diagnostic = {"ready": False, "policy": "qr_verified_observation_pose",
                      "delay_sec": self.delay_sec, "stand_axis_rad": None,
                      "facing_ready": False, "motion_authorized": False}
        if current.retained_backside_orientation is not None:
            diagnostic.update(delay_sec=0., stand_axis_rad=current.retained_backside_orientation['stand_axis_rad'],
                orientation_source='certified_backside', current_angle_refit=False)
        current.metadata["qr_observation_pose_fallback"] = diagnostic

        def reject(reason):
            diagnostic["reason"] = reason
            return None

        if (snapshot.poisoned or current.qr_binding.symbol_count > 1
                or len(set(current.observed_qr_texts)) > 1
                or self.qr_id is not None and current.observed_qr_texts
                and set(current.observed_qr_texts) != {self.qr_id}):
            self.poisoned = True
        if self.poisoned:
            return reject("poisoned_observation_epoch")
        if update.motion_epoch_reset:
            return reject("stationary_epoch_changed_awaiting_stopped_frame")
        if not update.frame_accepted or current.target_key != snapshot.target_key:
            return reject("current_sensor_frame_not_admitted")
        if current.stamp_sec <= self.latest_stamp:
            return reject("duplicate_or_out_of_order_qr_frame")
        self.latest_stamp = current.stamp_sec
        retained = current.retained_backside_orientation is not None
        crop_bound = (retained and current.qr_binding.accepted
                      and current.qr_binding.reason == "decoded_qr_exclusive_opposite_crop")
        if not update.qr_sample_accepted or (current.qr_corners is None and not crop_bound):
            return reject("fresh_independently_bound_qr_required")
        qr_id = current.qr_binding.qr_texts_for_evidence[0]
        if qr_id not in (snapshot.tentative_qr_id, snapshot.latched_qr_id):
            return reject("qr_identity_not_recorded_in_current_epoch")
        if self.first_checked_sec is None:
            self.first_checked_sec = observed_at_sec
            self.first_monotonic_sec = now_monotonic_sec
            self.qr_id = qr_id
        # Both sensor clock and monotonic time must cover the same-stop grace.
        # Neither a timestamp leap nor delayed processing alone completes it.
        delay = 0. if crop_bound or current.qr_binding.target_reconciliation is not None else self.delay_sec
        self.effective_delay_sec = delay
        diagnostic["delay_sec"] = delay
        if crop_bound:
            diagnostic.update(delay_sec=0., stand_axis_rad=current.retained_backside_orientation['stand_axis_rad'],
                              orientation_source='certified_backside', current_angle_refit=False)
        if (observed_at_sec - self.first_checked_sec + 1e-9 < delay
                or now_monotonic_sec - self.first_monotonic_sec + 1e-9 < delay):
            return reject("same_pose_geometry_grace_pending")
        diagnostic.update(ready=True, reason="fresh_qr_observation_pose_ready", qr_id=qr_id)
        return current, update, observed_at_sec


def record_qr_observation_pose(adapter, *, update, image_stamp_sec, observed_at_sec):
    adapter._qr_observation_pose_ready = None
    if getattr(adapter.args, "qr_observation_pose_json", None) is None:
        return
    current = getattr(adapter, "_pending_qr_observation_pose", None)
    if current is None or current.stamp_sec != image_stamp_sec:
        return
    fallback = getattr(adapter, "_qr_observation_pose_fallback", None)
    if fallback is None:
        fallback = adapter._qr_observation_pose_fallback = QrObservationPoseFallback(
            delay_sec=getattr(adapter.args, "qr_pose_fallback_delay_sec", 0.))
    adapter._qr_observation_pose_ready = fallback.observe(
        current, update=update, observed_at_sec=observed_at_sec, now_monotonic_sec=time.monotonic())


def qr_observation_grace_pending(adapter):
    fallback = getattr(adapter, "_qr_observation_pose_fallback", None)
    return fallback is not None and fallback.grace_pending(now_monotonic_sec=time.monotonic())


def commit_qr_observation_pose(adapter):
    ready = getattr(adapter, "_qr_observation_pose_ready", None)
    adapter._qr_observation_pose_ready = None
    output = getattr(adapter.args, "qr_observation_pose_json", None)
    if ready is None or output is None or getattr(adapter, "completed", False):
        return None
    current, update, checked_at_sec = ready
    args, profile = adapter.args, adapter.profile
    qr_id = current.qr_binding.qr_texts_for_evidence[0]
    try:
        payload = build_qr_verified_observation_pose(
            candidate_uid=args.stand_id, stream_id=args.stream_id, qr_id=qr_id,
            planning_frame=profile.map_frame,
            stand_center={"x_m": args.stand_x, "y_m": args.stand_y},
            robot_pose=asdict(current.robot_pose),
            sensor_stamp_sec=current.stamp_sec, scan_stamp_sec=current.scan_stamp_sec,
            checked_at_sec=checked_at_sec,
            robot_profile_sha256=real_robot_profile_sha256(profile),
            calibration_profile_sha256=camera_calibration_sha256(adapter.calibration),
            stand_model_profile_sha256=current.model_profile_sha256,
            target_key=current.target_key, motion_epoch=update.snapshot.motion_epoch,
            camera_signature=current.camera_signature, qr_corners_px=current.qr_corners,
            image_shape=current.image_shape, qr_binding=current.qr_binding.metadata(),
            **({} if current.retained_backside_orientation is None else
               {"retained_backside_orientation": current.retained_backside_orientation}),
            source_gates={key: True for key in SOURCE_GATES},
            localization_provenance={"map_frame": profile.map_frame, "base_frame": profile.base_frame,
                "scan_frame": profile.scan_frame, "camera_frame": profile.camera_optical_frame,
                "exact_image_transform_stamp_sec": current.stamp_sec,
                "exact_scan_transform_stamp_sec": current.scan_stamp_sec})
    except (OSError, TypeError, ValueError) as exc:
        current.metadata["qr_observation_pose_fallback"].update(
            ready=False, reason="qr_observation_receipt_rejected", detail=str(exc))
        return None
    if not adapter._commit_sensor_artifact(output, payload,
            image_stamp_sec=current.stamp_sec, scan_stamp_sec=current.scan_stamp_sec,
            artifact_kind="qr_verified_observation_pose"):
        return None
    adapter.completed = True
    return "qr_observation_pose_committed", {
        "qr_observation_pose": str(output),
        "qr_verified_observation_pose_sha256": payload["qr_verified_observation_pose_sha256"],
        "qr_texts": [qr_id], "stand_axis_rad": payload['stand_axis_rad'], "facing_ready": False,
        "completion_scope": "discovery_only", "admission_policy": "qr_verified_observation_pose"}
