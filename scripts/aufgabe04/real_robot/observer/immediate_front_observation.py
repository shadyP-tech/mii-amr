"""Join one current viewer-quality head fit to independently decoded identity.

The QR latch only retains identity and its image location. Every admission
still needs a new complete, candidate-associated head fit from current pixels.
The ordinary observer owns sensor, stationary-epoch and identity-conflict gates.
"""

from dataclasses import asdict, dataclass
import math

from scripts.aufgabe04.artifacts.current_head_front_observation import (
    build_current_head_front_evidence, qr_quad_inside_current_head,
)
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import recommendation_to_dict
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.camera_stand_observation import stand_axis_from_camera_yaw
from scripts.aufgabe04.perception.stand_axis.head_model_admission import admit_measured_head_model
from scripts.aufgabe04.qr_scanning.qr_observation import validated_qr_corners
from scripts.aufgabe04.real_robot.configuration.recommendation import build_real_viewpoint_recommendation


@dataclass(frozen=True)
class FrontHeadFrame:
    stamp_sec: float
    scan_stamp_sec: float
    robot_pose: Pose2D
    camera_heading_rad: float
    target_key: str
    camera_signature: tuple
    image_shape: tuple
    estimate: object
    debug: object
    head_admission: object
    head_association: object
    head_corners: tuple
    head_accepted: bool
    qr_binding: object
    qr_corners: tuple | None
    observed_qr_texts: tuple
    metadata: dict


@dataclass(frozen=True)
class BoundQrIdentity:
    qr_id: str
    stamp_sec: float
    scan_stamp_sec: float
    checked_at_sec: float
    corners: tuple
    binding: dict


def prepare_immediate_front(*, estimate, debug, association, crop, qr_binding,
                            qr_observations, observed_qr_texts, image_stamp_sec,
                            scan_stamp_sec, robot_pose, camera_heading_rad,
                            target_key, camera_signature, image_shape, roi, metadata):
    """Prepare independent channels; neither a proposal nor a QR supplies yaw."""
    yaw = estimate.yaw_deg
    admission = admit_measured_head_model(
        estimate=estimate, debug=debug,
        yaw_rad=math.radians(yaw) if type(yaw) in (int, float) else math.nan)
    wrapper = None if association is None else association.lidar_association
    cluster = None if wrapper is None else wrapper.search_association
    head_accepted = bool(
        admission.accepted and crop.accepted and association is not None
        and association.accepted and association.head_admission.accepted
        and cluster is not None and cluster.scan_stamp_sec == scan_stamp_sec)
    head_corners = tuple((p.u_px + roi.x0, p.v_px + roi.y0)
                         for p in (estimate.corners or ()))
    observations = tuple(qr_observations or ())
    qr_corners = None
    if (qr_binding.accepted and qr_binding.reason == "decoded_qr_target_associated"
            and qr_binding.symbol_count == 1 and len(observations) == 1
            and qr_binding.qr_texts_for_evidence == (observations[0].text,)):
        corners = validated_qr_corners(observations[0].corners,
            image_shape=(roi.y1 - roi.y0, roi.x1 - roi.x0))
        if corners is not None:
            qr_corners = tuple((u + roi.x0, v + roi.y0) for u, v in corners)
    return FrontHeadFrame(
        image_stamp_sec, scan_stamp_sec, robot_pose, camera_heading_rad,
        target_key, camera_signature, tuple(image_shape[:2]), estimate, debug,
        admission, association, head_corners, head_accepted, qr_binding,
        qr_corners, tuple(observed_qr_texts), metadata)


class ImmediateFrontAdmission:
    """A short identity latch, with no angle accumulation or prediction."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.context = None
        self.identity = None
        self.latest_stamp = -math.inf

    def observe(self, current, *, update, observed_at_sec):
        snapshot = update.snapshot
        context = (snapshot.target_key, snapshot.motion_epoch,
                   current.camera_signature, current.image_shape,
                   current.estimate.model_profile_sha256)
        if context != self.context or update.motion_epoch_reset or snapshot.poisoned:
            self.reset()
            self.context = context
        diagnostic = {"ready": False, "policy": "current_head_and_bound_qr",
                      "required_axis_samples": 1, "required_qr_samples": 1,
                      "motion_authorized": False}
        current.metadata["immediate_front_admission"] = diagnostic

        def reject(reason):
            diagnostic["reason"] = reason
            return None

        if snapshot.poisoned:
            return reject("poisoned_observation_epoch")
        if update.motion_epoch_reset:
            return reject("stationary_epoch_changed_awaiting_stopped_frame")
        if current.target_key != snapshot.target_key or not update.frame_accepted:
            return reject("current_sensor_frame_not_admitted")
        if current.stamp_sec <= self.latest_stamp:
            return reject("duplicate_or_out_of_order_front_frame")
        self.latest_stamp = current.stamp_sec
        ttl = min(1., snapshot.qr_ttl_sec)
        if self.identity is not None and not 0. <= observed_at_sec - self.identity.stamp_sec <= ttl:
            self.identity = None
        if (current.qr_binding.symbol_count > 1 or len(set(current.observed_qr_texts)) > 1
                or self.identity is not None and current.observed_qr_texts
                and set(current.observed_qr_texts) != {self.identity.qr_id}):
            self.identity = None
            return reject("current_qr_identity_conflict")
        if update.qr_sample_accepted and current.qr_corners is not None:
            qr_id = current.qr_binding.qr_texts_for_evidence[0]
            if qr_id not in (snapshot.tentative_qr_id, snapshot.latched_qr_id):
                self.identity = None
                return reject("qr_identity_not_recorded_in_current_epoch")
            self.identity = BoundQrIdentity(
                qr_id, current.stamp_sec, current.scan_stamp_sec, observed_at_sec,
                current.qr_corners, current.qr_binding.metadata())
        if not current.head_accepted:
            return reject("current_complete_associated_head_fit_required")
        identity = self.identity
        if identity is None:
            return reject("decoded_candidate_qr_required")
        if not 0. <= observed_at_sec - identity.stamp_sec <= ttl:
            self.identity = None
            return reject("bound_qr_identity_expired")
        if not qr_quad_inside_current_head(identity.corners, current.head_corners):
            return reject("bound_qr_outside_current_complete_head")
        diagnostic.update(ready=True, reason="current_head_and_bound_qr_ready",
                          qr_id=identity.qr_id, qr_sensor_stamp_sec=identity.stamp_sec,
                          qr_age_sec=observed_at_sec - identity.stamp_sec,
                          axis_sample_count=1)
        return current, identity, update, observed_at_sec


def record_immediate_front(adapter, *, update, image_stamp_sec, observed_at_sec):
    adapter._immediate_front_ready = None
    current = getattr(adapter, "_pending_immediate_front", None)
    if current is None or current.stamp_sec != image_stamp_sec:
        return
    admission = getattr(adapter, "_immediate_front_admission", None)
    if admission is None:
        admission = adapter._immediate_front_admission = ImmediateFrontAdmission()
    adapter._immediate_front_ready = admission.observe(
        current, update=update, observed_at_sec=observed_at_sec)


def commit_immediate_front(adapter):
    ready = getattr(adapter, "_immediate_front_ready", None)
    adapter._immediate_front_ready = None
    if ready is None or getattr(adapter, "completed", False):
        return None
    current, identity, update, checked_at_sec = ready
    args = adapter.args
    ttl = min(1., update.snapshot.qr_ttl_sec)

    def identity_fresh():
        now = adapter.node.get_clock().now().nanoseconds / 1e9
        return 0. <= now - identity.stamp_sec <= ttl

    if not identity_fresh():
        current.metadata["immediate_front_admission"].update(
            ready=False, reason="bound_qr_expired_before_publication")
        return None
    try:
        axis = stand_axis_from_camera_yaw(
            robot_x_m=current.robot_pose.x_m, robot_y_m=current.robot_pose.y_m,
            stand_x_m=args.stand_x, stand_y_m=args.stand_y,
            camera_yaw_rad=math.radians(current.estimate.yaw_deg),
            camera_heading_rad=current.camera_heading_rad)
        evidence = build_current_head_front_evidence(
            head_model_quality=asdict(current.debug.head_model_quality),
            head_admission=current.head_admission.metadata(),
            head_corners_px=current.head_corners, qr_corners_px=identity.corners,
            image_shape=current.image_shape,
            model_profile_sha256=current.estimate.model_profile_sha256,
            sensor_stamp_sec=current.stamp_sec, scan_stamp_sec=current.scan_stamp_sec,
            checked_at_sec=checked_at_sec, qr_sensor_stamp_sec=identity.stamp_sec,
            qr_scan_stamp_sec=identity.scan_stamp_sec, qr_checked_at_sec=identity.checked_at_sec,
            qr_id=identity.qr_id, qr_binding=identity.binding,
            head_lidar_association=asdict(current.head_association.lidar_association),
            camera_yaw_rad=math.radians(current.estimate.yaw_deg),
            camera_heading_rad=current.camera_heading_rad, stand_axis_rad=axis,
            target_key=current.target_key, motion_epoch=update.snapshot.motion_epoch,
            camera_signature=current.camera_signature,
            sample_gate_evidence={key: True for key in (
                "stationary", "synchronized", "lidar_associated", "source_fresh",
                "current_head_boundary", "complete_head", "identity_unambiguous")})
        recommendation = build_real_viewpoint_recommendation(
            stream_id=args.stream_id, stand_id=args.stand_id,
            planning_frame=adapter.profile.map_frame,
            stand_center=Pose2D(args.stand_x, args.stand_y),
            stand_radius_m=args.stand_radius_m, stand_uncertainty_m=args.stand_uncertainty_m,
            robot_pose=current.robot_pose, stand_axis_rad=axis,
            axis_confidence=0., axis_sample_count=1,
            sensor_stamp_sec=current.stamp_sec, expected_qr_id=identity.qr_id,
            observed_qr_ids=(identity.qr_id,), target_distance_m=args.target_distance_m,
            observation_unix_sec=current.stamp_sec, current_head_evidence=evidence)
        payload = recommendation_to_dict(recommendation)
    except (TypeError, ValueError) as exc:
        current.metadata["immediate_front_admission"].update(
            ready=False, reason="current_head_front_receipt_rejected", detail=str(exc))
        return None
    if not adapter._commit_sensor_artifact(
            args.recommended_pose_json, payload, image_stamp_sec=current.stamp_sec,
            scan_stamp_sec=current.scan_stamp_sec, artifact_kind="recommendation",
            additional_check=identity_fresh):
        return None
    adapter.completed = True
    return "recommendation_committed", {
        "recommendation": str(args.recommended_pose_json), "axis_sample_count": 1,
        "axis_confidence": 0., "qr_texts": [identity.qr_id],
        "admission_policy": "current_head_and_bound_qr", "temporal_consensus_claimed": False}
