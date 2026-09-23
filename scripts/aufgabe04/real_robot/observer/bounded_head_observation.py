"""Adapt independent head bounds and face evidence to passive receipts.

The ordinary evidence accumulator still owns freshness, stationary epochs and
QR conflicts. This adapter does not change any strict single-angle verdict.
"""

from dataclasses import dataclass

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import recommendation_to_dict
from scripts.aufgabe04.perception.camera_stand_observation import stand_axis_from_camera_yaw
from scripts.aufgabe04.perception.stand_axis.head_orientation_bounds import validated_current_head_orientation_bounds
from scripts.aufgabe04.real_robot.configuration.profile import camera_calibration_sha256, real_robot_profile_sha256
from scripts.aufgabe04.real_robot.configuration.recommendation import build_real_viewpoint_recommendation
from scripts.aufgabe04.real_robot.observer.backside_axis_observation import build_backside_axis_observation
from scripts.aufgabe04.real_robot.observer.bounded_head_window import BoundedHeadSample, BoundedHeadWindow
from scripts.aufgabe04.real_robot.observer.contract import (
    BACKSIDE_AXIS_SAMPLE_SOURCE, REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE,
    BACKSIDE_MODEL_EVIDENCE_STATE, BACKSIDE_VISIBLE_FACE,
)
from scripts.aufgabe04.real_robot.observer.registration_evidence import build_backside_target_registration_evidence


@dataclass(frozen=True)
class CurrentBoundedHead:
    sample: BoundedHeadSample
    scan_stamp_sec: float
    robot_pose: Pose2D
    proof: object
    appearance: object
    registration: dict | None
    debug: object
    metadata: dict
    target_reconciliation: dict | None
    head_position_evidence: dict | None


def prepare_bounded_head(*, estimate, debug, association, crop, appearance_crop,
                         qr_binding, marker_verified, marker_seen_in_epoch,
                         image_stamp_sec, scan_stamp_sec, robot_pose, camera_heading_rad,
                         stand_x_m, stand_y_m, camera_signature, roi, metadata,
                         projected_center_px, expected_head_height_px,
                         head_position_evidence):
    """Bind this frame's head to its own scan and independently observed face."""
    proof = getattr(debug, "head_orientation_bounds", None)
    if (not validated_current_head_orientation_bounds(proof, estimate=estimate, debug=debug)
            or association is None or association.accepted is not True
            or crop.accepted is not True):
        return None
    wrapper = association.lidar_association
    if (getattr(association, "head_orientation_bounds", None) != proof
            or wrapper is None or wrapper.search_association is None
            or wrapper.search_association.scan_stamp_sec != scan_stamp_sec):
        metadata["bounded_orientation_rejection"] = "current_head_scan_or_bounds_binding_mismatch"
        return None
    texts = qr_binding.qr_texts_for_evidence
    front = (qr_binding.accepted and qr_binding.reason == "decoded_qr_target_associated"
             and qr_binding.symbol_count == 1 and len(texts) == 1 and marker_verified is True)
    appearance = getattr(debug, "head_backside_appearance", None)
    backside = (not marker_seen_in_epoch and not texts and not marker_verified
                and debug.qr_detected is False and debug.qr_marker_verified is False
                and appearance_crop.accepted is True and appearance is not None
                and appearance.accepted is True and appearance.supplies_angle is False)
    if not front and not backside:
        return None
    registration = None
    if backside:
        try:
            registration = build_backside_target_registration_evidence(
                final_head_center_error_ratio=appearance.head_center_error_ratio,
                candidate_lidar_association=wrapper.search_association,
                registered_lidar_association=wrapper,
                current_head_association=association,
                allow_bounded_orientation=True, head_orientation_bounds=proof)
        except (AttributeError, TypeError, ValueError) as exc:
            metadata["bounded_orientation_rejection"] = str(exc)
            return None
    axis = stand_axis_from_camera_yaw(
        robot_x_m=robot_pose.x_m, robot_y_m=robot_pose.y_m,
        stand_x_m=stand_x_m, stand_y_m=stand_y_m,
        camera_yaw_rad=proof.center_rad, camera_heading_rad=camera_heading_rad)
    sample = BoundedHeadSample(
        image_stamp_sec, axis, proof.half_width_rad, estimate.model_profile_sha256,
        camera_signature, "front" if front else "backside", texts[0] if front else None,
        tuple((p.u_px + roi.x0, p.v_px + roi.y0) for p in estimate.corners),
        projected_center_px, expected_head_height_px)
    # Proofs travel with the current sample, independently of the diagnostic
    # dictionary's nesting. Only the accepted association can supply identity.
    return CurrentBoundedHead(sample, scan_stamp_sec, robot_pose, proof, appearance,
                              registration, debug, metadata,
                              target_reconciliation=association.target_reconciliation,
                              head_position_evidence=head_position_evidence)


def record_bounded_head(adapter, *, update, image_stamp_sec, observed_at_sec):
    """Run after the ordinary gate and independent repeated appearance update."""
    adapter._bounded_head_ready = None
    current = getattr(adapter, "_pending_bounded_head", None)
    if current is not None and current.sample.stamp_sec != image_stamp_sec:
        current = None
    window = getattr(adapter, "_bounded_head_window", None)
    if current is None and window is None:
        return
    if window is None:
        count = max(7, update.snapshot.required_axis_sample_count)
        if count > 32:
            return
        window = adapter._bounded_head_window = BoundedHeadWindow(
            required_samples=count, ttl_sec=update.snapshot.axis_ttl_sec)
    bounds = window.observe(None if current is None else current.sample,
                            update=update, observed_at_sec=observed_at_sec)
    if current is None:
        return
    current.metadata["bounded_orientation_window"] = dict(window.metadata)
    if bounds is None:
        return
    if current.sample.face == "front":
        if update.resolved_qr_id != current.sample.qr_id:
            return
    else:
        backside = (getattr(adapter, "_head_confidence_metadata", None) or {}).get("backside", {})
        if (backside.get("state") != "backside_supported"
                or backside.get("current_sample_accepted") is not True):
            return
    adapter._bounded_head_ready = (current, bounds, update)


def commit_bounded_head(adapter):
    """Try the distinct interval receipt before a weaker advisory exit."""
    ready = getattr(adapter, "_bounded_head_ready", None)
    adapter._bounded_head_ready = None  # A later status cannot reuse this frame.
    if ready is None or getattr(adapter, "completed", False):
        return None
    current, bounds, update = ready
    sample, args = current.sample, adapter.args
    if update.snapshot.poisoned:
        return None
    try:
        if sample.face == "front":
            output = args.recommended_pose_json
            recommendation = build_real_viewpoint_recommendation(
                stream_id=args.stream_id, stand_id=args.stand_id,
                planning_frame=adapter.profile.map_frame,
                stand_center=Pose2D(args.stand_x, args.stand_y),
                stand_radius_m=args.stand_radius_m, stand_uncertainty_m=args.stand_uncertainty_m,
                robot_pose=current.robot_pose, stand_axis_rad=bounds["center_rad"],
                axis_confidence=0., axis_sample_count=bounds["sample_count"],
                sensor_stamp_sec=sample.stamp_sec, expected_qr_id=sample.qr_id,
                observed_qr_ids=(sample.qr_id,), target_distance_m=args.target_distance_m,
                observation_unix_sec=sample.stamp_sec, bounded_orientation=bounds)
            payload = recommendation_to_dict(recommendation)
            state, kind = "recommendation_committed", "recommendation"
        else:
            output = args.axis_observation_json
            if output is None:
                return None
            snapshot, appearance = update.snapshot, current.appearance
            registration = current.registration
            confidence = adapter._head_confidence_metadata["backside"]["confidence"]
            payload = build_backside_axis_observation(
                target_reconciliation=current.target_reconciliation,
                head_position_evidence=current.head_position_evidence,
                stream_id=args.stream_id, stand_id=args.stand_id,
                planning_frame=adapter.profile.map_frame, stand_x_m=args.stand_x, stand_y_m=args.stand_y,
                robot_x_m=current.robot_pose.x_m, robot_y_m=current.robot_pose.y_m,
                robot_yaw_rad=current.robot_pose.yaw_rad, stand_axis_rad=bounds["center_rad"],
                axis_confidence=0., axis_sample_count=bounds["sample_count"],
                consensus_source=(REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE
                    if registration["mode"] == "bounded_camera_lidar_registration" else BACKSIDE_AXIS_SAMPLE_SOURCE),
                estimate_source=BACKSIDE_AXIS_SAMPLE_SOURCE,
                estimate_evidence_state=BACKSIDE_MODEL_EVIDENCE_STATE,
                estimate_visible_face=BACKSIDE_VISIBLE_FACE, visible_face_confidence=confidence,
                debug_qr_detected=False, qr_texts=(),
                evidence_qr_sample_count=snapshot.current_qr_sample_count,
                evidence_tentative_qr_id=snapshot.tentative_qr_id,
                evidence_latched_qr_id=snapshot.latched_qr_id,
                qr_marker_seen_in_stationary_epoch=adapter._qr_marker_seen_in_stationary_epoch,
                all_samples_stationary=True, all_samples_synchronized=True,
                all_samples_lidar_associated=True, sensor_stamp_sec=sample.stamp_sec,
                stand_model_profile_sha256=sample.model_sha256,
                stand_model_measurement_status="measured", head_scale_ratio=appearance.head_scale_ratio,
                head_center_error_ratio=appearance.head_center_error_ratio,
                pose_reprojection_rmse_px=current.debug.pose_reprojection_rmse_px,
                pose_ambiguity_gap_px=current.debug.pose_ambiguity_gap_px,
                robot_profile_sha256=real_robot_profile_sha256(adapter.profile),
                calibration_profile_sha256=camera_calibration_sha256(adapter.calibration),
                target_registration=registration, bounded_orientation=bounds)
            state, kind = "backside_axis_committed_qr_unresolved", "backside_axis_observation"
        payload["axis_measurement"] = {
            "source": "current_measured_head_bounded_orientation",
            "model_profile_sha256": sample.model_sha256,
            "model_measurement_status": "measured", "sensor_stamp_sec": sample.stamp_sec,
            "single_angle_confidence_claimed": False,
            "sample_gate_evidence": {key: True for key in (
                "all_samples_stationary", "all_samples_synchronized",
                "all_samples_lidar_associated", "all_samples_current_frame_model_geometry",
                "all_samples_fresh", "all_samples_complete_head",
                "all_samples_temporally_consistent_borders")},
            "bounded_orientation_window": dict(adapter._bounded_head_window.metadata),
        }
    except (TypeError, ValueError) as exc:
        current.metadata["bounded_orientation_rejection"] = str(exc)
        return None
    if not adapter._commit_sensor_artifact(output, payload, image_stamp_sec=sample.stamp_sec,
                                          scan_stamp_sec=current.scan_stamp_sec, artifact_kind=kind):
        return None
    adapter.completed = True
    if sample.face == "backside":
        adapter.axis_observation_committed = True
    return state, {kind: str(output), "axis_sample_count": bounds["sample_count"],
                   "bounded_orientation": bounds, "single_angle_confidence_claimed": False}
