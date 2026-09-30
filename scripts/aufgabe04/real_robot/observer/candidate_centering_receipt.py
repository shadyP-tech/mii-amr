"""Publish current target framing advice without admitting angle or motion.

The observer stages the measured head only after the ordinary crop and identity
gates. A reconciled decoded QR also supports framing when the head is clipped.
Centering is reviewed after stationary frame admission and before precise
axis accumulation. The advisory is consumed by its immediate status publication
and never survives a new sensor tuple or a motion epoch reset.
"""

from dataclasses import dataclass

from scripts.aufgabe04.real_robot.configuration.profile import (
    camera_calibration_sha256, real_robot_profile_sha256,
)
from scripts.aufgabe04.real_robot.observer.candidate_centering import (
    build_camera_centering_advisory,
)
from scripts.aufgabe04.real_robot.observer.inspection_framing import (
    review_centering_destination,
)
from scripts.aufgabe04.real_robot.observer.qr_target_support import (
    prepare_qr_target_support, QrTargetSupport,
)


@dataclass(frozen=True)
class CandidateCenteringFrame:
    image_stamp_sec: float
    scan_stamp_sec: float
    target_key: str
    robot_pose: object
    odom_pose: object
    association: object
    intrinsics: object
    scan_from_camera: object
    base_from_camera: object
    metadata: dict
    allow_advisory: bool = True


def prepare_candidate_centering(*, crop, association, image_stamp_sec,
        scan_stamp_sec, target_key, robot_pose, odom_pose, intrinsics,
        scan_from_camera, base_from_camera, metadata, allow_advisory=True,
        qr_observation=None):
    if odom_pose is None:
        return None
    if not crop.accepted or association is None or not association.accepted:
        # A complete, independently bound QR can locate a clipped head. It
        # supplies framing only; the head and angle admission stay unchanged.
        if (qr_observation is None or qr_observation.stamp_sec != image_stamp_sec
                or qr_observation.scan_stamp_sec != scan_stamp_sec
                or qr_observation.target_key != target_key
                or qr_observation.robot_pose != robot_pose):
            return None
        association = prepare_qr_target_support(qr_observation)
        if association is None:
            return None
    return CandidateCenteringFrame(image_stamp_sec, scan_stamp_sec, target_key,
        robot_pose, odom_pose, association, intrinsics, scan_from_camera,
        base_from_camera, metadata, allow_advisory)


def centering_observation_requested(args):
    # A final post-turn capture still verifies framing after motion is disabled.
    return (getattr(args, "candidate_centering_json", None) is not None
            or getattr(args, "observation_not_before_sec", None) is not None)


def candidate_centering_status(adapter):
    """Report measured framing without treating missing advice as success."""
    if not centering_observation_requested(adapter.args):
        return None
    current = getattr(adapter, "_pending_candidate_centering", None)
    diagnostic = (None if current is None else current.metadata.get("candidate_centering"))
    if diagnostic is None:
        return dict(state="blocked", camera_centered=False,
            reason="fresh_admitted_current_head_required", motion_authorized=False)
    result = dict(diagnostic)
    if not adapter._source_freshness(current.image_stamp_sec, current.scan_stamp_sec).accepted:
        result.update(state="blocked", camera_centered=False, ready=False,
            reason="centering_observation_expired_before_publication")
    return result


def review_candidate_centering(adapter, *, snapshot, image_stamp_sec,
        observed_at_sec, motion_epoch_reset=False):
    """Review admitted current framing without waiting for an axis sample."""
    adapter._candidate_centering_ready = None
    if not centering_observation_requested(adapter.args):
        return
    current = getattr(adapter, "_pending_candidate_centering", None)
    if current is None or current.image_stamp_sec != image_stamp_sec:
        return
    if (snapshot.poisoned or motion_epoch_reset
            or current.target_key != snapshot.target_key):
        return
    try:
        advisory = build_camera_centering_advisory(
            association=current.association, intrinsics=current.intrinsics,
            scan_from_camera=current.scan_from_camera,
            base_from_camera=current.base_from_camera,
            candidate_uid=adapter.args.stand_id, target_key=current.target_key,
            motion_epoch=snapshot.motion_epoch,
            anchor_pose=adapter._evidence_pose(current.robot_pose),
            anchor_odom_pose=adapter._evidence_pose(current.odom_pose),
            odom_stamp_sec=current.image_stamp_sec,
            planning_frame=adapter.profile.map_frame, stream_id=adapter.args.stream_id,
            image_stamp_sec=current.image_stamp_sec, now_sec=observed_at_sec,
            robot_profile_sha256=real_robot_profile_sha256(adapter.profile),
            calibration_profile_sha256=camera_calibration_sha256(adapter.calibration),
            stand_model_profile_sha256=adapter.stand_model_profile.sha256,
            max_age_sec=adapter.args.max_sensor_age_sec,
            max_image_scan_skew_sec=adapter.args.sync_tolerance_sec,
            diagnostics=current.metadata.setdefault("candidate_centering", {}))
    except (TypeError, ValueError, ArithmeticError) as exc:
        current.metadata["candidate_centering"] = {
            "state": "blocked", "camera_centered": False, "ready": False,
            "reason": "centering_advisory_rejected", "detail": str(exc)}
        return
    if advisory is None:
        return
    diagnostic = current.metadata["candidate_centering"]
    qr_support = isinstance(current.association, QrTargetSupport)
    if not current.allow_advisory:
        diagnostic.update(state="deferred", reason="preserve_retained_geometry_qr_completion")
        return
    if getattr(adapter.args, "candidate_centering_json", None) is None:
        diagnostic.update(state="blocked", reason="centering_motion_disabled_for_capture")
        return
    framing = review_centering_destination(advisory,
        search_association=current.association.lidar_association.search_association)
    current.metadata["centering_framing"] = framing.metadata()
    if not framing.allowed:
        diagnostic.update(state="blocked", reason=framing.reason)
        return
    current.metadata["candidate_centering"] = {**advisory.metadata(),
        "state": "correction_required", "ready": True, "camera_centered": False,
        "reason": ("fresh_reconciled_qr_off_center" if qr_support
                   else "fresh_current_head_off_center")}
    adapter._candidate_centering_ready = current, advisory


def commit_candidate_centering(adapter):
    ready = getattr(adapter, "_candidate_centering_ready", None)
    adapter._candidate_centering_ready = None
    output = getattr(adapter.args, "candidate_centering_json", None)
    if ready is None or output is None or getattr(adapter, "completed", False):
        return None
    current, advisory = ready
    payload = advisory.metadata()
    if not adapter._commit_sensor_artifact(
            output, payload, image_stamp_sec=current.image_stamp_sec,
            scan_stamp_sec=current.scan_stamp_sec, artifact_kind="candidate_centering"):
        return None
    adapter.completed = True
    return "candidate_centering_committed", {
        "candidate_centering": str(output),
        "camera_centering_advisory_sha256": payload["camera_centering_advisory_sha256"],
        "stand_axis_rad": None, "facing_ready": False,
        "completion_scope": "candidate_centering_advisory", "motion_authorized": False}
