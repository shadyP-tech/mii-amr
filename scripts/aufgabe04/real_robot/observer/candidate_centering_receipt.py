"""Publish a current head-centering advisory without admitting angle or motion.

The observer stages the measured head only after the ordinary crop and identity
gates. A successful stationary frame update is a separate prerequisite. The
advisory is consumed by its immediate status publication and never survives a
new sensor tuple or a motion epoch reset.
"""

from dataclasses import dataclass
import time

from scripts.aufgabe04.real_robot.configuration.profile import (
    camera_calibration_sha256, real_robot_profile_sha256,
)
from scripts.aufgabe04.real_robot.observer.candidate_centering import (
    build_camera_centering_advisory,
)
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import (
    qr_observation_grace_pending,
)
from scripts.aufgabe04.real_robot.observer.inspection_framing import (
    ProductiveViewHold, review_centering_destination,
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


def prepare_candidate_centering(*, crop, association, image_stamp_sec,
        scan_stamp_sec, target_key, robot_pose, odom_pose, intrinsics,
        scan_from_camera, base_from_camera, metadata):
    if (not crop.accepted or association is None or not association.accepted
            or odom_pose is None):
        return None
    return CandidateCenteringFrame(image_stamp_sec, scan_stamp_sec, target_key,
        robot_pose, odom_pose, association, intrinsics, scan_from_camera,
        base_from_camera, metadata)


def record_candidate_centering(adapter, *, update, image_stamp_sec, observed_at_sec):
    adapter._candidate_centering_ready = None
    if getattr(adapter.args, "candidate_centering_json", None) is None:
        return
    current = getattr(adapter, "_pending_candidate_centering", None)
    if current is None or current.image_stamp_sec != image_stamp_sec:
        return
    snapshot = update.snapshot
    if (not update.frame_accepted or snapshot.poisoned or update.motion_epoch_reset
            or current.target_key != snapshot.target_key):
        return
    hold = getattr(adapter, "_productive_view_hold", None)
    if hold is None:
        hold = adapter._productive_view_hold = ProductiveViewHold()
    now = time.monotonic()
    hold_pending = hold.observe(context=(snapshot.target_key, snapshot.motion_epoch,
        adapter.stand_model_profile.sha256, current.intrinsics, current.scan_from_camera,
        current.base_from_camera), now_sec=now, axis_sample_accepted=update.axis_sample_accepted)
    current.metadata["productive_view_opportunity"] = hold.metadata(now)
    if hold_pending:
        current.metadata["candidate_centering"] = dict(ready=False,
            reason="preserve_productive_geometry_view", motion_authorized=False)
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
            max_image_scan_skew_sec=adapter.args.sync_tolerance_sec)
    except (TypeError, ValueError, ArithmeticError) as exc:
        current.metadata["candidate_centering"] = {
            "ready": False, "reason": "centering_advisory_rejected", "detail": str(exc)}
        return
    if advisory is None:
        return
    framing = review_centering_destination(advisory,
        search_association=current.association.lidar_association.search_association)
    current.metadata["centering_framing"] = framing.metadata()
    if not framing.allowed:
        current.metadata["candidate_centering"] = dict(ready=False,
            reason=framing.reason, motion_authorized=False)
        return
    current.metadata["candidate_centering"] = advisory.metadata()
    adapter._candidate_centering_ready = current, advisory


def commit_candidate_centering(adapter):
    ready = getattr(adapter, "_candidate_centering_ready", None)
    adapter._candidate_centering_ready = None
    output = getattr(adapter.args, "candidate_centering_json", None)
    if (ready is None or output is None or getattr(adapter, "completed", False)
            or qr_observation_grace_pending(adapter)):
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
