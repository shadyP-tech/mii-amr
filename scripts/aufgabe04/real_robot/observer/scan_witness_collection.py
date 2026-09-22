"""Collect source-fresh scan witnesses independently of camera acquisition.

Exact-time TF lookup is injected by the ROS adapter. This helper does no robot
I/O and never substitutes the camera image's pose for a scan's own pose.
"""

import math

from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.configuration.geometry import pose2d_from_transform
from scripts.aufgabe04.real_robot.configuration.profile import transform_mismatches
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import (
    ScanPersistenceContext, scan_pose_in_map, scan_pose_from_camera_extrinsics,
)


def _transform_values(transform):
    t, r = transform.transform.translation, transform.transform.rotation
    return ((float(t.x), float(t.y), float(t.z)),
            (float(r.x), float(r.y), float(r.z), float(r.w)))


def plain_scan_from_sample(sample, *, topology_profile):
    message = sample.value
    return PlainLaserScan(
        ranges=tuple(float(value) for value in message.ranges),
        angle_min=float(message.angle_min), angle_increment=float(message.angle_increment),
        range_min=float(message.range_min), range_max=float(message.range_max),
        scan_frame_id=str(message.header.frame_id), scan_stamp_sec=sample.stamp_sec,
        receipt_sec=(sample.received_ros_sec if getattr(sample, "received_ros_sec", None) is not None
                     else sample.stamp_sec),
        angle_max=getattr(message, "angle_max", None),
        scan_topology_profile=topology_profile)


def collect_pending_scan_witnesses(pending, persistence, *, now_sec, lookup,
                                   args, profile, calibration, target_key, epoch_key,
                                   transform_error, count):
    """Drain in source order; False asks the owner to clear invalid context."""
    # The camera owner drains callback receipts independently of successful
    # fitting. Each witness owns its exact scan-time pose; TF ingestion can
    # continue while the owner is processing an image.
    for _ in range(len(pending)):
        sample = pending.popleft()
        now = now_sec()
        if not -args.max_future_timestamp_sec <= now - sample.stamp_sec <= args.max_sensor_age_sec:
            count("scan_witness_expired_before_tf")
            persistence.note_collection("tf_expired", stamp=sample.stamp_sec,
                                       reason="source_age_exceeded_before_exact_tf")
            # An unobserved intervening scan cannot silently preserve a chain
            # of consecutive unique witnesses across unavailable localization.
            persistence.reset()
            continue
        if str(sample.value.header.frame_id).strip("/") != profile.scan_frame:
            count("scan_witness_context_rejected")
            persistence.note_collection("context_rejected", stamp=sample.stamp_sec, reason="scan_frame_mismatch")
            return False
        try:
            stamp = sample.value.header.stamp
            scan_from_map = lookup(profile.scan_frame, profile.map_frame, stamp)
            map_from_base = lookup(profile.map_frame, profile.base_frame, stamp)
            base_from_camera = lookup(profile.base_frame, profile.camera_optical_frame)
            scan_from_camera = lookup(profile.scan_frame, profile.camera_optical_frame)
        except transform_error:
            # Preserve source order while exact TF catches up; newer scans
            # cannot advance the history past an unresolved earlier source.
            pending.appendleft(sample)
            count("scan_witness_tf_pending")
            persistence.note_collection("tf_pending", stamp=sample.stamp_sec, reason="exact_scan_tf_unavailable")
            return True
        if transform_mismatches(
                calibration.base_to_camera, base_from_camera,
                translation_tolerance_m=args.extrinsic_translation_tolerance_m,
                rotation_tolerance_rad=math.radians(args.extrinsic_rotation_tolerance_deg)):
            count("scan_witness_context_rejected")
            persistence.note_collection("context_rejected", stamp=sample.stamp_sec, reason="camera_extrinsic_mismatch")
            return False
        try:
            context = ScanPersistenceContext(
                target_key=target_key,
                epoch_key=epoch_key,
                robot_pose=pose2d_from_transform(map_from_base),
                scan_pose_map=scan_pose_in_map(*_transform_values(scan_from_map)),
                image_stamp_sec=sample.stamp_sec,
                candidate_x_m=args.stand_x, candidate_y_m=args.stand_y,
                stand_radius_m=args.stand_radius_m,
                stand_uncertainty_m=args.stand_uncertainty_m,
                lidar_range_tolerance_m=args.lidar_range_tolerance_m,
                scan_pose_robot=scan_pose_from_camera_extrinsics(
                    *_transform_values(base_from_camera), *_transform_values(scan_from_camera)))
            ingested = persistence.ingest_scan(
                plain_scan_from_sample(sample, topology_profile=getattr(args, "scan_topology_profile", "linear")), context=context,
                now_sec=now_sec(),
                max_scan_age_sec=args.max_sensor_age_sec)
            count("scan_witness_ingested" if ingested else "scan_witness_ingestion_rejected")
        except (TypeError, ValueError, ArithmeticError) as exc:
            count("scan_witness_context_rejected")
            persistence.note_collection("context_rejected", stamp=sample.stamp_sec, reason=str(exc))
            return False

    return True
