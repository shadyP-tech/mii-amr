"""Current-scan QR search and registration, independent of physical head fitting.

The broad envelope is only a uniqueness/search check. Ordinary admission needs
the decoded symbol's own ray, the original range and the narrow camera cone.
The certified opposite-side identity path adds exclusive crop validation.
Witnessed fragmentation may establish uniqueness without joining raw clusters.
"""

from dataclasses import asdict
import math

from scripts.aufgabe04.perception.candidate_lidar_association import (
    associate_candidate_lidar_target, normalize_certified_camera_map_bearing_limit,
)
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector, transform_point
from scripts.aufgabe04.real_robot.configuration.geometry import roi_from_projection, project_optical_point
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt
from scripts.aufgabe04.real_robot.observer.shared_scan_cluster import envelope_is_unique, envelope_from_proof


def retained_opposite_qr_search(*, orientation, camera_from_map, intrinsics,
        model_profile, image_stamp_sec, now_sec, max_scan_age_sec):
    """Project the retained target for decoding, without granting association.

    A failed current scan/head proof must not prevent looking at the target.
    This bounded search crop cannot bind either a payload or an image ray.
    """
    if not 0 <= now_sec-image_stamp_sec <= max_scan_age_sec:
        return None
    center = orientation.get('validated_target_center') or orientation['stand_center']
    point = transform_point((center['x_m'], center['y_m'],
        model_profile.head_center_height_m), camera_from_map)
    if point[2] <= 0:
        return None
    projection = project_optical_point(point, intrinsics,
        physical_size_m=max(model_profile.head_width_m, model_profile.head_height_m))
    roi = roi_from_projection(projection, intrinsics, padding_scale=2.4)
    if roi is None:
        return None
    return HeadRoiAttempt(roi, 'retained_opposite_qr_search_only', 2.4,
        projection.u_px, projection.v_px,
        intrinsics.fy_px*model_profile.head_height_m/point[2])


def qr_registration_envelope(scan, *, map_bearing_rad, cone_half_angle_rad,
        max_camera_map_bearing_delta_rad, accepted_range_m, now_sec, max_scan_age_sec,
        fragmentation=None, resolve_lidar_association=None):
    """Count even single-beam competitors over the entire correction envelope."""
    limit = normalize_certified_camera_map_bearing_limit(max_camera_map_bearing_delta_rad)
    raw = associate_candidate_lidar_target(scan, map_bearing_rad=map_bearing_rad,
        cone_half_angle_rad=limit + cone_half_angle_rad, accepted_range_m=accepted_range_m,
        now_sec=now_sec, max_scan_age_sec=max_scan_age_sec, min_cluster_sample_count=1)
    if resolve_lidar_association is not None:
        from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
        registered = associate_camera_registered_candidate_lidar_target(scan,
            map_bearing_rad=map_bearing_rad, observed_camera_bearing_rad=map_bearing_rad,
            cone_half_angle_rad=limit+cone_half_angle_rad, accepted_range_m=accepted_range_m,
            now_sec=now_sec, max_scan_age_sec=max_scan_age_sec, min_cluster_sample_count=1,
            max_camera_map_bearing_delta_rad=limit)
        fragmentation = resolve_lidar_association(registered, scan).witnessed_fragmentation
    if fragmentation is not None:
        return envelope_from_proof(fragmentation, scan=scan, map_bearing_rad=map_bearing_rad,
            cone_half_angle_rad=limit+cone_half_angle_rad, accepted_range_m=accepted_range_m,
            now_sec=now_sec, max_scan_age_sec=max_scan_age_sec)
    return raw


def current_scan_qr_search(*, scan, scan_from_map, camera_from_map, intrinsics,
        model_profile, image_stamp_sec, sync_tolerance_sec, target_reconciliation=None, **association_options):
    """Project a unique current return at measured head height for a small crop.

    This crop supplies neither symbol corners nor a head angle. Exact-time TF,
    freshness, separate identity binding and stationary gates remain mandatory.
    """
    info = dict(policy="current_scan_qr_search", accepted=False, motion_authorized=False,
                supplies_identity=False, supplies_head_geometry=False)
    if (scan is None or scan.scan_stamp_sec is None
            or scan.scan_frame_id != scan_from_map.parent_frame
            or scan_from_map.child_frame != camera_from_map.child_frame):
        return None, {**info, "reason": "search_transform_frame_mismatch"}
    now = association_options["now_sec"]
    age = association_options["max_scan_age_sec"]
    if (not all(math.isfinite(v) for v in (now, image_stamp_sec, sync_tolerance_sec))
            or sync_tolerance_sec <= 0 or not 0 <= now-image_stamp_sec <= age
            or not 0 <= now-scan.scan_stamp_sec <= age
            or abs(image_stamp_sec-scan.scan_stamp_sec) > sync_tolerance_sec):
        return None, {**info, "reason": "stale_or_unsynchronized_search"}
    envelope = qr_registration_envelope(scan, **association_options)
    if target_reconciliation is not None:
        try:
            from scripts.aufgabe04.real_robot.observer.candidate_position_epoch import validated_reconciliation_envelope
            _, envelope, _, _ = validated_reconciliation_envelope(target_reconciliation,
                scan=scan, image_stamp_sec=image_stamp_sec,
                map_bearing_rad=association_options['map_bearing_rad'],
                accepted_range_m=association_options['accepted_range_m'])
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return None, {**info, 'reason': str(exc)}
    info["envelope"] = asdict(envelope)
    if not envelope_is_unique(envelope):
        return None, {**info, "reason": "qr_search_cluster_not_unique"}
    points = tuple((scan.ranges[i]*math.cos(scan.angle_min+i*scan.angle_increment),
                    scan.ranges[i]*math.sin(scan.angle_min+i*scan.angle_increment))
                   for i in envelope.selected_cluster_source_indices)
    x, y = (sum(p[k] for p in points)/len(points) for k in (0, 1))
    tx, ty, tz = scan_from_map.translation_xyz_m
    qx, qy, qz, qw = scan_from_map.rotation_xyzw
    world = rotate_vector((x-tx, y-ty, -tz), (-qx, -qy, -qz, qw))
    height = model_profile.head_top_height_m-model_profile.head_height_m/2
    point = transform_point((*world[:2], height), camera_from_map)
    if point[2] <= 0:
        return None, {**info, "reason": "qr_search_behind_camera"}
    projection = project_optical_point(point, intrinsics,
        physical_size_m=max(model_profile.head_width_m, model_profile.head_height_m))
    roi = roi_from_projection(projection, intrinsics, padding_scale=2.4)
    if roi is None:
        return None, {**info, "reason": "qr_search_outside_image"}
    attempt = HeadRoiAttempt(roi, "current_unique_scan_qr_search", 2.4,
        projection.u_px, projection.v_px, intrinsics.fy_px*model_profile.head_height_m/point[2])
    return attempt, {**info, "accepted": True, "attempt": attempt.metadata()}
