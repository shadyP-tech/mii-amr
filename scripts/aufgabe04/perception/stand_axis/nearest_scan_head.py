"""Current scan locations for nearest-stand camera search, never head pixels."""

import math

from scripts.aufgabe04.perception.lidar_stand_detector import detect_stand_candidates_from_scan
from scripts.aufgabe04.perception.models import LidarStandDetectorConfig
from scripts.aufgabe04.perception.scan_topology import ScanTopology
from scripts.aufgabe04.perception.stand_axis.metric_head_search import metric_head_search
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector, transform_point


def _inverse_point(point, transform):
    x, y, z, w = transform.rotation_xyzw
    return rotate_vector(tuple(a-b for a, b in zip(point, transform.translation_xyz_m)),
                         (-x, -y, -z, w))


def nearest_scan_head_search(*, scan, image_stamp_sec, now_sec, max_scan_age_sec,
                             scan_from_camera, base_from_camera, model_profile,
                             fx, fy, cx, cy, image_shape, sync_tolerance_sec=.15,
                             position_uncertainty_m=.02):
    """Project the nearest isolated stand-sized scan cluster into the full image.

    Uses measured floor-relative extrinsics and model head height. This is a
    search prior only: scan points, projection and target choice cannot certify
    a camera border or a pose. Missing/stale/ambiguous context fails explicitly.
    """
    info = {"policy": "nearest_current_scan_head", "supplies_corners": False,
            "motion_authorized": False, "candidates": []}
    def fail(reason):
        info["reason"] = reason
        return None, info
    if scan is None or scan_from_camera is None or base_from_camera is None:
        return fail("nearest_head_scan_or_floor_extrinsics_unavailable")
    if scan.scan_frame_id.lstrip('/') != scan_from_camera.parent_frame.lstrip('/'):
        return fail("nearest_head_scan_frame_mismatch")
    stamps = (scan.scan_stamp_sec, scan.receipt_sec, image_stamp_sec, now_sec)
    if any(value is None or not math.isfinite(value) for value in stamps):
        return fail("nearest_head_timestamp_unavailable")
    if (not 0. <= now_sec-scan.scan_stamp_sec <= max_scan_age_sec
            or not 0. <= now_sec-scan.receipt_sec <= max_scan_age_sec
            or abs(scan.scan_stamp_sec-image_stamp_sec) > sync_tolerance_sec):
        return fail("nearest_head_scan_stale_or_unsynchronized")
    if (not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or min(fx, fy) <= 0.
            or not model_profile.committable or model_profile.environment != "physical"):
        return fail("nearest_head_metric_context_invalid")
    topology = ScanTopology(len(scan.ranges), scan.angle_min, scan.angle_increment,
                            scan.angle_max, scan.scan_topology_profile)
    info["scan_topology"] = topology.evidence()
    config = LidarStandDetectorConfig(min_width_m=.012,
        max_width_m=1.5*model_profile.head_width_m + 2*model_profile.tolerance_m,
        max_cluster_gap_m=.04, min_cluster_points=2)
    clusters = detect_stand_candidates_from_scan(scan.ranges,
        angle_min_rad=scan.angle_min, angle_increment_rad=scan.angle_increment,
        angle_max_rad=scan.angle_max, scan_topology_profile=scan.scan_topology_profile,
        config=config)
    projected = []
    for cluster in clusters:
        surface_camera = _inverse_point((cluster.center_x_m, cluster.center_y_m, 0.), scan_from_camera)
        surface_base = transform_point(surface_camera, base_from_camera)
        center_base = (surface_base[0], surface_base[1],
                       model_profile.head_top_height_m-.5*model_profile.head_height_m)
        point = _inverse_point(center_base, base_from_camera)
        if point[2] <= .05:
            continue
        u, v = fx*point[0]/point[2]+cx, fy*point[1]/point[2]+cy
        height = fy*model_profile.head_height_m/point[2]
        if not (0. <= u < image_shape[1] and 0. <= v < image_shape[0] and height >= 8.):
            continue
        distance = math.hypot(*center_base[:2])
        projected.append((distance, u, v, height, cluster, point[2]))
    projected.sort(key=lambda p: p[0])
    info["candidates"] = [{"range_m": d, "center_u_px": u, "center_v_px": v,
                           "height_px": h, "scan_indices": c.source_indices,
                           "wraps_scan_seam": c.wraps_scan_seam}
                          for d, u, v, h, c, depth in projected]
    if not projected:
        return fail("nearest_head_no_visible_scan_candidate")
    if len(projected) > 1 and projected[1][0]-projected[0][0] < .03:
        return fail("nearest_head_scan_candidates_ambiguous")
    distance, u, v, height, cluster, depth = projected[0]
    x, y, z, w = base_from_camera.rotation_xyzw
    try:
        search = metric_head_search(model_profile=model_profile, depth_m=depth,
            fx=fx, fy=fy, cx=cx, cy=cy, image_shape=image_shape, center=(u, v),
            camera_vertical=rotate_vector((0., 0., 1.), (-x, -y, -z, w)),
            position_uncertainty_m=position_uncertainty_m,
            max_center_offset_ratio=.70)
    except ValueError:
        return fail("nearest_head_metric_context_invalid")
    info.update(reason="nearest_head_current_scan_hint", search=search.diagnostics(),
                scan_stamp_sec=scan.scan_stamp_sec, image_stamp_sec=image_stamp_sec)
    return search, info
