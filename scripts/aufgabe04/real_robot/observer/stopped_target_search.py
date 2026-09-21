"""Current scan search hint, bounded by the original candidate registration.

The hint never edits survey geometry, supplies borders, or authorizes motion.
Final measured-head association must use the same scan and original map bearing.
"""

from dataclasses import dataclass, field
import math

from scripts.aufgabe04.perception.candidate_lidar_association import (
    associate_candidate_lidar_target, normalize_certified_camera_map_bearing_limit,
)
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector, transform_point
from scripts.aufgabe04.real_robot.configuration.geometry import project_optical_point


@dataclass(frozen=True)
class StoppedTargetSearch:
    original_projection: object
    projection: object
    candidate_xy: tuple[float, float]
    search_xy: tuple[float, float]
    map_bearing_rad: float
    scan: object = field(repr=False)
    source_indices: tuple[int, ...]
    image_stamp_sec: float
    residual_m: float

    def metadata(self):
        return {"policy": "current_unique_scan_search_hint", "accepted": True,
                "candidate_xy_m": self.candidate_xy, "search_xy_m": self.search_xy,
                "residual_m": self.residual_m, "scan_stamp_sec": self.scan.scan_stamp_sec,
                "image_stamp_sec": self.image_stamp_sec, "source_indices": self.source_indices,
                "original_center_px": [self.original_projection.u_px, self.original_projection.v_px],
                "search_center_px": [self.projection.u_px, self.projection.v_px],
                "candidate_geometry_updated": False, "final_association_required": True,
                "motion_authorized": False}


def reconcile_stopped_target_search(*, scan, candidate_xy, original_projection,
        map_bearing_rad, cone_half_angle_rad, max_bearing_delta_rad, accepted_range_m,
        now_sec, image_stamp_sec, max_age_sec, sync_tolerance_sec,
        scan_from_map, camera_from_map, intrinsics, model_profile,
        stand_radius_m, stand_uncertainty_m):
    """Locate one compact current cluster within the existing registration budget.

    Use the whole search envelope for uniqueness, including one-point competitors.
    A selected cluster needs three contiguous raw samples; fragmentation never
    becomes authority to shift a hint. The caller has already checked stationarity.
    """
    info = {"policy": "current_unique_scan_search_hint", "accepted": False,
            "candidate_geometry_updated": False, "motion_authorized": False}
    def reject(reason):
        return None, {**info, "reason": reason}
    if (not all(math.isfinite(v) for v in (*candidate_xy, stand_radius_m,
            stand_uncertainty_m, max_age_sec, sync_tolerance_sec))
            or stand_radius_m <= 0 or stand_uncertainty_m < 0
            or min(max_age_sec, sync_tolerance_sec) <= 0):
        return reject("invalid_search_context")
    if scan is None or scan.scan_stamp_sec is None:
        return reject("current_scan_required")
    if (scan.scan_frame_id != scan_from_map.parent_frame
            or scan_from_map.child_frame != camera_from_map.child_frame):
        return reject("search_transform_frame_mismatch")
    if (not all(math.isfinite(v) for v in (now_sec, image_stamp_sec, scan.scan_stamp_sec))
            or not 0 <= now_sec-image_stamp_sec <= max_age_sec
            or not 0 <= now_sec-scan.scan_stamp_sec <= max_age_sec
            or abs(image_stamp_sec-scan.scan_stamp_sec) > sync_tolerance_sec):
        return reject("stale_or_unsynchronized_search_hint")
    nominal = associate_candidate_lidar_target(
        scan, map_bearing_rad=map_bearing_rad, cone_half_angle_rad=cone_half_angle_rad,
        accepted_range_m=accepted_range_m, now_sec=now_sec,
        max_scan_age_sec=max_age_sec, min_cluster_sample_count=1)
    if nominal.eligible_cluster_count:
        # Preserve the established search when the original cone has a target
        # (or competing targets). This recovery is only for a displaced target.
        return reject("original_target_search_retained")
    limit = normalize_certified_camera_map_bearing_limit(max_bearing_delta_rad)
    association = associate_candidate_lidar_target(
        scan, map_bearing_rad=map_bearing_rad,
        cone_half_angle_rad=limit+cone_half_angle_rad,
        accepted_range_m=accepted_range_m, now_sec=now_sec,
        max_scan_age_sec=max_age_sec, min_cluster_sample_count=1)
    if not association.associated or association.eligible_cluster_count != 1:
        return reject("search_cluster_not_unique")
    indices = association.selected_cluster_source_indices
    if len(indices) < 3:
        return reject("search_cluster_insufficient_contiguous_samples")
    points = tuple((scan.ranges[i]*math.cos(scan.angle_min+i*scan.angle_increment),
                    scan.ranges[i]*math.sin(scan.angle_min+i*scan.angle_increment)) for i in indices)
    x, y = (sum(p[k] for p in points)/len(points) for k in (0, 1))
    delta = math.remainder(math.atan2(y, x)-map_bearing_rad, 2*math.pi)
    if abs(delta) > limit:
        return reject("search_cluster_outside_registration_bearing")
    if max(math.dist(a, b) for a in points for b in points) > 2*(stand_radius_m+stand_uncertainty_m):
        return reject("search_cluster_not_compact")
    # Invert the exact scan-time map->scan transform. Only horizontal position
    # is a scan hint; upright head height still comes from the measured model.
    tx, ty, tz = scan_from_map.translation_xyz_m
    qx, qy, qz, qw = scan_from_map.rotation_xyzw
    world = rotate_vector((x-tx, y-ty, -tz), (-qx, -qy, -qz, qw))
    search_xy = world[:2]
    residual = math.dist(candidate_xy, search_xy)
    if residual > 2*(stand_radius_m+stand_uncertainty_m):
        return reject("search_cluster_outside_candidate_distance")
    center_height = model_profile.head_top_height_m-model_profile.head_height_m/2
    camera_point = transform_point((*search_xy, center_height), camera_from_map)
    projection = project_optical_point(camera_point, intrinsics,
        physical_size_m=max(model_profile.head_width_m, model_profile.head_height_m))
    if projection.depth_m <= 0 or not projection.inside_image:
        return reject("search_projection_outside_image")
    hint = StoppedTargetSearch(original_projection, projection, tuple(candidate_xy),
        tuple(search_xy), map_bearing_rad, scan, tuple(indices), image_stamp_sec, residual)
    return hint, hint.metadata()
