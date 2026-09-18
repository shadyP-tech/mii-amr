"""Project a mapped LiDAR candidate's possible head volume into an image.

This is a search region, not a depth segmentation or a measured silhouette.
Unknown head yaw is covered by a horizontal bounding box. Only locator edges
are masked; the original image and edges must certify all final borders.
"""

from dataclasses import dataclass
from itertools import product
import math

from scripts.aufgabe04.perception.stand_axis_handoff.geometry import transform_point


@dataclass(frozen=True)
class LidarHeadEdgeRegion:
    shape: tuple[int, int]
    bounds: tuple[int, int, int, int]  # Exclusive upper bounds, full-image pixels.

    def __post_init__(self):
        h, w = self.shape
        x0, y0, x1, y1 = self.bounds
        if (any(type(v) is not int for v in (*self.shape, *self.bounds))
                or not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h):
            raise ValueError("invalid LiDAR head edge region")

    def locator_edges(self, raw_edges):
        import numpy as np

        if raw_edges.shape != self.shape:
            raise ValueError("LiDAR head region must match the exact processing image")
        x0, y0, x1, y1 = self.bounds
        selected = np.zeros_like(raw_edges)
        selected[y0:y1, x0:x1] = raw_edges[y0:y1, x0:x1]
        return selected

    def contains(self, corners):
        x0, y0, x1, y1 = self.bounds
        return bool(corners) and all(
            x0 <= p.u_px < x1 and y0 <= p.v_px < y1 for p in corners)

    def diagnostics(self):
        return {"policy": "lidar_candidate_head_volume", "bounds_xyxy": self.bounds,
                "image_shape": self.shape, "supplies_corners": False,
                "pixel_depth_measured": False, "raw_edges_unchanged": True,
                "motion_authorized": False}


def project_lidar_candidate_head_region(
    *, candidate_xy, camera_from_map, model_profile, intrinsics, image_shape,
    position_uncertainty_m, surface_center_margin_m, association,
    image_stamp_sec, now_sec, max_sensor_age_sec, sync_tolerance_sec,
    pixel_margin=6,
):
    """Use exploration coordinates and this image's exact-time camera transform.

    A unique current scan association must corroborate the mapped candidate.
    Missing, stale or ambiguous evidence leaves the existing visual search
    available, with an explicit diagnostic explaining why no mask was applied.
    The surface-to-center margin covers a scan-derived location on a base or
    support rather than at the head axis. Position uncertainty also pads height
    to tolerate floor/camera registration error; no panel orientation is assumed
    beyond the measured upright stand construction.
    """
    info = {"policy": "lidar_candidate_head_volume", "applied": False,
            "supplies_corners": False, "pixel_depth_measured": False}

    def fail(reason):
        info["reason"] = reason
        return None, info

    if (association is None or not association.associated
            or association.eligible_cluster_count != 1):
        return fail("candidate_scan_not_uniquely_associated")
    stamp = association.scan_stamp_sec
    values = (stamp, image_stamp_sec, now_sec, max_sensor_age_sec, sync_tolerance_sec)
    if (any(v is None or not math.isfinite(v) for v in values)
            or min(max_sensor_age_sec, sync_tolerance_sec) <= 0
            or not 0 <= now_sec - stamp <= max_sensor_age_sec
            or not 0 <= now_sec - image_stamp_sec <= max_sensor_age_sec
            or abs(stamp - image_stamp_sec) > sync_tolerance_sec):
        return fail("candidate_scan_stale_or_unsynchronized")
    try:
        x, y = candidate_xy
        fx, fy, cx, cy = intrinsics
        rows, cols = image_shape[:2]
        if (not all(math.isfinite(v) for v in
                    (x, y, fx, fy, cx, cy, position_uncertainty_m,
                     surface_center_margin_m, pixel_margin))
                or min(fx, fy) <= 0 or min(rows, cols) <= 0
                or min(position_uncertainty_m, surface_center_margin_m, pixel_margin) < 0
                or not model_profile.committable or model_profile.environment != "physical"
                or getattr(model_profile, "head_top_height_m", None) is None):
            return fail("candidate_head_metric_context_invalid")
        tolerance = model_profile.tolerance_m
        # Enclose every yaw of the measured thin rectangular head.
        radius = (math.hypot(.5 * model_profile.head_width_m, model_profile.head_depth_m)
                  + position_uncertainty_m + surface_center_margin_m + tolerance)
        top = model_profile.head_top_height_m + position_uncertainty_m + tolerance
        bottom = (model_profile.head_top_height_m - model_profile.head_height_m
                  - position_uncertainty_m - tolerance)
        points = [transform_point(p, camera_from_map) for p in product(
            (x-radius, x+radius), (y-radius, y+radius), (bottom, top))]
        if any(p[2] <= .01 for p in points):
            return fail("candidate_head_volume_crosses_camera_plane")
        pixels = [(fx * px / pz + cx, fy * py / pz + cy) for px, py, pz in points]
        us, vs = zip(*pixels)
        bounds = (max(0, math.floor(min(us)-pixel_margin)),
                  max(0, math.floor(min(vs)-pixel_margin)),
                  min(cols, math.ceil(max(us)+pixel_margin)+1),
                  min(rows, math.ceil(max(vs)+pixel_margin)+1))
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            return fail("candidate_head_volume_outside_image")
        if min(bounds[2]-bounds[0], bounds[3]-bounds[1]) < 8:
            return fail("candidate_head_region_too_small")
        region = LidarHeadEdgeRegion((rows, cols), bounds)
    except (TypeError, ValueError, ArithmeticError):
        return fail("candidate_head_metric_context_invalid")
    info.update(region.diagnostics(), applied=True, reason="projected_candidate_head_volume",
                candidate_xy_m=(x, y), position_uncertainty_m=position_uncertainty_m,
                surface_center_margin_m=surface_center_margin_m,
                image_stamp_sec=image_stamp_sec, scan_stamp_sec=stamp)
    return region, info
