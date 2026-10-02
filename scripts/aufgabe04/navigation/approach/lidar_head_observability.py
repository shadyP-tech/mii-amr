"""Require the configured measured head and current laser-plane intersection.

This is a check of measured dimensions and recorded exact-time transforms;
it cannot prove physical mounting calibration or identify a face by itself.
"""

from __future__ import annotations

import math

from scripts.aufgabe04.navigation.approach.lidar_head_geometry import HEAD_MODEL
from scripts.aufgabe04.perception.stand_axis.model_profile import StandModelProfile


def lidar_head_model_admission(stand_model) -> dict[str, object]:
    result = {"accepted": False, "reason": "measured_head_model_unavailable",
              "motion_authorized": False, "stand_axis_authorized": False}
    if not isinstance(stand_model, StandModelProfile):
        return result
    result["stand_model_profile_sha256"] = stand_model.sha256
    if stand_model.environment != "physical" or stand_model.measurement_status != "measured":
        return {**result, "reason": "measured_physical_head_model_required"}
    dimensions = {"width_m": stand_model.head_width_m, "depth_m": stand_model.head_depth_m,
                  "tolerance_m": stand_model.tolerance_m}
    result["head_fit_dimensions"] = dimensions
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or abs(value - getattr(HEAD_MODEL, name)) > 1e-9
           for name, value in dimensions.items()):
        return {**result, "reason": "configured_head_model_differs_from_lidar_fit"}
    if (stand_model.head_top_height_m is None
            or not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in
                       (stand_model.head_top_height_m, stand_model.head_height_m))
            or stand_model.head_top_height_m <= stand_model.head_height_m):
        return {**result, "reason": "measured_head_height_unavailable"}
    return {**result, "accepted": True, "reason": "measured_head_model_matches_lidar_fit"}


def verify_lidar_head_observability(
    *, stand_model, base_frame, mount_evidence, target_range_m, source_scan_stamps_sec,
) -> dict[str, object]:
    """Check every beam plane in one complete captured cohort against the head.

    ``target_range_m`` must upper-bound scan-origin to head range, including
    center uncertainty and head half-width. Bounding every ray direction at
    that radius is conservative without retaining per-fit beam indices.
    Source stamps come from the validated capture receipts, independently of
    mount records. Both production eight-scan and legacy three-scan cohorts
    require one exact-time mount record for every scan, in the same order.
    Capture/arrival admission independently owns freshness.
    """
    model_review = lidar_head_model_admission(stand_model)
    result = {**model_review, "accepted": False,
              "head_cross_section_supported": False, "physical_mount_calibration_verified": False}
    if not model_review["accepted"]:
        return result
    def reject(reason):
        return {**result, "reason": reason}
    try:
        if not isinstance(base_frame, str) or base_frame.lstrip("/").split("/")[-1] != "base_footprint":
            return reject("ground_referenced_base_footprint_required")
        if isinstance(target_range_m, bool) or not math.isfinite(target_range_m) or target_range_m <= 0:
            return reject("head_slice_target_range_invalid")
        source_stamps = tuple(source_scan_stamps_sec)
        if (len(source_stamps) not in (3, 8)
                or any(isinstance(stamp, bool) or not math.isfinite(stamp) or stamp <= 0
                       for stamp in source_stamps)
                or any(a >= b for a, b in zip(source_stamps, source_stamps[1:]))):
            return reject("head_slice_source_scan_stamps_invalid")
        records = tuple(mount_evidence)
        result.update(source_scan_count=len(source_stamps), mount_record_count=len(records))
        if len(records) != len(source_stamps):
            return reject("complete_exact_scan_mount_records_required")
        lower = stand_model.head_top_height_m - stand_model.head_height_m + stand_model.tolerance_m
        upper = stand_model.head_top_height_m - stand_model.tolerance_m
        intervals, stamps = [], []
        for record, source_stamp in zip(records, source_stamps):
            if record["ground_frame"] != base_frame:
                return reject("head_slice_ground_frame_mismatch")
            stamp = record["stamp_sec"]
            keys = ("scan_height_above_ground_m", "scan_vertical_direction_x",
                    "scan_vertical_direction_y", "scan_vertical_direction_z")
            values = [record[key] for key in keys]
            exact_stamp = record["exact_transform_stamp_sec"]
            if (any(isinstance(v, bool) or not math.isfinite(v)
                    for v in (*values, stamp, exact_stamp))
                    or stamp <= 0 or record["exact_transform_stamp_sec"] != stamp):
                return reject("head_slice_exact_transform_invalid")
            if stamp != source_stamp:
                return reject("head_slice_mount_scan_stamp_mismatch")
            height, vx, vy, vz = values
            if abs(vx*vx + vy*vy + vz*vz - 1.) > 1e-6 or vz <= 0:
                return reject("head_slice_vertical_direction_invalid")
            vertical_excursion = target_range_m * math.hypot(vx, vy)
            interval = (height - vertical_excursion, height + vertical_excursion)
            intervals.append(interval)
            stamps.append(stamp)
        if any(a >= b for a, b in zip(stamps, stamps[1:])):
            return reject("head_slice_scan_stamps_not_unique_ordered")
        result.update(ground_frame=base_frame, source_scan_stamps_sec=stamps,
                      target_range_bound_m=target_range_m, measured_head_interior_m=[lower, upper],
                      beam_height_intervals_m=[list(v) for v in intervals])
        if any(low < lower or high > upper for low, high in intervals):
            return reject("laser_plane_not_inside_measured_head")
        return {**result, "accepted": True, "head_cross_section_supported": True,
                "reason": "exact_tf_laser_plane_inside_measured_head"}
    except (AttributeError, KeyError, TypeError, ValueError, ArithmeticError):
        return reject("head_slice_mount_evidence_invalid")
