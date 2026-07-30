"""Temporally pooled LiDAR line fitting for a coarse stand-face axis."""

from __future__ import annotations

import math
import statistics
from typing import Sequence

from scripts.aufgabe04.perception.stand_axis_handoff.geometry import (
    axial_normalize_rad,
)
from scripts.aufgabe04.perception.stand_axis_handoff.models import (
    LidarAxisEstimate,
)
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _valid_ranges_in_cone(
    scan: PlainLaserScan,
    *,
    bearing_rad: float,
    half_angle_rad: float,
) -> tuple[tuple[float, float], ...]:
    selected = []
    for index, raw_range in enumerate(scan.ranges):
        distance = float(raw_range)
        if (
            not math.isfinite(distance)
            or distance < scan.range_min
            or distance > scan.range_max
        ):
            continue
        angle = scan.angle_min + index * scan.angle_increment
        if abs(_normalize_angle(angle - bearing_rad)) <= half_angle_rad:
            selected.append((angle, distance))
    return tuple(selected)


def estimate_pooled_lidar_axis(
    scans: Sequence[PlainLaserScan],
    *,
    target_bearing_rad: float,
    bearing_half_angle_rad: float = math.radians(8.0),
    target_range_m: float | None = None,
    range_tolerance_m: float = 0.12,
    min_points: int = 20,
    min_linearity: float = 0.90,
    min_length_m: float = 0.04,
    max_length_m: float = 0.12,
) -> LidarAxisEstimate:
    """Fit a PCA tangent from associated returns across stationary scans."""

    if not scans:
        return LidarAxisEstimate(False, "no_scans")
    if not math.isfinite(target_bearing_rad):
        raise ValueError("target bearing must be finite")
    if not math.isfinite(bearing_half_angle_rad) or bearing_half_angle_rad <= 0.0:
        raise ValueError("bearing half-angle must be finite and positive")
    if not math.isfinite(range_tolerance_m) or range_tolerance_m <= 0.0:
        raise ValueError("range tolerance must be finite and positive")
    frames = {scan.scan_frame_id for scan in scans if scan.scan_frame_id}
    if len(frames) > 1:
        return LidarAxisEstimate(False, "scan_frame_mismatch")

    cone_returns = [
        _valid_ranges_in_cone(
            scan,
            bearing_rad=target_bearing_rad,
            half_angle_rad=bearing_half_angle_rad,
        )
        for scan in scans
    ]
    if target_range_m is None:
        nearest_per_scan = [
            min(distance for _angle, distance in selected)
            for selected in cone_returns
            if selected
        ]
        if len(nearest_per_scan) < 3:
            return LidarAxisEstimate(False, "target_range_unavailable")
        target_range_m = statistics.median(nearest_per_scan)
    if not math.isfinite(target_range_m) or target_range_m <= 0.0:
        raise ValueError("target range must be finite and positive")

    points = []
    contributing_scans = 0
    for selected in cone_returns:
        scan_points = [
            (distance * math.cos(angle), distance * math.sin(angle))
            for angle, distance in selected
            if abs(distance - target_range_m) <= range_tolerance_m
        ]
        if scan_points:
            contributing_scans += 1
            points.extend(scan_points)
    if len(points) < min_points:
        return LidarAxisEstimate(
            False,
            "too_few_associated_points",
            sample_count=len(points),
            scan_count=contributing_scans,
            target_range_m=target_range_m,
            target_bearing_rad=target_bearing_rad,
        )

    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    xx = sum((point[0] - center_x) ** 2 for point in points) / len(points)
    xy = sum(
        (point[0] - center_x) * (point[1] - center_y) for point in points
    ) / len(points)
    yy = sum((point[1] - center_y) ** 2 for point in points) / len(points)
    axis = axial_normalize_rad(0.5 * math.atan2(2.0 * xy, xx - yy))
    trace = xx + yy
    discriminant = math.sqrt(max(0.0, (xx - yy) ** 2 + 4.0 * xy * xy))
    major = 0.5 * (trace + discriminant)
    minor = 0.5 * (trace - discriminant)
    linearity = 0.0 if major <= 1.0e-12 else max(0.0, 1.0 - minor / major)
    tangent = (math.cos(axis), math.sin(axis))
    normal = (-tangent[1], tangent[0])
    along = [
        (point[0] - center_x) * tangent[0]
        + (point[1] - center_y) * tangent[1]
        for point in points
    ]
    across = [
        (point[0] - center_x) * normal[0]
        + (point[1] - center_y) * normal[1]
        for point in points
    ]
    length = _percentile(along, 0.95) - _percentile(along, 0.05)
    width = _percentile(across, 0.95) - _percentile(across, 0.05)
    reason = "axis_estimated"
    usable = True
    if linearity < min_linearity:
        usable, reason = False, "linearity_below_gate"
    elif length < min_length_m:
        usable, reason = False, "line_length_below_gate"
    elif length > max_length_m:
        usable, reason = False, "line_length_above_gate"
    confidence = min(
        1.0,
        linearity * min(1.0, len(points) / max(float(min_points), 1.0)),
    )
    return LidarAxisEstimate(
        usable=usable,
        reason=reason,
        angle_rad=axis,
        confidence=confidence,
        sample_count=len(points),
        scan_count=contributing_scans,
        target_range_m=target_range_m,
        target_bearing_rad=target_bearing_rad,
        center_xy_m=(center_x, center_y),
        length_m=length,
        width_m=width,
        linearity=linearity,
    )
