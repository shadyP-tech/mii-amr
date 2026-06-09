from __future__ import annotations

import math

from .math_utils import normalize_angle_rad
from .models import ScanSafety


FORWARD_SOFT_STOP_MIN_CLOSE_RANGES = 2


def percentile(values, percent):
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percent / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def valid_scan_ranges(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    sector_half_angle_deg=None,
):
    selected = []
    half_angle_rad = (
        math.radians(sector_half_angle_deg)
        if sector_half_angle_deg is not None
        else None
    )
    for index, raw_range in enumerate(ranges):
        if not math.isfinite(raw_range):
            continue
        if raw_range < range_min or raw_range > range_max:
            continue
        if half_angle_rad is not None:
            angle = normalize_angle_rad(angle_min + index * angle_increment)
            if abs(angle) > half_angle_rad:
                continue
        selected.append(float(raw_range))
    return selected


def evaluate_scan_safety(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    mode,
    scan_half_angle_deg,
    hard_stop_range_m,
    min_scan_range_m,
    rotation_stop_range_m,
):
    if mode not in {"forward", "rotate"}:
        raise ValueError(f"unsupported scan mode: {mode!r}")

    sector = scan_half_angle_deg if mode == "forward" else None
    selected = valid_scan_ranges(
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max,
        sector_half_angle_deg=sector,
    )
    if not selected:
        return ScanSafety(False, "no_valid_scan_ranges", 0, None, None)

    min_range = min(selected)
    percentile_5 = percentile(selected, 5.0)
    soft_threshold = min_scan_range_m if mode == "forward" else rotation_stop_range_m

    if min_range < hard_stop_range_m:
        return ScanSafety(False, "hard_stop", len(selected), min_range, percentile_5)
    if mode == "forward":
        close_count = sum(1 for value in selected if value < min_scan_range_m)
        if close_count >= FORWARD_SOFT_STOP_MIN_CLOSE_RANGES:
            return ScanSafety(False, "soft_stop", len(selected), min_range, percentile_5)
    if percentile_5 < soft_threshold:
        return ScanSafety(False, "soft_stop", len(selected), min_range, percentile_5)
    return ScanSafety(True, "clear", len(selected), min_range, percentile_5)

