"""Candidate bearing and surface-range bounds in one LiDAR coordinate frame.

LaserScan ranges originate at the scanner, which need not coincide with the
robot's base frame. The caller supplies the candidate transformed at the scan
timestamp; this module preserves the existing surface and tolerance policy.
"""

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class ScanTargetGeometry:
    center_range_m: float
    bearing_rad: float
    accepted_range_m: tuple[float, float]


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def scan_target_geometry(
    scan_point_xyz_m: Sequence[float],
    *,
    stand_radius_m: float,
    stand_uncertainty_m: float,
    lidar_range_tolerance_m: float,
) -> ScanTargetGeometry:
    """Derive both gates from the same exact-time candidate point in scan.

    Only the planar range is relevant to LaserScan. A nonfinite height still
    rejects a malformed transform result. Negative lower bounds retain the
    established surface-range policy; this helper does not widen any gate.
    """

    if (
        isinstance(scan_point_xyz_m, (str, bytes))
        or not isinstance(scan_point_xyz_m, Sequence)
        or len(scan_point_xyz_m) != 3
    ):
        raise ValueError("scan_point_xyz_m must contain three finite coordinates")
    x_m, y_m, _ = (
        _finite(value, f"scan_point_xyz_m[{index}]")
        for index, value in enumerate(scan_point_xyz_m)
    )
    radius = _finite(stand_radius_m, "stand_radius_m")
    uncertainty = _finite(stand_uncertainty_m, "stand_uncertainty_m")
    tolerance = _finite(lidar_range_tolerance_m, "lidar_range_tolerance_m")
    if min(radius, uncertainty, tolerance) < 0.0:
        raise ValueError("scan range policy values must be nonnegative")
    distance = math.hypot(x_m, y_m)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("candidate planar scan range must be finite and positive")
    lower = distance - 2.0 * radius - uncertainty - tolerance
    upper = distance + tolerance
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("candidate scan range bounds must be finite")
    return ScanTargetGeometry(distance, math.atan2(y_m, x_m), (lower, upper))
