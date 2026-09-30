"""Bounded planar head geometry from observed returns, without motion authority.

The default dimensions are the measured physical_stand_measured_20260826_v2
cross-section. Noise is an explicit engineering allowance, not a calibrated
confidence interval. Missing endpoints leave a finite interval for the center;
repeating an identical scan never makes that interval smaller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from scripts.aufgabe04.perception.stand_axis_handoff.geometry import axial_difference_rad, axial_normalize_rad


@dataclass(frozen=True)
class LidarHeadModel:
    width_m: float = .078
    depth_m: float = .006
    tolerance_m: float = .002
    point_noise_m: float = .003


HEAD_MODEL = LidarHeadModel()
MAX_ANGLE_UNCERTAINTY_RAD = math.radians(15)
MAX_CENTER_UNCERTAINTY_M = .035


@dataclass(frozen=True)
class HeadSurfaceFit:
    tangent_rad: float
    center_x_m: float
    center_y_m: float
    along_uncertainty_m: float
    across_uncertainty_m: float
    angle_uncertainty_rad: float
    observed_span_m: float
    point_count: int

    @property
    def center_uncertainty_m(self) -> float:
        return math.hypot(self.along_uncertainty_m, self.across_uncertainty_m)


def adjacent_returns(
    first_range: float, second_range: float, angular_step: float,
    *, model: LidarHeadModel = HEAD_MODEL,
) -> bool:
    """Range-aware gap plus a separate depth-discontinuity limit.

    Callers must additionally require original consecutive beam indices. The
    finite width bound and per-scan line fit still reject broad/mixed surfaces.
    The factor two allows incidence up to roughly 60 degrees, not arbitrary
    joining across missing returns or the scan seam.
    """
    if (not all(math.isfinite(v) and v > 0 for v in
                (first_range, second_range, abs(angular_step)))
            or abs(angular_step) > math.pi / 4):
        return False
    spacing = 2 * min(first_range, second_range) * math.sin(abs(angular_step) / 2)
    gap = math.sqrt(max(0., (first_range - second_range) ** 2
                        + 4 * first_range * second_range * math.sin(angular_step / 2) ** 2))
    maximum_gap = min(model.width_m + model.tolerance_m,
                      2 * spacing + 2 * model.point_noise_m)
    maximum_depth_jump = min(.035, math.sqrt(3) * spacing + 2 * model.point_noise_m)
    return gap <= maximum_gap and abs(first_range - second_range) <= maximum_depth_jump


def fit_head_surface(
    points: Sequence[tuple[float, float]], *, sensor_position: tuple[float, float],
    model: LidarHeadModel = HEAD_MODEL,
) -> HeadSurfaceFit | None:
    """Fit one spatially supported scan, retaining unobserved endpoint freedom."""
    if (len(points) < 4 or not all(math.isfinite(v) for p in points for v in p)
            or not all(math.isfinite(v) for v in sensor_position)):
        return None
    cx, cy = (sum(p[k] for p in points) / len(points) for k in (0, 1))
    xx = sum((x - cx) ** 2 for x, y in points) / len(points)
    yy = sum((y - cy) ** 2 for x, y in points) / len(points)
    xy = sum((x - cx) * (y - cy) for x, y in points) / len(points)
    tangent = axial_normalize_rad(.5 * math.atan2(2 * xy, xx - yy))
    ct, st = math.cos(tangent), math.sin(tangent)
    along = [(x - cx) * ct + (y - cy) * st for x, y in points]
    across = [-(x - cx) * st + (y - cy) * ct for x, y in points]
    span = max(along) - min(along)
    major = sum(v * v for v in along) / len(points)
    minor = sum(v * v for v in across) / len(points)
    residual = max(abs(v) for v in across)
    max_extent = math.hypot(model.width_m + model.tolerance_m,
                            model.depth_m + model.tolerance_m)
    if (span < .04 or span > max_extent + 2 * model.point_noise_m
            or major <= 0 or minor / major > .05 or residual > .006):
        return None
    angle_error = math.atan2(2 * (model.point_noise_m + residual), span)
    if angle_error > MAX_ANGLE_UNCERTAINTY_RAD:
        return None
    # Any center in this interval can explain the visible partial segment.
    # Its midpoint is a proposal, not an assertion that endpoints were seen.
    along_low = max(along) - max_extent / 2 - model.point_noise_m
    along_high = min(along) + max_extent / 2 + model.point_noise_m
    if along_low > along_high:
        return None
    along_center = (along_low + along_high) / 2
    # The observed broad face is nearer the sensor than the slab center.
    sensor_side = (-st * (sensor_position[0] - cx)
                   + ct * (sensor_position[1] - cy))
    if abs(sensor_side) < .1:
        return None  # Near edge-on geometry does not establish a broad face.
    across_center = -math.copysign(model.depth_m / 2, sensor_side)
    across_error = (model.point_noise_m + residual + model.tolerance_m / 2
                    + max_extent / 2 * math.sin(angle_error))
    return HeadSurfaceFit(
        tangent, cx + ct * along_center - st * across_center,
        cy + st * along_center + ct * across_center,
        (along_high - along_low) / 2, across_error, angle_error, span, len(points),
    )


def combine_head_surfaces(
    fits: Sequence[HeadSurfaceFit], *, tangent_rad: float,
) -> HeadSurfaceFit | None:
    """Intersect conservative center bounds; preserve systematic angle error.

    The caller establishes independent viewpoints and consistency. Each scan
    already needs enough spatial support; this function cannot pool sparse
    beams to invent orientation. Bounds are rotated into one common frame
    before intersection, so complementary observations can constrain center.
    """
    if not fits or not math.isfinite(tangent_rad):
        return None
    ct, st = math.cos(tangent_rad), math.sin(tangent_rad)
    intervals = []
    angle_error = 0.
    for fit in fits:
        delta = axial_difference_rad(tangent_rad, fit.tangent_rad)
        angle_error = max(angle_error, delta + fit.angle_uncertainty_rad)
        along_error = (abs(math.cos(delta)) * fit.along_uncertainty_m
                       + abs(math.sin(delta)) * fit.across_uncertainty_m)
        across_error = (abs(math.sin(delta)) * fit.along_uncertainty_m
                        + abs(math.cos(delta)) * fit.across_uncertainty_m)
        along = ct * fit.center_x_m + st * fit.center_y_m
        across = -st * fit.center_x_m + ct * fit.center_y_m
        intervals.append((along - along_error, along + along_error,
                          across - across_error, across + across_error))
    low_t, high_t = max(i[0] for i in intervals), min(i[1] for i in intervals)
    low_n, high_n = max(i[2] for i in intervals), min(i[3] for i in intervals)
    if low_t > high_t or low_n > high_n or angle_error > MAX_ANGLE_UNCERTAINTY_RAD:
        return None
    t, n = (low_t + high_t) / 2, (low_n + high_n) / 2
    # Do not report sub-noise precision even when intervals barely intersect.
    ut = max(HEAD_MODEL.point_noise_m, (high_t - low_t) / 2)
    un = max(HEAD_MODEL.point_noise_m, (high_n - low_n) / 2)
    if math.hypot(ut, un) > MAX_CENTER_UNCERTAINTY_M:
        return None
    return HeadSurfaceFit(tangent_rad, ct * t - st * n, st * t + ct * n,
                          ut, un, angle_error, min(f.observed_span_m for f in fits),
                          min(f.point_count for f in fits))
