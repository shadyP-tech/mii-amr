#!/usr/bin/env python3
"""
Offline rectangular-arena geometry estimator for TurtleBot scan data.

This module is intentionally ROS-free. It analyzes LaserScan-like samples and
returns a conservative pose-prior diagnostic. Live robot motion and AMCL
initialization belong in later commits.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


WALL_UNKNOWN = "unknown"
WALL_HEATER = "heater"
WALL_CLEAN = "clean"
WALL_AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class ScanSample:
    ranges: Sequence[float]
    angle_min: float
    angle_increment: float
    range_min: float = 0.0
    range_max: float = float("inf")
    odom_pose: Pose2D | None = None


@dataclass(frozen=True)
class ArenaGeometryConfig:
    arena_length_m: float = 3.90
    arena_width_m: float | None = None
    heater_side_width_m: float = 2.016
    clean_side_width_m: float = 1.967
    width_match_min_margin_m: float = 0.015
    max_short_wall_range_sum_error_m: float = 0.15
    map_center_x: float = 0.0
    map_center_y: float = 0.0
    map_yaw_deg: float = 0.0
    heater_wall_side: str = "+x"
    min_wall_points: int = 20
    max_wall_separation_error_m: float = 0.20
    max_line_rmse_m: float = 0.08
    min_parallel_score: float = 0.90
    min_short_wall_points: int = 8
    min_short_wall_confidence: float = 0.75
    min_classification_margin: float = 0.15
    forced_short_wall_side: str | None = None
    forced_short_wall_type: str | None = None
    short_wall_band_m: float = 0.25
    short_wall_outer_band_m: float = 0.06
    min_short_wall_visible_width_m: float = 0.35
    heater_protrusion_min_m: float = 0.06
    profile_min_points: int = 20
    profile_min_visible_width_m: float = 0.45
    profile_max_line_rmse_m: float = 0.035
    profile_heater_relaxed_max_line_rmse_m: float = 0.055
    profile_min_any_line_support_fraction: float = 0.20
    profile_cluster_bin_width_m: float = 0.075
    profile_cluster_gap_tolerance_bins: int = 1
    profile_min_confidence: float = 0.75
    profile_min_assignment_margin: float = 0.20
    profile_min_heater_clean_contrast: float = 0.20
    profile_min_heater_like_score: float = 0.70
    profile_min_clean_like_score: float = 0.70
    profile_max_opposite_score: float = 0.55
    profile_protrusion_min_m: float = 0.04
    profile_heater_depth_p95_low_m: float = 0.03
    profile_heater_depth_p95_high_m: float = 0.08
    profile_heater_protrusion_fraction_low: float = 0.12
    profile_heater_protrusion_fraction_high: float = 0.45
    profile_heater_cluster_count_low: float = 1.0
    profile_heater_cluster_count_high: float = 3.0
    profile_heater_largest_cluster_width_low_m: float = 0.05
    profile_heater_largest_cluster_width_high_m: float = 0.20
    profile_heater_roughness_low_m: float = 0.01
    profile_heater_roughness_high_m: float = 0.04
    profile_heater_line_support_low: float = 0.35
    profile_heater_line_support_high: float = 0.75
    profile_clean_depth_p95_low_m: float = 0.02
    profile_clean_depth_p95_high_m: float = 0.06
    profile_clean_protrusion_fraction_low: float = 0.03
    profile_clean_protrusion_fraction_high: float = 0.15
    profile_clean_cluster_count_low: float = 0.0
    profile_clean_cluster_count_high: float = 2.0
    profile_clean_roughness_low_m: float = 0.005
    profile_clean_roughness_high_m: float = 0.03
    profile_clean_line_support_low: float = 0.45
    profile_clean_line_support_high: float = 0.80
    profile_relative_heater_min_score: float = 0.85
    profile_relative_heater_min_delta: float = 0.15
    profile_relative_opposite_max_heater_score: float = 0.80
    profile_relative_selected_max_clean_score: float = 0.55
    profile_relative_min_protrusion_clusters: int = 2
    profile_relative_min_protrusion_fraction: float = 0.10
    profile_relative_confidence_cap: float = 0.85
    profile_heater_weight_protrusion_fraction: float = 0.40
    profile_heater_weight_width_coverage: float = 0.20
    profile_heater_weight_depth: float = 0.15
    profile_heater_weight_low_flat_support: float = 0.15
    profile_heater_weight_roughness_refined: float = 0.10
    profile_heater_clean_rail_penalty: float = 0.20
    profile_heater_high_flat_support_penalty: float = 0.25
    profile_heater_high_flat_support_penalty_low: float = 0.75
    profile_heater_high_flat_support_penalty_high: float = 0.90
    profile_heater_width_coverage_low: float = 0.08
    profile_heater_width_coverage_high: float = 0.35
    profile_heater_protrusion_depth_p90_low_m: float = 0.05
    profile_heater_protrusion_depth_p90_high_m: float = 0.14
    profile_clean_rail_flat_support_min: float = 0.55
    profile_clean_rail_protrusion_fraction_max: float = 0.10
    profile_clean_rail_depth_p90_max_m: float = 0.07
    profile_clean_rail_dominant_cluster_fraction_min: float = 0.45
    profile_broad_flat_wall_flat_support_min: float = 0.75
    profile_broad_flat_wall_dominant_cluster_fraction_min: float = 0.45
    profile_broad_flat_wall_cluster_width_min_m: float = 0.75
    profile_broad_flat_wall_cluster_count_max: int = 2
    profile_broad_flat_wall_penalty: float = 0.25
    profile_broad_flat_wall_heater_cap: float = 0.69
    angle_search_step_deg: float = 2.0


@dataclass(frozen=True)
class LineFit:
    point_count: int
    normal_x: float
    normal_y: float
    offset: float
    direction_angle_rad: float
    rmse_m: float

    def to_dict(self):
        return {
            "point_count": self.point_count,
            "normal": [self.normal_x, self.normal_y],
            "offset": self.offset,
            "direction_angle_deg": math.degrees(self.direction_angle_rad),
            "rmse_m": self.rmse_m,
        }


@dataclass(frozen=True)
class LongWallFit:
    ok: bool
    reason: str
    axis_angle_rad: float | None = None
    normal_angle_rad: float | None = None
    lower_wall_points: int = 0
    upper_wall_points: int = 0
    wall_separation_m: float | None = None
    wall_separation_error_m: float | None = None
    observed_wall_separation_m: float | None = None
    matched_width_profile_label: str | None = None
    expected_wall_width_m: float | None = None
    width_error_m: float | None = None
    width_match_margin_m: float | None = None
    width_match_mode: str | None = None
    width_match_ambiguous: bool = False
    width_profile_errors_m: dict[str, float] | None = None
    long_wall_rmse_m: float | None = None
    parallel_angle_error_deg: float | None = None
    parallel_score: float | None = None
    lateral_offset_m: float | None = None
    lower_projection_m: float | None = None
    upper_projection_m: float | None = None
    lower_line: LineFit | None = None
    upper_line: LineFit | None = None

    def to_dict(self):
        return {
            "ok": self.ok,
            "reason": self.reason,
            "axis_angle_deg": (
                None if self.axis_angle_rad is None else math.degrees(self.axis_angle_rad)
            ),
            "normal_angle_deg": (
                None if self.normal_angle_rad is None else math.degrees(self.normal_angle_rad)
            ),
            "lower_wall_points": self.lower_wall_points,
            "upper_wall_points": self.upper_wall_points,
            "wall_separation_m": self.wall_separation_m,
            "wall_separation_error_m": self.wall_separation_error_m,
            "observed_wall_separation_m": self.observed_wall_separation_m,
            "matched_width_profile_label": self.matched_width_profile_label,
            "expected_wall_width_m": self.expected_wall_width_m,
            "width_error_m": self.width_error_m,
            "width_match_margin_m": self.width_match_margin_m,
            "width_match_mode": self.width_match_mode,
            "width_match_ambiguous": self.width_match_ambiguous,
            "width_profile_errors_m": self.width_profile_errors_m,
            "long_wall_rmse_m": self.long_wall_rmse_m,
            "parallel_angle_error_deg": self.parallel_angle_error_deg,
            "parallel_score": self.parallel_score,
            "lateral_offset_m": self.lateral_offset_m,
            "lower_projection_m": self.lower_projection_m,
            "upper_projection_m": self.upper_projection_m,
            "lower_line": None if self.lower_line is None else self.lower_line.to_dict(),
            "upper_line": None if self.upper_line is None else self.upper_line.to_dict(),
        }


@dataclass(frozen=True)
class ShortWallClassification:
    wall_type: str
    reason: str
    observed_axis_side: str | None = None
    confidence: float = 0.0
    heater_feature_score: float = 0.0
    clean_feature_score: float = 0.0
    classification_margin: float = 0.0
    short_wall_candidate_range_m: float | None = None
    short_wall_visible_width_m: float | None = None
    short_wall_rmse_m: float | None = None
    short_wall_range_sum_m: float | None = None
    short_wall_range_sum_error_m: float | None = None
    point_count: int = 0
    profile_features: dict | None = None
    heater_profile_score: float = 0.0
    clean_profile_score: float = 0.0
    pairwise_assignment_score: float | None = None
    pairwise_assignment_margin: float | None = None
    heater_clean_contrast: float | None = None
    short_wall_range_sum_expected_m: float | None = None
    short_wall_range_sum_tolerance_m: float | None = None
    selected_assignment: str | None = None
    validity_failed_reason: str | None = None
    heater_profile_delta: float | None = None
    relative_heater_score: float | None = None
    relative_opposite_heater_score: float | None = None
    relative_opposite_max_heater_score: float | None = None
    relative_min_protrusion_clusters: int | None = None
    relative_min_protrusion_fraction: float | None = None
    relative_confidence_raw: float | None = None

    def to_dict(self):
        return {
            "wall_type": self.wall_type,
            "reason": self.reason,
            "axis_side": self.observed_axis_side,
            "observed_axis_side": self.observed_axis_side,
            "confidence": self.confidence,
            "heater_feature_score": self.heater_feature_score,
            "clean_feature_score": self.clean_feature_score,
            "classification_margin": self.classification_margin,
            "short_wall_candidate_range_m": self.short_wall_candidate_range_m,
            "short_wall_visible_width_m": self.short_wall_visible_width_m,
            "short_wall_rmse_m": self.short_wall_rmse_m,
            "short_wall_range_sum_m": self.short_wall_range_sum_m,
            "short_wall_range_sum_error_m": self.short_wall_range_sum_error_m,
            "point_count": self.point_count,
            "profile_features": self.profile_features,
            "heater_profile_score": self.heater_profile_score,
            "clean_profile_score": self.clean_profile_score,
            "pairwise_assignment_score": self.pairwise_assignment_score,
            "pairwise_assignment_margin": self.pairwise_assignment_margin,
            "heater_clean_contrast": self.heater_clean_contrast,
            "range_sum_m": self.short_wall_range_sum_m,
            "range_sum_expected_m": self.short_wall_range_sum_expected_m,
            "range_sum_error_m": self.short_wall_range_sum_error_m,
            "range_sum_tolerance_m": self.short_wall_range_sum_tolerance_m,
            "selected_assignment": self.selected_assignment,
            "validity_failed_reason": self.validity_failed_reason,
            "heater_profile_delta": self.heater_profile_delta,
            "relative_heater_score": self.relative_heater_score,
            "relative_opposite_heater_score": self.relative_opposite_heater_score,
            "relative_opposite_max_heater_score": self.relative_opposite_max_heater_score,
            "relative_min_protrusion_clusters": self.relative_min_protrusion_clusters,
            "relative_min_protrusion_fraction": self.relative_min_protrusion_fraction,
            "relative_confidence_raw": self.relative_confidence_raw,
        }


@dataclass(frozen=True)
class PairwiseShortWallClassification:
    applicable: bool
    accepted: bool
    reason: str
    assignment: str | None = None
    confidence: float = 0.0
    margin: float = 0.0
    heater_clean_contrast: float = 0.0
    winner_score: float = 0.0
    loser_score: float = 0.0
    range_sum_m: float | None = None
    range_sum_expected_m: float | None = None
    range_sum_error_m: float | None = None
    range_sum_tolerance_m: float | None = None
    negative_wall_type: str = WALL_UNKNOWN
    positive_wall_type: str = WALL_UNKNOWN
    heater_profile_delta: float | None = None
    relative_heater_score: float | None = None
    relative_opposite_heater_score: float | None = None
    relative_opposite_max_heater_score: float | None = None
    relative_min_protrusion_clusters: int | None = None
    relative_min_protrusion_fraction: float | None = None
    relative_confidence_raw: float | None = None


@dataclass(frozen=True)
class ArenaGeometryResult:
    success: bool
    failure_reason: str
    pose_unique: bool
    yaw_ambiguity_resolved: bool
    estimated_pose_prior: Pose2D | None
    estimated_covariance: dict[str, float] | None
    long_wall_fit: LongWallFit
    short_wall_classification: ShortWallClassification
    short_wall_candidates: dict[str, ShortWallClassification]
    diagnostics: dict[str, float | int | str | None]

    def to_dict(self):
        return {
            "success": self.success,
            "failure_reason": self.failure_reason,
            "pose_unique": self.pose_unique,
            "yaw_ambiguity_resolved": self.yaw_ambiguity_resolved,
            "estimated_pose_prior": (
                None
                if self.estimated_pose_prior is None
                else {
                    "x": self.estimated_pose_prior.x,
                    "y": self.estimated_pose_prior.y,
                    "yaw_deg": self.estimated_pose_prior.yaw_deg,
                }
            ),
            "estimated_covariance": self.estimated_covariance,
            "long_wall_fit": self.long_wall_fit.to_dict(),
            "short_wall_candidates": {
                side: candidate.to_dict()
                for side, candidate in self.short_wall_candidates.items()
            },
            "short_wall_classification": self.short_wall_classification.to_dict(),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class WidthMatch:
    observed_wall_separation_m: float
    matched_width_profile_label: str
    expected_wall_width_m: float
    width_error_m: float
    width_match_margin_m: float | None
    width_match_mode: str
    width_match_ambiguous: bool
    width_profile_errors_m: dict[str, float]

    @property
    def abs_error_m(self):
        return abs(self.width_error_m)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def normalize_angle_rad(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def normalize_undirected_angle_rad(angle_rad):
    angle = angle_rad % math.pi
    return angle if angle >= 0.0 else angle + math.pi


def undirected_angle_delta_rad(a, b):
    delta = abs(normalize_angle_rad(a - b))
    return min(delta, abs(math.pi - delta))


def percentile_sorted(ordered, percent):
    if not ordered:
        raise ValueError("percentile requires values")
    if len(ordered) == 1:
        return ordered[0]
    rank = (percent / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def percentile(values, percent):
    return percentile_sorted(sorted(values), percent)


def median(values):
    return percentile(values, 50.0)


def finite_scan_points(sample: ScanSample, range_stride=1):
    points = []
    for index, raw_range in enumerate(sample.ranges):
        if index % range_stride != 0:
            continue
        if not math.isfinite(raw_range):
            continue
        if raw_range < sample.range_min or raw_range > sample.range_max:
            continue
        angle = sample.angle_min + index * sample.angle_increment
        points.append((raw_range * math.cos(angle), raw_range * math.sin(angle)))
    return points


def transform_point(point, pose: Pose2D):
    cos_yaw = math.cos(math.radians(pose.yaw_deg))
    sin_yaw = math.sin(math.radians(pose.yaw_deg))
    x, y = point
    return (
        pose.x + cos_yaw * x - sin_yaw * y,
        pose.y + sin_yaw * x + cos_yaw * y,
    )


def relative_pose(pose: Pose2D, origin: Pose2D):
    dx = pose.x - origin.x
    dy = pose.y - origin.y
    yaw = math.radians(-origin.yaw_deg)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return Pose2D(
        x=cos_yaw * dx - sin_yaw * dy,
        y=sin_yaw * dx + cos_yaw * dy,
        yaw_deg=pose.yaw_deg - origin.yaw_deg,
    )


def accumulate_scan_points(samples: Sequence[ScanSample], range_stride=1, max_points=None):
    if not samples:
        return []
    origin = next((sample.odom_pose for sample in samples if sample.odom_pose is not None), None)
    points = []
    for sample in samples:
        pose = Pose2D()
        if sample.odom_pose is not None and origin is not None:
            pose = relative_pose(sample.odom_pose, origin)
        for point in finite_scan_points(sample, range_stride=range_stride):
            points.append(transform_point(point, pose))
            if max_points is not None and len(points) >= max_points:
                return points
    return points


def fit_line(points: Sequence[tuple[float, float]]):
    if len(points) < 2:
        raise ValueError("line fit needs at least two points")
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    sxx = sum((point[0] - mean_x) ** 2 for point in points) / len(points)
    syy = sum((point[1] - mean_y) ** 2 for point in points) / len(points)
    sxy = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points) / len(points)
    direction = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    direction = normalize_undirected_angle_rad(direction)
    normal_x = -math.sin(direction)
    normal_y = math.cos(direction)
    offset = normal_x * mean_x + normal_y * mean_y
    if offset < 0.0:
        normal_x = -normal_x
        normal_y = -normal_y
        offset = -offset
    residuals = [normal_x * point[0] + normal_y * point[1] - offset for point in points]
    rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    return LineFit(
        point_count=len(points),
        normal_x=normal_x,
        normal_y=normal_y,
        offset=offset,
        direction_angle_rad=direction,
        rmse_m=rmse,
    )


def vector_from_angle(angle_rad):
    return math.cos(angle_rad), math.sin(angle_rad)


def dot(point, vector):
    return point[0] * vector[0] + point[1] * vector[1]


def projection_clusters(points, normal, lower_center, upper_center, threshold):
    lower = []
    upper = []
    for point in points:
        projection = dot(point, normal)
        if abs(projection - lower_center) <= threshold:
            lower.append(point)
        if abs(projection - upper_center) <= threshold:
            upper.append(point)
    return lower, upper


def width_profiles(config: ArenaGeometryConfig):
    if config.arena_width_m is not None:
        return "single", [("arena_single", config.arena_width_m)]
    return "dual", [
        ("heater_side_width", config.heater_side_width_m),
        ("clean_side_width", config.clean_side_width_m),
    ]


def match_width_profile(observed_wall_separation_m, config: ArenaGeometryConfig):
    mode, profiles = width_profiles(config)
    errors = {
        label: observed_wall_separation_m - expected_width
        for label, expected_width in profiles
    }
    ranked = sorted(
        profiles,
        key=lambda item: abs(errors[item[0]]),
    )
    best_label, best_width = ranked[0]
    best_abs_error = abs(errors[best_label])
    margin = None
    ambiguous = False
    if len(ranked) > 1:
        second_label, _second_width = ranked[1]
        margin = abs(errors[second_label]) - best_abs_error
        ambiguous = margin < config.width_match_min_margin_m
    return WidthMatch(
        observed_wall_separation_m=observed_wall_separation_m,
        matched_width_profile_label=best_label,
        expected_wall_width_m=best_width,
        width_error_m=errors[best_label],
        width_match_margin_m=margin,
        width_match_mode=mode,
        width_match_ambiguous=ambiguous,
        width_profile_errors_m=errors,
    )


def fit_long_walls(points: Sequence[tuple[float, float]], config: ArenaGeometryConfig):
    if len(points) < 2 * config.min_wall_points:
        return LongWallFit(False, "insufficient_points")

    threshold = max(0.06, 2.0 * config.max_line_rmse_m)
    best = None
    step = max(0.5, config.angle_search_step_deg)
    steps = int(math.ceil(180.0 / step))

    for index in range(steps):
        axis_angle = math.radians(index * step)
        normal_angle = axis_angle + math.pi / 2.0
        normal = vector_from_angle(normal_angle)
        projections = [dot(point, normal) for point in points]
        ordered = sorted(projections)
        p30 = percentile_sorted(ordered, 30.0)
        p70 = percentile_sorted(ordered, 70.0)
        lower_values = [value for value in projections if value <= p30]
        upper_values = [value for value in projections if value >= p70]
        lower_center = median(lower_values)
        upper_center = median(upper_values)
        separation = upper_center - lower_center
        lower_points, upper_points = projection_clusters(
            points,
            normal,
            lower_center,
            upper_center,
            threshold,
        )
        if len(lower_points) < config.min_wall_points or len(upper_points) < config.min_wall_points:
            continue
        try:
            lower_line = fit_line(lower_points)
            upper_line = fit_line(upper_points)
        except ValueError:
            continue
        parallel_error = math.degrees(
            undirected_angle_delta_rad(
                lower_line.direction_angle_rad,
                upper_line.direction_angle_rad,
            )
        )
        parallel_score = clamp(1.0 - parallel_error / 90.0, 0.0, 1.0)
        rmse = math.sqrt(
            (
                lower_line.rmse_m * lower_line.rmse_m
                + upper_line.rmse_m * upper_line.rmse_m
            )
            / 2.0
        )
        width_match = match_width_profile(separation, config)
        separation_error = width_match.abs_error_m
        score = (
            len(lower_points)
            + len(upper_points)
            - 250.0 * separation_error
            - 250.0 * rmse
            - 2.0 * parallel_error
        )
        candidate = (
            score,
            axis_angle,
            normal_angle,
            lower_center,
            upper_center,
            separation,
            separation_error,
            width_match,
            rmse,
            parallel_error,
            parallel_score,
            lower_points,
            upper_points,
            lower_line,
            upper_line,
        )
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return LongWallFit(False, "no_parallel_long_wall_candidates")

    (
        _score,
        axis_angle,
        normal_angle,
        lower_center,
        upper_center,
        separation,
        separation_error,
        width_match,
        rmse,
        parallel_error,
        parallel_score,
        lower_points,
        upper_points,
        lower_line,
        upper_line,
    ) = best

    if separation_error > config.max_wall_separation_error_m:
        reason = "wall_separation_out_of_tolerance"
        ok = False
    elif rmse > config.max_line_rmse_m:
        reason = "long_wall_rmse_above_threshold"
        ok = False
    elif parallel_score < config.min_parallel_score:
        reason = "long_walls_not_parallel"
        ok = False
    else:
        reason = "ok"
        ok = True

    center_projection = (lower_center + upper_center) / 2.0
    return LongWallFit(
        ok=ok,
        reason=reason,
        axis_angle_rad=normalize_undirected_angle_rad(axis_angle),
        normal_angle_rad=normalize_undirected_angle_rad(normal_angle),
        lower_wall_points=len(lower_points),
        upper_wall_points=len(upper_points),
        wall_separation_m=separation,
        wall_separation_error_m=separation_error,
        observed_wall_separation_m=width_match.observed_wall_separation_m,
        matched_width_profile_label=width_match.matched_width_profile_label,
        expected_wall_width_m=width_match.expected_wall_width_m,
        width_error_m=width_match.width_error_m,
        width_match_margin_m=width_match.width_match_margin_m,
        width_match_mode=width_match.width_match_mode,
        width_match_ambiguous=width_match.width_match_ambiguous,
        width_profile_errors_m=width_match.width_profile_errors_m,
        long_wall_rmse_m=rmse,
        parallel_angle_error_deg=parallel_error,
        parallel_score=parallel_score,
        lateral_offset_m=-center_projection,
        lower_projection_m=lower_center,
        upper_projection_m=upper_center,
        lower_line=lower_line,
        upper_line=upper_line,
    )


def wall_side_for_type(wall_type, config: ArenaGeometryConfig):
    heater = config.heater_wall_side
    clean = "-x" if heater == "+x" else "+x"
    if wall_type == WALL_HEATER:
        return heater
    if wall_type == WALL_CLEAN:
        return clean
    return None


def clipped_score(value, low, high):
    if high <= low:
        return 1.0 if value >= high else 0.0
    return clamp((value - low) / (high - low), 0.0, 1.0)


def cluster_bins(bin_indices: Sequence[int], gap_tolerance_bins: int):
    if not bin_indices:
        return []
    ordered = sorted(set(bin_indices))
    clusters = [[ordered[0], ordered[0]]]
    for bin_index in ordered[1:]:
        if bin_index - clusters[-1][1] <= gap_tolerance_bins + 1:
            clusters[-1][1] = bin_index
        else:
            clusters.append([bin_index, bin_index])
    return clusters


def compute_short_wall_profile_features(
    band_points,
    axis,
    normal,
    side,
    edge_projection,
    line: LineFit | None,
    config: ArenaGeometryConfig,
):
    if not band_points:
        return {
            "point_count": 0,
            "visible_width_m": None,
            "line_rmse_m": None,
            "edge_projection_m": edge_projection,
            "candidate_axis_sign": 1.0 if side == "axis_positive" else -1.0,
            "depth_positive_meaning": "residual_from_clean_edge_toward_arena_center",
            "protrusion_depth_p90_m": 0.0,
            "protrusion_width_coverage_fraction": 0.0,
            "dominant_cluster_width_fraction": 0.0,
            "flat_outer_support_fraction": 0.0,
            "clean_rail_artifact_score": 0.0,
            "broad_flat_wall_artifact_score": 0.0,
            "validity_failed_reason": "profile_point_count_too_low",
        }

    axis_sign = 1.0 if side == "axis_positive" else -1.0
    width_values = [dot(point, normal) for point in band_points]
    depth_values = [
        axis_sign * (edge_projection - dot(point, axis))
        for point in band_points
    ]
    sorted_depth = sorted(depth_values)
    visible_width = max(width_values) - min(width_values)
    depth_p75 = percentile_sorted(sorted_depth, 75.0)
    depth_p90 = percentile_sorted(sorted_depth, 90.0)
    depth_p95 = percentile_sorted(sorted_depth, 95.0)
    depth_p10 = percentile_sorted(sorted_depth, 10.0)
    profile_roughness = max(0.0, depth_p90 - depth_p10)
    outer_support_count = sum(
        1
        for depth in depth_values
        if abs(depth) <= config.short_wall_outer_band_m
    )
    outer_line_support_fraction = outer_support_count / len(depth_values)

    min_width = min(width_values)
    protrusion_bins = []
    protrusion_depths = []
    for width, depth in zip(width_values, depth_values):
        if depth > config.profile_protrusion_min_m:
            protrusion_depths.append(depth)
            protrusion_bins.append(
                int(math.floor((width - min_width) / config.profile_cluster_bin_width_m))
            )
    clusters = cluster_bins(protrusion_bins, config.profile_cluster_gap_tolerance_bins)
    largest_cluster_width = 0.0
    if clusters:
        largest_cluster_width = max(
            (end - start + 1) * config.profile_cluster_bin_width_m
            for start, end in clusters
        )
    protrusion_fraction = len(protrusion_bins) / len(depth_values)
    total_width_bins = max(
        1,
        int(math.floor(visible_width / config.profile_cluster_bin_width_m)) + 1,
    )
    protrusion_width_coverage = len(set(protrusion_bins)) / total_width_bins
    dominant_cluster_width_fraction = (
        0.0 if visible_width <= 0.0 else clamp(largest_cluster_width / visible_width, 0.0, 1.0)
    )
    protrusion_depth_p90 = (
        percentile_sorted(sorted(protrusion_depths), 90.0)
        if protrusion_depths
        else 0.0
    )
    flat_outer_support_fraction = outer_line_support_fraction
    if protrusion_bins:
        flat_support_score = clipped_score(
            flat_outer_support_fraction,
            config.profile_clean_rail_flat_support_min,
            1.0,
        )
        low_protrusion_score = 1.0 - clipped_score(
            protrusion_fraction,
            0.0,
            config.profile_clean_rail_protrusion_fraction_max,
        )
        shallow_depth_score = 1.0 - clipped_score(
            protrusion_depth_p90,
            config.profile_protrusion_min_m,
            config.profile_clean_rail_depth_p90_max_m,
        )
        dominant_cluster_score = clipped_score(
            dominant_cluster_width_fraction,
            config.profile_clean_rail_dominant_cluster_fraction_min,
            1.0,
        )
        clean_rail_artifact_score = min(
            flat_support_score,
            low_protrusion_score,
            shallow_depth_score,
            dominant_cluster_score,
        )
    else:
        clean_rail_artifact_score = 0.0
    if clusters:
        broad_flat_support_score = clipped_score(
            flat_outer_support_fraction,
            config.profile_broad_flat_wall_flat_support_min,
            1.0,
        )
        broad_dominant_cluster_score = clipped_score(
            dominant_cluster_width_fraction,
            config.profile_broad_flat_wall_dominant_cluster_fraction_min,
            1.0,
        )
        broad_cluster_width_score = clipped_score(
            largest_cluster_width,
            config.profile_broad_flat_wall_cluster_width_min_m,
            max(config.profile_broad_flat_wall_cluster_width_min_m, visible_width),
        )
        low_cluster_count_score = 1.0 - clipped_score(
            float(len(clusters)),
            float(config.profile_broad_flat_wall_cluster_count_max),
            float(config.profile_broad_flat_wall_cluster_count_max + 2),
        )
        broad_flat_wall_artifact_score = min(
            broad_flat_support_score,
            broad_dominant_cluster_score,
            broad_cluster_width_score,
            low_cluster_count_score,
        )
    else:
        broad_flat_wall_artifact_score = 0.0

    validity_failed_reason = None
    line_rmse = None if line is None else line.rmse_m
    if len(band_points) < config.profile_min_points:
        validity_failed_reason = "profile_point_count_too_low"
    elif visible_width < config.profile_min_visible_width_m:
        validity_failed_reason = "profile_visible_width_too_low"
    elif line_rmse is None:
        validity_failed_reason = "profile_line_fit_unavailable"
    elif line_rmse > config.profile_max_line_rmse_m:
        validity_failed_reason = "profile_line_rmse_too_high"
    elif outer_line_support_fraction < config.profile_min_any_line_support_fraction:
        validity_failed_reason = "profile_line_support_too_low"

    return {
        "point_count": len(band_points),
        "visible_width_m": visible_width,
        "line_rmse_m": line_rmse,
        "edge_projection_m": edge_projection,
        "candidate_axis_sign": axis_sign,
        "depth_positive_meaning": "residual_from_clean_edge_toward_arena_center",
        "depth_p75_m": depth_p75,
        "depth_p90_m": depth_p90,
        "depth_p95_m": depth_p95,
        "protrusion_fraction": protrusion_fraction,
        "protrusion_cluster_count": len(clusters),
        "largest_protrusion_cluster_width_m": largest_cluster_width,
        "protrusion_depth_p90_m": protrusion_depth_p90,
        "protrusion_width_coverage_fraction": protrusion_width_coverage,
        "dominant_cluster_width_fraction": dominant_cluster_width_fraction,
        "flat_outer_support_fraction": flat_outer_support_fraction,
        "clean_rail_artifact_score": clean_rail_artifact_score,
        "broad_flat_wall_artifact_score": broad_flat_wall_artifact_score,
        "profile_roughness_m": profile_roughness,
        "outer_line_support_fraction": outer_line_support_fraction,
        "validity_failed_reason": validity_failed_reason,
    }


def score_short_wall_profile(features, config: ArenaGeometryConfig):
    if not features:
        return 0.0, 0.0
    protrusion_fraction = features.get("protrusion_fraction") or 0.0
    width_coverage = features.get("protrusion_width_coverage_fraction") or 0.0
    protrusion_depth_p90 = features.get("protrusion_depth_p90_m") or 0.0
    roughness = features.get("profile_roughness_m") or 0.0
    line_support = features.get("outer_line_support_fraction") or 0.0
    flat_support = features.get("flat_outer_support_fraction")
    if flat_support is None:
        flat_support = line_support
    clean_rail_artifact_score = features.get("clean_rail_artifact_score") or 0.0
    broad_flat_wall_artifact_score = (
        features.get("broad_flat_wall_artifact_score") or 0.0
    )

    protrusion_fraction_score = clipped_score(
        protrusion_fraction,
        config.profile_heater_protrusion_fraction_low,
        config.profile_heater_protrusion_fraction_high,
    )
    width_coverage_score = clipped_score(
        width_coverage,
        config.profile_heater_width_coverage_low,
        config.profile_heater_width_coverage_high,
    )
    protrusion_depth_score = clipped_score(
        protrusion_depth_p90,
        config.profile_heater_protrusion_depth_p90_low_m,
        config.profile_heater_protrusion_depth_p90_high_m,
    )
    low_flat_support_score = 1.0 - clipped_score(
        flat_support,
        config.profile_heater_line_support_low,
        config.profile_heater_line_support_high,
    )
    roughness_score = clipped_score(
        roughness,
        config.profile_heater_roughness_low_m,
        config.profile_heater_roughness_high_m,
    )
    high_flat_support_penalty_score = clipped_score(
        flat_support,
        config.profile_heater_high_flat_support_penalty_low,
        config.profile_heater_high_flat_support_penalty_high,
    )
    heater_score = clamp(
        config.profile_heater_weight_protrusion_fraction * protrusion_fraction_score
        + config.profile_heater_weight_width_coverage * width_coverage_score
        + config.profile_heater_weight_depth * protrusion_depth_score
        + config.profile_heater_weight_low_flat_support * low_flat_support_score
        + config.profile_heater_weight_roughness_refined * roughness_score
        - config.profile_heater_clean_rail_penalty * clean_rail_artifact_score,
        0.0,
        1.0,
    )
    heater_score = clamp(
        heater_score
        - config.profile_heater_high_flat_support_penalty
        * high_flat_support_penalty_score,
        0.0,
        1.0,
    )
    if broad_flat_wall_artifact_score > 0.0:
        heater_score = clamp(
            heater_score
            - config.profile_broad_flat_wall_penalty * broad_flat_wall_artifact_score,
            0.0,
            config.profile_broad_flat_wall_heater_cap,
        )

    clean_score = sum(
        [
            clipped_score(
                flat_support,
                config.profile_clean_line_support_low,
                config.profile_clean_line_support_high,
            ),
            1.0
            - clipped_score(
                protrusion_fraction,
                config.profile_clean_protrusion_fraction_low,
                config.profile_clean_protrusion_fraction_high,
            ),
            1.0
            - clipped_score(
                width_coverage,
                config.profile_heater_width_coverage_low,
                config.profile_heater_width_coverage_high,
            ),
            1.0
            - clipped_score(
                protrusion_depth_p90,
                config.profile_heater_protrusion_depth_p90_low_m,
                config.profile_heater_protrusion_depth_p90_high_m,
            ),
        ]
    ) / 4.0
    return heater_score, clean_score


def is_profile_heater_like(candidate: ShortWallClassification, config: ArenaGeometryConfig):
    return (
        candidate.heater_profile_score >= config.profile_min_heater_like_score
        and candidate.clean_profile_score <= config.profile_max_opposite_score
    )


def is_profile_clean_like(candidate: ShortWallClassification, config: ArenaGeometryConfig):
    return (
        candidate.clean_profile_score >= config.profile_min_clean_like_score
        and candidate.heater_profile_score <= config.profile_max_opposite_score
    )


def is_profile_weak(candidate: ShortWallClassification, config: ArenaGeometryConfig):
    return (
        candidate.heater_profile_score < config.profile_min_heater_like_score
        and candidate.clean_profile_score < config.profile_min_clean_like_score
    )


def classify_short_wall_relative_heater(
    negative: ShortWallClassification,
    positive: ShortWallClassification,
    common: dict,
    config: ArenaGeometryConfig,
):
    if negative.heater_profile_score >= positive.heater_profile_score:
        selected = negative
        opposite = positive
        assignment = "negative_heater"
    else:
        selected = positive
        opposite = negative
        assignment = "positive_heater"

    selected_features = selected.profile_features or {}
    selected_heater_score = selected.heater_profile_score
    opposite_heater_score = opposite.heater_profile_score
    heater_delta = selected_heater_score - opposite_heater_score
    protrusion_clusters = int(selected_features.get("protrusion_cluster_count") or 0)
    protrusion_fraction = selected_features.get("protrusion_fraction") or 0.0

    relative_common = {
        **common,
        "assignment": assignment,
        "confidence": min(selected_heater_score, config.profile_relative_confidence_cap),
        "margin": heater_delta,
        "winner_score": selected_heater_score,
        "loser_score": opposite_heater_score,
        "heater_profile_delta": heater_delta,
        "relative_heater_score": selected_heater_score,
        "relative_opposite_heater_score": opposite_heater_score,
        "relative_opposite_max_heater_score": config.profile_relative_opposite_max_heater_score,
        "relative_min_protrusion_clusters": config.profile_relative_min_protrusion_clusters,
        "relative_min_protrusion_fraction": config.profile_relative_min_protrusion_fraction,
        "relative_confidence_raw": selected_heater_score,
    }

    if selected_heater_score < config.profile_relative_heater_min_score:
        return PairwiseShortWallClassification(
            reason="pairwise_profile_relative_heater_score_too_low",
            **relative_common,
        )
    if opposite_heater_score > config.profile_relative_opposite_max_heater_score:
        return PairwiseShortWallClassification(
            reason="pairwise_profile_relative_opposite_too_heater_like",
            **relative_common,
        )
    if heater_delta < config.profile_relative_heater_min_delta:
        return PairwiseShortWallClassification(
            reason="pairwise_profile_relative_heater_delta_too_low",
            **relative_common,
        )
    if selected.clean_profile_score > config.profile_relative_selected_max_clean_score:
        return PairwiseShortWallClassification(
            reason="pairwise_profile_relative_selected_contradictory",
            **relative_common,
        )
    if (
        protrusion_clusters < config.profile_relative_min_protrusion_clusters
        or protrusion_fraction < config.profile_relative_min_protrusion_fraction
    ):
        return PairwiseShortWallClassification(
            reason="pairwise_profile_relative_evidence_not_distributed",
            **relative_common,
        )

    negative_type = WALL_HEATER if assignment == "negative_heater" else WALL_CLEAN
    positive_type = WALL_CLEAN if assignment == "negative_heater" else WALL_HEATER
    return PairwiseShortWallClassification(
        **{**relative_common, "accepted": True},
        reason="pairwise_profile_relative_heater_valid",
        negative_wall_type=negative_type,
        positive_wall_type=positive_type,
    )


def short_wall_profile_validity_failure(candidate: ShortWallClassification):
    failed = candidate.validity_failed_reason
    if failed is None and candidate.profile_features is not None:
        failed = candidate.profile_features.get("validity_failed_reason")
    return failed


def allows_relaxed_heater_profile_rmse(
    candidate: ShortWallClassification,
    config: ArenaGeometryConfig,
):
    if short_wall_profile_validity_failure(candidate) != "profile_line_rmse_too_high":
        return False

    features = candidate.profile_features or {}
    line_rmse = features.get("line_rmse_m")
    if line_rmse is None or line_rmse > config.profile_heater_relaxed_max_line_rmse_m:
        return False

    protrusion_clusters = int(features.get("protrusion_cluster_count") or 0)
    protrusion_fraction = features.get("protrusion_fraction") or 0.0
    return (
        candidate.heater_profile_score >= config.profile_relative_heater_min_score
        and candidate.clean_profile_score <= config.profile_relative_selected_max_clean_score
        and protrusion_clusters >= config.profile_relative_min_protrusion_clusters
        and protrusion_fraction >= config.profile_relative_min_protrusion_fraction
    )


def classify_short_wall_pairwise(
    candidates: dict[str, ShortWallClassification],
    config: ArenaGeometryConfig,
):
    negative = candidates.get("axis_negative")
    positive = candidates.get("axis_positive")
    if negative is None or positive is None:
        return PairwiseShortWallClassification(False, False, "pairwise_profile_missing_candidate")
    if negative.profile_features is None or positive.profile_features is None:
        return PairwiseShortWallClassification(False, False, "pairwise_profile_unavailable")

    if (
        negative.short_wall_candidate_range_m is None
        or positive.short_wall_candidate_range_m is None
    ):
        return PairwiseShortWallClassification(
            True,
            False,
            "pairwise_profile_range_missing",
            range_sum_expected_m=config.arena_length_m,
            range_sum_tolerance_m=config.max_short_wall_range_sum_error_m,
        )

    range_sum = negative.short_wall_candidate_range_m + positive.short_wall_candidate_range_m
    range_sum_error = range_sum - config.arena_length_m
    abs_range_sum_error = abs(range_sum_error)

    for candidate in (negative, positive):
        failed = short_wall_profile_validity_failure(candidate)
        if failed is not None and not allows_relaxed_heater_profile_rmse(candidate, config):
            return PairwiseShortWallClassification(
                True,
                False,
                "pairwise_profile_candidate_invalid",
                confidence=0.0,
                range_sum_m=range_sum,
                range_sum_expected_m=config.arena_length_m,
                range_sum_error_m=abs_range_sum_error,
                range_sum_tolerance_m=config.max_short_wall_range_sum_error_m,
            )

    if abs_range_sum_error > config.max_short_wall_range_sum_error_m:
        reason = (
            "pairwise_profile_range_sum_too_short"
            if range_sum < config.arena_length_m
            else "pairwise_profile_range_sum_too_long"
        )
        return PairwiseShortWallClassification(
            True,
            False,
            reason,
            range_sum_m=range_sum,
            range_sum_expected_m=config.arena_length_m,
            range_sum_error_m=abs_range_sum_error,
            range_sum_tolerance_m=config.max_short_wall_range_sum_error_m,
        )

    negative_heater_like = is_profile_heater_like(negative, config)
    positive_heater_like = is_profile_heater_like(positive, config)
    negative_clean_like = is_profile_clean_like(negative, config)
    positive_clean_like = is_profile_clean_like(positive, config)

    score_neg_heater = negative.heater_profile_score + positive.clean_profile_score
    score_pos_heater = negative.clean_profile_score + positive.heater_profile_score
    if score_neg_heater >= score_pos_heater:
        assignment = "negative_heater"
        winner_score = score_neg_heater
        loser_score = score_pos_heater
    else:
        assignment = "positive_heater"
        winner_score = score_pos_heater
        loser_score = score_neg_heater
    confidence = winner_score / 2.0
    margin = (winner_score - loser_score) / 2.0
    contrast = min(
        abs(negative.heater_profile_score - positive.heater_profile_score),
        abs(negative.clean_profile_score - positive.clean_profile_score),
    )
    heater_delta = abs(negative.heater_profile_score - positive.heater_profile_score)

    common = {
        "applicable": True,
        "accepted": False,
        "assignment": assignment,
        "confidence": confidence,
        "margin": margin,
        "heater_clean_contrast": contrast,
        "winner_score": winner_score,
        "loser_score": loser_score,
        "range_sum_m": range_sum,
        "range_sum_expected_m": config.arena_length_m,
        "range_sum_error_m": abs_range_sum_error,
        "range_sum_tolerance_m": config.max_short_wall_range_sum_error_m,
        "heater_profile_delta": heater_delta,
    }

    if negative_heater_like and positive_heater_like:
        return classify_short_wall_relative_heater(negative, positive, common, config)
    if negative_clean_like and positive_clean_like:
        return PairwiseShortWallClassification(reason="pairwise_profile_both_clean_like", **common)
    if is_profile_weak(negative, config) and is_profile_weak(positive, config):
        return classify_short_wall_relative_heater(negative, positive, common, config)

    expected_assignment = None
    if negative_heater_like and positive_clean_like:
        expected_assignment = "negative_heater"
    elif positive_heater_like and negative_clean_like:
        expected_assignment = "positive_heater"
    else:
        return classify_short_wall_relative_heater(negative, positive, common, config)

    if assignment != expected_assignment:
        return PairwiseShortWallClassification(
            reason="pairwise_profile_assignment_label_mismatch",
            **common,
        )
    if confidence < config.profile_min_confidence:
        return PairwiseShortWallClassification(reason="pairwise_profile_confidence_too_low", **common)
    if margin < config.profile_min_assignment_margin:
        return PairwiseShortWallClassification(reason="pairwise_profile_margin_too_low", **common)
    if contrast < config.profile_min_heater_clean_contrast:
        return PairwiseShortWallClassification(reason="pairwise_profile_contrast_too_low", **common)

    negative_type = WALL_HEATER if assignment == "negative_heater" else WALL_CLEAN
    positive_type = WALL_CLEAN if assignment == "negative_heater" else WALL_HEATER
    return PairwiseShortWallClassification(
        **{**common, "accepted": True},
        reason="pairwise_profile_heater_clean_valid",
        negative_wall_type=negative_type,
        positive_wall_type=positive_type,
    )


def copy_candidate_with_pairwise_result(
    candidate: ShortWallClassification,
    pairwise: PairwiseShortWallClassification,
    wall_type=None,
):
    raw_validity_failed_reason = short_wall_profile_validity_failure(candidate)
    validity_failed_reason = None if pairwise.accepted else raw_validity_failed_reason
    profile_features = candidate.profile_features
    if pairwise.accepted and raw_validity_failed_reason is not None:
        profile_features = dict(candidate.profile_features or {})
        profile_features["raw_validity_failed_reason"] = raw_validity_failed_reason
        profile_features["validity_failed_reason"] = None
        profile_features["relaxed_validity_reason"] = "accepted_by_pairwise_classifier"

    return ShortWallClassification(
        wall_type=wall_type or candidate.wall_type,
        reason=pairwise.reason,
        observed_axis_side=candidate.observed_axis_side,
        confidence=pairwise.confidence,
        heater_feature_score=candidate.heater_profile_score,
        clean_feature_score=candidate.clean_profile_score,
        classification_margin=pairwise.margin,
        short_wall_candidate_range_m=candidate.short_wall_candidate_range_m,
        short_wall_visible_width_m=candidate.short_wall_visible_width_m,
        short_wall_rmse_m=candidate.short_wall_rmse_m,
        short_wall_range_sum_m=pairwise.range_sum_m,
        short_wall_range_sum_error_m=pairwise.range_sum_error_m,
        point_count=candidate.point_count,
        profile_features=profile_features,
        heater_profile_score=candidate.heater_profile_score,
        clean_profile_score=candidate.clean_profile_score,
        pairwise_assignment_score=pairwise.winner_score,
        pairwise_assignment_margin=pairwise.margin,
        heater_clean_contrast=pairwise.heater_clean_contrast,
        short_wall_range_sum_expected_m=pairwise.range_sum_expected_m,
        short_wall_range_sum_tolerance_m=pairwise.range_sum_tolerance_m,
        selected_assignment=pairwise.assignment,
        validity_failed_reason=validity_failed_reason,
        heater_profile_delta=pairwise.heater_profile_delta,
        relative_heater_score=pairwise.relative_heater_score,
        relative_opposite_heater_score=pairwise.relative_opposite_heater_score,
        relative_opposite_max_heater_score=pairwise.relative_opposite_max_heater_score,
        relative_min_protrusion_clusters=pairwise.relative_min_protrusion_clusters,
        relative_min_protrusion_fraction=pairwise.relative_min_protrusion_fraction,
        relative_confidence_raw=pairwise.relative_confidence_raw,
    )


def pairwise_result_to_classification(
    candidates: dict[str, ShortWallClassification],
    pairwise: PairwiseShortWallClassification,
):
    if pairwise.accepted:
        heater_side = (
            "axis_negative"
            if pairwise.assignment == "negative_heater"
            else "axis_positive"
        )
        return copy_candidate_with_pairwise_result(
            candidates[heater_side],
            pairwise,
            wall_type=WALL_HEATER,
        )
    best = max(
        candidates.values(),
        key=lambda item: max(item.heater_profile_score, item.clean_profile_score),
    )
    return copy_candidate_with_pairwise_result(best, pairwise, wall_type=WALL_UNKNOWN)


def annotate_pairwise_candidates(
    candidates: dict[str, ShortWallClassification],
    pairwise: PairwiseShortWallClassification,
):
    if not pairwise.applicable:
        return candidates
    annotated = {}
    for side, candidate in candidates.items():
        wall_type = WALL_UNKNOWN
        if pairwise.accepted:
            if side == "axis_negative":
                wall_type = pairwise.negative_wall_type
            elif side == "axis_positive":
                wall_type = pairwise.positive_wall_type
        annotated[side] = copy_candidate_with_pairwise_result(candidate, pairwise, wall_type)
    return annotated


def classify_candidate(
    points,
    axis,
    normal,
    side,
    edge_projection,
    config: ArenaGeometryConfig,
):
    band_points = []
    for point in points:
        t = dot(point, axis)
        if abs(t - edge_projection) <= config.short_wall_band_m:
            band_points.append(point)
    if len(band_points) < config.min_short_wall_points:
        profile_features = compute_short_wall_profile_features(
            band_points,
            axis,
            normal,
            side,
            edge_projection,
            None,
            config,
        )
        return ShortWallClassification(
            WALL_UNKNOWN,
            "insufficient_short_wall_points",
            observed_axis_side=side,
            short_wall_candidate_range_m=abs(edge_projection),
            point_count=len(band_points),
            profile_features=profile_features,
            validity_failed_reason=profile_features.get("validity_failed_reason"),
        )

    normal_values = [dot(point, normal) for point in band_points]
    visible_width = max(normal_values) - min(normal_values)
    outer_points = [
        point
        for point in band_points
        if abs(dot(point, axis) - edge_projection) <= config.short_wall_outer_band_m
    ]
    if len(outer_points) < max(3, config.min_short_wall_points // 2):
        outer_points = band_points
    try:
        line = fit_line(outer_points)
    except ValueError:
        profile_features = compute_short_wall_profile_features(
            band_points,
            axis,
            normal,
            side,
            edge_projection,
            None,
            config,
        )
        return ShortWallClassification(
            WALL_UNKNOWN,
            "short_wall_line_fit_failed",
            observed_axis_side=side,
            short_wall_candidate_range_m=abs(edge_projection),
            short_wall_visible_width_m=visible_width,
            point_count=len(band_points),
            profile_features=profile_features,
            validity_failed_reason=profile_features.get("validity_failed_reason"),
        )

    profile_features = compute_short_wall_profile_features(
        band_points,
        axis,
        normal,
        side,
        edge_projection,
        line,
        config,
    )
    heater_score, clean_score = score_short_wall_profile(profile_features, config)
    profile_margin = abs(heater_score - clean_score)
    if is_profile_heater_like(
        ShortWallClassification(
            WALL_UNKNOWN,
            "profile_candidate_scoring",
            heater_profile_score=heater_score,
            clean_profile_score=clean_score,
        ),
        config,
    ):
        profile_wall_type = WALL_HEATER
        profile_reason = "profile_candidate_heater_like"
    elif is_profile_clean_like(
        ShortWallClassification(
            WALL_UNKNOWN,
            "profile_candidate_scoring",
            heater_profile_score=heater_score,
            clean_profile_score=clean_score,
        ),
        config,
    ):
        profile_wall_type = WALL_CLEAN
        profile_reason = "profile_candidate_clean_like"
    else:
        profile_wall_type = WALL_UNKNOWN
        profile_reason = "profile_candidate_ambiguous_scores"

    validity_failed_reason = profile_features.get("validity_failed_reason")
    if validity_failed_reason is not None:
        profile_wall_type = WALL_UNKNOWN
        profile_reason = validity_failed_reason

    return ShortWallClassification(
        wall_type=profile_wall_type,
        reason=profile_reason,
        observed_axis_side=side,
        confidence=max(heater_score, clean_score),
        heater_feature_score=heater_score,
        clean_feature_score=clean_score,
        classification_margin=profile_margin,
        short_wall_candidate_range_m=abs(edge_projection),
        short_wall_visible_width_m=visible_width,
        short_wall_rmse_m=line.rmse_m,
        point_count=len(band_points),
        profile_features=profile_features,
        heater_profile_score=heater_score,
        clean_profile_score=clean_score,
        validity_failed_reason=validity_failed_reason,
    )


def empty_short_wall_candidates(reason):
    return {
        "axis_negative": ShortWallClassification(
            WALL_UNKNOWN,
            reason,
            observed_axis_side="axis_negative",
        ),
        "axis_positive": ShortWallClassification(
            WALL_UNKNOWN,
            reason,
            observed_axis_side="axis_positive",
        ),
    }


def classify_short_wall_candidates(
    points: Sequence[tuple[float, float]],
    long_wall_fit: LongWallFit,
    config: ArenaGeometryConfig,
):
    if not long_wall_fit.ok or long_wall_fit.axis_angle_rad is None:
        return empty_short_wall_candidates("long_wall_fit_unavailable")

    axis = vector_from_angle(long_wall_fit.axis_angle_rad)
    normal = vector_from_angle(long_wall_fit.normal_angle_rad or long_wall_fit.axis_angle_rad + math.pi / 2.0)
    projections = [dot(point, axis) for point in points]
    lower_edge = percentile(projections, 5.0)
    upper_edge = percentile(projections, 95.0)
    return {
        "axis_negative": classify_candidate(
            points,
            axis,
            normal,
            "axis_negative",
            lower_edge,
            config,
        ),
        "axis_positive": classify_candidate(
            points,
            axis,
            normal,
            "axis_positive",
            upper_edge,
            config,
        ),
    }


def is_valid_short_wall_candidate(candidate: ShortWallClassification, config: ArenaGeometryConfig):
    if candidate.wall_type not in {WALL_HEATER, WALL_CLEAN}:
        return False
    if candidate.confidence < config.min_short_wall_confidence:
        return False
    if candidate.classification_margin < config.min_classification_margin:
        return False
    if candidate.point_count < config.min_short_wall_points:
        return False
    if candidate.short_wall_rmse_m is None:
        return False
    return candidate.short_wall_rmse_m <= config.max_line_rmse_m


def copy_candidate_with_reason(
    candidate: ShortWallClassification,
    reason,
    wall_type=None,
    short_wall_range_sum_m=None,
    short_wall_range_sum_error_m=None,
):
    return ShortWallClassification(
        wall_type=wall_type or candidate.wall_type,
        reason=reason,
        observed_axis_side=candidate.observed_axis_side,
        confidence=candidate.confidence,
        heater_feature_score=candidate.heater_feature_score,
        clean_feature_score=candidate.clean_feature_score,
        classification_margin=candidate.classification_margin,
        short_wall_candidate_range_m=candidate.short_wall_candidate_range_m,
        short_wall_visible_width_m=candidate.short_wall_visible_width_m,
        short_wall_rmse_m=candidate.short_wall_rmse_m,
        short_wall_range_sum_m=short_wall_range_sum_m,
        short_wall_range_sum_error_m=short_wall_range_sum_error_m,
        point_count=candidate.point_count,
        profile_features=candidate.profile_features,
        heater_profile_score=candidate.heater_profile_score,
        clean_profile_score=candidate.clean_profile_score,
        pairwise_assignment_score=candidate.pairwise_assignment_score,
        pairwise_assignment_margin=candidate.pairwise_assignment_margin,
        heater_clean_contrast=candidate.heater_clean_contrast,
        short_wall_range_sum_expected_m=candidate.short_wall_range_sum_expected_m,
        short_wall_range_sum_tolerance_m=candidate.short_wall_range_sum_tolerance_m,
        selected_assignment=candidate.selected_assignment,
        validity_failed_reason=candidate.validity_failed_reason,
    )


def complementary_short_wall_pair(accepted: Sequence[ShortWallClassification]):
    if len(accepted) != 2:
        return None
    heater = next((candidate for candidate in accepted if candidate.wall_type == WALL_HEATER), None)
    clean = next((candidate for candidate in accepted if candidate.wall_type == WALL_CLEAN), None)
    if heater is None or clean is None:
        return None
    if (
        heater.short_wall_candidate_range_m is None
        or clean.short_wall_candidate_range_m is None
    ):
        return None
    return heater, clean


def forced_short_wall_classification(
    candidates: dict[str, ShortWallClassification],
    config: ArenaGeometryConfig,
):
    if config.forced_short_wall_side is None and config.forced_short_wall_type is None:
        return None
    if config.forced_short_wall_side not in candidates:
        return ShortWallClassification(
            WALL_UNKNOWN,
            "forced_short_wall_side_missing",
            observed_axis_side=config.forced_short_wall_side,
        )
    if config.forced_short_wall_type not in {WALL_HEATER, WALL_CLEAN}:
        candidate = candidates[config.forced_short_wall_side]
        return copy_candidate_with_reason(
            candidate,
            "forced_short_wall_type_invalid",
            wall_type=WALL_UNKNOWN,
        )

    candidate = candidates[config.forced_short_wall_side]
    if candidate.point_count < config.min_short_wall_points:
        return copy_candidate_with_reason(
            candidate,
            "forced_short_wall_candidate_insufficient_points",
            wall_type=WALL_UNKNOWN,
        )
    if candidate.short_wall_rmse_m is None or candidate.short_wall_rmse_m > config.max_line_rmse_m:
        return copy_candidate_with_reason(
            candidate,
            "forced_short_wall_candidate_bad_fit",
            wall_type=WALL_UNKNOWN,
        )
    return copy_candidate_with_reason(
        candidate,
        "forced_short_wall_classification",
        wall_type=config.forced_short_wall_type,
    )


def select_short_wall_classification(
    candidates: dict[str, ShortWallClassification],
    config: ArenaGeometryConfig,
):
    forced = forced_short_wall_classification(candidates, config)
    if forced is not None:
        return forced

    pairwise = classify_short_wall_pairwise(candidates, config)
    if pairwise.applicable:
        return pairwise_result_to_classification(candidates, pairwise)

    ordered = list(candidates.values())
    accepted = [
        candidate
        for candidate in ordered
        if is_valid_short_wall_candidate(candidate, config)
    ]
    ambiguous = [candidate for candidate in ordered if candidate.wall_type == WALL_AMBIGUOUS]
    if len(accepted) > 1:
        complementary_pair = complementary_short_wall_pair(accepted)
        best = max(accepted, key=lambda item: item.confidence)
        if complementary_pair is None:
            return copy_candidate_with_reason(
                best,
                "both_axis_candidates_valid",
                wall_type=WALL_AMBIGUOUS,
            )
        heater, clean = complementary_pair
        range_sum = (
            heater.short_wall_candidate_range_m
            + clean.short_wall_candidate_range_m
        )
        range_sum_error = abs(range_sum - config.arena_length_m)
        if range_sum_error > config.max_short_wall_range_sum_error_m:
            return copy_candidate_with_reason(
                best,
                "short_wall_range_inconsistent",
                wall_type=WALL_AMBIGUOUS,
                short_wall_range_sum_m=range_sum,
                short_wall_range_sum_error_m=range_sum_error,
            )
        return copy_candidate_with_reason(
            heater,
            "complementary_short_walls_valid",
            short_wall_range_sum_m=range_sum,
            short_wall_range_sum_error_m=range_sum_error,
        )
    if len(accepted) == 1:
        return accepted[0]
    if ambiguous:
        return max(ambiguous, key=lambda item: item.confidence)
    return max(ordered, key=lambda item: item.confidence)


def build_pose_prior(
    points: Sequence[tuple[float, float]],
    long_wall_fit: LongWallFit,
    classification: ShortWallClassification,
    config: ArenaGeometryConfig,
    candidates: dict[str, ShortWallClassification] | None = None,
):
    if (
        long_wall_fit.axis_angle_rad is None
        or classification.observed_axis_side is None
        or classification.wall_type not in {WALL_HEATER, WALL_CLEAN}
    ):
        return None

    axis_angle = long_wall_fit.axis_angle_rad
    observed_map_side = wall_side_for_type(classification.wall_type, config)
    observed_positive = classification.observed_axis_side == "axis_positive"
    observed_is_map_positive = observed_map_side == "+x"
    local_map_plus_angle = axis_angle if observed_positive == observed_is_map_positive else axis_angle + math.pi
    local_map_plus = vector_from_angle(local_map_plus_angle)

    complementary_pair = None
    if (
        candidates is not None
        and classification.reason
        in {
            "complementary_short_walls_valid",
            "pairwise_profile_heater_clean_valid",
            "pairwise_profile_relative_heater_valid",
        }
    ):
        complementary_pair = complementary_short_wall_pair(list(candidates.values()))

    if complementary_pair is not None:
        heater, clean = complementary_pair
        heater_range = heater.short_wall_candidate_range_m or 0.0
        clean_range = clean.short_wall_candidate_range_m or 0.0
        if config.heater_wall_side == "+x":
            robot_x_arena = (clean_range - heater_range) / 2.0
        else:
            robot_x_arena = (heater_range - clean_range) / 2.0
    else:
        axis_values = [dot(point, local_map_plus) for point in points]
        if observed_map_side == "+x":
            wall_coord = percentile(axis_values, 95.0)
            robot_x_arena = config.arena_length_m / 2.0 - wall_coord
        else:
            wall_coord = percentile(axis_values, 5.0)
            robot_x_arena = -config.arena_length_m / 2.0 - wall_coord

    center_projection = (
        (long_wall_fit.lower_projection_m or 0.0)
        + (long_wall_fit.upper_projection_m or 0.0)
    ) / 2.0
    robot_y_arena = -center_projection

    map_yaw = math.radians(config.map_yaw_deg)
    map_x = (
        config.map_center_x
        + robot_x_arena * math.cos(map_yaw)
        - robot_y_arena * math.sin(map_yaw)
    )
    map_y = (
        config.map_center_y
        + robot_x_arena * math.sin(map_yaw)
        + robot_y_arena * math.cos(map_yaw)
    )
    robot_yaw = normalize_angle_rad(map_yaw - local_map_plus_angle)
    return Pose2D(map_x, map_y, math.degrees(robot_yaw))


def estimate_covariance(long_wall_fit: LongWallFit, classification: ShortWallClassification):
    if (
        long_wall_fit.long_wall_rmse_m is None
        or long_wall_fit.wall_separation_error_m is None
        or long_wall_fit.parallel_angle_error_deg is None
    ):
        return None
    yaw_std = max(
        math.radians(2.0),
        math.radians(long_wall_fit.parallel_angle_error_deg)
        + long_wall_fit.long_wall_rmse_m,
    )
    y_std = max(
        0.03,
        long_wall_fit.long_wall_rmse_m + long_wall_fit.wall_separation_error_m / 2.0,
    )
    short_wall_rmse = classification.short_wall_rmse_m or 0.15
    margin_factor = 1.0 - clamp(classification.classification_margin, 0.0, 1.0)
    x_std = max(0.20, short_wall_rmse + 0.50 * margin_factor)
    return {
        "x_m2": x_std * x_std,
        "y_m2": y_std * y_std,
        "yaw_rad2": yaw_std * yaw_std,
    }


def analyze_points(points: Sequence[tuple[float, float]], config: ArenaGeometryConfig | None = None):
    config = config or ArenaGeometryConfig()
    long_fit = fit_long_walls(points, config)
    if not long_fit.ok:
        classification = ShortWallClassification(WALL_UNKNOWN, "long_wall_fit_failed")
        candidates = empty_short_wall_candidates("long_wall_fit_failed")
        return ArenaGeometryResult(
            success=False,
            failure_reason=long_fit.reason,
            pose_unique=False,
            yaw_ambiguity_resolved=False,
            estimated_pose_prior=None,
            estimated_covariance=None,
            long_wall_fit=long_fit,
            short_wall_classification=classification,
            short_wall_candidates=candidates,
            diagnostics={
                "num_scan_samples_used": 0,
                "num_points_used": len(points),
            },
        )

    candidates = classify_short_wall_candidates(points, long_fit, config)
    classification = select_short_wall_classification(candidates, config)
    if classification.reason.startswith("pairwise_profile_"):
        pairwise = classify_short_wall_pairwise(candidates, config)
        candidates = annotate_pairwise_candidates(candidates, pairwise)
        classification = pairwise_result_to_classification(candidates, pairwise)
    pose_unique = classification.wall_type in {WALL_HEATER, WALL_CLEAN}
    pose = (
        build_pose_prior(points, long_fit, classification, config, candidates)
        if pose_unique
        else None
    )
    covariance = estimate_covariance(long_fit, classification) if pose_unique else None
    if pose_unique and pose is not None and covariance is not None:
        success = True
        failure_reason = ""
    else:
        success = False
        failure_reason = "pose_not_unique"

    return ArenaGeometryResult(
        success=success,
        failure_reason=failure_reason,
        pose_unique=pose_unique,
        yaw_ambiguity_resolved=pose_unique,
        estimated_pose_prior=pose,
        estimated_covariance=covariance,
        long_wall_fit=long_fit,
        short_wall_classification=classification,
        short_wall_candidates=candidates,
        diagnostics={
            "num_scan_samples_used": 0,
            "num_points_used": len(points),
            "wall_width_estimate_m": long_fit.wall_separation_m,
            "lateral_offset_m": long_fit.lateral_offset_m,
            "yaw_axis_estimate_deg": (
                None if long_fit.axis_angle_rad is None else math.degrees(long_fit.axis_angle_rad)
            ),
        },
    )


def analyze_scan_samples(
    samples: Sequence[ScanSample],
    config: ArenaGeometryConfig | None = None,
    range_stride=1,
    max_points=None,
):
    points = accumulate_scan_points(samples, range_stride=range_stride, max_points=max_points)
    result = analyze_points(points, config)
    diagnostics = dict(result.diagnostics)
    diagnostics["num_scan_samples_used"] = len(samples)
    return ArenaGeometryResult(
        success=result.success,
        failure_reason=result.failure_reason,
        pose_unique=result.pose_unique,
        yaw_ambiguity_resolved=result.yaw_ambiguity_resolved,
        estimated_pose_prior=result.estimated_pose_prior,
        estimated_covariance=result.estimated_covariance,
        long_wall_fit=result.long_wall_fit,
        short_wall_classification=result.short_wall_classification,
        short_wall_candidates=result.short_wall_candidates,
        diagnostics=diagnostics,
    )


def write_json(path: Path | str, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def load_scan_samples_json(path: Path | str):
    path = Path(path)
    with path.open() as file:
        data = json.load(file)
    samples = []
    for row in data.get("scan_samples", data if isinstance(data, list) else []):
        odom = row.get("odom_pose")
        samples.append(
            ScanSample(
                ranges=row["ranges"],
                angle_min=float(row["angle_min"]),
                angle_increment=float(row["angle_increment"]),
                range_min=float(row.get("range_min", 0.0)),
                range_max=float(row.get("range_max", float("inf"))),
                odom_pose=(
                    None
                    if odom is None
                    else Pose2D(
                        float(odom.get("x", 0.0)),
                        float(odom.get("y", 0.0)),
                        float(odom.get("yaw_deg", 0.0)),
                    )
                ),
            )
        )
    return samples


def iter_points(points: Iterable[tuple[float, float]]):
    for x, y in points:
        yield float(x), float(y)
