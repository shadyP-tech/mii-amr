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
    arena_width_m: float = 1.898
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
    short_wall_band_m: float = 0.25
    short_wall_outer_band_m: float = 0.06
    min_short_wall_visible_width_m: float = 0.35
    heater_protrusion_min_m: float = 0.06
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
    point_count: int = 0

    def to_dict(self):
        return {
            "wall_type": self.wall_type,
            "reason": self.reason,
            "observed_axis_side": self.observed_axis_side,
            "confidence": self.confidence,
            "heater_feature_score": self.heater_feature_score,
            "clean_feature_score": self.clean_feature_score,
            "classification_margin": self.classification_margin,
            "short_wall_candidate_range_m": self.short_wall_candidate_range_m,
            "short_wall_visible_width_m": self.short_wall_visible_width_m,
            "short_wall_rmse_m": self.short_wall_rmse_m,
            "point_count": self.point_count,
        }


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
            "short_wall_classification": self.short_wall_classification.to_dict(),
            "diagnostics": self.diagnostics,
        }


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
        separation_error = abs(separation - config.arena_width_m)
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


def classify_candidate(
    points,
    axis,
    normal,
    side,
    edge_projection,
    config: ArenaGeometryConfig,
):
    sign = 1.0 if side == "axis_positive" else -1.0
    band_points = []
    for point in points:
        t = dot(point, axis)
        if abs(t - edge_projection) <= config.short_wall_band_m:
            band_points.append(point)
    if len(band_points) < config.min_short_wall_points:
        return ShortWallClassification(WALL_UNKNOWN, "insufficient_short_wall_points")

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
        return ShortWallClassification(WALL_UNKNOWN, "short_wall_line_fit_failed")

    short_wall_expected_angle = math.atan2(normal[1], normal[0])
    perpendicular_error = math.degrees(
        undirected_angle_delta_rad(line.direction_angle_rad, short_wall_expected_angle)
    )
    geometric_quality = clamp(1.0 - line.rmse_m / config.max_line_rmse_m, 0.0, 1.0)
    perpendicular_quality = clamp(1.0 - perpendicular_error / 30.0, 0.0, 1.0)
    coverage_quality = clamp(
        visible_width / config.min_short_wall_visible_width_m,
        0.0,
        1.0,
    )

    protrusion_count = 0
    for point in band_points:
        inward = sign * (edge_projection - dot(point, axis))
        if inward > config.heater_protrusion_min_m:
            protrusion_count += 1
    protrusion_fraction = protrusion_count / len(band_points)
    protrusion_score = clamp((protrusion_fraction - 0.05) / 0.25, 0.0, 1.0)

    base_quality = geometric_quality * perpendicular_quality * coverage_quality
    heater_score = base_quality * protrusion_score
    clean_score = base_quality * (1.0 - protrusion_score)
    margin = abs(heater_score - clean_score)

    if (
        heater_score >= config.min_short_wall_confidence
        and clean_score >= config.min_short_wall_confidence
    ):
        wall_type = WALL_AMBIGUOUS
        reason = "heater_and_clean_scores_both_high"
    elif max(heater_score, clean_score) < config.min_short_wall_confidence:
        wall_type = WALL_UNKNOWN
        reason = "classification_confidence_too_low"
    elif margin < config.min_classification_margin:
        wall_type = WALL_UNKNOWN
        reason = "classification_margin_too_low"
    elif heater_score > clean_score:
        wall_type = WALL_HEATER
        reason = "heater_score_dominant"
    else:
        wall_type = WALL_CLEAN
        reason = "clean_score_dominant"

    return ShortWallClassification(
        wall_type=wall_type,
        reason=reason,
        observed_axis_side=side,
        confidence=max(heater_score, clean_score),
        heater_feature_score=heater_score,
        clean_feature_score=clean_score,
        classification_margin=margin,
        short_wall_candidate_range_m=abs(edge_projection),
        short_wall_visible_width_m=visible_width,
        short_wall_rmse_m=line.rmse_m,
        point_count=len(band_points),
    )


def classify_short_wall(
    points: Sequence[tuple[float, float]],
    long_wall_fit: LongWallFit,
    config: ArenaGeometryConfig,
):
    if not long_wall_fit.ok or long_wall_fit.axis_angle_rad is None:
        return ShortWallClassification(WALL_UNKNOWN, "long_wall_fit_unavailable")

    axis = vector_from_angle(long_wall_fit.axis_angle_rad)
    normal = vector_from_angle(long_wall_fit.normal_angle_rad or long_wall_fit.axis_angle_rad + math.pi / 2.0)
    projections = [dot(point, axis) for point in points]
    lower_edge = percentile(projections, 5.0)
    upper_edge = percentile(projections, 95.0)
    candidates = [
        classify_candidate(points, axis, normal, "axis_negative", lower_edge, config),
        classify_candidate(points, axis, normal, "axis_positive", upper_edge, config),
    ]
    accepted = [
        candidate
        for candidate in candidates
        if candidate.wall_type in {WALL_HEATER, WALL_CLEAN}
    ]
    ambiguous = [candidate for candidate in candidates if candidate.wall_type == WALL_AMBIGUOUS]
    if ambiguous and not accepted:
        return max(ambiguous, key=lambda item: item.confidence)
    if not accepted:
        return max(candidates, key=lambda item: item.confidence)
    accepted.sort(key=lambda item: item.confidence, reverse=True)
    if len(accepted) > 1 and accepted[0].confidence - accepted[1].confidence < config.min_classification_margin:
        best = accepted[0]
        return ShortWallClassification(
            wall_type=WALL_AMBIGUOUS,
            reason="multiple_short_wall_candidates_with_low_margin",
            observed_axis_side=best.observed_axis_side,
            confidence=best.confidence,
            heater_feature_score=best.heater_feature_score,
            clean_feature_score=best.clean_feature_score,
            classification_margin=best.classification_margin,
            short_wall_candidate_range_m=best.short_wall_candidate_range_m,
            short_wall_visible_width_m=best.short_wall_visible_width_m,
            short_wall_rmse_m=best.short_wall_rmse_m,
            point_count=best.point_count,
        )
    return accepted[0]


def build_pose_prior(
    points: Sequence[tuple[float, float]],
    long_wall_fit: LongWallFit,
    classification: ShortWallClassification,
    config: ArenaGeometryConfig,
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
        return ArenaGeometryResult(
            success=False,
            failure_reason=long_fit.reason,
            pose_unique=False,
            yaw_ambiguity_resolved=False,
            estimated_pose_prior=None,
            estimated_covariance=None,
            long_wall_fit=long_fit,
            short_wall_classification=classification,
            diagnostics={
                "num_scan_samples_used": 0,
                "num_points_used": len(points),
            },
        )

    classification = classify_short_wall(points, long_fit, config)
    pose_unique = classification.wall_type in {WALL_HEATER, WALL_CLEAN}
    pose = build_pose_prior(points, long_fit, classification, config) if pose_unique else None
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
