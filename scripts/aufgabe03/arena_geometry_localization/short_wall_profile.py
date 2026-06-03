from __future__ import annotations

import math
from typing import Sequence

from .geometry import clamp, dot, percentile_sorted
from .models import ArenaGeometryConfig, LineFit, ShortWallClassification


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


def _empty_profile_features(side, edge_projection):
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


def _compute_short_wall_profile_features_from_projected(
    projected_band_points,
    side,
    edge_projection,
    line: LineFit | None,
    config: ArenaGeometryConfig,
):
    if not projected_band_points:
        return _empty_profile_features(side, edge_projection)

    axis_sign = 1.0 if side == "axis_positive" else -1.0
    width_values = [
        normal_projection
        for _point, _axis_projection, normal_projection in projected_band_points
    ]
    depth_values = [
        axis_sign * (edge_projection - axis_projection)
        for _point, axis_projection, _normal_projection in projected_band_points
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
    if len(projected_band_points) < config.profile_min_points:
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
        "point_count": len(projected_band_points),
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


def compute_short_wall_profile_features(
    band_points,
    axis,
    normal,
    side,
    edge_projection,
    line: LineFit | None,
    config: ArenaGeometryConfig,
):
    projected_band_points = [
        (point, dot(point, axis), dot(point, normal))
        for point in band_points
    ]
    return _compute_short_wall_profile_features_from_projected(
        projected_band_points,
        side,
        edge_projection,
        line,
        config,
    )


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
