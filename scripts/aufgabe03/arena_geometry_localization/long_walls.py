from __future__ import annotations

import math
from typing import Sequence

from .geometry import (
    clamp,
    fit_line,
    median,
    normalize_undirected_angle_rad,
    percentile_sorted,
    undirected_angle_delta_rad,
    vector_from_angle,
)
from .models import ArenaGeometryConfig, LongWallFit, WidthMatch


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


def _project_points(points: Sequence[tuple[float, float]], normal):
    normal_x, normal_y = normal
    return [
        (point, point[0] * normal_x + point[1] * normal_y)
        for point in points
    ]


def _projection_clusters_from_projected(projected, lower_center, upper_center, threshold):
    lower = []
    upper = []
    for point, projection in projected:
        if abs(projection - lower_center) <= threshold:
            lower.append(point)
        if abs(projection - upper_center) <= threshold:
            upper.append(point)
    return lower, upper


def _normalize_angle_deg(angle_deg):
    return angle_deg % 180.0


def _dedupe_angles_deg(angles_deg):
    deduped = []
    seen = set()
    for angle_deg in angles_deg:
        normalized = _normalize_angle_deg(angle_deg)
        key = round(normalized, 6)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _exhaustive_angles_deg(config: ArenaGeometryConfig):
    step = max(0.5, config.angle_search_step_deg)
    steps = int(math.ceil(180.0 / step))
    return [index * step for index in range(steps)]


def _coarse_angles_deg(config: ArenaGeometryConfig):
    step = max(0.5, config.long_wall_coarse_step_deg)
    steps = int(math.ceil(180.0 / step))
    return [index * step for index in range(steps)]


def _refine_angles_deg(coarse_candidates, config: ArenaGeometryConfig):
    top_k = max(1, int(config.long_wall_refine_top_k))
    window = max(0.0, config.long_wall_refine_window_deg)
    step = max(0.5, config.long_wall_refine_step_deg)
    angles = []
    for candidate in coarse_candidates[:top_k]:
        center_deg = math.degrees(candidate[1])
        offset = -window
        while offset <= window + 1.0e-9:
            angles.append(center_deg + offset)
            offset += step
    return _dedupe_angles_deg(angles)


def _score_long_wall_angles(
    points: Sequence[tuple[float, float]],
    config: ArenaGeometryConfig,
    angles_deg,
    use_prefit_width_gate,
):
    threshold = max(0.06, 2.0 * config.max_line_rmse_m)
    best = None
    candidates = []
    angle_count = 0
    prefit_skipped_count = 0
    width_skip_threshold = (
        config.max_wall_separation_error_m
        + max(0.0, config.long_wall_prefit_width_skip_margin_m)
    )

    for angle_deg in _dedupe_angles_deg(angles_deg):
        angle_count += 1
        axis_angle = math.radians(angle_deg)
        normal_angle = axis_angle + math.pi / 2.0
        normal = vector_from_angle(normal_angle)
        projected = _project_points(points, normal)
        projections = [projection for _point, projection in projected]
        ordered = sorted(projections)
        p30 = percentile_sorted(ordered, 30.0)
        p70 = percentile_sorted(ordered, 70.0)
        lower_values = [value for value in projections if value <= p30]
        upper_values = [value for value in projections if value >= p70]
        lower_center = median(lower_values)
        upper_center = median(upper_values)
        separation = upper_center - lower_center
        width_match = match_width_profile(separation, config)
        separation_error = width_match.abs_error_m
        if use_prefit_width_gate and separation_error > width_skip_threshold:
            prefit_skipped_count += 1
            continue
        lower_points, upper_points = _projection_clusters_from_projected(
            projected,
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
        candidates.append(candidate)
        if best is None or candidate[0] > best[0]:
            best = candidate

    candidates.sort(key=lambda item: item[0], reverse=True)
    return {
        "best": best,
        "candidates": candidates,
        "angle_count": angle_count,
        "candidate_count": len(candidates),
        "prefit_skipped_count": prefit_skipped_count,
    }


def _combine_search_stats(*stats_items):
    return {
        "angle_count": sum(item["angle_count"] for item in stats_items),
        "candidate_count": sum(item["candidate_count"] for item in stats_items),
        "prefit_skipped_count": sum(item["prefit_skipped_count"] for item in stats_items),
    }


def _long_wall_fit_from_best(
    best,
    config: ArenaGeometryConfig,
    search_mode,
    stats,
    search_fallback_used=False,
):
    if best is None:
        return LongWallFit(
            False,
            "no_parallel_long_wall_candidates",
            search_mode=search_mode,
            search_angle_count=stats["angle_count"],
            search_candidate_count=stats["candidate_count"],
            search_prefit_skipped_count=stats["prefit_skipped_count"],
            search_fallback_used=search_fallback_used,
        )

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
        search_mode=search_mode,
        search_angle_count=stats["angle_count"],
        search_candidate_count=stats["candidate_count"],
        search_prefit_skipped_count=stats["prefit_skipped_count"],
        search_fallback_used=search_fallback_used,
    )


def _fit_long_walls_exhaustive(
    points: Sequence[tuple[float, float]],
    config: ArenaGeometryConfig,
    search_mode="exhaustive",
    search_fallback_used=False,
):
    stats = _score_long_wall_angles(
        points,
        config,
        _exhaustive_angles_deg(config),
        use_prefit_width_gate=False,
    )
    return _long_wall_fit_from_best(
        stats["best"],
        config,
        search_mode,
        stats,
        search_fallback_used=search_fallback_used,
    )


def _fit_long_walls_coarse_to_fine(
    points: Sequence[tuple[float, float]],
    config: ArenaGeometryConfig,
):
    coarse = _score_long_wall_angles(
        points,
        config,
        _coarse_angles_deg(config),
        use_prefit_width_gate=True,
    )
    if not coarse["candidates"]:
        return _fit_long_walls_exhaustive(
            points,
            config,
            search_mode="coarse_to_fine_fallback",
            search_fallback_used=True,
        )

    refine = _score_long_wall_angles(
        points,
        config,
        _refine_angles_deg(coarse["candidates"], config),
        use_prefit_width_gate=True,
    )
    combined_stats = _combine_search_stats(coarse, refine)
    best = refine["best"] or coarse["best"]
    fit = _long_wall_fit_from_best(best, config, "coarse_to_fine", combined_stats)
    if not fit.ok:
        return _fit_long_walls_exhaustive(
            points,
            config,
            search_mode="coarse_to_fine_fallback",
            search_fallback_used=True,
        )
    return fit


def fit_long_walls(points: Sequence[tuple[float, float]], config: ArenaGeometryConfig):
    if len(points) < 2 * config.min_wall_points:
        return LongWallFit(
            False,
            "insufficient_points",
            search_mode=config.long_wall_search_mode,
        )

    if config.long_wall_search_mode == "exhaustive":
        return _fit_long_walls_exhaustive(points, config)
    return _fit_long_walls_coarse_to_fine(points, config)
