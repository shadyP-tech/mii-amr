from __future__ import annotations

import math
from typing import Sequence

from .long_walls import fit_long_walls
from .models import (
    WALL_CLEAN,
    WALL_HEATER,
    WALL_UNKNOWN,
    ArenaGeometryConfig,
    ArenaGeometryResult,
    ScanSample,
    ShortWallClassification,
)
from .pose_prior import build_pose_prior, estimate_covariance
from .scan_points import accumulate_scan_points
from .short_wall_classifier import (
    _select_short_wall_classification_with_pairwise,
    annotate_pairwise_candidates,
    classify_short_wall_candidates,
    empty_short_wall_candidates,
)


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
    classification, pairwise = _select_short_wall_classification_with_pairwise(
        candidates,
        config,
    )
    if pairwise is not None and classification.reason.startswith("pairwise_profile_"):
        candidates = annotate_pairwise_candidates(candidates, pairwise)
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
    point_cache=None,
    sample_point_limits=None,
):
    points = accumulate_scan_points(
        samples,
        range_stride=range_stride,
        max_points=max_points,
        point_cache=point_cache,
        sample_point_limits=sample_point_limits,
    )
    result = analyze_points(points, config)
    diagnostics = dict(result.diagnostics)
    diagnostics["num_scan_samples_used"] = len(samples)
    if sample_point_limits is not None:
        diagnostics["sample_point_limits_used"] = True
        diagnostics["sample_point_limit_count"] = len(sample_point_limits)
        diagnostics["sample_point_limit_total"] = sum(sample_point_limits.values())
    if point_cache is not None:
        diagnostics["scan_point_cache"] = point_cache.to_dict()
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
