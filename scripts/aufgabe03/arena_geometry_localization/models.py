from __future__ import annotations

import math
from dataclasses import dataclass



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
    profile_relative_heater_min_delta: float = 0.12
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
    long_wall_search_mode: str = "coarse_to_fine"
    long_wall_prefit_width_skip_margin_m: float = 0.15
    long_wall_coarse_step_deg: float = 10.0
    long_wall_refine_top_k: int = 2
    long_wall_refine_window_deg: float = 5.0
    long_wall_refine_step_deg: float = 0.5


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
    search_mode: str | None = None
    search_angle_count: int = 0
    search_candidate_count: int = 0
    search_prefit_skipped_count: int = 0
    search_fallback_used: bool = False

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
            "search_mode": self.search_mode,
            "search_angle_count": self.search_angle_count,
            "search_candidate_count": self.search_candidate_count,
            "search_prefit_skipped_count": self.search_prefit_skipped_count,
            "search_fallback_used": self.search_fallback_used,
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
