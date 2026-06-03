"""ROS-free rectangular-arena geometry localization helpers."""

from .analysis import analyze_points, analyze_scan_samples
from .geometry import (
    clamp,
    dot,
    fit_line,
    median,
    normalize_angle_rad,
    normalize_undirected_angle_rad,
    percentile,
    percentile_sorted,
    projection_clusters,
    undirected_angle_delta_rad,
    vector_from_angle,
)
from .long_walls import fit_long_walls, match_width_profile, width_profiles
from .models import (
    WALL_AMBIGUOUS,
    WALL_CLEAN,
    WALL_HEATER,
    WALL_UNKNOWN,
    ArenaGeometryConfig,
    ArenaGeometryResult,
    LineFit,
    LongWallFit,
    PairwiseShortWallClassification,
    Pose2D,
    ScanSample,
    ShortWallClassification,
    WidthMatch,
)
from .pose_prior import build_pose_prior, estimate_covariance, wall_side_for_type
from .scan_points import (
    ScanPointCache,
    accumulate_scan_points,
    finite_scan_points,
    relative_pose,
    transform_point,
)
from .serialization import iter_points, load_scan_samples_json, write_json
from .short_wall_classifier import (
    allows_relaxed_heater_profile_rmse,
    annotate_pairwise_candidates,
    classify_candidate,
    classify_short_wall_candidates,
    classify_short_wall_pairwise,
    classify_short_wall_relative_heater,
    complementary_short_wall_pair,
    copy_candidate_with_pairwise_result,
    copy_candidate_with_reason,
    empty_short_wall_candidates,
    forced_short_wall_classification,
    is_valid_short_wall_candidate,
    pairwise_result_to_classification,
    select_short_wall_classification,
    short_wall_profile_validity_failure,
)
from .short_wall_profile import (
    clipped_score,
    cluster_bins,
    compute_short_wall_profile_features,
    is_profile_clean_like,
    is_profile_heater_like,
    is_profile_weak,
    score_short_wall_profile,
)


__all__ = [name for name in globals() if not name.startswith("_")]
