from __future__ import annotations

from typing import Sequence

from .geometry import fit_line, percentile_sorted, vector_from_angle
from .models import (
    WALL_AMBIGUOUS,
    WALL_CLEAN,
    WALL_HEATER,
    WALL_UNKNOWN,
    ArenaGeometryConfig,
    PairwiseShortWallClassification,
    ShortWallClassification,
)
from .short_wall_profile import (
    _compute_short_wall_profile_features_from_projected,
    is_profile_clean_like,
    is_profile_heater_like,
    is_profile_weak,
    score_short_wall_profile,
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
        "relative_strong_heater_min_score": config.profile_relative_strong_heater_min_score,
        "relative_strong_opposite_max_heater_score": (
            config.profile_relative_strong_opposite_max_heater_score
        ),
        "relative_min_protrusion_clusters": config.profile_relative_min_protrusion_clusters,
        "relative_min_protrusion_fraction": config.profile_relative_min_protrusion_fraction,
        "relative_confidence_raw": selected_heater_score,
    }
    strong_winner_with_tolerable_opposite = (
        selected_heater_score >= config.profile_relative_strong_heater_min_score
        and opposite_heater_score
        <= config.profile_relative_strong_opposite_max_heater_score
        and heater_delta >= config.profile_relative_heater_min_delta
    )

    if selected_heater_score < config.profile_relative_heater_min_score:
        return PairwiseShortWallClassification(
            reason="pairwise_profile_relative_heater_score_too_low",
            **relative_common,
        )
    if (
        opposite_heater_score > config.profile_relative_opposite_max_heater_score
        and not strong_winner_with_tolerable_opposite
    ):
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
        relative_strong_heater_min_score=pairwise.relative_strong_heater_min_score,
        relative_strong_opposite_max_heater_score=(
            pairwise.relative_strong_opposite_max_heater_score
        ),
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


def _project_points(points, axis, normal):
    axis_x, axis_y = axis
    normal_x, normal_y = normal
    return [
        (
            point,
            point[0] * axis_x + point[1] * axis_y,
            point[0] * normal_x + point[1] * normal_y,
        )
        for point in points
    ]


def _classify_projected_candidate(
    projected_points,
    side,
    edge_projection,
    config: ArenaGeometryConfig,
):
    band_projected_points = []
    for projected in projected_points:
        if abs(projected[1] - edge_projection) <= config.short_wall_band_m:
            band_projected_points.append(projected)

    if len(band_projected_points) < config.min_short_wall_points:
        profile_features = _compute_short_wall_profile_features_from_projected(
            band_projected_points,
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
            point_count=len(band_projected_points),
            profile_features=profile_features,
            validity_failed_reason=profile_features.get("validity_failed_reason"),
        )

    normal_values = [
        normal_projection
        for _point, _axis_projection, normal_projection in band_projected_points
    ]
    visible_width = max(normal_values) - min(normal_values)
    band_points = [point for point, _axis_projection, _normal_projection in band_projected_points]
    outer_points = [
        point
        for point, axis_projection, _normal_projection in band_projected_points
        if abs(axis_projection - edge_projection) <= config.short_wall_outer_band_m
    ]
    if len(outer_points) < max(3, config.min_short_wall_points // 2):
        outer_points = band_points
    try:
        line = fit_line(outer_points)
    except ValueError:
        profile_features = _compute_short_wall_profile_features_from_projected(
            band_projected_points,
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
            point_count=len(band_projected_points),
            profile_features=profile_features,
            validity_failed_reason=profile_features.get("validity_failed_reason"),
        )

    profile_features = _compute_short_wall_profile_features_from_projected(
        band_projected_points,
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
        point_count=len(band_projected_points),
        profile_features=profile_features,
        heater_profile_score=heater_score,
        clean_profile_score=clean_score,
        validity_failed_reason=validity_failed_reason,
    )


def _forced_short_wall_fast_path_enabled(config: ArenaGeometryConfig):
    return (
        config.forced_short_wall_side in {"axis_negative", "axis_positive"}
        and config.forced_short_wall_type in {WALL_HEATER, WALL_CLEAN}
    )


def _forced_candidate_valid_for_fast_path(
    candidate: ShortWallClassification,
    config: ArenaGeometryConfig,
):
    if candidate.point_count < config.min_short_wall_points:
        return False
    if candidate.short_wall_rmse_m is None:
        return False
    return candidate.short_wall_rmse_m <= config.max_line_rmse_m


def _skipped_forced_short_wall_candidate(side, edge_projection):
    return ShortWallClassification(
        WALL_UNKNOWN,
        "forced_short_wall_side_skipped",
        observed_axis_side=side,
        short_wall_candidate_range_m=abs(edge_projection),
    )


def classify_candidate(
    points,
    axis,
    normal,
    side,
    edge_projection,
    config: ArenaGeometryConfig,
):
    return _classify_projected_candidate(
        _project_points(points, axis, normal),
        side,
        edge_projection,
        config,
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
    projected_points = _project_points(points, axis, normal)
    axis_projections = sorted(
        axis_projection
        for _point, axis_projection, _normal_projection in projected_points
    )
    lower_edge = percentile_sorted(axis_projections, 5.0)
    upper_edge = percentile_sorted(axis_projections, 95.0)
    if _forced_short_wall_fast_path_enabled(config):
        forced_side = config.forced_short_wall_side
        forced_edge = lower_edge if forced_side == "axis_negative" else upper_edge
        forced_candidate = _classify_projected_candidate(
            projected_points,
            forced_side,
            forced_edge,
            config,
        )
        if _forced_candidate_valid_for_fast_path(forced_candidate, config):
            other_side = (
                "axis_positive"
                if forced_side == "axis_negative"
                else "axis_negative"
            )
            other_edge = upper_edge if other_side == "axis_positive" else lower_edge
            return {
                forced_side: forced_candidate,
                other_side: _skipped_forced_short_wall_candidate(other_side, other_edge),
            }

    return {
        "axis_negative": _classify_projected_candidate(
            projected_points,
            "axis_negative",
            lower_edge,
            config,
        ),
        "axis_positive": _classify_projected_candidate(
            projected_points,
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


def _select_short_wall_classification_with_pairwise(
    candidates: dict[str, ShortWallClassification],
    config: ArenaGeometryConfig,
):
    forced = forced_short_wall_classification(candidates, config)
    if forced is not None:
        return forced, None

    pairwise = classify_short_wall_pairwise(candidates, config)
    if pairwise.applicable:
        return pairwise_result_to_classification(candidates, pairwise), pairwise

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
            ), None
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
            ), None
        return copy_candidate_with_reason(
            heater,
            "complementary_short_walls_valid",
            short_wall_range_sum_m=range_sum,
            short_wall_range_sum_error_m=range_sum_error,
        ), None
    if len(accepted) == 1:
        return accepted[0], None
    if ambiguous:
        return max(ambiguous, key=lambda item: item.confidence), None
    return max(ordered, key=lambda item: item.confidence), None


def select_short_wall_classification(
    candidates: dict[str, ShortWallClassification],
    config: ArenaGeometryConfig,
):
    classification, _pairwise = _select_short_wall_classification_with_pairwise(
        candidates,
        config,
    )
    return classification
