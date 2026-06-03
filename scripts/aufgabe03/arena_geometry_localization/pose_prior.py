from __future__ import annotations

import math
from typing import Sequence

from .geometry import clamp, dot, normalize_angle_rad, percentile, vector_from_angle
from .models import (
    WALL_CLEAN,
    WALL_HEATER,
    ArenaGeometryConfig,
    LongWallFit,
    Pose2D,
    ShortWallClassification,
)
from .short_wall_classifier import complementary_short_wall_pair


def wall_side_for_type(wall_type, config: ArenaGeometryConfig):
    heater = config.heater_wall_side
    clean = "-x" if heater == "+x" else "+x"
    if wall_type == WALL_HEATER:
        return heater
    if wall_type == WALL_CLEAN:
        return clean
    return None


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
