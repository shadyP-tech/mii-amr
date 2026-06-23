from __future__ import annotations

import math
from typing import Iterable, List, Sequence

from .models import BaseFramePoint, LidarStandDetectorConfig, StandCandidate


def _valid_range(raw_range: float, config: LidarStandDetectorConfig) -> bool:
    return (
        raw_range is not None
        and math.isfinite(float(raw_range))
        and float(raw_range) >= config.min_range_m
        and float(raw_range) < config.max_range_m
    )


def scan_points_from_ranges(
    ranges: Sequence[float],
    *,
    angle_min_rad: float,
    angle_increment_rad: float,
    config: LidarStandDetectorConfig | None = None,
) -> List[BaseFramePoint]:
    cfg = config or LidarStandDetectorConfig()
    points: List[BaseFramePoint] = []
    for index, raw_range in enumerate(ranges):
        if not _valid_range(raw_range, cfg):
            continue
        distance = float(raw_range)
        bearing = angle_min_rad + index * angle_increment_rad
        points.append(
            BaseFramePoint(
                x_m=distance * math.cos(bearing),
                y_m=distance * math.sin(bearing),
                bearing_rad=bearing,
                range_m=distance,
                source_index=index,
            )
        )
    return points


def _distance(a: BaseFramePoint, b: BaseFramePoint) -> float:
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)


def cluster_scan_points(
    points: Sequence[BaseFramePoint],
    *,
    config: LidarStandDetectorConfig | None = None,
) -> List[List[BaseFramePoint]]:
    cfg = config or LidarStandDetectorConfig()
    ordered = sorted(points, key=lambda point: point.source_index)
    clusters: List[List[BaseFramePoint]] = []
    current: List[BaseFramePoint] = []

    for point in ordered:
        if not current:
            current = [point]
            continue
        previous = current[-1]
        adjacent_index = point.source_index == previous.source_index + 1
        close_enough = _distance(previous, point) <= cfg.max_cluster_gap_m
        if adjacent_index and close_enough:
            current.append(point)
            continue
        clusters.append(current)
        current = [point]

    if current:
        clusters.append(current)
    return clusters


def _cluster_width(cluster: Sequence[BaseFramePoint]) -> float:
    if len(cluster) < 2:
        return 0.0
    return max(_distance(a, b) for a in cluster for b in cluster)


def _candidate_from_cluster(
    cluster: Sequence[BaseFramePoint],
    *,
    candidate_index: int,
    config: LidarStandDetectorConfig,
) -> StandCandidate | None:
    if len(cluster) < config.min_cluster_points:
        return None
    width = _cluster_width(cluster)
    if width < config.min_width_m or width > config.max_width_m:
        return None

    center_x = sum(point.x_m for point in cluster) / len(cluster)
    center_y = sum(point.y_m for point in cluster) / len(cluster)
    distance = math.hypot(center_x, center_y)
    bearing = math.atan2(center_y, center_x)

    point_score = min(1.0, len(cluster) / max(config.min_cluster_points * 2.0, 1.0))
    width_midpoint = (config.min_width_m + config.max_width_m) / 2.0
    width_half_span = max((config.max_width_m - config.min_width_m) / 2.0, 0.001)
    width_score = max(0.0, 1.0 - abs(width - width_midpoint) / width_half_span)
    confidence = max(0.0, min(1.0, 0.45 + 0.30 * point_score + 0.25 * width_score))

    return StandCandidate(
        candidate_id=f"stand_candidate_{candidate_index:02d}",
        bearing_rad=bearing,
        distance_m=distance,
        approximate_width_m=width,
        center_x_m=center_x,
        center_y_m=center_y,
        point_count=len(cluster),
        confidence=confidence,
    )


def detect_stand_candidates(
    points: Sequence[BaseFramePoint],
    *,
    config: LidarStandDetectorConfig | None = None,
) -> List[StandCandidate]:
    cfg = config or LidarStandDetectorConfig()
    candidates: List[StandCandidate] = []
    for cluster in cluster_scan_points(points, config=cfg):
        candidate = _candidate_from_cluster(
            cluster,
            candidate_index=len(candidates) + 1,
            config=cfg,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def detect_stand_candidates_from_scan(
    ranges: Sequence[float],
    *,
    angle_min_rad: float,
    angle_increment_rad: float,
    config: LidarStandDetectorConfig | None = None,
) -> List[StandCandidate]:
    cfg = config or LidarStandDetectorConfig()
    points = scan_points_from_ranges(
        ranges,
        angle_min_rad=angle_min_rad,
        angle_increment_rad=angle_increment_rad,
        config=cfg,
    )
    return detect_stand_candidates(points, config=cfg)

