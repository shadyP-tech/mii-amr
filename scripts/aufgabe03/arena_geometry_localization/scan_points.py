from __future__ import annotations

import math
from typing import Sequence

from .models import Pose2D, ScanSample


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


def _pose_key(pose: Pose2D | None):
    if pose is None:
        return None
    return (pose.x, pose.y, pose.yaw_deg)


def _limit_points_evenly(points, limit):
    if limit is None or limit >= len(points):
        return points
    limit = int(limit)
    if limit <= 0:
        return []
    if limit == 1:
        return [points[len(points) // 2]]
    max_position = len(points) - 1
    selected = []
    seen = set()
    for selection_index in range(limit):
        position = round(selection_index * max_position / (limit - 1))
        if position in seen:
            continue
        seen.add(position)
        selected.append(points[position])
    return selected


class ScanPointCache:
    def __init__(self):
        self._local_points = {}
        self._transformed_points = {}
        self.local_hits = 0
        self.local_misses = 0
        self.transformed_hits = 0
        self.transformed_misses = 0

    def local_points(self, sample: ScanSample, range_stride=1):
        key = (id(sample), int(range_stride))
        if key in self._local_points:
            self.local_hits += 1
            return self._local_points[key]
        self.local_misses += 1
        points = finite_scan_points(sample, range_stride=range_stride)
        self._local_points[key] = points
        return points

    def transformed_points(self, sample: ScanSample, origin: Pose2D | None, range_stride=1):
        pose = Pose2D()
        if sample.odom_pose is not None and origin is not None:
            pose = relative_pose(sample.odom_pose, origin)
        key = (
            id(sample),
            int(range_stride),
            _pose_key(origin),
            _pose_key(sample.odom_pose),
            _pose_key(pose),
        )
        if key in self._transformed_points:
            self.transformed_hits += 1
            return self._transformed_points[key]
        self.transformed_misses += 1
        cos_yaw = math.cos(math.radians(pose.yaw_deg))
        sin_yaw = math.sin(math.radians(pose.yaw_deg))
        transformed = [
            (
                pose.x + cos_yaw * x - sin_yaw * y,
                pose.y + sin_yaw * x + cos_yaw * y,
            )
            for x, y in self.local_points(sample, range_stride=range_stride)
        ]
        self._transformed_points[key] = transformed
        return transformed

    def to_dict(self):
        return {
            "local_entries": len(self._local_points),
            "transformed_entries": len(self._transformed_points),
            "local_hits": self.local_hits,
            "local_misses": self.local_misses,
            "transformed_hits": self.transformed_hits,
            "transformed_misses": self.transformed_misses,
        }


def _transformed_points_for_sample(sample, origin, range_stride, point_cache):
    if point_cache is not None:
        return point_cache.transformed_points(
            sample,
            origin,
            range_stride=range_stride,
        )

    pose = Pose2D()
    if sample.odom_pose is not None and origin is not None:
        pose = relative_pose(sample.odom_pose, origin)
    yaw = math.radians(pose.yaw_deg)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    transformed = []
    for index, raw_range in enumerate(sample.ranges):
        if index % range_stride != 0:
            continue
        if not math.isfinite(raw_range):
            continue
        if raw_range < sample.range_min or raw_range > sample.range_max:
            continue
        angle = sample.angle_min + index * sample.angle_increment
        x = raw_range * math.cos(angle)
        y = raw_range * math.sin(angle)
        transformed.append((
            pose.x + cos_yaw * x - sin_yaw * y,
            pose.y + sin_yaw * x + cos_yaw * y,
        ))
    return transformed


def accumulate_scan_points(
    samples: Sequence[ScanSample],
    range_stride=1,
    max_points=None,
    point_cache: ScanPointCache | None = None,
    sample_point_limits=None,
):
    if not samples:
        return []
    origin = next((sample.odom_pose for sample in samples if sample.odom_pose is not None), None)
    points = []
    for sample in samples:
        limit = None
        if sample_point_limits is not None:
            limit = sample_point_limits.get(id(sample))
        transformed = _limit_points_evenly(
            _transformed_points_for_sample(sample, origin, range_stride, point_cache),
            limit,
        )
        for point in transformed:
            points.append(point)
            if max_points is not None and len(points) >= max_points:
                return points
    return points
