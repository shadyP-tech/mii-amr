#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import io
import math
import pstats
import time

from arena_geometry_localization import ArenaGeometryConfig, analyze_points


def rotate_point(point, yaw_deg):
    yaw = math.radians(yaw_deg)
    x, y = point
    return (
        math.cos(yaw) * x - math.sin(yaw) * y,
        math.sin(yaw) * x + math.cos(yaw) * y,
    )


def transform_points(points, yaw_deg=0.0, lateral_offset_m=0.0):
    transformed = []
    for x, y in points:
        transformed.append(rotate_point((x, y - lateral_offset_m), yaw_deg))
    return transformed


def rectangular_points(
    include_clean=True,
    include_heater=True,
    yaw_deg=0.0,
    lateral_offset_m=0.0,
    width_m=1.898,
):
    half_length = 3.90 / 2.0
    half_width = width_m / 2.0
    points = []
    for index in range(61):
        x = -1.50 + index * 0.05
        points.append((x, -half_width))
        points.append((x, half_width))
    if include_clean:
        for index in range(39):
            y = -0.90 + index * 0.05
            points.append((-half_length, y))
    if include_heater:
        for index in range(39):
            y = -0.90 + index * 0.05
            points.append((half_length, y))
        for y_center in (-0.35, 0.35):
            for offset_index in range(10):
                y = y_center - 0.12 + offset_index * 0.025
                points.append((half_length - 0.16, y))
    return transform_points(points, yaw_deg=yaw_deg, lateral_offset_m=lateral_offset_m)


def benchmark_points(width_m, copies):
    base = rectangular_points(width_m=width_m)
    if copies <= 1:
        return base

    offsets = []
    center = (copies - 1) / 2.0
    for index in range(copies):
        offsets.append((index - center) * 0.002)

    points = []
    for dx in offsets:
        for x, y in base:
            points.append((x + dx, y))
    return points


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile arena geometry localization on synthetic points.",
    )
    parser.add_argument("--calls", type=int, default=100, help="analyze_points calls")
    parser.add_argument("--copies", type=int, default=7, help="jittered copies of the synthetic arena")
    parser.add_argument("--profile-lines", type=int, default=20, help="cProfile rows to print")
    parser.add_argument(
        "--search-mode",
        choices=("coarse_to_fine", "exhaustive"),
        default="coarse_to_fine",
        help="long-wall search mode",
    )
    parser.add_argument("--width-m", type=float, default=1.898, help="synthetic arena width")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.calls < 1:
        raise SystemExit("--calls must be >= 1")
    if args.copies < 1:
        raise SystemExit("--copies must be >= 1")
    if args.profile_lines < 1:
        raise SystemExit("--profile-lines must be >= 1")

    points = benchmark_points(args.width_m, args.copies)
    config = ArenaGeometryConfig(
        arena_width_m=args.width_m,
        long_wall_search_mode=args.search_mode,
    )

    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    result = None
    for _index in range(args.calls):
        result = analyze_points(points, config)
    profile.disable()
    elapsed = time.perf_counter() - start

    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(
        args.profile_lines,
    )

    print(f"points={len(points)}")
    print(f"calls={args.calls}")
    print(f"search_mode={args.search_mode}")
    print(f"total_sec={elapsed:.6f}")
    print(f"sec_per_call={elapsed / args.calls:.6f}")
    if result is not None:
        print(f"success={result.success}")
        print(f"failure_reason={result.failure_reason}")
        print(f"short_wall_reason={result.short_wall_classification.reason}")
        print(f"long_wall_search_mode={result.long_wall_fit.search_mode}")
        print(f"long_wall_search_angle_count={result.long_wall_fit.search_angle_count}")
        print(f"long_wall_search_candidate_count={result.long_wall_fit.search_candidate_count}")
        print(f"long_wall_search_prefit_skipped_count={result.long_wall_fit.search_prefit_skipped_count}")
        print(f"long_wall_search_fallback_used={result.long_wall_fit.search_fallback_used}")
    print(stream.getvalue())


if __name__ == "__main__":
    main()
