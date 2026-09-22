"""A camera search location covering compatible raw endpoint fragments.

This never declares one LiDAR target, supplies corners, or proves continuity.
Mission association still requires its independent temporal witnesses.
"""

import math

from scripts.aufgabe04.perception.scan_endpoint_fragments import bounded_endpoint_fragments


def endpoint_search_center(scan, candidates, *, max_width_m,
                           max_point_gap_m=.04, max_range_jump_m=.05):
    groups = tuple(candidate.source_indices for candidate in candidates)
    if not bounded_endpoint_fragments(scan, groups):
        return None
    points = {}
    for group in groups:
        for index in group:
            distance = scan.ranges[index]
            if (not math.isfinite(distance) or not scan.range_min <= distance < scan.range_max):
                return None
            angle = scan.angle_min + index * scan.angle_increment
            points[index] = (distance * math.cos(angle), distance * math.sin(angle))
    if (abs(scan.ranges[0] - scan.ranges[-1]) > max_range_jump_m
            or math.dist(points[0], points[len(scan.ranges)-1]) > max_point_gap_m
            or any(math.dist(a, b) > max_width_m for a in points.values() for b in points.values())):
        return None
    return tuple(sum(point[axis] for point in points.values()) / len(points) for axis in (0, 1))
