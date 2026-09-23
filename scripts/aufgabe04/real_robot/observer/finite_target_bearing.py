"""Finite-distance camera rays in the scan frame, including camera translation."""
import math
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector


def point_on_scan_range(*, center_px, intrinsics, scan_from_camera, distance_m):
    values = (*center_px, intrinsics.fx_px, intrinsics.fy_px, intrinsics.cx_px,
              intrinsics.cy_px, distance_m, *scan_from_camera.translation_xyz_m)
    if not all(math.isfinite(v) for v in values) or min(intrinsics.fx_px, intrinsics.fy_px, distance_m) <= 0:
        raise ValueError('invalid finite target projection')
    ray = ((center_px[0]-intrinsics.cx_px)/intrinsics.fx_px,
           (center_px[1]-intrinsics.cy_px)/intrinsics.fy_px, 1.)
    d = rotate_vector(ray, scan_from_camera.rotation_xyzw)
    t = scan_from_camera.translation_xyz_m
    a, b = d[0]**2+d[1]**2, 2*(t[0]*d[0]+t[1]*d[1])
    c = t[0]**2+t[1]**2-distance_m**2
    disc = b*b-4*a*c
    if a <= 1e-12 or disc < 0:
        raise ValueError('camera ray misses scan range')
    roots = [z for z in ((-b-math.sqrt(disc))/(2*a), (-b+math.sqrt(disc))/(2*a)) if z > 1e-9]
    if len(roots) != 1:
        raise ValueError('camera ray has no unique positive depth')
    return tuple(ti+roots[0]*di for ti, di in zip(t, d)), roots[0]


def finite_target_bearing(*, center_px, intrinsics, scan_from_camera, distance_m, range_interval_m):
    lo, hi = range_interval_m
    if not 0 < lo <= distance_m <= hi:
        raise ValueError('invalid target range interval')
    common = dict(center_px=center_px, intrinsics=intrinsics, scan_from_camera=scan_from_camera)
    point, depth = point_on_scan_range(**common, distance_m=distance_m)
    bearing = math.atan2(point[1], point[0])
    ends = [point_on_scan_range(**common, distance_m=r)[0] for r in (lo, hi)]
    uncertainty = max(abs(math.remainder(math.atan2(p[1], p[0])-bearing, math.tau)) for p in ends)
    return bearing, uncertainty, depth
