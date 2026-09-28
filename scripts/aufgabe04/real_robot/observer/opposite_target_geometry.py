"""One certified target position for opposite-view search and scan witnesses.

The survey center remains the candidate identity. A retained metric head center
is a bounded search hypothesis, never a replacement for current scan evidence.
"""
from dataclasses import replace
import math

from scripts.aufgabe04.artifacts.retained_backside_orientation import validate_retained_orientation
from scripts.aufgabe04.real_robot.observer.scan_target_geometry import scan_target_geometry

RETAINED_TARGET = 'retained_validated_center'


def retained_target_center(record, *, stand_center):
    if record is None:
        return None
    record = validate_retained_orientation(record, candidate_uid=record['candidate_uid'],
        planning_frame=record['planning_frame'], stand_center=stand_center,
        model_sha256=record['stand_model_profile_sha256'])
    return record.get('validated_target_center')


def retained_scan_target(point, *, center, stand_radius_m, stand_uncertainty_m,
                         lidar_range_tolerance_m):
    uncertainty = center['uncertainty_m']
    if (type(uncertainty) not in (int, float) or not math.isfinite(uncertainty)
            or not 0 < uncertainty <= .3
            or not 0 <= lidar_range_tolerance_m <= .05+1e-9):
        raise ValueError('retained target range uncertainty exceeds bound')
    target = scan_target_geometry(point, stand_radius_m=stand_radius_m,
        stand_uncertainty_m=stand_uncertainty_m,
        lidar_range_tolerance_m=lidar_range_tolerance_m)
    lo, hi = target.accepted_range_m
    return replace(target, accepted_range_m=(lo-uncertainty, hi+uncertainty))
