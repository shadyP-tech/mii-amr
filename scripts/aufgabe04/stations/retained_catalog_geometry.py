"""Keep a retained angle's interval and frame ancestry in durable catalogs."""
from dataclasses import replace
import math
from scripts.aufgabe04.artifacts.projected_retained_facing import (
    projected_retained_from_evidence, projected_retained_candidate,
)
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import recommendation_axis_estimator


def validate_retained_catalog_record(record):
    if record.retained_facing is None or record.axis.bounded_orientation is None:
        raise ValueError('retained catalog geometry requires both bounds and projection evidence')
    from scripts.aufgabe04.perception.arrival_pose_estimator import arrival_pose_record_from_recommendation
    recommendation = projected_retained_from_evidence(record.retained_facing)
    if record.candidate_uid != recommendation.stand_id:
        raise ValueError('retained catalog candidate differs from original QR receipt')
    expected = arrival_pose_record_from_recommendation(recommendation,
        candidate_uid=record.candidate_uid,
        map_yaml_sha256=record.validation.validated_map_yaml_sha256,
        corridor_length_m=record.corridor_length_m,
        validated_unix_sec=record.validation.validated_unix_sec,
        axis_sample_count=recommendation.axis_sample_count,
        estimator=recommendation_axis_estimator(recommendation), source=record.source,
        source_observation_ids=record.source_observation_ids)
    # Server station identity is bound independently by the identity registry.
    for field in ('stand', 'axis', 'face', 'arrival_pose', 'corridor_entry_pose',
                  'standoff_m', 'sensor_stamp_sec'):
        if getattr(record, field) != getattr(expected, field):
            raise ValueError(f'retained catalog {field} differs from authenticated geometry')


def require_retained_candidate_anchor(record, candidate):
    validate_retained_catalog_record(record)
    original = projected_retained_candidate(record.retained_facing)
    if original != candidate:
        raise ValueError('retained catalog frozen candidate anchor changed')


def retained_catalog_keepouts(record, candidate, config, clearance):
    """Preserve both centers' clearance during common-frame route validation."""
    active = clearance.get('minimum_active_standoff_m')
    collision = clearance.get('minimum_collision_standoff_m')
    if (any(type(value) not in (int, float) or not math.isfinite(value)
            for value in (active, collision)) or not 0 < collision <= active
            or collision > candidate.geometry.keepout_radius_m + 1e-9):
        raise ValueError('retained catalog requires collision clearance covered by its frozen envelope')
    if (record.standoff_m + 1e-9 < active or
            math.hypot(record.arrival_pose.x_m-candidate.geometry.x_m,
                       record.arrival_pose.y_m-candidate.geometry.y_m) + 1e-9 < active):
        raise ValueError('retained catalog target violates active clearance around one of its centers')
    original = replace(config, stand_radius_m=candidate.geometry.radius_m,
        stand_position_uncertainty_m=candidate.geometry.uncertainty_m)
    measured = replace(config, stand_radius_m=record.stand.radius_m,
        stand_position_uncertainty_m=record.stand.uncertainty_m)
    measured_floor = collision + max(0.,record.stand.uncertainty_m-candidate.geometry.uncertainty_m)
    return (
        (candidate.geometry.x_m,candidate.geometry.y_m,
         max(original.stand_keepout_radius_m,candidate.geometry.keepout_radius_m+config.tracking_margin_m)),
        (record.stand.x_m,record.stand.y_m,max(measured_floor,measured.stand_keepout_radius_m)),
    )
