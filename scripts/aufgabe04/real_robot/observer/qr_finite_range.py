"""Resolve range from the decoded QR's entire calibrated ray interval.

No head fit is required. The union of narrow cones for every allowed range
must contain one provable target before that target supplies measured depth.
"""
from dataclasses import asdict
import math

from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
from scripts.aufgabe04.real_robot.observer.finite_target_bearing import finite_target_bearing
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import registered_target_is_unique, validated_witnessed_fragmentation
from scripts.aufgabe04.real_robot.observer.shared_scan_cluster import WitnessedEnvelope, bind_ray_to_envelope

POLICY = 'current_qr_entire_range_ray_unique_target'


def resolve_qr_range(*, scan, center_px, intrinsics, scan_from_camera,
        map_bearing_rad, cone_half_angle_rad, accepted_range_m, now_sec,
        max_scan_age_sec, max_camera_map_bearing_delta_rad, min_cluster_sample_count,
        resolver=None):
    if not 0 < cone_half_angle_rad <= math.radians(3)+1e-9:
        raise ValueError('QR range ray cone exceeds bound')
    common = dict(center_px=center_px, intrinsics=intrinsics,
        scan_from_camera=scan_from_camera, range_interval_m=accepted_range_m)
    middle = sum(accepted_range_m)/2
    ray, spread, _ = finite_target_bearing(**common, distance_m=middle)
    if abs(math.remainder(ray-map_bearing_rad,math.tau))+spread > max_camera_map_bearing_delta_rad:
        raise ValueError('camera_map_bearing_interval_exceeds_limit')
    params = dict(map_bearing_rad=map_bearing_rad, accepted_range_m=accepted_range_m,
        now_sec=now_sec,max_scan_age_sec=max_scan_age_sec,min_cluster_sample_count=1,
        max_camera_map_bearing_delta_rad=max_camera_map_bearing_delta_rad)
    support = associate_camera_registered_candidate_lidar_target(scan, **params,
        observed_camera_bearing_rad=ray, cone_half_angle_rad=cone_half_angle_rad+spread)
    if resolver is not None:
        support = resolver(support, scan)
    if not registered_target_is_unique(support):
        raise ValueError('finite_qr_ray_target_not_unique')
    bearing, uncertainty, depth = finite_target_bearing(**common,distance_m=support.distance_m)
    result = associate_camera_registered_candidate_lidar_target(scan,
        **{**params,'min_cluster_sample_count':min_cluster_sample_count},
        observed_camera_bearing_rad=bearing,cone_half_angle_rad=cone_half_angle_rad)
    if support.witnessed_fragmentation is not None:
        envelope = WitnessedEnvelope(**asdict(support.search_association),
                                    witnessed_fragmentation=support.witnessed_fragmentation)
        result = bind_ray_to_envelope(result, scan, envelope,
            now_sec=now_sec,max_scan_age_sec=max_scan_age_sec)
    if (not registered_target_is_unique(result) or not set(result.search_association.selected_cluster_source_indices)
            .issubset(support.search_association.selected_cluster_source_indices)):
        raise ValueError('finite_qr_ray_cluster_changed')
    raw = asdict(scan)
    raw['ranges'] = [v if math.isfinite(v) else None for v in scan.ranges]
    proof = dict(policy=POLICY, scan=raw, support=asdict(support),
        center_px=list(center_px),intrinsics=asdict(intrinsics),scan_from_camera=asdict(scan_from_camera),
        parameters={**params,'min_cluster_sample_count':min_cluster_sample_count,
                    'cone_half_angle_rad':cone_half_angle_rad})
    finite = dict(policy='calibrated_scan_range_ray',bearing_rad=bearing,
        uncertainty_rad=uncertainty,optical_depth_m=depth,range_m=support.distance_m,
        range_interval_m=accepted_range_m,scan_from_camera=asdict(scan_from_camera),intrinsics=asdict(intrinsics))
    return result, finite, proof


def validate_qr_range(proof):
    from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
    from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics
    from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
    from scripts.aufgabe04.real_robot.observer.shared_scan_cluster import _equal
    if proof.get('policy') != POLICY:
        raise ValueError('invalid QR range resolution')
    raw = proof['scan']
    scan = PlainLaserScan(**{**raw,'ranges':tuple(math.nan if x is None else x for x in raw['ranges'])})
    def resolver(current, _scan):
        witness = proof['support'].get('witnessed_fragmentation')
        if witness is None:
            result = current
        else:
            result = validated_witnessed_fragmentation(witness)
            # Its scan, exact parameters and freshness must describe this ray.
            entry = witness['current']
            if not _equal(entry['scan'],raw):
                raise ValueError('QR range witness scan changed')
            expected = associate_camera_registered_candidate_lidar_target(scan,
                **entry['parameters'],now_sec=proof['parameters']['now_sec'],
                max_scan_age_sec=proof['parameters']['max_scan_age_sec'])
            if expected != current:
                raise ValueError('QR range witness ray changed')
        if not _equal(asdict(result),proof['support']):
            raise ValueError('QR range support changed')
        return result
    result, finite, reproduced = resolve_qr_range(scan=scan,center_px=proof['center_px'],
        intrinsics=CameraIntrinsics(**proof['intrinsics']),scan_from_camera=RigidTransform(**proof['scan_from_camera']),
        **proof['parameters'],resolver=resolver)
    if proof.get('candidate_association') is not None:
        from scripts.aufgabe04.real_robot.observer.qr_ray_candidate import validate_ray_candidate
        validate_ray_candidate(proof,proof['candidate_association'])
        reproduced['candidate_association']=proof['candidate_association']
    if not _equal(reproduced,proof):
        raise ValueError('QR range proof changed')
    return result, finite
