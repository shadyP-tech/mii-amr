"""Recomputable broad target proof and its narrower current camera ray.

Raw cluster counts stay intact. Only real current beams from an independently
witnessed envelope may support the narrow ray; a second target is never hidden.
"""
from dataclasses import asdict, dataclass, replace
import json
import math

from scripts.aufgabe04.perception import candidate_lidar_association as lidar

SUBSET_KIND = 'witnessed_registration_envelope_subset'


@dataclass(frozen=True)
class WitnessedEnvelope(lidar.CandidateLidarAssociation):
    witnessed_fragmentation: dict | None = None


def _equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def envelope_from_proof(proof, *, scan, map_bearing_rad, cone_half_angle_rad,
                        accepted_range_m, now_sec, max_scan_age_sec):
    from scripts.aufgabe04.real_robot.observer.scan_target_persistence import validated_witnessed_fragmentation
    if proof.get('kind') == SUBSET_KIND:
        raise ValueError('a narrow ray cannot certify the broad envelope')
    result = validated_witnessed_fragmentation(proof)
    raw = asdict(scan)
    raw['ranges'] = [v if math.isfinite(v) else None for v in scan.ranges]
    cluster = result.search_association
    if (not _equal(raw, proof['current']['scan'])
            or result.map_bearing_rad != map_bearing_rad
            or result.registered_search_bearing_rad != map_bearing_rad
            or cluster.cone_half_angle_rad != cone_half_angle_rad
            or tuple(cluster.accepted_range_m) != tuple(accepted_range_m)
            or not 0 <= now_sec-scan.scan_stamp_sec <= max_scan_age_sec):
        raise ValueError('witnessed envelope differs from current scan or search bounds')
    return WitnessedEnvelope(**asdict(replace(cluster, scan_age_sec=now_sec-scan.scan_stamp_sec)),
                             witnessed_fragmentation=proof)


def envelope_metadata_is_unique(value):
    if not isinstance(value, dict) or value.get('associated') is not True:
        return False
    proof = value.get('witnessed_fragmentation')
    if proof is None:
        return type(value.get('eligible_cluster_count')) is int and value['eligible_cluster_count'] == 1
    try:
        from scripts.aufgabe04.real_robot.observer.scan_target_persistence import validated_witnessed_fragmentation
        if proof.get('kind') == SUBSET_KIND:
            return False
        source = validated_witnessed_fragmentation(proof).search_association
        expected = {**asdict(source), 'scan_age_sec': value['scan_age_sec'], 'witnessed_fragmentation': proof}
        return 0 <= value['scan_age_sec'] <= .5 and _equal(expected, value)
    except (ValueError, TypeError, KeyError, ArithmeticError):
        return False


def envelope_is_unique(envelope):
    return envelope_metadata_is_unique(asdict(envelope))


def validate_cluster_receipt_context(metadata, receipt):
    proof = metadata.get('witnessed_fragmentation')
    if proof is None:
        return
    if proof.get('kind') == SUBSET_KIND:
        proof = proof['envelope']
    context = proof['current']['context']
    if (context['target_key'] != receipt['target_key']
            or context['epoch_key'] != str(receipt['motion_epoch'])
            or context['candidate_x_m'] != receipt['stand_center']['x_m']
            or context['candidate_y_m'] != receipt['stand_center']['y_m']
            or context['image_stamp_sec'] != receipt['sensor_stamp_sec']
            or context['robot_pose'] != receipt['robot_pose']):
        raise ValueError('witnessed cluster differs from QR candidate, pose or epoch')


def bind_ray_to_envelope(association, scan, envelope, *, now_sec, max_scan_age_sec):
    """Derive the narrow cone from a proved envelope, retaining every raw gate."""
    if association.associated or getattr(envelope, 'witnessed_fragmentation', None) is None:
        return association
    if association.rejection_reason != 'ambiguous_registered_camera_clusters':
        return association
    search = association.search_association
    proof = dict(schema_version=1, kind=SUBSET_KIND, envelope=envelope.witnessed_fragmentation,
        parameters=dict(map_bearing_rad=association.map_bearing_rad,
            observed_camera_bearing_rad=association.registered_search_bearing_rad,
            max_camera_map_bearing_delta_rad=association.max_camera_map_bearing_delta_rad,
            cone_half_angle_rad=search.cone_half_angle_rad, accepted_range_m=list(search.accepted_range_m),
            min_cluster_sample_count=search.min_cluster_sample_count,
            max_range_jump_m=search.max_range_jump_m, max_point_gap_m=search.max_point_gap_m),
        now_sec=now_sec, max_scan_age_sec=max_scan_age_sec)
    # Recheck that the caller supplied exactly the scan in the broad proof.
    raw = asdict(scan)
    raw['ranges'] = [v if math.isfinite(v) else None for v in scan.ranges]
    if (not envelope_is_unique(envelope) or not _equal(raw, proof['envelope']['current']['scan'])
            or not 0 <= now_sec-scan.scan_stamp_sec <= max_scan_age_sec):
        raise ValueError('ray envelope differs from current scan')
    return validate_envelope_subset(proof)


def validate_envelope_subset(proof):
    from scripts.aufgabe04.real_robot.observer.scan_target_persistence import validated_witnessed_fragmentation
    from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
    if (set(proof) != {'schema_version','kind','envelope','parameters','now_sec','max_scan_age_sec'}
            or type(proof['schema_version']) is not int or proof['schema_version'] != 1
            or proof['kind'] != SUBSET_KIND or proof['envelope'].get('kind') == SUBSET_KIND):
        raise ValueError('invalid witnessed envelope subset')
    broad = validated_witnessed_fragmentation(proof['envelope']).search_association
    raw = proof['envelope']['current']['scan']
    scan = PlainLaserScan(**{**raw, 'ranges': tuple(math.nan if v is None else v for v in raw['ranges'])})
    p = proof['parameters']
    if (not 0 < proof['max_scan_age_sec'] <= .5
            or not 0 <= proof['now_sec']-scan.scan_stamp_sec <= proof['max_scan_age_sec']
            or not 0 < p['cone_half_angle_rad'] <= math.radians(3)+1e-9
            or p['min_cluster_sample_count'] < 1
            or p['max_range_jump_m'] != broad.max_range_jump_m or p['max_point_gap_m'] != broad.max_point_gap_m
            or tuple(p['accepted_range_m']) != tuple(broad.accepted_range_m)
            or abs(math.remainder(p['observed_camera_bearing_rad']-broad.map_bearing_rad, math.tau))
                + p['cone_half_angle_rad'] > broad.cone_half_angle_rad+1e-9):
        raise ValueError('current QR ray exceeds its witnessed envelope')
    result = lidar.associate_camera_registered_candidate_lidar_target(scan, **p,
        now_sec=proof['now_sec'], max_scan_age_sec=proof['max_scan_age_sec'])
    if result.rejection_reason != 'ambiguous_registered_camera_clusters':
        raise ValueError('subset proof must resolve current fragmentation only')
    samples = tuple(s for s in lidar._valid_samples_in_map_cone(scan,
        map_bearing_rad=p['observed_camera_bearing_rad'], cone_half_angle_rad=p['cone_half_angle_rad'])
        if p['accepted_range_m'][0] <= s.distance_m <= p['accepted_range_m'][1])
    if (len(samples) < p['min_cluster_sample_count'] or
            not {s.index for s in samples}.issubset(broad.selected_cluster_source_indices)):
        raise ValueError('current ray contains a competing target')
    cluster = lidar._build_cluster(samples)
    search = replace(result.search_association, associated=True, rejection_reason='',
        distance_m=cluster.distance_m, selected_cluster_sample_count=len(samples),
        selected_cluster_start_index=samples[0].index, selected_cluster_end_index=samples[-1].index,
        selected_cluster_source_indices=tuple(s.index for s in samples),
        selected_cluster_bearing_rad=cluster.bearing_rad,
        selected_cluster_bearing_delta_from_map_rad=abs(math.remainder(cluster.bearing_rad-p['observed_camera_bearing_rad'],math.tau)),
        selection_source=SUBSET_KIND)
    return replace(result, associated=True, distance_m=cluster.distance_m, rejection_reason='',
                   search_association=search, unique_eligible_cluster_required=False, witnessed_fragmentation=proof)


class ScanWitnessFanout:
    """Share exact-time scan ingestion, while keeping broad/narrow histories separate."""
    def __init__(self, *owners):
        self.owners = owners

    def reset(self):
        for owner in self.owners:
            owner.reset()

    def note_collection(self, *args, **kwargs):
        for owner in self.owners:
            owner.note_collection(*args, **kwargs)

    def ingest_scan(self, *args, **kwargs):
        results = [owner.ingest_scan(*args, **kwargs) for owner in self.owners]
        return all(results)
