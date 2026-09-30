"""Recorded partial envelope recovery keeps complete raw-cluster identity."""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.observer import candidate_position_epoch as epoch_policy
from scripts.aufgabe04.real_robot.observer.target_reconciliation import StoppedTargetReconciliation, validate_reconciliation
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot

ROOT = Path(__file__).parent/'fixtures/candidate_position_epoch_clipped'
UID = 'survey_candidate_0001'


def metadata():
    """Three compact original tuples; the last row matches frame_000005.jpg."""
    return json.loads((ROOT/'observations.json').read_text())


def transform(row, parent, child):
    sample = next(t for t in row['tf_samples']
                  if t['target_frame'] == parent and t['source_frame'] == child)
    return RigidTransform(parent, child, tuple(sample['translation_xyz_m']), tuple(sample['rotation_xyzw']))


def pose(row, parent):
    tf = transform(row, parent, 'base_footprint')
    q = tf.rotation_xyzw
    yaw = math.atan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))
    return (*tf.translation_xyz_m[:2], yaw)


def recorded_rows():
    """Yield reconciliation kwargs rebuilt only from the committed fixture."""
    snapshot = load_candidate_snapshot(ROOT/'candidate_snapshot.json')
    geometry = snapshot.candidate_for(UID).geometry
    for row in metadata()['rows']:
        raw = row['sensors']['scan']
        scan = PlainLaserScan(
            ranges=tuple(math.nan if value is None else float(value) for value in raw['ranges']),
            **{key:raw[key] for key in ('angle_min','angle_max','angle_increment','range_min','range_max')},
            scan_frame_id=raw['header']['frame_id'], scan_stamp_sec=row['scan_stamp_sec'],
            receipt_sec=row['scan_received_ros_sec'], scan_topology_profile='full_rotation')
        options = dict(row['original_options'])
        options['accepted_range_m'] = tuple(options['accepted_range_m'])
        yield dict(snapshot_path=ROOT/'candidate_snapshot.json', candidate_uid=UID,
            planning_frame='map', stand_center=(geometry.x_m,geometry.y_m),
            target_key='recorded-clipped-target', epoch=0, scan=scan,
            scan_from_map=transform(row,'base_scan','map'), robot_pose=pose(row,'map'),
            image_stamp_sec=row['image_stamp_sec'], now_sec=row['now_sec'], options=options,
            position_epoch_path=ROOT/'candidate_frame_projection.json')


def recorded_proof():
    tracker = StoppedTargetReconciliation()
    rows = list(recorded_rows())
    assert tracker.observe(**rows[0]) is None
    assert tracker.observe(**rows[1]) is None
    proof = tracker.observe(**rows[2])
    assert proof is not None, tracker.metadata
    return proof, rows[-1]


def association(row, half_angle):
    return associate_candidate_lidar_target(row['scan'],
        map_bearing_rad=row['options']['map_bearing_rad'], cone_half_angle_rad=math.radians(half_angle),
        accepted_range_m=row['options']['accepted_range_m'], now_sec=row['now_sec'], max_scan_age_sec=.5)


def epoch_result(row):
    snapshot = load_candidate_snapshot(row['snapshot_path'])
    entry = dict(position_epoch=epoch_policy.epoch_reference(row['position_epoch_path']),
        checked_at_sec=row['now_sec'], robot_pose=row['robot_pose'], options=row['options'])
    return epoch_policy.epoch_cluster(entry,snapshot,UID,row['scan'],row['scan_from_map'])


def test_three_recorded_partial_clusters_reconcile_the_complete_stand():
    for row in recorded_rows():
        assert association(row,15).selected_cluster_source_indices == (6,)
        assert association(row,35).selected_cluster_source_indices == (6,7,8,9,10)
    proof, row = recorded_proof()
    _, envelope, world, bearing = validate_reconciliation(proof,
        candidate_uid=UID, image_stamp_sec=row['image_stamp_sec'], scan_stamp_sec=row['scan'].scan_stamp_sec)
    assert envelope.selected_cluster_source_indices == (6,7,8,9,10)
    assert .53 < envelope.distance_m < .56
    assert .16 < math.dist(world,proof['stand_center']) < .19
    assert 13 < math.degrees(bearing) < 15
    assert proof['entries'][-1]['options'] == row['options']
    assert proof['stand_center'] == list(row['stand_center'])
    assert all(entry.get('position_epoch') for entry in proof['entries'])
    assert proof['candidate_geometry_updated'] is False
    assert proof['motion_authorized'] is False


@pytest.mark.parametrize('ordinary_samples, bearing_shift', [(1,0.), (2,.03), (3,.06), (4,.09)])
def test_clipped_beams_recover_the_same_complete_raw_cluster(ordinary_samples,bearing_shift):
    row = list(recorded_rows())[-1]
    # Shift only the synthetic candidate envelope across one beam boundary.
    row['options']['map_bearing_rad'] += bearing_shift
    ordinary = association(row,15)
    assert ordinary.selected_cluster_sample_count == ordinary_samples
    _, envelope, _, _ = epoch_result(row)
    assert envelope.selected_cluster_source_indices == (6,7,8,9,10)
    assert set(ordinary.selected_cluster_source_indices) < set(envelope.selected_cluster_source_indices)


@pytest.mark.parametrize('defect', ['competitor','two_beams','no_epoch','stale'])
def test_partial_beams_do_not_bypass_unique_complete_fresh_epoch_support(defect):
    tracker = StoppedTargetReconciliation()
    for row in recorded_rows():
        ranges = list(row['scan'].ranges)
        if defect == 'competitor':
            index = round((math.radians(25)-row['scan'].angle_min)/row['scan'].angle_increment)
            ranges[index] = .45
        elif defect == 'two_beams':
            ranges = [math.inf]*len(ranges)
            ranges[6:8] = row['scan'].ranges[6:8]
        elif defect == 'no_epoch':
            row.pop('position_epoch_path')
        else:
            row['now_sec'] += 1.
        row['scan'] = replace(row['scan'],ranges=tuple(ranges))
        assert tracker.observe(**row) is None
    expected = {'competitor':'competing clusters','two_beams':'three-beam',
                'no_epoch':'three-beam','stale':'stale'}
    assert expected[defect] in tracker.metadata['reason']


def test_complete_ordinary_cluster_cannot_borrow_epoch_recovery():
    row = list(recorded_rows())[-1]
    row['options']['map_bearing_rad'] = math.radians(14)
    assert association(row,15).selected_cluster_source_indices == (6,7,8,9,10)
    with pytest.raises(ValueError,match='clipped subset'):
        epoch_result(row)


def test_multiple_ordinary_fragments_are_rejected_even_when_expanded_range_connects_them():
    row = list(recorded_rows())[-1]
    ranges = [math.inf]*len(row['scan'].ranges)
    # The frozen range starts 1.35 cm nearer. Its middle beam can bridge two
    # ordinary fragments, but a broad connected component is not enough to
    # certify that the ordinary envelope contained just one clipped cluster.
    ranges[6:9] = (.38,.37,.38)
    row['scan'] = replace(row['scan'],ranges=tuple(ranges))
    row['options']['map_bearing_rad'] = math.radians(14)
    assert association(row,15).eligible_cluster_count == 2
    broad = associate_candidate_lidar_target(row['scan'],map_bearing_rad=math.radians(14),
        cone_half_angle_rad=math.radians(35),accepted_range_m=(.3638,.5973))
    assert broad.selected_cluster_source_indices == (6,7,8)
    with pytest.raises(ValueError,match='clipped subset'):
        epoch_result(row)


def test_equal_counts_with_different_raw_indices_are_not_a_recovery_witness(monkeypatch):
    row = list(recorded_rows())[-1]
    original = epoch_policy.associate_candidate_lidar_target
    def different_ordinary_cluster(scan, **options):
        result = original(scan,**options)
        if options['cone_half_angle_rad'] == math.radians(15):
            return replace(result,selected_cluster_source_indices=(20,),
                selected_cluster_start_index=20,selected_cluster_end_index=20)
        return result
    # Defensive component-boundary test: a matching count must never replace
    # the source-index membership proof, even if an upstream result changes.
    monkeypatch.setattr(epoch_policy,'associate_candidate_lidar_target',different_ordinary_cluster)
    with pytest.raises(ValueError,match='clipped subset'):
        epoch_result(row)


@pytest.mark.parametrize('defect', ['stale','moved','reused','epoch_hash','missing_sample','scan_binding'])
def test_clipped_recovery_proof_keeps_source_and_stationarity_guards(defect):
    proof, row = recorded_proof()
    entry = proof['entries'][-1]
    if defect == 'stale':
        entry['checked_at_sec'] += 1.
    elif defect == 'moved':
        entry['robot_pose'][0] += .04
    elif defect == 'reused':
        proof['entries'][-1] = copy.deepcopy(proof['entries'][-2])
    elif defect == 'epoch_hash':
        for sample in proof['entries']:
            sample['position_epoch']['sha256'] = '0'*64
    elif defect == 'missing_sample':
        proof['entries'].pop()
    else:
        ranges = list(row['scan'].ranges)
        ranges[6] += .001
        row['scan'] = replace(row['scan'],ranges=tuple(ranges))
    with pytest.raises(ValueError):
        epoch_policy.validated_reconciliation_envelope(proof,scan=row['scan'],
            map_bearing_rad=row['options']['map_bearing_rad'],accepted_range_m=row['options']['accepted_range_m'])
