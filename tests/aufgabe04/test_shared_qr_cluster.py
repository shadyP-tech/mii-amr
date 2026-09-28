"""Recorded scans plus explicitly offline-decoded QR_002 corners; no robot I/O."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import pytest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, ImageRoi
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import ScanPersistenceContext, StoppedScanTargetPersistence
from scripts.aufgabe04.real_robot.observer.qr_target_binding import bind_qr_observations_to_target
from scripts.aufgabe04.real_robot.observer.qr_finite_range import validate_qr_range

DATA = json.loads((Path(__file__).parent/'fixtures/qr002_cluster_20260923.json').read_text())


def unpack(row):
    return (PlainLaserScan(**{**row['scan'],'ranges':tuple(math.nan if x is None else x for x in row['scan']['ranges'])}),
        ScanPersistenceContext(**{**row['context'],**{k:Pose2D(**row['context'][k])
            for k in ('robot_pose','scan_pose_map','scan_pose_robot')}}))


def replay():
    state = StoppedScanTargetPersistence()
    witnesses = sorted(DATA['witnesses'],key=lambda w:w['scan']['scan_stamp_sec'])
    index = 0
    results = []
    for row in DATA['frames']:
        scan, context = unpack(row)
        while index < len(witnesses) and witnesses[index]['scan']['scan_stamp_sec'] < scan.scan_stamp_sec:
            old, old_context = unpack(witnesses[index])
            state.ingest_scan(old,context=old_context,now_sec=witnesses[index]['now_sec'],max_scan_age_sec=.5)
            index += 1
        raw = associate_camera_registered_candidate_lidar_target(scan,**row['options'],
            observed_camera_bearing_rad=row['head_bearing_rad'],now_sec=row['now_sec'],max_scan_age_sec=.5)
        state.resolve(raw,scan,context=context,now_sec=row['now_sec'],max_scan_age_sec=.5)
        observations = tuple(DecodedQrObservation(**{**o,'corners':None if o['corners'] is None
            else tuple(map(tuple,o['corners']))}) for o in row['offline_decoded_qr'])
        options = dict(roi=ImageRoi(0,0,800,600,100),intrinsics=CameraIntrinsics(**DATA['intrinsics']),
            scan_from_camera=RigidTransform(**DATA['scan_from_camera']),scan=scan,**row['options'],
            now_sec=row['now_sec'],max_scan_age_sec=.5,min_cluster_sample_count=1,
            camera_registration_accepted=True,allow_independent_registration=True,
            candidate_context=dict(snapshot_path=str(Path(__file__).parent/'fixtures/qr002_candidates_20260923.json'),
                candidate_uid='survey_candidate_0002',scan_from_map=dict(parent_frame='base_scan',child_frame='map',
                    translation_xyz_m=row['scan_from_map']['translation_xyz_m'],
                    rotation_xyzw=row['scan_from_map']['rotation_xyzw'])),
            resolve_lidar_association=lambda a,s:state.preview(a,s,context=context,now_sec=row['now_sec'],max_scan_age_sec=.5))
        result = bind_qr_observations_to_target(observations,**options)
        results.append((row,result,options,observations))
    return results


@pytest.fixture(scope='module')
def recorded_results():
    return replay()


def test_recorded_qr_can_use_its_proved_ray_without_broad_cluster_uniqueness(recorded_results):
    from unittest.mock import patch
    with patch('scripts.aufgabe04.real_robot.observer.qr_finite_range.resolve_qr_range',
               side_effect=ValueError('finite_qr_range_not_unique')):
        baseline=[i for i,(_,r,_,_) in enumerate(replay(),1) if r.accepted]
    assert baseline==[35,41,112]
    accepted = [i for i,(_,r,_,_) in enumerate(recorded_results,1) if r.accepted]
    assert len(accepted) > 3, accepted  # Original run had only 3 QR association frames.
    proofs = [r.range_resolution for _,r,_,_ in recorded_results if r.accepted and r.range_resolution]
    assert proofs
    for proof in proofs:
        result,finite = validate_qr_range(json.loads(json.dumps(proof)))
        assert result.associated and finite['range_interval_m']


@pytest.mark.parametrize('mutation',[
    lambda p:p['center_px'].__setitem__(0,p['center_px'][0]+20),
    lambda p:p['parameters'].update(now_sec=p['parameters']['now_sec']+1),
    lambda p:p['parameters'].update(cone_half_angle_rad=.2),
    lambda p:p['support'].update(distance_m=.1),
    lambda p:p['scan']['ranges'].__setitem__(0,.15),
])
def test_range_proof_cannot_change_pixels_source_gates_or_cluster(recorded_results,mutation):
    proof=deepcopy(next(r.range_resolution for _,r,_,_ in recorded_results if r.accepted and r.range_resolution))
    mutation(proof)
    with pytest.raises(ValueError):validate_qr_range(proof)


def test_stale_and_conflicting_symbols_still_fail(recorded_results):
    _,_,options,observations=next(x for x in recorded_results if x[1].accepted and x[1].range_resolution)
    stale={**options,'now_sec':options['now_sec']+1}
    assert not bind_qr_observations_to_target(observations,**stale).accepted
    other=replace(observations[0],text='QR_999')
    assert not bind_qr_observations_to_target((*observations,other),**options).accepted


def test_competing_return_inside_qr_ray_is_not_discarded(recorded_results):
    _,_,options,observations=next(x for x in recorded_results if x[1].accepted and x[1].range_resolution)
    # A separate in-range object in the same QR cone contradicts all old proof.
    scan=options['scan'];ranges=list(scan.ranges)
    ranges[0]=options['accepted_range_m'][0]+.005
    result=bind_qr_observations_to_target(observations,**{**options,'scan':replace(scan,ranges=tuple(ranges)),
                                                       'resolve_lidar_association':None})
    assert not result.accepted


def test_recorded_range_proof_roundtrips_through_discovery_receipt(recorded_results):
    from scripts.aufgabe04.artifacts.qr_verified_observation_pose import build_qr_verified_observation_pose, SOURCE_GATES
    for row,binding,_,observations in recorded_results:
        if not binding.accepted or not binding.range_resolution:
            continue
        context=row['context']
        qr=build_qr_verified_observation_pose(candidate_uid='survey_candidate_0002',stream_id='offline',qr_id='QR_002',
            planning_frame='map',stand_center=dict(x_m=context['candidate_x_m'],y_m=context['candidate_y_m']),
            robot_pose=context['robot_pose'],sensor_stamp_sec=context['image_stamp_sec'],
            scan_stamp_sec=row['scan']['scan_stamp_sec'],checked_at_sec=row['now_sec'],
            robot_profile_sha256='a'*64,calibration_profile_sha256='b'*64,stand_model_profile_sha256='c'*64,
            target_key=context['target_key'],motion_epoch=0,camera_signature=(640.,640.,400.,300.),
            qr_corners_px=observations[0].corners,image_shape=(600,800),qr_binding=binding.metadata(),
            source_gates={k:True for k in SOURCE_GATES},localization_provenance=dict(map_frame='map',
                base_frame='base_footprint',scan_frame='base_scan',camera_frame='camera',
                exact_image_transform_stamp_sec=context['image_stamp_sec'],exact_scan_transform_stamp_sec=row['scan']['scan_stamp_sec']))
        assert qr['completion_authorized'] and not qr['facing_ready']


def test_ray_resolution_recomputes_witnessed_endpoint_support():
    from tests.aufgabe04.test_scan_endpoint_persistence import inputs, seed
    from scripts.aufgabe04.real_robot.observer.qr_finite_range import resolve_qr_range
    state=seed();scan,context,p=inputs(57)
    # Synthetic calibrated QR ray, pointed at this recorded fragmented target.
    options=dict(scan=scan,center_px=(400-640*math.tan(p['observed_camera_bearing_rad']),300),
        intrinsics=CameraIntrinsics(800,600,640.,640.,400.,300.),
        scan_from_camera=RigidTransform('base_scan','camera',(0.,0.,0.),(-.5,.5,-.5,.5)),
        map_bearing_rad=p['map_bearing_rad'],cone_half_angle_rad=p['cone_half_angle_rad'],
        accepted_range_m=p['accepted_range_m'],now_sec=p['now_sec'],max_scan_age_sec=p['max_scan_age_sec'],
        max_camera_map_bearing_delta_rad=math.radians(12),min_cluster_sample_count=1,
        resolver=lambda a,s:state.preview(a,s,context=context,now_sec=p['now_sec'],max_scan_age_sec=p['max_scan_age_sec']))
    result,_,proof=resolve_qr_range(**options)
    assert result.associated and result.search_association.eligible_cluster_count==2
    assert proof['support']['witnessed_fragmentation'] is not None
    assert validate_qr_range(json.loads(json.dumps(proof)))[0]==result
    bad=deepcopy(proof);bad['support']['witnessed_fragmentation']['witnesses'].pop()
    with pytest.raises(ValueError):validate_qr_range(bad)


def test_shared_envelope_handles_internal_gap_in_valid_circular_cluster():
    from scripts.aufgabe04.real_robot.observer.qr_candidate_search import qr_registration_envelope
    from scripts.aufgabe04.real_robot.observer.shared_scan_cluster import envelope_is_unique, envelope_metadata_is_unique
    row=DATA['frames'][31];scan,context=unpack(row);state=StoppedScanTargetPersistence()
    # Controlled witnesses fill the recorded single missing beam. They are
    # synthetic continuity evidence, never reported as recorded robot success.
    ranges=list(scan.ranges);ranges[215]=(ranges[214]+ranges[216])/2
    def observe(s,c,now):
        return qr_registration_envelope(s,**row['options'],now_sec=now,max_scan_age_sec=.5,
            resolve_lidar_association=lambda a,b:state.resolve(a,b,context=c,now_sec=now,max_scan_age_sec=.5))
    for stamp in (100.,100.1,100.2):
        witness=replace(scan,ranges=tuple(ranges),scan_stamp_sec=stamp,receipt_sec=stamp)
        assert envelope_is_unique(observe(witness,replace(context,image_stamp_sec=stamp),stamp+.05))
    current=replace(scan,scan_stamp_sec=100.3,receipt_sec=100.3)
    envelope=observe(current,replace(context,image_stamp_sec=100.3),100.35)
    assert envelope.eligible_cluster_count==2 and envelope_is_unique(envelope)
    assert envelope.selected_cluster_wraps_scan_seam
    persisted=json.loads(json.dumps(asdict(envelope)))
    assert envelope_metadata_is_unique(persisted)
    persisted['selected_cluster_source_indices']=[0]
    assert not envelope_metadata_is_unique(persisted)


def test_neighbor_candidate_explanation_blocks_qr_ray(recorded_results,tmp_path):
    from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot, write_candidate_snapshot
    from scripts.aufgabe04.real_robot.observer.qr_ray_candidate import bind_ray_candidate
    _,binding,options,_=next(x for x in recorded_results if x[1].accepted and x[1].range_resolution)
    context=options['candidate_context'];snapshot=load_candidate_snapshot(Path(context['snapshot_path']))
    target=snapshot.candidate_for('survey_candidate_0002')
    other=next(c for c in snapshot.candidates if c.candidate_uid!=target.candidate_uid)
    overlapping=replace(other,geometry=replace(other.geometry,x_m=target.geometry.x_m+.02,y_m=target.geometry.y_m))
    modified=replace(snapshot,candidates=tuple(overlapping if c.candidate_uid==other.candidate_uid else c for c in snapshot.candidates))
    path=tmp_path/'competing.json';write_candidate_snapshot(path,modified)
    with pytest.raises(ValueError,match='another candidate'):
        bind_ray_candidate(binding.range_resolution,{**context,'snapshot_path':str(path)})
