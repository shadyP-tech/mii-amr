"""Retained-center admission and bounded failure for the 20260928 real run."""
from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from tests.aufgabe04.opposite_center_fixture import recorded_center, ROOT
from scripts.aufgabe04.perception.camera_calibration import rectify_bgr_frame
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.real_robot.observer.target_reconciliation import StoppedTargetReconciliation, validate_reconciliation, RETAINED_POLICY
from scripts.aufgabe04.real_robot.observer.opposite_identity import process_opposite_identity
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import ScanPersistenceContext, _target_for_context, scan_pose_in_map, scan_pose_from_camera_extrinsics
from scripts.aufgabe04.real_robot.observer.opposite_target_geometry import retained_target_center
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search


# Genuine WeChat scale-4 symbol corners from this fixture's rectified frame 24,
# recorded with OpenCV 4.14. The backend is stubbed below to isolate association
# and publication from native timing; these are not the decoder's input extent.
DECODED_CORNERS = ((353.69232177734375, 227.87181091308594),
    (455.0843811035156, 228.6617889404297), (456.62994384765625, 332.7156982421875),
    (356.08385467529297, 332.16233825683594))


@pytest.fixture
def recorded(tmp_path):
    return recorded_center(tmp_path)


def test_recorded_center_fixes_bearing_and_range_without_joining_scan_seam(recorded):
    data,rows,_,_,snapshot,orientation,_ = recorded
    tracker = StoppedTargetReconciliation()
    proof = tracker.observe(**rows[1])
    assert proof is not None, tracker.metadata
    assert proof['policy'] == RETAINED_POLICY
    assert len(proof['entries']) == 1  # Confirmation of an already certified metric center.
    _,envelope,_,_ = validate_reconciliation(proof)
    assert envelope.eligible_cluster_count == 1
    assert envelope.selected_cluster_sample_count >= 3
    assert rows[1]['options']['accepted_range_m'][1] > .52
    assert data['frames'][1]['search']['envelope']['accepted_range_m'][1] < .481
    assert math.degrees(rows[1]['options']['map_bearing_rad']) < 1
    assert math.degrees(data['frames'][1]['search']['envelope']['map_bearing_rad']) > 12
    assert proof['retained_orientation']['bounded_orientation'] == orientation['bounded_orientation']
    for row in (rows[0],rows[2],rows[3]):
        assert tracker.observe(**row) is None
        assert 'compact three-beam cluster' in tracker.metadata['reason']


def test_retained_confirmation_rejects_missing_changed_stale_or_competing_sources(recorded,tmp_path):
    from scripts.aufgabe04.stations.candidate_snapshot import write_candidate_snapshot
    _,rows,_,_,snapshot,orientation,_ = recorded
    r = rows[1];proof=StoppedTargetReconciliation().observe(**r)
    for mutate in (
        lambda p:p.pop('retained_orientation'),
        lambda p:p['retained_orientation']['validated_target_center'].__setitem__('y_m',0.),
        lambda p:p['entries'][0].__setitem__('checked_at_sec',r['now_sec']+1),
        lambda p:p['entries'][0]['options'].__setitem__('map_bearing_rad',.3),
        lambda p:p['entries'][0]['options'].__setitem__('accepted_range_m',(.1,1.)),
        lambda p:p['entries'][0].pop('target_geometry_basis'),
        lambda p:p['entries'].append(deepcopy(p['entries'][0])),
    ):
        bad=deepcopy(proof)
        mutate(bad)
        with pytest.raises((ValueError,KeyError)):validate_reconciliation(bad)
    _,_,world,_=validate_reconciliation(proof)
    other=snapshot.candidate_for('survey_candidate_0003')
    other=replace(other,geometry=replace(other.geometry,x_m=world[0],y_m=world[1]))
    snapshot=replace(snapshot,candidates=tuple(other if c.candidate_uid==other.candidate_uid else c for c in snapshot.candidates))
    path=tmp_path/'competitor.json';write_candidate_snapshot(path,snapshot)
    tracker=StoppedTargetReconciliation()
    assert tracker.observe(**{**r,'snapshot_path':path}) is None
    assert 'another candidate' in tracker.metadata['reason']
    with pytest.raises(ValueError):retained_target_center(orientation,stand_center=dict(x_m=0.,y_m=0.))


def frame_context(recorded):
    data,rows,intrinsics,camera,snapshot,orientation,model_path=recorded
    r=rows[1];f=data['frames'][1]
    def tf(parent,child):
        t=next(t for t in f['tf_samples'] if t['target_frame']==parent and t['source_frame']==child)
        return RigidTransform(parent,child,tuple(t['translation_xyz_m']),tuple(t['rotation_xyzw']))
    ci=dict(f['sensors']['camera_info']);ci['header']=SimpleNamespace(**ci['header'])
    image=rectify_bgr_frame(cv2.imread(str(ROOT/'frame_000024.jpg')),SimpleNamespace(**ci),cv2,np)
    return r,tf,image,load_measured_physical_stand_model(model_path)


def test_camera_and_independent_scan_witness_use_identical_certified_geometry(recorded):
    from dataclasses import asdict
    _,_,_,_,snapshot,orientation,_=recorded
    r,tf,_,_=frame_context(recorded);g=snapshot.candidate_for(r['candidate_uid']).geometry
    base=tf('base_footprint','camera');laser=tf('base_scan','camera')
    context=ScanPersistenceContext(target_key=r['target_key'],epoch_key='0',
        robot_pose=Pose2D(*r['robot_pose']),scan_pose_map=scan_pose_in_map(r['scan_from_map'].translation_xyz_m,r['scan_from_map'].rotation_xyzw),
        image_stamp_sec=r['image_stamp_sec'],candidate_x_m=g.x_m,candidate_y_m=g.y_m,
        stand_radius_m=g.radius_m,stand_uncertainty_m=g.uncertainty_m,lidar_range_tolerance_m=.04,
        scan_pose_robot=scan_pose_from_camera_extrinsics(base.translation_xyz_m,base.rotation_xyzw,laser.translation_xyz_m,laser.rotation_xyzw),
        retained_orientation=orientation)
    actual=_target_for_context(asdict(context))
    assert actual.accepted_range_m == pytest.approx(r['options']['accepted_range_m'])
    assert actual.bearing_rad == pytest.approx(r['options']['map_bearing_rad'])


@pytest.mark.parametrize('payload', ['symbol', 'cornerless', 'full_roi', 'background', 'multiple', 'late'])
@pytest.mark.parametrize("confirmed", [True, False])
def test_recorded_producer_decodes_before_binding_and_keeps_retained_angle(recorded, confirmed, payload):
    from scripts.aufgabe04.real_robot.observer.qr_observation_pose import QrObservationPoseFallback
    from scripts.aufgabe04.artifacts.qr_verified_observation_pose import build_qr_verified_observation_pose, validate_qr_verified_observation_pose, SOURCE_GATES
    _,_,intrinsics,_,snapshot,orientation,_=recorded
    r,tf,image,model=frame_context(recorded);proof=StoppedTargetReconciliation().observe(**r)
    clock = {'now': r['now_sec']}
    frames=[];states=[]
    snapshot_update=SimpleNamespace(target_key=r['target_key'],motion_epoch=0,poisoned=False,
        tentative_qr_id='Start',latched_qr_id=None,as_dict=lambda:{})
    update=SimpleNamespace(snapshot=snapshot_update,motion_epoch_reset=False,frame_accepted=True,qr_sample_accepted=True)
    adapter=SimpleNamespace(args=SimpleNamespace(stand_id=r['candidate_uid'],sync_tolerance_sec=.1,
        lidar_cone_half_angle_deg=3.,backside_registration_max_bearing_delta_deg=12.,max_sensor_age_sec=.5,
        candidate_centering_json=None),cv2=cv2,stand_model_profile=model,_capture_pending={},
        node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=int(clock['now']*1e9)))),
        _target_evidence_key=lambda:r['target_key'],_write_status=lambda state,**kw:states.append((state,kw)),
        _record_observation_frame=lambda **kw:frames.append(kw) or update)
    search_options = dict(scan=r['scan'], scan_from_map=r['scan_from_map'],
        camera_from_map=tf('camera','map'), intrinsics=intrinsics, model_profile=model,
        image_stamp_sec=r['image_stamp_sec'], sync_tolerance_sec=.1,
        target_reconciliation=proof if confirmed else None,
        now_sec=r['now_sec'], max_scan_age_sec=.5, **r['options'])
    search, _ = current_scan_qr_search(**search_options)
    roi = search.roi
    def decoded(pixels, _cv2, **options):
        assert options['identity_only'] is False
        assert np.array_equal(pixels, image[roi.y0:roi.y1, roi.x0:roi.x1])
        corners = tuple((x-roi.x0,y-roi.y0) for x,y in DECODED_CORNERS)
        if payload == 'cornerless':
            corners = None
        elif payload == 'full_roi':
            h,w = pixels.shape[:2]
            corners = ((0.,0.),(float(w-1),0.),(float(w-1),float(h-1)),(0.,float(h-1)))
        elif payload == 'background':
            # The small background candidate's QR cannot borrow the foreground
            # head's retained angle and unique scan cluster.
            corners = ((178.,181.),(201.,181.),(201.,204.),(178.,204.))
        elif payload == 'late':
            clock['now'] = r['image_stamp_sec'] + .6
        result = (DecodedQrObservation('Start',corners,'recorded_wechat',4.),)
        return result + (DecodedQrObservation('Other',corners,'test',4.),) if payload == 'multiple' else result
    # Real frame/head search, scan association, candidate exclusion and receipt
    # validation; only the decoder's current output is replayed deterministically.
    with patch('scripts.aufgabe04.real_robot.observer.opposite_identity.detect_qr_observations_bgr',
               side_effect=decoded) as decode:
        process_opposite_identity(adapter,context=SimpleNamespace(orientation=orientation,snapshot=snapshot),
            frame=image,intrinsics=intrinsics,robot_pose=Pose2D(*r['robot_pose']),camera_signature=(1,2,3,4),
            image_stamp_sec=r['image_stamp_sec'],scan=r['scan'],scan_from_map=r['scan_from_map'],camera_from_map=tf('camera','map'),
            map_bearing_rad=r['options']['map_bearing_rad'],accepted_range_m=r['options']['accepted_range_m'],
            scan_from_camera=tf('base_scan','camera'),base_from_camera=tf('base_footprint','camera'),image_stamp=None,
            target_reconciliation=proof if confirmed else None,require_target_reconciliation=True)
        decode.assert_called_once()
        metadata = adapter._capture_pending['detector_metadata']
        assert metadata['current_angle_refit'] is False
        assert metadata['provisional_qr_search']['attempted'] is True
        if payload != 'symbol':
            assert not frames[-1]['qr_texts'] and frames[-1]['axis_yaw_rad'] is None
            assert not metadata['provisional_qr_search']['accepted']
            assert getattr(adapter,'_pending_qr_observation_pose',None) is None
            return
    assert frames[-1]['qr_texts']==('Start',) and frames[-1]['axis_yaw_rad'] is None
    current=adapter._pending_qr_observation_pose
    assert QrObservationPoseFallback(delay_sec=1.5).observe(current,update=update,
        observed_at_sec=r['now_sec'],now_monotonic_sec=1.) is not None
    receipt=build_qr_verified_observation_pose(candidate_uid=r['candidate_uid'],stream_id='recorded',planning_frame='map',
        qr_id='Start',stand_center=dict(zip(('x_m','y_m'),r['stand_center'])),robot_pose=dict(zip(('x_m','y_m','yaw_rad'),r['robot_pose'])),
        sensor_stamp_sec=r['image_stamp_sec'],scan_stamp_sec=r['scan'].scan_stamp_sec,checked_at_sec=r['now_sec'],
        robot_profile_sha256=orientation['robot_profile_sha256'],calibration_profile_sha256=orientation['calibration_profile_sha256'],
        stand_model_profile_sha256=model.sha256,target_key=r['target_key'],motion_epoch=0,
        camera_signature=current.camera_signature,qr_corners_px=None,image_shape=current.image_shape,
        qr_binding=current.qr_binding.metadata(),source_gates={k:True for k in SOURCE_GATES},
        retained_backside_orientation=orientation,arrival_target_reconciliation=proof if confirmed else None,
        localization_provenance=dict(map_frame='map',base_frame='base_footprint',scan_frame='base_scan',camera_frame='camera',
            exact_image_transform_stamp_sec=r['image_stamp_sec'],exact_scan_transform_stamp_sec=r['scan'].scan_stamp_sec))
    assert validate_qr_verified_observation_pose(receipt)['qr_id']=='Start'
    assert receipt['stand_axis_rad']==orientation['stand_axis_rad']
    assert adapter._opposite_identity_failure is None
    if not confirmed:
        from scripts.aufgabe04.artifacts.content_store import content_hashed_payload
        from scripts.aufgabe04.artifacts.qr_verified_observation_pose import HASH_FIELD
        for mutation in ('remove_proof', 'policy', 'center', 'scan', 'candidate_uid'):
            forged = deepcopy(receipt)
            crop = forged['qr_binding']['current_head_binding']
            if mutation == 'remove_proof':
                crop.pop('decoded_symbol_range_resolution')
            elif mutation == 'policy':
                crop['search']['policy'] = 'current_scan_qr_search'
            elif mutation == 'center':
                crop['decoded_symbol_range_resolution']['center_px'][0] += 2.
            elif mutation == 'candidate_uid':
                forged['candidate_uid'] = crop['candidate_uid'] = 'survey_candidate_0002'
            else:
                ray = crop['decoded_symbol_range_resolution']
                index = ray['support']['search_association']['selected_cluster_source_indices'][0]
                ray['scan']['ranges'][index] += .1
            forged.pop(HASH_FIELD)
            with pytest.raises(ValueError):
                validate_qr_verified_observation_pose(content_hashed_payload(forged, hash_field=HASH_FIELD))


@pytest.mark.parametrize('scene', ['one_ray', 'two_rays', 'neighbor'])
def test_decoded_symbol_uses_its_ray_when_broad_search_has_two_clusters(recorded, scene):
    """Synthetic distractors modify the real replay; they are not run evidence."""
    from scripts.aufgabe04.real_robot.observer.qr_candidate_search import retained_opposite_qr_search
    from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import decoded_symbol_identity_crop
    from scripts.aufgabe04.artifacts.qr_verified_observation_pose import _validate_opposite_crop
    _,_,intrinsics,_,snapshot,orientation,_ = recorded
    r,tf,image,model = frame_context(recorded)
    ranges = list(r['scan'].ranges)
    ranges[6] = ranges[7] = .46  # Inside broad map cone, outside the decoded ray.
    if scene == 'two_rays':
        ranges[220] = .35  # Separate in-range cluster inside the symbol's ray.
    if scene == 'neighbor':
        target = snapshot.candidate_for(r['candidate_uid'])
        other = next(c for c in snapshot.candidates if c.candidate_uid != target.candidate_uid)
        center = orientation['validated_target_center']
        overlapping = replace(target.geometry, x_m=center['x_m'], y_m=center['y_m'])
        snapshot = replace(snapshot, candidates=tuple(replace(c, geometry=overlapping)
            if c.candidate_uid == other.candidate_uid else c for c in snapshot.candidates))
    options = dict(scan=replace(r['scan'],ranges=tuple(ranges)), scan_from_map=r['scan_from_map'],
        camera_from_map=tf('camera','map'), intrinsics=intrinsics, model_profile=model,
        image_stamp_sec=r['image_stamp_sec'], sync_tolerance_sec=.1,
        now_sec=r['now_sec'], max_scan_age_sec=.5, **r['options'])
    search, info = current_scan_qr_search(**options)
    assert search is None and info['envelope']['eligible_cluster_count'] >= 2
    search = retained_opposite_qr_search(orientation=orientation, camera_from_map=options['camera_from_map'],
        intrinsics=intrinsics, model_profile=model, image_stamp_sec=r['image_stamp_sec'],
        now_sec=r['now_sec'], max_scan_age_sec=.5)
    corners = tuple((x-search.roi.x0, y-search.roi.y0) for x,y in DECODED_CORNERS)
    diagnostic = {}
    attempt,crop,support = decoded_symbol_identity_crop(
        (DecodedQrObservation('Start',corners,'recorded_wechat',4.),),
        search_attempt=search, image_shape=image.shape[:2], candidate_uid=r['candidate_uid'],
        snapshot=snapshot, scan_from_camera=tf('base_scan','camera'), diagnostics=diagnostic, **options)
    if scene != 'one_ray':
        assert attempt is None
        assert diagnostic['reason'] == ('finite_qr_ray_target_not_unique' if scene == 'two_rays'
                                        else 'target_crop_overlap_unresolved')
        return
    assert attempt is not None and support is not None
    assert crop['search']['envelope']['eligible_cluster_count'] == 1
    assert crop['search']['policy'] == 'current_decoded_qr_ray'
    data = dict(candidate_uid=r['candidate_uid'], qr_binding={'association':crop['search']['envelope']})
    _validate_opposite_crop(crop, data, r['image_stamp_sec'], r['scan'].scan_stamp_sec, image.shape[:2])
