"""Regressions for the recorded off-center Start stand after the opposite move."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import math
import pytest
import cv2

from tests.aufgabe04.opposite_reconciliation_fixture import recorded_opposite
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search
from scripts.aufgabe04.real_robot.observer.opposite_target_support import detect_opposite_target_support, validate_target_support
from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import exclusive_identity_crop, bind_crop_text
from scripts.aufgabe04.real_robot.observer.candidate_centering import build_camera_centering_advisory
from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose
from scripts.aufgabe04.real_robot.observer.target_reconciliation import StoppedTargetReconciliation, validate_reconciliation
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.artifacts.backside_axis_observation import load_backside_axis_observation
from scripts.aufgabe04.artifacts.retained_backside_orientation import orientation_record


@pytest.fixture
def recorded(tmp_path):
    return recorded_opposite(tmp_path)


def support_for(recorded, *, proof=True):
    image,options,snapshot,tf,reconciliation,orientation,row,axis=recorded
    search=current_scan_qr_search(**options)
    keep=('intrinsics','model_profile','scan','image_stamp_sec','now_sec','map_bearing_rad',
          'cone_half_angle_rad','accepted_range_m','max_scan_age_sec','max_camera_map_bearing_delta_rad')
    support=detect_opposite_target_support(image,cv2,attempt=search[0],
        scan_from_camera=tf('base_scan','camera'),target_reconciliation=reconciliation if proof else None,
        max_elapsed_sec=.5,**{k:options[k] for k in keep})
    return support,search


def test_recorded_opposite_support_requires_retained_center_and_current_scans(recorded):
    _,options,snapshot,tf,proof,orientation,row,axis=recorded
    assert support_for(recorded,proof=False)[0] is None
    tracker=StoppedTargetReconciliation()
    assert tracker.observe(**row) is None
    assert 'original registration bounds' in tracker.metadata['reason']
    support,search=support_for(recorded)
    assert support is not None
    assert math.dist(row['stand_center'],validate_reconciliation(proof)[2]) > .08
    assert support.lidar_association.max_camera_map_bearing_delta_rad == math.radians(3)
    attempt,crop=exclusive_identity_crop(candidate_uid=row['candidate_uid'],snapshot=snapshot,
        support=support,search_result=search,**options)
    assert attempt is not None and crop['sampling']=='isolated_current_qr_quad'
    assert any(c['occluded_by_target_symbol'] for c in crop['competitors'])
    assert all(attempt.roi.x0<=x<attempt.roi.x1 and attempt.roi.y0<=y<attempt.roi.y1 for x,y in support.corners_px)
    assert bind_crop_text((DecodedQrObservation('Start',None,'payload_only'),),crop).accepted
    source=load_backside_axis_observation(axis)
    assert .02 <= source.validated_target_center['uncertainty_m'] < .03
    assert orientation['bounded_orientation']['half_width_rad']==source.bounded_orientation['half_width_rad']


def test_recorded_calibrated_turn_exceeds_budget_explicitly(recorded):
    _,options,_,tf,proof,_,row,_=recorded
    support,_=support_for(recorded);assert support is not None
    diagnostics={}
    advisory=build_camera_centering_advisory(association=support,
        intrinsics=options['intrinsics'],scan_from_camera=tf('base_scan','camera'),
        base_from_camera=tf('base_footprint','camera'),candidate_uid=row['candidate_uid'],
        target_key=row['target_key'],stream_id='recorded',planning_frame='map',motion_epoch=0,
        anchor_pose=EvidencePose(*row['robot_pose']),anchor_odom_pose=EvidencePose(0.,0.,0.),
        odom_stamp_sec=row['image_stamp_sec'],image_stamp_sec=row['image_stamp_sec'],now_sec=row['now_sec'],
        robot_profile_sha256='a'*64,calibration_profile_sha256='b'*64,
        stand_model_profile_sha256=options['model_profile'].sha256,diagnostics=diagnostics)
    assert advisory is None
    assert diagnostics['reason']=='centering_budget_exceeded'
    assert diagnostics['recovery']=='bounded_inspection_view'
    assert diagnostics['motion_authorized'] is False


def test_current_proof_and_retained_source_cannot_be_rebound_or_edited(recorded):
    _,_,_,_,proof,orientation,_,axis=recorded
    support,_=support_for(recorded);value=support.metadata()
    bad=deepcopy(value);bad['center_px']=(400.,300.)
    with pytest.raises(ValueError):validate_target_support(bad)
    bad=deepcopy(proof);bad['retained_orientation']['validated_target_center']['y_m']+=.05
    with pytest.raises(ValueError):validate_reconciliation(bad)
    bad=deepcopy(proof);bad['entries'][0]=bad['entries'][1]
    with pytest.raises(ValueError):validate_reconciliation(bad)
    # A warm immutable-chain cache must still notice source mutation.
    orientation_record(orientation['path'])
    raw=axis.read_text();axis.write_text(raw.replace('0.13884297539521773','0.01'))
    with pytest.raises(ValueError):orientation_record(orientation['path'])


def test_producer_uses_reconciliation_decodes_and_does_not_fit_angle(recorded):
    from scripts.aufgabe04.real_robot.observer.opposite_identity import process_opposite_identity
    from scripts.aufgabe04.navigation.foundation.models import Pose2D
    image,options,snapshot,tf,proof,orientation,row,_=recorded
    staged=[];states=[]
    adapter=SimpleNamespace(args=SimpleNamespace(stand_id=row['candidate_uid'],sync_tolerance_sec=.1,
        lidar_cone_half_angle_deg=3.,backside_registration_max_bearing_delta_deg=12.,max_sensor_age_sec=.5,
        candidate_centering_json=None),cv2=cv2,stand_model_profile=options['model_profile'],
        node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=int(row['now_sec']*1e9)))),
        _target_evidence_key=lambda:row['target_key'],_capture_pending={},
        _write_status=lambda state,**details:states.append((state,details)))
    def record(**fields):
        staged.append(fields)
        return SimpleNamespace(snapshot=SimpleNamespace(as_dict=lambda:{}))
    adapter._record_observation_frame=record
    # Exercise real scan reconciliation, complete-outline support and crop
    # exclusion; decoder output is independently checked below on saved pixels.
    module='scripts.aufgabe04.real_robot.observer.opposite_identity.'
    with patch(module+'detect_qr_observations_bgr',return_value=(DecodedQrObservation('Start',None,'recorded'),)) as decoder:
        process_opposite_identity(adapter,context=SimpleNamespace(orientation=orientation,snapshot=snapshot),
            frame=image,intrinsics=options['intrinsics'],robot_pose=Pose2D(*row['robot_pose']),
            camera_signature=(1,2,3,4),image_stamp_sec=row['image_stamp_sec'],scan=options['scan'],
            scan_from_map=tf('base_scan','map'),camera_from_map=tf('camera','map'),
            map_bearing_rad=options['map_bearing_rad'],accepted_range_m=options['accepted_range_m'],
            scan_from_camera=tf('base_scan','camera'),base_from_camera=tf('base_footprint','camera'),
            image_stamp=None,target_reconciliation=proof)
        decoder.assert_called_once()
    assert staged[-1]['qr_texts']==('Start',)
    assert staged[-1]['axis_yaw_rad'] is None and staged[-1]['lidar_associated']
    assert adapter._pending_qr_observation_pose is not None
    assert states[-1][1]['stand_axis_debug']['current_angle_refit'] is False

    from scripts.aufgabe04.real_robot.observer.qr_observation_pose import QrObservationPoseFallback
    from scripts.aufgabe04.artifacts.qr_verified_observation_pose import (
        build_qr_verified_observation_pose, validate_qr_verified_observation_pose, SOURCE_GATES)
    current=adapter._pending_qr_observation_pose
    update=SimpleNamespace(snapshot=SimpleNamespace(target_key=row['target_key'],motion_epoch=0,
        poisoned=False,tentative_qr_id='Start',latched_qr_id=None),motion_epoch_reset=False,
        frame_accepted=True,qr_sample_accepted=True)
    assert QrObservationPoseFallback(delay_sec=1.5).observe(current,update=update,
        observed_at_sec=row['now_sec'],now_monotonic_sec=1.) is not None
    payload=build_qr_verified_observation_pose(candidate_uid=row['candidate_uid'],stream_id='recorded',
        planning_frame='map',qr_id='Start',stand_center=dict(zip(('x_m','y_m'),row['stand_center'])),
        robot_pose=dict(zip(('x_m','y_m','yaw_rad'),row['robot_pose'])),
        sensor_stamp_sec=row['image_stamp_sec'],scan_stamp_sec=row['scan'].scan_stamp_sec,
        checked_at_sec=row['now_sec'],robot_profile_sha256=orientation['robot_profile_sha256'],
        calibration_profile_sha256=orientation['calibration_profile_sha256'],
        stand_model_profile_sha256=options['model_profile'].sha256,
        target_key=row['target_key'],motion_epoch=0,camera_signature=current.camera_signature,
        qr_corners_px=None,image_shape=current.image_shape,qr_binding=current.qr_binding.metadata(),
        source_gates={key:True for key in SOURCE_GATES},retained_backside_orientation=orientation,
        localization_provenance=dict(map_frame='map',base_frame='base_footprint',scan_frame='base_scan',
            camera_frame='camera',exact_image_transform_stamp_sec=row['image_stamp_sec'],
            exact_scan_transform_stamp_sec=row['scan'].scan_stamp_sec))
    assert validate_qr_verified_observation_pose(payload)['qr_id']=='Start'


def test_recorded_pixels_contain_decodable_start(recorded):
    # Payload availability is independent of the producer's admission gates.
    # Production's isolated-quad decoder is covered with its deployed backend
    # by the earlier overlap regression; this fixture uses native OpenCV here.
    image,options,*_=recorded
    attempt,_=current_scan_qr_search(**options);roi=attempt.roi
    pixels=cv2.resize(image[roi.y0:roi.y1,roi.x0:roi.x1],None,fx=4,fy=4)
    text,_,_=cv2.QRCodeDetector().detectAndDecode(pixels)
    if not text and cv2.__version__.startswith('4.') and not hasattr(cv2,'wechat_qrcode_WeChatQRCode'):
        pytest.skip('native OpenCV build lacks payload decoder')
    if not text and hasattr(cv2,'wechat_qrcode_WeChatQRCode'):
        decoded=cv2.wechat_qrcode_WeChatQRCode().detectAndDecode(pixels)[0]
        text=decoded[0] if decoded else ''
    assert text=='Start'


def test_recorded_position_and_angle_plan_and_seal_together(recorded,tmp_path):
    from scripts.aufgabe04.navigation.approach.candidate_preapproach_materialization import plan_candidate_preapproach
    from scripts.aufgabe04.navigation.foundation.models import Pose2D
    from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot
    from tests.aufgabe04.opposite_reconciliation_fixture import ROOT
    from tests.aufgabe04.test_candidate_preapproach_planning import CandidatePreapproachPlanningTest
    import json
    source=load_backside_axis_observation(recorded[-1])
    snapshot=load_candidate_snapshot(ROOT/'backside_snapshot.json')
    repo=Path(__file__).resolve().parents[2]
    clearance=dict(minimum_active_standoff_m=.33,minimum_candidate_transit_radius_m=.34,minimum_static_inflation_m=.25)
    outputs=plan_candidate_preapproach(map_yaml=repo/'maps/aufgabe03/arena_1p898x3p9_auto.yaml',
        semantic_map_id='arena_1p898x3p9_auto',plan=CandidatePreapproachPlanningTest._plan(snapshot.map_bundle_sha256),
        snapshot=snapshot,snapshot_path=ROOT/'backside_snapshot.json',candidate_uid=source.stand_id,
        start=Pose2D(source.robot_x_m,source.robot_y_m,source.target_reconciliation['entries'][-1]['robot_pose'][2]),
        output_dir=tmp_path/'opposite_plan',approach_offset_m=.5,inflation_radius_m=.25,
        candidate_transit_radius_m=.34,physical_clearance=clearance,
        approach_normal_rad=source.opposite_face_normal_rad,axis_observation_path=recorded[-1])
    metadata=json.loads(Path(outputs['diagnostics_json']).read_text())['metadata']
    center=source.validated_target_center;goal=metadata['selected_approach_pose']
    assert metadata['validated_target_center']==center
    assert math.dist((center['x_m'],center['y_m']),(source.stand_x_m,source.stand_y_m))>.08
    assert abs(math.remainder(goal['yaw_rad']-math.atan2(center['y_m']-goal['y_m'],center['x_m']-goal['x_m']),math.tau))<1e-9
    assert metadata['bounded_orientation_view']['stand_center_uncertainty_m']==center['uncertainty_m']
    assert Path(outputs['route_certificate_json']).is_file()


def test_arrival_keeps_validated_center_instead_of_turning_back_to_survey(recorded,tmp_path):
    from scripts.aufgabe04.real_robot.candidate.approach import _admit_camera_arrival_geometry
    from scripts.aufgabe04.navigation.foundation.models import Pose2D
    from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, payload_sha256
    import json
    _,_,snapshot,_,_,orientation,row,_=recorded
    center=orientation['validated_target_center']
    robot=Pose2D(row['robot_pose'][0],row['robot_pose'][1],
        math.atan2(center['y_m']-row['robot_pose'][1],center['x_m']-row['robot_pose'][0]))
    assert abs(robot.yaw_rad-row['robot_pose'][2])>math.radians(6)
    target=Path(orientation['path']).with_name('target.json')
    digest=payload_sha256(load_content_hashed_json(target,hash_field='candidate_frame_projection_sha256'))
    config=SimpleNamespace(physical_clearance={'minimum_active_standoff_m':.33},approach_offset_m=.5,
        camera_arrival_range_slack_m=.1,camera_arrival_max_bearing_error_rad=math.radians(3))
    artifacts=SimpleNamespace(config=SimpleNamespace(snapshot=snapshot),evidence_path=target,evidence_sha256=digest,
        snapshot_path=Path(row['snapshot_path']),snapshot_sha256=recorded[4]['snapshot_sha256'],
        camera_decision_binding=lambda:SimpleNamespace(projection_path=target,projection_sha256=digest))
    effects=SimpleNamespace(admit_planning_frame=object(),run_centering_turn=object())
    module='scripts.aufgabe04.real_robot.candidate.approach.'
    with patch(module+'_run_planning_frame_admission',return_value=SimpleNamespace(current_pose=robot)), \
         patch(module+'_materialize_candidate_frame_projection',return_value=artifacts):
        arrived=_admit_camera_arrival_geometry(source_config=config,effects=effects,source_registry=object(),
            candidate_uid=row['candidate_uid'],candidate_root=tmp_path/'arrival',observation_attempt_index=1,
            allow_centering_acquisition=True,retained_backside_axis_path=Path(orientation['path']))
    assert arrived.candidate==snapshot.candidate_for(row['candidate_uid'])
    assert orientation_record(arrived.retained_backside_axis_path)['validated_target_center']==center
    evidence=json.loads((tmp_path/'arrival/camera_attempt_01_arrival/admission.json').read_text())
    assert evidence['accepted'] and evidence['validated_target_center']==center
    assert evidence['camera_centered'] is False


def test_metric_position_tamper_cannot_borrow_valid_backside_angle(recorded,tmp_path):
    import json
    source=json.loads(recorded[-1].read_text())
    source['head_position_evidence']['head_bounds']['hypotheses'][0]['translation_xyz_m'][0]+=.03
    forged=tmp_path/'forged.json';forged.write_text(json.dumps(source))
    with pytest.raises(ValueError,match='validated metric head geometry'):
        load_backside_axis_observation(forged)


@pytest.mark.parametrize('outline_available',[True,False])
def test_identity_overlap_does_not_block_target_recovery(recorded,outline_available,tmp_path):
    from scripts.aufgabe04.real_robot.observer.opposite_identity import process_opposite_identity
    from scripts.aufgabe04.navigation.foundation.models import Pose2D
    image,options,snapshot,tf,proof,orientation,row,_=recorded
    support,search=support_for(recorded)
    frames=[]
    adapter=SimpleNamespace(args=SimpleNamespace(stand_id=row['candidate_uid'],sync_tolerance_sec=.1,
        lidar_cone_half_angle_deg=3.,backside_registration_max_bearing_delta_deg=12.,max_sensor_age_sec=.5,
        candidate_centering_json=tmp_path/'centering.json'),cv2=cv2,stand_model_profile=options['model_profile'],
        profile=SimpleNamespace(odom_frame='odom',base_frame='base_footprint'),TransformException=RuntimeError,
        _lookup=lambda *args:None,
        node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=int(row['now_sec']*1e9)))),
        _target_evidence_key=lambda:row['target_key'],_write_status=lambda *args,**kwargs:None,
        _record_observation_frame=lambda **kwargs:frames.append(kwargs) or SimpleNamespace(snapshot=SimpleNamespace(as_dict=lambda:{})))
    module='scripts.aufgabe04.real_robot.observer.opposite_identity.'
    with patch(module+'detect_opposite_target_support',return_value=support if outline_available else None), \
         patch(module+'exclusive_identity_crop',return_value=(None,dict(accepted=False,reason='target_crop_overlap_unresolved'))), \
         patch(module+'pose2d_from_transform',return_value=Pose2D(0.,0.,0.)), \
         patch(module+'detect_qr_observations_bgr') as decoder:
        process_opposite_identity(adapter,context=SimpleNamespace(orientation=orientation,snapshot=snapshot),
            frame=image,intrinsics=options['intrinsics'],robot_pose=Pose2D(*row['robot_pose']),
            camera_signature=(1,2,3,4),image_stamp_sec=row['image_stamp_sec'],scan=options['scan'],
            scan_from_map=tf('base_scan','map'),camera_from_map=tf('camera','map'),
            map_bearing_rad=options['map_bearing_rad'],accepted_range_m=options['accepted_range_m'],
            scan_from_camera=tf('base_scan','camera'),base_from_camera=tf('base_footprint','camera'),
            image_stamp=None,target_reconciliation=proof)
        decoder.assert_not_called()
    assert frames[-1]['lidar_associated'] and not frames[-1]['qr_texts']
    if outline_available:
        assert adapter._pending_candidate_centering.association is support
    else:
        assert getattr(adapter,'_pending_candidate_centering',None) is None
