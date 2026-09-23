"""Recorded off-center QR and bounded candidate reconciliation regressions."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import math
import unittest

from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import intrinsics_from_camera_info, ImageRoi
from scripts.aufgabe04.real_robot.observer.target_reconciliation import StoppedTargetReconciliation, validate_reconciliation
from scripts.aufgabe04.real_robot.observer.qr_target_binding import bind_qr_observations_to_target
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot, write_candidate_snapshot

ROOT = Path(__file__).parent/'fixtures/target_reconciliation_20260923'


def recorded_inputs(data=None, *, root=ROOT, candidate_uid="survey_candidate_0005"):
    snapshot_path = root/"candidate_snapshot.json"
    data = data or json.loads((ROOT/'inputs.json').read_text())
    snapshot = load_candidate_snapshot(snapshot_path)
    g = snapshot.candidate_for(candidate_uid).geometry
    rows = []
    for f in data['frames']:
        s = f['sensors']['scan']
        scan = PlainLaserScan(ranges=tuple(v if type(v) in (int,float) else math.nan for v in s['ranges']),
            angle_min=s['angle_min'],angle_increment=s['angle_increment'],angle_max=s['angle_max'],
            range_min=s['range_min'],range_max=s['range_max'],scan_frame_id=s['header']['frame_id'],
            scan_stamp_sec=f['scan_stamp_sec'],receipt_sec=f['scan_received_ros_sec'],scan_topology_profile='full_rotation')
        def tf(parent,child):
            t = next(t for t in f['tf_samples'] if t['target_frame']==parent and t['source_frame']==child)
            return RigidTransform(parent,child,tuple(t['translation_xyz_m']),tuple(t['rotation_xyzw']))
        base = tf('map','base_footprint');q=base.rotation_xyzw
        pose = (*base.translation_xyz_m[:2], math.atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2)))
        e=f['search']['envelope']
        options=dict(map_bearing_rad=e['map_bearing_rad'],cone_half_angle_rad=math.radians(3),
            max_camera_map_bearing_delta_rad=math.radians(12),accepted_range_m=tuple(e['accepted_range_m']))
        ci=dict(f['sensors']['camera_info']);ci['header']=SimpleNamespace(**ci['header'])
        rows.append(dict(snapshot_path=snapshot_path,candidate_uid=candidate_uid,
            planning_frame='map',stand_center=(g.x_m,g.y_m),target_key=f"fixture/{candidate_uid}",epoch=0,
            scan=scan,scan_from_map=tf('base_scan','map'),robot_pose=pose,image_stamp_sec=f['image_stamp_sec'],
            now_sec=max(f['image_stamp_sec'],scan.scan_stamp_sec,scan.receipt_sec)+.1,options=options))
    return data, rows, intrinsics_from_camera_info(SimpleNamespace(**ci)), tf('base_scan','camera')


class TargetReconciliationTest(unittest.TestCase):
    def setUp(self):
        self.data,self.rows,self.intrinsics,self.camera = recorded_inputs()

    def proof(self):
        tracker=StoppedTargetReconciliation()
        self.assertIsNone(tracker.observe(**self.rows[0]))
        self.assertIsNone(tracker.observe(**self.rows[1]))
        result=tracker.observe(**self.rows[2])
        self.assertIsNotNone(result,tracker.metadata)
        return result

    def bind(self,proof=None,**overrides):
        r=self.rows[-1]
        options=dict(roi=ImageRoi(0,0,800,600,100),intrinsics=self.intrinsics,
            scan_from_camera=self.camera,scan=r['scan'],now_sec=r['now_sec'],max_scan_age_sec=.5,
            min_cluster_sample_count=1,camera_registration_accepted=False,allow_independent_registration=True,
            target_reconciliation=proof,**r['options'])
        options.update(overrides)
        return bind_qr_observations_to_target((DecodedQrObservation('QR_004',tuple(map(tuple,self.data['qr_corners_px'])),'recorded'),),**options)

    def test_recorded_qr_requires_validated_reconciliation_after_parallax_correction(self):
        raw=self.bind()
        self.assertFalse(raw.accepted)
        self.assertEqual(raw.reason,'camera_map_bearing_interval_exceeds_limit')
        proof=self.proof()
        result=self.bind(proof)
        self.assertTrue(result.accepted,result.reason)
        self.assertEqual(result.qr_texts_for_evidence,('QR_004',))
        self.assertLess(abs(result.camera_bearing_rad),math.radians(13))
        self.assertFalse(proof['candidate_geometry_updated'])
        self.assertTrue(set(result.association['search_association']['selected_cluster_source_indices']).issubset(
            result.independent_registration['envelope']['selected_cluster_source_indices']))

    def test_processing_time_advance_preserves_same_current_cluster(self):
        proof = self.proof()
        result = self.bind(proof,now_sec=self.rows[-1]['now_sec']+.15)
        self.assertTrue(result.accepted,result.reason)

    def test_reused_tuple_motion_expiry_and_changed_context_cannot_reconcile(self):
        tracker=StoppedTargetReconciliation()
        tracker.observe(**self.rows[0]);tracker.observe(**self.rows[1])
        self.assertIsNone(tracker.observe(**self.rows[1]))
        for change in ({'epoch':1},{'target_key':'other'}, {'robot_pose':(2.,2.,1.)},
                       {'now_sec':self.rows[-1]['now_sec']+1}):
            tracker=StoppedTargetReconciliation()
            tracker.observe(**self.rows[0]);tracker.observe(**self.rows[1])
            self.assertIsNone(tracker.observe(**{**self.rows[2],**change}))

    def test_receipt_rejects_forged_history_and_rebinding(self):
        proof=self.proof()
        for mutate in (lambda p:p['entries'].__setitem__(0,p['entries'][1]),
                       lambda p:p['entries'][0]['robot_pose'].__setitem__(0,3.),
                       lambda p:p.__setitem__('snapshot_sha256','0'*64),
                       lambda p:p['entries'][0]['options'].__setitem__('max_camera_map_bearing_delta_rad',math.radians(20))):
            bad=copy.deepcopy(proof);mutate(bad)
            with self.assertRaises(ValueError):validate_reconciliation(bad)
        with self.assertRaises(ValueError):validate_reconciliation(proof,candidate_uid='survey_candidate_0004')
        self.assertFalse(self.bind(proof,now_sec=self.rows[-1]['now_sec']+1).accepted)

    def test_competing_candidate_prevents_reconciliation(self):
        import tempfile
        proof=self.proof();_,_,world,_=validate_reconciliation(proof)
        snapshot=load_candidate_snapshot(ROOT/'candidate_snapshot.json')
        neighbor=snapshot.candidate_for('survey_candidate_0004')
        neighbor=replace(neighbor,geometry=replace(neighbor.geometry,x_m=world[0],y_m=world[1]))
        snapshot=replace(snapshot,candidates=tuple(neighbor if c.candidate_uid==neighbor.candidate_uid else c for c in snapshot.candidates))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'snapshot.json';write_candidate_snapshot(path,snapshot)
            tracker=StoppedTargetReconciliation()
            for row in self.rows:self.assertIsNone(tracker.observe(**{**row,'snapshot_path':path}))
            self.assertIn('another candidate',tracker.metadata['reason'])

if __name__=='__main__':unittest.main()


def test_reconciled_qr_receipt_round_trip_and_tamper_rejection():
    from dataclasses import asdict
    from scripts.aufgabe04.artifacts.qr_verified_observation_pose import build_qr_verified_observation_pose, validate_qr_verified_observation_pose, SOURCE_GATES, HASH_FIELD
    from scripts.aufgabe04.artifacts.content_store import content_hashed_payload
    case=TargetReconciliationTest();case.setUp();proof=case.proof();binding=case.bind(proof);r=case.rows[-1]
    fields=dict(candidate_uid=r['candidate_uid'],stream_id='fixture',planning_frame='map',qr_id='QR_004',
        stand_center=dict(zip(('x_m','y_m'),r['stand_center'])),robot_pose=dict(zip(('x_m','y_m','yaw_rad'),r['robot_pose'])),
        sensor_stamp_sec=r['image_stamp_sec'],scan_stamp_sec=r['scan'].scan_stamp_sec,checked_at_sec=r['now_sec'],
        robot_profile_sha256='a'*64,calibration_profile_sha256='b'*64,stand_model_profile_sha256='c'*64,
        target_key=r['target_key'],motion_epoch=0,camera_signature=(case.intrinsics.fx_px,case.intrinsics.fy_px,case.intrinsics.cx_px,case.intrinsics.cy_px),
        qr_corners_px=case.data['qr_corners_px'],image_shape=(600,800),qr_binding=binding.metadata(),
        source_gates={k:True for k in SOURCE_GATES},localization_provenance=dict(map_frame='map',base_frame='base_footprint',scan_frame='base_scan',camera_frame='camera',exact_image_transform_stamp_sec=r['image_stamp_sec'],exact_scan_transform_stamp_sec=r['scan'].scan_stamp_sec))
    receipt=build_qr_verified_observation_pose(**fields)
    assert validate_qr_verified_observation_pose(json.loads(json.dumps(receipt)))['qr_id']=='QR_004'
    for mutate in (lambda d:d['qr_binding']['finite_bearing'].__setitem__('range_m',1.),
                   lambda d:d.__setitem__('motion_epoch',1),
                   lambda d:d['qr_corners_px'][0].__setitem__(0,400.)):
        bad=copy.deepcopy(receipt);mutate(bad);bad.pop(HASH_FIELD)
        try:validate_qr_verified_observation_pose(content_hashed_payload(bad,hash_field=HASH_FIELD))
        except ValueError:pass
        else:raise AssertionError('altered proof was admitted')


def test_reconciled_current_head_can_produce_bounded_centering_without_new_axis():
    from scripts.aufgabe04.real_robot.observer.candidate_centering import build_camera_centering_advisory, validate_camera_centering_advisory
    from scripts.aufgabe04.real_robot.observer.finite_target_bearing import finite_target_bearing
    from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose
    from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
    case=TargetReconciliationTest();case.setUp();proof=case.proof();r=case.rows[-1]
    _,env,_,reference=validate_reconciliation(proof)
    center=(260.,300.)
    bearing,_,_=finite_target_bearing(center_px=center,intrinsics=case.intrinsics,scan_from_camera=case.camera,
        distance_m=env.distance_m,range_interval_m=env.accepted_range_m)
    lidar=associate_camera_registered_candidate_lidar_target(r['scan'],map_bearing_rad=reference,
        observed_camera_bearing_rad=bearing,cone_half_angle_rad=math.radians(3),accepted_range_m=env.accepted_range_m,
        now_sec=r['now_sec'],max_scan_age_sec=.5,max_camera_map_bearing_delta_rad=math.radians(3))
    association=SimpleNamespace(accepted=True,lidar_association=lidar,head_admission=SimpleNamespace(accepted=False),
        head_orientation_bounds=object(),full_image_center_px=center,target_reconciliation=proof)
    base=RigidTransform('base_footprint','camera',(.04553724525972719,-.004971307208605384,.12574663531904093),case.camera.rotation_xyzw)
    result=build_camera_centering_advisory(association=association,intrinsics=case.intrinsics,scan_from_camera=case.camera,
        base_from_camera=base,candidate_uid=r['candidate_uid'],target_key=r['target_key'],stream_id='fixture',planning_frame='map',motion_epoch=0,
        anchor_pose=EvidencePose(*r['robot_pose']),anchor_odom_pose=EvidencePose(0,0,0),odom_stamp_sec=r['image_stamp_sec'],
        image_stamp_sec=r['image_stamp_sec'],now_sec=r['now_sec'],robot_profile_sha256='a'*64,calibration_profile_sha256='b'*64,stand_model_profile_sha256='c'*64)
    assert result is not None
    assert 0 < result.requested_yaw_rad <= math.radians(6)
    validate_camera_centering_advisory(json.loads(json.dumps(result.metadata())))


def test_reconciled_identity_completes_immediately_without_axis_consensus():
    from scripts.aufgabe04.real_robot.observer.qr_observation_pose import prepare_qr_observation_pose, QrObservationPoseFallback
    from scripts.aufgabe04.real_robot.observer.evidence import PassiveObserverEvidence, EvidencePose
    case=TargetReconciliationTest();case.setUp();proof=case.proof();binding=case.bind(proof);r=case.rows[-1]
    pose=EvidencePose(*r['robot_pose'])
    evidence=PassiveObserverEvidence(target_key=r['target_key'],anchor_pose=pose,required_axis_samples=7,max_axis_deviation_rad=.1)
    update=evidence.record_frame(target_key=r['target_key'],pose=pose,frame_stamp_sec=r['image_stamp_sec'],lidar_stamp_sec=r['scan'].scan_stamp_sec,
        observed_at_sec=r['now_sec'],lidar_associated=True,qr_texts=('QR_004',),qr_symbol_count=1,axis_yaw_rad=None,axis_source=None)
    frame=prepare_qr_observation_pose(qr_binding=binding,qr_observations=(DecodedQrObservation('QR_004',tuple(map(tuple,case.data['qr_corners_px'])),'recorded'),),
        observed_qr_texts=('QR_004',),image_stamp_sec=r['image_stamp_sec'],scan_stamp_sec=r['scan'].scan_stamp_sec,robot_pose=pose,target_key=r['target_key'],
        camera_signature=(640,640,400,300),image_shape=(600,800),roi=ImageRoi(0,0,800,600,100),model_profile_sha256='c'*64,metadata={})
    ready=QrObservationPoseFallback(delay_sec=1.5).observe(frame,update=update,observed_at_sec=r['now_sec'],now_monotonic_sec=10.)
    assert ready is not None
    assert update.snapshot.current_axis_sample_count==0
    assert frame.metadata['qr_observation_pose_fallback']['delay_sec']==0.
