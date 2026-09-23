"""Recorded 11:30 run: three opposite tuples and its preceding backside proof."""
from pathlib import Path
from types import SimpleNamespace
import json
import cv2
import numpy as np

from scripts.aufgabe04.artifacts.retained_backside_orientation import orientation_record
from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import write_backside_axis_frame_projection
from scripts.aufgabe04.perception.camera_calibration import rectify_bgr_frame
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.real_robot.observer.target_reconciliation import StoppedTargetReconciliation
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot
from tests.aufgabe04.test_target_reconciliation import recorded_inputs

ROOT = Path(__file__).parent/'fixtures/opposite_reconciliation_20260923'


def recorded_opposite(root):
    data, rows, intrinsics, camera = recorded_inputs(json.loads((ROOT/'inputs.json').read_text()),
        root=ROOT,candidate_uid='survey_candidate_0001')
    raw = json.loads((ROOT/'backside_observation.json').read_text())
    raw['target_reconciliation']['snapshot_path'] = str((ROOT/'backside_snapshot.json').resolve())
    raw['head_position_evidence']['model_path']=str((Path(__file__).resolve().parents[2]/raw['head_position_evidence']['model_path']).resolve())
    axis = root/'axis.json';axis.write_text(json.dumps(raw))
    from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
    source = root/'source.json';target = root/'target.json'
    uid=raw['stand_id']
    snapshot = load_candidate_snapshot(ROOT/'candidate_snapshot.json');g=snapshot.candidate_for(uid).geometry
    digests=[]
    for label,path,snapshot_name in [('source',source,'backside_snapshot.json'),('target',target,'candidate_snapshot.json')]:
        evidence=json.loads((ROOT/(label+'_projection.json')).read_text())
        hash_key='candidate_frame_projection_sha256'
        evidence.pop(hash_key)
        evidence['source_candidate_snapshot_path']=str((ROOT/'canonical_snapshot.json').resolve())
        evidence['projected_candidate_snapshot_path']=str((ROOT/snapshot_name).resolve())
        digests.append(write_content_hashed_json(path,evidence,hash_field=hash_key))
    sha,target_sha=digests
    projection=root/'orientation.json'
    write_backside_axis_frame_projection(projection,axis_evidence_path=axis,
        source_candidate_projection_path=source,source_candidate_projection_sha256=sha,
        target_candidate_projection_path=target,target_candidate_projection_sha256=target_sha,
        target_candidate_x_m=g.x_m,target_candidate_y_m=g.y_m)
    orientation=orientation_record(projection)
    tracker=StoppedTargetReconciliation();proof=None
    for row in rows:
        proof=tracker.observe(**row,retained_orientation=orientation)
    assert proof is not None,tracker.metadata
    f=data['frames'][-1]
    def tf(parent,child):
        t=next(t for t in f['tf_samples'] if t['target_frame']==parent and t['source_frame']==child)
        return RigidTransform(parent,child,tuple(t['translation_xyz_m']),tuple(t['rotation_xyzw']))
    ci=dict(f['sensors']['camera_info']);ci['header']=SimpleNamespace(**ci['header'])
    image=rectify_bgr_frame(cv2.imread(str(ROOT/'frame.jpg')),SimpleNamespace(**ci),cv2,np)
    model=load_measured_physical_stand_model(Path(__file__).resolve().parents[2]/'configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json')
    row=rows[-1]
    options=dict(scan=row['scan'],scan_from_map=row['scan_from_map'],camera_from_map=tf('camera','map'),
        intrinsics=intrinsics,model_profile=model,image_stamp_sec=row['image_stamp_sec'],
        sync_tolerance_sec=.1,now_sec=row['now_sec'],max_scan_age_sec=.5,**row['options'])
    return image,options,snapshot,tf,proof,orientation,row,axis
