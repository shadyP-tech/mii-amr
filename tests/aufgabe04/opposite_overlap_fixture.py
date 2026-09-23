"""Recorded sensor inputs only; no hand-authored target corners."""
from pathlib import Path
import json, math
from types import SimpleNamespace
import numpy as np
import cv2
from scripts.aufgabe04.perception.camera_calibration import rectify_bgr_frame
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.real_robot.configuration.geometry import intrinsics_from_camera_info


def recorded_options():
    repo = Path(__file__).resolve().parents[2]
    root = Path(__file__).parent/'fixtures/opposite_identity_background_overlap'
    data = json.loads((root/'input.json').read_text())
    ci = dict(data['sensors']['camera_info']); ci['header'] = SimpleNamespace(**ci['header'])
    info = SimpleNamespace(**ci)
    frame = rectify_bgr_frame(cv2.imread(str(root/'frame.jpg')), info, cv2, np)
    def transform(parent, child):
        tf = next(t for t in data['tf_samples'] if t['target_frame']==parent and t['source_frame']==child)
        return RigidTransform(parent, child, tuple(tf['translation_xyz_m']), tuple(tf['rotation_xyzw']))
    model = load_measured_physical_stand_model(repo/'configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json')
    s=data['sensors']['scan']
    scan=PlainLaserScan(ranges=tuple(v if isinstance(v,(int,float)) else math.nan for v in s['ranges']),
        angle_min=s['angle_min'],angle_increment=s['angle_increment'],angle_max=s['angle_max'],
        range_min=s['range_min'],range_max=s['range_max'],scan_frame_id=s['header']['frame_id'],
        scan_stamp_sec=data['scan_stamp_sec'],receipt_sec=data['scan_received_ros_sec'],scan_topology_profile='full_rotation')
    snapshot=SimpleNamespace(snapshot_id='recorded', candidates=tuple(SimpleNamespace(candidate_uid=c['candidate_uid'],
        geometry=SimpleNamespace(**c['geometry'])) for c in data['candidates']))
    snapshot.candidate_for=lambda uid:next((c for c in snapshot.candidates if c.candidate_uid==uid),None)
    options=dict(scan=scan,scan_from_map=transform('base_scan','map'),camera_from_map=transform('camera','map'),
        intrinsics=intrinsics_from_camera_info(info),model_profile=model,image_stamp_sec=data['image_stamp_sec'],
        sync_tolerance_sec=.1,map_bearing_rad=data['search']['envelope']['map_bearing_rad'],
        cone_half_angle_rad=math.radians(3),max_camera_map_bearing_delta_rad=math.radians(12),
        accepted_range_m=tuple(data['search']['envelope']['accepted_range_m']),
        now_sec=max(data['image_stamp_sec'],data['scan_stamp_sec'])+.1,max_scan_age_sec=.5)
    return frame, options, snapshot, transform, data
