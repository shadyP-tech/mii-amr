"""Original Oct 1 opposite tuples; only file references are rebased locally.

The fixture includes complete original parsed payloads and byte/canonical hashes.
Locally regenerated path-bearing projections are derivative test artifacts, not
the original files. No ranges, timestamps, poses or uncertainty are adjusted.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.artifacts.retained_backside_orientation import orientation_record
from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import write_backside_axis_frame_projection
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.camera_calibration import camera_calibration_from_info, rectify_bgr_frame
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import transform_point
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics
from scripts.aufgabe04.real_robot.configuration.profile import load_camera_calibration, camera_calibration_sha256
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import (
    ScanPersistenceContext, scan_pose_in_map, scan_pose_from_camera_extrinsics,
)
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / 'fixtures/opposite_endpoint_20261001.json'


def recorded_endpoint(tmp_path, capture_index=26):
    """Return authentic hint kwargs and tuple geometry, without granting support."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = json.loads(FIXTURE.read_text())
    artifacts = data['artifacts']
    for name, value in artifacts.items():
        canonical = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        assert hashlib.sha256(canonical.encode()).hexdigest() == data['provenance'][name]['canonical_payload_sha256']
    for name in ('canonical_snapshot', 'backside_snapshot', 'candidate_snapshot'):
        (tmp_path / f'{name}.json').write_text(json.dumps(artifacts[name]))
    model_path = ROOT / data['provenance']['stand_model']['repository_path']
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == data['provenance']['stand_model']['source_sha256']
    axis = copy.deepcopy(artifacts['backside_observation'])
    axis['target_reconciliation']['snapshot_path'] = str(tmp_path / 'backside_snapshot.json')
    axis['head_position_evidence']['model_path'] = str(model_path)
    (tmp_path / 'axis.json').write_text(json.dumps(axis))
    digests = []
    for label, name in [('source', 'backside_snapshot'), ('target', 'candidate_snapshot')]:
        projection = copy.deepcopy(artifacts[f'{label}_projection'])
        projection.pop('candidate_frame_projection_sha256')
        projection['source_candidate_snapshot_path'] = str(tmp_path / 'canonical_snapshot.json')
        projection['projected_candidate_snapshot_path'] = str(tmp_path / f'{name}.json')
        digests.append(write_content_hashed_json(tmp_path / f'{label}.json', projection,
            hash_field='candidate_frame_projection_sha256'))
    snapshot_path = tmp_path / 'candidate_snapshot.json'
    snapshot = load_candidate_snapshot(snapshot_path)
    uid = data['candidate_uid']
    g = snapshot.candidate_for(uid).geometry
    write_backside_axis_frame_projection(tmp_path / 'orientation.json', axis_evidence_path=tmp_path / 'axis.json',
        source_candidate_projection_path=tmp_path / 'source.json', source_candidate_projection_sha256=digests[0],
        target_candidate_projection_path=tmp_path / 'target.json', target_candidate_projection_sha256=digests[1],
        target_candidate_x_m=g.x_m, target_candidate_y_m=g.y_m)
    orientation = orientation_record(tmp_path / 'orientation.json')
    assert orientation['validated_target_center'] == data['expected_original']['retained_center']
    calibration_profile_path = tmp_path / 'sealed_camera_calibration.json'
    calibration_profile_path.write_text(json.dumps(artifacts['sealed_camera_calibration']))
    calibration_profile = load_camera_calibration(calibration_profile_path)
    assert camera_calibration_sha256(calibration_profile) == orientation['calibration_profile_sha256']
    frame = next(f for f in data['frames'] if f['capture_index'] == capture_index)
    canonical = json.dumps(frame, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == data['provenance'][f'frame_{capture_index:06d}']['canonical_payload_sha256']
    m = frame['metadata']

    def tf(parent, child):
        value = next(t for t in m['tf_samples'] if t['target_frame'] == parent and t['source_frame'] == child)
        return RigidTransform(parent, child, tuple(value['translation_xyz_m']), tuple(value['rotation_xyzw']))

    raw = m['sensors']['scan']
    envelope = m['detector_metadata']['identity_crop']['search']['envelope']
    scan = PlainLaserScan(tuple(float(v) for v in raw['ranges']), raw['angle_min'], raw['angle_increment'],
        raw['range_min'], raw['range_max'], raw['header']['frame_id'], m['scan_stamp_sec'],
        m['scan_received_ros_sec'], raw['angle_max'], envelope['scan_topology']['profile'])
    now = scan.receipt_sec + envelope['scan_age_sec']
    base, scan_from_map = tf('map', 'base_footprint'), tf('base_scan', 'map')
    q = base.rotation_xyzw
    robot = Pose2D(*base.translation_xyz_m[:2], math.atan2(2*q[3]*q[2], 1-2*q[2]**2))
    base_camera, scan_camera = tf('base_footprint', 'camera'), tf('base_scan', 'camera')
    camera_map = tf('camera', 'map')
    ci = dict(m['sensors']['camera_info'])
    ci['header'] = SimpleNamespace(**ci['header'])
    calibration = camera_calibration_from_info(SimpleNamespace(**ci))
    intrinsics = CameraIntrinsics(calibration.width_px, calibration.height_px,
        calibration.fx_px, calibration.fy_px, calibration.cx_px, calibration.cy_px)
    model = load_measured_physical_stand_model(model_path)
    observation = m['outcome']['observation_evidence']
    target_key, epoch = observation['target_key'], observation['motion_epoch']
    center = orientation['validated_target_center']
    point = transform_point((center['x_m'], center['y_m'], 0.), scan_from_map)
    tolerance = envelope['accepted_range_m'][1] - math.hypot(*point[:2]) - center['uncertainty_m']
    # Recover the actual .04m run setting from its recorded surface envelope.
    # Rounding removes only transform arithmetic noise (<1e-12), not evidence.
    assert abs(tolerance - .04) < 1e-12
    tolerance = .04
    context = ScanPersistenceContext(target_key=target_key, epoch_key=str(epoch), robot_pose=robot,
        scan_pose_map=scan_pose_in_map(scan_from_map.translation_xyz_m, scan_from_map.rotation_xyzw),
        image_stamp_sec=m['image_stamp_sec'], candidate_x_m=g.x_m, candidate_y_m=g.y_m,
        stand_radius_m=g.radius_m, stand_uncertainty_m=g.uncertainty_m, lidar_range_tolerance_m=tolerance,
        scan_pose_robot=scan_pose_from_camera_extrinsics(base_camera.translation_xyz_m, base_camera.rotation_xyzw,
            scan_camera.translation_xyz_m, scan_camera.rotation_xyzw), retained_orientation=orientation)
    options = dict(map_bearing_rad=envelope['map_bearing_rad'], cone_half_angle_rad=math.radians(3),
        max_camera_map_bearing_delta_rad=math.radians(12), accepted_range_m=tuple(envelope['accepted_range_m']))
    row = dict(snapshot_path=snapshot_path, candidate_uid=uid, planning_frame='map',
        stand_center=(g.x_m, g.y_m), target_key=target_key, epoch=epoch, scan=scan, scan_from_map=scan_from_map,
        robot_pose=(robot.x_m, robot.y_m, robot.yaw_rad), image_stamp_sec=m['image_stamp_sec'], now_sec=now, options=options,
        retained_orientation=orientation, use_retained_target=True)
    hint_kwargs = dict(scan=scan, persistence_context=context, snapshot_path=snapshot_path,
        intrinsics=intrinsics, scan_from_camera=scan_camera, camera_from_map=camera_map,
        scan_from_map=scan_from_map, model_profile=model, model_path=model_path, now_sec=now,
        max_scan_age_sec=.5, cone_half_angle_rad=math.radians(3),
        max_camera_map_bearing_delta_rad=math.radians(12),
        map_bearing_rad=options['map_bearing_rad'], accepted_range_m=options['accepted_range_m'])
    outline = copy.deepcopy(data.get('derived_rectified_outlines', {}).get(str(capture_index)))
    image_path = FIXTURE.with_suffix('.jpg') if capture_index == 26 else None
    return dict(data=data, frame=frame, metadata=m, row=row, hint_kwargs=hint_kwargs, image_path=image_path,
        outline=outline, model=model, model_path=model_path, snapshot=snapshot, snapshot_path=snapshot_path,
        orientation=orientation, scan=scan, intrinsics=intrinsics, calibration=calibration,
        calibration_profile=calibration_profile, calibration_profile_path=calibration_profile_path,
        scan_from_camera=scan_camera, camera_from_map=camera_map, scan_from_map=scan_from_map,
        robot_pose=robot, persistence_context=context, now_sec=now)


def rectified_endpoint_image(value, image_path):
    """Decode an original byte-verified image and apply recorded CameraInfo."""
    import cv2
    import numpy as np
    raw = Path(image_path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == value['frame']['image_sha256']
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    return rectify_bgr_frame(image, value['calibration'], cv2, np)
