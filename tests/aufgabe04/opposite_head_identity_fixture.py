"""Latest failed opposite run: original pixels, scans, TF, and retained angle.

Only path-bearing artifact references are rebased. Current reconciliation is
rebuilt from recorded raw scans through its production validator; no successful
head measurement, QR binding, or observation pose is supplied by the fixture.
"""
import copy
from dataclasses import asdict
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
from scripts.aufgabe04.real_robot.observer.target_reconciliation import StoppedTargetReconciliation, validate_reconciliation
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / 'fixtures/opposite_head_identity_20261001'


def _payload_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def recorded_head_identity(tmp_path, capture_index=31):
    """Rebuild certified orientation and three-scan proof from original sources."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = json.loads((FIXTURE / 'input.json').read_text())
    artifacts = data['artifacts']
    for name, value in artifacts.items():
        assert _payload_hash(value) == data['provenance'][name]['canonical_payload_sha256']
    for name in ('canonical_snapshot', 'backside_snapshot', 'candidate_snapshot'):
        (tmp_path / f'{name}.json').write_text(json.dumps(artifacts[name]))
    model_path = ROOT / data['provenance']['stand_model']['repository_path']
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == data['provenance']['stand_model']['source_sha256']
    axis = artifacts['backside_observation']
    assert not axis.get('target_reconciliation') and not axis.get('head_position_evidence')
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
    assert 'validated_target_center' not in orientation
    assert orientation['axis_sample_count'] == 7
    calibration_path = tmp_path / 'sealed_camera_calibration.json'
    calibration_path.write_text(json.dumps(artifacts['sealed_camera_calibration']))
    calibration_profile = load_camera_calibration(calibration_path)
    assert camera_calibration_sha256(calibration_profile) == orientation['calibration_profile_sha256']
    frames = {f['capture_index']: f for f in data['frames']}
    tracker = StoppedTargetReconciliation()
    for index in range(capture_index - 2, capture_index + 1):
        frame = frames[index]
        assert _payload_hash(frame) == data['provenance'][f'frame_{index:06d}']['canonical_payload_sha256']
        m = frame['metadata']

        def tf(parent, child):
            value = next(t for t in m['tf_samples'] if (t['target_frame'], t['source_frame']) == (parent, child))
            return RigidTransform(parent, child, tuple(value['translation_xyz_m']), tuple(value['rotation_xyzw']))

        raw = m['sensors']['scan']
        scan = PlainLaserScan(tuple(float(v) for v in raw['ranges']), raw['angle_min'], raw['angle_increment'],
            raw['range_min'], raw['range_max'], raw['header']['frame_id'], m['scan_stamp_sec'],
            m['scan_received_ros_sec'], raw['angle_max'], 'full_rotation')
        search = m['detector_metadata']['identity_crop']['search']
        if 'envelope' in search:
            now = scan.receipt_sec + search['envelope']['scan_age_sec']
        else:
            # Frame 29 expired later in image processing. Its reconciliation
            # happened at selection: map the recorded monotonic selection time
            # through this same frame's recorded ROS/monotonic outcome pair.
            now = m['selected_monotonic_sec'] + m['outcome_ros_sec'] - m['outcome_monotonic_sec']
        base, scan_map = tf('map', 'base_footprint'), tf('base_scan', 'map')
        q = base.rotation_xyzw
        robot = Pose2D(*base.translation_xyz_m[:2], math.atan2(2*q[3]*q[2], 1-2*q[2]**2))
        point = transform_point((g.x_m, g.y_m, 0.), scan_map)
        distance = math.hypot(*point[:2])
        options = dict(map_bearing_rad=math.atan2(point[1], point[0]), cone_half_angle_rad=math.radians(3),
            max_camera_map_bearing_delta_rad=math.radians(12),
            accepted_range_m=(distance - 2*g.radius_m - g.uncertainty_m - .04, distance + .04))
        observation = m['outcome']['observation_evidence']
        row = dict(snapshot_path=snapshot_path, candidate_uid=uid, planning_frame='map',
            stand_center=(g.x_m, g.y_m), target_key=observation['target_key'], epoch=observation['motion_epoch'],
            scan=scan, scan_from_map=scan_map, robot_pose=tuple(asdict(robot).values()),
            image_stamp_sec=m['image_stamp_sec'], now_sec=now, options=options,
            retained_orientation=orientation, position_epoch_path=tmp_path / 'target.json')
        proof = tracker.observe(**row)
    validate_reconciliation(proof, candidate_uid=uid,
        image_stamp_sec=m['image_stamp_sec'], scan_stamp_sec=scan.scan_stamp_sec)
    info = dict(m['sensors']['camera_info'])
    info['header'] = SimpleNamespace(**info['header'])
    calibration = camera_calibration_from_info(SimpleNamespace(**info))
    intrinsics = CameraIntrinsics(calibration.width_px, calibration.height_px,
        calibration.fx_px, calibration.fy_px, calibration.cx_px, calibration.cy_px)
    return dict(data=data, frame=frame, metadata=m, row=row, model_path=model_path,
        model=load_measured_physical_stand_model(model_path), orientation=orientation,
        snapshot=snapshot, snapshot_path=snapshot_path, calibration=calibration,
        calibration_profile=calibration_profile, intrinsics=intrinsics, robot_pose=robot,
        scan=scan, scan_from_map=scan_map, camera_from_map=tf('camera', 'map'),
        scan_from_camera=tf('base_scan', 'camera'), base_from_camera=tf('base_footprint', 'camera'),
        now_sec=now, reconciliation=proof, tracker=tracker,
        image_path=FIXTURE / f'frame_{capture_index:06d}.jpg')


def recorded_head_image(case):
    import cv2
    import numpy as np
    raw = case['image_path'].read_bytes()
    assert hashlib.sha256(raw).hexdigest() == case['frame']['image_sha256']
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    return rectify_bgr_frame(image, case['calibration'], cv2, np)
