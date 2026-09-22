"""Real third-view pixels: automated search, own corners, independent binding."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
cv2 = pytest.importorskip('cv2')
np = pytest.importorskip('numpy')
from scripts.aufgabe04.perception.camera_calibration import rectify_bgr_frame
from scripts.aufgabe04.perception.stand_axis_handoff.models import RigidTransform
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, ImageRoi
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search
from scripts.aufgabe04.real_robot.observer.qr_target_binding import bind_qr_observations_to_target

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name('fixtures') / 'qr_admission_20260922'


def recorded_context(row):
    ci = SimpleNamespace(**row['sensors']['camera_info'])
    ci.header = SimpleNamespace(**ci.header)
    raw = (FIXTURE / row['image_file']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['image_sha256']
    frame = rectify_bgr_frame(cv2.imdecode(np.frombuffer(raw, np.uint8), 1), ci, cv2, np)
    s = row['sensors']['scan']
    scan = PlainLaserScan(ranges=tuple(v if isinstance(v, (int, float)) else math.nan for v in s['ranges']),
        angle_min=s['angle_min'], angle_increment=s['angle_increment'], angle_max=s['angle_max'],
        range_min=s['range_min'], range_max=s['range_max'], scan_frame_id=s['header']['frame_id'],
        scan_stamp_sec=row['scan_stamp_sec'], receipt_sec=row['scan_received_ros_sec'], scan_topology_profile='full_rotation')
    def tf(parent, child):
        t = next(t for t in row['tf_samples'] if t['target_frame'] == parent and t['source_frame'] == child)
        return RigidTransform(parent, child, tuple(t['translation_xyz_m']), tuple(t['rotation_xyzw']))
    intrinsics = CameraIntrinsics(800, 600, ci.p[0], ci.p[5], ci.p[2], ci.p[6])
    common = {k: row['association'][k] for k in ('map_bearing_rad', 'cone_half_angle_rad', 'accepted_range_m')}
    common.update(scan=scan, now_sec=row['outcome_ros_sec'], max_scan_age_sec=.5,
                  max_camera_map_bearing_delta_rad=math.radians(12))
    search = dict(common, intrinsics=intrinsics, scan_from_map=tf('base_scan', 'map'),
        camera_from_map=tf('camera', 'map'), image_stamp_sec=row['image_stamp_sec'], sync_tolerance_sec=.1,
        model_profile=SimpleNamespace(**json.loads((ROOT / 'configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json').read_text())))
    binding = dict(common, intrinsics=intrinsics, scan_from_camera=tf('base_scan', 'camera'),
        min_cluster_sample_count=row['association']['min_cluster_sample_count'],
        camera_registration_accepted=False, roi=ImageRoi(0, 0, 800, 600, 100))
    return frame, search, binding


@pytest.mark.parametrize('row', json.loads((FIXTURE / 'inputs.json').read_text())['rows'], ids=['third_view_early', 'third_view_late'])
def test_recorded_start_decodes_and_binds_without_a_head_angle(row):
    frame, search, binding = recorded_context(row)
    attempt, info = current_scan_qr_search(**search)
    assert info['accepted'] and not info['supplies_head_geometry']
    r = attempt.roi
    # CI does not benchmark hardware. The deterministic work-policy tests cover
    # deadline ordering; this replay checks the actual image/corner geometry.
    decoded = detect_qr_observations_bgr(frame[r.y0:r.y1, r.x0:r.x1], cv2,
        prefer_native_geometry=True, preferred_scale=4)
    assert [d.text for d in decoded] == ['Start']
    assert all(d.corners is not None for d in decoded)
    restored = tuple(replace(d, corners=tuple((u+r.x0, v+r.y0) for u, v in d.corners)) for d in decoded)
    assert not bind_qr_observations_to_target(restored, **binding).accepted
    result = bind_qr_observations_to_target(restored, **binding, allow_independent_registration=True)
    assert result.accepted, result.reason
    assert result.qr_texts_for_evidence == ('Start',)
    assert result.independent_registration['envelope']['eligible_cluster_count'] == 1
    assert not result.metadata()['motion_authorized']
    assert not result.metadata()['completion_authorized']
    assert not bind_qr_observations_to_target(restored, **dict(binding, now_sec=row['scan_stamp_sec']+.501),
                                             allow_independent_registration=True).accepted
    assert current_scan_qr_search(**dict(search, image_stamp_sec=row['scan_stamp_sec']-.101))[0] is None
