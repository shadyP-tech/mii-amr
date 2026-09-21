"""Replay the displaced grey backside without hand-supplied head geometry."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import pytest

cv2 = pytest.importorskip('cv2')
np = pytest.importorskip('numpy')

from scripts.aufgabe04.perception.camera_calibration import CameraCalibration, rectify_bgr_frame, rectified_source_support
from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
from scripts.aufgabe04.perception.stand_axis.image_source_support import ImageSourceSupport
from scripts.aufgabe04.perception.stand_axis.lidar_head_edge_region import project_lidar_candidate_head_region
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform, rectified_pixel_bearing_in_scan
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.qr_scanning.native_qr_observations import detect_native_qr_observations_bgr
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, OpticalProjection
from scripts.aufgabe04.real_robot.observer.stopped_target_search import reconcile_stopped_target_search
from scripts.aufgabe04.real_robot.observer.current_scan_head_proposal_filter import CurrentScanHeadProposalFilter
from scripts.aufgabe04.real_robot.observer.current_head_association import associate_current_measured_head
from scripts.aufgabe04.real_robot.observer.qr_acquisition_policy import QrAcquisitionPolicy
from scripts.aufgabe04.real_robot.observer.qr_decode_cache import RoiQrDecodeCache
from scripts.aufgabe04.real_robot.observer.viewer_head_acquisition import evaluate_viewer_head, classify_viewer_head
from scripts.aufgabe04.real_robot.observer.tracked_head_registration import (
    tracked_head_selection, register_current_tracked_head, review_current_tracked_head_crop,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / 'fixtures/stopped_target_search_20260921'


def recorded_options(index=9):
    data = json.loads((FIXTURE / f'frame_{index:06d}.json').read_text())
    c, s = data['sensors']['camera_info'], data['sensors']['scan']
    cal = CameraCalibration(c['width'], c['height'], 'camera', tuple(c['k']),
                            tuple(c['d']), tuple(c['r']), tuple(c['p']))
    intr = CameraIntrinsics(c['width'], c['height'], cal.fx_px, cal.fy_px, cal.cx_px, cal.cy_px)
    def tf(parent, child):
        t = next(t for t in data['tf_samples'] if t['target_frame'] == parent and t['source_frame'] == child)
        return RigidTransform(parent, child, tuple(t['translation_xyz_m']), tuple(t['rotation_xyzw']))
    scan = PlainLaserScan(tuple(float(r) for r in s['ranges']), s['angle_min'], s['angle_increment'],
        s['range_min'], s['range_max'], 'base_scan', data['scan_stamp_sec'], data['scan_received_ros_sec'],
        s['angle_max'], 'full_rotation')
    a = data['preliminary_association']
    model = load_measured_physical_stand_model(ROOT / 'configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json')
    options = dict(scan=scan, candidate_xy=tuple(data['candidate_xy']),
        original_projection=OpticalProjection(**data['target_projection']),
        map_bearing_rad=a['map_bearing_rad'], cone_half_angle_rad=a['cone_half_angle_rad'],
        max_bearing_delta_rad=math.radians(12), accepted_range_m=tuple(a['accepted_range_m']),
        now_sec=data['now_sec'], image_stamp_sec=data['image_stamp_sec'], max_age_sec=.5,
        sync_tolerance_sec=.15, scan_from_map=tf('base_scan', 'map'), camera_from_map=tf('camera', 'map'),
        intrinsics=intr, model_profile=model, stand_radius_m=.06, stand_uncertainty_m=.02)
    return options, data, cal, tf


def test_fresh_unique_scan_reconciles_search_without_mutating_candidate():
    options, _, _, _ = recorded_options()
    original = dict(options)
    hint, info = reconcile_stopped_target_search(**options)
    assert hint is not None, info
    assert options == original
    assert .10 < hint.residual_m < .15
    assert 210 < hint.projection.u_px < 270
    assert hint.original_projection.u_px > 370
    assert not info['candidate_geometry_updated'] and not info['motion_authorized']
    assert info['final_association_required']


@pytest.mark.parametrize('change', [
    {'now_sec': 1.}, {'image_stamp_sec': 1.}, {'image_stamp_sec': math.nan},
    {'candidate_xy': (0., 0.)}, {'candidate_xy': (math.nan, 0.)},
    {'stand_radius_m': -1.}, {'accepted_range_m': (.8, 1.)},
    {'map_bearing_rad': math.radians(-5)},
])
def test_invalid_stale_out_of_range_or_displaced_hint_is_rejected(change):
    options, _, _, _ = recorded_options()
    hint, info = reconcile_stopped_target_search(**{**options, **change})
    assert hint is None and not info['accepted']


def test_single_point_competitor_and_fragmentation_prevent_search_shift():
    options, _, _, _ = recorded_options()
    for mutation in ('competitor', 'split', 'two_samples', 'wrong_frame'):
        scan = options['scan']; ranges = list(scan.ranges)
        hint, _ = reconcile_stopped_target_search(**options)
        if mutation == 'competitor':
            ranges[0] = .53  # Inside the original cone and range, isolated from the target.
        elif mutation == 'split':
            ranges[hint.source_indices[len(hint.source_indices)//2]] = math.nan
        elif mutation == 'two_samples':
            for i in hint.source_indices[2:]: ranges[i] = math.nan
        changed = replace(scan, ranges=tuple(ranges), scan_frame_id='wrong' if mutation == 'wrong_frame' else scan.scan_frame_id)
        result, info = reconcile_stopped_target_search(**{**options, 'scan': changed})
        assert result is None, (mutation, info)


@pytest.mark.parametrize('index', (4, 9))
def test_recorded_backside_passes_current_geometry_association_and_marker_absence(index):
    options, data, cal, tf = recorded_options(index)
    hint, _ = reconcile_stopped_target_search(**options)
    assert hint is not None
    image = (FIXTURE / f'frame_{index:06d}.jpg').read_bytes()
    assert hashlib.sha256(image).hexdigest() == data['image_sha256']
    frame = rectify_bgr_frame(cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR), cal, cv2, np)
    intr, model, projection = options['intrinsics'], options['model_profile'], options['original_projection']
    common = {key: options[key] for key in ('scan', 'map_bearing_rad', 'cone_half_angle_rad', 'accepted_range_m', 'now_sec')}
    wide = associate_candidate_lidar_target(**{**common, 'cone_half_angle_rad': math.radians(15)}, max_scan_age_sec=.5)
    region, region_info = project_lidar_candidate_head_region(candidate_xy=hint.search_xy,
        camera_from_map=tf('camera', 'map'), model_profile=model,
        intrinsics=(intr.fx_px, intr.fy_px, intr.cx_px, intr.cy_px), image_shape=frame.shape,
        position_uncertainty_m=.02, surface_center_margin_m=.06, association=wide,
        image_stamp_sec=options['image_stamp_sec'], now_sec=common['now_sec'], max_sensor_age_sec=.5, sync_tolerance_sec=.15)
    assert region is not None
    filtering = CurrentScanHeadProposalFilter(**common, intrinsics=intr, scan_from_camera=tf('base_scan', 'camera'),
        max_scan_age_sec=.5, min_cluster_sample_count=1, max_camera_map_bearing_delta_rad=math.radians(12),
        optical_depth_m=hint.projection.depth_m, depth_uncertainty_m=.08)
    # Real image and native marker processing, with no supplied corners, yaw,
    # tracker pose, persisted scan witness or decoder result. Timing is tested
    # separately: a slow CI machine must not turn this geometric regression flaky.
    started = time.monotonic()
    budget = QrAcquisitionPolicy().begin_frame(target_key='recorded_candidate',
        image_stamp_sec=options['image_stamp_sec'], started_ros_sec=common['now_sec'],
        started_monotonic_sec=started, max_sensor_age_sec=10.)
    height = intr.fy_px * model.head_height_m / projection.depth_m
    evaluation = evaluate_viewer_head(cv2, frame, model_profile=model, intrinsics=intr, pose_hint=None,
        projection=hint.projection, expected_head_height_px=height, fallback_attempt=None,
        cache=RoiQrDecodeCache(), budget=budget,
        native_decoder=lambda crop: detect_native_qr_observations_bgr(crop, cv2),
        full_decoder=lambda crop, limit, provenance: detect_qr_observations_bgr(crop, cv2,
            diagnostics=provenance, max_elapsed_sec=limit, prefer_native_geometry=True),
        deadline_monotonic_sec=None, proposal_filter=filtering, lidar_edge_region=region,
        lidar_edge_region_diagnostics=region_info, depth_uncertainty_m=.08, position_uncertainty_m=.08,
        camera_vertical=rotate_vector((0., 0., 1.), tf('camera', 'map').rotation_xyzw),
        source_support=ImageSourceSupport(cv2, rectified_source_support(cal, cv2, np)))
    evaluation = classify_viewer_head(evaluation, model_profile=model, intrinsics=intr, expected_head_height_px=height)
    association_args = dict(**common, estimate=evaluation.estimate, debug=evaluation.debug,
        profile_sha256=model.sha256, attempt=evaluation.attempt, projection=projection,
        expected_head_height_px=height, intrinsics=intr, scan_from_camera=tf('base_scan', 'camera'),
        max_scan_age_sec=.5, min_cluster_sample_count=1, max_center_offset_ratio=1.5,
        max_camera_map_bearing_delta_rad=math.radians(12), search_reconciliation=hint)
    association = associate_current_measured_head(**association_args)
    assert association.accepted, association.reason
    assert 200 < association.full_image_center_px[0] < 250
    if index == 9:
        assert evaluation.estimate.usable, evaluation.estimate.reason
        assert evaluation.estimate.visible_face == 'backside_candidate'
    selection = register_current_tracked_head(tracked_head_selection(evaluation), association=association,
        observed_at_sec=options['image_stamp_sec'], now_sec=common['now_sec'], max_age_sec=.5,
        expected_model_sha256=model.sha256)
    review = review_current_tracked_head_crop(selection, require_marker_absence=True)
    assert review.accepted, review.reason
    assert evaluation.debug.qr_detected is False and evaluation.qr_observations == ()
    # Reconciliation cannot lend registration to another scan, map prior or cluster.
    for changes in (
        {'scan': replace(options['scan'])},
        {'projection': replace(projection, u_px=projection.u_px+1)},
        {'map_bearing_rad': common['map_bearing_rad']+.01},
        {'search_reconciliation': replace(hint, source_indices=(99,))},
        {'now_sec': common['now_sec']+1.},
    ):
        rejected = associate_current_measured_head(**{**association_args, **changes})
        assert not rejected.accepted, changes
    # Finite-distance projection preserves the existing 12-degree map gate.
    pixels = association.full_image_center_px
    ray = dict(u_px=pixels[0], v_px=pixels[1], fx_px=intr.fx_px, fy_px=intr.fy_px,
               cx_px=intr.cx_px, cy_px=intr.cy_px, scan_from_camera=tf('base_scan', 'camera'))
    distant_bearing = rectified_pixel_bearing_in_scan(**ray)
    finite_bearing = rectified_pixel_bearing_in_scan(**ray, optical_depth_m=hint.projection.depth_m)
    assert abs(distant_bearing-common['map_bearing_rad']) > math.radians(12)
    assert abs(finite_bearing-common['map_bearing_rad']) < math.radians(12)


@pytest.mark.parametrize('depth', (0., -1., math.nan, math.inf))
def test_invalid_finite_depth_cannot_supply_a_scan_bearing(depth):
    _, _, _, tf = recorded_options()
    with pytest.raises(ValueError, match='optical depth'):
        rectified_pixel_bearing_in_scan(u_px=228., v_px=289., fx_px=640., fy_px=640.,
            cx_px=400., cy_px=300., scan_from_camera=tf('base_scan', 'camera'), optical_depth_m=depth)


def test_finite_projection_rejects_nonfinite_translation():
    _, _, _, tf = recorded_options()
    with pytest.raises(ValueError, match='translation'):
        rectified_pixel_bearing_in_scan(u_px=228., v_px=289., fx_px=640., fy_px=640.,
            cx_px=400., cy_px=300., optical_depth_m=.5,
            scan_from_camera=replace(tf('base_scan', 'camera'), translation_xyz_m=(math.nan, 0., 0.)))


def test_nominal_target_keeps_existing_search_even_when_an_offset_hint_is_possible():
    options, _, _, _ = recorded_options()
    hint, _ = reconcile_stopped_target_search(**options)
    scan = options['scan']
    points = [(scan.ranges[i]*math.cos(scan.angle_min+i*scan.angle_increment),
               scan.ranges[i]*math.sin(scan.angle_min+i*scan.angle_increment)) for i in hint.source_indices]
    bearing = math.atan2(sum(p[1] for p in points), sum(p[0] for p in points))
    unchanged, info = reconcile_stopped_target_search(**{**options, 'map_bearing_rad': bearing})
    assert unchanged is None and info['reason'] == 'original_target_search_retained'
