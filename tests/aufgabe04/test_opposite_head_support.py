"""Current head pixels replace QR outlines without granting identity or angle."""
from dataclasses import replace
import math
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from scripts.aufgabe04.real_robot.observer.opposite_head_support import (
    HEAD_POLICY, HEAD_REGION_POLICY, detect_opposite_head_region,
    head_region_from_corners,
    support_opposite_head_region, validate_opposite_head_region,
)
from scripts.aufgabe04.real_robot.observer.opposite_target_support import validate_target_support
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search
from tests.aufgabe04.opposite_head_identity_fixture import recorded_head_identity, recorded_head_image

MODULE = 'scripts.aufgabe04.real_robot.observer.opposite_head_support'


@pytest.fixture(scope='module')
def recorded(tmp_path_factory):
    case = recorded_head_identity(tmp_path_factory.mktemp('head-support'), 31)
    image = recorded_head_image(case)
    search_options = dict(intrinsics=case['intrinsics'], model_profile=case['model'],
        scan=case['scan'], scan_from_map=case['scan_from_map'], camera_from_map=case['camera_from_map'],
        image_stamp_sec=case['metadata']['image_stamp_sec'], now_sec=case['now_sec'],
        max_scan_age_sec=.5, sync_tolerance_sec=.1, target_reconciliation=case['reconciliation'],
        **case['row']['options'])
    attempt, _ = current_scan_qr_search(**search_options)
    support_options = {k: v for k, v in search_options.items()
                       if k not in ('scan_from_map', 'camera_from_map', 'sync_tolerance_sec')}
    support_options.update(attempt=attempt, scan_from_camera=case['scan_from_camera'])
    # The fixture retains the recorded current scan clocks and proof. Detection
    # alone has a larger offline budget to avoid platform-speed test flakiness.
    return SimpleNamespace(case=case, image=image, attempt=attempt, options=support_options)


def region(recorded, **extra):
    options = dict(attempt=recorded.attempt, model_profile=recorded.case['model'],
        image_stamp_sec=recorded.case['metadata']['image_stamp_sec'],
        now_sec=recorded.case['now_sec'], max_scan_age_sec=.5, max_elapsed_sec=.2)
    options.update(extra)
    return detect_opposite_head_region(recorded.image, cv2, **options)


def test_real_current_border_needs_no_qr_detector_or_angle_fit(recorded, monkeypatch):
    forbidden = Mock(side_effect=AssertionError('QR geometry must not run'))
    monkeypatch.setattr(cv2, 'QRCodeDetector', forbidden)
    resources = SimpleNamespace(decoder=forbidden)
    current = region(recorded, resources=resources)
    assert current is not None
    assert 500 < current.full_image_center_px[0] < 505
    assert 283 < current.full_image_center_px[1] < 288
    metadata = validate_opposite_head_region(current.metadata())
    assert metadata['policy'] == HEAD_REGION_POLICY
    for field in ('supplies_angle', 'supplies_identity', 'supplies_target_uniqueness', 'motion_authorized'):
        assert metadata[field] is False
        with pytest.raises(ValueError, match='cannot grant authority'):
            validate_opposite_head_region({**metadata, field: True})
    forbidden.assert_not_called()


def test_recorded_head_has_separate_unique_synchronized_scan_support(recorded):
    current = region(recorded)
    support = support_opposite_head_region(current,
        image_shape=recorded.image.shape[:2], **recorded.options)
    assert support is not None
    evidence = validate_target_support(support.metadata())
    assert evidence['policy'] == HEAD_POLICY
    assert 'expected_symbol_height_px' not in evidence
    assert support.target_reconciliation == recorded.case['reconciliation']
    indices = tuple(support.lidar_association.search_association.selected_cluster_source_indices)
    assert indices == (207, 208, 209)
    assert 0 < support.depth_m < .7
    assert not evidence['supplies_angle'] and not evidence['supplies_identity']


def test_existing_current_head_corners_do_not_trigger_new_detection(recorded, monkeypatch):
    current = region(recorded)
    monkeypatch.setattr(cv2, 'createLineSegmentDetector', Mock(side_effect=AssertionError('redetection')))
    rebuilt = head_region_from_corners(current.corners_px, image_shape=current.image_shape,
        image_stamp_sec=current.image_stamp_sec, attempt=recorded.attempt)
    assert rebuilt == current
    assert support_opposite_head_region(rebuilt,
        image_shape=recorded.image.shape[:2], **recorded.options) is not None


@pytest.mark.parametrize('height_ratio', [.65, 1.33])
def test_already_measured_head_keeps_existing_physical_scale_limits(recorded, height_ratio):
    current = region(recorded)
    corners = current.corners_px
    height = (math.dist(corners[0], corners[3])+math.dist(corners[1], corners[2]))/2
    attempt = replace(recorded.attempt, expected_head_height_px=height/height_ratio)
    reused = head_region_from_corners(corners, image_shape=current.image_shape,
        image_stamp_sec=current.image_stamp_sec, attempt=attempt)
    assert reused is not None
    options = {**recorded.options, 'attempt': attempt}
    assert support_opposite_head_region(reused, image_shape=current.image_shape, **options) is not None


def test_already_registered_head_keeps_current_scan_authority_outside_narrow_locator(recorded):
    current = region(recorded)
    # A shifted projection remains merely a hint. Already measured real pixels
    # keep their unchanged exact-tuple/3-degree scan registration.
    attempt = replace(recorded.attempt,
        expected_center_u_px=current.full_image_center_px[0]-1.2*current.expected_head_height_px,
        expected_center_v_px=current.full_image_center_px[1])
    reused = head_region_from_corners(current.corners_px, image_shape=current.image_shape,
        image_stamp_sec=current.image_stamp_sec, attempt=attempt)
    assert reused is not None
    assert head_region_from_corners(current.corners_px, image_shape=current.image_shape,
        image_stamp_sec=current.image_stamp_sec, attempt=attempt, max_center_offset_ratio=.75) is None
    options = {**recorded.options, 'attempt': attempt}
    assert support_opposite_head_region(reused, image_shape=current.image_shape, **options) is not None


@pytest.mark.parametrize('change', ['stamp', 'shape', 'scale', 'stale', 'search_scale'])
def test_current_region_cannot_be_reused_for_changed_source(recorded, change):
    current = region(recorded)
    options = dict(recorded.options, image_shape=recorded.image.shape[:2])
    if change == 'stamp':
        current = replace(current, image_stamp_sec=current.image_stamp_sec-.01)
    elif change == 'shape':
        options['image_shape'] = (recorded.image.shape[0]+1, recorded.image.shape[1])
    elif change == 'scale':
        current = replace(current, expected_head_height_px=current.expected_head_height_px+1)
    elif change == 'stale':
        options['now_sec'] = current.image_stamp_sec+.501
    else:
        options['attempt'] = replace(recorded.attempt,
            expected_head_height_px=recorded.attempt.expected_head_height_px+1)
    assert support_opposite_head_region(current, **options) is None


@pytest.mark.parametrize('returns', ['absent', 'fragmented'])
def test_current_border_cannot_create_scan_uniqueness(recorded, returns):
    current = region(recorded)
    scan = recorded.options['scan']
    ranges = list(scan.ranges)
    if returns == 'absent':
        ranges = [math.nan]*len(ranges)
    else:
        ranges[208] = math.nan
    options = dict(recorded.options, scan=replace(scan, ranges=tuple(ranges)),
        target_reconciliation=None, image_shape=recorded.image.shape[:2])
    assert support_opposite_head_region(current, **options) is None


def test_unfinished_border_comparison_never_accepts_first_head(recorded, monkeypatch):
    monkeypatch.setattr(MODULE+'._rough_proposals', lambda *a, **kw: [(0, ())]*13)
    diagnostics = {}
    assert region(recorded, diagnostics=diagnostics) is None
    assert diagnostics['reason'] == 'head_region_verification_budget_exceeded'


def test_distinct_supported_regions_do_not_select_one(recorded, monkeypatch):
    # The real frame has several complete rail fits belonging to one border
    # family. Explicitly make those families distinct to exercise ambiguity.
    monkeypatch.setattr(MODULE+'._same_head', lambda *args: False)
    diagnostics = {}
    assert region(recorded, diagnostics=diagnostics) is None
    assert diagnostics['reason'] == 'multiple_current_head_regions'


def test_expired_budget_stale_image_and_blank_pixels_fail_closed(recorded):
    assert region(recorded, max_elapsed_sec=0) is None
    assert region(recorded, now_sec=recorded.options['image_stamp_sec']+.501) is None
    options = {k: recorded.options[k] for k in
        ('attempt', 'model_profile', 'image_stamp_sec', 'now_sec', 'max_scan_age_sec')}
    assert detect_opposite_head_region(np.zeros_like(recorded.image), cv2,
        max_elapsed_sec=.2, **options) is None


@pytest.mark.parametrize('change', ['clipped', 'incomplete', 'scale', 'center'])
def test_persisted_head_region_geometry_is_checked(recorded, change):
    current = region(recorded)
    metadata = current.metadata()
    if change == 'clipped':
        metadata['corners_px'] = ((-1., 0.), *current.corners_px[1:])
    elif change == 'incomplete':
        metadata['corners_px'] = current.corners_px[:3]
    elif change == 'scale':
        metadata['expected_head_height_px'] *= 2
    else:
        metadata['center_px'] = (0., 0.)
    with pytest.raises(ValueError):
        validate_opposite_head_region(metadata)
