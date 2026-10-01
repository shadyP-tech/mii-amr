"""Visual search hints never replace a current, independently proved target."""

from dataclasses import replace
import math
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from scripts.aufgabe04.real_robot.observer.opposite_target_support import (
    OUTLINE_POLICY,
    detect_opposite_qr_outline,
    support_opposite_qr_outline,
    validate_opposite_qr_outline,
    validate_target_support,
)
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search
from tests.aufgabe04.opposite_reconciliation_fixture import recorded_opposite


# Native detectMulti(scale=4) on the calibrated, rectified frame.jpg in
# fixtures/opposite_reconciliation_20260923. These are full-image coordinates;
# the mock converts them back to the native crop coordinates deterministically.
RECORDED_CORNERS = (
    (510.5, 221.25), (616.4409790039062, 217.70496368408203),
    (624.5, 326.25), (514.5, 328.5),
)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    image, options, _, tf, proof, *_ = recorded_opposite(
        tmp_path_factory.mktemp("opposite_outline"))
    attempt, _ = current_scan_qr_search(**options)
    assert attempt is not None
    keys = (
        "intrinsics", "model_profile", "scan", "image_stamp_sec", "now_sec",
        "map_bearing_rad", "cone_half_angle_rad", "accepted_range_m",
        "max_scan_age_sec", "max_camera_map_bearing_delta_rad",
    )
    return SimpleNamespace(image=image, attempt=attempt, options=options,
        support_options=dict(attempt=attempt, image_shape=image.shape[:2],
            scan_from_camera=tf("base_scan", "camera"), target_reconciliation=proof,
            **{key: options[key] for key in keys}))


def acquire(recorded, quads=(RECORDED_CORNERS,)):
    roi = recorded.attempt.roi
    points = np.asarray([[(4*(u-roi.x0), 4*(v-roi.y0)) for u, v in corners]
                         for corners in quads], dtype=np.float64)
    # The available native object has no payload decoder at all.
    detector = SimpleNamespace(detectMulti=Mock(return_value=(True, points)))
    resources = SimpleNamespace(decoder=Mock(return_value=detector))
    diagnostics = {}
    result = detect_opposite_qr_outline(recorded.image, cv2,
        attempt=recorded.attempt, model_profile=recorded.options["model_profile"],
        image_stamp_sec=recorded.options["image_stamp_sec"],
        now_sec=recorded.options["now_sec"], max_scan_age_sec=.5,
        max_elapsed_sec=.5, resources=resources, diagnostics=diagnostics)
    resources.decoder.assert_called_once_with("native")
    return result, detector, diagnostics


def test_one_complete_current_outline_has_no_target_identity_or_motion_authority(recorded):
    outline, detector, diagnostics = acquire(recorded)
    assert outline is not None
    assert outline.corners_px == RECORDED_CORNERS
    detector.detectMulti.assert_called_once()
    assert diagnostics["reason"] == "current_outline_search_only"
    metadata = validate_opposite_qr_outline(outline.metadata())
    assert metadata["policy"] == OUTLINE_POLICY
    assert not metadata.get("accepted", False)
    assert "lidar_association" not in metadata
    for field in ("supplies_angle", "supplies_identity", "supplies_target_uniqueness",
                  "motion_authorized"):
        assert metadata[field] is False
        with pytest.raises(ValueError, match="cannot grant authority"):
            validate_opposite_qr_outline({**metadata, field: True})
    with pytest.raises(ValueError, match="support missing"):
        validate_target_support(metadata)


def test_multiple_eligible_outlines_do_not_select_one(recorded):
    # A second nearby detector result stays within the same scale and center
    # limits. It is an explicit ambiguity injection, not positive run evidence.
    shifted = tuple((u+10, v) for u, v in RECORDED_CORNERS)
    assert acquire(recorded, (shifted,))[0] is not None
    outline, detector, diagnostics = acquire(recorded, (RECORDED_CORNERS, shifted))
    assert outline is None
    detector.detectMulti.assert_called_once()
    assert diagnostics["reason"] == "multiple_search_outlines"


def test_incomplete_outline_is_not_a_visual_witness(recorded):
    outline, _, diagnostics = acquire(recorded, (RECORDED_CORNERS[:3],))
    assert outline is None
    assert diagnostics["reason"] == "no_complete_target_outline"


def test_same_current_outline_can_use_existing_reconciliation_without_redetection(recorded):
    outline, detector, _ = acquire(recorded)
    support = support_opposite_qr_outline(outline, **recorded.support_options)
    assert support is not None
    assert support.corners_px == RECORDED_CORNERS
    assert support.image_stamp_sec == recorded.options["image_stamp_sec"]
    assert support.target_reconciliation == recorded.support_options["target_reconciliation"]
    validate_target_support(support.metadata())
    detector.detectMulti.assert_called_once()


@pytest.mark.parametrize("change", ["stamp", "shape", "scale", "stale", "hint_scale"])
def test_reused_outline_rejects_changed_or_stale_source(recorded, change):
    outline, _, _ = acquire(recorded)
    options = dict(recorded.support_options)
    if change == "stamp":
        outline = replace(outline, image_stamp_sec=outline.image_stamp_sec-.01)
    elif change == "shape":
        options["image_shape"] = (recorded.image.shape[0]+1, recorded.image.shape[1])
    elif change == "scale":
        outline = replace(outline, expected_symbol_height_px=outline.expected_symbol_height_px+1)
    elif change == "stale":
        options["now_sec"] = outline.image_stamp_sec+.501
    else:
        options["attempt"] = replace(recorded.attempt,
            expected_head_height_px=recorded.attempt.expected_head_height_px+1)
    assert support_opposite_qr_outline(outline, **options) is None


@pytest.mark.parametrize("returns", ["absent", "fragmented"])
def test_search_hint_and_complete_outline_cannot_grant_scan_support(recorded, returns):
    outline, _, _ = acquire(recorded)
    scan = recorded.options["scan"]
    ranges = list(scan.ranges)
    if returns == "absent":
        ranges = [math.nan]*len(ranges)
    else:
        # The recorded target is the contiguous group [207, 208, 209]. Remove
        # its middle real return: the visual hint cannot join the two fragments.
        ranges[208] = math.nan
    diagnostics = {}
    options = {**recorded.support_options, "scan": replace(scan, ranges=tuple(ranges)),
               "target_reconciliation": None, "diagnostics": diagnostics}
    assert support_opposite_qr_outline(outline, **options) is None
    assert diagnostics["reason"] == "scan_cluster_not_unique"
