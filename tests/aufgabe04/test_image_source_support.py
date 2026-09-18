from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.stand_axis.image_source_support import ImageSourceSupport
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.head_geometry_acquisition import estimate_current_head_geometry
from tests.aufgabe04.test_physical_head_pipeline import profile, head_image, estimate


def quad(x, y, size):
    return tuple(ImagePoint(*p) for p in ((x,y),(x+size,y),(x+size,y+size),(x,y+size)))


def test_camera_support_excludes_canvas_not_dark_scene_pixels():
    valid = np.ones((240, 320), np.uint8)
    valid[:8] = 0
    original = valid.copy()
    support = ImageSourceSupport(cv2, valid)
    assert not support.accepts(quad(30, 8, 80))
    assert support.accepts(quad(30, 30, 80))
    assert not support.segment((30,8),(110,8))
    assert support.segment((30,30),(110,30))
    np.testing.assert_array_equal(valid, original)
    valid[:] = 0  # Construction took its own immutable copy.
    assert support.accepts(quad(30,30,80))


def test_support_checks_whole_border_and_composes_preview_separately():
    valid = np.ones((240,320), np.uint8)
    valid[60:70,98:103] = 0
    support = ImageSourceSupport(cv2, valid)
    assert not support.accepts(quad(20,20,80))
    other = Mock(return_value=False)
    other.preview = Mock(return_value=True)
    filtering = support.filter(other)
    candidate = SimpleNamespace(corners=quad(140,30,60))
    assert filtering.preview(candidate)
    assert not filtering(candidate)
    other.preview.assert_called_once_with(candidate)
    other.assert_called_once_with(candidate)


def test_tracked_fit_cannot_bypass_current_source_support(profile):
    frame, _ = head_image(profile, angle=35.)
    fitted, debug = estimate(profile, frame)
    assert fitted.usable
    camera = dict(camera_fx_px=640.,camera_fy_px=640.,camera_cx_px=400.,camera_cy_px=300.)
    support = ImageSourceSupport(cv2, np.zeros(frame.shape[:2],np.uint8))
    est, dbg = estimate_current_head_geometry(cv2,frame,model_profile=profile,
        source_support=support,pose_hint=debug.model_pose,
        estimator=Mock(return_value=(fitted,debug)),**camera)
    assert not est.usable and est.corners is None
    assert dbg.model_pose is None and dbg.head_outer_recovery is None
    assert est.reason == "head_border_outside_source_image"


def test_expired_model_work_never_computes_canny(profile):
    from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
    frame, _ = head_image(profile)
    with patch("scripts.aufgabe04.perception.stand_axis.model_pipeline._canny_edges_from_frame") as edges:
        est, dbg = estimate_stand_axis_from_metric_model(cv2,frame,model_profile=profile,
            camera_fx_px=640.,camera_fy_px=640.,camera_cx_px=400.,camera_cy_px=300.,
            deadline_monotonic_sec=0.)
    assert est.reason == "head_acquisition_deadline_exceeded"
    edges.assert_not_called()
    assert dbg.head_acquisition_diagnostics["deadline_stage"] == "before_edge_preprocessing"


def test_delayed_or_ambiguous_border_is_display_only(profile):
    from scripts.aufgabe04.perception.debug.viewer_model_overlay_policy import current_border_diagnostic, current_model_overlay_state
    frame, _ = head_image(profile,angle=35.)
    est, dbg = estimate(profile,frame)
    assert est.usable
    delayed = current_border_diagnostic(estimate=est,artifacts=dbg,local_result_fresh=True,source_fresh=False)
    assert delayed["visible"] and delayed["state"] == "delayed_image_borders"
    assert not delayed["motion_authorized"] and not delayed["pose_authorized"]
    assert not current_model_overlay_state(inputs_ready=True,estimate=est,artifacts=dbg,result_fresh=False).current_fit_accepted
    assert not current_border_diagnostic(estimate=est,artifacts=dbg,local_result_fresh=False,source_fresh=False)["visible"]
    ambiguous = replace(est,usable=False,reason="head_model_planar_axis_ambiguous",yaw_deg=None)
    assert current_border_diagnostic(estimate=ambiguous,artifacts=dbg,local_result_fresh=True,source_fresh=True)["visible"]
