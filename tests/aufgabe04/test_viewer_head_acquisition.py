"""Full-image viewer geometry and head-local marker evidence remain separate."""

from dataclasses import replace
import math
from unittest.mock import Mock, patch

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.stand_axis.head_geometry_acquisition import estimate_current_head_geometry
from scripts.aufgabe04.perception.stand_axis.head_model_admission import admit_measured_head_model
from scripts.aufgabe04.perception.stand_axis.qr_marker_validation import QrMarkerEvidence
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import QrQuadDetection, detect_qr_quad
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, ImageRoi, OpticalProjection
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt
from scripts.aufgabe04.real_robot.observer.qr_acquisition_policy import QrAcquisitionPolicy
from scripts.aufgabe04.real_robot.observer.qr_decode_cache import RoiQrDecodeCache
from scripts.aufgabe04.real_robot.observer.viewer_head_acquisition import (
    VIEWER_HEAD_SOURCE, classify_viewer_head, evaluate_viewer_head,
)
from tests.aufgabe04 import test_head_boundary_independence as fixtures


MODULE = "scripts.aufgabe04.real_robot.observer.viewer_head_acquisition"
QR = DecodedQrObservation("QR_003", ((20., 20.), (50., 20.), (50., 50.), (20., 50.)), "test")


@pytest.fixture(scope="module")
def scene():
    fixtures.HeadBoundaryIndependenceTest.setUpClass()
    fixture = fixtures.HeadBoundaryIndependenceTest()
    head, camera = fixture.projection(angle_deg=20.)
    image = np.zeros((600, 800, 3), np.uint8)
    cv2.fillConvexPoly(image, np.rint([(p.u_px, p.v_px) for p in head]).astype(np.int32), (200, 200, 200))
    height = sum(math.hypot(head[i].u_px-head[j].u_px, head[i].v_px-head[j].v_px)
                 for i, j in ((0, 3), (1, 2)))/2
    intrinsics = CameraIntrinsics(800, 600, camera.fx_px, camera.fy_px, camera.cx_px, camera.cy_px)
    fit = estimate_current_head_geometry(cv2, image, model_profile=fixture.profile,
        camera_fx_px=camera.fx_px, camera_fy_px=camera.fy_px,
        camera_cx_px=camera.cx_px, camera_cy_px=camera.cy_px)
    assert fit[0].usable, fit[0].reason
    return image, fixture.profile, intrinsics, height, fit


def evaluate(scene, *, native=None, full=None, available=2., clock=None, **changes):
    image, profile, intrinsics, height, fit = scene
    if clock is None:
        clock = [10.]
    budget = QrAcquisitionPolicy().begin_frame(target_key="candidate", image_stamp_sec=10.,
        started_ros_sec=10., started_monotonic_sec=10., max_sensor_age_sec=available)
    options = dict(model_profile=profile, intrinsics=intrinsics, pose_hint=None,
        # Displacement stays inside association bounds; it must not crop or seed geometry.
        projection=OpticalProjection(340., 260., .35, height, True),
        expected_head_height_px=height, fallback_attempt=None, cache=RoiQrDecodeCache(),
        budget=budget, native_decoder=native or (lambda crop: ()),
        full_decoder=full or (lambda *args: ()), deadline_monotonic_sec=None,
        estimator=Mock(return_value=fit), now=lambda: clock[0])
    options.update(changes)
    result = evaluate_viewer_head(cv2, image, **options)
    return result, options["estimator"]


def classified(scene, result, **changes):
    options = dict(model_profile=scene[1], intrinsics=scene[2], expected_head_height_px=scene[3])
    options.update(changes)
    return classify_viewer_head(result, **options)


def test_geometry_uses_original_image_intrinsics_and_only_a_conservative_candidate_screen(scene):
    result, estimator = evaluate(scene, native=lambda crop: (QR,))
    estimator.assert_called_once()
    assert estimator.call_args.args[1] is scene[0]
    options = estimator.call_args.kwargs
    assert options["camera_cx_px"] == scene[2].cx_px
    assert options["camera_cy_px"] == scene[2].cy_px
    assert options["min_edge_height_px"] == 8.
    assert options["qr_marker_policy"] == "disabled"
    assert not any(key.startswith("expected_head_") for key in options)
    assert "current_head_proposal_corners" not in options
    assert options["candidate_search"].center == (340., 260.)
    assert options["candidate_search"].height == pytest.approx(scene[2].fy_px*scene[1].head_height_m/.35)
    assert options["candidate_search"].pixel_size.depth_m == .35
    assert options["proposal_filter"] is None
    assert result.qr_decode_metadata["current_scan_proposal_filter_applied"] is False
    assert result.qr_decode_metadata["candidate_screen"]["supplies_corners"] is False
    assert result.frame is scene[0]
    assert result.attempt.source == VIEWER_HEAD_SOURCE
    assert result.attempt.roi == ImageRoi(0, 0, 800, 600, scene[3])
    assert result.estimate is scene[4][0]
    assert result.debug.model_pose is scene[4][1].model_pose


def test_optional_current_scan_filter_reaches_full_image_geometry_unchanged(scene):
    scan_filter = lambda proposal: True
    result, estimator = evaluate(scene, proposal_filter=scan_filter, native=lambda crop: (QR,))
    assert estimator.call_args.args[1] is scene[0]
    assert estimator.call_args.kwargs["proposal_filter"] is scan_filter
    assert result.qr_decode_metadata["current_scan_proposal_filter_applied"] is True
    assert result.estimate is scene[4][0]


@pytest.mark.parametrize("projection", (None,
    OpticalProjection(math.nan, 260., .5, 100., False),
    OpticalProjection(340., 260., -.5, 100., True)))
def test_invalid_projection_keeps_unscreened_full_image_geometry(scene, projection):
    result, estimator = evaluate(scene, projection=projection, native=lambda crop: (QR,))
    assert estimator.call_args.args[1] is scene[0]
    assert estimator.call_args.kwargs["candidate_search"] is None
    assert result.qr_decode_metadata["candidate_screen"] is None


def test_default_clock_resolves_current_monotonic_seam(scene):
    with patch(f"{MODULE}.time.monotonic", return_value=10.25):
        result, _ = evaluate(scene, native=lambda crop: (QR,), now=None)
    assert result.qr_decode_metadata["geometry_completed_monotonic_sec"] == 10.25


def test_qr_crop_follows_complete_head_and_translates_once(scene):
    decoder = Mock(return_value=(QR,))
    with patch(f"{MODULE}.detect_qr_quad", side_effect=AssertionError("decoded identity needs no finder search")):
        result, _ = evaluate(scene, native=decoder)
    x0, y0, x1, y1 = result.qr_decode_metadata["identity_roi"]
    assert x0 > 250 and y0 > 150  # Follows current head, not displaced projection.
    crop = decoder.call_args.args[0]
    assert crop.shape == (y1-y0, x1-x0, 3)
    assert np.shares_memory(crop, scene[0])
    assert result.qr_observations[0].corners == tuple((u+x0, v+y0) for u, v in QR.corners)
    assert result.debug.qr_detected is True
    assert result.debug.qr_marker_verified is True
    assert classified(scene, result).estimate.visible_face is None


def test_complete_decoder_extent_is_not_promoted_to_qr_geometry(scene):
    def decoder(crop):
        height, width = crop.shape[:2]
        return (replace(QR, corners=((0., 0.), (float(width-1), 0.),
                    (float(width-1), float(height-1)), (0., float(height-1)))),)
    result, _ = evaluate(scene, native=decoder)
    assert result.qr_observations[0].text == QR.text
    assert result.qr_observations[0].corners is None
    assert result.qr_decode_metadata["acquisition"]["allowed"]  # Recover actual symbol corners.
    assert result.debug.qr_marker_verified is True


def test_empty_decode_without_finder_budget_is_unknown_not_backside(scene):
    clock = [10.]
    def native(crop):
        clock[0] += .04
        return ()
    with patch(f"{MODULE}.detect_qr_quad", side_effect=AssertionError("marker budget exhausted")):
        result, _ = evaluate(scene, native=native, clock=clock, available=.14)
    assert result.qr_observations == ()
    assert result.qr_decode_metadata["geometry_completed_monotonic_sec"] == 10.
    assert clock[0] > result.qr_decode_metadata["geometry_completed_monotonic_sec"]
    assert result.debug.qr_detected is None
    assert result.debug.qr_marker_verified is None
    assert classified(scene, result).estimate.visible_face is None


def test_no_identity_budget_runs_no_decoder_or_marker_work(scene):
    decoder = Mock(side_effect=AssertionError("identity budget exhausted"))
    result, estimator = evaluate(scene, native=decoder, full=decoder, available=.10)
    estimator.assert_called_once()
    assert result.estimate.usable
    assert result.qr_observations is None
    assert result.debug.qr_detected is None
    assert not result.qr_decode_metadata["performed"]


def test_actual_native_finder_absence_classifies_only_after_geometry_return(scene):
    with patch(f"{MODULE}.detect_qr_quad", wraps=detect_qr_quad) as finder:
        result, estimator = evaluate(scene)
    estimator.assert_called_once()
    finder.assert_called_once()
    assert finder.call_args.args[1].shape != scene[0].shape
    assert finder.call_args.kwargs == dict(scales=(1.,), allow_decode_fallback=False,
                                          allow_native_decode_fallback=False)
    assert result.debug.qr_detected is False and result.debug.qr_marker_verified is False
    assert result.estimate.visible_face is None
    side = classified(scene, result)
    assert side.estimate.visible_face == "backside_candidate"
    assert side.estimate.corners == result.estimate.corners
    assert side.estimate.yaw_deg == result.estimate.yaw_deg
    assert side.debug.model_pose is result.debug.model_pose
    assert side.debug.head_acquisition_diagnostics["original_candidate_association_required"]
    assert classified(scene, result, expected_head_height_px=scene[3]*4).estimate.visible_face is None


def test_undecoded_qr_quadrilateral_vetoes_backside_even_if_finders_fail(scene):
    detection = QrQuadDetection(tuple(ImagePoint(*point) for point in QR.corners), 1.)
    with patch(f"{MODULE}.detect_qr_quad", return_value=detection), patch(
        f"{MODULE}.validate_qr_marker", return_value=QrMarkerEvidence(False, "qr_finder_patterns_unverified")):
        result, _ = evaluate(scene)
    assert result.debug.qr_detected is True
    assert result.debug.qr_marker_verified is False
    assert classified(scene, result).estimate.visible_face is None


def test_late_native_miss_cannot_become_backside_evidence(scene):
    clock = [10.]
    def late_detector(*args, **kwargs):
        clock[0] += 2.
        return None
    with patch(f"{MODULE}.detect_qr_quad", side_effect=late_detector):
        result, _ = evaluate(scene, clock=clock)
    assert result.debug.qr_detected is None and result.debug.qr_marker_verified is None
    assert result.debug.qr_marker_reason == "qr_marker_completion_deadline_exceeded"
    assert not result.qr_decode_metadata["acquisition"]["allowed"]
    assert classified(scene, result).estimate.visible_face is None


@pytest.mark.parametrize("fallback", (False, True))
def test_head_miss_decodes_only_bounded_fallback_without_claiming_absence(scene, fallback):
    estimate, debug = scene[4]
    miss = (replace(estimate, usable=False, corners=None, yaw_deg=None),
            replace(debug, head_model_quality=None, model_pose=None))
    decoder = Mock(return_value=(QR,))
    attempt = HeadRoiAttempt(ImageRoi(120, 150, 300, 400, scene[3]), "nominal_projection", 3., 200., 200., scene[3])
    with patch(f"{MODULE}.detect_qr_quad", side_effect=AssertionError("no complete head for absence check")):
        result, _ = evaluate(scene, native=decoder, fallback_attempt=attempt if fallback else None,
                             estimator=Mock(return_value=miss))
    if fallback:
        decoder.assert_called_once()
        assert decoder.call_args.args[0].shape == (250, 180, 3)
        assert result.qr_observations[0].corners[0] == (140., 170.)
    else:
        decoder.assert_not_called()
        assert result.qr_observations is None
        assert result.qr_decode_metadata["reason"] == "no_bounded_identity_crop"
    assert classified(scene, result).estimate.visible_face is None


@pytest.mark.parametrize("angle", (20., 45.))
def test_real_full_frame_fit_and_marker_decoration_solve_geometry_once(scene, angle):
    head, _ = fixtures.HeadBoundaryIndependenceTest().projection(angle_deg=angle)
    image = np.zeros_like(scene[0])
    cv2.fillConvexPoly(image, np.rint([(p.u_px, p.v_px) for p in head]).astype(np.int32), (200, 200, 200))
    actual_scene = (image, *scene[1:])
    from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import estimate_planar_pose_ippe
    with patch("scripts.aufgabe04.perception.stand_axis.head_model_fit.estimate_planar_pose_ippe",
               wraps=estimate_planar_pose_ippe) as solve:
        result, _ = evaluate(actual_scene, native=lambda crop: (QR,), estimator=None)
        side = classified(actual_scene, result)
    solve.assert_called_once()
    assert result.estimate.usable, result.estimate.reason
    assert result.estimate.yaw_deg == pytest.approx(-angle, abs=3.)
    assert side.estimate.corners == result.estimate.corners
    assert side.estimate.yaw_deg == result.estimate.yaw_deg
    assert admit_measured_head_model(estimate=side.estimate, debug=side.debug,
        yaw_rad=math.radians(side.estimate.yaw_deg)).accepted
