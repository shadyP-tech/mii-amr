"""The physical viewer and observer share QR-free current-head geometry."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.camera_calibration import rectify_bgr_frame
from scripts.aufgabe04.perception.debug.viewer_model_overlay_policy import current_model_overlay_state
from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from tests.aufgabe04.recorded_head_proposal_fixture import (
    RECORDED_HEAD_PROPOSALS, recorded_head_proposal_image,
)

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = "scripts.aufgabe04.perception.stand_axis.model_pipeline"
COLD = "scripts.aufgabe04.perception.stand_axis.head_cold_acquisition.acquire_cold_head_proposal"


@pytest.fixture
def profile():
    return load_measured_physical_stand_model(
        ROOT / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")


def head_image(profile, angle=45., distance=.35):
    matrix = np.array(((640., 0., 400.), (0., 640., 300.), (0., 0., 1.)))
    points = np.array([(p.x_m, p.y_m, p.z_m) for p in profile.head_corners])
    pixels = cv2.projectPoints(points, np.array((0., math.radians(angle), 0.)),
        np.array((0., 0., distance)), matrix, np.zeros(4))[0].reshape(-1, 2)
    corners = tuple(ImagePoint(float(u), float(v)) for u, v in pixels)
    image = np.zeros((600, 800, 3), np.uint8)
    cv2.polylines(image, [np.rint(pixels).astype(np.int32)], True, (255, 255, 255), 1)
    return image, corners


def estimate(profile, image, **changes):
    options = dict(model_profile=profile, camera_fx_px=640., camera_fy_px=640.,
        camera_cx_px=400., camera_cy_px=300., qr_observations=(), blur_kernel=1)
    options.update(changes)
    # Any physical invocation of legacy QR or joint pose fitting is a regression.
    with patch(f"{PIPELINE}.estimate_planar_pose_ippe", side_effect=AssertionError("QR pose used")), \
         patch(f"{PIPELINE}.collect_metric_model_diagnostics", side_effect=AssertionError("joint fit used")):
        return estimate_stand_axis_from_metric_model(cv2, image, **options)


@pytest.mark.parametrize("angle", (20., 45., 60., 75.))
def test_cold_plain_head_reaches_3d_overlay_without_qr_neck_or_history(profile, angle):
    image, _ = head_image(profile, angle)
    fitted, debug = estimate(profile, image)
    assert fitted.usable, fitted.reason
    assert abs(fitted.yaw_deg + angle) < 3.
    assert fitted.source == "model_current_measured_head"
    assert fitted.visible_face is None  # No registered candidate or directed side proof.
    assert debug.head_neck_junction is None
    assert debug.head_acquisition_diagnostics["source"] == "cold_current_head_search"
    assert debug.head_acquisition_diagnostics["qr_used_for_geometry"] is False
    assert "head_back_top_left" in debug.projected_landmarks
    state = current_model_overlay_state(inputs_ready=True, estimate=fitted,
        artifacts=debug, result_fresh=True)
    assert state.current_fit_accepted
    assert state.geometry_color == (180, 0, 180)
    expired = current_model_overlay_state(inputs_ready=True, estimate=fitted,
        artifacts=debug, result_fresh=False)
    assert not expired.current_fit_accepted


def test_qr_identity_and_size_change_only_marker_evidence(profile):
    image, corners = head_image(profile)
    variants = [()]
    for scale in (.50, .95, 1.15):
        quad = tuple((400. + scale*(p.u_px-400.), 300. + scale*(p.v_px-300.)) for p in corners)
        variants.append((DecodedQrObservation("QR_003", quad, "test"),))
    variants.append(variants[-1] + (DecodedQrObservation("QR_004", None, "test"),))
    outputs = [estimate(profile, image, qr_observations=qr) for qr in variants]
    base, _ = outputs[0]
    assert base.usable, base.reason
    for index, (fitted, debug) in enumerate(outputs):
        assert fitted.usable
        assert fitted.yaw_deg == base.yaw_deg
        assert fitted.corners == base.corners
        assert fitted.visible_face is None
        assert debug.qr_marker_verified is (index > 0)
        assert debug.head_marker_boundary is None
        assert debug.model_diagnostics is None
    assert outputs[-1][1].qr_marker_reason == "multiple_decoded_qr_identities"


def test_qr_or_tracked_pose_cannot_replace_current_head_pixels(profile):
    image, _ = head_image(profile)
    fitted, debug = estimate(profile, image)
    assert fitted.usable
    for qr in ((), (DecodedQrObservation("QR_003", None, "test"),)):
        missing, missing_debug = estimate(profile, np.zeros_like(image),
            pose_hint=debug.model_pose, qr_observations=qr)
        assert not missing.usable
        assert missing.yaw_deg is None
        assert missing_debug.model_pose is None
        assert missing_debug.projected_landmarks is None


def test_named_candidate_cannot_escape_its_projection_via_cold_search(profile):
    image, _ = head_image(profile)
    with patch(COLD, side_effect=AssertionError("unassociated global search")):
        missing, debug = estimate(profile, image, expected_head_center_u_px=70.,
            expected_head_center_v_px=70., expected_head_height_px=60.)
    assert not missing.usable
    assert debug.head_acquisition_diagnostics["source"] == "candidate_projection"
    assert debug.head_acquisition_diagnostics["acquisition"]["proposal"] is None
    with patch(COLD, side_effect=AssertionError("incomplete candidate escaped")):
        incomplete, _ = estimate(profile, image, expected_head_center_u_px=70.)
    assert incomplete.reason == "head_candidate_projection_incomplete"


def test_named_candidate_uses_associated_hint_and_refits_crop_adjusted_current_pixels(profile):
    image, _ = head_image(profile, angle=45.)
    initial, initial_debug = estimate(profile, image)
    assert initial.usable
    current, _ = head_image(profile, angle=47.)
    x0, y0, x1, y1 = 260, 180, 540, 420
    # An associated camera pose remains camera-relative across changed crops.
    # Both acquisition alternatives are prohibited on this bounded current fit.
    with patch(COLD, side_effect=AssertionError("candidate escaped globally")), \
         patch("scripts.aufgabe04.perception.stand_axis.physical_head_pipeline.acquire_head_proposal",
               side_effect=AssertionError("candidate ignored associated pose")):
        fitted, debug = estimate(profile, current[y0:y1, x0:x1], pose_hint=initial_debug.model_pose,
            camera_cx_px=400.-x0, camera_cy_px=300.-y0,
            expected_head_center_u_px=400.-x0, expected_head_center_v_px=300.-y0,
            expected_head_height_px=140.)
    assert fitted.usable, fitted.reason
    assert abs(fitted.yaw_deg + 47.) < 3.
    assert fitted.corners != initial.corners
    assert debug.head_acquisition_diagnostics["source"] == "candidate_tracked_head_search"
    assert debug.head_acquisition_diagnostics["acquisition"] is None
    uncropped, _ = estimate(profile, current, pose_hint=initial_debug.model_pose)
    assert abs(fitted.yaw_deg - uncropped.yaw_deg) < 1e-3
    assert np.allclose([(p.u_px+x0, p.v_px+y0) for p in fitted.corners],
                       [(p.u_px, p.v_px) for p in uncropped.corners])


def test_named_tracked_candidate_head_loss_does_not_retry_acquisition(profile):
    image, _ = head_image(profile)
    initial, debug = estimate(profile, image)
    assert initial.usable
    with patch(COLD, side_effect=AssertionError("lost candidate escaped")), \
         patch("scripts.aufgabe04.perception.stand_axis.physical_head_pipeline.acquire_head_proposal",
               side_effect=AssertionError("lost candidate retried on same image")):
        missing, missing_debug = estimate(profile, np.zeros_like(image), pose_hint=debug.model_pose,
            expected_head_center_u_px=400., expected_head_center_v_px=300., expected_head_height_px=140.)
    assert not missing.usable
    assert missing.yaw_deg is None
    assert missing_debug.model_pose is None


def test_named_tracked_candidate_ambiguity_keeps_current_rejection(profile):
    image, corners = head_image(profile, angle=0., distance=.7)
    uncertain, debug = estimate(profile, image, current_head_proposal_corners=corners)
    assert not uncertain.usable
    with patch(COLD, side_effect=AssertionError("ambiguous candidate escaped")), \
         patch("scripts.aufgabe04.perception.stand_axis.physical_head_pipeline.acquire_head_proposal",
               side_effect=AssertionError("ambiguous angle tried another locator")):
        tracked, _ = estimate(profile, image, pose_hint=debug.head_pose_hypotheses[0],
            expected_head_center_u_px=400., expected_head_center_v_px=300., expected_head_height_px=70.)
    assert not tracked.usable
    assert tracked.yaw_deg is None


def test_off_center_tracked_backside_classifies_using_current_verified_head_center(profile):
    image, corners = head_image(profile, angle=45.)
    initial, initial_debug = estimate(profile, image)
    assert initial.usable
    # The map projection may be offset within the observer's 1.5-head
    # association window. Reapplying the .25-head classification crop gate to
    # that nominal point used to drop an already registered backside.
    expected = dict(expected_head_center_u_px=510., expected_head_center_v_px=300.,
                    expected_head_height_px=145.)
    nominal, _ = estimate(profile, image, current_head_proposal_corners=corners, **expected)
    assert nominal.usable
    assert nominal.visible_face is None
    tracked, debug = estimate(profile, image, pose_hint=initial_debug.model_pose, **expected)
    assert tracked.usable
    assert tracked.evidence_state == "fresh_backside"
    assert tracked.visible_face == "backside_candidate"
    assert debug.head_center_error_ratio == 0.
    assert debug.head_acquisition_diagnostics["side_projection_source"] == "current_verified_head_pixels"
    assert debug.head_acquisition_diagnostics["original_candidate_association_required"] is True
    assert debug.head_acquisition_diagnostics["side_projection_center_px"] == (
        sum(p.u_px for p in tracked.corners)/4, sum(p.v_px for p in tracked.corners)/4)
    assert abs(tracked.yaw_deg-initial.yaw_deg) < .1


@pytest.mark.parametrize("decoded", (False, True))
def test_off_center_current_front_marker_vetoes_backside_even_with_backside_hint(profile, decoded):
    image, corners = head_image(profile, angle=45.)
    seed, seed_debug = estimate(profile, image, expected_head_center_u_px=400.,
                                expected_head_center_v_px=300., expected_head_height_px=145.)
    assert seed.evidence_state == "fresh_backside"
    quad = tuple((400. + .65*(p.u_px-400.), 300. + .65*(p.v_px-300.)) for p in corners)
    if decoded:
        current, debug = estimate(profile, image, pose_hint=seed_debug.model_pose,
            expected_head_center_u_px=510., expected_head_center_v_px=300., expected_head_height_px=145.,
            qr_observations=(DecodedQrObservation("QR_003", quad, "test"),))
    else:
        # A current native quadrilateral alone also prevents committing absence;
        # the test deliberately grants no decoded identity or historical marker.
        from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import QrQuadDetection
        with patch(f"{PIPELINE}.detect_qr_quad", return_value=QrQuadDetection(
                tuple(ImagePoint(u, v) for u, v in quad), 1.)):
            current, debug = estimate(profile, image, pose_hint=seed_debug.model_pose,
                expected_head_center_u_px=510., expected_head_center_v_px=300., expected_head_height_px=145.)
    assert current.usable
    assert current.evidence_state == "fresh_refined"
    assert current.visible_face is None
    assert debug.qr_detected
    assert not debug.head_backside_classification.accepted


def test_verified_current_borders_with_uncertain_angle_cannot_use_cold_retry(profile):
    image, corners = head_image(profile, angle=0., distance=.7)
    uncertain, debug = estimate(profile, image, current_head_proposal_corners=corners)
    assert not uncertain.usable
    assert debug.head_model_quality.outer_border_verified
    # A previous pose may locate the head, never select a more convenient angle.
    pose = debug.head_pose_hypotheses[0]
    with patch(COLD, side_effect=AssertionError("ambiguous angle retried globally")):
        tracked, _ = estimate(profile, image, pose_hint=pose)
    assert not tracked.usable
    assert tracked.yaw_deg is None


def test_recorded_backside_cold_acquires_but_preserves_angle_uncertainty_gate(profile):
    fixtures = ROOT / "tests/aufgabe04/fixtures/qr_marker_validation"
    sample = json.loads((fixtures / "manifest.json").read_text())["samples"]["backside_000022"]
    data = (fixtures / sample["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == sample["source_compressed_sha256"]
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    image = rectify_bgr_frame(image, SimpleNamespace(**sample["recorded_camera_info"]), cv2, np)
    x0, y0, x1, y1 = (sample["roi"][key] for key in ("x0", "y0", "x1", "y1"))
    k = sample["recorded_camera_info"]["k"]
    fitted, debug = estimate(profile, image[y0:y1, x0:x1], blur_kernel=5,
        camera_fx_px=k[0], camera_fy_px=k[4], camera_cx_px=k[2]-x0, camera_cy_px=k[5]-y0)
    acquisition = debug.head_acquisition_diagnostics["acquisition"]
    assert acquisition["proposal"] is not None  # No QR/projection/previous fit required.
    assert debug.head_model_quality.outer_border_verified
    # This original small frontal head is an acquisition fixture, not ground truth.
    assert not fitted.usable
    assert fitted.reason == "head_model_yaw_uncertainty_too_high"
    assert debug.head_model_quality.yaw_std_deg > debug.head_model_quality.max_yaw_std_deg
    assert not current_model_overlay_state(inputs_ready=True, estimate=fitted,
        artifacts=debug, result_fresh=True).current_fit_accepted


def test_recorded_backside_with_full_border_margin_reaches_purple_3d_overlay(profile):
    # Original lossless rectified crop (260,260,435,390), not the tight nominal
    # crop above. The helper verifies its committed source-pixel hash.
    image = recorded_head_proposal_image(cv2, np, "back22")
    metadata = RECORDED_HEAD_PROPOSALS["back22"]
    fitted, debug = estimate(profile, image, blur_kernel=5,
        **{key: value for key, value in metadata.items() if key.startswith("camera_")})
    assert fitted.usable, fitted.reason
    assert -17. < fitted.yaw_deg < -12.  # Regression interval, not angle ground truth.
    assert debug.head_model_quality.yaw_std_deg < 3.
    assert debug.head_model_quality.reprojection_rmse_px < 2.
    assert debug.head_neck_junction is None
    assert not debug.qr_detected
    assert debug.head_acquisition_diagnostics["source"] == "cold_current_head_search"
    assert debug.projected_landmarks is not None
    assert current_model_overlay_state(inputs_ready=True, estimate=fitted,
        artifacts=debug, result_fresh=True).geometry_color == (180, 0, 180)
