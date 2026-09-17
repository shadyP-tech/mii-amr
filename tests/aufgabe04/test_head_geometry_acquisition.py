"""Full-image acquisition is shared without candidate or QR pose seeds."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from scripts.aufgabe04.perception.stand_axis.head_geometry_acquisition import (
    DEFAULT_MIN_EDGE_HEIGHT_PX,
    create_head_geometry_tracker,
    estimate_current_head_geometry,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation


MODULE = "scripts.aufgabe04.perception.stand_axis.head_geometry_acquisition"
PIPELINE = "scripts.aufgabe04.perception.stand_axis.model_pipeline"
ROOT = Path(__file__).resolve().parents[2]
CAMERA = dict(camera_fx_px=640., camera_fy_px=641., camera_cx_px=400., camera_cy_px=300.)


@pytest.mark.parametrize("hint", (None, object()))
def test_cold_and_tracked_calls_use_full_image_without_preselected_candidate(hint):
    frame, cv2, profile = object(), object(), object()
    result = object()
    fit = Mock(return_value=result)
    with patch(f"{MODULE}.estimate_stand_axis_from_metric_model", fit):
        actual = estimate_current_head_geometry(cv2, frame, model_profile=profile,
            pose_hint=hint, deadline_monotonic_sec=42., **CAMERA)
    assert actual is result
    assert fit.call_args.args == (cv2, frame)
    kwargs = fit.call_args.kwargs
    assert kwargs["pose_hint"] is hint
    assert {key: kwargs[key] for key in CAMERA} == CAMERA
    assert kwargs["min_edge_height_px"] == DEFAULT_MIN_EDGE_HEIGHT_PX == 8.
    assert kwargs["qr_marker_policy"] == "disabled"
    assert kwargs["deadline_monotonic_sec"] == 42.
    assert kwargs["edge_preprocess"] == "channel_union"
    assert (kwargs["blur_kernel"], kwargs["canny_low"], kwargs["canny_high"]) == (5, 20, 60)
    assert not any(key.startswith("expected_head_") for key in kwargs)
    assert "current_head_proposal_corners" not in kwargs
    assert "current_head_proposal_verified" not in kwargs
    assert "current_head_refinement" not in kwargs


def test_same_image_qr_refresh_forwards_caches_and_optional_marker_policy():
    frame, cache, holder = object(), object(), object()
    qr = (DecodedQrObservation("QR_003", ((10., 10.), (20., 10.), (20., 20.), (10., 20.)), "test"),)
    fit = Mock(return_value=(object(), object()))
    actual = estimate_current_head_geometry(object(), frame, model_profile=object(),
        input_cache=cache, input_cache_roi=(0, 0, 800, 600),
        current_image_head_fit=holder, qr_observations=qr,
        qr_marker_policy="supplied_only", estimator=fit, **CAMERA)
    assert actual is fit.return_value
    assert fit.call_args.args[1] is frame
    kwargs = fit.call_args.kwargs
    assert kwargs["input_cache"] is cache
    assert kwargs["input_cache_roi"] == (0, 0, 800, 600)
    assert kwargs["current_image_head_fit"] is holder
    assert kwargs["qr_observations"] is qr
    assert kwargs["qr_marker_policy"] == "supplied_only"


def test_tracker_keeps_only_a_bounded_search_hint_and_clears_after_three_misses():
    tracker = create_head_geometry_tracker()
    assert tracker.prediction_ttl_sec == .25
    assert tracker.search_hint_ttl_sec == 2.
    assert tracker.max_soft_misses == 2
    pose, signature = object(), tuple(CAMERA.values())
    tracker.accept(pose, now_sec=10., profile_sha256="profile", camera_signature=signature)
    for index in range(3):
        observed = 10.4 + index * .4
        update = tracker.update_from_observation(None, None, observed_at_sec=observed,
            completed_at_sec=observed + .05, profile_sha256="profile", camera_signature=signature)
        assert not update.accepted
        prediction = tracker.prediction(now_sec=observed + .1,
            profile_sha256="profile", camera_signature=signature)
        if index < 2:
            assert prediction.pose is pose
            assert prediction.state == "predicted_only"
            assert prediction.age_sec == pytest.approx(observed + .1 - 10.)
        else:
            assert prediction.pose is None
    tracker.accept(pose, now_sec=20., profile_sha256="profile", camera_signature=signature)
    assert tracker.prediction(now_sec=22.01, profile_sha256="profile",
        camera_signature=signature).pose is None


def test_current_physical_pixels_determine_pose_with_or_without_qr_or_history():
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    from tests.aufgabe04.test_physical_head_pipeline import head_image

    profile = load_measured_physical_stand_model(
        ROOT / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
    frame, corners = head_image(profile, angle=45.)
    options = dict(model_profile=profile, camera_fx_px=640., camera_fy_px=640.,
        camera_cx_px=400., camera_cy_px=300., blur_kernel=1)
    qr = (DecodedQrObservation("QR_003",
        tuple((p.u_px, p.v_px) for p in corners), "test"),)
    # Neither an identity nor a prediction may replace the current head fit.
    with patch(f"{PIPELINE}.estimate_planar_pose_ippe", side_effect=AssertionError("QR pose used")), \
         patch(f"{PIPELINE}.collect_metric_model_diagnostics", side_effect=AssertionError("joint fit used")):
        cold, cold_debug = estimate_current_head_geometry(cv2, frame, **options)
        decoded, decoded_debug = estimate_current_head_geometry(cv2, frame,
            qr_observations=qr, qr_marker_policy="supplied_only", **options)
        tracked, tracked_debug = estimate_current_head_geometry(cv2, frame,
            pose_hint=cold_debug.model_pose, **options)
        missing, missing_debug = estimate_current_head_geometry(cv2, numpy.zeros_like(frame),
            pose_hint=cold_debug.model_pose, qr_observations=qr, **options)
    assert cold.usable, cold.reason
    assert decoded.usable, decoded.reason
    assert tracked.usable, tracked.reason
    assert cold_debug.head_acquisition_diagnostics["source"] == "cold_current_head_search"
    assert tracked_debug.head_acquisition_diagnostics["source"] == "tracked_head_search"
    assert decoded.yaw_deg == cold.yaw_deg
    assert decoded.corners == cold.corners
    assert decoded_debug.qr_marker_verified
    assert abs(tracked.yaw_deg - cold.yaw_deg) < 3.
    assert not missing.usable
    assert missing.yaw_deg is None
    assert missing_debug.model_pose is None
