"""Geometry and marker work have independent evidence and timing contracts."""

from unittest.mock import patch

import pytest

from scripts.aufgabe04.perception.stand_axis.marker_work_schedule import schedule_marker_work


def decision(**changes):
    options = dict(policy="auto", positive_observations=False, physical_head=True,
                   head_available=True, now_monotonic_sec=10., deadline_monotonic_sec=10.2)
    return schedule_marker_work(**{**options, **changes})


@pytest.mark.parametrize("changes,reason", [
    ({"policy": "disabled"}, "qr_marker_checks_disabled"),
    ({"policy": "supplied_only"}, "qr_marker_supplied_evidence_unavailable"),
    ({"head_available": False}, "qr_marker_head_geometry_unavailable"),
    ({"deadline_monotonic_sec": 10.03}, "qr_marker_processing_budget_exhausted"),
])
def test_unchecked_marker_absence_remains_explicit(changes, reason):
    result = decision(**changes)
    assert result.action == "skip"
    assert result.reason == reason


def test_positive_identity_is_used_without_native_decode_even_after_acquisition_miss():
    assert decision(positive_observations=True, head_available=False).action == "supplied"
    assert decision(policy="supplied_only", positive_observations=True).action == "supplied"
    assert decision().action == "native"
    with pytest.raises(ValueError):
        decision(policy="unchecked_is_backside")


def test_disabled_marker_work_keeps_current_3d_fit_without_backside_commit():
    pytest.importorskip("cv2")
    from tests.aufgabe04.test_physical_head_pipeline import head_image, estimate, ROOT, PIPELINE
    from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
    profile = load_measured_physical_stand_model(
        ROOT / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
    image, corners = head_image(profile)
    expected = dict(current_head_proposal_corners=corners, expected_head_center_u_px=400.,
                    expected_head_center_v_px=300., expected_head_height_px=145.)
    with patch(f"{PIPELINE}.detect_qr_quad", side_effect=AssertionError("hidden native QR work")):
        fitted, debug = estimate(profile, image, qr_marker_policy="disabled", **expected)
    assert fitted.usable
    assert fitted.visible_face is None
    assert debug.qr_marker_verified is None
    assert debug.qr_marker_reason == "qr_marker_checks_disabled"
    assert not debug.head_backside_classification.accepted
    assert debug.projected_landmarks is not None
    with patch(f"{PIPELINE}.detect_qr_quad", return_value=None):
        backside, checked = estimate(profile, image, **expected)
    assert backside.visible_face == "backside_candidate"
    assert checked.qr_marker_verified is False
    assert fitted.yaw_deg == backside.yaw_deg


def test_head_miss_and_supplied_identity_never_run_native_marker_detector():
    pytest.importorskip("cv2")
    import numpy as np
    from tests.aufgabe04.test_physical_head_pipeline import head_image, estimate, ROOT, PIPELINE
    from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
    from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
    profile = load_measured_physical_stand_model(
        ROOT / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
    image, corners = head_image(profile)
    with patch(f"{PIPELINE}.detect_qr_quad", side_effect=AssertionError("hidden native QR work")):
        missing, debug = estimate(profile, np.zeros_like(image))
        fitted, supplied = estimate(profile, image, current_head_proposal_corners=corners,
            qr_observations=(DecodedQrObservation("Start", tuple((p.u_px, p.v_px) for p in corners), "test"),))
    assert not missing.usable
    assert debug.qr_marker_verified is None
    assert debug.qr_marker_reason == "qr_marker_head_geometry_unavailable"
    assert fitted.usable
    assert supplied.qr_marker_verified is True
    assert supplied.qr_marker_reason == "decoded_qr_identity"


def test_physical_native_marker_check_never_calls_payload_decoder():
    from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import _detect_qr_quad_corners_native
    from types import SimpleNamespace
    from unittest.mock import Mock
    detector = SimpleNamespace(detectMulti=Mock(return_value=(False, None)),
        detect=Mock(return_value=(False, None)),
        detectAndDecodeMulti=Mock(side_effect=AssertionError("hidden payload decode")))
    cv2 = SimpleNamespace(QRCodeDetector=lambda: detector)
    assert _detect_qr_quad_corners_native(cv2, object(), allow_decode_fallback=False) is None
    detector.detectAndDecodeMulti.assert_not_called()


def test_geometry_only_then_marker_decoration_reuses_same_current_fit_once():
    pytest.importorskip("cv2")
    from tests.aufgabe04.test_physical_head_pipeline import head_image, estimate, ROOT, PIPELINE
    from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
    from scripts.aufgabe04.perception.stand_axis.current_image_head_fit import CurrentImageHeadFit
    from scripts.aufgabe04.perception.stand_axis.physical_head_pipeline import fit_physical_head_in_frame
    profile = load_measured_physical_stand_model(
        ROOT / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
    image, corners = head_image(profile)
    holder = CurrentImageHeadFit()
    expected = dict(current_head_proposal_corners=corners, current_image_head_fit=holder,
        expected_head_center_u_px=400., expected_head_center_v_px=300., expected_head_height_px=145.)
    with patch(f"{PIPELINE}.fit_physical_head_in_frame", wraps=fit_physical_head_in_frame) as fit, \
         patch(f"{PIPELINE}.detect_qr_quad", return_value=None) as native:
        geometry, unchecked = estimate(profile, image, qr_marker_policy="disabled", **expected)
        native.assert_not_called()
        decorated, checked = estimate(profile, image, qr_marker_policy="auto", **expected)
    assert fit.call_count == 1
    assert native.call_count == 1
    assert holder.reused
    assert geometry.corners == decorated.corners
    assert geometry.yaw_deg == decorated.yaw_deg
    assert unchecked.qr_marker_verified is None
    assert checked.qr_marker_verified is False
    assert decorated.visible_face == "backside_candidate"


def test_native_overrun_does_not_supply_backside_absence():
    pytest.importorskip("cv2")
    from tests.aufgabe04.test_physical_head_pipeline import head_image, estimate, ROOT, PIPELINE
    from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
    profile = load_measured_physical_stand_model(
        ROOT / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
    image, corners = head_image(profile)
    initial, debug = estimate(profile, image, current_head_proposal_corners=corners,
                              qr_marker_policy="disabled")
    clock = [10.]
    def late_miss(*args, **kwargs):
        clock[0] = 10.3
        return None
    with patch(f"{PIPELINE}.fit_physical_head_in_frame", return_value=(initial, debug, None)), \
         patch(f"{PIPELINE}.time.monotonic", side_effect=lambda: clock[0]), \
         patch(f"{PIPELINE}.detect_qr_quad", side_effect=late_miss):
        current, artifacts = estimate(profile, image, deadline_monotonic_sec=10.2,
            expected_head_center_u_px=400., expected_head_center_v_px=300., expected_head_height_px=145.)
    assert current.usable  # Angle freshness is still checked by its consumer.
    assert current.visible_face is None
    assert artifacts.qr_marker_verified is None
    assert artifacts.qr_marker_reason == "qr_marker_completion_deadline_exceeded"


def test_complete_head_with_uncertain_angle_still_gets_independent_marker_checks():
    pytest.importorskip("cv2")
    from tests.aufgabe04.test_physical_head_pipeline import head_image, estimate, ROOT, PIPELINE
    from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
    profile = load_measured_physical_stand_model(
        ROOT / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
    image, corners = head_image(profile, angle=0., distance=.7)
    with patch(f"{PIPELINE}.detect_qr_quad", return_value=None) as marker:
        uncertain, debug = estimate(profile, image, current_head_proposal_corners=corners,
            expected_head_center_u_px=400., expected_head_center_v_px=300., expected_head_height_px=70.)
    assert not uncertain.usable
    assert uncertain.yaw_deg is None
    assert debug.head_model_quality.outer_border_verified
    marker.assert_called_once()
    assert debug.qr_marker_verified is False
    assert debug.head_backside_appearance.accepted
    assert not debug.head_backside_classification.accepted
