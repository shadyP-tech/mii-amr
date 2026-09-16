"""Geometry and QR progress independently without borrowing an old angle."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.stand_axis.current_image_head_fit import CurrentImageHeadFit
from scripts.aufgabe04.perception.stand_axis.models import StandAxisEdgeDebugArtifacts
from scripts.aufgabe04.perception.stand_axis.physical_head_pipeline import fit_physical_head_in_frame
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.observer.camera_target_registration import (
    CameraTargetRegistrationSelection, HeadRoiEvaluation,
)
from scripts.aufgabe04.real_robot.observer.independent_qr_acquisition import (
    evaluate_geometry_then_identity, probe_identity_after_head_miss,
)
from scripts.aufgabe04.real_robot.observer.qr_acquisition_policy import QrAcquisitionPolicy
from scripts.aufgabe04.real_robot.observer.qr_decode_cache import RoiQrDecodeCache
from tests.aufgabe04.test_current_image_head_fit import profile, fit, PIPELINE
from tests.aufgabe04.test_physical_head_pipeline import head_image


def budget(policy=None, age=.01):
    return (policy or QrAcquisitionPolicy()).begin_frame(
        target_key="candidate", image_stamp_sec=100., started_ros_sec=100. + age,
        started_monotonic_sec=10., max_sensor_age_sec=.5)


def test_geometry_precedes_decoders_and_exact_fit_is_used_once(profile):
    image, corners = head_image(profile)
    holder, order = CurrentImageHeadFit(), []
    qr = (DecodedQrObservation("Start", ((360., 260.), (440., 260.),
        (440., 340.), (360., 340.)), "recovered"),)
    def geometry():
        order.append("geometry")
        return fit(profile, image, holder, None, qr_marker_policy="disabled",
                   current_head_proposal_corners=corners)
    def native(_):
        order.append("native")
        return ()
    def full(*_):
        order.append("full")
        return qr
    def decorate(observations):
        order.append("decorate")
        return fit(profile, image, holder, observations,
                   current_head_proposal_corners=corners)
    with patch(f"{PIPELINE}.fit_physical_head_in_frame", wraps=fit_physical_head_in_frame) as fitter:
        result, debug, observations, metadata = evaluate_geometry_then_identity(
            frame=image, roi=(0, 0, 800, 600), roi_source="registered_head",
            cache=RoiQrDecodeCache(), budget=budget(), native_decoder=native,
            full_decoder=full, geometry_only=geometry, decorate=decorate,
            now=lambda: 10.01, current_image_head_fit=holder)
    assert order == ["geometry", "native", "full", "decorate"]
    assert fitter.call_count == 1
    assert result.usable and debug.qr_marker_verified
    assert observations == qr
    assert metadata["current_image_geometry_reused"]
    assert not metadata["current_image_geometry_refit"]


def test_expired_identity_budget_returns_geometry_with_unknown_side():
    estimate = SimpleNamespace(usable=True)
    debug = StandAxisEdgeDebugArtifacts(edges=None, qr_detected=None, qr_marker_verified=None)
    native, full, decorate = Mock(), Mock(), Mock()
    result = evaluate_geometry_then_identity(
        frame=object(), roi=(0, 0, 100, 100), roi_source="nominal_projection",
        cache=RoiQrDecodeCache(), budget=budget(age=.44), native_decoder=native,
        full_decoder=full, geometry_only=lambda: (estimate, debug), decorate=decorate,
        now=lambda: 10.02, current_image_head_fit=CurrentImageHeadFit())
    assert result[0] is estimate and result[1].qr_detected is None
    assert result[2] is None and not result[3]["performed"]
    native.assert_not_called()
    full.assert_not_called()
    decorate.assert_not_called()


@pytest.mark.parametrize("observations", [(), (DecodedQrObservation("Start", None, "text_only"),)])
def test_head_miss_can_decode_but_never_claims_marker_absence(observations):
    attempt = SimpleNamespace(roi=SimpleNamespace(x0=5, y0=6, x1=105, y1=106), source="expanded")
    current = HeadRoiEvaluation(attempt, object(), SimpleNamespace(usable=False),
        StandAxisEdgeDebugArtifacts(edges=None), qr_observations=None,
        qr_decode_metadata={"performed": False, "reason": "head_proposal_unavailable"})
    selected = CameraTargetRegistrationSelection(current, (current,), None, None, None, None)
    decoder = Mock(return_value=observations)
    result = probe_identity_after_head_miss(selected, cache=RoiQrDecodeCache(),
        budget=budget(), full_decoder=decoder, now=lambda: 10.02)
    decoder.assert_called_once()
    assert result.selected.estimate is current.estimate
    assert not result.selected.estimate.usable
    assert result.selected.qr_observations == observations
    assert result.selected.debug.qr_detected is (True if observations else None)
    assert result.selected is result.evaluations[0]


def test_repeated_head_miss_reserves_probe_time_without_extending_sensor_deadline():
    value = budget()
    assert value.head_deadline_with_identity_reserve(
        now_monotonic_sec=10.01, previous_head_miss=True) == pytest.approx(10.36)
    assert value.head_deadline_with_identity_reserve(
        now_monotonic_sec=10.01, previous_head_miss=False) == pytest.approx(10.44)
    value.request(roi=(0, 0, 100, 100), roi_source="expanded", now_monotonic_sec=10.36,
                  current_qr_signal=False, identity_geometry_available=False, selected_crop=True)
    assert value.head_deadline_with_identity_reserve(
        now_monotonic_sec=10.37, previous_head_miss=True) == pytest.approx(10.44)


def test_head_miss_probe_reuses_exact_result_even_when_native_cache_is_full():
    attempt = SimpleNamespace(roi=SimpleNamespace(x0=5, y0=6, x1=105, y1=106), source="expanded")
    current = HeadRoiEvaluation(attempt, object(), SimpleNamespace(usable=False),
        StandAxisEdgeDebugArtifacts(edges=None), qr_observations=None,
        qr_decode_metadata={"performed": False, "reason": "head_proposal_unavailable"})
    selection = CameraTargetRegistrationSelection(current, (current,), None, None, None, None)
    cache, work, decoder = RoiQrDecodeCache(), budget(), Mock(return_value=())
    for x in (0, 10, 20):
        cache.decode(roi=(x, 0, x+100, 100), mode="native", frame=object(), decoder=lambda _: ())
    for _ in range(2):
        result = probe_identity_after_head_miss(selection, cache=cache, budget=work,
            full_decoder=decoder, now=lambda: 10.02)
    decoder.assert_called_once()
    assert result.selected.qr_decode_metadata["cache_hit"]
