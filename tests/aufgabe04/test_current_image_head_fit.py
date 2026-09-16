"""Recovered QR changes semantics, never repeats or borrows head geometry."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.stand_axis.current_image_head_fit import CurrentImageHeadFit
from scripts.aufgabe04.perception.stand_axis.head_backside_classification import is_classified_measured_head_backside
from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.physical_head_pipeline import fit_physical_head_in_frame
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import PlanarPoseHypothesis
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.observer.qr_acquisition_policy import QrAcquisitionPolicy, evaluate_roi_with_qr_acquisition
from scripts.aufgabe04.real_robot.observer.qr_decode_cache import RoiQrDecodeCache
from tests.aufgabe04.test_head_backside_classification import classified_head
from tests.aufgabe04.test_physical_head_pipeline import head_image

PIPELINE = "scripts.aufgabe04.perception.stand_axis.model_pipeline"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def profile():
    return load_measured_physical_stand_model(
        ROOT / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")


def fit(profile, image, holder, observations=(), **changes):
    options = dict(model_profile=profile, camera_fx_px=640., camera_fy_px=640.,
        camera_cx_px=400., camera_cy_px=300., qr_observations=observations,
        blur_kernel=1, current_image_head_fit=holder)
    options.update(changes)
    return estimate_stand_axis_from_metric_model(cv2, image, **options)


def test_qr_acquisition_fits_geometry_once_and_charges_both_passes(profile):
    image, _ = head_image(profile)
    holder = CurrentImageHeadFit()
    qr = (DecodedQrObservation("QR_003", ((360., 260.), (440., 260.),
        (440., 340.), (360., 340.)), "recovered"),)
    budget = QrAcquisitionPolicy().begin_frame(target_key="candidate", image_stamp_sec=100.,
        started_ros_sec=100.01, started_monotonic_sec=10., max_sensor_age_sec=.5)
    outputs = []
    def estimate(observations):
        result = fit(profile, image, holder, observations)
        outputs.append(result)
        return result
    times = iter((10., 10.02, 10.02))
    with patch(f"{PIPELINE}.fit_physical_head_in_frame", wraps=fit_physical_head_in_frame) as geometry:
        result, debug, observations, metadata = evaluate_roi_with_qr_acquisition(
            frame=image, roi=(0, 0, 800, 600), roi_source="nominal_projection",
            cache=RoiQrDecodeCache(), budget=budget, native_decoder=lambda _: (),
            full_decoder=lambda *_: qr, estimate=estimate, now=lambda: next(times),
            current_image_head_fit=holder)
    assert geometry.call_count == 1
    assert len(outputs) == 2
    assert result.usable
    assert result.yaw_deg == outputs[0][0].yaw_deg
    assert result.corners == outputs[0][0].corners
    assert debug.qr_marker_verified
    assert observations == qr
    assert metadata["current_image_geometry_reused"]
    assert not metadata["current_image_geometry_refit"]
    assert debug.stage_timings_ms["initial_geometry_pass_ms"] == pytest.approx(20.)
    assert debug.stage_timings_ms["total"] >= 20.
    assert "edge_preprocessing" in debug.stage_timings_ms


def test_refresh_restores_undecorated_front_geometry_after_backside_classification(profile):
    raw, raw_debug, _ = classified_head(profile_sha256=profile.sha256)
    raw = replace(raw, camera_face_normal_xyz=(.6, 0., -.8),
        camera_face_center_xyz_m=(0., 0., .5))
    holder = CurrentImageHeadFit()
    image = np.zeros((200, 200, 3), np.uint8)
    options = dict(camera_cx_px=80., camera_cy_px=80., expected_head_center_u_px=80.,
        expected_head_center_v_px=80., expected_head_height_px=90.)
    with patch(f"{PIPELINE}.fit_physical_head_in_frame", return_value=(raw, raw_debug, None)) as geometry:
        backside, back_debug = fit(profile, image, holder, **options)
        qr = (DecodedQrObservation("QR_003", None, "recovered"),)
        front, front_debug = fit(profile, image, holder, qr, **options)
    assert is_classified_measured_head_backside(backside, back_debug)
    assert backside.camera_face_normal_xyz is None
    assert front.camera_face_normal_xyz == raw.camera_face_normal_xyz
    assert front.camera_face_center_xyz_m == raw.camera_face_center_xyz_m
    assert front.source == raw.source
    assert front_debug.qr_marker_verified
    assert not front_debug.head_backside_classification.accepted
    assert front_debug.head_backside_classification.reason == "backside_current_marker_absence_required"
    assert not raw_debug.qr_detected  # The retained undecorated tuple was not mutated.
    geometry.assert_called_once()


@pytest.mark.parametrize("change", ("new_image", "mutated_pixels", "intrinsics", "projection",
    "settings", "new_roi", "profile", "seed", "proposal", "proposal_verified"))
def test_different_geometry_context_cannot_reuse_even_identical_head_pixels(profile, change):
    image, corners = head_image(profile)
    holder = CurrentImageHeadFit()
    options = dict(current_head_proposal_corners=corners)
    with patch(f"{PIPELINE}.fit_physical_head_in_frame", wraps=fit_physical_head_in_frame) as geometry:
        fit(profile, image, holder, **options)
        if change == "new_image":
            image = image.copy()
        elif change == "mutated_pixels":
            image[0, 0] = 255
        elif change == "intrinsics":
            options["camera_cx_px"] = 401.
        elif change == "projection":
            options["expected_head_center_u_px"] = 400.
        elif change == "settings":
            options["canny_low"] = 21
        elif change == "new_roi":
            image = image[1:, 1:]
        elif change == "profile":
            profile = replace(profile, head_width_m=.079)
        elif change == "seed":
            options["pose_hint"] = PlanarPoseHypothesis((0., .4, 0.), (0., 0., .4),
                (0., 0., 1.), 25., .5, True)
        elif change == "proposal":
            options["current_head_proposal_corners"] = tuple(
                replace(point, u_px=point.u_px + 1.) for point in corners)
        elif change == "proposal_verified":
            options["current_head_proposal_verified"] = True
        fit(profile, image, holder, (DecodedQrObservation("QR_003", None, "recovered"),), **options)
    assert geometry.call_count == 2
    assert not holder.reused


def test_qr_refresh_cannot_promote_missing_head_geometry(profile):
    image = np.zeros((200, 200, 3), np.uint8)
    holder = CurrentImageHeadFit()
    with patch(f"{PIPELINE}.fit_physical_head_in_frame", wraps=fit_physical_head_in_frame) as geometry:
        missing, _ = fit(profile, image, holder)
        qr, debug = fit(profile, image, holder,
            (DecodedQrObservation("QR_003", None, "recovered"),))
    geometry.assert_called_once()
    assert holder.reused
    assert not missing.usable
    assert not qr.usable
    assert qr.yaw_deg is None
    assert qr.reason == missing.reason
    assert debug.qr_marker_verified


def test_proposal_verification_context_requires_boolean_to_avoid_equal_numeric_keys(profile):
    image, corners = head_image(profile)
    holder = CurrentImageHeadFit()
    # In Python True == 1. A non-boolean verification state must never alias
    # a verified cache context while skipping the producer's `is True` guard.
    with pytest.raises(ValueError, match="verification must be a boolean"):
        fit(profile, image, holder, current_head_proposal_corners=corners,
            current_head_proposal_verified=1)


def test_only_one_refresh_is_retained_and_geometry_exceptions_are_not_cached():
    image = np.zeros((10, 10, 3), np.uint8)
    holder = CurrentImageHeadFit()
    producer = Mock(side_effect=(ValueError("no geometry"), object(), object()))
    with pytest.raises(ValueError, match="no geometry"):
        holder.compute(image, context=(), producer=producer)
    first = holder.compute(image, context=(), producer=producer)
    assert holder.compute(image, context=(), producer=producer) is first
    assert holder.reused
    assert holder.compute(image, context=(), producer=producer) is not first
    assert not holder.reused
    assert producer.call_count == 3


def test_legacy_nonphysical_path_still_recomputes(profile):
    image = np.zeros((200, 200, 3), np.uint8)
    profile = replace(profile, environment="simulation")
    holder = CurrentImageHeadFit()
    with patch(f"{PIPELINE}._canny_edges_from_frame", return_value=image[:, :, 0]) as edges, \
         patch(f"{PIPELINE}.detect_qr_quad", return_value=None), \
         patch(f"{PIPELINE}.fit_physical_head_in_frame") as geometry:
        fit(profile, image, holder)
        fit(profile, image, holder, (DecodedQrObservation("QR_003", None, "recovered"),))
    assert edges.call_count == 2
    geometry.assert_not_called()
    assert not holder.reused
