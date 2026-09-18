"""Distance constrains search but cannot supply missing camera evidence."""

from dataclasses import replace
import math
from unittest.mock import Mock, patch

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.stand_axis.metric_head_search import projected_head_size, metric_head_search
from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import acquire_cold_head_proposal
from scripts.aufgabe04.perception.stand_axis.head_frame_detection import head_frame_detection
from scripts.aufgabe04.perception.stand_axis.head_geometry_acquisition import estimate_current_head_geometry
from tests.aufgabe04.test_physical_head_pipeline import profile, head_image, estimate
from tests.aufgabe04.test_head_cold_acquisition import frame_with_heads


def search(profile, **changes):
    options = dict(model_profile=profile, depth_m=.5, fx=640., fy=640.,
                   cx=400., cy=300., image_shape=(600, 800))
    options.update(changes)
    return metric_head_search(**options)


@pytest.mark.parametrize("depth,expected", ((.4, 124.8), (.5, 99.84), (.6, 83.2)))
def test_known_dimension_and_camera_depth_predict_pixel_height(profile, depth, expected):
    prior = search(profile, depth_m=depth).pixel_size
    assert prior.height_px == pytest.approx(expected)
    assert prior.frontal_width_px == pytest.approx(expected)
    assert prior.min_height_px < expected < prior.max_height_px


def test_distance_uncertainty_widens_bounds_and_unknown_position_keeps_full_image(profile):
    narrow, wide = search(profile), search(profile, depth_uncertainty_m=.07)
    assert wide.pixel_size.min_height_px < narrow.pixel_size.min_height_px
    assert wide.pixel_size.max_height_px > narrow.pixel_size.max_height_px
    assert narrow.image_bounds((600, 800)) == (0, 0, 800, 600)
    assert narrow.pixel_size.accepts(15., 100.)  # Oblique width is not frontal width.
    assert not narrow.pixel_size.accepts(100., 250.)
    assert not narrow.pixel_size.accepts(300., 100.)


@pytest.mark.parametrize("change", ({"depth_m": 0.}, {"depth_m": math.nan},
    {"depth_m": .02}, {"depth_uncertainty_m": -.1}, {"camera_vertical": (0., 2., 0.)}))
def test_invalid_metric_context_never_supplies_a_search(profile, change):
    with pytest.raises(ValueError):
        search(profile, **change)


@pytest.mark.parametrize("pitch", (-.3, 0., .3))
def test_projected_side_bounds_cover_yaw_and_depth_error_with_tilted_camera(profile, pitch):
    camera_rotation = cv2.Rodrigues(np.array((pitch, 0., 0.)))[0]
    vertical = camera_rotation @ np.array((0., 1., 0.))
    prior = projected_head_size(model_profile=profile, depth_m=.5, fx=640., fy=641.,
        center_normalized=(.2, -.1), camera_vertical=tuple(vertical))
    points = np.array([(p.x_m, p.y_m, p.z_m) for p in profile.head_corners])
    for yaw in np.linspace(-1.2, 1.2, 9):
        rotation = camera_rotation @ cv2.Rodrigues(np.array((0., yaw, 0.)))[0]
        for dz in (-.02, .02):
            camera_points = points @ rotation.T + np.array((.1, -.05, .5+dz))
            pixels = camera_points[:, :2]/camera_points[:, 2:]
            pixels *= (640., 641.)
            height = (np.linalg.norm(pixels[0]-pixels[3])+np.linalg.norm(pixels[1]-pixels[2]))/2
            width = (np.linalg.norm(pixels[0]-pixels[1])+np.linalg.norm(pixels[2]-pixels[3]))/2
            assert prior.accepts(width, height), (pitch, yaw, dz, prior)


def test_small_background_contours_do_not_spend_head_search_quota(profile):
    frame = frame_with_heads()
    for x in range(12, 340, 38):
        cv2.rectangle(frame, (x, 8), (x+30, 38), (255, 255, 255), 1)
    prior = search(profile, cx=190., cy=130., image_shape=frame.shape)
    with patch("scripts.aufgabe04.perception.stand_axis.head_cold_acquisition.MAX_CONTOURS", 8):
        unconstrained = acquire_cold_head_proposal(cv2, frame)
        constrained = acquire_cold_head_proposal(cv2, frame, candidate_search=prior)
    assert unconstrained.reason == "head_cold_acquisition_contour_budget_exceeded"
    assert constrained.proposal is not None, constrained
    assert constrained.joint_border_diagnostics["metric_contours_rejected"] > 8


def test_size_prior_does_not_invent_missing_borders(profile):
    frame = np.zeros((600, 800, 3), np.uint8)
    result, debug = estimate_current_head_geometry(cv2, frame, model_profile=profile,
        camera_fx_px=640., camera_fy_px=640., camera_cx_px=400., camera_cy_px=300.,
        candidate_search=search(profile))
    assert result.corners is None and not result.usable
    assert not head_frame_detection(result, debug)["head_frame_detected"]


def test_tracked_measurement_must_still_match_current_depth(profile):
    image, _ = head_image(profile, distance=.35)
    fitted, debug = estimate(profile, image)
    assert fitted.usable
    result, artifacts = estimate_current_head_geometry(cv2, image, model_profile=profile,
        camera_fx_px=640., camera_fy_px=640., camera_cx_px=400., camera_cy_px=300.,
        candidate_search=search(profile, depth_m=1.), pose_hint=debug.model_pose,
        estimator=Mock(return_value=(fitted, debug)))
    assert result.reason == "head_border_outside_metric_pixel_bounds"
    assert result.corners is None and artifacts.head_outer_recovery is None
    assert not head_frame_detection(result, artifacts)["head_frame_detected"]


def test_verified_frame_is_reported_when_yaw_is_uncertain(profile):
    image, _ = head_image(profile)
    fitted, debug = estimate(profile, image)
    uncertain = replace(fitted, usable=False, yaw_deg=None, reason="head_pose_ambiguous")
    detection = head_frame_detection(uncertain, debug)
    assert detection["head_frame_detected"]
    assert not detection["yaw_reliable"] and not detection["motion_authorized"]


def test_debug_viewer_explicit_depth_requires_metric_profile_and_unique_target():
    from scripts.aufgabe04.perception.debug.stand_axis_viewer import build_parser, _validate_runtime_args
    parser = build_parser()
    args = parser.parse_args(["--compressed-image-topic", "/camera/image/compressed", "--head-depth-m", ".4"])
    with pytest.raises(ValueError, match="head-target unique"):
        _validate_runtime_args(args)
    args.head_target = "unique"
    with pytest.raises(ValueError, match="stand-model-profile"):
        _validate_runtime_args(args)
    args.stand_model_profile = "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"
    _validate_runtime_args(args)
    args.head_depth_m = math.nan
    with pytest.raises(ValueError, match="finite and positive"):
        _validate_runtime_args(args)
