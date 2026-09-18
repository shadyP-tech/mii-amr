"""Geometry bounds, recorded clutter and colour hints without invented pixels."""
import hashlib
import json
import math
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.camera_calibration import CameraCalibration, rectify_bgr_frame, rectified_source_support
from scripts.aufgabe04.perception.stand_axis.head_geometry_acquisition import estimate_current_head_geometry
from scripts.aufgabe04.perception.stand_axis.head_frame_detection import head_frame_detection
from scripts.aufgabe04.perception.stand_axis.image_source_support import ImageSourceSupport
from scripts.aufgabe04.perception.stand_axis.metric_head_search import metric_head_search, projected_head_size
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector
from scripts.aufgabe04.perception.stand_axis_handoff.models import RigidTransform
from tests.aufgabe04.test_physical_head_pipeline import profile, head_image


def prior(profile, **changes):
    options = dict(model_profile=profile, depth_m=.5, fx=640., fy=641., cx=400., cy=300.,
        image_shape=(600, 800), center=(400., 300.), position_uncertainty_m=.02)
    options.update(changes)
    return metric_head_search(**options)


@pytest.mark.parametrize("pitch", (-.5, -.1, 0., .1, .5))
def test_correlated_bounds_cover_physical_projection_with_position_and_depth_error(profile, pitch):
    camera = cv2.Rodrigues(np.array((pitch, 0., 0.)))[0]
    vertical = tuple(camera @ np.array((0., 1., 0.)))
    vertices = np.array([(p.x_m, p.y_m, p.z_m) for p in profile.head_corners])
    for depth in (.25, .4, .8):
        search = prior(profile, depth_m=depth, center=(460., 260.), camera_vertical=vertical)
        point = np.array((60.*depth/640., -40.*depth/641., depth))
        for yaw in np.linspace(-1.3, 1.3, 11):
            rotation = camera @ cv2.Rodrigues(np.array((0., yaw, 0.)))[0]
            for error in ((-.02, -.02, -.02), (.02, .02, .02), (.02, -.02, -.02)):
                xyz = vertices @ rotation.T + point + error
                pixels = xyz[:, :2]/xyz[:, 2:] * (640., 641.) + (400., 300.)
                corners = tuple(ImagePoint(*map(float, p)) for p in pixels)
                lengths = np.linalg.norm(np.roll(pixels, -1, axis=0)-pixels, axis=1)
                assert search.pixel_size.accepts((lengths[0]+lengths[2])/2., (lengths[1]+lengths[3])/2.)
                assert all(search.pixel_size.accepts((lengths[0]+lengths[2])/2., side)
                           for side in (lengths[1], lengths[3]))
                assert search.edge_region.contains(corners)
                assert search.accepts_center(search.projected_center(corners))


def test_small_camera_tilt_no_longer_doubles_allowed_head_height(profile):
    v = (0., math.cos(.1), math.sin(.1))
    size = projected_head_size(model_profile=profile, depth_m=.364, fx=641., fy=641., camera_vertical=v)
    assert size.min_height_px > 100.
    assert size.max_height_px < 185.


def test_depth_only_does_not_claim_tight_position_uncertainty(profile):
    search = prior(profile, position_uncertainty_m=None)
    assert search.center_bounds_px is None and search.edge_region is None
    assert search.accepts_center((460., 340.))
    assert not prior(profile).accepts_center((500., 300.))


def test_short_side_cannot_hide_behind_plausible_average_height(profile):
    search = prior(profile, depth_m=.364)
    corners = tuple(ImagePoint(x, y) for x, y in ((333., 230.), (467., 230.), (467., 330.), (333., 364.)))
    assert search.pixel_size.accepts(134., 117.)
    assert not search.accepts_measurement(corners)


def test_position_error_expands_tilted_camera_size_bounds(profile):
    options = dict(camera_vertical=(0., math.cos(.5), math.sin(.5)))
    narrow, wide = prior(profile, **options), prior(profile, position_uncertainty_m=.15, **options)
    assert wide.pixel_size.min_height_px < narrow.pixel_size.min_height_px
    assert wide.pixel_size.max_height_px > narrow.pixel_size.max_height_px


@pytest.mark.parametrize("error", (-.01, math.inf, math.nan))
def test_invalid_position_uncertainty_is_rejected(profile, error):
    with pytest.raises(ValueError):
        prior(profile, position_uncertainty_m=error)


@pytest.fixture
def recorded(profile):
    directory = Path(__file__).parent/'fixtures/head_border_clutter_20260918'
    data = json.loads((directory/'inputs.json').read_text())
    source = directory/'source_000003.png'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == data['source_sha256']
    cal = CameraCalibration(**data['calibration']['calibration'])
    tf = RigidTransform(**data['calibration']['base_from_camera'])
    x, y, z, w = tf.rotation_xyzw
    vertical = rotate_vector((0., 0., 1.), (-x, -y, -z, w))
    frame = rectify_bgr_frame(cv2.imread(str(source)), cal, cv2, np)
    s = data['recorded_search']
    search = prior(profile, depth_m=s['pixel_size']['depth_m'], fx=cal.fx_px, fy=cal.fy_px,
        cx=cal.cx_px, cy=cal.cy_px, center=(s['center_u_px'], s['center_v_px']),
        camera_vertical=vertical, max_center_offset_ratio=.70)
    support = ImageSourceSupport(cv2, rectified_source_support(cal, cv2, np), blur_kernel=5)
    return frame, cal, search, support, data


@pytest.mark.parametrize("colour", (False, True))
def test_recorded_clutter_preserves_border_ambiguity_and_measurement_pixels(profile, recorded, colour):
    frame, cal, search, support, data = recorded
    original = frame.copy()
    estimate, debug = estimate_current_head_geometry(cv2, frame, model_profile=profile,
        camera_fx_px=cal.fx_px, camera_fy_px=cal.fy_px, camera_cx_px=cal.cx_px, camera_cy_px=cal.cy_px,
        candidate_search=search, source_support=support, use_color_prior=colour)
    detection = head_frame_detection(estimate, debug)
    # Outer and inset rails remain plausible in this frame. Do not count
    # removal of a competing valid fit as successful outer-border detection.
    assert not detection['head_frame_detected']
    assert estimate.reason == 'head_proposal_ambiguous'
    np.testing.assert_array_equal(frame, original)
    expected_edges = _canny_edges_from_frame(cv2, frame, edge_preprocess='channel_union',
        blur_kernel=5, canny_low=20, canny_high=60)
    np.testing.assert_array_equal(debug.raw_edges, expected_edges)
    acquisition = debug.head_acquisition_diagnostics['acquisition']
    assert acquisition['raw_verifications'] <= 12
    assert acquisition['joint_border_diagnostics']['unverified_independent_hypotheses'] == 0
    assert not detection['motion_authorized']


@pytest.mark.parametrize("fill", (0, 255))
def test_colour_mask_cannot_supply_edges_on_blank_image(profile, fill):
    frame = np.zeros((600, 800, 3), np.uint8)
    mask = np.full(frame.shape[:2], fill, np.uint8)
    estimate, debug = estimate_current_head_geometry(cv2, frame, model_profile=profile,
        camera_fx_px=640., camera_fy_px=641., camera_cx_px=400., camera_cy_px=300.,
        candidate_search=prior(profile), color_support_mask=mask)
    assert not head_frame_detection(estimate, debug)['head_frame_detected']
    assert not np.any(debug.raw_edges)


def test_empty_colour_mask_does_not_erase_an_uncoloured_physical_head(profile):
    frame, _ = head_image(profile, angle=45., distance=.5)
    estimate, debug = estimate_current_head_geometry(cv2, frame, model_profile=profile,
        camera_fx_px=640., camera_fy_px=641., camera_cx_px=400., camera_cy_px=300.,
        candidate_search=prior(profile), color_support_mask=np.zeros(frame.shape[:2], np.uint8),
        blur_kernel=1)
    assert head_frame_detection(estimate, debug)['head_frame_detected'], estimate.reason


def test_two_independent_heads_remain_ambiguous_with_colour_prior(profile):
    frame = np.zeros((600, 800, 3), np.uint8)
    for x in (275, 425):
        cv2.rectangle(frame, (x, 245), (x+90, 335), (180, 180, 180), 2)
    estimate, debug = estimate_current_head_geometry(cv2, frame, model_profile=profile,
        camera_fx_px=640., camera_fy_px=641., camera_cx_px=400., camera_cy_px=300.,
        candidate_search=prior(profile, position_uncertainty_m=.15, depth_uncertainty_m=.05),
        color_support_mask=np.full(frame.shape[:2], 255, np.uint8), blur_kernel=1)
    assert not head_frame_detection(estimate, debug)['head_frame_detected']
    assert not estimate.usable


def test_colour_breaks_equal_rail_tie_using_only_observed_pixels():
    from scripts.aufgabe04.perception.stand_axis.metric_edge_association import coherent_metric_rail_points
    xs = np.arange(20., 120.)
    points = np.concatenate([np.column_stack((xs, np.full(100, y))) for y in (47., 53.)])
    bins = np.tile(np.arange(100), 2)
    mask = np.zeros((100, 150), np.uint8)
    mask[53, 20:120] = 255
    options = dict(band_px=5., expected_length_px=100., minimum_coverage=.8)
    args = (cv2, points, bins, ImagePoint(20., 50.), ImagePoint(120., 50.))
    assert len(coherent_metric_rail_points(*args, **options)) == 0
    selected = coherent_metric_rail_points(*args, color_support_mask=mask, **options)
    assert len(selected) == 100
    assert np.all(selected[:, 1] == 53.)
    assert all(tuple(p) in set(map(tuple, points)) for p in selected)


def test_failed_enclosing_search_preserves_verified_original_and_growth_searches(profile, monkeypatch):
    from types import SimpleNamespace
    from scripts.aufgabe04.perception.stand_axis import head_outer_border as outer
    corners = tuple(ImagePoint(x, y) for x, y in ((100., 100.), (200., 100.), (200., 200.), (100., 200.)))
    original = SimpleNamespace(accepted=True, corners=corners)
    calls = []
    def missing(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(accepted=False, corners=None)
    monkeypatch.setattr(outer, 'refine_projected_head_border', missing)
    selected, evidence = outer.select_current_outer_head_border(cv2, None,
        model_profile=profile, refinement=original, corridor_half_width_px=6.,
        prefer_outer_metric_rail=True)
    assert selected is original and evidence.accepted
    assert evidence.current_raw_alternatives == (corners,)
    assert evidence.attempted_growth_factors == outer.OUTER_HEAD_SEARCH_GROWTH_FACTORS
    assert len(calls) == 3 and calls[-1]['prefer_outer_metric_rail']
