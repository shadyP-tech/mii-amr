"""Color gating must remove clutter without fabricating border evidence."""

from pathlib import Path
from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.debug.viewer_color_edges import (
    color_edge_support, color_edge_exclusion,
)
from scripts.aufgabe04.perception.debug.stand_axis_viewer import build_parser
from scripts.aufgabe04.perception.models import ColorRange
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame
from scripts.aufgabe04.perception.stand_axis.head_geometry_acquisition import estimate_current_head_geometry
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.model_input_cache import MetricModelInputCache
from scripts.aufgabe04.perception.stand_axis.current_image_head_fit import CurrentImageHeadFit
from scripts.aufgabe04.perception.stand_axis_image import estimate_stand_axis_from_edges
from tests.aufgabe04.test_physical_head_pipeline import head_image


@pytest.mark.parametrize("hue", [0, 179, 65, 115, 140])
def test_palette_preserves_thin_borders_but_removes_neutral_background(hue):
    frame = np.zeros((100, 180, 3), np.uint8)
    color = cv2.cvtColor(np.array([[[hue, 220, 220]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    cv2.rectangle(frame, (80, 20), (160, 80), tuple(int(v) for v in color), 1)
    cv2.rectangle(frame, (10, 10), (50, 90), (255, 255, 255), 2)
    original = frame.copy()
    raw = _canny_edges_from_frame(cv2, frame, edge_preprocess="channel_union",
                                 blur_kernel=1, canny_low=20, canny_high=60)
    support = color_edge_support(cv2, np, frame)
    filtered = cv2.bitwise_and(raw, support)
    assert np.count_nonzero(raw[:, :60]) > 100
    assert not np.any(filtered[:, :60])
    np.testing.assert_array_equal(filtered[:, 70:], raw[:, 70:])
    np.testing.assert_array_equal(frame, original)
    assert np.count_nonzero(filtered) > 100
    assert not np.any(support[40:60, 100:140])  # Neutral head interior stays excluded.


def test_selected_color_and_tuned_ranges_do_not_fall_back_to_other_colors():
    frame = np.zeros((50, 100, 3), np.uint8)
    frame[10:40, 10:40] = (0, 255, 0)
    frame[10:40, 60:90] = (255, 0, 0)
    support = color_edge_support(cv2, np, frame, color="green")
    assert support[10, 20] == 255
    assert support[20, 70] == 0
    tuned = color_edge_support(cv2, np, frame, color="green",
        ranges=[ColorRange("custom", (100, 100, 100), (130, 255, 255))])
    assert tuned[20, 20] == 0
    assert tuned[10, 70] == 255
    assert not np.any(color_edge_support(cv2, np, frame, color="red"))


@pytest.mark.parametrize("hsv", [(122, 60, 150), (136, 35, 145)])
def test_light_grey_preserves_recorded_cool_cast_border(hsv):
    pixels = np.zeros((50, 50, 3), np.uint8)
    pixels[10:40, 10:40] = hsv
    frame = cv2.cvtColor(pixels, cv2.COLOR_HSV2BGR)
    support = color_edge_support(cv2, np, frame, color="light-grey")
    assert np.all(support[10, 10:40] == 255)
    assert not support[25, 25]


def test_colored_outer_border_excludes_enclosed_qr_edges():
    frame = np.full((100, 120, 3), (160, 160, 160), np.uint8)
    cv2.rectangle(frame, (20, 15), (100, 85), (0, 255, 0), 4)
    for x in range(35, 90, 10):
        cv2.line(frame, (x, 30), (x, 70), (0, 0, 0), 3)
    raw = cv2.Canny(frame, 20, 60)
    filtered = raw & color_edge_support(cv2, np, frame)
    assert np.any(raw[30:70, 35:90])
    assert not np.any(filtered[30:70, 35:90])
    assert np.any(filtered[10:25, 20:100])


def test_five_color_palette_excludes_yellow():
    frame = cv2.cvtColor(np.full((20, 20, 3), (28, 220, 220), np.uint8), cv2.COLOR_HSV2BGR)
    assert not np.any(color_edge_support(cv2, np, frame))


def test_roi_color_filter_composes_with_existing_wall_exclusion():
    support = np.zeros((60, 80), np.uint8)
    support[20:40, 30:50] = 255
    roi = SimpleNamespace(x0=20, x1=60, y0=10, y1=50)
    wall = np.zeros((40, 40), np.uint8)
    wall[20, :] = 255
    exclusion = color_edge_exclusion(cv2, support, roi=roi, existing=wall)
    assert exclusion.shape == wall.shape
    assert exclusion[15, 15] == 0
    assert exclusion[20, 15] == 255
    assert exclusion[0, 0] == 255


def test_metric_fit_uses_filtered_edges_and_rechecks_changed_mask_on_same_image():
    profile = load_measured_physical_stand_model(Path(__file__).resolve().parents[2]
        / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
    frame, _ = head_image(profile)
    frame[np.any(frame > 0, axis=2)] = (0, 255, 0)
    for x in range(20, 180, 20):
        cv2.line(frame, (x, 20), (x, 580), (255, 255, 255), 2)
    original = frame.copy()
    cache, holder = MetricModelInputCache(frame), CurrentImageHeadFit()
    kwargs = dict(model_profile=profile, camera_fx_px=640., camera_fy_px=640.,
        camera_cx_px=400., camera_cy_px=300., blur_kernel=1,
        input_cache=cache, input_cache_roi=(0, 0, 800, 600), current_image_head_fit=holder)
    for color, expected_usable in (("green", True), ("red", False), ("green", True)):
        support = color_edge_support(cv2, np, frame, color=color)
        result, debug = estimate_current_head_geometry(cv2, frame,
            edge_exclusion_mask=color_edge_exclusion(cv2, support), **kwargs)
        assert result.usable is expected_usable, result.reason
        assert not np.any(debug.raw_edges[:, :180])
        assert not np.any(debug.raw_edges[support == 0])
        if expected_usable:
            assert abs(result.yaw_deg + 45.) < 3.
        else:
            assert not np.any(debug.raw_edges)
    np.testing.assert_array_equal(frame, original)


def test_legacy_edge_path_does_not_reintroduce_excluded_background():
    frame = np.zeros((120, 200, 3), np.uint8)
    cv2.rectangle(frame, (100, 25), (180, 95), (0, 255, 0), 3)
    cv2.rectangle(frame, (10, 10), (70, 110), (255, 255, 255), 3)
    support = color_edge_support(cv2, np, frame)
    _, debug = estimate_stand_axis_from_edges(cv2, frame,
        edge_preprocess="channel_union", blur_kernel=1, canny_low=20, canny_high=60,
        edge_exclusion_mask=color_edge_exclusion(cv2, support))
    assert not np.any(debug.edges[:, :80])
    assert np.any(debug.edges[:, 90:])


def test_viewer_defaults_to_all_palette_colors_with_explicit_opt_out():
    source = ["--compressed-image-topic", "/camera/image_raw/compressed"]
    args = build_parser().parse_args(source)
    assert args.color_edge_mask and args.edge_color == "all"
    args = build_parser().parse_args(source + ["--no-color-edge-mask", "--edge-color", "green"])
    assert not args.color_edge_mask and args.edge_color == "green"
