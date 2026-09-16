"""Map reuse saves setup cost while every output uses the current image."""

from dataclasses import replace
from unittest.mock import Mock
import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.camera_calibration import (
    RectificationMapCache, camera_calibration_from_info, rectify_bgr_frame,
)
from tests.aufgabe04.test_camera_calibration import camera_info


def test_map_cache_reuses_only_unchanged_calibration_and_never_image_pixels():
    calibration = camera_calibration_from_info(camera_info())
    cache = RectificationMapCache()
    backend = Mock(wraps=cv2)
    backend.CV_32FC1 = cv2.CV_32FC1
    backend.INTER_LINEAR = cv2.INTER_LINEAR
    backend.BORDER_CONSTANT = cv2.BORDER_CONSTANT
    dark = np.zeros((600, 800, 3), np.uint8)
    bright = np.full_like(dark, 255)
    first = rectify_bgr_frame(dark, calibration, backend, np, map_cache=cache)
    second = rectify_bgr_frame(bright, calibration, backend, np, map_cache=cache)
    assert backend.initUndistortRectifyMap.call_count == 1
    assert not np.array_equal(first, second)
    assert np.array_equal(second, rectify_bgr_frame(bright, calibration, cv2, np))
    changed = replace(calibration, distortion=(0.,) * len(calibration.distortion))
    rectify_bgr_frame(bright, changed, backend, np, map_cache=cache)
    assert backend.initUndistortRectifyMap.call_count == 2
    with pytest.raises(ValueError, match="dimensions"):
        rectify_bgr_frame(bright[:300], changed, backend, np, map_cache=cache)
    assert backend.initUndistortRectifyMap.call_count == 2
