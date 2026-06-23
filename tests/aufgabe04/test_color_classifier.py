import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.color_classifier import (  # noqa: E402
    classify_hsv_pixels,
    hsv_pixel_in_range,
    validate_color_range,
)
from scripts.aufgabe04.perception.models import ColorClassifierConfig, ColorRange  # noqa: E402


class ColorClassifierTest(unittest.TestCase):
    def test_classifies_canonical_target_color_patch(self):
        patch = [[(65, 210, 210) for _ in range(8)] for _ in range(6)]

        result = classify_hsv_pixels(patch, config=ColorClassifierConfig(min_confidence=0.40))

        self.assertEqual(result.label, "green")
        self.assertAlmostEqual(result.confidence, 1.0)
        self.assertEqual(result.matched_pixels, 48)

    def test_rejects_low_saturation_gray_patch(self):
        patch = [[(65, 20, 210) for _ in range(8)] for _ in range(6)]

        result = classify_hsv_pixels(patch, config=ColorClassifierConfig(min_confidence=0.20))

        self.assertEqual(result.label, "unknown")
        self.assertEqual(result.matched_pixels, 0)

    def test_rejects_off_hue_patch_with_same_brightness(self):
        patch = [[(140, 210, 210) for _ in range(8)] for _ in range(6)]

        result = classify_hsv_pixels(patch, config=ColorClassifierConfig(min_confidence=0.20))

        self.assertEqual(result.label, "unknown")

    def test_classification_is_stable_under_moderate_lighting_shift(self):
        bright_patch = [[(65, 210, 240) for _ in range(5)] for _ in range(5)]
        dim_patch = [[(65, 210, 120) for _ in range(5)] for _ in range(5)]

        bright = classify_hsv_pixels(bright_patch, config=ColorClassifierConfig(min_confidence=0.40))
        dim = classify_hsv_pixels(dim_patch, config=ColorClassifierConfig(min_confidence=0.40))

        self.assertEqual(bright.label, "green")
        self.assertEqual(dim.label, "green")
        self.assertGreaterEqual(dim.confidence, 0.95)

    def test_wraparound_hsv_range_matches_both_sides_of_red(self):
        red_wrap = ColorRange("red", (170, 70, 50), (10, 255, 255))

        self.assertTrue(hsv_pixel_in_range((175, 200, 200), red_wrap))
        self.assertTrue(hsv_pixel_in_range((5, 200, 200), red_wrap))
        self.assertFalse(hsv_pixel_in_range((90, 200, 200), red_wrap))

    def test_classifies_red_on_low_and_high_hue_ranges(self):
        low_hue_red = [[(5, 210, 210) for _ in range(4)] for _ in range(4)]
        high_hue_red = [[(175, 210, 210) for _ in range(4)] for _ in range(4)]

        low = classify_hsv_pixels(low_hue_red, config=ColorClassifierConfig(min_confidence=0.40))
        high = classify_hsv_pixels(high_hue_red, config=ColorClassifierConfig(min_confidence=0.40))

        self.assertEqual(low.label, "red")
        self.assertEqual(high.label, "red")
        self.assertAlmostEqual(low.confidence, 1.0)
        self.assertAlmostEqual(high.confidence, 1.0)

    def test_rejects_empty_pixel_input(self):
        result = classify_hsv_pixels([], config=ColorClassifierConfig(min_confidence=0.20))

        self.assertEqual(result.label, "unknown")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.total_pixels, 0)

    def test_mixed_color_roi_confidence_reflects_matched_fraction(self):
        pixels = [(65, 210, 210)] * 6 + [(110, 210, 210)] * 4

        result = classify_hsv_pixels(pixels, config=ColorClassifierConfig(min_confidence=0.40))

        self.assertEqual(result.label, "green")
        self.assertAlmostEqual(result.confidence, 0.6)
        self.assertEqual(result.matched_pixels, 6)
        self.assertEqual(result.total_pixels, 10)

    def test_validates_hsv_bounds(self):
        with self.assertRaisesRegex(ValueError, "hue"):
            validate_color_range(ColorRange("bad", (-1, 70, 50), (10, 255, 255)))
        with self.assertRaisesRegex(ValueError, "saturation/value"):
            validate_color_range(ColorRange("bad", (0, 300, 50), (10, 255, 255)))
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            validate_color_range(ColorRange("bad", (0, 200, 50), (10, 100, 255)))


if __name__ == "__main__":
    unittest.main()
