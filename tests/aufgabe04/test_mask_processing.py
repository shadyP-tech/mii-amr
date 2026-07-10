import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.mask_processing import (  # noqa: E402
    apply_morphology,
    build_mask_for_ranges,
    classify_mask_roi,
)
from scripts.aufgabe04.perception.models import ColorRange  # noqa: E402
from scripts.aufgabe04.perception.roi import Rect  # noqa: E402


try:
    import cv2
    import numpy
except ImportError:
    cv2 = None
    numpy = None


@unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for mask tests")
class MaskProcessingTest(unittest.TestCase):
    def test_build_mask_unions_multiple_ranges(self):
        hsv = numpy.array(
            [
                [[5, 200, 200], [175, 200, 200], [90, 200, 200]],
                [[5, 20, 200], [175, 200, 200], [90, 200, 200]],
            ],
            dtype=numpy.uint8,
        )
        ranges = (
            ColorRange("red", (0, 70, 50), (10, 255, 255)),
            ColorRange("red", (170, 70, 50), (179, 255, 255)),
        )

        mask = build_mask_for_ranges(cv2, numpy, hsv, ranges)

        self.assertEqual(int(mask[0, 0]), 255)
        self.assertEqual(int(mask[0, 1]), 255)
        self.assertEqual(int(mask[0, 2]), 0)
        self.assertEqual(int(mask[1, 0]), 0)
        self.assertEqual(int(mask[1, 1]), 255)

    def test_apply_morphology_returns_same_mask_when_kernel_disabled(self):
        mask = numpy.zeros((3, 3), dtype=numpy.uint8)

        self.assertIs(apply_morphology(cv2, mask, kernel_size=1, close_iterations=2, open_iterations=1), mask)

    def test_apply_morphology_runs_configured_close_open_path(self):
        mask = numpy.zeros((7, 7), dtype=numpy.uint8)
        mask[2:5, 2:5] = 255

        cleaned = apply_morphology(cv2, mask, kernel_size=3, close_iterations=1, open_iterations=1)

        self.assertEqual(cleaned.shape, mask.shape)
        self.assertEqual(cleaned.dtype, mask.dtype)

    def test_classify_mask_roi_counts_matched_pixels(self):
        mask = numpy.zeros((4, 4), dtype=numpy.uint8)
        mask[1:3, 1:3] = 255

        result = classify_mask_roi(cv2, mask, Rect(1, 1, 2, 2), "green")

        self.assertEqual(result.label, "green")
        self.assertAlmostEqual(result.confidence, 1.0)
        self.assertEqual(result.matched_pixels, 4)
        self.assertEqual(result.total_pixels, 4)

    def test_classify_mask_roi_returns_unknown_for_empty_clipped_roi(self):
        mask = numpy.zeros((4, 4), dtype=numpy.uint8)

        result = classify_mask_roi(cv2, mask, Rect(10, 1, 2, 2), "green")

        self.assertEqual(result.label, "unknown")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.matched_pixels, 0)
        self.assertEqual(result.total_pixels, 0)


if __name__ == "__main__":
    unittest.main()
