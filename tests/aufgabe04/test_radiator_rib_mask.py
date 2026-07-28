from __future__ import annotations

import unittest

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover - optional outside the project environment
    cv2 = None
    numpy = None

from scripts.aufgabe04.perception.stand_axis.radiator_rib_mask import (
    repeated_vertical_rib_exclusion_mask,
)


@unittest.skipIf(
    cv2 is None or numpy is None,
    "numpy and OpenCV are required for repeated-rib suppression",
)
class RepeatedVerticalRibMaskTest(unittest.TestCase):
    def test_masks_regular_radiator_rails_but_preserves_two_stand_rails(self):
        edges = numpy.zeros((180, 300), dtype=numpy.uint8)
        radiator_xs = (22, 34, 46, 58, 70, 82)
        for x in radiator_xs:
            cv2.line(edges, (x, 12), (x, 128), 255, 2)
        for x in (170, 250):
            cv2.line(edges, (x, 30), (x, 140), 255, 2)

        result = repeated_vertical_rib_exclusion_mask(cv2, edges)

        self.assertGreaterEqual(result.suppressed_rail_count, len(radiator_xs))
        for x in radiator_xs:
            self.assertEqual(int(result.mask[70, x]), 255)
        self.assertEqual(int(result.mask[80, 170]), 0)
        self.assertEqual(int(result.mask[80, 250]), 0)

    def test_does_not_mask_an_isolated_pair_of_stand_rails(self):
        edges = numpy.zeros((180, 300), dtype=numpy.uint8)
        for x in (130, 220):
            cv2.line(edges, (30, 28), (30, 140), 255, 2)
            cv2.line(edges, (x, 28), (x, 140), 255, 2)

        result = repeated_vertical_rib_exclusion_mask(cv2, edges)

        self.assertEqual(result.suppressed_rail_count, 0)
        self.assertEqual(int(result.mask.max()), 0)
