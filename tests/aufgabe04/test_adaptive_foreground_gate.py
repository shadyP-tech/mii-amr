from __future__ import annotations

import unittest

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover - optional outside the project environment
    cv2 = None
    numpy = None

from scripts.aufgabe04.perception.stand_axis.adaptive_foreground_gate import (
    adaptive_foreground_gate_from_background,
)


@unittest.skipIf(
    cv2 is None or numpy is None,
    "numpy and OpenCV are required for adaptive foreground gating",
)
class AdaptiveForegroundGateTest(unittest.TestCase):
    def _scene_with_colored_head(self, bgr):
        frame = numpy.full((180, 300, 3), (180, 170, 160), dtype=numpy.uint8)
        # The background sample comes from a geometry-confirmed radiator band.
        seed = numpy.zeros(frame.shape[:2], dtype=numpy.uint8)
        seed[16:132, 18:92] = 255
        for x in (28, 40, 52, 64, 76):
            cv2.line(frame, (x, 18), (x, 130), (145, 138, 132), 2)

        cv2.rectangle(frame, (165, 36), (246, 116), bgr, thickness=-1)
        # The physical head may carry an achromatic QR board.  It must remain
        # inside the dilated foreground support even though no hue is assumed.
        cv2.rectangle(frame, (180, 50), (228, 101), (24, 24, 24), thickness=-1)
        cv2.rectangle(frame, (188, 58), (220, 93), (235, 235, 235), thickness=3)
        return frame, seed

    def test_gate_supports_different_foreground_hues_without_configured_hue(self):
        for bgr in ((210, 45, 35), (35, 45, 210), (45, 210, 50)):
            with self.subTest(bgr=bgr):
                frame, seed = self._scene_with_colored_head(bgr)
                result = adaptive_foreground_gate_from_background(cv2, numpy, frame, seed)

                self.assertTrue(result.applied)
                self.assertEqual(result.reason, "applied")
                self.assertEqual(int(result.gate[76, 205]), 255)
                self.assertEqual(int(result.gate[76, 10]), 0)

    def test_returns_no_gate_without_a_geometry_confirmed_background_sample(self):
        frame, _seed = self._scene_with_colored_head((210, 45, 35))
        empty_seed = numpy.zeros(frame.shape[:2], dtype=numpy.uint8)

        result = adaptive_foreground_gate_from_background(cv2, numpy, frame, empty_seed)

        self.assertFalse(result.applied)
        self.assertIsNone(result.gate)
        self.assertEqual(result.reason, "insufficient_background_sample")

    def test_gate_is_only_a_foreground_topology_prior(self):
        frame, seed = self._scene_with_colored_head((210, 45, 35))
        result = adaptive_foreground_gate_from_background(cv2, numpy, frame, seed)
        self.assertTrue(result.applied)

        raw_edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 20, 60)
        topology_edges = cv2.bitwise_and(raw_edges, result.gate)

        # The learned foreground support may guide topology, but it is not a
        # direct heater/Canny mask. Raw support still contains both the stand
        # boundary and any radiator rail for independent final edge fitting.
        self.assertGreater(int(numpy.count_nonzero(raw_edges[30:120, 25:32])), 0)
        self.assertGreater(int(numpy.count_nonzero(raw_edges[45:108, 160:171])), 0)
        self.assertGreater(int(numpy.count_nonzero(topology_edges[45:108, 160:171])), 0)

    def test_rejects_an_overly_broad_foreground_gate(self):
        frame = numpy.full((120, 180, 3), (24, 40, 210), dtype=numpy.uint8)
        frame[10:30, 10:50] = (180, 170, 160)
        seed = numpy.zeros(frame.shape[:2], dtype=numpy.uint8)
        seed[10:30, 10:50] = 255

        result = adaptive_foreground_gate_from_background(cv2, numpy, frame, seed)

        self.assertFalse(result.applied)
        self.assertIsNone(result.gate)
        self.assertEqual(result.reason, "foreground_coverage_unreliable")
