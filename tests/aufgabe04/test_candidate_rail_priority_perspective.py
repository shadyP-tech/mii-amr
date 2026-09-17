"""Candidate-height ranking must retain foreshortened and neck-split rails.

The synthetic scenes have known projected geometry and complete raw support.
Remote open line clutter must not remove the candidate's only viable locators.
"""

import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.candidate_head_search import CandidateHeadSearch
from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import (
    MAX_RAILS_PER_DIRECTION, acquire_cold_head_proposal,
)
from scripts.aufgabe04.perception.stand_axis.head_proposal import _extent
from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
from tests.aufgabe04 import test_head_model_angle_reference as angle_fixtures


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class CandidateRailPriorityPerspectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        angle_fixtures.HeadModelAngleReferenceTest.setUpClass()
        cls.fixture = angle_fixtures.HeadModelAngleReferenceTest()
        cls.profile = cls.fixture.profile

    def scene(self, *, split_bottom, clutter):
        corners, _camera = self.fixture.projection(45. if split_bottom else 75.)
        _width, height, center = _extent(corners)
        raw = np.zeros((600, 1000), np.uint8)
        pixels = np.rint([(p.u_px, p.v_px) for p in corners]).astype(int)
        for index, (a, b) in enumerate(zip(pixels, np.roll(pixels, -1, axis=0))):
            if split_bottom and index == 2:
                # Preserve both measured outer bottom intervals. A real neck
                # interrupts only their central gap and extends downwards.
                for start, stop in ((0., .42), (.58, 1.)):
                    first = np.rint(a + (b-a) * start).astype(int)
                    last = np.rint(a + (b-a) * stop).astype(int)
                    cv2.line(raw, tuple(first), tuple(last), 255, 1)
                for fraction in (.42, .58):
                    origin = np.rint(a + (b-a) * fraction).astype(int)
                    cv2.line(raw, tuple(origin), (int(origin[0]), 540), 255, 1)
            else:
                cv2.line(raw, tuple(a), tuple(b), 255, 1)
        if clutter:
            # More than the complete quota in each direction. These distant
            # open lines resemble room/rack edges, not another eligible head.
            for index in range(MAX_RAILS_PER_DIRECTION + 6):
                cv2.line(raw, (20, 20 + 6*index),
                          (20 + round(height), 20 + 6*index), 255, 1)
                cv2.line(raw, (650 + 6*index, 420),
                          (650 + 6*index, 420 + round(height)), 255, 1)
        return raw, corners, CandidateHeadSearch(center, height)

    def assert_acquisition_survives_clutter(self, *, split_bottom):
        for clutter in (False, True):
            with self.subTest(clutter=clutter):
                raw, corners, search = self.scene(split_bottom=split_bottom, clutter=clutter)
                if not split_bottom:
                    width, height, _center = _extent(corners)
                    self.assertLess(width, .35 * height)
                # First establish that the raw foreground really supports the
                # model. Only acquisition below must discover its own corners.
                measured = refine_projected_head_border(cv2, raw, corners,
                                                       corridor_half_width_px=4.)
                self.assertTrue(measured.accepted, measured.reason)
                result = acquire_cold_head_proposal(
                    cv2, cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), raw_edges=raw,
                    model_profile=self.profile, candidate_search=search)
                self.assertIsNotNone(result.proposal, result.reason)
                self.assertAlmostEqual(result.proposal.center_u_px, search.center[0], delta=3.)
                self.assertAlmostEqual(result.proposal.center_v_px, search.center[1], delta=3.)
                self.assertTrue(search.accepts_measurement(result.proposal.corners))

    def test_closed_high_yaw_head_remains_acquirable_with_candidate_screen(self):
        self.assert_acquisition_survives_clutter(split_bottom=False)

    def test_neck_split_head_remains_acquirable_with_candidate_screen(self):
        self.assert_acquisition_survives_clutter(split_bottom=True)


if __name__ == "__main__":
    unittest.main()
