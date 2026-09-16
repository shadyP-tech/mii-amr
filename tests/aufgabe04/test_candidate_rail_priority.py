"""A named candidate keeps complete measured rails amid background grid clutter."""

import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import (
    MAX_RAILS_PER_DIRECTION, _rail_endpoint_hints, acquire_cold_head_proposal,
)
from scripts.aufgabe04.perception.stand_axis.head_search_bounds import HeadSearchBounds
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class CandidateRailPriorityTests(unittest.TestCase):
    def scene(self):
        frame = np.zeros((460, 600, 3), np.uint8)
        cv2.rectangle(frame, (200, 100), (300, 200), (200, 200, 200), 2)
        # A rack/window grid below the stand supplies many longer lines and
        # short intersecting fragments within the conservative search canvas.
        # Its complete rectangles lie outside the candidate's vertical band.
        for x in range(180, 331, 10):
            cv2.line(frame, (x, 220), (x, 370), (100, 100, 100), 1)
        for y in range(220, 371, 10):
            cv2.line(frame, (180, y), (330, y), (100, 100, 100), 1)
        raw = _canny_edges_from_frame(cv2, frame, edge_preprocess="channel_union",
                                      blur_kernel=5, canny_low=20, canny_high=60)
        return frame, raw

    def test_complete_candidate_rail_hint_survives_the_fixed_clutter_quota(self):
        frame, raw = self.scene()
        bounds = HeadSearchBounds.optional(250., 150., 100., 1.5, .3)
        groups = []
        hints, counts = _rail_endpoint_hints(
            cv2, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), raw,
            search_bounds=bounds, rail_groups_out=groups)
        self.assertEqual(counts, (MAX_RAILS_PER_DIRECTION, MAX_RAILS_PER_DIRECTION))
        # Check the actual detected geometry, without prescribing ordering or
        # a particular line detector. All four exterior rails must remain
        # available for intersections; retaining only inset edges is not enough.
        for direction, coordinate, exterior_sign in ((0, 100., -1), (0, 200., 1),
                                                     (1, 200., -1), (1, 300., 1)):
            with self.subTest(direction=direction, coordinate=coordinate):
                cross = 1 - direction
                self.assertTrue(any(
                    90. < length < 110.
                    and all(0. <= exterior_sign * (p[cross] - coordinate) < 4.
                            for p in (a, b))
                    for length, a, b in groups[direction]))
        complete = [hint for hint in hints
                    if abs(min(p[0] for p in hint) - 200.) < 5.
                    and abs(max(p[0] for p in hint) - 300.) < 5.
                    and abs(min(p[1] for p in hint) - 100.) < 5.
                    and abs(max(p[1] for p in hint) - 200.) < 5.]
        self.assertTrue(complete, "clutter consumed all complete candidate rail hints")

    def test_surviving_locator_still_requires_current_raw_border_measurement(self):
        frame, raw = self.scene()
        options = dict(expected_head_center_u_px=250., expected_head_center_v_px=150.,
                       expected_head_height_px=100., max_center_offset_ratio=1.5)
        result = acquire_cold_head_proposal(cv2, frame, raw_edges=raw, **options)
        self.assertIsNotNone(result.proposal, result.reason)
        self.assertAlmostEqual(result.proposal.center_u_px, 250., delta=3.)
        self.assertAlmostEqual(result.proposal.center_v_px, 150., delta=3.)
        self.assertGreater(result.proposal.raw_edge_support, .9)
        self.assertIsNone(acquire_cold_head_proposal(
            cv2, frame, raw_edges=np.zeros_like(raw), **options).proposal)


if __name__ == "__main__":
    unittest.main()
