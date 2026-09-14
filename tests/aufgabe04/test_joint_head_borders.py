"""Independent current rails close heads without complete locator endpoints."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposal, acquire_head_proposal, _same_head
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class JointHeadBordersTests(unittest.TestCase):
    def image_and_fragments(self, lefts=(270,)):
        image = np.zeros((360, 520, 3), np.uint8)
        fragments = []
        for x in lefts:
            cv2.rectangle(image, (x, 100), (x + 100, 200), (190, 190, 190), 2)
            # Each line is observed, but opposite fragments have inconsistent
            # endpoints. Neither endpoint pairing describes the whole head.
            fragments.extend(((x + 20, 100, x + 80, 100),
                              (x + 10, 200, x + 75, 200),
                              (x, 140, x, 198),
                              (x + 100, 102, x + 100, 155)))
        return image, np.asarray(fragments, np.float32).reshape(-1, 1, 4)

    def acquire(self, image, fragments, *, center=260.):
        # Only locator output is injected. Gradient scoring, unchanged raw
        # Canny/refinement/corner checks and final proposal selection are real.
        locator = SimpleNamespace(detect=lambda _: (fragments, None, None, None))
        with patch.object(cv2, "createLineSegmentDetector", return_value=locator), \
                patch.object(cv2, "HoughLinesP", return_value=fragments):
            return acquire_head_proposal(
                cv2, image, expected_head_center_u_px=center,
                expected_head_center_v_px=150., expected_head_height_px=100.,
            )

    def test_independent_partial_rail_endpoints_recover_complete_current_head(self):
        image, fragments = self.image_and_fragments()
        result = self.acquire(image, fragments)
        self.assertIsNotNone(result.proposal, result.reason)
        x0, y0, x1, y1 = result.proposal.head_bounds_xyxy
        for actual, expected in zip((x0, y0, x1, y1), (270, 100, 370, 200)):
            self.assertAlmostEqual(actual, expected, delta=3.)
        self.assertLessEqual(result.raw_verifications, 12)
        self.assertTrue(any(item["accepted"] for item in
                            result.joint_border_diagnostics["strict_verifications"]))
        self.assertIsNone(getattr(result.proposal, "yaw_deg", None))

    def test_locator_intersections_cannot_replace_a_missing_current_border(self):
        image, fragments = self.image_and_fragments()
        image[94:107, 260:381] = 0
        result = self.acquire(image, fragments)
        self.assertIsNone(result.proposal)
        self.assertEqual(result.reason, "head_proposal_unavailable")

    def test_locator_order_does_not_change_current_border_selection(self):
        image, fragments = self.image_and_fragments()
        first = self.acquire(image, fragments)
        reversed_order = self.acquire(image, fragments[::-1].copy())
        self.assertIsNotNone(first.proposal)
        self.assertEqual(first.proposal.corners, reversed_order.proposal.corners)
        self.assertEqual(first.joint_border_diagnostics["selected_corners"],
                         reversed_order.joint_border_diagnostics["selected_corners"])

    def test_two_complete_heads_remain_ambiguous_with_partial_locators(self):
        image, fragments = self.image_and_fragments((50, 270))
        result = self.acquire(image, fragments, center=210.)
        self.assertIsNone(result.proposal)
        self.assertEqual(result.reason, "head_proposal_ambiguous")
        self.assertLessEqual(result.raw_verifications, 12)

    def test_distinct_nested_head_sizes_are_not_collapsed_into_a_printed_inset(self):
        def proposal(size):
            low, high = 150 - size / 2, 150 + size / 2
            corners = tuple(ImagePoint(x, y) for x, y in
                            ((low, low), (high, low), (high, high), (low, high)))
            return HeadProposal(corners, (0, 0, 300, 300), (low, low, high, high),
                                150., 150., size, size / 100., 0., 1., 1.)
        self.assertFalse(_same_head(proposal(70.), proposal(130.)))
        self.assertTrue(_same_head(proposal(100.), proposal(91.)))
        self.assertTrue(_same_head(proposal(100.), proposal(79.5)))

    def test_dense_clutter_has_explicit_candidate_and_strict_fit_bounds(self):
        random = np.random.default_rng(2)
        image = np.zeros((480, 640, 3), np.uint8)
        for _ in range(180):
            x, y = random.integers(0, 450, 2)
            width, height = random.integers(35, 145, 2)
            color = tuple(int(value) for value in random.integers(30, 255, 3))
            cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), color, 2)
        result = acquire_head_proposal(
            cv2, image, expected_head_center_u_px=320.,
            expected_head_center_v_px=240., expected_head_height_px=100.,
        )
        self.assertIsNone(result.proposal)
        self.assertEqual(result.reason, "head_proposal_ambiguous")
        bounds = result.joint_border_diagnostics
        self.assertLessEqual(bounds["compared_pair_combinations"], bounds["max_pair_combinations"])
        self.assertLessEqual(bounds["scored_current_hypotheses"], bounds["max_scored_current_hypotheses"])
        self.assertLessEqual(bounds["retained_hypotheses"], 64)
        self.assertLessEqual(result.raw_verifications, 12)


if __name__ == "__main__":
    unittest.main()
