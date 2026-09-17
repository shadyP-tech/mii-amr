"""Current-image offsets repair locators without supplying missing borders."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import HeadAcquisitionDeadlineExceeded
from scripts.aufgabe04.perception.stand_axis.head_hint_neighborhood import (
    MAX_HINT_NEIGHBORS, observed_head_hint_neighborhood,
)
from scripts.aufgabe04.perception.stand_axis.metric_edge_association import metric_corner_arm_support
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


def quad(x0, y0, x1, y1):
    return tuple(ImagePoint(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class HeadHintNeighborhoodTests(unittest.TestCase):
    def setUp(self):
        # No QR dimensions, identity, pose or manually chosen scale factor are
        # available to the helper; the profile bounds only its search corridor.
        self.profile = SimpleNamespace(head_width_m=.078, head_height_m=.078,
                                       tolerance_m=.0005, measurement_status="measured")
        self.raw = np.zeros((250, 300), np.uint8)
        self.hint = quad(55., 55., 185., 185.)

    def draw(self, corners, raw=None):
        raw = self.raw if raw is None else raw
        cv2.polylines(raw, [np.rint([(p.u_px, p.v_px) for p in corners]).astype(np.int32)],
                      True, 255, 1, cv2.LINE_8)

    def nearby(self, variants, target, tolerance=1.5):
        return any(max(np.hypot(a.u_px-b.u_px, a.v_px-b.v_px)
                       for a, b in zip(candidate, target)) <= tolerance for candidate in variants)

    def run_locator(self, corners=None, **kwargs):
        return observed_head_hint_neighborhood(cv2, self.raw,
            self.hint if corners is None else corners, model_profile=self.profile, **kwargs)

    def test_large_neutral_hint_locates_nearby_current_head_without_fixed_scale(self):
        target = quad(60., 60., 180., 180.)
        self.draw(target)
        before = self.raw.copy()
        diagnostics = {}
        variants = self.run_locator(diagnostics=diagnostics)
        self.assertTrue(self.nearby(variants, target))
        self.assertNotIn(self.hint, variants)
        self.assertLessEqual(len(variants), MAX_HINT_NEIGHBORS)
        self.assertFalse(diagnostics["supplies_measurement"])
        np.testing.assert_array_equal(self.raw, before)

    def test_observed_outward_and_inward_modes_both_remain_unverified_hints(self):
        hint = quad(60., 60., 180., 180.)
        inset, outer = quad(64., 64., 176., 176.), quad(56., 56., 184., 184.)
        self.draw(inset)
        self.draw(outer)
        variants = self.run_locator(hint)
        self.assertTrue(self.nearby(variants, inset))
        self.assertTrue(self.nearby(variants, outer))
        self.assertLessEqual(len(variants), MAX_HINT_NEIGHBORS)
        self.assertEqual(len(variants), len(set(variants)))

    def test_missing_side_cannot_be_invented_from_remaining_three(self):
        for first, last in (((60, 180), (60, 60)), ((60, 60), (180, 60)),
                            ((180, 60), (180, 180))):
            cv2.line(self.raw, first, last, 255, 1)
        self.assertEqual(self.run_locator(), ())

    def test_background_intersections_with_missing_corner_arms_are_rejected(self):
        # Long segments support every trimmed side; their four apparent line
        # intersections are absent in the untouched image.
        for first, last in (((72, 60), (168, 60)), ((72, 180), (168, 180)),
                            ((60, 72), (60, 168)), ((180, 72), (180, 168))):
            cv2.line(self.raw, first, last, 255, 1)
        self.assertEqual(self.run_locator(), ())

    def test_long_background_continuations_do_not_move_local_hint(self):
        target = quad(60., 60., 180., 180.)
        self.draw(target)
        baseline = self.run_locator()
        for y in (15, 35, 210, 230):
            cv2.line(self.raw, (0, y), (299, y), 255, 1)
        for x in (10, 30, 220, 240, 270):
            cv2.line(self.raw, (x, 0), (x, 249), 255, 1)
        cv2.line(self.raw, (0, 60), (299, 60), 255, 1)
        cv2.line(self.raw, (60, 0), (60, 249), 255, 1)
        variants = self.run_locator()
        self.assertTrue(self.nearby(variants, target))
        self.assertEqual(variants, baseline)

    def test_each_variant_has_current_corner_arms_and_stays_in_image(self):
        self.draw(quad(5., 5., 125., 125.))
        for x in (3, 7, 123, 127):
            cv2.line(self.raw, (x, 0), (x, 132), 255, 1)
        variants = self.run_locator(quad(2., 2., 128., 128.))
        self.assertTrue(variants)
        self.assertLessEqual(len(variants), MAX_HINT_NEIGHBORS)
        for variant in variants:
            self.assertTrue(metric_corner_arm_support(cv2, self.raw, variant).accepted)
            self.assertTrue(all(0 <= p.u_px < self.raw.shape[1]
                                and 0 <= p.v_px < self.raw.shape[0] for p in variant))

    def test_perspective_seed_keeps_observed_directions(self):
        hint = tuple(ImagePoint(*p) for p in ((50., 50.), (190., 62.), (184., 200.), (57., 183.)))
        center = np.mean([(p.u_px, p.v_px) for p in hint], axis=0)
        target = tuple(ImagePoint(*(center + .96 * (np.asarray((p.u_px, p.v_px)) - center)))
                       for p in hint)
        self.draw(target)
        variants = self.run_locator(hint)
        self.assertTrue(self.nearby(variants, target, tolerance=2.))

    def test_invalid_and_blank_inputs_have_no_locators(self):
        self.assertEqual(self.run_locator(), ())
        self.assertEqual(self.run_locator(quad(-10., 10., 100., 100.)), ())
        self.assertEqual(self.run_locator((ImagePoint(1., 1.),) * 4), ())

    def test_deadline_prevents_work_and_interrupts_sampling(self):
        self.draw(quad(60., 60., 180., 180.))
        with self.assertRaises(HeadAcquisitionDeadlineExceeded):
            self.run_locator(deadline_monotonic_sec=0.)
        with patch("scripts.aufgabe04.perception.stand_axis.head_acquisition_budget.time.monotonic",
                   side_effect=(1., 1., 3.)):
            with self.assertRaises(HeadAcquisitionDeadlineExceeded):
                self.run_locator(deadline_monotonic_sec=2.)


if __name__ == "__main__":
    unittest.main()
