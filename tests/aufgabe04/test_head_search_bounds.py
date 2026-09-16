"""Candidate locators obey the observer's existing vertical association limit."""

import math
import unittest

from scripts.aufgabe04.perception.stand_axis.head_search_bounds import HeadSearchBounds
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


def rectangle(u, v, *, height=100., width=100., angle=0.):
    c, s = math.cos(angle), math.sin(angle)
    return tuple(ImagePoint(u + c * x - s * y, v + s * x + c * y)
                 for x, y in ((-width/2, -height/2), (width/2, -height/2),
                              (width/2, height/2), (-width/2, height/2)))


class HeadSearchBoundsTests(unittest.TestCase):
    def bounds(self, offset=1.5):
        return HeadSearchBounds.optional(500., 500., 100., offset, .3)

    def test_vertical_limit_does_not_reduce_horizontal_search_reach(self):
        bounds = self.bounds()
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                self.assertTrue(bounds.accepts(rectangle(500., 500. + sign * 75.)))
                self.assertFalse(bounds.accepts(rectangle(500., 500. + sign * 75.01)))
                self.assertTrue(bounds.accepts(rectangle(500. + sign * 150., 500.)))
                self.assertFalse(bounds.accepts(rectangle(500. + sign * 150.01, 500.)))
        # The original radial cap still applies inside the vertical strip.
        self.assertFalse(bounds.accepts(rectangle(650., 575.)))
        diagnostic = bounds.diagnostics()
        self.assertEqual(diagnostic["max_center_offset_ratio"], 1.5)
        self.assertEqual(diagnostic["max_vertical_center_offset_ratio"], .75)

    def test_smaller_configured_cap_remains_binding_in_both_directions(self):
        bounds = self.bounds(.4)
        for dx, dy in ((40., 0.), (-40., 0.), (0., 40.), (0., -40.)):
            with self.subTest(dx=dx, dy=dy):
                self.assertTrue(bounds.accepts(rectangle(500. + dx, 500. + dy)))
                self.assertFalse(bounds.accepts(rectangle(500. + dx * 1.001, 500. + dy * 1.001)))
        diagnostic = bounds.diagnostics()
        self.assertEqual(diagnostic["max_center_offset_ratio"], .4)
        self.assertEqual(diagnostic["max_vertical_center_offset_ratio"], .4)

    def test_search_canvas_matches_asymmetric_limits_and_clips_to_image(self):
        # Both axes retain the same conservative corner/support margin; only
        # the center search reach differs by 75 pixels for this 100-pixel head.
        self.assertEqual(self.bounds().image_bounds((1000, 1000)), (233, 308, 768, 693))
        self.assertEqual(self.bounds(.4).image_bounds((1000, 1000)), (343, 343, 658, 658))
        clipped = HeadSearchBounds.optional(50., 60., 100., 1.5, .3)
        self.assertEqual(clipped.image_bounds((200, 250)), (0, 0, 250, 200))

    def test_canvas_contains_complete_rotated_heads_at_accepted_offsets(self):
        bounds = self.bounds()
        x0, y0, x1, y1 = bounds.image_bounds((1000, 1000))
        for dx, dy in ((149., 0.), (-149., 0.), (0., 74.), (0., -74.),
                       (120., 70.), (-120., -70.)):
            for angle in (-.3, 0., .3):
                with self.subTest(dx=dx, dy=dy, angle=angle):
                    corners = rectangle(500. + dx, 500. + dy,
                                        height=120., width=162., angle=angle)
                    self.assertTrue(bounds.accepts(corners))
                    self.assertTrue(all(x0 <= p.u_px < x1 and y0 <= p.v_px < y1
                                        for p in corners))


if __name__ == "__main__":
    unittest.main()
