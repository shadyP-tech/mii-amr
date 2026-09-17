"""Candidate screening preserves every permitted raw-refinement outcome."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from scripts.aufgabe04.perception.stand_axis.candidate_head_search import CandidateHeadSearch
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


def rectangle(u=400., v=300., height=100., width=None):
    width = height if width is None else width
    return tuple(ImagePoint(u + x, v + y) for x, y in
                 ((-width/2, -height/2), (width/2, -height/2),
                  (width/2, height/2), (-width/2, height/2)))


class CandidateHeadSearchTests(unittest.TestCase):
    def setUp(self):
        self.search = CandidateHeadSearch((400., 300.), 100.)

    def test_hint_envelope_includes_inner_panel_outer_growth_and_refinement_shift(self):
        minimum = .60 * 100. / (1.15 * 1.25)
        maximum = 1.35 * 100. / .85
        for height in (minimum, .071/.078*100., 100., maximum):
            self.assertTrue(self.search.accepts_hint(rectangle(height=height)))
        self.assertFalse(self.search.accepts_hint(rectangle(height=minimum-.001)))
        self.assertFalse(self.search.accepts_hint(rectangle(height=maximum+.001)))
        for u, v in ((560., 300.), (240., 300.), (400., 460.), (400., 140.)):
            self.assertTrue(self.search.accepts_hint(rectangle(u, v)))
        self.assertFalse(self.search.accepts_hint(rectangle(560.001, 300.)))
        self.assertFalse(self.search.accepts_hint(rectangle(400., 460.001)))

    def test_measured_filter_uses_original_limits_without_hint_allowance(self):
        for height in (60., 62., 100., 133., 135.):
            self.assertTrue(self.search.accepts_measurement(rectangle(height=height)))
        for height in (59.999, 135.001):
            self.assertFalse(self.search.accepts_measurement(rectangle(height=height)))
        for u, v in ((550., 300.), (250., 300.), (400., 450.), (400., 150.), (400., 400.)):
            self.assertTrue(self.search.accepts_measurement(rectangle(u, v)))
        self.assertFalse(self.search.accepts_measurement(rectangle(550.001, 300.)))
        self.assertFalse(self.search.accepts_measurement(rectangle(400., 450.001)))
        self.assertFalse(self.search.accepts_measurement(rectangle(550., 375.)))

    def test_sixty_pixel_displacement_and_smaller_configured_bounds(self):
        displaced = rectangle(460., 300.)
        self.assertTrue(self.search.accepts_hint(displaced))
        self.assertTrue(self.search.accepts_measurement(displaced))
        tighter = replace(self.search, max_center_offset_ratio=.4)
        self.assertTrue(tighter.accepts_hint(rectangle(450., 300.)))
        self.assertFalse(tighter.accepts_hint(rectangle(450.001, 300.)))
        self.assertFalse(tighter.accepts_measurement(displaced))

    def test_every_permitted_synthetic_refinement_retains_its_hint(self):
        # Sample the exact scale/center envelope, including limiting positions.
        for final_height in (60., 100., 135.):
            for growth in (1., 1.10, 1.25):
                for refinement_scale in (.85, 1., 1.15):
                    for final_center in ((400., 300.), (548., 300.), (400., 373.), (510., 365.)):
                        final = rectangle(*final_center, height=final_height)
                        self.assertTrue(self.search.accepts_measurement(final))
                        for angle in (0., math.pi/4., math.pi/2., math.pi):
                            hint_center = (final_center[0] + 10.*math.cos(angle),
                                           final_center[1] + 10.*math.sin(angle))
                            hint = rectangle(*hint_center, height=final_height/(growth*refinement_scale))
                            self.assertTrue(self.search.accepts_hint(hint), (growth, refinement_scale, hint_center))

    def test_invalid_projection_falls_back_without_a_screen(self):
        for values in ((None, 300., 100.), (math.nan, 300., 100.),
                       (400., math.inf, 100.), (400., 300., 0.), (400., 300., -100.)):
            self.assertIsNone(CandidateHeadSearch.optional(*values))
        self.assertEqual(CandidateHeadSearch.optional(400., 300., 100.), self.search)

    def test_measured_scale_matrix_matches_current_post_fit_association_gate(self):
        from scripts.aufgabe04.real_robot.observer.head_model_admission import head_scale_gate
        for expected in (50., 100., 200.):
            screen = CandidateHeadSearch((400., 300.), expected)
            for ratio in (.59, .60, .62, 1., 1.33, 1.35, 1.36):
                for balance in (.64, .65, .66, 1.):
                    left = 2. * ratio * expected / (1. + balance)
                    right = left * balance
                    corners = (ImagePoint(350., 300.-left/2), ImagePoint(450., 300.-right/2),
                               ImagePoint(450., 300.+right/2), ImagePoint(350., 300.+left/2))
                    measured_left = math.dist((corners[0].u_px, corners[0].v_px),
                                              (corners[3].u_px, corners[3].v_px))
                    measured_right = math.dist((corners[1].u_px, corners[1].v_px),
                                               (corners[2].u_px, corners[2].v_px))
                    expected_decision = head_scale_gate(expected_size_px=expected,
                        left_height_px=measured_left, right_height_px=measured_right)["accepted"]
                    self.assertEqual(screen.accepts_measurement(corners), expected_decision,
                                     (expected, ratio, balance))


class CandidateColdScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise unittest.SkipTest("OpenCV and NumPy required")
        cls.cv2, cls.np = cv2, np

    def test_distant_complete_heads_do_not_spend_selected_candidates_strict_budget(self):
        from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import acquire_cold_head_proposal
        from tests.aufgabe04.test_head_cold_acquisition import frame_with_heads
        frame = frame_with_heads((((190, 130), (100, 100), 0), ((460, 250), (100, 100), 0)))
        before = frame.copy()
        result = acquire_cold_head_proposal(self.cv2, frame,
            candidate_search=CandidateHeadSearch((190., 130.), 100.))
        self.assertIsNotNone(result.proposal, result.reason)
        self.assertAlmostEqual(result.proposal.center_u_px, 190., delta=3.)
        self.assertGreater(result.joint_border_diagnostics["candidate_screen_hint_rejections"], 0)
        for check in result.joint_border_diagnostics["strict_verifications"]:
            self.assertTrue(CandidateHeadSearch((190., 130.), 100.).accepts_hint(
                tuple(ImagePoint(*p) for p in check["locator_corners"])))
        self.np.testing.assert_array_equal(frame, before)

    def test_small_supported_hints_survive_as_texture_without_proposal_filter(self):
        from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import acquire_cold_head_proposal
        from tests.aufgabe04.test_head_cold_acquisition import frame_with_heads
        frame = frame_with_heads((((190, 130), (100, 100), 0), ((190, 130), (30, 30), 0)))
        filtered_heights = []
        def candidate_filter(proposal):
            filtered_heights.append(proposal.observed_height_px)
            return True
        captured = []
        from scripts.aufgabe04.perception.stand_axis.head_proposal_selection import select_verified_head
        def select(*args, **kwargs):
            captured.extend(kwargs["texture_hypotheses"])
            return select_verified_head(*args, **kwargs)
        with patch("scripts.aufgabe04.perception.stand_axis.head_cold_acquisition.select_verified_head", side_effect=select):
            result = acquire_cold_head_proposal(self.cv2, frame,
                candidate_search=CandidateHeadSearch((190., 130.), 100.), proposal_filter=candidate_filter)
        self.assertGreater(result.joint_border_diagnostics["texture_only_hypotheses"], 0)
        self.assertTrue(any(max(p.v_px for p in item[1])-min(p.v_px for p in item[1]) < 40.
                            for item in captured))
        self.assertTrue(all(height > 40. for height in filtered_heights))

    def test_two_current_heads_within_candidate_envelope_remain_ambiguous(self):
        from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import acquire_cold_head_proposal
        from tests.aufgabe04.test_head_cold_acquisition import frame_with_heads
        frame = frame_with_heads((((190, 130), (100, 100), 0), ((340, 130), (100, 100), 0)))
        result = acquire_cold_head_proposal(self.cv2, frame,
            candidate_search=CandidateHeadSearch((265., 130.), 100.))
        self.assertIsNone(result.proposal)
        self.assertEqual(result.reason, "head_proposal_ambiguous")

    def test_measured_corners_must_pass_the_exact_gate_after_a_permitted_hint(self):
        from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import acquire_cold_head_proposal
        from tests.aufgabe04.test_head_cold_acquisition import frame_with_heads
        frame = frame_with_heads()
        result = acquire_cold_head_proposal(self.cv2, frame,
            candidate_search=CandidateHeadSearch((190., 130.), 70.))
        self.assertIsNone(result.proposal)
        self.assertGreater(result.raw_verifications, 0)
        self.assertGreater(result.joint_border_diagnostics["candidate_screen_measurement_rejections"], 0)


if __name__ == "__main__":
    unittest.main()
