"""A completed candidate miss may retry locators within the original budget.

Only contour/rail locator output is controlled. Raw support, physical-frame
refinement, candidate admission and ambiguity decisions use production code.
"""

from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis import head_cold_acquisition as acquisition
from scripts.aufgabe04.perception.stand_axis.candidate_head_search import CandidateHeadSearch
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model


def rectangle(x, y, height):
    return ((x-height/2, y-height/2), (x+height/2, y-height/2),
            (x+height/2, y+height/2), (x-height/2, y+height/2))


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class CandidateHeadRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_measured_physical_stand_model(
            Path(__file__).resolve().parents[2]
            / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")

    def setUp(self):
        self.screen = CandidateHeadSearch((320., 240.), 100.)
        # The small rectangle passes the conservative locator screen but its
        # measured 50-pixel height cannot meet the final 60-pixel minimum.
        self.small = rectangle(320., 340., 50.)
        self.head = rectangle(260., 240., 100.)
        self.other_head = rectangle(380., 240., 100.)
        self.deadline = 10.
        self.clock = 1.

    def acquire(self, primary, retry=(), *, options=None, expire_on_locator=None,
                expire_after_fit=None):
        self.raw = np.zeros((480, 640), np.uint8)
        for corners in (*primary, *retry):
            cv2.polylines(self.raw, [np.array(corners, np.int32)], True, 255, 1)
        self.frame = cv2.cvtColor(self.raw, cv2.COLOR_GRAY2BGR)
        self.locator_calls, self.fit_calls = [], []
        self.refinements = {"stale": object()}
        actual_fit = acquisition.refine_current_physical_head

        def hints(cv, gray, raw, **kwargs):
            self.locator_calls.append((raw, kwargs))
            number = len(self.locator_calls)
            if expire_on_locator == number:
                self.clock = self.deadline
            return tuple(primary if number == 1 else retry), (0, 0)

        def fit(*args, **kwargs):
            result = actual_fit(*args, **kwargs)
            self.fit_calls.append(kwargs)
            if expire_after_fit == len(self.fit_calls):
                self.clock = self.deadline
            return result

        kwargs = dict(raw_edges=self.raw, model_profile=self.profile,
                      candidate_search=self.screen, refinement_out=self.refinements,
                      deadline_monotonic_sec=self.deadline)
        kwargs.update(options or {})
        with patch.object(cv2, "findContours", return_value=([], None)), \
             patch.object(acquisition, "_rail_endpoint_hints", side_effect=hints), \
             patch.object(acquisition, "refine_current_physical_head", side_effect=fit), \
             patch("scripts.aufgabe04.perception.stand_axis.head_acquisition_budget.time.monotonic",
                   side_effect=lambda: self.clock):
            return acquisition.acquire_cold_head_proposal(cv2, self.frame, **kwargs)

    def test_completed_small_frame_miss_retries_real_head_with_original_pixels_and_deadline(self):
        result = self.acquire([self.small], [self.head])
        self.assertIsNotNone(result.proposal, result.reason)
        self.assertAlmostEqual(result.proposal.observed_height_px, 100., delta=1.)
        self.assertEqual(len(self.locator_calls), 2)
        self.assertEqual([call[1]["preferred_head_height_px"] for call in self.locator_calls],
                         [None, 100.])
        self.assertTrue(all(raw is self.raw for raw, _options in self.locator_calls))
        self.assertTrue(all(options["deadline_monotonic_sec"] == self.deadline
                            for _raw, options in self.locator_calls))
        self.assertTrue(all(options["deadline_monotonic_sec"] == self.deadline
                            for options in self.fit_calls))
        diagnostic = result.joint_border_diagnostics
        retry = diagnostic["candidate_size_retry"]
        self.assertEqual(retry["initial_reason"], "head_proposal_unavailable")
        self.assertEqual(retry["initial_raw_verifications"], 1)
        self.assertEqual(retry["remaining_raw_verifications"], acquisition.MAX_RAW_VERIFICATIONS - 1)
        self.assertEqual(retry["retry_raw_verifications"], 1)
        self.assertFalse(retry["deadline_extended"])
        self.assertEqual(result.raw_verifications, 2)
        self.assertEqual(len(diagnostic["strict_verifications"]), 2)
        self.assertEqual(len(self.fit_calls), 2)
        # This proves the returned physical-frame receipt is bound to the
        # original frame, not to a fresh image or a synthetic retry proposal.
        self.assertEqual(set(self.refinements), {"selected"})
        proof = self.refinements["selected"]
        measured, _outer, _seed = proof.resolve(
            self.frame, self.raw, model_profile=self.profile,
            proposal_corners=result.proposal.corners)
        self.assertEqual(measured.corners, result.proposal.corners)

    def test_retry_gets_only_the_remaining_one_of_twelve_strict_checks(self):
        primary = [rectangle(x, y, 50.) for y in (140., 340.)
                   for x in (230., 290., 350., 410.)]
        primary += [rectangle(175., 210., 50.), rectangle(175., 270., 50.),
                    rectangle(465., 210., 50.)]
        self.assertEqual(len(primary), 11)
        result = self.acquire(primary, [self.head, self.other_head])
        self.assertIsNone(result.proposal)
        self.assertEqual(result.reason, "head_cold_acquisition_verification_budget_exceeded")
        self.assertEqual(len(self.locator_calls), 2)
        self.assertEqual(len(self.fit_calls), acquisition.MAX_RAW_VERIFICATIONS)
        self.assertEqual(result.raw_verifications, acquisition.MAX_RAW_VERIFICATIONS)
        diagnostic = result.joint_border_diagnostics
        self.assertEqual(len(diagnostic["strict_verifications"]), acquisition.MAX_RAW_VERIFICATIONS)
        self.assertEqual(diagnostic["candidate_size_retry"]["initial_raw_verifications"], 11)
        self.assertEqual(diagnostic["candidate_size_retry"]["remaining_raw_verifications"], 1)
        self.assertEqual(diagnostic["candidate_size_retry"]["retry_raw_verifications"], 1)
        self.assertGreater(diagnostic["unverified_independent_hypotheses"], 0)
        self.assertEqual(self.refinements, {})

    def test_completed_empty_retry_stops_after_one_attempt(self):
        result = self.acquire([], [])
        self.assertEqual(result.reason, "head_proposal_unavailable")
        self.assertEqual(len(self.locator_calls), 2)
        self.assertEqual(result.raw_verifications, 0)
        self.assertEqual(self.refinements, {})

    def test_success_ambiguity_and_unresolved_verification_budget_never_retry(self):
        cases = (([self.head], {}, "current_head_proposal"),
                 ([self.head, self.other_head], {}, "head_proposal_ambiguous"),
                 ([self.head, self.other_head], {"_verification_limit": 1},
                  "head_cold_acquisition_verification_budget_exceeded"))
        for primary, options, reason in cases:
            with self.subTest(reason=reason):
                result = self.acquire(primary, [self.head], options=options)
                self.assertEqual(result.reason, reason)
                self.assertEqual(len(self.locator_calls), 1)
                self.assertNotIn("candidate_size_retry", result.joint_border_diagnostics)
                if reason != "current_head_proposal":
                    self.assertIsNone(result.proposal)

    def test_absent_screen_and_legacy_bounds_do_not_request_size_retry(self):
        for options in ({"candidate_search": None},
                        {"expected_head_center_u_px": 320., "expected_head_center_v_px": 240.,
                         "expected_head_height_px": 100.}):
            with self.subTest(options=options):
                result = self.acquire([], [self.head], options=options)
                self.assertEqual(result.reason, "head_proposal_unavailable")
                self.assertEqual(len(self.locator_calls), 1)
                self.assertNotIn("candidate_size_retry", result.joint_border_diagnostics)

    def test_deadline_expiry_in_primary_cannot_start_retry(self):
        result = self.acquire([self.small], [self.head], expire_on_locator=1)
        self.assertEqual(result.reason, "head_acquisition_deadline_exceeded")
        self.assertIsNone(result.proposal)
        self.assertEqual(len(self.locator_calls), 1)
        self.assertEqual(self.fit_calls, [])
        self.assertEqual(self.refinements, {})

    def test_retry_deadline_keeps_completed_primary_and_retry_work_in_diagnostics(self):
        result = self.acquire([self.small], [self.head], expire_after_fit=2)
        self.assertEqual(result.reason, "head_acquisition_deadline_exceeded")
        self.assertIsNone(result.proposal)
        self.assertEqual(len(self.locator_calls), 2)
        self.assertEqual(len(self.fit_calls), 2)
        self.assertEqual(result.raw_verifications, 2)
        self.assertEqual(len(result.joint_border_diagnostics["strict_verifications"]), 2)
        self.assertEqual(result.joint_border_diagnostics["candidate_size_retry"]["retry_raw_verifications"], 1)
        self.assertTrue(all(options["deadline_monotonic_sec"] == self.deadline
                            for _raw, options in self.locator_calls))
        self.assertEqual(self.refinements, {})


if __name__ == "__main__":
    unittest.main()
