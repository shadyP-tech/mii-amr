"""Final candidate guidance only locates borders supported by the current image."""

import math
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis import head_cold_acquisition as acquisition
from scripts.aufgabe04.perception.stand_axis import candidate_rail_hints as guided_rails
from scripts.aufgabe04.perception.stand_axis.candidate_head_search import CandidateHeadSearch
from scripts.aufgabe04.perception.stand_axis.candidate_rail_hints import candidate_observed_rail_hints
from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import HeadAcquisitionDeadlineExceeded
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics
from scripts.aufgabe04.real_robot.observer.current_scan_head_proposal_filter import CurrentScanHeadProposalFilter


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class CandidateGuidedAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_measured_physical_stand_model(
            Path(__file__).resolve().parents[2]
            / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")

    def scene(self, *, width=98, center_x=390, clutter=True, missing_bottom=False):
        raw = np.zeros((600, 1000), np.uint8)
        groups = [[], []]

        def rail(first, last, direction, draw=True):
            if draw:
                cv2.line(raw, first, last, 255, 1)
            groups[direction].append((math.dist(first, last), first, last))

        if clutter:
            # Both pools exceed the global24 quota. Their 100px lengths rank
            # before the physical98px head in the size-only retry as well.
            # All rails occupy the same image third and scale strata.
            for index in range(30):
                rail((370, 10+4*index), (470, 10+4*index), 0)
                rail((480+4*index, 360), (480+4*index, 460), 1)
        x0, x1 = center_x-width//2, center_x+width//2
        rail((x0, 191), (x1, 191), 0)
        rail((x0, 289), (x1, 289), 0, draw=not missing_bottom)
        rail((x0, 191), (x0, 289), 1)
        rail((x1, 191), (x1, 289), 1)
        return raw, groups, CandidateHeadSearch((center_x-60., 240.), 100.)

    @staticmethod
    def detected_segments(groups):
        return np.asarray([(*first, *last) for group in groups for _length, first, last in group],
                          np.float32).reshape(-1, 1, 4)

    def test_supported_pair_priority_retains_offset_head_behind_both_global_quotas(self):
        raw, groups, search = self.scene()
        segments = self.detected_segments(groups)
        detector = Mock(detect=Mock(return_value=(segments, None, None, None)))
        for preferred_height in (None, 100.):
            with self.subTest(preferred_height=preferred_height), \
                 patch.object(cv2, "createLineSegmentDetector", return_value=detector), \
                 patch.object(cv2, "HoughLinesP", return_value=None):
                retained = []
                acquisition._rail_endpoint_hints(cv2, raw, raw,
                    preferred_head_height_px=preferred_height, rail_groups_out=retained)
                self.assertEqual([len(group) for group in retained], [24, 24])
                self.assertFalse(any(length == 98. for group in retained for length, _a, _b in group))
        diagnostic, priority = {}, []
        original = raw.copy()
        hints = candidate_observed_rail_hints(cv2, raw, groups, search,
            diagnostics=diagnostic, prioritized_groups_out=priority)
        self.assertEqual(len(hints), 1)
        self.assertEqual(set(hints[0]), {(341., 191.), (439., 191.), (439., 289.), (341., 289.)})
        self.assertEqual([len(group) for group in priority], [2, 2])
        self.assertEqual(diagnostic["input_rails"], [32, 32])
        self.assertFalse(diagnostic["overflow"])
        self.assertFalse(diagnostic["supplies_measurement"])
        np.testing.assert_array_equal(raw, original)

    def test_short_foreshortened_sides_remain_observed_locators(self):
        raw, groups, search = self.scene(width=26)
        priority = []
        hints = candidate_observed_rail_hints(cv2, raw, groups, search,
                                             prioritized_groups_out=priority)
        self.assertEqual(len(hints), 1)
        self.assertEqual(set(hints[0]), {(377., 191.), (403., 191.), (403., 289.), (377., 289.)})
        self.assertEqual({item[0] for item in priority[0]}, {26.})
        self.assertLess(26./98., .35)

    def test_blank_missing_or_clipped_borders_cannot_become_candidate_rectangles(self):
        blank, groups, search = self.scene(clutter=False)
        blank.fill(0)
        missing = self.scene(clutter=False, missing_bottom=True)
        clipped = self.scene(clutter=False, center_x=40)
        for name, (raw, rails, candidate) in (("blank", (blank, groups, search)),
                                              ("missing_bottom", missing), ("clipped", clipped)):
            with self.subTest(scene=name):
                diagnostic = {}
                self.assertEqual(candidate_observed_rail_hints(
                    cv2, raw, rails, candidate, diagnostics=diagnostic), ())
                self.assertEqual(diagnostic["supported_hints"], 0)
                self.assertFalse(diagnostic["supplies_measurement"])

    def test_guided_pair_corner_work_stops_at_original_deadline(self):
        raw, groups, search = self.scene(clutter=False)
        original = raw.copy()
        clock = [1.]
        actual_corner_check = guided_rails.metric_corner_arm_support

        def corner_check(*args, **options):
            result = actual_corner_check(*args, **options)
            self.assertTrue(result.accepted)
            clock[0] = 10.
            return result

        with patch.object(guided_rails, "metric_corner_arm_support", side_effect=corner_check) as check, \
             patch("scripts.aufgabe04.perception.stand_axis.head_acquisition_budget.time.monotonic",
                   side_effect=lambda: clock[0]), \
             self.assertRaises(HeadAcquisitionDeadlineExceeded):
            candidate_observed_rail_hints(cv2, raw, groups, search, deadline_monotonic_sec=10.)
        # The original and extended endpoint quads share this same pair-loop
        # iteration. A deadline check only at the next pair is too late.
        self.assertEqual(check.call_count, 1)
        np.testing.assert_array_equal(raw, original)

    def test_actual_final_locator_recovers_current_frame_after_both_quotas_miss(self):
        raw, groups, search = self.scene()
        frame, proof = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), {}
        original = raw.copy()
        detector = Mock(detect=Mock(return_value=(self.detected_segments(groups), None, None, None)))
        with patch.object(cv2, "createLineSegmentDetector", return_value=detector), \
             patch.object(cv2, "HoughLinesP", return_value=None), \
             patch.object(cv2, "findContours", return_value=([], None)), \
             patch.object(acquisition, "_rail_endpoint_hints", wraps=acquisition._rail_endpoint_hints) as locate, \
             patch.object(acquisition, "refine_current_physical_head",
                          wraps=acquisition.refine_current_physical_head) as fit, \
             patch("scripts.aufgabe04.perception.stand_axis.head_acquisition_budget.time.monotonic",
                   return_value=1.):
            result = acquisition.acquire_cold_head_proposal(cv2, frame, raw_edges=raw,
                candidate_search=search, model_profile=self.profile, refinement_out=proof,
                deadline_monotonic_sec=10.)
        self.assertIsNotNone(result.proposal, result.reason)
        self.assertEqual(locate.call_count, 3)
        self.assertEqual([call.kwargs["preferred_head_height_px"] for call in locate.call_args_list],
                         [None, 100., 100.])
        self.assertEqual([call.kwargs["candidate_search"] for call in locate.call_args_list],
                         [None, None, search])
        self.assertTrue(all(call.args[2] is raw for call in locate.call_args_list))
        self.assertTrue(all(call.kwargs["deadline_monotonic_sec"] == 10.
                            for call in locate.call_args_list + fit.call_args_list))
        self.assertAlmostEqual(result.proposal.center_u_px-search.center[0], 60.)
        self.assertAlmostEqual(result.proposal.observed_height_px, 98.)
        self.assertEqual(result.raw_verifications, 1)
        self.assertEqual(fit.call_count, 1)
        diagnostic = result.joint_border_diagnostics
        self.assertEqual(len(diagnostic["strict_verifications"]), 1)
        primary, final = diagnostic["candidate_size_retry"], diagnostic["candidate_size_retry"]["next_attempt"]
        self.assertFalse(primary["guided_rails"])
        self.assertTrue(final["guided_rails"])
        for attempt in (primary, final):
            self.assertEqual(attempt["initial_reason"], "head_proposal_unavailable")
            self.assertEqual(attempt["initial_raw_verifications"], 0)
            self.assertTrue(attempt["same_raw_image"])
            self.assertFalse(attempt["deadline_extended"])
        self.assertFalse(diagnostic["candidate_guided_rails"]["supplies_measurement"])
        measured, _outer, _seed = proof["selected"].resolve(
            frame, raw, model_profile=self.profile, proposal_corners=result.proposal.corners)
        self.assertEqual(measured.corners, result.proposal.corners)
        np.testing.assert_array_equal(raw, original)

    @staticmethod
    def rectangle(x, y, height):
        return ((x-height/2, y-height/2), (x+height/2, y-height/2),
                (x+height/2, y+height/2), (x-height/2, y+height/2))

    def controlled_attempts(self, attempts, *, verification_limit=12, expire_after_fit=None,
                            proposal_filter=None):
        """Only locator availability is controlled; every measured border is real."""
        self.raw = np.zeros((480, 640), np.uint8)
        for corners in (corners for attempt in attempts for corners in attempt):
            cv2.polylines(self.raw, [np.asarray(corners, np.int32)], True, 255, 1)
        self.frame = cv2.cvtColor(self.raw, cv2.COLOR_GRAY2BGR)
        self.locator_calls, self.fit_calls, self.proof = [], [], {}
        self.clock = 1.
        actual_fit = acquisition.refine_current_physical_head

        def locate(cv, gray, raw, **options):
            self.locator_calls.append((raw, options))
            # Current raw lines exist before the quota. The final fallback is
            # therefore eligible after the first two complete geometry misses.
            options["all_rail_groups_out"].extend((
                ((100., (210., 190.), (310., 190.)),), ()))
            return tuple(attempts[len(self.locator_calls)-1]), (1, 0)

        def fit(*args, **options):
            measured = actual_fit(*args, **options)
            self.fit_calls.append(options)
            if len(self.fit_calls) == expire_after_fit:
                self.clock = 10.
            return measured

        with patch.object(cv2, "findContours", return_value=([], None)), \
             patch.object(acquisition, "_rail_endpoint_hints", side_effect=locate), \
             patch.object(acquisition, "refine_current_physical_head", side_effect=fit), \
             patch("scripts.aufgabe04.perception.stand_axis.head_acquisition_budget.time.monotonic",
                   side_effect=lambda: self.clock):
            return acquisition.acquire_cold_head_proposal(cv2, self.frame, raw_edges=self.raw,
                model_profile=self.profile, candidate_search=CandidateHeadSearch((320., 240.), 100.),
                refinement_out=self.proof, deadline_monotonic_sec=10.,
                _verification_limit=verification_limit, proposal_filter=proposal_filter)

    def test_wrong_bearing_rough_head_cannot_suppress_later_scan_bound_target(self):
        bearing = math.atan(60./520.)
        scan = PlainLaserScan((math.inf, math.inf, .6, .601, .6, math.inf, math.inf),
                              bearing-math.radians(3), math.radians(1), .1, 3.5,
                              "scan", 10., 10.01)
        screen = CurrentScanHeadProposalFilter(
            intrinsics=CameraIntrinsics(640, 480, 520., 515., 320., 240.),
            scan_from_camera=RigidTransform("scan", "camera", (0., 0., 0.), (.5, -.5, .5, -.5)),
            scan=scan, map_bearing_rad=0., cone_half_angle_rad=math.radians(3),
            accepted_range_m=(.49, .71), now_sec=10.1, max_scan_age_sec=.5,
            min_cluster_sample_count=2, max_camera_map_bearing_delta_rad=math.radians(12))
        # The fully observed background frame is inside candidate image bounds,
        # but even its rough-center envelope misses the certified bearing gate.
        # The actual head is only available after both initial locator quotas.
        result = self.controlled_attempts(([self.rectangle(460., 240., 100.)], [],
                                           [self.rectangle(260., 240., 100.)]), proposal_filter=screen)
        self.assertIsNotNone(result.proposal, result.reason)
        self.assertAlmostEqual(result.proposal.center_u_px, 260.)
        self.assertEqual(len(self.locator_calls), 3)
        self.assertEqual(result.raw_verifications, 1)
        self.assertEqual(len(self.fit_calls), 1)
        retry = result.joint_border_diagnostics["candidate_size_retry"]
        self.assertEqual(retry["initial_reason"], "head_proposal_candidate_association_rejected")
        self.assertEqual(retry["initial_raw_verifications"], 0)
        self.assertEqual(retry["remaining_raw_verifications"], 12)
        self.assertTrue(retry["next_attempt"]["guided_rails"])
        association = screen.metadata()
        self.assertEqual(association["rough_rejections"], 1)
        self.assertEqual(association["measured_scan_previews"], 1)
        self.assertTrue(association["measured_associations"][0]["associated"])
        measured, _outer, _seed = self.proof["selected"].resolve(
            self.frame, self.raw, model_profile=self.profile, proposal_corners=result.proposal.corners)
        self.assertEqual(measured.corners, result.proposal.corners)

    def test_final_fallback_spends_only_one_remaining_strict_check(self):
        small = [self.rectangle(x, y, 50.) for y in (140., 340.)
                 for x in (230., 290., 350., 410.)]
        small += [self.rectangle(175., 210., 50.), self.rectangle(175., 270., 50.),
                  self.rectangle(465., 210., 50.)]
        result = self.controlled_attempts((small[:6], small[6:],
            [self.rectangle(260., 240., 100.), self.rectangle(380., 240., 100.)]))
        self.assertIsNone(result.proposal)
        self.assertEqual(result.reason, "head_cold_acquisition_verification_budget_exceeded")
        self.assertEqual(len(self.locator_calls), 3)
        self.assertEqual(len(self.fit_calls), 12)
        self.assertEqual(result.raw_verifications, 12)
        diagnostic = result.joint_border_diagnostics
        self.assertEqual(len(diagnostic["strict_verifications"]), 12)
        primary, final = diagnostic["candidate_size_retry"], diagnostic["candidate_size_retry"]["next_attempt"]
        self.assertEqual(primary["initial_raw_verifications"], 6)
        self.assertEqual(primary["retry_raw_verifications"], 6)
        self.assertEqual(final["initial_raw_verifications"], 5)
        self.assertEqual(final["remaining_raw_verifications"], 1)
        self.assertEqual(final["retry_raw_verifications"], 1)
        self.assertGreater(diagnostic["unverified_independent_hypotheses"], 0)
        self.assertEqual(self.proof, {})

    def test_success_ambiguity_or_unresolved_second_pass_never_enter_final_fallback(self):
        head, other = self.rectangle(260., 240., 100.), self.rectangle(380., 240., 100.)
        for seeds, limit, reason in (([head], 12, "current_head_proposal"),
                                    ([head, other], 12, "head_proposal_ambiguous"),
                                    ([head, other], 1, "head_cold_acquisition_verification_budget_exceeded")):
            with self.subTest(reason=reason):
                result = self.controlled_attempts(([], seeds), verification_limit=limit)
                self.assertEqual(result.reason, reason)
                self.assertEqual(len(self.locator_calls), 2)
                self.assertTrue(all(options["candidate_search"] is None
                                    for _raw, options in self.locator_calls))
                self.assertNotIn("next_attempt", result.joint_border_diagnostics["candidate_size_retry"])

    def test_complete_second_miss_with_no_remaining_checks_cannot_start_guidance(self):
        result = self.controlled_attempts(([self.rectangle(230., 340., 50.)],
                                           [self.rectangle(410., 340., 50.)]), verification_limit=2)
        self.assertEqual(result.reason, "head_proposal_unavailable")
        self.assertIsNone(result.proposal)
        self.assertEqual(len(self.locator_calls), 2)
        self.assertEqual(result.raw_verifications, 2)
        self.assertNotIn("next_attempt", result.joint_border_diagnostics["candidate_size_retry"])
        self.assertEqual(self.proof, {})

    def test_completed_final_miss_never_starts_a_fourth_locator_pass(self):
        result = self.controlled_attempts(([], [], []))
        self.assertEqual(result.reason, "head_proposal_unavailable")
        self.assertEqual(len(self.locator_calls), 3)
        self.assertEqual(result.raw_verifications, 0)
        final = result.joint_border_diagnostics["candidate_size_retry"]["next_attempt"]
        self.assertTrue(final["guided_rails"])
        self.assertNotIn("next_attempt", final)
        self.assertEqual(self.proof, {})

    def test_expiring_final_fit_retains_all_three_pass_counts_without_a_proof(self):
        result = self.controlled_attempts(([self.rectangle(230., 340., 50.)],
            [self.rectangle(410., 340., 50.)], [self.rectangle(260., 240., 100.)]),
            expire_after_fit=3)
        self.assertEqual(result.reason, "head_acquisition_deadline_exceeded")
        self.assertIsNone(result.proposal)
        self.assertEqual(len(self.locator_calls), 3)
        self.assertEqual(result.raw_verifications, 3)
        self.assertEqual(len(result.joint_border_diagnostics["strict_verifications"]), 3)
        self.assertTrue(all(raw is self.raw for raw, _options in self.locator_calls))
        self.assertTrue(all(options["deadline_monotonic_sec"] == 10.
                            for _raw, options in self.locator_calls))
        self.assertTrue(all(options["deadline_monotonic_sec"] == 10. for options in self.fit_calls))
        self.assertEqual(self.proof, {})


if __name__ == "__main__":
    unittest.main()
