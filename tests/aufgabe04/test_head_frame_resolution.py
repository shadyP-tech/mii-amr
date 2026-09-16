"""Only current supporting rails deduplicate physical head measurements."""

from itertools import permutations
from dataclasses import replace
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.perception.stand_axis.head_border_families import CurrentBorderFamilies
from scripts.aufgabe04.perception.stand_axis.head_frame_resolution import (
    distinct_current_frames, resolved_current_frame,
    resolved_tested_head_hint, resolve_measured_current_frames, CurrentFrameResolutionRecord,
)
from scripts.aufgabe04.perception.stand_axis.head_outer_border import HeadOuterBorderEvidence
from scripts.aufgabe04.perception.stand_axis.head_proposal_selection import select_verified_head
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None


def square(low, high, *, dx=0.):
    return tuple(ImagePoint(x + dx, y)
                 for x, y in ((low, low), (high, low), (high, high), (low, high)))


def frame(corners):
    return SimpleNamespace(corners=corners)


def recovered_record(original, recovered, *, hint=None):
    evidence = HeadOuterBorderEvidence(True, "current_raw_outer_head_recovered",
        original, recovered.corners, "measured-profile", recovered=True,
        neutral_proposal_corners=hint or original,
        current_raw_alternatives=(original, recovered.corners),
        head_size_m=(.078, .078), largest_inset_size_m=(.071, .071), model_tolerance_m=.002)
    return CurrentFrameResolutionRecord(hint or original, original, recovered, evidence)


def widen_left(corners, pixels):
    return tuple(ImagePoint(point.u_px - (pixels if index in (0, 3) else 0.), point.v_px)
                 for index, point in enumerate(corners))


class HeadFrameResolutionTests(unittest.TestCase):
    def test_completed_one_side_recovery_resolves_same_measured_initial_frame(self):
        # Recorded opposite-side frame 6: both hints measured the same frame,
        # but one current raw recovery found its left rail seven pixels outward.
        original = square(50., 150.)
        first = frame(square(50., 150., dx=.3))
        recovered = frame(widen_left(original, 7.))
        entry = recovered_record(original, recovered, hint=square(54., 146.))
        families = CurrentBorderFamilies()
        self.assertFalse(families.same(first.corners, recovered.corners))
        self.assertFalse(entry.recovery_evidence.independently_resolved)
        resolved = resolve_measured_current_frames([first, recovered], [entry], families)
        self.assertTrue(all(item is recovered for item in resolved))
        self.assertEqual(distinct_current_frames(resolved, families), [recovered])
        # A completed measured relation does not relax untested neutral hints.
        self.assertIsNone(resolved_tested_head_hint(first.corners, [entry], families))

    def test_enclosure_without_completed_raw_recovery_does_not_resolve_a_frame(self):
        original = frame(square(50., 150.))
        outer = frame(widen_left(original.corners, 7.))
        entry = recovered_record(original.corners, outer)
        incomplete = replace(entry.recovery_evidence,
                             current_raw_alternatives=(original.corners,))
        entries = (replace(entry, recovery_evidence=None),
                   replace(entry, recovery_evidence=incomplete),
                   replace(entry, recovery_evidence=replace(entry.recovery_evidence, recovered=False)))
        for current in entries:
            with self.subTest(evidence=current.recovery_evidence):
                result = resolve_measured_current_frames([original, outer], [current], CurrentBorderFamilies())
                self.assertIs(result[0], original)

    def test_completed_recovery_cannot_import_an_unassociated_final_proposal(self):
        original = frame(square(50., 150.))
        unassociated = frame(widen_left(original.corners, 7.))
        result = resolve_measured_current_frames([original],
            [recovered_record(original.corners, unassociated)], CurrentBorderFamilies())
        self.assertEqual(result, [original])

    def test_conflicting_completed_recoveries_preserve_original_and_both_finals(self):
        original = frame(square(50., 150.))
        left = frame(widen_left(original.corners, 7.))
        right = frame(tuple(ImagePoint(point.u_px + (7. if index in (1, 2) else 0.), point.v_px)
                            for index, point in enumerate(original.corners)))
        entries = [recovered_record(original.corners, target) for target in (left, right)]
        for ordered in permutations([original, left, right]):
            families = CurrentBorderFamilies()
            resolved = resolve_measured_current_frames(ordered, entries, families)
            remaining = distinct_current_frames(resolved, families)
            self.assertEqual({id(item) for item in remaining}, {id(original), id(left), id(right)})

    def test_completed_recovery_substitution_does_not_follow_transitive_edges(self):
        first = frame(square(50., 150.))
        middle = frame(widen_left(first.corners, 7.))
        last = frame(widen_left(first.corners, 14.))
        entries = [recovered_record(first.corners, middle), recovered_record(middle.corners, last)]
        families = CurrentBorderFamilies()
        resolved = resolve_measured_current_frames([first, middle, last], entries, families)
        remaining = distinct_current_frames(resolved, families)
        self.assertEqual({id(item) for item in remaining}, {id(middle), id(last)})

    def test_tested_current_inset_mapping_covers_only_its_same_rail_aliases(self):
        hint, original = square(50., 150.), square(50., 150., dx=.3)
        canonical = frame(square(40., 160.))
        entry = CurrentFrameResolutionRecord(hint, original, canonical)
        families = CurrentBorderFamilies()
        later_hint = square(50., 150., dx=.6)
        self.assertFalse(families.same(later_hint, canonical.corners))
        self.assertIs(resolved_tested_head_hint(later_hint, [entry], families), canonical)
        # An unrelated contained rectangle does not inherit this measured map.
        self.assertIsNone(resolved_tested_head_hint(square(58., 142.), [entry], families))
        self.assertIsNone(resolved_tested_head_hint(later_hint, [], families))

    def test_tested_mapping_retains_area_guard_at_the_observed_origin(self):
        origin = square(50., 60.)
        entry = CurrentFrameResolutionRecord(origin, origin, frame(square(40., 70.)))
        outward_hint = square(49.5, 60.5)
        families = CurrentBorderFamilies()
        self.assertTrue(families.same(outward_hint, origin))
        self.assertIsNone(resolved_tested_head_hint(outward_hint, [entry], families))

    def test_tested_mapping_rejects_conflicting_final_frames(self):
        origin = square(50., 150.)
        entries = [CurrentFrameResolutionRecord(origin, origin, frame(square(low, high)))
                   for low, high in ((40., 160.), (35., 165.))]
        self.assertIsNone(resolved_tested_head_hint(origin, entries, CurrentBorderFamilies()))

    def test_tested_mapping_does_not_bridge_different_original_measurements(self):
        left, middle, right = (square(50., 150., dx=delta) for delta in (0., 1.2, 2.4))
        canonical = frame(square(40., 165.))
        entries = [CurrentFrameResolutionRecord(middle, origin, canonical)
                   for origin in (left, right)]
        self.assertIsNone(resolved_tested_head_hint(middle, entries, CurrentBorderFamilies()))

    def test_same_original_with_different_neutral_hint_is_not_a_tested_alias(self):
        original = square(50., 150.)
        tested_hint = square(50., 150., dx=5.)
        entry = CurrentFrameResolutionRecord(tested_hint, original, frame(square(40., 160.)))
        families = CurrentBorderFamilies()
        self.assertTrue(families.same(original, entry.original_corners))
        self.assertFalse(families.same(original, entry.hint_corners))
        self.assertIsNone(resolved_tested_head_hint(original, [entry], families))

    def test_alias_cannot_bridge_its_tested_hint_and_initial_measurement(self):
        left, middle, right = (square(50., 150., dx=delta) for delta in (0., 1.2, 2.4))
        entry = CurrentFrameResolutionRecord(left, right, frame(square(40., 165.)))
        families = CurrentBorderFamilies()
        self.assertTrue(families.same(middle, left))
        self.assertTrue(families.same(middle, right))
        self.assertIsNone(resolved_tested_head_hint(middle, [entry], families))

    def test_outward_hint_keeps_existing_area_guard_even_with_nearby_rails(self):
        measured, outward = frame(square(40., 50.)), square(39.5, 50.5)
        families = CurrentBorderFamilies()
        self.assertTrue(families.same(outward, measured.corners))
        self.assertIsNone(resolved_current_frame(outward, [measured], families))
        self.assertIs(resolved_current_frame(measured.corners, [measured], families), measured)

    def test_intermediate_hint_cannot_resolve_two_disagreeing_measured_frames(self):
        left, middle, right = (square(40., 140., dx=delta) for delta in (0., 1.2, 2.4))
        families = CurrentBorderFamilies()
        self.assertTrue(families.same(left, middle))
        self.assertTrue(families.same(middle, right))
        self.assertFalse(families.same(left, right))
        self.assertIsNone(resolved_current_frame(middle, [frame(left), frame(right)], families))

    def test_bridge_preserves_all_conflicting_evidence_in_every_input_order(self):
        proposals = [frame(square(40., 140., dx=delta)) for delta in (0., 1.2, 2.4)]
        for ordered in permutations(proposals):
            with self.subTest(order=[item.corners[0].u_px for item in ordered]):
                remaining = distinct_current_frames(ordered, CurrentBorderFamilies())
                self.assertEqual({id(item) for item in remaining}, {id(item) for item in proposals})

    def test_isolated_pairwise_agreement_can_remove_only_its_own_aliases(self):
        aliases = [frame(square(40., 140., dx=delta)) for delta in (0., .25, .5)]
        unrelated = frame(square(160., 190.))
        remaining = distinct_current_frames([*aliases, unrelated], CurrentBorderFamilies())
        self.assertEqual(len(remaining), 2)
        self.assertIn(id(unrelated), {id(item) for item in remaining})
        self.assertEqual(sum(id(item) in {id(alias) for alias in aliases}
                             for item in remaining), 1)

    @unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
    def test_two_complete_nested_raw_frames_remain_competing(self):
        outer, inner = square(40., 160.), square(48., 152.)
        bgr = np.full((200, 200, 3), 255, np.uint8)
        for corners in (outer, inner):
            cv2.polylines(bgr, [np.asarray([(p.u_px, p.v_px) for p in corners], np.int32)],
                          True, (0, 0, 0), 1)
        raw = cv2.cvtColor(255 - bgr, cv2.COLOR_BGR2GRAY)
        families = CurrentBorderFamilies(raw, bgr)
        self.assertFalse(families.same(outer, inner))
        self.assertIsNone(resolved_current_frame(inner, [frame(outer)], families))
        self.assertIsNone(resolved_tested_head_hint(inner,
            [CurrentFrameResolutionRecord(outer, outer, frame(outer))], families))
        remaining = distinct_current_frames([frame(outer), frame(inner)], families)
        self.assertEqual(len(remaining), 2)
        selected, reason, _diagnostic = select_verified_head(cv2, remaining,
            raw_edges=raw, frame_bgr=bgr, border_families=families)
        self.assertIsNone(selected)
        self.assertEqual(reason, "head_proposal_ambiguous")

    @unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
    def test_preserved_bridge_still_rejects_downstream_instead_of_remerging(self):
        proposals = [frame(square(40., 140., dx=delta)) for delta in (0., 1.2, 2.4)]
        for ordered in permutations(proposals):
            families = CurrentBorderFamilies()
            remaining = distinct_current_frames(ordered, families)
            selected, reason, _diagnostic = select_verified_head(
                cv2, remaining, border_families=families)
            self.assertIsNone(selected)
            self.assertEqual(reason, "head_proposal_ambiguous")


if __name__ == "__main__":
    unittest.main()
