"""Recentered crop selection cannot erase current-image marker conflicts."""

from types import SimpleNamespace
import unittest

from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
from scripts.aufgabe04.real_robot.observer.front_observation import front_observation_decision
from scripts.aufgabe04.real_robot.observer.roi_qr_evidence import summarize_roi_qr_evidence


QUAD = ((10., 10.), (30., 10.), (30., 30.), (10., 30.))


def observation(text="QR_003", corners=QUAD):
    return DecodedQrObservation(text, corners, "test_decoder")


def evaluated(*observations, roi=ImageRoi(0, 0, 100, 100, 30), marker=False, verified=False):
    return SimpleNamespace(
        attempt=SimpleNamespace(roi=roi), qr_observations=observations,
        debug=SimpleNamespace(qr_detected=marker, qr_marker_verified=verified),
    )


def summarize(*evaluations):
    return summarize_roi_qr_evidence(SimpleNamespace(
        evaluations=evaluations, selected=evaluations[-1],
    ))


class RoiQrEvidenceTest(unittest.TestCase):
    def test_different_nominal_and_recentered_identities_conflict(self):
        result = summarize(evaluated(observation("QR_001")), evaluated(observation("QR_002")))
        self.assertEqual(result.qr_texts, ("QR_001", "QR_002"))
        self.assertEqual(result.symbol_count, 2)
        self.assertEqual(result.conflict_reason, "conflicting_qr_identities_across_rois")
        self.assertTrue(result.marker_verified)

    def test_two_symbols_in_one_crop_conflict_even_with_same_text(self):
        result = summarize(evaluated(observation(), observation()), evaluated(observation()))
        self.assertEqual(result.conflict_reason, "multiple_qr_symbols_in_evaluated_roi")
        self.assertEqual(result.symbol_count, 2)

    def test_crop_adjusted_same_symbol_is_counted_once(self):
        # The second crop's symbol corners differ but identify the same pixels.
        other_roi = ImageRoi(5, 6, 70, 80, 30)
        shifted = tuple((u - 5, v - 6) for u, v in QUAD)
        result = summarize(evaluated(observation()), evaluated(observation(corners=shifted), roi=other_roi))
        self.assertEqual(result.symbol_count, 1)
        self.assertIsNone(result.conflict_reason)

    def test_same_local_corners_in_disjoint_crops_are_distinct_symbols(self):
        result = summarize(evaluated(observation()), evaluated(
            observation(), roi=ImageRoi(60, 0, 160, 100, 30),
        ))
        self.assertEqual(result.symbol_count, 2)
        self.assertEqual(result.conflict_reason, "disjoint_qr_symbols_across_rois")

    def test_small_corner_bias_and_reverse_winding_preserve_same_symbol(self):
        biased = tuple((u + 2, v + 1) for u, v in reversed(QUAD))
        result = summarize(evaluated(observation()), evaluated(observation(corners=biased)))
        self.assertEqual(result.symbol_count, 1)
        self.assertIsNone(result.conflict_reason)

    def test_bbox_overlap_does_not_merge_disjoint_diagonal_quads(self):
        first = ((10., 10.), (30., 30.), (27., 33.), (7., 13.))
        second = ((20., 10.), (40., 30.), (37., 33.), (17., 13.))
        result = summarize(evaluated(observation(corners=first)), evaluated(observation(corners=second)))
        self.assertEqual(result.symbol_count, 2)
        self.assertEqual(result.conflict_reason, "disjoint_qr_symbols_across_rois")

    def test_overlap_chain_cannot_hide_disjoint_first_and_last_markers(self):
        second = tuple((u + 8, v) for u, v in QUAD)
        third = tuple((u + 16, v) for u, v in QUAD)
        result = summarize(*(evaluated(observation(corners=corners)) for corners in (QUAD, second, third)))
        self.assertEqual(result.symbol_count, 2)
        self.assertIsNotNone(result.conflict_reason)

    def test_verified_marker_outside_selected_crop_retains_front_veto(self):
        result = summarize(evaluated(marker=True, verified=True), evaluated())
        self.assertTrue(result.marker_verified)
        decision = front_observation_decision(
            qr_texts=result.qr_texts, qr_marker_detected=result.marker_detected,
            qr_marker_verified=result.marker_verified,
            estimate_source="model_backside_current_frame", marker_seen_in_stationary_epoch=False,
        )
        self.assertTrue(decision.withhold_backside_axis)
        self.assertTrue(decision.marker_seen_in_stationary_epoch)

    def test_tentative_nominal_quad_only_vetoes_current_frame(self):
        result = summarize(evaluated(marker=True, verified=False), evaluated())
        self.assertTrue(result.marker_detected)
        self.assertFalse(result.marker_verified)
        decision = front_observation_decision(
            qr_texts=result.qr_texts, qr_marker_detected=result.marker_detected,
            qr_marker_verified=result.marker_verified,
            estimate_source="model_backside_current_frame", marker_seen_in_stationary_epoch=False,
        )
        self.assertTrue(decision.withhold_backside_axis)
        self.assertFalse(decision.marker_seen_in_stationary_epoch)

    def test_text_without_geometry_retains_marker_but_no_new_binding(self):
        primary = evaluated(observation(corners=None))
        selected = evaluated()
        result = summarize(primary, selected)
        self.assertTrue(result.marker_verified)
        self.assertEqual(result.qr_texts, ("QR_003",))
        self.assertEqual(selected.qr_observations, ())
        self.assertEqual(result.metadata()["identity_authority"], "selected_crop_target_binding_only")
        self.assertFalse(result.metadata()["motion_authorized"])

    def test_text_only_mismatch_still_conflicts(self):
        result = summarize(evaluated(observation("A", None)), evaluated(observation("B")))
        self.assertEqual(result.conflict_reason, "conflicting_qr_identities_across_rois")

    def test_legacy_marker_verification_is_preserved_only_for_positive_marker(self):
        result = summarize(evaluated(marker=True, verified=None), evaluated())
        self.assertIsNone(result.marker_verified)
        self.assertTrue(result.marker_detected)
        result = summarize(evaluated(marker=False, verified=None), evaluated())
        self.assertFalse(result.marker_verified)
        self.assertFalse(result.marker_detected)

    def test_decoded_identity_overrides_legacy_or_unverified_marker_state(self):
        result = summarize(evaluated(marker=True, verified=None), evaluated(observation(), verified=False))
        self.assertTrue(result.marker_verified)
        self.assertTrue(result.marker_detected)

    def test_invalid_crop_geometry_cannot_create_false_spatial_binding(self):
        outside = tuple((u + 100, v) for u, v in QUAD)
        result = summarize(evaluated(observation(corners=outside)), evaluated(observation()))
        self.assertEqual(result.symbol_count, 1)
        self.assertTrue(result.marker_verified)
        self.assertEqual(result.qr_texts, ("QR_003",))


if __name__ == "__main__":
    unittest.main()
