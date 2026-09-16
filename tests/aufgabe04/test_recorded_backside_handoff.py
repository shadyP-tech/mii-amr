"""Historical pixels verify current selection/binding, not angle ground truth.

The adjacent frames still choose slightly different right rails (about three
pixels, roughly six degrees). These tests must not be read as stable multi-frame
orientation or seven fresh live observations. They verify current evidence and
that an unresolved current marker blocks reuse of an older backside label.
"""

import math
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.real_robot.observer.backside_proposal_reuse import BacksideProposalReuse
from scripts.aufgabe04.perception.stand_axis.head_model_quality import validated_head_model_quality
from tests.aufgabe04.recorded_backside_fixture import RecordedBacksideFixture


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required for recorded images")
class RecordedBacksideHandoffTests(unittest.TestCase):
    def setUp(self):
        self.fixture = RecordedBacksideFixture(cv2, np)

    def assert_current_registered_head(self, selection, *, distance_m=.758):
        selected = selection.selected
        self.assertTrue(selection.registered, selected.estimate.reason)
        self.assertTrue(selected.estimate.usable, selected.estimate.reason)
        self.assertEqual(selected.estimate.source,
            "model_backside_current_frame" if selected.estimate.visible_face == "backside_candidate"
            else "model_current_measured_head")
        self.assertEqual(selected.debug.model_pose_fit_source, "model_current_measured_head")
        self.assertEqual(selected.qr_observations, ())
        self.assertIsNotNone(selected.debug.model_pose)
        self.assertTrue(math.isfinite(selected.estimate.yaw_deg))
        self.assertTrue(validated_head_model_quality(selected.debug.head_model_quality))
        self.assertTrue(selected.debug.head_model_quality.raw_corner_support_accepted)
        self.assertEqual(len(selected.estimate.corners), 4)
        current_binding = selected.debug.head_acquisition_diagnostics["selected_border_binding"]
        self.assertTrue(current_binding["performed"])
        self.assertTrue(current_binding["accepted"])
        self.assertFalse(current_binding["historical_measurement_reused"])
        binding = selection.head_acquisition["lidar_association"]
        self.assertTrue(binding["associated"])
        self.assertAlmostEqual(binding["distance_m"], distance_m, places=3)
        self.assertLess(math.degrees(binding["camera_map_bearing_delta_rad"]), 12.)

    def test_actual_cold_acquisition_is_identical_with_exact_crop_cache(self):
        uncached, old_meta = self.fixture.evaluate(
            "frame_000008", BacksideProposalReuse(), cache_inputs=False
        )
        cold, meta = self.fixture.evaluate("frame_000008", BacksideProposalReuse())
        self.assert_current_registered_head(uncached)
        self.assert_current_registered_head(cold)
        self.assertTrue(cold.selected.debug.head_backside_classification.accepted)
        self.assertFalse(cold.selected.debug.qr_detected)
        self.assertEqual(len(old_meta["calls"]), 2)
        self.assertEqual(len(meta["calls"]), 2)
        self.assertEqual(cold.selected.estimate.corners, uncached.selected.estimate.corners)
        self.assertEqual(cold.selected.estimate.yaw_deg, uncached.selected.estimate.yaw_deg)
        self.assertEqual([c["qr_decode"]["cache_hit"] for c in meta["calls"]], [False, False])
        self.assertTrue(meta["head_acquisition"]["candidate_associated"])

    def test_repeated_content_refits_current_pixels_and_only_retains_search_hint(self):
        # This synthetic test clock does not manufacture independent images or
        # live freshness. The same saved image is processed twice from scratch.
        reuse = BacksideProposalReuse()
        cold, _ = self.fixture.evaluate("frame_000008", reuse, test_stamp=100.0)
        warm, meta = self.fixture.evaluate("frame_000008", reuse, test_stamp=100.1)
        self.assert_current_registered_head(warm)
        self.assertIsNot(warm.selected.frame, cold.selected.frame)
        self.assertIsNot(warm.selected.estimate, cold.selected.estimate)
        self.assertEqual(len(meta["calls"]), 2)
        self.assertTrue(meta["proposal_reuse"]["hint_retained"])
        self.assertTrue(meta["proposal_reuse"]["complete_head_reverified"])
        self.assertFalse(meta["proposal_reuse"]["measurement_reused"])
        self.assertEqual(meta["calls"][0]["qr_decode"]["mode"], "full")
        self.assertFalse(meta["calls"][0]["qr_decode"]["cache_hit"])
        self.assertFalse(meta["calls"][0]["input_cache"]["edge_preprocessing"]["cache_hit"])
        # Native marker work is optional after an unavailable head. If run on
        # a complete current head, it must not reuse another image's result.
        native = meta["calls"][0]["input_cache"].get("qr_detection")
        if native is not None:
            self.assertFalse(native["cache_hit"])

    def test_subsequent_current_marker_uncertainty_cannot_inherit_backside_hint(self):
        reuse = BacksideProposalReuse()
        cold, _ = self.fixture.evaluate("frame_000008", reuse)
        self.assert_current_registered_head(cold)
        current, meta = self.fixture.evaluate("frame_000010", reuse)
        self.assert_current_registered_head(current, distance_m=.757)
        self.assertEqual(len(meta["calls"]), 2)
        self.assertFalse(meta["proposal_reuse"]["hint_retained"])
        self.assertFalse(meta["proposal_reuse"]["measurement_reused"])
        self.assertIsNot(current.selected.estimate, cold.selected.estimate)
        # This is evidence of the remaining rail-choice instability, not proof
        # that either single-frame angle equals the physical stand orientation.
        self.assertNotEqual(current.selected.estimate.yaw_deg, cold.selected.estimate.yaw_deg)
        self.assertTrue(current.selected.debug.qr_detected)
        self.assertEqual(current.selected.debug.qr_marker_reason, "invalid_marker_quadrilateral")
        self.assertFalse(current.selected.debug.head_backside_classification.accepted)
        self.assertEqual(current.selected.debug.head_backside_classification.reason,
                         "backside_current_marker_absence_required")
        self.assertIsNone(current.selected.estimate.visible_face)
        self.assertFalse(meta["calls"][0]["qr_decode"]["cache_hit"])


if __name__ == "__main__":
    unittest.main()
