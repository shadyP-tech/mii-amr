"""Real pixels exercise cold acquisition, strict current fits and rejection."""

import math
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.real_robot.observer.backside_proposal_reuse import BacksideProposalReuse
from tests.aufgabe04.recorded_backside_fixture import RecordedBacksideFixture


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required for recorded images")
class RecordedBacksideHandoffTests(unittest.TestCase):
    def setUp(self):
        self.fixture = RecordedBacksideFixture(cv2, np)

    def assert_usable_registered_backside(self, selection):
        selected = selection.selected
        self.assertTrue(selection.registered)
        self.assertTrue(selected.estimate.usable)
        self.assertEqual(selected.estimate.evidence_state, "fresh_backside")
        self.assertEqual(selected.estimate.source, "model_backside_current_frame")
        self.assertEqual(selected.qr_observations, ())
        self.assertFalse(selected.debug.qr_detected)
        self.assertIsNone(selected.debug.model_pose)
        self.assertAlmostEqual(selected.estimate.yaw_deg, -2.631965, places=4)
        binding = self.fixture.registered_binding("frame_000008", selection)
        self.assertTrue(binding.associated)
        self.assertAlmostEqual(binding.search_association.distance_m, 0.758, places=3)
        self.assertAlmostEqual(math.degrees(binding.camera_map_bearing_delta_rad), 8.503256, places=4)

    def test_actual_cold_acquisition_matches_legacy_fit_with_exact_crop_cache(self):
        legacy, old_meta = self.fixture.evaluate(
            "frame_000008", BacksideProposalReuse(), cache_inputs=False
        )
        cold, meta = self.fixture.evaluate("frame_000008", BacksideProposalReuse())
        self.assert_usable_registered_backside(legacy)
        self.assert_usable_registered_backside(cold)
        self.assertEqual(len(old_meta["calls"]), 3)
        self.assertEqual(len(meta["calls"]), 3)
        self.assertEqual(
            cold.selected.estimate.corners, legacy.selected.estimate.corners
        )
        self.assertEqual([c["qr_decode"]["cache_hit"] for c in meta["calls"]], [False, False, True])
        self.assertTrue(meta["calls"][2]["input_cache"]["edge_preprocessing"]["cache_hit"])
        self.assertTrue(meta["calls"][2]["input_cache"]["qr_detection"]["cache_hit"])

    def test_warmed_hint_refits_original_pixels_and_redecodes_qr_for_current_array(self):
        # Repeated content under a synthetic test clock is a performance/fit
        # regression, never seven independent frames or historical freshness.
        reuse = BacksideProposalReuse()
        cold, _ = self.fixture.evaluate("frame_000008", reuse, test_stamp=100.0)
        warm, meta = self.fixture.evaluate("frame_000008", reuse, test_stamp=100.1)
        self.assert_usable_registered_backside(warm)
        self.assertIsNot(warm.selected.frame, cold.selected.frame)
        self.assertIsNot(warm.selected.estimate, cold.selected.estimate)
        self.assertEqual(len(meta["calls"]), 1)
        self.assertEqual(meta["proposal_reuse"]["mode"], "strict_current_image_hint")
        self.assertFalse(meta["proposal_reuse"]["measurement_reused"])
        self.assertEqual(meta["calls"][0]["qr_decode"]["mode"], "full")
        self.assertFalse(meta["calls"][0]["qr_decode"]["cache_hit"])
        self.assertFalse(meta["calls"][0]["input_cache"]["edge_preprocessing"]["cache_hit"])
        self.assertFalse(meta["calls"][0]["input_cache"]["qr_detection"]["cache_hit"])

    def test_subsequent_actual_geometry_failure_clears_hint_without_old_angle(self):
        reuse = BacksideProposalReuse()
        cold, _ = self.fixture.evaluate("frame_000008", reuse)
        self.assertTrue(cold.selected.estimate.usable)
        failed, meta = self.fixture.evaluate("frame_000010", reuse)
        self.assertEqual(len(meta["calls"]), 1)
        self.assertEqual(meta["proposal_reuse"]["mode"], "strict_current_image_hint")
        self.assertFalse(meta["proposal_reuse"]["hint_retained"])
        self.assertFalse(failed.selected.estimate.usable)
        self.assertIsNone(failed.selected.estimate.yaw_deg)
        self.assertEqual(failed.selected.estimate.reason, "model_backside_head_and_neck_unavailable")
        self.assertFalse(meta["calls"][0]["qr_decode"]["cache_hit"])


if __name__ == "__main__":
    unittest.main()
