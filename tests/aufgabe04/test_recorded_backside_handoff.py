"""Real pixels exercise cold acquisition, strict current fits and rejection."""

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

    def assert_rejected_registered_head(self, selection):
        selected = selection.selected
        self.assertTrue(selection.registered)
        self.assertFalse(selected.estimate.usable)
        self.assertEqual(selected.estimate.reason, "model_corner_evidence_insufficient")
        self.assertEqual(selected.estimate.source, "model_current_measured_head")
        self.assertEqual(selected.debug.model_pose_fit_source, "model_current_measured_head")
        self.assertFalse(selected.debug.head_backside_classification.accepted)
        self.assertEqual(selected.qr_observations, ())
        self.assertFalse(selected.debug.qr_detected)
        self.assertIsNone(selected.debug.model_pose)
        # Current joint acquisition reaches a larger border than historical
        # locators. Its failed strict corner support cannot inherit an older
        # fit, angle or side classification.
        self.assertIsNone(selected.estimate.yaw_deg)
        self.assertFalse(validated_head_model_quality(selected.debug.head_model_quality))
        self.assertFalse(selected.debug.head_model_quality.raw_corner_support_accepted)
        self.assertIsNone(selected.estimate.corners)
        self.assertIsNone(selected.estimate.visible_face)
        self.assertIsNone(selected.estimate.camera_face_normal_xyz)
        binding = selection.head_acquisition["lidar_association"]
        self.assertTrue(binding["associated"])
        self.assertAlmostEqual(binding["distance_m"], 0.758, places=3)
        self.assertLess(math.degrees(binding["camera_map_bearing_delta_rad"]), 12.)

    def test_actual_cold_acquisition_is_identical_with_exact_crop_cache(self):
        uncached, old_meta = self.fixture.evaluate(
            "frame_000008", BacksideProposalReuse(), cache_inputs=False
        )
        cold, meta = self.fixture.evaluate("frame_000008", BacksideProposalReuse())
        self.assert_rejected_registered_head(uncached)
        self.assert_rejected_registered_head(cold)
        self.assertEqual(len(old_meta["calls"]), 2)
        self.assertEqual(len(meta["calls"]), 2)
        self.assertEqual(
            cold.selected.estimate.corners, uncached.selected.estimate.corners
        )
        self.assertEqual([c["qr_decode"]["cache_hit"] for c in meta["calls"]], [False, False])
        self.assertTrue(meta["head_acquisition"]["candidate_associated"])

    def test_repeated_rejected_head_refits_current_pixels_without_retaining_a_hint(self):
        # Repeated content under a synthetic test clock is a performance/fit
        # regression, never seven independent frames or historical freshness.
        reuse = BacksideProposalReuse()
        cold, _ = self.fixture.evaluate("frame_000008", reuse, test_stamp=100.0)
        warm, meta = self.fixture.evaluate("frame_000008", reuse, test_stamp=100.1)
        self.assert_rejected_registered_head(warm)
        self.assertIsNot(warm.selected.frame, cold.selected.frame)
        self.assertIsNot(warm.selected.estimate, cold.selected.estimate)
        self.assertEqual(len(meta["calls"]), 2)
        self.assertFalse(meta["proposal_reuse"]["hint_retained"])
        self.assertFalse(meta["proposal_reuse"]["measurement_reused"])
        self.assertEqual(meta["calls"][0]["qr_decode"]["mode"], "full")
        self.assertFalse(meta["calls"][0]["qr_decode"]["cache_hit"])
        self.assertFalse(meta["calls"][0]["input_cache"]["edge_preprocessing"]["cache_hit"])
        self.assertFalse(meta["calls"][0]["input_cache"]["qr_detection"]["cache_hit"])

    def test_subsequent_rejected_border_has_no_old_angle_or_side_hint(self):
        reuse = BacksideProposalReuse()
        cold, _ = self.fixture.evaluate("frame_000008", reuse)
        self.assert_rejected_registered_head(cold)
        failed, meta = self.fixture.evaluate("frame_000010", reuse)
        self.assertEqual(len(meta["calls"]), 2)
        self.assertFalse(meta["proposal_reuse"]["hint_retained"])
        self.assertFalse(failed.selected.estimate.usable)
        self.assertIsNone(failed.selected.estimate.yaw_deg)
        self.assertEqual(failed.selected.estimate.reason, "model_corner_evidence_insufficient")
        self.assertFalse(meta["calls"][0]["qr_decode"]["cache_hit"])


if __name__ == "__main__":
    unittest.main()
