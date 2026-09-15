"""Cold acquisition is QR/neck independent and never fills missing borders."""

from unittest import mock
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import (
    MAX_CONTOURS, MAX_IMAGE_PIXELS, MAX_RAW_VERIFICATIONS, acquire_cold_head_proposal,
)
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame


def frame_with_heads(heads=(((190, 130), (100, 100), 0),), missing_side=None, neck=False):
    frame = np.zeros((360, 520, 3), np.uint8)
    for rectangle in heads:
        corners = np.rint(cv2.boxPoints(rectangle)).astype(int)
        for index, (a, b) in enumerate(zip(corners, np.roll(corners, -1, axis=0))):
            if index != missing_side:
                cv2.line(frame, tuple(a), tuple(b), (200, 200, 200), 2)
        if neck:
            x, y = rectangle[0]
            cv2.line(frame, (int(x - 6), int(y + 50)), (int(x - 6), int(y + 95)), (200, 200, 200), 2)
            cv2.line(frame, (int(x + 6), int(y + 50)), (int(x + 6), int(y + 95)), (200, 200, 200), 2)
    return frame


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class ColdHeadAcquisitionTests(unittest.TestCase):
    def test_complete_plain_head_needs_no_qr_neck_or_pose(self):
        result = acquire_cold_head_proposal(cv2, frame_with_heads())
        self.assertIsNotNone(result.proposal, result)
        self.assertAlmostEqual(result.proposal.center_u_px, 190., delta=2.)
        self.assertGreater(result.proposal.raw_edge_support, .90)
        self.assertLessEqual(result.raw_verifications, MAX_RAW_VERIFICATIONS)
        self.assertFalse(result.joint_border_diagnostics["angle_authorized"])
        self.assertIsNone(getattr(result.proposal, "yaw_deg", None))

    def test_off_center_rotated_head_is_located_from_current_pixels(self):
        result = acquire_cold_head_proposal(cv2, frame_with_heads((((405, 260), (100, 100), 15),)))
        self.assertIsNotNone(result.proposal, result)
        self.assertAlmostEqual(result.proposal.center_u_px, 405., delta=2.)
        self.assertAlmostEqual(result.proposal.center_v_px, 260., delta=2.)

    def test_neck_presence_does_not_change_head_acquisition(self):
        result = acquire_cold_head_proposal(cv2, frame_with_heads(neck=True))
        self.assertIsNotNone(result.proposal, result)
        self.assertAlmostEqual(result.proposal.observed_height_px, 100., delta=4.)

    def test_equal_luminance_colour_head_with_neck_uses_current_channel_edges(self):
        frame = np.empty((360, 520, 3), np.uint8)
        frame[:] = (255, 0, 0)
        cv2.rectangle(frame, (190, 100), (290, 200), (0, 0, 97), -1)
        cv2.rectangle(frame, (233, 200), (247, 270), (0, 0, 97), -1)
        self.assertEqual(len(np.unique(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))), 1)
        result = acquire_cold_head_proposal(cv2, frame)
        self.assertIsNotNone(result.proposal, result)
        self.assertAlmostEqual(result.proposal.center_u_px, 240., delta=3.)
        self.assertAlmostEqual(result.proposal.observed_height_px, 100., delta=4.)

    def test_high_angle_current_head_keeps_the_metric_pixel_quality_gate(self):
        from tests.aufgabe04.test_head_model_angle_reference import HeadModelAngleReferenceTest
        from scripts.aufgabe04.perception.stand_axis.head_model_fit import fit_current_measured_head
        HeadModelAngleReferenceTest.setUpClass()
        fixture = HeadModelAngleReferenceTest()
        corners, camera = fixture.projection(75.)
        raw = fixture.raster(corners, include_neck=False)
        result = acquire_cold_head_proposal(cv2, cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), raw_edges=raw)
        self.assertIsNotNone(result.proposal, result)
        estimate, _debug, _pose = fit_current_measured_head(
            cv2, raw, model_profile=fixture.profile, camera=camera,
            proposal_corners=result.proposal.corners,
        )
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertAlmostEqual(estimate.yaw_deg, -75., delta=2.)

    def test_blank_open_and_clipped_heads_remain_unavailable(self):
        frames = [np.zeros((360, 520, 3), np.uint8), frame_with_heads()[:, :205]]
        frames.extend(frame_with_heads(missing_side=side) for side in range(4))
        for index, frame in enumerate(frames):
            with self.subTest(index=index):
                self.assertIsNone(acquire_cold_head_proposal(cv2, frame).proposal)

    def test_two_distinct_heads_are_ambiguous_even_with_different_sizes(self):
        result = acquire_cold_head_proposal(cv2, frame_with_heads((
            ((130, 130), (110, 110), 0), ((370, 220), (70, 70), 0),
        )))
        self.assertIsNone(result.proposal)
        self.assertEqual(result.reason, "head_proposal_ambiguous")

    def test_large_enclosing_background_cannot_swallow_a_smaller_head(self):
        for outside, inside in (((250, 250), (80, 80)), ((400, 300), (80, 80)), ((200, 200), (60, 60))):
            with self.subTest(outside=outside, inside=inside):
                result = acquire_cold_head_proposal(cv2, frame_with_heads((
                    ((260, 180), outside, 0), ((260, 180), inside, 0),
                )))
                self.assertIsNone(result.proposal)
                self.assertEqual(result.reason, "head_proposal_ambiguous")

    def test_recorded_backside_acquires_without_expected_geometry_or_qr(self):
        from tests.aufgabe04.recorded_head_proposal_fixture import recorded_head_proposal_image
        result = acquire_cold_head_proposal(cv2, recorded_head_proposal_image(cv2, np, "back22"))
        self.assertIsNotNone(result.proposal, result)
        self.assertAlmostEqual(result.proposal.center_u_px, 80.5, delta=3.)
        self.assertAlmostEqual(result.proposal.center_v_px, 54., delta=3.)
        self.assertGreater(result.proposal.raw_edge_support, .90)
        self.assertLessEqual(result.raw_verifications, MAX_RAW_VERIFICATIONS)

    def test_rectification_canvas_boundary_is_not_a_head(self):
        frame = np.zeros((360, 520, 3), np.uint8)
        cv2.rectangle(frame, (8, 8), (511, 351), (200, 200, 200), 2)
        self.assertIsNone(acquire_cold_head_proposal(cv2, frame).proposal)

    def test_current_raw_pixels_are_required_and_not_modified(self):
        frame = frame_with_heads()
        edges = _canny_edges_from_frame(cv2, frame, edge_preprocess="channel_union",
                                        blur_kernel=5, canny_low=20, canny_high=60)
        original_frame, original_edges = frame.copy(), edges.copy()
        self.assertIsNotNone(acquire_cold_head_proposal(cv2, frame, raw_edges=edges).proposal)
        np.testing.assert_array_equal(frame, original_frame)
        np.testing.assert_array_equal(edges, original_edges)
        self.assertIsNone(acquire_cold_head_proposal(cv2, frame, raw_edges=np.zeros_like(edges)).proposal)

    def test_workload_exhaustion_cannot_choose_first_head(self):
        frame = frame_with_heads()
        with mock.patch.object(cv2, "findContours", return_value=([np.zeros((4, 1, 2), np.int32)] * (MAX_CONTOURS + 1), None)):
            result = acquire_cold_head_proposal(cv2, frame)
        self.assertEqual(result.reason, "head_cold_acquisition_contour_budget_exceeded")
        self.assertEqual(result.raw_verifications, 0)
        self.assertIsNone(result.proposal)
        oversized = np.zeros((1081, 1920, 3), np.uint8)
        self.assertGreater(oversized.shape[0] * oversized.shape[1], MAX_IMAGE_PIXELS)
        self.assertEqual(acquire_cold_head_proposal(cv2, oversized).reason,
                         "head_cold_acquisition_image_budget_exceeded")


if __name__ == "__main__":
    unittest.main()
