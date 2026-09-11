"""A candidate crop follows observed complete heads before QR/PnP success."""

from dataclasses import fields
import unittest

try:
    import cv2
    import numpy
except ImportError:
    cv2 = numpy = None

from scripts.aufgabe04.perception.stand_axis.head_proposal import (
    HeadProposal, acquire_head_proposal,
)
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame
from tests.aufgabe04.recorded_head_proposal_fixture import (
    RECORDED_HEAD_PROPOSALS, recorded_head_proposal_image,
)


def _synthetic(*, heads=(270,), neck=True, missing_side=None, inset=False):
    frame = numpy.zeros((360, 520, 3), dtype=numpy.uint8)
    for x in heads:
        corners = ((x, 100), (x + 100, 100), (x + 100, 200), (x, 200))
        for index, (first, last) in enumerate(zip(corners, corners[1:] + corners[:1])):
            if index != missing_side:
                cv2.line(frame, first, last, (200, 200, 200), 2)
        if neck:
            for rail in (44, 56):
                cv2.line(frame, (x + rail, 201), (x + rail, 244), (200, 200, 200), 1)
        if inset:
            cv2.rectangle(frame, (x + 12, 112), (x + 88, 188), (200, 200, 200), 2)
    return frame


@unittest.skipIf(cv2 is None or numpy is None, "OpenCV and NumPy required")
class HeadProposalTests(unittest.TestCase):
    def acquire(self, frame, **options):
        return acquire_head_proposal(
            cv2, frame,
            expected_head_center_u_px=options.pop("expected_head_center_u_px", 260.0),
            expected_head_center_v_px=150.0,
            expected_head_height_px=options.pop("expected_head_height_px", 100.0),
            **options,
        )

    def test_off_center_head_is_acquired_without_qr_or_pose(self):
        result = self.acquire(_synthetic())
        self.assertEqual(result.reason, "current_head_proposal")
        self.assertAlmostEqual(result.proposal.center_u_px, 320.0, delta=3.0)
        self.assertGreater(result.proposal.center_offset_head_heights, 0.5)
        names = {field.name for field in fields(HeadProposal)}
        self.assertFalse(names & {"yaw_deg", "axis", "pose", "qr_identity", "front", "backside"})
        x0, y0, x1, y1 = result.proposal.bounds_xyxy
        self.assertLess(x0, 270)
        self.assertGreater(x1, 370)
        self.assertLess(y0, 100)
        self.assertGreater(y1, 230)

    def test_inner_printed_rectangle_does_not_replace_outer_head(self):
        result = self.acquire(_synthetic(inset=True))
        self.assertIsNotNone(result.proposal, result.reason)
        left, _top, right, _bottom = result.proposal.head_bounds_xyxy
        self.assertLess(left, 273)
        self.assertGreater(right, 367)

    def test_both_front_and_plain_backside_have_same_neutral_geometry(self):
        plain = self.acquire(_synthetic())
        printed = self.acquire(_synthetic(inset=True))
        self.assertEqual(plain.proposal.corners, printed.proposal.corners)

    def test_missing_each_raw_border_is_rejected(self):
        for side in range(4):
            with self.subTest(side=side):
                self.assertIsNone(self.acquire(_synthetic(missing_side=side)).proposal)

    def test_rectangle_without_centered_paired_neck_is_rejected(self):
        self.assertIsNone(self.acquire(_synthetic(neck=False)).proposal)

    def test_clipped_head_is_not_expanded_into_a_measurement(self):
        self.assertIsNone(self.acquire(_synthetic()[:, :330]).proposal)

    def test_two_distinct_complete_heads_are_ambiguous(self):
        result = self.acquire(_synthetic(heads=(50, 270)), expected_head_center_u_px=210)
        self.assertIsNone(result.proposal)
        self.assertEqual(result.reason, "head_proposal_ambiguous")

    def test_projection_size_and_displacement_remain_bounded(self):
        self.assertIsNone(self.acquire(_synthetic(), expected_head_height_px=50).proposal)
        self.assertIsNone(self.acquire(_synthetic(), expected_head_center_u_px=100).proposal)

    def test_no_current_raw_evidence_means_no_proposal(self):
        frame = _synthetic()
        edges = numpy.zeros(frame.shape[:2], dtype=numpy.uint8)
        self.assertIsNone(self.acquire(frame, raw_edges=edges).proposal)
        self.assertFalse(numpy.any(edges))

    def test_search_image_and_raw_evidence_are_immutable(self):
        frame = _synthetic(inset=True)
        edges = _canny_edges_from_frame(cv2, frame, edge_preprocess="channel_union", blur_kernel=5, canny_low=20, canny_high=60)
        original_frame, original_edges = frame.copy(), edges.copy()
        result = self.acquire(frame, raw_edges=edges)
        self.assertIsNotNone(result.proposal)
        self.assertLessEqual(result.raw_verifications, 12)
        numpy.testing.assert_array_equal(frame, original_frame)
        numpy.testing.assert_array_equal(edges, original_edges)

    def test_invalid_inputs_fail_without_detector_authority(self):
        for height in (0, 11, float("nan"), float("inf")):
            with self.subTest(height=height):
                self.assertEqual(self.acquire(_synthetic(), expected_head_height_px=height).reason, "head_proposal_input_invalid")
        self.assertEqual(self.acquire(None).reason, "head_proposal_input_invalid")

    def recorded(self, name):
        metadata = RECORDED_HEAD_PROPOSALS[name]
        frame = recorded_head_proposal_image(cv2, numpy, name)
        result = acquire_head_proposal(cv2, frame, **{
            field: metadata[field] for field in (
                "expected_head_center_u_px", "expected_head_center_v_px", "expected_head_height_px",
            )
        })
        return result, metadata, frame

    def test_recorded_front45_corrects_crop_before_a_successful_pose_exists(self):
        result, metadata, frame = self.recorded("front45")
        self.assertEqual(result.reason, "current_head_proposal")
        proposal = result.proposal
        origin_x = metadata["source_crop_xyxy"][0]
        full_left = proposal.head_bounds_xyxy[0] + origin_x
        full_right = proposal.head_bounds_xyxy[2] + origin_x
        self.assertAlmostEqual(full_left, 446, delta=4)
        self.assertAlmostEqual(full_right, 581, delta=4)
        self.assertGreater(full_right, metadata["nominal_roi_xyxy"][2] + 50)
        self.assertGreater(proposal.bounds_xyxy[2] + origin_x, full_right)
        self.assertGreater(proposal.raw_edge_support, 0.9)
        self.assertGreater(proposal.center_offset_head_heights, 0.75)
        # The nominal clip really loses the observed border before processing.
        nominal_right = metadata["nominal_roi_xyxy"][2] - origin_x
        clipped = acquire_head_proposal(cv2, frame[:, :nominal_right], **{
            field: metadata[field] for field in (
                "expected_head_center_u_px", "expected_head_center_v_px", "expected_head_height_px",
            )
        })
        self.assertIsNone(clipped.proposal)

    def test_recorded_valid_backside_uses_same_pose_independent_acquisition(self):
        from types import SimpleNamespace
        from tests.aufgabe04.recorded_backside_fixture import RecordedBacksideFixture, FIXTURE_ROOT
        from scripts.aufgabe04.perception.camera_calibration import rectify_bgr_frame

        fixture = RecordedBacksideFixture(cv2, numpy)
        record = fixture.inputs["frames"]["frame_000008"]
        original = cv2.imread(str(FIXTURE_ROOT / record["image"]))
        rectified = rectify_bgr_frame(original, SimpleNamespace(**record["camera_info"]), cv2, numpy)
        attempt = record["attempts"][1]
        roi = attempt["roi"]
        result = acquire_head_proposal(
            cv2, rectified[roi["y0"]:roi["y1"], roi["x0"]:roi["x1"]],
            expected_head_center_u_px=attempt["expected_center_u_px"] - roi["x0"],
            expected_head_center_v_px=attempt["expected_center_v_px"] - roi["y0"],
            expected_head_height_px=attempt["expected_head_height_px"],
        )
        self.assertEqual(result.reason, "current_head_proposal")
        self.assertEqual(result.locator, "raw_lines")
        self.assertGreater(result.proposal.raw_edge_support, 0.9)
        self.assertGreater(result.proposal.center_offset_head_heights, 1.0)
        self.assertIsNone(getattr(result.proposal, "yaw_deg", None))

    def test_recorded_false_qr_back22_does_not_bypass_raw_neck_requirement(self):
        result, _metadata, _frame = self.recorded("back22")
        self.assertIsNone(result.proposal)
        self.assertEqual(result.reason, "head_proposal_unavailable")


if __name__ == "__main__":
    unittest.main()
