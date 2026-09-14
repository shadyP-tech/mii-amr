from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.aufgabe04.artifacts.backside_axis_observation import MINIMUM_BACKSIDE_FACE_CONFIDENCE

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover
    cv2 = None
    numpy = None

from scripts.aufgabe04.perception.stand_axis.model_backside_topology import (
    backside_topology_proposal_batches,
)
from scripts.aufgabe04.perception.stand_axis.model_backside_acquisition import estimate_stand_axis_from_model_backside
from scripts.aufgabe04.perception.stand_axis.model_pipeline import (
    estimate_stand_axis_from_metric_model,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import (
    load_measured_physical_stand_model,
)
from scripts.aufgabe04.perception.stand_axis.preprocessing import (
    _canny_edges_from_frame,
    _topology_edges_from_frame,
)
from tests.aufgabe04.recorded_backside_camera_fixture import (
    ROI_CAMERA_OPTIONS,
    recorded_backside_roi,
)


@unittest.skipIf(cv2 is None or numpy is None, "OpenCV/numpy unavailable")
class BacksideTopologyRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_measured_physical_stand_model(
            Path(__file__).resolve().parents[2]
            / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"
        )

    def estimate(self, frame=None, **overrides):
        options = {**ROI_CAMERA_OPTIONS, **overrides}
        return estimate_stand_axis_from_metric_model(
            cv2,
            recorded_backside_roi(cv2, numpy) if frame is None else frame,
            model_profile=self.profile,
            **options,
        )

    def test_recorded_connected_neck_keeps_head_angle_with_separate_backside_evidence(self):
        estimate, debug = self.estimate()

        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(estimate.reason, "axis_estimated_current_measured_head_backside")
        self.assertEqual(estimate.source, "model_backside_current_frame")
        self.assertEqual(estimate.visible_face, "backside_candidate")
        self.assertEqual(estimate.evidence_state, "fresh_backside")
        self.assertFalse(debug.qr_detected)
        self.assertIsNone(estimate.camera_face_normal_xyz)
        self.assertIsNone(estimate.camera_face_center_xyz_m)
        self.assertIsNotNone(estimate.yaw_deg)
        self.assertLess(estimate.pose_reprojection_rmse_px, 1.0)
        self.assertGreaterEqual(estimate.visible_face_confidence, MINIMUM_BACKSIDE_FACE_CONFIDENCE)
        self.assertEqual(debug.model_pose_fit_source, "model_current_measured_head")
        self.assertTrue(debug.head_backside_classification.accepted)
        self.assertEqual(debug.head_backside_classification.confidence, estimate.visible_face_confidence)
        self.assertEqual(debug.head_backside_classification.yaw_deg, estimate.yaw_deg)
        self.assertEqual(debug.model_pose.yaw_deg, estimate.yaw_deg)
        self.assertTrue(debug.head_model_quality.accepted)
        self.assertEqual(debug.head_neck_junction.core_start_gap_px, 4)
        self.assertEqual(debug.head_neck_junction.start_gap_px, 0)
        self.assertTrue(debug.head_neck_junction.raw_continuation.accepted)
        for path in debug.head_neck_junction.raw_continuation.paths_px:
            self.assertTrue(all(debug.raw_edges[y, x] > 0 for x, y in path))
        self.assertIsNone(debug.head_outer_recovery)
        # The outer head spans x=11..80, y=18..92. The interior Start label
        # must not supply the recovered head's bottom or side measurements.
        self.assertAlmostEqual(min(p.u_px for p in estimate.corners), 11, delta=2)
        self.assertAlmostEqual(max(p.u_px for p in estimate.corners), 80, delta=2)
        self.assertGreater(min(p.v_px for p in estimate.corners[2:]), 88)

    def test_recorded_neck_with_a_missing_raw_row_remains_rejected(self):
        _estimate, debug = self.estimate()
        raw = debug.raw_edges.copy()
        raw[93, :] = 0
        with patch(
            "scripts.aufgabe04.perception.stand_axis.model_pipeline._canny_edges_from_frame",
            return_value=raw,
        ):
            estimate, debug = self.estimate()
        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.reason, "head_neck_junction_gap_too_large")
        self.assertIsNone(estimate.yaw_deg)
        self.assertIsNone(estimate.visible_face)
        self.assertFalse(debug.head_neck_junction.raw_continuation.accepted)
        self.assertEqual(debug.head_neck_junction.start_gap_px,
                         debug.head_neck_junction.core_start_gap_px)

    def test_recorded_frame_reproduces_failure_without_recovery_batch(self):
        def filtered_only(*args, **kwargs):
            yield next(backside_topology_proposal_batches(*args, **kwargs))

        with patch(
            "scripts.aufgabe04.perception.stand_axis.model_backside_acquisition."
            "backside_topology_proposal_batches",
            side_effect=filtered_only,
        ):
            estimate, _debug = self.estimate()

        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.reason, "model_backside_head_and_neck_unavailable")

    def test_existing_filtered_head_never_enters_raw_recovery(self):
        frame = numpy.zeros((130, 131, 3), dtype=numpy.uint8)
        cv2.rectangle(frame, (30, 20), (100, 90), (255, 255, 255), -1)
        cv2.rectangle(frame, (58, 90), (72, 129), (255, 255, 255), -1)

        def filtered_then_forbidden(*args, **kwargs):
            yield next(backside_topology_proposal_batches(*args, **kwargs))
            self.fail("a filtered head must not be displaced by raw topology")

        with patch(
            "scripts.aufgabe04.perception.stand_axis.model_backside_acquisition."
            "backside_topology_proposal_batches",
            side_effect=filtered_then_forbidden,
        ):
            # Exercise retained fallback topology directly. Its historical
            # backside policy is not the independent head-angle policy.
            options = {**ROI_CAMERA_OPTIONS, "expected_head_center_u_px": 65,
                       "expected_head_center_v_px": 55, "expected_head_height_px": 70}
            estimate, _debug = estimate_stand_axis_from_model_backside(
                cv2, frame, model_profile=self.profile,
                raw_edges=_canny_edges_from_frame(cv2, frame, edge_preprocess="channel_union",
                                                  blur_kernel=5, canny_low=20, canny_high=60),
                max_reprojection_rmse_px=2.0, **options,
            )
        self.assertTrue(estimate.usable, estimate.reason)
        current, debug = self.estimate(
            frame, expected_head_center_u_px=65,
            expected_head_center_v_px=55, expected_head_height_px=70,
        )
        self.assertFalse(current.usable)
        self.assertTrue(debug.head_neck_junction.accepted)
        self.assertEqual(current.reason, "head_model_yaw_uncertainty_too_high")
        self.assertIsNone(debug.model_pose)

    def test_raw_recovery_keeps_metric_scale_and_target_association_gates(self):
        for overrides in (
            {"expected_head_height_px": 54.0},
            {"expected_head_center_u_px": 95.0},
        ):
            with self.subTest(overrides=overrides):
                estimate, _debug = self.estimate(**overrides)
                self.assertFalse(estimate.usable, estimate)
                self.assertIsNone(estimate.visible_face)

    def test_recorded_label_and_head_without_neck_cannot_be_a_stand(self):
        original = recorded_backside_roi(cv2, numpy)
        no_neck = original.copy()
        no_neck[94:, :] = (143, 143, 143)
        only_label = numpy.full_like(original, 143)
        only_label[39:69, 28:72] = original[39:69, 28:72]
        for name, frame in (
            ("head_without_neck", no_neck),
            ("interior_start_label", only_label),
        ):
            with self.subTest(name=name):
                estimate, _debug = self.estimate(frame)
                self.assertFalse(estimate.usable, estimate)
                self.assertIsNone(estimate.visible_face)

    def test_absent_raw_top_border_cannot_be_reconstructed_by_topology(self):
        frame = recorded_backside_roi(cv2, numpy)
        raw = _canny_edges_from_frame(
            cv2, frame, edge_preprocess="channel_union", blur_kernel=5,
            canny_low=20, canny_high=60,
        )
        # Remove measurement support directly. Painting over image pixels
        # would introduce a new real contrast border at the patch boundary.
        raw[:30, :] = 0
        with patch(
            "scripts.aufgabe04.perception.stand_axis.model_pipeline."
            "_canny_edges_from_frame",
            return_value=raw,
        ):
            estimate, _debug = self.estimate(frame)
        self.assertFalse(estimate.usable, estimate)
        self.assertIsNone(estimate.visible_face)

    def test_repeated_background_rails_cannot_supply_a_target_head(self):
        frame = numpy.full((130, 131, 3), 143, dtype=numpy.uint8)
        for x in range(8, 130, 12):
            cv2.line(frame, (x, 0), (x, 129), (85, 85, 85), 2)
        for y in (18, 90):
            cv2.line(frame, (0, y), (130, y), (85, 85, 85), 2)
        estimate, _debug = self.estimate(frame)
        self.assertFalse(estimate.usable, estimate)
        self.assertIsNone(estimate.visible_face)

    def test_qr_texture_without_head_or_neck_never_confirms_backside(self):
        frame = numpy.full((130, 131, 3), 255, dtype=numpy.uint8)
        qr = cv2.QRCodeEncoder_create().encode("QR_001")
        qr = cv2.resize(qr, (76, 76), interpolation=cv2.INTER_NEAREST)
        frame[26:102, 27:103] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
        estimate, _debug = self.estimate(frame)
        self.assertFalse(estimate.usable, estimate)
        self.assertNotEqual(estimate.visible_face, "backside_candidate")

    def test_proposal_batches_are_bounded_and_preserve_raw_pixels(self):
        frame = recorded_backside_roi(cv2, numpy)
        raw = _canny_edges_from_frame(
            cv2, frame, edge_preprocess="channel_union", blur_kernel=5,
            canny_low=20, canny_high=60,
        )
        before = raw.copy()
        filtered = _topology_edges_from_frame(
            cv2, frame, edge_preprocess="channel_union", canny_low=20,
            canny_high=60, fallback_edges=raw,
        )
        batches = list(backside_topology_proposal_batches(
            cv2, filtered, raw, edge_preprocess="channel_union",
        ))
        self.assertEqual([len(batch) for batch in batches], [3, 3])
        numpy.testing.assert_array_equal(raw, before)
        # Other preprocessing modes already use raw topology and cannot add
        # an identical or unconstrained recovery branch.
        self.assertEqual(len(list(backside_topology_proposal_batches(
            cv2, raw, raw, edge_preprocess="channel_union",
        ))), 1)
        self.assertEqual(len(list(backside_topology_proposal_batches(
            cv2, filtered, raw, edge_preprocess="gray",
        ))), 1)


if __name__ == "__main__":
    unittest.main()
