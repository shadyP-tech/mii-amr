"""Current raw rails tolerate rasterization while preserving physical gates."""

from pathlib import Path
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.raw_neck_support import measure_raw_neck_support
from scripts.aufgabe04.perception.stand_axis.head_model_fit import fit_current_measured_head
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import RectifiedCameraMatrix
from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
from scripts.aufgabe04.perception.stand_axis.head_neck_connectivity import trace_raw_neck_junction
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from tests.aufgabe04.recorded_backside_neck_fixture import (
    CAMERA, CORNERS, EXPECTED_HEAD, MIN_EDGE_HEIGHT_PX, recorded_backside_neck,
)


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class RawNeckSupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_measured_physical_stand_model(
            Path(__file__).resolve().parents[2]
            / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")

    def head(self):
        return tuple(ImagePoint(*p) for p in ((50., 50.), (150., 50.), (150., 150.), (50., 150.)))

    def rails(self, slope=.12, start=151, end=190, gap=14):
        raw = np.zeros((220, 220), np.uint8)
        for y in range(start, end):
            x = round(93 + slope * (y - 151))
            raw[y, x] = raw[y, x + gap] = 255
        return raw

    def assert_witness(self, raw, evidence):
        self.assertTrue(evidence.accepted, evidence.reason)
        self.assertEqual(len(evidence.paths_px), 2)
        self.assertGreaterEqual(evidence.paired_run, evidence.required_run)
        for path in evidence.paths_px:
            self.assertTrue(all(raw[y, x] > 0 for x, y in path))
            self.assertTrue(all(y1 - y0 == 1 and abs(x1 - x0) <= 1
                                for (x0, y0), (x1, y1) in zip(path, path[1:])))
        self.assertTrue(all(7 <= right[0] - left[0] <= 34
                            for left, right in zip(*evidence.paths_px)))

    def test_recorded_backside_has_current_continuous_raw_witnesses(self):
        _frame, raw = recorded_backside_neck(cv2, np)
        before = raw.copy()
        evidence = measure_raw_neck_support(raw, tuple(ImagePoint(*p) for p in CORNERS))
        self.assert_witness(raw, evidence)
        self.assertEqual(evidence.required_run, 12)
        np.testing.assert_array_equal(raw, before)

    def test_recorded_head_metric_fit_retains_strict_boundary_and_pose_checks(self):
        _frame, raw = recorded_backside_neck(cv2, np)
        corners = tuple(ImagePoint(*p) for p in CORNERS)
        estimate, debug, _pose = fit_current_measured_head(
            cv2, raw, model_profile=self.profile,
            camera=RectifiedCameraMatrix(**CAMERA), proposal_corners=corners)
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(estimate.reason, "axis_estimated_current_measured_head")
        self.assertTrue(debug.head_model_quality.accepted)
        self.assertIsNone(debug.head_neck_junction)
        self.assertTrue(debug.head_model_quality.outer_border_verified)
        self.assertIsNone(estimate.visible_face)
        self.assertAlmostEqual(estimate.yaw_deg, -12.93, delta=1.0)  # Replay bound, not metrology.
        self.assertLess(estimate.pose_reprojection_rmse_px, 2.0)

    def test_whole_recorded_image_fits_current_head_and_preserves_marker_veto(self):
        frame, _raw = recorded_backside_neck(cv2, np)
        options = dict(model_profile=self.profile,
                       camera_fx_px=CAMERA["fx_px"], camera_fy_px=CAMERA["fy_px"],
                       camera_cx_px=CAMERA["cx_px"], camera_cy_px=CAMERA["cy_px"],
                       expected_head_center_u_px=EXPECTED_HEAD[0],
                       expected_head_center_v_px=EXPECTED_HEAD[1],
                       expected_head_height_px=EXPECTED_HEAD[2], min_edge_height_px=MIN_EDGE_HEIGHT_PX)
        estimate, debug = estimate_stand_axis_from_metric_model(cv2, frame, qr_observations=(), **options)
        # Joint cold acquisition can select a different complete current
        # border than the historical two-rail locator. It must still pass the
        # unchanged independent pose/uncertainty checks without neck evidence.
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(debug.model_pose_fit_source, "model_current_measured_head")
        self.assertTrue(debug.head_model_quality.outer_border_verified)
        self.assertTrue(debug.head_model_quality.accepted)
        self.assertFalse(debug.head_model_quality.axis_ambiguous)
        self.assertIsNone(debug.head_neck_junction)
        self.assertIsNone(estimate.camera_face_normal_xyz)
        marked, marked_debug = estimate_stand_axis_from_metric_model(
            cv2, frame, qr_observations=(DecodedQrObservation("QR_001", None, "test_current_identity"),), **options)
        self.assertTrue(marked_debug.qr_detected)
        self.assertTrue(marked.usable, marked.reason)
        self.assertEqual(marked.corners, estimate.corners)
        self.assertAlmostEqual(marked.yaw_deg, estimate.yaw_deg, places=8)
        self.assertNotEqual(marked.visible_face, "backside_candidate")

    def test_neck_cue_does_not_authorize_an_inner_rectangle_pose(self):
        _frame, raw = recorded_backside_neck(cv2, np)
        cx, cy = (sum(p[i] for p in CORNERS) / 4 for i in (0, 1))
        inner = tuple(ImagePoint(cx + (x - cx) * 71 / 78, cy + (y - cy) * 71 / 78)
                      for x, y in CORNERS)
        # The visible post overlaps the lower head in this image. Sharing its
        # raw pixels is only a neck cue: an inset proposal still needs an
        # independent unambiguous metric fit and cannot inherit the outer pose.
        estimate, debug, _pose = fit_current_measured_head(
            cv2, raw, model_profile=self.profile,
            camera=RectifiedCameraMatrix(**CAMERA), proposal_corners=inner)
        self.assertFalse(estimate.usable)
        self.assertFalse(debug.head_model_quality.accepted)
        self.assertIsNone(estimate.yaw_deg)
        self.assertIsNone(estimate.visible_face)

    def test_oppositely_slanted_pairs_keep_current_pixel_connectivity(self):
        for slope in (-.12, .12):
            with self.subTest(slope=slope):
                raw = self.rails(slope)
                before = raw.copy()
                self.assert_witness(raw, measure_raw_neck_support(raw, self.head()))
                np.testing.assert_array_equal(raw, before)

    def test_missing_row_is_never_filled(self):
        raw = self.rails()
        raw[159, :] = 0  # Earlier fragment too short; later start exceeds the original 8px cap.
        evidence = measure_raw_neck_support(raw, self.head())
        self.assertFalse(evidence.accepted)
        self.assertEqual(evidence.maximum_start_gap_px, 8)

    def test_single_thick_rail_and_too_close_pair_remain_rejected(self):
        single = np.zeros((220, 220), np.uint8)
        single[151:190, 98:101] = 255
        for raw in (single, self.rails(gap=4), self.rails(gap=36)):
            with self.subTest(nonzero=int(np.count_nonzero(raw))):
                self.assertFalse(measure_raw_neck_support(raw, self.head()).accepted)

    def test_disconnected_horizontal_jumps_are_not_continuous_rails(self):
        raw = np.zeros((220, 220), np.uint8)
        for y in range(151, 190):
            x = 92 + (4 if y % 2 else 0)
            raw[y, x] = raw[y, x + 14] = 255
        self.assertFalse(measure_raw_neck_support(raw, self.head()).accepted)

    def test_late_existing_core_remains_diagnostic_only(self):
        raw = self.rails(slope=0, start=161)
        evidence = measure_raw_neck_support(raw, self.head())
        self.assertFalse(evidence.accepted)
        self.assertEqual(evidence.reason, "raw_neck_start_gap_too_large")
        self.assertEqual(evidence.start_gap_px, 10)
        self.assertTrue(evidence.paths_px)

    def test_junction_continuation_revalidates_supplied_core_pixels(self):
        raw = self.rails()
        core = measure_raw_neck_support(raw, self.head())
        options = dict(bottom_edge_px=((50., 150.), (150., 150.)),
                       rail_columns_px=tuple(path[0][0] for path in core.paths_px),
                       core_start_row_px=core.paths_px[0][0][1],
                       core_run_length_px=core.required_run, min_rail_gap_px=7,
                       max_rail_gap_px=34, max_start_gap_px=0, core_paths_px=core.paths_px)
        self.assertTrue(trace_raw_neck_junction(raw, **options).accepted)
        changed = raw.copy()
        x, y = core.paths_px[0][5]
        changed[y, x] = 0
        rejected = trace_raw_neck_junction(changed, **options)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "raw_neck_continuation_core_unavailable")

    def test_malformed_masks_and_corners_fail_closed(self):
        for raw in (None, [], np.zeros((4, 4, 3)), np.full((220, 220), np.nan)):
            with self.subTest(raw_type=type(raw).__name__):
                self.assertFalse(measure_raw_neck_support(raw, self.head()).accepted)
        for corners in (None, (), self.head()[:3],
                        (ImagePoint(float("nan"), 1),) * 4, (object(),) * 4):
            with self.subTest(corners=corners):
                self.assertFalse(measure_raw_neck_support(self.rails(), corners).accepted)


if __name__ == "__main__":
    unittest.main()
