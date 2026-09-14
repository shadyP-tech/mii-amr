"""Saved current pixels cover independent head geometry and front-marker veto.

These are geometry regressions, not replayed freshness or hardware evidence.
Lossless fixtures retain original recording provenance and calibration. The
development OpenCV4.13 backend differs from the deployed4.5.4 runtime.
"""

import hashlib
import json
import math
from pathlib import Path
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.head_model_fit import fit_current_measured_head
from scripts.aufgabe04.perception.stand_axis.head_model_quality import validated_head_model_quality
from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import RectifiedCameraMatrix
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures/stand_views_20260914_123717"


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required for saved image regressions")
class RecordedStandViews20260914Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = json.loads((FIXTURE_ROOT / "inputs.json").read_text())
        model_path = ROOT / cls.inputs["model_profile"]
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != cls.inputs["model_file_sha256"]:
            raise AssertionError("Recorded geometry must use its unchanged measured stand profile")
        cls.profile = load_measured_physical_stand_model(model_path)

    def fixture(self, name):
        record = self.inputs["records"][name]
        path = FIXTURE_ROOT / record["image"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["image_sha256"])
        frame = cv2.imread(str(path))
        self.assertIsNotNone(frame)
        self.assertEqual(hashlib.sha256(frame.tobytes()).hexdigest(), record["pixel_sha256"])
        raw = _canny_edges_from_frame(
            cv2, frame, edge_preprocess=record["profile"]["edge_preprocess"],
            blur_kernel=5, canny_low=record["profile"]["canny_low"],
            canny_high=record["profile"]["canny_high"],
        )
        return record, frame, raw

    def fit(self, record, raw):
        return fit_current_measured_head(
            cv2, raw, model_profile=self.profile,
            camera=RectifiedCameraMatrix(**record["camera"]),
            proposal_corners=tuple(ImagePoint(**p) for p in record["diagnostic_proposal_corners"]),
            min_edge_height_px=record["profile"]["min_edge_height_px"],
        )

    def pipeline(self, record, frame, *, observations=(), seeded=False):
        camera = record["camera"]
        expected = record["expected_head"]
        return estimate_stand_axis_from_metric_model(
            cv2, frame, model_profile=self.profile,
            camera_fx_px=camera["fx_px"], camera_fy_px=camera["fy_px"],
            camera_cx_px=camera["cx_px"], camera_cy_px=camera["cy_px"],
            expected_head_center_u_px=expected[0], expected_head_center_v_px=expected[1],
            expected_head_height_px=expected[2],
            backside_target_crop_horizontal_half_width_ratio=record["crop_half_width_ratio"],
            edge_preprocess=record["profile"]["edge_preprocess"],
            canny_low=record["profile"]["canny_low"], canny_high=record["profile"]["canny_high"],
            min_edge_height_px=record["profile"]["min_edge_height_px"],
            qr_observations=observations,
            current_head_proposal_corners=(
                tuple(ImagePoint(**p) for p in record["diagnostic_proposal_corners"])
                if seeded else None
            ),
        )

    def assert_current_physical_head(self, estimate, debug):
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertTrue(validated_head_model_quality(debug.head_model_quality))
        self.assertTrue(debug.head_model_quality.outer_border_verified)
        self.assertEqual(debug.model_pose_fit_source, "model_current_measured_head")
        self.assertEqual(estimate.model_profile_sha256, self.profile.sha256)
        self.assertIsNotNone(estimate.yaw_deg)
        self.assertTrue(math.isfinite(estimate.yaw_deg))

    def assert_recorded_backside_geometry(self, name, estimate, debug):
        self.assertTrue(debug.head_model_quality.outer_border_verified)
        self.assertFalse(debug.head_model_quality.centered_neck_supported)
        self.assertFalse(debug.head_model_quality.neck_junction_verified)
        if name == "backside_000010":
            # Complete outer borders must reach the head-only pose test, but
            # this image's two planar orientations remain indistinguishable.
            self.assertFalse(estimate.usable)
            self.assertEqual(estimate.reason, "head_model_planar_axis_ambiguous")
            self.assertTrue(debug.head_model_quality.axis_ambiguous)
            self.assertIsNone(estimate.yaw_deg)
        else:
            self.assert_current_physical_head(estimate, debug)

    def test_recorded_complete_heads_reach_pose_tests_after_neck_cues_fail(self):
        expected_old_failures = {
            "backside_000010": "raw_neck_paired_continuity_insufficient",
            "backside_000012": "raw_neck_start_gap_too_large",
        }
        for name, old_reason in expected_old_failures.items():
            with self.subTest(name=name):
                record, _frame, raw = self.fixture(name)
                self.assertEqual(record["old_neck_replay"]["reason"], old_reason)
                estimate, debug, pose = self.fit(record, raw)
                self.assert_recorded_backside_geometry(name, estimate, debug)
                self.assertIsNotNone(pose)
                self.assertGreater(len(pose.hypotheses), 0)
                self.assertIsNone(estimate.visible_face)  # Head fit alone has no side authority.

    def test_removing_current_neck_pixels_does_not_change_the_head_angle(self):
        for name in ("backside_000010", "backside_000012"):
            with self.subTest(name=name):
                record, _frame, raw = self.fixture(name)
                estimate, debug, _pose = self.fit(record, raw)
                self.assert_recorded_backside_geometry(name, estimate, debug)
                removed = raw.copy()
                # Preserve the actual bottom rail and its raw corner arms;
                # erase the longer post/neck runs outside the measured head.
                first_removed_row = math.ceil(max(p.v_px for p in estimate.corners)) + 4
                removed[first_removed_row:, :] = 0
                self.assertGreater(np.count_nonzero(raw != removed), 0)
                without_neck, changed_debug, _ = self.fit(record, removed)
                self.assert_recorded_backside_geometry(name, without_neck, changed_debug)
                self.assertEqual(without_neck.corners, estimate.corners)
                self.assertEqual(without_neck.reason, estimate.reason)
                if estimate.yaw_deg is None:
                    self.assertIsNone(without_neck.yaw_deg)
                else:
                    self.assertAlmostEqual(without_neck.yaw_deg, estimate.yaw_deg, places=9)

    def test_real_backside_pixels_reach_the_whole_metric_pipeline(self):
        for name in ("backside_000010", "backside_000012"):
            with self.subTest(name=name):
                record, frame, _raw = self.fixture(name)
                estimate, debug = self.pipeline(record, frame)
                # Joint current-border acquisition can select a better
                # supported quadrilateral than the diagnostic seed. The
                # direct seed's ambiguity remains covered separately above.
                self.assert_current_physical_head(estimate, debug)
                self.assertFalse(debug.head_model_quality.centered_neck_supported)
                self.assertFalse(debug.head_model_quality.neck_junction_verified)
                for selected, recorded in zip(estimate.corners, record["diagnostic_proposal_corners"]):
                    self.assertLess(math.dist(
                        (selected.u_px, selected.v_px),
                        (recorded["u_px"], recorded["v_px"])), 8.)
                self.assertFalse(debug.qr_detected)
                self.assertFalse(debug.qr_marker_verified)
                # Receipt emission still belongs to the observer's distinct
                # frame/scan/epoch consensus gates, exercised by the existing
                # measured_backside_observer_handoff integration suite.

    def test_complete_front_decodes_its_own_qr_and_never_becomes_backside(self):
        record, frame, raw = self.fixture("front_complete")
        observations = detect_qr_observations_bgr(frame, cv2)
        self.assertEqual(tuple(o.text for o in observations), (record["expected_qr_text"],))
        self.assertIsNotNone(observations[0].corners)
        estimate, debug = self.pipeline(record, frame, observations=observations, seeded=True)
        self.assert_current_physical_head(estimate, debug)
        self.assertTrue(debug.qr_detected)
        self.assertTrue(debug.qr_marker_verified)
        self.assertNotEqual(estimate.visible_face, "backside_candidate")
        self.assertNotEqual(estimate.source, "model_backside_current_frame")
        self.assertIsNotNone(debug.head_backside_appearance)
        self.assertFalse(debug.head_backside_appearance.accepted)
        head_only, direct_debug, _ = self.fit(record, raw)
        self.assert_current_physical_head(head_only, direct_debug)
        self.assertAlmostEqual(estimate.yaw_deg, head_only.yaw_deg, places=9)

    def test_complete_front_finders_veto_backside_without_a_decoded_identity(self):
        record, frame, _raw = self.fixture("front_complete")
        estimate, debug = self.pipeline(record, frame, observations=(), seeded=True)
        self.assert_current_physical_head(estimate, debug)
        self.assertTrue(debug.qr_detected)
        self.assertTrue(debug.qr_marker_verified)
        self.assertEqual(debug.qr_marker_reason, "three_current_pixel_qr_finders")
        self.assertNotEqual(estimate.visible_face, "backside_candidate")
        self.assertNotEqual(estimate.source, "model_backside_current_frame")
        self.assertIsNotNone(debug.head_backside_appearance)
        self.assertFalse(debug.head_backside_appearance.accepted)

    def test_clipped_recorded_front_does_not_turn_marker_absence_into_backside(self):
        record, frame, _raw = self.fixture("front_clipped")
        self.assertFalse(record["complete_head_visible"])
        self.assertGreater(record["actual_full_head_bounds"][2], record["crop_xyxy"][2])
        estimate, debug = self.pipeline(record, frame)
        self.assertNotEqual(estimate.visible_face, "backside_candidate")
        self.assertNotEqual(estimate.evidence_state, "fresh_backside")
        classification = debug.head_backside_classification
        self.assertTrue(classification is None or not classification.accepted)


if __name__ == "__main__":
    unittest.main()
