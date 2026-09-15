"""QR evidence controls identity/side, never the qualified measured-head angle."""

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.perception.stand_axis.head_backside_classification import (
    classify_current_head_backside,
)
from scripts.aufgabe04.perception.stand_axis.head_outer_border import (
    HeadMarkerBoundaryEvidence, current_head_boundary_eligible,
)
from scripts.aufgabe04.perception.stand_axis_consensus import axis_conditioning
from scripts.aufgabe04.real_robot.observer.axis_sample_policy import admit_axis_sample
from scripts.aufgabe04.real_robot.observer.head_observation_window import current_head_window_input
from tests.aufgabe04.test_head_backside_classification import classified_head


class HeadQrIndependenceContractTests(unittest.TestCase):
    @staticmethod
    def marker_diagnostics():
        # Include an older report's reconsideration flag: it must have no
        # authority over current physical-head consumers after this correction.
        return (
            None,
            HeadMarkerBoundaryEvidence(True, "no_current_verified_symbol_scale_reference"),
            HeadMarkerBoundaryEvidence(True, "current_head_symbol_boundary_not_contradicted",
                                       observed_symbol_spans=(.795, .795)),
            HeadMarkerBoundaryEvidence(False, "current_border_matches_verified_qr_panel",
                                       observed_symbol_spans=(.875, .875),
                                       requests_reconsideration=True),
            HeadMarkerBoundaryEvidence(False, "unreliable_marker_geometry",
                                       observed_symbol_spans=(math.nan, math.inf)),
            object(),
        )

    @staticmethod
    def admission(estimate, debug, qr_texts=(), associated=True):
        yaw = math.radians(estimate.yaw_deg)
        return admit_axis_sample(
            estimate=estimate, debug=debug, yaw_rad=yaw,
            conditioning=axis_conditioning(yaw), qr_texts=qr_texts,
            lidar_target_associated=associated,
        )

    @staticmethod
    def window(estimate, debug):
        return current_head_window_input(
            estimate, debug, frame_stamp_sec=10., camera_signature=(640., 640., 80., 80.),
            roi=SimpleNamespace(x0=10, y0=20), projected_center_px=(90., 100.),
            expected_head_height_px=90.,
        )

    def test_qr_size_presence_and_identity_cannot_change_angle_or_temporal_input(self):
        estimate, debug, _ = classified_head(yaw_deg=45.)
        admitted, temporal = self.admission(estimate, debug), self.window(estimate, debug)
        self.assertTrue(admitted.accepted)
        self.assertIsNotNone(temporal)
        # No independently recovered enclosing contour is required merely
        # because a QR happens to be present in this otherwise identical fit.
        self.assertFalse(debug.head_outer_recovery.recovered)
        for diagnostic in self.marker_diagnostics():
            for detected, verified, texts in (
                (False, False, ()), (True, False, ()),
                (True, True, ("QR_003",)), (True, True, ("QR_003", "QR_004")),
            ):
                with self.subTest(marker=diagnostic, texts=texts, detected=detected):
                    current = replace(debug, head_marker_boundary=diagnostic,
                                      qr_detected=detected, qr_marker_verified=verified)
                    self.assertTrue(current_head_boundary_eligible(estimate, current))
                    self.assertEqual(self.admission(estimate, current, texts), admitted)
                    self.assertEqual(self.window(estimate, current), temporal)

    def test_marker_only_changes_backside_classification(self):
        estimate, debug, options = classified_head()
        for diagnostic in self.marker_diagnostics():
            for detected, verified in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(marker=diagnostic, detected=detected, verified=verified):
                    current = replace(debug, head_marker_boundary=diagnostic,
                                      qr_detected=detected, qr_marker_verified=verified)
                    side, proof = classify_current_head_backside(estimate, current, **options)
                    self.assertEqual(proof.head_backside_classification.accepted,
                                     not detected and not verified)
                    self.assertEqual(side.yaw_deg, estimate.yaw_deg)
                    self.assertEqual(side.corners, estimate.corners)
                    self.assertTrue(self.admission(estimate, current).accepted)

    def test_qr_cannot_promote_missing_raw_border_bad_quality_or_association(self):
        estimate, debug, _ = classified_head()
        for diagnostic in self.marker_diagnostics():
            marked = replace(debug, head_marker_boundary=diagnostic,
                             qr_detected=True, qr_marker_verified=True)
            for boundary in (None, replace(debug.head_outer_recovery, accepted=False),
                             replace(debug.head_outer_recovery, profile_sha256="b" * 64)):
                with self.subTest(marker=diagnostic, boundary=boundary):
                    current = replace(marked, head_outer_recovery=boundary)
                    self.assertFalse(self.admission(estimate, current, ("QR_003",)).accepted)
                    self.assertIsNone(self.window(estimate, current))
            poor_quality = replace(marked, head_model_quality=replace(
                debug.head_model_quality, accepted=False, raw_corner_support_accepted=False))
            self.assertFalse(self.admission(estimate, poor_quality, ("QR_003",)).accepted)
            self.assertIsNone(self.window(estimate, poor_quality))
            self.assertFalse(self.admission(estimate, marked, ("QR_003",), associated=False).accepted)

    def test_old_qr_diagnostic_reason_cannot_veto_an_otherwise_current_head(self):
        estimate, debug, _ = classified_head()
        diagnostic = replace(estimate, reason="current_border_matches_verified_qr_panel")
        self.assertEqual(self.window(diagnostic, debug), self.window(estimate, debug))
        self.assertEqual(self.admission(diagnostic, debug), self.admission(estimate, debug))


if __name__ == "__main__":
    unittest.main()
