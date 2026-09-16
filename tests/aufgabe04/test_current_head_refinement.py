"""Cold border selection and metric fitting share the current physical rails."""

import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.current_head_border_binding import bind_selected_current_head
from scripts.aufgabe04.perception.stand_axis.current_head_refinement import refine_current_physical_head
from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import HeadAcquisitionDeadlineExceeded
from scripts.aufgabe04.perception.stand_axis.head_model_fit import fit_current_measured_head
from scripts.aufgabe04.perception.stand_axis.head_outer_border import validated_current_head_boundary
from tests.aufgabe04 import test_head_boundary_independence as fixtures


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class CurrentHeadRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.HeadBoundaryIndependenceTest.setUpClass()
        cls.fixture = fixtures.HeadBoundaryIndependenceTest()
        cls.profile = cls.fixture.profile

    def test_current_panel_seed_resolves_physical_rails_before_metric_border_binding(self):
        corridors = []
        for angle in (20., 45., 60.):
            with self.subTest(angle=angle):
                outer, camera = self.fixture.projection(angle_deg=angle)
                panel, _ = self.fixture.projection(.071/.078, angle_deg=angle)
                raw = self.fixture.raw(outer, panel)
                before = raw.copy()
                measurement, boundary, seed = refine_current_physical_head(
                    cv2, raw, model_profile=self.profile, proposal_corners=panel)
                self.assertTrue(measurement.accepted, measurement.reason)
                self.assertTrue(boundary.recovered)
                self.assertTrue(validated_current_head_boundary(boundary,
                    corners=measurement.corners, profile_sha256=self.profile.sha256))
                self.assertEqual(seed.corners, panel)
                corridors.append(seed.corridor_half_width_px)
                fitted = fit_current_measured_head(cv2, raw,
                    model_profile=self.profile, camera=camera,
                    proposal_corners=measurement.corners)
                bound, diagnostic = bind_selected_current_head(
                    fitted, measurement.corners, raw_edges=raw,
                    frame_bgr=cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR))
                self.assertTrue(diagnostic["accepted"], diagnostic)
                self.assertTrue(bound[0].usable, bound[0].reason)
                np.testing.assert_array_equal(raw, before)
        self.assertGreater(max(corridors), 4.)

    def test_raw_refinement_matches_metric_evidence_without_solving_a_pose(self):
        outer, camera = self.fixture.projection()
        panel, _ = self.fixture.projection(.071/.078)
        raw = self.fixture.raw(outer, panel)
        with patch("scripts.aufgabe04.perception.stand_axis.head_model_fit.estimate_planar_pose_ippe",
                   side_effect=AssertionError("border canonicalization cannot solve a pose")):
            measurement, boundary, seed = refine_current_physical_head(
                cv2, raw, model_profile=self.profile, proposal_corners=panel)
        estimate, debug, _ = fit_current_measured_head(cv2, raw,
            model_profile=self.profile, camera=camera, proposal_corners=panel)
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(estimate.corners, measurement.corners)
        self.assertEqual(debug.head_outer_recovery, boundary)
        self.assertEqual(debug.model_corridor_half_width_px, seed.corridor_half_width_px)
        self.assertIsNone(getattr(measurement, "yaw_deg", None))

    def test_missing_raw_border_or_corner_cannot_be_supplied_by_canonicalization(self):
        outer, _camera = self.fixture.projection()
        raw = self.fixture.raw(outer)
        missing = raw.copy()
        # A complete absent side remains absent even though its proposal exists.
        left = (outer[0], outer[3])
        cv2.line(missing, tuple(np.rint((left[0].u_px, left[0].v_px)).astype(int)),
                 tuple(np.rint((left[1].u_px, left[1].v_px)).astype(int)), 0, 12)
        for image in (np.zeros_like(raw), missing):
            with self.subTest(blank=not np.any(image)):
                measurement, boundary, _seed = refine_current_physical_head(
                    cv2, image, model_profile=self.profile, proposal_corners=outer)
                self.assertFalse(measurement.accepted)
                self.assertFalse(boundary.accepted)

    def test_initial_raw_fit_expiry_stops_before_any_outer_search(self):
        from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
        corners, _camera = self.fixture.projection()
        raw = self.fixture.raw(corners)
        clock = [0.]
        def measured(*args, **kwargs):
            result = refine_projected_head_border(*args, **kwargs)
            clock[0] = 2.
            return result
        module = "scripts.aufgabe04.perception.stand_axis.current_head_refinement."
        with patch("scripts.aufgabe04.perception.stand_axis.head_acquisition_budget.time.monotonic",
                   side_effect=lambda: clock[0]), \
             patch(module + "refine_projected_head_border", side_effect=measured) as initial, \
             patch(module + "select_current_outer_head_border") as outer:
            with self.assertRaises(HeadAcquisitionDeadlineExceeded):
                refine_current_physical_head(cv2, raw, model_profile=self.profile,
                    proposal_corners=corners, deadline_monotonic_sec=1.)
        initial.assert_called_once()
        outer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
