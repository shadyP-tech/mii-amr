"""A nearby real neck must not authenticate a detached inner paper border."""

import math
from dataclasses import replace
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.head_candidates import _short_centered_neck_support
from scripts.aufgabe04.perception.stand_axis.head_model_fit import fit_current_measured_head
from scripts.aufgabe04.perception.stand_axis.head_model_neck import measure_head_neck_junction
from scripts.aufgabe04.perception.stand_axis.head_proposal import acquire_head_proposal
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.real_robot.observer.backside_proposal_reuse import BacksideProposalReuse
from tests.aufgabe04.recorded_backside_fixture import RecordedBacksideFixture
from tests.aufgabe04 import test_geometry_contract as geometry_fixture
from tests.aufgabe04 import test_head_model_angle_reference as reference_fixture


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class HeadModelNeckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference_fixture.HeadModelAngleReferenceTest.setUpClass()
        cls.reference = reference_fixture.HeadModelAngleReferenceTest()
        cls.profile = cls.reference.profile

    def test_pixel_gap_cap_tightens_when_measured_panel_inset_is_small(self):
        for height, max_gap, resolved in ((40, 0, False), (55, 0, True), (70, 1, True), (140, 2, True)):
            corners = tuple(ImagePoint(x, y) for x, y in (
                (40, 30), (40 + height, 30), (40 + height, 30 + height), (40, 30 + height),
            ))
            for gap in (0, 2, 3, 6):
                with self.subTest(height=height, gap=gap):
                    raw = np.zeros((300, 300), np.uint8)
                    center = 40 + height / 2
                    y0 = 31 + height + gap
                    for sign in (-1, 1):
                        x = round(center + sign * .07 * height)
                        cv2.line(raw, (x, y0), (x, y0 + 25), 255, 1)
                    original = raw.copy()
                    junction = measure_head_neck_junction(raw, corners, self.profile)
                    self.assertEqual(junction.start_gap_px, gap)
                    self.assertEqual(junction.accepted, resolved and gap <= max_gap)
                    self.assertEqual(junction.max_start_gap_px, max_gap)
                    self.assertEqual(junction.pixel_uncertainty_allowance_px, 2.0)
                    if not resolved:
                        self.assertEqual(junction.reason, "head_neck_panel_separation_unresolved")
                    np.testing.assert_array_equal(raw, original)

    def test_qr_symbol_dimensions_do_not_change_the_measured_panel_gate(self):
        corners, _camera = self.reference.projection(45.)
        raw = self.reference.raster(corners)
        expected = measure_head_neck_junction(raw, corners, self.profile)
        self.assertTrue(expected.accepted)
        changed = replace(self.profile, qr_symbol_width_m=.050, qr_symbol_height_m=.055)
        self.assertEqual(measure_head_neck_junction(raw, corners, changed), expected)
        unavailable = replace(self.profile, qr_panel_height_m=None)
        self.assertFalse(measure_head_neck_junction(raw, corners, unavailable).accepted)

    def test_two_separate_contiguous_rails_are_required(self):
        corners = tuple(ImagePoint(x, y) for x, y in ((40, 30), (140, 30), (140, 130), (40, 130)))
        for mode in ("blank", "single", "interrupted"):
            with self.subTest(mode=mode):
                raw = np.zeros((210, 210), np.uint8)
                if mode != "blank":
                    cv2.line(raw, (83, 131), (83, 160), 255, 1)
                if mode == "interrupted":
                    cv2.line(raw, (97, 131), (97, 160), 255, 1)
                    raw[131:161:3, 83:98] = 0
                result = measure_head_neck_junction(raw, corners, self.profile)
                self.assertFalse(result.accepted)
                self.assertIsNone(result.start_gap_px)

    def test_legacy_neck_diagnostic_does_not_determine_a_lone_quads_scale(self):
        for distance, angle in ((.35, 35.), (.35, 45.), (.35, 60.), (.9, 35.), (.9, 45.)):
            with self.subTest(distance=distance, angle=angle):
                outer, camera = self.reference.projection(angle, distance_m=distance)
                matrix = np.array(((camera.fx_px, 0., camera.cx_px),
                                   (0., camera.fy_px, camera.cy_px), (0., 0., 1.)))
                # Test scene: the 71 mm paper is visible; all 78 mm outer-head
                # rails are absent. The physical head's genuine neck remains.
                paper_points = np.array([(p.x_m * 71 / 78, p.y_m * 71 / 78, p.z_m)
                                         for p in self.profile.head_corners])
                paper = cv2.projectPoints(paper_points, np.array((0., math.radians(angle), 0.)),
                    np.array((0., 0., distance)), matrix, np.zeros(4))[0].reshape(-1, 2)
                raw = np.zeros((600, 800), np.uint8)
                cv2.polylines(raw, [np.rint(paper).astype(np.int32)], True, 255, 1)
                points = np.array([(p.u_px, p.v_px) for p in outer])
                center = (points[2] + points[3]) / 2.
                width = np.linalg.norm(points[2] - points[3])
                height = (np.linalg.norm(points[0] - points[3]) + np.linalg.norm(points[1] - points[2])) / 2.
                for sign in (-1, 1):
                    x, y = round(center[0] + sign * .07 * width), round(center[1])
                    cv2.line(raw, (x, y), (x, round(y + .6 * height)), 255, 1)
                acquisition = acquire_head_proposal(cv2, cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR),
                    raw_edges=raw, expected_head_center_u_px=float(points[:, 0].mean()),
                    expected_head_center_v_px=float(points[:, 1].mean()), expected_head_height_px=height)
                self.assertIsNotNone(acquisition.proposal, acquisition.reason)
                corners = acquisition.proposal.corners
                self.assertTrue(_short_centered_neck_support(raw, corners))
                junction = measure_head_neck_junction(raw, corners, self.profile)
                self.assertFalse(junction.accepted)
                self.assertEqual(junction.reason, "head_neck_junction_gap_too_large")
                self.assertGreater(junction.start_gap_px, junction.max_start_gap_px)
                if distance == .9:
                    # The formerly accepted two-row paper/neck gap is now
                    # below the resolution needed for the old fixed2px cap.
                    self.assertEqual(junction.start_gap_px, 2)
                    self.assertEqual(junction.max_start_gap_px, 0)
                estimate, debug, _pose = fit_current_measured_head(cv2, raw,
                    model_profile=self.profile, camera=camera, proposal_corners=corners)
                # A standalone quad has an angle but no independent scale
                # identity. Its detached neck is not admissibility evidence.
                self.assertTrue(estimate.usable, estimate.reason)
                self.assertIsNone(estimate.visible_face)
                self.assertIsNone(debug.head_neck_junction)
                self.assertEqual(debug.head_outer_recovery.scale_identifiability,
                                 "conditional_on_candidate_and_measured_profile")

    def test_recorded_junction_gaps_preserve_pixels_without_tuning_to_old_fit(self):
        _fixture, edges, corners, camera = self.reference.recorded_inputs()
        estimate, debug, _pose = fit_current_measured_head(cv2, edges, model_profile=self.profile,
            camera=camera, proposal_corners=corners)
        latest = measure_head_neck_junction(edges, debug.refined_corners, self.profile)
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertTrue(latest.accepted)
        self.assertEqual(latest.start_gap_px, 2)
        self.assertEqual(latest.max_start_gap_px, 2)

        # This older fit has a four-row gap. Its old low head RMSE does not
        # justify widening the junction tolerance or moving the observed edge.
        geometry_fixture.GeometryContractTest.setUpClass()
        previous = geometry_fixture.GeometryContractTest.debug
        recorded_corners = tuple(ImagePoint(**p) for p in
            geometry_fixture.GeometryContractTest.frame["geometry_refined_corners"])
        gap = measure_head_neck_junction(previous.raw_edges, recorded_corners, self.profile)
        self.assertEqual(gap.start_gap_px, 4)
        self.assertFalse(gap.accepted)
        recovered = measure_head_neck_junction(previous.raw_edges, previous.refined_corners, self.profile)
        self.assertTrue(recovered.accepted)
        self.assertEqual(recovered.start_gap_px, 0)
        self.assertTrue(geometry_fixture.GeometryContractTest.estimate.usable)

        selection, _metadata = RecordedBacksideFixture(cv2, np).evaluate("frame_000008", BacksideProposalReuse())
        backside = selection.selected.debug
        before = backside.raw_edges.copy()
        gap = measure_head_neck_junction(backside.raw_edges, backside.refined_corners, self.profile)
        # Shared current-border acquisition now selects a complete bound head.
        # The legacy neck diagnostic remains independent: it neither selects
        # these rails nor supplies angle, identity, or backside admission proof.
        np.testing.assert_array_equal(backside.raw_edges, before)
        self.assertTrue(selection.selected.estimate.usable)
        self.assertEqual(len(backside.refined_corners), 4)
        self.assertIsNone(backside.head_neck_junction)
        self.assertTrue(backside.head_acquisition_diagnostics["selected_border_binding"]["accepted"])
        self.assertTrue(gap.accepted)
        self.assertEqual(gap.start_gap_px, 0)


if __name__ == "__main__":
    unittest.main()
