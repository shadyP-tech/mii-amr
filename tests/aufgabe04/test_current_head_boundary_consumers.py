"""All head consumers bind physical-border evidence to this exact current fit."""

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.perception.stand_axis.head_backside_appearance import (
    assess_current_head_backside_appearance,
)
from scripts.aufgabe04.perception.stand_axis.head_backside_classification import (
    classify_current_head_backside,
)
from scripts.aufgabe04.perception.stand_axis.head_model_admission import (
    admit_measured_head_model,
)
from scripts.aufgabe04.perception.stand_axis.head_outer_border import (
    HeadMarkerBoundaryEvidence,
)
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.real_robot.observer.head_observation_window import (
    current_head_window_input,
)
from tests.aufgabe04.test_head_backside_classification import classified_head
from tests.aufgabe04.test_head_model_admission import outer_boundary, quality


def shifted(corners, du=.01, dv=0.):
    return tuple(ImagePoint(p.u_px + du, p.v_px + dv) for p in corners)


class CurrentHeadBoundaryConsumerTests(unittest.TestCase):
    @staticmethod
    def recovered_head():
        estimate, debug, options = classified_head()
        corners = estimate.corners
        inner = tuple(ImagePoint(80. + (p.u_px - 80.) / 1.1,
                                 80. + (p.v_px - 80.) / 1.1) for p in corners)
        boundary = outer_boundary(
            corners, reason="current_raw_outer_head_recovered", original_corners=inner,
            recovered=True, independently_resolved=True,
            current_raw_alternatives=(inner, corners),
            head_size_m=(.078, .078), largest_inset_size_m=(.071, .071),
            model_tolerance_m=.002,
        )
        return estimate, replace(debug, head_outer_recovery=boundary,
                                 predicted_corners=corners), options

    def window(self, estimate, debug):
        return current_head_window_input(
            estimate, debug, frame_stamp_sec=10., camera_signature=(640., 640., 80., 80.),
            roi=SimpleNamespace(x0=10, y0=20), projected_center_px=(90., 100.),
            expected_head_height_px=90.,
        )

    def decisions(self, estimate, debug, options):
        _, classified = classify_current_head_backside(estimate, debug, **options)
        return {
            "angle": admit_measured_head_model(
                estimate=estimate, debug=debug, yaw_rad=math.radians(estimate.yaw_deg),
            ).accepted,
            "appearance": assess_current_head_backside_appearance(
                estimate, debug, **options,
            ).accepted,
            "classification": classified.head_backside_classification.accepted,
            "temporal": self.window(estimate, debug) is not None,
        }

    def assert_all_reject(self, estimate, debug, options):
        for consumer, accepted in self.decisions(estimate, debug, options).items():
            with self.subTest(consumer=consumer):
                self.assertFalse(accepted)

    def test_same_current_border_is_accepted_across_all_consumers(self):
        estimate, debug, options = classified_head()
        self.assertTrue(all(self.decisions(estimate, debug, options).values()))
        current = self.window(estimate, debug)
        self.assertEqual(current.corners_full_image,
                         tuple((p.u_px + 10, p.v_px + 20) for p in estimate.corners))

    def test_missing_rejected_or_untyped_boundary_is_rejected_everywhere(self):
        estimate, debug, options = classified_head()
        for boundary in (
            None,
            replace(debug.head_outer_recovery, accepted=False),
            replace(debug.head_outer_recovery, reason="current_physical_head_boundary_unresolved"),
            SimpleNamespace(**vars(debug.head_outer_recovery)),
        ):
            with self.subTest(boundary=boundary):
                self.assert_all_reject(estimate, replace(debug, head_outer_recovery=boundary), options)

    def test_exact_corner_profile_and_proposal_mismatches_reject_everywhere(self):
        estimate, debug, options = classified_head()
        proof = debug.head_outer_recovery
        mutations = {
            "different_current_fit": (replace(estimate, corners=shifted(estimate.corners)), debug),
            "different_corner_order": (
                replace(estimate, corners=estimate.corners[1:] + estimate.corners[:1]), debug),
            "proof_selected_pixel_changed": (estimate, replace(
                debug, head_outer_recovery=replace(proof, recovered_corners=shifted(estimate.corners)))),
            "proof_profile_changed": (estimate, replace(
                debug, head_outer_recovery=replace(proof, profile_sha256="b" * 64))),
            "proof_head_dimensions_changed": (estimate, replace(
                debug, head_outer_recovery=replace(proof, head_size_m=(.10, .10)))),
            "debug_profile_changed": (estimate, replace(debug, model_profile_sha256="b" * 64)),
            "different_current_proposal": (estimate, replace(
                debug, predicted_corners=shifted(estimate.corners))),
            "proof_proposal_changed": (estimate, replace(
                debug, predicted_corners=estimate.corners,
                head_outer_recovery=replace(proof, neutral_proposal_corners=shifted(estimate.corners)))),
            "proof_original_not_current": (estimate, replace(
                debug, head_outer_recovery=replace(proof, original_corners=shifted(estimate.corners)))),
            "proof_selected_not_current": (estimate, replace(
                debug, head_outer_recovery=replace(proof, current_raw_alternatives=()))),
            "proof_is_prediction": (estimate, replace(
                debug, head_outer_recovery=replace(proof, selection_source="previous_frame_prediction"))),
        }
        for case, (current, artifact) in mutations.items():
            with self.subTest(case=case):
                self.assert_all_reject(current, artifact, options)

    def test_unresolved_marker_disagreement_does_not_enter_any_head_consumer(self):
        estimate, debug, options = classified_head()
        diagnostic = HeadMarkerBoundaryEvidence(
            False, "current_border_matches_verified_qr_panel", requests_reconsideration=True,
        )
        self.assert_all_reject(estimate, replace(debug, head_marker_boundary=diagnostic), options)

    def test_independently_recovered_head_is_accepted_by_all_consumers(self):
        estimate, debug, options = self.recovered_head()
        self.assertTrue(all(self.decisions(estimate, debug, options).values()))

    def test_ratio_disagreement_is_supporting_but_current_marker_still_vetoes_backside(self):
        estimate, debug, options = self.recovered_head()
        diagnostic = HeadMarkerBoundaryEvidence(
            False, "current_border_matches_verified_qr_panel", requests_reconsideration=True,
        )
        for qr_detected, marker_verified in ((False, True), (True, True), (True, False)):
            current = replace(debug, qr_detected=qr_detected, qr_marker_verified=marker_verified,
                              head_marker_boundary=diagnostic)
            with self.subTest(qr_detected=qr_detected, marker_verified=marker_verified):
                self.assertEqual(self.decisions(estimate, current, options), {
                    "angle": True, "appearance": False, "classification": False, "temporal": True,
                })
                self.assertTrue(diagnostic.diagnostic_only)
                self.assertFalse(diagnostic.supplies_angle)
                self.assertEqual(self.window(estimate, current).yaw_rad,
                                 math.radians(estimate.yaw_deg))

    def test_thick_paper_or_unproved_growth_cannot_override_ratio_disagreement(self):
        estimate, debug, options = self.recovered_head()
        proof = debug.head_outer_recovery
        diagnostic = HeadMarkerBoundaryEvidence(
            False, "current_border_matches_verified_qr_panel", requests_reconsideration=True,
        )
        # Two pixels between a rail's inner and outer strokes can have enough
        # area growth for a rectangle detector, but not physical frame evidence.
        near_inner = tuple(ImagePoint(80. + (p.u_px - 80.) * 88./90.,
                                      80. + (p.v_px - 80.) * 88./90.) for p in estimate.corners)
        one_axis_inner = tuple(ImagePoint(80. + (p.u_px - 80.) * 88./90.,
                                          80. + (p.v_px - 80.) * 80./90.) for p in estimate.corners)
        unproved = (
            replace(proof, independently_resolved=False),
            replace(proof, model_tolerance_m=None),
            replace(proof, largest_inset_size_m=None),
            replace(proof, original_corners=near_inner,
                    current_raw_alternatives=(near_inner, estimate.corners)),
            replace(proof, original_corners=one_axis_inner,
                    current_raw_alternatives=(one_axis_inner, estimate.corners)),
        )
        for boundary in unproved:
            with self.subTest(boundary=boundary):
                self.assert_all_reject(estimate, replace(
                    debug, head_outer_recovery=boundary, head_marker_boundary=diagnostic,
                ), options)

    def test_angle_ambiguity_does_not_destroy_independent_backside_appearance(self):
        estimate, debug, options = classified_head()
        estimate = replace(estimate, usable=False, reason="head_model_planar_axis_ambiguous")
        debug = replace(
            debug, head_model_quality=quality(
                accepted=False, reason="head_model_planar_axis_ambiguous", axis_ambiguous=True,
            ),
            head_pose_hypotheses=(SimpleNamespace(
                positive_depth=True, reprojection_rmse_px=.4, yaw_deg=37.815,
            ),),
        )
        decisions = self.decisions(estimate, debug, options)
        self.assertEqual(decisions, {
            "angle": False, "appearance": True, "classification": False, "temporal": True,
        })

    def test_temporal_ambiguity_keeps_the_producers_full_pixel_noise_neighborhood(self):
        estimate, debug, _options = classified_head()
        estimate = replace(estimate, usable=False, yaw_deg=None,
                           reason="head_model_planar_axis_ambiguous")
        debug = replace(debug, head_model_quality=quality(
            accepted=False, reason="head_model_planar_axis_ambiguous", axis_ambiguous=True,
            reprojection_rmse_px=.2), head_pose_hypotheses=tuple(
                SimpleNamespace(positive_depth=True, reprojection_rmse_px=residual, yaw_deg=yaw)
                for residual, yaw in ((.2, -7.), (.6, 7.))))
        current = self.window(estimate, debug)
        self.assertEqual(current.plausible_yaws_rad, (math.radians(-7.), math.radians(7.)))

    def test_retained_diagnostic_poses_cannot_promote_other_failed_quality(self):
        estimate, debug, _options = classified_head()
        estimate = replace(estimate, usable=False, yaw_deg=None)
        retained = (SimpleNamespace(positive_depth=True, reprojection_rmse_px=.2, yaw_deg=28.),)
        for changes in (
            dict(accepted=False, reason="head_model_yaw_uncertainty_too_high", yaw_std_deg=3.1),
            dict(accepted=False, reason="head_model_pixel_span_insufficient", minimum_edge_length_px=20.),
            dict(accepted=False, reason="head_model_outer_border_unverified", outer_border_verified=False),
            dict(accepted=False, reason="head_model_planar_axis_ambiguous", axis_ambiguous=True,
                 minimum_edge_length_px=20.),
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(self.window(estimate, replace(
                    debug, head_model_quality=quality(**changes), head_pose_hypotheses=retained)))


if __name__ == "__main__":
    unittest.main()
