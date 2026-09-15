"""The viewer must not make rejected or expired models look admissible."""

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.perception.debug.stand_model_overlay import (
    annotate_model_prediction, annotate_projected_model_landmarks,
)
from scripts.aufgabe04.perception.debug.viewer_frame_timing import ViewerFrameTiming
from scripts.aufgabe04.perception.debug.viewer_model_overlay_policy import (
    current_crop_head_estimate, current_model_overlay_state, estimate_in_full_image,
)
from scripts.aufgabe04.perception.debug.viewer_axis_admission import (
    current_axis_evidence_ready, viewer_axis_admission,
)
from scripts.aufgabe04.perception.debug.stand_axis_viewer import _select_axis_pipeline_result
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis_consensus import AxisConsensus
from tests.aufgabe04.test_head_model_admission import head_debug, head_estimate, quality, outer_boundary


class Drawing:
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self):
        self.lines = []
        self.labels = []

    def line(self, frame, start, end, color, thickness):
        self.lines.append((start, end, color, thickness))

    def putText(self, frame, text, position, font, scale, color, thickness):
        self.labels.append((text, position, color))


class ViewerModelOverlayStateTests(unittest.TestCase):
    def state(self, **changes):
        return current_model_overlay_state(**{
            "inputs_ready": True, "estimate": head_estimate(44.2),
            "artifacts": head_debug(), "result_fresh": True, **changes,
        })

    @staticmethod
    def landmarks():
        return {
            name: ImagePoint(x, y) for name, x, y in (
                ("head_top_left", 10, 10), ("head_top_right", 50, 10),
                ("head_bottom_right", 50, 50), ("head_bottom_left", 10, 50),
                ("head_back_top_left", 20, 20), ("head_back_top_right", 60, 20),
                ("head_back_bottom_right", 60, 60), ("head_back_bottom_left", 20, 60),
            )
        }

    def test_current_measured_fit_uses_shared_quality_without_neck_or_consensus(self):
        state = self.state(artifacts=head_debug(head_model_quality=quality(
            centered_neck_supported=False, neck_junction_verified=False,
        )))
        self.assertEqual(state.state, "current_head_fit")
        self.assertTrue(state.current_fit_accepted)
        self.assertEqual(state.geometry_color, (180, 0, 180))
        self.assertEqual(state.reason, "measured_head_geometry_quality_accepted")

    def test_final_panel_rejection_is_gray_even_with_retained_good_pose(self):
        estimate = replace(head_estimate(), usable=False, reason="head_border_matches_qr_panel")
        state = self.state(estimate=estimate)
        self.assertFalse(state.current_fit_accepted)
        self.assertEqual(state.state, "rejected_fit")
        self.assertEqual(state.reason, estimate.reason)
        drawing = Drawing()
        annotate_projected_model_landmarks(
            drawing, object(), self.landmarks(), color=state.geometry_color,
        )
        self.assertGreater(len(drawing.lines), 4)
        self.assertEqual({line[2] for line in drawing.lines}, {(150, 150, 150)})

    def test_claimed_accepted_quality_does_not_override_shared_limits(self):
        for changes in ({"yaw_std_deg": 4.0}, {"outer_border_verified": False},
                        {"axis_ambiguous": True}):
            with self.subTest(changes=changes):
                state = self.state(artifacts=head_debug(head_model_quality=quality(**changes)))
                self.assertFalse(state.current_fit_accepted)
                self.assertEqual(state.state, "rejected_fit")
                self.assertNotEqual(state.geometry_color, (180, 0, 180))

    def test_result_can_expire_between_detection_and_render_without_changing_fit(self):
        timing = ViewerFrameTiming(10.0, 9.98)
        estimate = head_estimate()
        options = dict(max_result_age_sec=.18, max_frame_age_sec=.25)
        completed = timing.assess(now_sec=10.10, **options)
        rendered = timing.assess(now_sec=10.20, **options)
        self.assertTrue(self.state(estimate=estimate, result_fresh=completed.accepted).current_fit_accepted)
        state = self.state(
            estimate=estimate, result_fresh=rendered.accepted,
            freshness_reason=rendered.reason,
        )
        self.assertEqual(state.state, "obsolete_result")
        self.assertEqual(state.reason, "observation_too_old")
        self.assertFalse(state.current_fit_accepted)
        self.assertTrue(estimate.usable)

    def test_predicted_corners_are_amber_and_explicitly_labeled_proposal(self):
        estimate = head_estimate(evidence_state="predicted_only")
        state = self.state(estimate=estimate)
        self.assertEqual(state.state, "proposal_only")
        self.assertFalse(state.current_fit_accepted)
        drawing = Drawing()
        annotate_model_prediction(drawing, object(), estimate.corners, x_offset=90, y_offset=80)
        self.assertEqual({line[2] for line in drawing.lines}, {(0, 190, 255)})
        self.assertEqual(drawing.labels[0][0], "proposal")
        self.assertEqual(drawing.lines[0][0], (
            round(estimate.corners[0].u_px + 90), round(estimate.corners[0].v_px + 80),
        ))

    def test_landmarks_without_admission_default_to_gray_diagnostics(self):
        drawing = Drawing()
        annotate_projected_model_landmarks(drawing, object(), self.landmarks())
        self.assertEqual({line[2] for line in drawing.lines}, {(150, 150, 150)})

    def test_accepted_geometry_keeps_crop_offsets_and_dashed_design(self):
        drawing = Drawing()
        state = self.state()
        annotate_projected_model_landmarks(
            drawing, object(), self.landmarks(), x_offset=100, y_offset=200,
            color=state.geometry_color,
        )
        self.assertGreater(len(drawing.lines), 8)
        self.assertEqual(drawing.lines[0][0], (120, 220))
        self.assertEqual({line[2] for line in drawing.lines}, {(180, 0, 180)})

    def test_missing_metric_inputs_do_not_label_fallback_an_accepted_model(self):
        state = self.state(inputs_ready=False, artifacts=None)
        self.assertEqual(state.state, "inputs_unavailable")
        self.assertFalse(state.current_fit_accepted)

    def test_nonzero_crop_keeps_overlay_and_handoff_in_exact_proof_coordinates(self):
        corners = tuple(ImagePoint(x, y) for x, y in (
            (.1, .2), (90.1, .2), (90.1, 90.2), (.1, 90.2),
        ))
        metric = head_estimate(44.2, corners=corners)
        artifacts = head_debug(head_outer_recovery=outer_boundary(corners))
        crop, selected_artifacts = _select_axis_pipeline_result(
            model_only=True, metric_estimate=metric, metric_artifacts=artifacts,
            fallback_estimate=None, fallback_artifacts=None,
        )
        roi = SimpleNamespace(x0=93, y0=67)
        display = estimate_in_full_image(crop, roi)
        # This reproduces the regression: image-coordinate corners cannot be
        # checked against a crop-coordinate physical border receipt.
        self.assertFalse(current_axis_evidence_ready(display, selected_artifacts))
        admitted = current_crop_head_estimate(
            crop_estimate=crop, displayed_estimate=display, roi=roi,
            current_accepted=True, held=False,
        )
        self.assertIs(admitted, crop)
        self.assertIs(admitted.corners, corners)
        self.assertIs(selected_artifacts, artifacts)
        self.assertTrue(current_axis_evidence_ready(admitted, selected_artifacts))
        state = self.state(estimate=admitted, artifacts=selected_artifacts)
        self.assertEqual(state.geometry_color, (180, 0, 180))
        consensus = AxisConsensus(math.radians(44.2), 7, .01, admitted.source)
        handoff = viewer_axis_admission(
            consensus=consensus, estimate=admitted, artifacts=selected_artifacts,
            max_obliqueness_rad=math.radians(35.),
        )
        self.assertTrue(handoff.accepted)
        self.assertEqual(display.corners[0], ImagePoint(93.1, 67.2))

    def test_held_changed_or_rejected_display_cannot_borrow_current_crop_admission(self):
        crop, roi = head_estimate(), SimpleNamespace(x0=93, y0=67)
        display = estimate_in_full_image(crop, roi)
        cases = (
            (display, True, True),
            (display, False, False),
            (replace(display, yaw_deg=12.), True, False),
            (replace(display, source="previous_head"), True, False),
            (replace(display, corners=display.corners[1:] + display.corners[:1]), True, False),
        )
        for selected, fresh, held in cases:
            with self.subTest(fresh=fresh, held=held, selected=selected):
                admitted = current_crop_head_estimate(
                    crop_estimate=crop, displayed_estimate=selected, roi=roi,
                    current_accepted=fresh, held=held,
                )
                self.assertFalse(current_axis_evidence_ready(admitted, head_debug()))
                self.assertFalse(self.state(estimate=admitted).current_fit_accepted)


if __name__ == "__main__":
    unittest.main()
