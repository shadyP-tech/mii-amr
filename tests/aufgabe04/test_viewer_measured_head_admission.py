"""Viewer handoff must agree with the model source and expose stale fits."""

from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.perception.debug.viewer_axis_admission import (
    current_head_quality_ready, viewer_axis_admission,
    viewer_color_side_allowed, viewer_face_export_allowed,
)
from scripts.aufgabe04.perception.debug.stand_model_overlay import annotate_metric_model_status
from scripts.aufgabe04.perception.stand_axis.model_profile import load_stand_model
from scripts.aufgabe04.perception.stand_axis_consensus import AxisConsensus
from tests.aufgabe04.test_head_model_admission import head_debug, head_estimate, quality


class MeasuredHeadViewerTests(unittest.TestCase):
    def test_color_only_head_cannot_export_an_opposite_side_observation(self):
        from scripts.aufgabe04.perception.stand_side_classification import classify_stand_side
        head = head_estimate(44.2)
        side = classify_stand_side(
            qr_texts=(), color_confidence=1., min_color_confidence=.6,
            allow_color_only=viewer_color_side_allowed(head, requested=True),
        )
        self.assertEqual(side.side, "unknown_side")
        self.assertFalse(viewer_face_export_allowed(head, side.side))
        self.assertFalse(viewer_face_export_allowed(head, "basic_color_side"))
        self.assertTrue(viewer_face_export_allowed(head, "qr_code_side"))
        self.assertTrue(viewer_color_side_allowed(
            replace(head, source="legacy_color_head"), requested=True))

    def admission(self, estimate, debug):
        return viewer_axis_admission(
            consensus=AxisConsensus(math.radians(estimate.yaw_deg), 7, .01, estimate.source),
            estimate=estimate, artifacts=debug, max_obliqueness_rad=math.radians(30.),
        )

    def test_high_angle_model_quality_does_not_change_legacy_silhouette_limit(self):
        head = head_estimate(44.2)
        admitted = self.admission(head, head_debug())
        self.assertTrue(admitted.accepted)
        self.assertIsNone(admitted.max_obliqueness_rad)
        self.assertEqual(admitted.policy, "current_measured_head_quality")
        legacy = self.admission(replace(head, source="model_current_frame_refined"), head_debug())
        self.assertFalse(legacy.accepted)
        self.assertEqual(legacy.reason, "oblique_silhouette")

    def test_quality_failure_or_prediction_never_enters_viewer_consensus(self):
        for head, debug in (
            (head_estimate(20.), head_debug(head_model_quality=quality(yaw_std_deg=4.))),
            (head_estimate(44.2, evidence_state="predicted_only"), head_debug()),
            (head_estimate(44.2, usable=False), head_debug()),
        ):
            self.assertFalse(current_head_quality_ready(head, debug))
            self.assertFalse(self.admission(head, debug).accepted)

    def test_overlay_labels_obsolete_fit_without_changing_purple_model_style(self):
        profile = load_stand_model(Path("configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"))
        for fresh in (True, False):
            rows = []
            cursor = SimpleNamespace(draw=lambda _cv2, _frame, text, **kwargs: rows.append((text, kwargs)))
            annotate_metric_model_status(
                SimpleNamespace(FONT_HERSHEY_SIMPLEX=0), object(), profile=profile,
                inputs_ready=True, estimate=head_estimate(44.2), artifacts=head_debug(),
                text_cursor=cursor, result_fresh=fresh,
            )
            joined = " ".join(row[0] for row in rows)
            self.assertIn("pixel_model_yaw_std=1.46deg", joined)
            if fresh:
                self.assertIn("current_head_yaw=44.2deg", joined)
                self.assertEqual(rows[0][1]["color"], (0, 255, 0))
            else:
                self.assertIn("model=obsolete_result", joined)
                self.assertNotIn("current_head_yaw=", joined)
                self.assertEqual(rows[0][1]["color"], (255, 0, 255))


if __name__ == "__main__":
    unittest.main()
