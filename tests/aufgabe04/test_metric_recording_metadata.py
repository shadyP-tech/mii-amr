from dataclasses import replace
import json
from pathlib import Path
import unittest

from scripts.aufgabe04.perception.debug.recording_metadata import recording_metadata
from scripts.aufgabe04.perception.debug.stand_axis_viewer import (
    _unavailable_target_estimate,
    build_parser,
)
from scripts.aufgabe04.perception.stand_axis.metric_edge_association import MetricCornerArmSupport
from scripts.aufgabe04.perception.stand_axis.model_diagnostics import metric_fit_diagnostics_payload
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint, StandAxisEdgeDebugArtifacts


class MetricRecordingMetadataTest(unittest.TestCase):
    def test_rejected_fit_keeps_candidate_evidence_without_claiming_refinement(self):
        corners = tuple(ImagePoint(x, y) for x, y in ((0, 0), (10, 0), (10, 10), (0, 10)))
        support = MetricCornerArmSupport(
            radius_px=4.0, minimum_bins=2,
            bins_by_corner={
                "head_top_left": {"left": 0, "top": 3},
                "head_top_right": {"top": 3, "right": 3},
                "head_bottom_right": {"right": 3, "bottom": 3},
                "head_bottom_left": {"bottom": 3, "left": 3},
            },
        )
        debug = StandAxisEdgeDebugArtifacts(
            edges=None, candidate_corners=corners, corner_arm_support=support,
            evidence_state="predicted_only",
        )
        payload = metric_fit_diagnostics_payload(debug)
        self.assertIsNone(payload["refined_corners"])
        self.assertEqual(len(payload["candidate_corners"]), 4)
        self.assertFalse(payload["corner_arm_support"]["accepted"])
        self.assertEqual(payload["corner_arm_support"]["bins_by_corner"]["head_top_left"]["left"], 0)
        json.dumps(payload, allow_nan=False)

    def test_viewer_options_and_unavailable_estimates_are_strict_json(self):
        args = build_parser().parse_args(["--compressed-image-topic", "/camera/image_raw/compressed"])
        estimate = replace(_unavailable_target_estimate("no_fit"), left_height_px=float("nan"))
        payload = recording_metadata({"options": vars(args), "estimate": estimate, "file": Path("capture.png")})
        result = json.loads(json.dumps(payload, allow_nan=False))
        self.assertIsNone(result["estimate"]["left_height_px"])
        self.assertFalse(result["estimate"]["usable"])
        self.assertEqual(result["file"], "capture.png")

    def test_unsupported_payload_objects_are_not_silently_stringified(self):
        with self.assertRaises(TypeError):
            recording_metadata({"image": object()})


if __name__ == "__main__":
    unittest.main()
