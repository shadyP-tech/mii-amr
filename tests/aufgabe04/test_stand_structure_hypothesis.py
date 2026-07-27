import sys
import unittest
from pathlib import Path

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover
    cv2 = None
    numpy = None


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.stand_structure_hypothesis import (  # noqa: E402
    evaluate_stand_structure,
)


@unittest.skipIf(cv2 is None or numpy is None, "OpenCV and numpy are required")
class StandStructureHypothesisTest(unittest.TestCase):
    def make_edges(
        self,
        *,
        include_top=True,
        include_left=True,
        include_right=True,
        include_stem=True,
        include_base=True,
    ):
        edges = numpy.zeros((360, 320), dtype=numpy.uint8)
        if include_top:
            cv2.line(edges, (90, 45), (230, 42), 255, 2)
        if include_left:
            cv2.line(edges, (90, 45), (94, 175), 255, 2)
        if include_right:
            cv2.line(edges, (230, 42), (226, 174), 255, 2)
        # Deliberately omit the lower head edge at the neck.
        if include_stem:
            cv2.line(edges, (153, 176), (153, 292), 255, 2)
            cv2.line(edges, (167, 176), (167, 292), 255, 2)
        if include_base:
            cv2.line(edges, (160, 292), (45, 329), 255, 2)
            cv2.line(edges, (160, 292), (276, 326), 255, 2)
            cv2.line(edges, (45, 329), (276, 326), 255, 2)
        return edges

    @property
    def rough_corners(self):
        return ((90.0, 44.0), (230.0, 43.0), (227.0, 175.0), (93.0, 175.0))

    def evaluate(self, edges):
        return evaluate_stand_structure(
            cv2,
            edges,
            self.rough_corners,
            stem_center_x=160.0,
            stem_top_y=175.0,
            min_aspect_ratio=0.70,
            max_aspect_ratio=1.30,
        )

    def test_accepts_three_sided_head_owned_by_stem_and_base(self):
        evidence = self.evaluate(self.make_edges())

        self.assertTrue(evidence.accepted, evidence.reason)
        self.assertTrue(evidence.tracking_supported)
        self.assertEqual(evidence.reason, "structure_owned_head_supported")
        self.assertIsNotNone(evidence.corners)
        self.assertGreaterEqual(evidence.head_top_support, 0.55)
        self.assertGreaterEqual(evidence.head_left_support, 0.55)
        self.assertGreaterEqual(evidence.head_right_support, 0.55)
        self.assertGreaterEqual(evidence.base_support, 0.55)

    def test_rejects_stem_and_base_without_head_side(self):
        evidence = self.evaluate(self.make_edges(include_left=False))

        self.assertFalse(evidence.accepted)
        self.assertFalse(evidence.tracking_supported)
        self.assertIn("head_left", evidence.reason)

    def test_rejects_head_and_stem_without_base(self):
        evidence = self.evaluate(self.make_edges(include_base=False))

        self.assertFalse(evidence.accepted)
        self.assertTrue(evidence.tracking_supported)
        self.assertEqual(evidence.reason, "structure_base_unsupported")

    def test_rejects_full_width_floor_seam_as_base(self):
        edges = self.make_edges(include_base=False)
        cv2.line(edges, (0, 326), (319, 326), 255, 2)

        evidence = self.evaluate(edges)

        self.assertFalse(evidence.accepted)
        self.assertEqual(evidence.reason, "structure_base_unsupported")

    def test_status_dict_excludes_pixel_mask(self):
        evidence = self.evaluate(self.make_edges())

        status = evidence.status_dict()

        self.assertNotIn("evidence_mask", status)
        self.assertTrue(status["accepted"])


if __name__ == "__main__":
    unittest.main()
