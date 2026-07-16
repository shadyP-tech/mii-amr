import math
import unittest

from scripts.aufgabe04.perception.stand_axis_consensus import (
    AxisConsensusAccumulator,
    axis_conditioning,
)


class AxisConsensusTest(unittest.TestCase):
    def test_conditioning_rejects_highly_oblique_stable_axis(self):
        result = axis_conditioning(
            math.radians(31), max_obliqueness_rad=math.radians(30)
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "oblique_silhouette")

    def test_conditioning_accepts_frontal_axis_at_threshold(self):
        result = axis_conditioning(
            math.radians(-30), max_obliqueness_rad=math.radians(30)
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "well_conditioned")

    def test_requires_stable_same_source_window(self):
        accumulator = AxisConsensusAccumulator(required_samples=3, max_deviation_rad=math.radians(5))
        self.assertIsNone(accumulator.add(yaw_rad=0.01, source="edges", side="qr_code_side", qr_texts=("A",)))
        self.assertIsNone(accumulator.add(yaw_rad=0.02, source="edges", side="qr_code_side", qr_texts=("A",)))
        result = accumulator.add(yaw_rad=0.00, source="edges", side="qr_code_side", qr_texts=("A",))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.yaw_rad, 0.01, places=3)

    def test_source_change_and_outlier_do_not_emit(self):
        accumulator = AxisConsensusAccumulator(required_samples=3, max_deviation_rad=math.radians(5))
        for yaw in (0.0, 0.01):
            self.assertIsNone(accumulator.add(yaw_rad=yaw, source="edges", side="qr_code_side", qr_texts=("A",)))
        self.assertIsNone(accumulator.add(yaw_rad=1.0, source="stem", side="qr_code_side", qr_texts=("A",)))
        self.assertIsNone(accumulator.add(yaw_rad=0.0, source="stem", side="qr_code_side", qr_texts=("A",)))
        self.assertIsNone(accumulator.add(yaw_rad=0.0, source="stem", side="qr_code_side", qr_texts=("A",)))


if __name__ == "__main__":
    unittest.main()
