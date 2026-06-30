import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.stand_side_classification import classify_stand_side  # noqa: E402


class StandSideClassificationTest(unittest.TestCase):
    def test_qr_code_side_wins_over_color_confidence(self):
        side = classify_stand_side(
            qr_texts=(" QR_001 ",),
            color_confidence=0.95,
            min_color_confidence=0.20,
        )

        self.assertEqual(side.side, "qr_code_side")
        self.assertEqual(side.reason, "qr_detected")
        self.assertEqual(side.qr_texts, ("QR_001",))
        self.assertAlmostEqual(side.color_confidence, 0.95)

    def test_basic_color_side_when_color_is_visible_and_no_qr(self):
        side = classify_stand_side(
            qr_texts=(),
            color_confidence=0.45,
            min_color_confidence=0.20,
        )

        self.assertEqual(side.side, "basic_color_side")
        self.assertEqual(side.reason, "stand_color_detected_without_qr")
        self.assertGreater(side.confidence, 0.0)

    def test_unknown_side_when_no_qr_and_color_is_weak(self):
        side = classify_stand_side(
            qr_texts=("", " "),
            color_confidence=0.05,
            min_color_confidence=0.20,
        )

        self.assertEqual(side.side, "unknown_side")
        self.assertEqual(side.reason, "no_qr_and_low_color_confidence")
        self.assertEqual(side.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
