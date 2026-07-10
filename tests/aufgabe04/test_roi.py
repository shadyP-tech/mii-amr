import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.roi import Rect, clamp_roi, parse_roi  # noqa: E402


class RoiTest(unittest.TestCase):
    def test_parse_roi_accepts_x_y_width_height(self):
        self.assertEqual(parse_roi("1, 2, 30, 40"), Rect(1, 2, 30, 40))

    def test_parse_roi_rejects_wrong_arity(self):
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "x,y,w,h"):
            parse_roi("1,2,3")

    def test_parse_roi_rejects_non_integer_values(self):
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "integers"):
            parse_roi("1,2,nope,4")

    def test_parse_roi_rejects_non_positive_size(self):
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "positive"):
            parse_roi("1,2,0,4")

    def test_clamp_roi_keeps_in_bounds_roi(self):
        roi = Rect(2, 3, 4, 5)

        self.assertEqual(clamp_roi(roi, (20, 30, 3)), roi)

    def test_clamp_roi_clamps_negative_origin(self):
        self.assertEqual(clamp_roi(Rect(-3, -2, 10, 9), (20, 30)), Rect(0, 0, 10, 9))

    def test_clamp_roi_clamps_oversized_roi(self):
        self.assertEqual(clamp_roi(Rect(25, 18, 10, 10), (20, 30)), Rect(25, 18, 5, 2))

    def test_clamp_roi_returns_zero_area_when_out_of_frame(self):
        self.assertEqual(clamp_roi(Rect(40, 5, 10, 10), (20, 30)), Rect(30, 5, 0, 10))


if __name__ == "__main__":
    unittest.main()
