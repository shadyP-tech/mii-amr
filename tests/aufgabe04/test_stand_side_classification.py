import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.stand_axis_image import ImagePoint, StandAxisImageEstimate  # noqa: E402
from scripts.aufgabe04.perception.stand_side_classification import (  # noqa: E402
    classify_stand_side,
    classify_stand_side_from_frame,
)


class FakeFrame:
    shape = (120, 160, 3)

    def __init__(self, name="full"):
        self.name = name

    def __getitem__(self, key):
        return FakeFrame("crop")


class FakeCv2:
    def getPerspectiveTransform(self, src, dst):
        return object()

    def warpPerspective(self, frame, transform, size):
        return FakeFrame("rectified")


class FakeNumpy:
    float32 = "float32"

    def array(self, values, dtype=None):
        return values


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

    def test_color_only_can_be_forced_to_fail_closed(self):
        side = classify_stand_side(
            qr_texts=(),
            color_confidence=0.9,
            allow_color_only=False,
        )
        self.assertEqual(side.side, "unknown_side")
        self.assertEqual(side.reason, "color_only_evidence_not_allowed")

    def test_frame_classifier_tries_rectified_crop_before_full_frame(self):
        estimate = StandAxisImageEstimate(
            usable=True,
            reason="axis_estimated",
            mode="face_visible",
            corners=(
                ImagePoint(20, 20),
                ImagePoint(100, 20),
                ImagePoint(100, 100),
                ImagePoint(20, 100),
            ),
            axis_line=None,
            left_height_px=80.0,
            right_height_px=80.0,
            height_ratio=1.0,
            yaw_proxy=0.0,
            yaw_deg=None,
            closer_side="equal",
            contour_area_px=6400.0,
        )
        scanned = []

        def detect_qr_texts_bgr(frame, cv2):
            scanned.append(frame.name)
            return ("QR_001",) if frame.name == "crop" else ()

        side = classify_stand_side_from_frame(
            FakeCv2(),
            FakeNumpy(),
            FakeFrame(),
            None,
            estimate,
            detect_qr_texts_bgr=detect_qr_texts_bgr,
        )

        self.assertEqual(side.side, "qr_code_side")
        self.assertEqual(side.qr_texts, ("QR_001",))
        self.assertEqual(scanned, ["rectified", "crop"])


if __name__ == "__main__":
    unittest.main()
