import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_texts_bgr  # noqa: E402


class FakeDetector:
    def __init__(self, multi_result, single_result):
        self.multi_result = multi_result
        self.single_result = single_result

    def detectAndDecodeMulti(self, frame):
        return self.multi_result

    def detectAndDecode(self, frame):
        return self.single_result


class FakeCv2:
    def __init__(self, detector, wechat_detector=None):
        self.detector = detector
        if wechat_detector is not None:
            self.wechat_qrcode_WeChatQRCode = lambda: wechat_detector

    def QRCodeDetector(self):
        return self.detector


class FakeWeChatDetector:
    def __init__(self, result):
        self.result = result

    def detectAndDecode(self, frame):
        return self.result


class OpenCvQRDetectorTest(unittest.TestCase):
    def test_returns_nonblank_multi_detect_texts(self):
        cv2 = FakeCv2(FakeDetector((True, (" QR_001 ", "", "DEPOT_01"), None, None), ("", None, None)))

        self.assertEqual(detect_qr_texts_bgr(object(), cv2), ("QR_001", "DEPOT_01"))

    def test_falls_back_to_single_detect(self):
        cv2 = FakeCv2(FakeDetector((False, (), None, None), (" qr_002 ", None, None)))

        self.assertEqual(detect_qr_texts_bgr(object(), cv2), ("qr_002",))

    def test_falls_back_when_multi_detect_has_only_blanks(self):
        cv2 = FakeCv2(FakeDetector((True, (" ", ""), None, None), (" depot_01 ", None, None)))

        self.assertEqual(detect_qr_texts_bgr(object(), cv2), ("depot_01",))

    def test_tolerates_short_or_odd_opencv_results(self):
        cv2 = FakeCv2(FakeDetector((True,), ("", None, None)))

        self.assertEqual(detect_qr_texts_bgr(object(), cv2), ())

    def test_falls_back_to_wechat_qr_detector_when_qrcode_detector_decodes_nothing(self):
        cv2 = FakeCv2(
            FakeDetector((False, (), None, None), ("", None, None)),
            FakeWeChatDetector(((" QR_003 ", ""), None)),
        )

        self.assertEqual(detect_qr_texts_bgr(object(), cv2), ("QR_003",))

    def test_qrcode_detector_result_wins_over_wechat_fallback(self):
        cv2 = FakeCv2(
            FakeDetector((False, (), None, None), ("QR_004", None, None)),
            FakeWeChatDetector((("QR_005",), None)),
        )

        self.assertEqual(detect_qr_texts_bgr(object(), cv2), ("QR_004",))


if __name__ == "__main__":
    unittest.main()
