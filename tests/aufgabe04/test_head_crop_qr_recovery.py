"""Current-head QR recovery prioritizes actual symbol geometry within budget."""

import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr


MODULE = "scripts.aufgabe04.qr_scanning.opencv_qr_detector."
QUAD = ((20., 20.), (80., 20.), (80., 80.), (20., 80.))


@unittest.skipIf(cv2 is None, "OpenCV unavailable")
class HeadCropQrRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((120, 120, 3), 60, np.uint8)
        self.calls = []

    def backend(self, *, quad=QUAD, multi_points=None, multi_texts=(),
                single_text="", isolated_texts=("Start",)):
        calls = self.calls

        class Native:
            def detectAndDecodeMulti(self, frame):
                calls.append("multi")
                return bool(multi_texts), multi_texts, multi_points, ()

            def detectAndDecode(self, frame):
                calls.append("single")
                return single_text, quad, None

        class Wechat:
            def detectAndDecode(self, frame):
                h, w = frame.shape[:2]
                isolated = (h, w) != (120, 120)
                calls.append("isolated" if isolated else "whole_crop")
                bounds = np.array([((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1))])
                return isolated_texts if isolated else ("Start",), bounds

        class Backend:
            def QRCodeDetector(self):
                return Native()

            def wechat_qrcode_WeChatQRCode(self):
                return Wechat()

            def __getattr__(self, name):
                return getattr(cv2, name)

        return Backend()

    def decode(self, backend, **kwargs):
        trace = {}
        with patch(MODULE + "_qr_decode_candidates_with_geometry",
                   return_value=((self.frame, 1., 0),)):
            result = detect_qr_observations_bgr(
                self.frame, backend, diagnostics=trace,
                prefer_native_geometry=True, **kwargs,
            )
        return result, trace

    def test_original_single_quad_is_recovered_before_whole_crop_or_pyramid(self):
        def variants(*_args):
            yield self.frame, 1., 0
            self.fail("Recovery should precede enlarged full-crop work")

        trace = {}
        with patch(MODULE + "_qr_decode_candidates_with_geometry", side_effect=variants):
            result = detect_qr_observations_bgr(
                self.frame, self.backend(), diagnostics=trace,
                prefer_native_geometry=True,
            )
        self.assertEqual(self.calls, ["multi", "single", "isolated"])
        self.assertEqual(result[0].text, "Start")
        self.assertEqual(result[0].corners, QUAD)
        self.assertEqual(result[0].detector, "opencv_quad_wechat_rectified")
        self.assertEqual(trace["events"][-1]["stage"], "opencv_single_isolated_current_head")
        self.assertEqual(trace["search_policy"], "current_head_native_geometry_first")

    def test_missing_native_corners_keep_extent_payload_provisional(self):
        result, _ = self.decode(self.backend(quad=None))
        self.assertEqual(self.calls, ["multi", "single", "whole_crop"])
        self.assertEqual(tuple(item.text for item in result), ("Start",))
        self.assertIsNone(result[0].corners)

    def test_multiple_undecoded_native_symbols_block_isolated_geometry(self):
        neighbor = tuple((u + 30., v + 30.) for u, v in QUAD)
        result, trace = self.decode(self.backend(multi_points=(QUAD, neighbor)))
        self.assertNotIn("isolated", self.calls)
        self.assertEqual(tuple(item.text for item in result), ("Start",))
        self.assertIsNone(result[0].corners)
        self.assertEqual(trace["result"]["geometry_suppressed_reason"], "multiple_native_quads")

    def test_conflicting_current_payload_is_preserved_after_isolated_recovery(self):
        result, _ = self.decode(self.backend(multi_texts=("Neighbor",)))
        self.assertEqual(tuple(item.text for item in result), ("Neighbor", "Start"))
        self.assertIsNone(result[0].corners)
        self.assertEqual(result[1].corners, QUAD)

    def test_two_isolated_symbols_never_gain_unique_geometry_even_if_text_matches(self):
        for texts in (("Start", "Neighbor"), ("Start", "Start")):
            with self.subTest(texts=texts):
                result, _ = self.decode(self.backend(isolated_texts=texts))
                self.assertEqual(tuple(item.text for item in result), texts)
                self.assertTrue(all(item.corners is None for item in result))

    def test_budget_expiry_after_native_call_does_not_start_isolation_or_wechat(self):
        now = [0.]
        backend = self.backend()
        native = backend.QRCodeDetector()
        original = native.detectAndDecodeMulti

        def slow_multi(frame):
            now[0] += .13
            return original(frame)

        native.detectAndDecodeMulti = slow_multi
        backend.QRCodeDetector = lambda: native
        with patch(MODULE + "monotonic", side_effect=lambda: now[0]):
            result, trace = self.decode(backend, max_elapsed_sec=.12)
        self.assertEqual(result, ())
        self.assertEqual(self.calls, ["multi"])
        self.assertTrue(trace["processing_budget"]["exhausted"])

    def test_default_decoder_order_is_unchanged(self):
        trace = {}
        with patch(MODULE + "_qr_decode_candidates_with_geometry",
                   return_value=((self.frame, 1., 0),)):
            result = detect_qr_observations_bgr(self.frame, self.backend(), diagnostics=trace)
        self.assertEqual(self.calls, ["whole_crop", "multi", "single", "isolated"])
        self.assertEqual(result[0].corners, QUAD)
        self.assertEqual(trace["events"][-1]["stage"], "opencv_single_isolated_deferred")
        self.assertEqual(trace["search_policy"], "default")


if __name__ == "__main__":
    unittest.main()
