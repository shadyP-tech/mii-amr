"""Camera identity decoding must not depend on QR outline or corner evidence."""

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr


MODULE = "scripts.aufgabe04.qr_scanning.opencv_qr_detector."


class IdentityOnlyQrTest(unittest.TestCase):
    def setUp(self):
        self.frame = SimpleNamespace(shape=(100, 100, 3))
        self.native = Mock()
        self.native.detectAndDecodeMulti.return_value = (False, (), None)
        self.native.detectAndDecode.return_value = ("", None)
        self.wechat = Mock()
        self.wechat.detectAndDecode.return_value = ((), None)
        self.cv2 = SimpleNamespace(
            QRCodeDetector=Mock(return_value=self.native),
            wechat_qrcode_WeChatQRCode=Mock(return_value=self.wechat),
        )

    def decode(self, **kwargs):
        with ExitStack() as stack:
            # Geometry is prohibited in this mode, even if a payload backend
            # happens to return point-like output alongside its decoded text.
            geometry = [stack.enter_context(patch(MODULE + name)) for name in (
                "validated_qr_corners", "qr_corner_groups", "decode_isolated_native_quad",
            )]
            stack.enter_context(patch(MODULE + "_qr_decode_candidates_with_geometry",
                                      return_value=iter(((self.frame, 1., 0),))))
            result = detect_qr_observations_bgr(self.frame, self.cv2, identity_only=True, **kwargs)
            for operation in geometry:
                operation.assert_not_called()
        self.native.detect.assert_not_called()
        self.native.detectMulti.assert_not_called()
        return result

    def test_wechat_crop_extent_returns_id_immediately_without_native_or_corner_search(self):
        self.wechat.detectAndDecode.return_value = (
            (" Start ",), (((0, 0), (99, 0), (99, 99), (0, 99)),),
        )
        diagnostics = {}
        # ID-only takes precedence over a legacy geometry preference.
        result = self.decode(diagnostics=diagnostics, prefer_native_geometry=True)
        self.assertEqual(tuple(item.text for item in result), ("Start",))
        self.assertIsNone(result[0].corners)
        self.cv2.QRCodeDetector.assert_not_called()
        self.wechat.detectAndDecode.assert_called_once_with(self.frame)
        self.assertEqual(diagnostics["search_policy"], "identity_only")
        self.assertEqual([item["stage"] for item in diagnostics["events"]], ["wechat"])
        self.assertEqual(diagnostics["result"]["own_corner_count"], 0)

    def test_all_decoded_symbols_remain_ambiguous_including_duplicate_ids(self):
        for texts in (("Start", "QR_001"), ("Start", "Start")):
            with self.subTest(texts=texts):
                self.wechat.detectAndDecode.return_value = (("", *texts, " "), None)
                result = self.decode()
                self.assertEqual(tuple(item.text for item in result), texts)
                self.assertTrue(all(item.corners is None for item in result))
        self.cv2.QRCodeDetector.assert_not_called()

    def test_native_multi_payload_fallback_does_not_seek_single_or_isolated_geometry(self):
        self.wechat.detectAndDecode.side_effect = RuntimeError("unavailable backend")
        self.native.detectAndDecodeMulti.return_value = (True, ("Start",), object())
        result = self.decode()
        self.assertEqual(result[0].text, "Start")
        self.assertEqual(result[0].detector, "opencv_multi")
        self.assertIsNone(result[0].corners)
        self.native.detectAndDecode.assert_not_called()

    def test_native_single_payload_fallback_when_multi_has_no_id(self):
        self.native.detectAndDecodeMulti.return_value = (True, ("",), object())
        self.native.detectAndDecode.return_value = (" QR_004 ", object())
        result = self.decode()
        self.assertEqual(tuple(item.text for item in result), ("QR_004",))
        self.assertEqual(result[0].detector, "opencv_single")
        self.assertIsNone(result[0].corners)

    def test_native_multi_preserves_all_payloads(self):
        self.native.detectAndDecodeMulti.return_value = (True, ("Start", "QR_001"), object())
        result = self.decode()
        self.assertEqual(tuple(item.text for item in result), ("Start", "QR_001"))
        self.native.detectAndDecode.assert_not_called()

    def test_no_payload_does_not_invent_identity_from_backend_geometry(self):
        self.native.detectAndDecodeMulti.return_value = (False, (), object())
        self.native.detectAndDecode.return_value = ("", object())
        self.assertEqual(self.decode(), ())

    def test_budget_expiration_stops_before_the_next_backend(self):
        now = [0.]

        def slow_empty_decode(_frame):
            now[0] = .2
            return (), None

        self.wechat.detectAndDecode.side_effect = slow_empty_decode
        diagnostics = {}
        with patch(MODULE + "monotonic", side_effect=lambda: now[0]):
            result = self.decode(max_elapsed_sec=.12, diagnostics=diagnostics)
        self.assertEqual(result, ())
        self.cv2.QRCodeDetector.assert_not_called()
        self.assertTrue(diagnostics["processing_budget"]["exhausted"])


if __name__ == "__main__":
    unittest.main()
