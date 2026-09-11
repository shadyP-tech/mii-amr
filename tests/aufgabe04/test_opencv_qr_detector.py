import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.qr_scanning.opencv_qr_detector import (  # noqa: E402
    detect_qr_observations_bgr, detect_qr_texts_bgr,
)
from scripts.aufgabe04.qr_scanning.qr_observation import validated_qr_corners

try:
    import numpy
except ImportError:
    numpy = None


class FakeDetector:
    def __init__(self, multi_result, single_result):
        self.multi_result = multi_result
        self.single_result = single_result

    def detectAndDecodeMulti(self, frame):
        return self.multi_result

    def detectAndDecode(self, frame):
        return self.single_result


class RaisingDetector:
    def detectAndDecodeMulti(self, frame):
        raise RuntimeError("opencv decoder failed")

    def detectAndDecode(self, frame):
        raise RuntimeError("opencv decoder failed")


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
    @unittest.skipIf(numpy is None, "NumPy unavailable")
    def test_wechat_numpy_results_bind_text_to_its_own_quad_after_blank(self):
        first = ((2, 2), (12, 2), (12, 12), (2, 12))
        second = ((30, 30), (50, 30), (50, 50), (30, 50))
        cv2 = FakeCv2(
            RaisingDetector(),
            FakeWeChatDetector((numpy.array(["", " Start "]), numpy.array([first, second]))),
        )
        observations = detect_qr_observations_bgr(numpy.zeros((80, 80, 3)), cv2)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].text, "Start")
        self.assertEqual(observations[0].corners, second)
        self.assertEqual(observations[0].detector, "wechat")

    @unittest.skipIf(numpy is None, "NumPy unavailable")
    def test_native_single_and_multi_numpy_shapes_preserve_same_geometry(self):
        quad = numpy.array(((2, 2), (12, 2), (12, 12), (2, 12)))
        for points in (quad, quad.reshape(1, 4, 2)):
            for single in (False, True):
                with self.subTest(shape=points.shape, single=single):
                    cv2 = FakeCv2(FakeDetector(
                        (False, (), None) if single else (True, numpy.array(["Start"]), points),
                        ("Start", points),
                    ))
                    observations = detect_qr_observations_bgr(numpy.zeros((80, 80, 3)), cv2)
                    self.assertEqual(observations[0].corners, tuple(map(tuple, quad)))

    def test_invalid_quad_keeps_text_without_borrowing_native_geometry(self):
        for points in (None, 17, ((1, 2),), ((1, 1), (3, 3), (1, 3), (3, 1))):
            with self.subTest(points=points):
                cv2 = FakeCv2(RaisingDetector(), FakeWeChatDetector((("Start",), points)))
                observations = detect_qr_observations_bgr(object(), cv2)
                self.assertEqual(len(observations), 1)
                self.assertEqual(observations[0].text, "Start")
                self.assertIsNone(observations[0].corners)

    def test_corner_restore_accounts_for_scale_and_quiet_border(self):
        original = ((10, 20), (30, 20), (30, 40), (10, 40))
        transformed = tuple((x * 4 + 64, y * 4 + 64) for x, y in original)
        self.assertEqual(validated_qr_corners(
            transformed, image_shape=(100, 100, 3), scale=4, border_px=64,
        ), original)
        self.assertIsNone(validated_qr_corners(
            transformed, image_shape=(30, 100, 3), scale=4, border_px=64,
        ))

    def test_optional_corner_diagnostics_distinguish_extent_from_restore_failure(self):
        for points, scale, border, expected in (
            (None, 1, 0, "missing"),
            (((0, 0), (99, 0), (99, 99), (0, 99)), 1, 0, "full_input_extent"),
            (((0, 0), (499, 0), (499, 499), (0, 499)), 4, 50, "out_of_bounds"),
            (((10, 20), (30, 20), (30, 40), (10, 40)), 1, 0, "valid"),
        ):
            with self.subTest(expected=expected):
                diagnostics = {}
                without = validated_qr_corners(points, image_shape=(100, 100), scale=scale, border_px=border)
                with_trace = validated_qr_corners(points, image_shape=(100, 100), scale=scale,
                                                  border_px=border, diagnostics=diagnostics)
                self.assertEqual(without, with_trace)
                self.assertEqual(diagnostics["reason"], expected)
                if points is not None:
                    self.assertEqual(len(diagnostics["raw_bounds"]), 4)
                    self.assertEqual(len(diagnostics["normalized_bounds"]), 4)

    def test_multiple_same_identity_symbols_remain_distinct(self):
        first = ((2, 2), (12, 2), (12, 12), (2, 12))
        second = tuple((u + 30, v) for u, v in first)
        cv2 = FakeCv2(FakeDetector((True, ("Start", "Start"), (first, second)), ("", None)))
        observations = detect_qr_observations_bgr(object(), cv2)
        self.assertEqual(tuple(item.text for item in observations), ("Start", "Start"))
        self.assertNotEqual(observations[0].corners, observations[1].corners)

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

    def test_conflicting_payloads_after_text_only_wechat_remain_ambiguous(self):
        cv2 = FakeCv2(
            FakeDetector((False, (), None, None), ("QR_004", None, None)),
            FakeWeChatDetector((("QR_005",), None)),
        )

        self.assertEqual(detect_qr_texts_bgr(object(), cv2), ("QR_005", "QR_004"))

    def test_recorded_wechat_crop_bounds_cannot_supply_symbol_geometry(self):
        bounds = ((0., 0.), (498., 0.), (498., 497.), (0., 497.))
        cv2 = FakeCv2(RaisingDetector(), FakeWeChatDetector((("Start",), (bounds,))))
        observations = detect_qr_observations_bgr(SimpleNamespace(shape=(498, 499, 3)), cv2)
        self.assertEqual(observations[0].text, "Start")
        self.assertIsNone(observations[0].corners)

    def test_placeholder_quad_requires_independently_decoded_matching_native_quad(self):
        bounds = ((0., 0.), (99., 0.), (99., 99.), (0., 99.))
        symbol = ((40., 40.), (60., 40.), (60., 60.), (40., 60.))
        for native_text in ("Start", "Neighbor"):
            cv2 = FakeCv2(
                FakeDetector((True, (native_text,), (symbol,)), ("", None)),
                FakeWeChatDetector((("Start",), (bounds,))),
            )
            observations = detect_qr_observations_bgr(SimpleNamespace(shape=(100, 100, 3)), cv2)
            if native_text == "Start":
                self.assertEqual(len(observations), 1)
                self.assertEqual(observations[0].detector, "opencv_multi")
                self.assertEqual(observations[0].corners, symbol)
            else:
                self.assertEqual(tuple(x.text for x in observations), ("Start", "Neighbor"))

    def test_opencv_decoder_errors_fall_back_to_wechat_detector(self):
        cv2 = FakeCv2(
            RaisingDetector(),
            FakeWeChatDetector((("QR_006",), None)),
        )

        self.assertEqual(detect_qr_texts_bgr(object(), cv2), ("QR_006",))

    def test_opencv_decoder_errors_return_empty_without_wechat_detector(self):
        cv2 = FakeCv2(RaisingDetector())

        self.assertEqual(detect_qr_texts_bgr(object(), cv2), ())


if __name__ == "__main__":
    unittest.main()
