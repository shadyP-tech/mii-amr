"""Tracking QR refreshes must not silently invoke acquisition decoders."""

from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import numpy
except ImportError:  # pragma: no cover
    numpy = None

from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import detect_qr_quad
from scripts.aufgabe04.qr_scanning.native_qr_observations import (
    detect_native_qr_observations_bgr,
)
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation


QR_QUAD = ((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0))
GENERIC_DECODER = (
    "scripts.aufgabe04.perception.stand_axis.qr_pose_seed."
    "detect_qr_observations_bgr"
)


class NativeDetector:
    def __init__(self, *, decoded_points=None):
        self.decoded_points = decoded_points
        self.calls = []

    def detectMulti(self, frame):
        self.calls.append(("detectMulti", frame.shape))
        return False, None

    def detect(self, frame):
        self.calls.append(("detect", frame.shape))
        return False, None

    def detectAndDecodeMulti(self, frame):
        self.calls.append(("detectAndDecodeMulti", frame.shape))
        if self.decoded_points is None:
            return False, (), None, ()
        return True, ("Start",), self.decoded_points, ()


class NativeCv2:
    INTER_CUBIC = 2

    def __init__(self, detector):
        self.detector = detector
        self.resize_scales = []

    def QRCodeDetector(self):
        return self.detector

    def resize(self, frame, dsize, *, fx, fy, interpolation):
        self.resize_scales.append((fx, fy))
        return numpy.zeros(
            (round(frame.shape[0] * fy), round(frame.shape[1] * fx), 3),
            dtype=numpy.uint8,
        )


class IdentityDecoder:
    def __init__(self, multi_result, single_result):
        self.multi_result = multi_result
        self.single_result = single_result
        self.calls = []

    def detectAndDecodeMulti(self, frame):
        self.calls.append(("multi", frame))
        if isinstance(self.multi_result, Exception):
            raise self.multi_result
        return self.multi_result

    def detectAndDecode(self, frame):
        self.calls.append(("single", frame))
        if isinstance(self.single_result, Exception):
            raise self.single_result
        return self.single_result


@unittest.skipIf(numpy is None, "NumPy unavailable")
class MetricQrDetectionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.frame = numpy.zeros((80, 80, 3), dtype=numpy.uint8)

    def test_native_tracking_miss_never_invokes_generic_decoder(self):
        detector = NativeDetector()
        cv2 = NativeCv2(detector)
        with patch(GENERIC_DECODER, side_effect=AssertionError("unexpected acquisition")):
            result = detect_qr_quad(
                cv2, self.frame, scales=(1.0,), allow_decode_fallback=False,
            )

        self.assertIsNone(result)
        self.assertEqual([name for name, _shape in detector.calls], [
            "detectMulti", "detect", "detectAndDecodeMulti",
        ])
        self.assertEqual(cv2.resize_scales, [])

    def test_acquisition_keeps_full_pyramid_and_generic_fallback_by_default(self):
        detector = NativeDetector()
        cv2 = NativeCv2(detector)
        observation = DecodedQrObservation("Start", QR_QUAD, "wechat", 4.0)
        with patch(GENERIC_DECODER, return_value=(observation,)) as fallback:
            result = detect_qr_quad(cv2, self.frame)

        self.assertIsNotNone(result)
        self.assertEqual(result.text, "Start")
        self.assertEqual(result.detector, "wechat")
        self.assertEqual(result.scale, 4.0)
        self.assertEqual(result.corners, tuple(ImagePoint(u, v) for u, v in QR_QUAD))
        self.assertEqual(cv2.resize_scales, [(2.0, 2.0), (4.0, 4.0)])
        self.assertEqual(len(detector.calls), 9)
        fallback.assert_called_once_with(self.frame, cv2)

    def test_native_decode_multi_corner_fallback_remains_available_during_tracking(self):
        detector = NativeDetector(decoded_points=numpy.asarray([QR_QUAD]))
        cv2 = NativeCv2(detector)
        with patch(GENERIC_DECODER, side_effect=AssertionError("unexpected acquisition")):
            result = detect_qr_quad(
                cv2, self.frame, scales=(1.0,), allow_decode_fallback=False,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.detector, "opencv_native")
        self.assertEqual(result.corners, tuple(ImagePoint(u, v) for u, v in QR_QUAD))
        self.assertIsNone(result.text)
        self.assertEqual(detector.calls[-1][0], "detectAndDecodeMulti")

    def test_supplied_decoded_geometry_remains_authoritative_in_both_policies(self):
        observation = DecodedQrObservation("Start", QR_QUAD, "wechat", 2.0)
        for allow_fallback in (False, True):
            with self.subTest(allow_decode_fallback=allow_fallback):
                detector = NativeDetector()
                cv2 = NativeCv2(detector)
                with patch(GENERIC_DECODER, side_effect=AssertionError("duplicate decoding")):
                    result = detect_qr_quad(
                        cv2, self.frame, decoded_observations=(observation,),
                        allow_decode_fallback=allow_fallback,
                    )

                self.assertIsNotNone(result)
                self.assertEqual(result.text, observation.text)
                self.assertEqual(result.detector, observation.detector)
                self.assertEqual(result.scale, observation.scale)
                self.assertEqual(result.corners, tuple(ImagePoint(u, v) for u, v in QR_QUAD))
                self.assertEqual(detector.calls, [])

    def test_ambiguous_or_text_only_observations_never_borrow_native_corners(self):
        first = DecodedQrObservation("Start", QR_QUAD, "wechat")
        second_quad = tuple((u + 35.0, v) for u, v in QR_QUAD)
        cases = (
            (DecodedQrObservation("Start", None, "wechat"),),
            (first, DecodedQrObservation("Start", second_quad, "wechat")),
            (first, DecodedQrObservation("Goal", second_quad, "wechat")),
        )
        for allow_fallback in (False, True):
            for observations in cases:
                with self.subTest(observations=observations, allow_fallback=allow_fallback):
                    detector = NativeDetector(decoded_points=numpy.asarray([QR_QUAD]))
                    cv2 = NativeCv2(detector)
                    with patch(GENERIC_DECODER, side_effect=AssertionError("duplicate decoding")):
                        result = detect_qr_quad(
                            cv2, self.frame, decoded_observations=observations,
                            allow_decode_fallback=allow_fallback,
                        )

                    self.assertIsNone(result)
                    self.assertEqual(detector.calls, [])

    def test_explicit_empty_observations_do_not_repeat_generic_decoding(self):
        detector = NativeDetector()
        cv2 = NativeCv2(detector)
        with patch(GENERIC_DECODER, side_effect=AssertionError("duplicate decoding")):
            result = detect_qr_quad(
                cv2, self.frame, scales=(1.0,), decoded_observations=(),
                allow_decode_fallback=True,
            )

        self.assertIsNone(result)
        self.assertEqual(len(detector.calls), 3)

    def test_native_identity_miss_uses_only_current_frame_and_native_decoders(self):
        detector = IdentityDecoder((False, (), None), ("", None))
        cv2 = NativeCv2(detector)
        with patch(GENERIC_DECODER, side_effect=AssertionError("unexpected acquisition")):
            result = detect_native_qr_observations_bgr(self.frame, cv2)

        self.assertEqual(result, ())
        self.assertEqual([method for method, _frame in detector.calls], ["multi", "single"])
        self.assertTrue(all(frame is self.frame for _method, frame in detector.calls))
        self.assertEqual(cv2.resize_scales, [])

    def test_native_identity_preserves_own_multi_symbol_geometry_and_ambiguity(self):
        second_quad = tuple((u + 35.0, v) for u, v in QR_QUAD)
        for second_text in ("Start", "Goal"):
            with self.subTest(second_text=second_text):
                detector = IdentityDecoder(
                    (True, ("Start", second_text), numpy.asarray([QR_QUAD, second_quad])),
                    AssertionError("multi-symbol evidence must not become one identity"),
                )
                result = detect_native_qr_observations_bgr(self.frame, NativeCv2(detector))

                self.assertEqual(tuple(item.text for item in result), ("Start", second_text))
                self.assertEqual(tuple(item.corners for item in result), (QR_QUAD, second_quad))
                self.assertEqual(tuple(item.detector for item in result), ("opencv_multi",) * 2)
                self.assertEqual(len(detector.calls), 1)

    def test_native_identity_single_fallback_keeps_current_frame_coordinates(self):
        detector = IdentityDecoder(
            RuntimeError("native multi unavailable"),
            ("Start", numpy.asarray([QR_QUAD])),
        )

        result = detect_native_qr_observations_bgr(self.frame, NativeCv2(detector))

        self.assertEqual(result, (DecodedQrObservation("Start", QR_QUAD, "opencv_single"),))
        self.assertTrue(all(frame is self.frame for _method, frame in detector.calls))

    def test_native_identity_text_only_results_remain_provisional_or_conflicting(self):
        for single_result, expected in (
            (("", None), (DecodedQrObservation("Start", None, "opencv_multi"),)),
            (("Start", numpy.asarray([QR_QUAD])),
             (DecodedQrObservation("Start", QR_QUAD, "opencv_single"),)),
            (("Goal", numpy.asarray([QR_QUAD])),
             (DecodedQrObservation("Start", None, "opencv_multi"),
              DecodedQrObservation("Goal", QR_QUAD, "opencv_single"))),
        ):
            with self.subTest(expected=expected):
                detector = IdentityDecoder((True, ("Start",), None), single_result)

                result = detect_native_qr_observations_bgr(self.frame, NativeCv2(detector))

                self.assertEqual(result, expected)

    def test_native_identity_never_reuses_previous_frame_payload_after_a_miss(self):
        detector = IdentityDecoder((True, ("Start",), numpy.asarray([QR_QUAD])), ("", None))
        cv2 = NativeCv2(detector)
        first = detect_native_qr_observations_bgr(self.frame, cv2)
        detector.multi_result = (False, (), None)

        second = detect_native_qr_observations_bgr(numpy.zeros_like(self.frame), cv2)

        self.assertEqual(first[0].text, "Start")
        self.assertEqual(second, ())


if __name__ == "__main__":
    unittest.main()
