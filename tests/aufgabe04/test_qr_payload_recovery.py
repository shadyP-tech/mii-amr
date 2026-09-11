"""Current-symbol recovery, ambiguity preservation, and recorded QR regressions."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.camera_calibration import rectify_bgr_frame
from scripts.aufgabe04.qr_scanning.isolated_qr_identity import decode_isolated_native_quad
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import (
    detect_qr_observations_bgr, _qr_decode_candidates_with_geometry,
)
from scripts.aufgabe04.qr_scanning.qr_observation import validated_qr_corners


FIXTURES = Path(__file__).parent / "fixtures/qr_recovery_20260911"
QUAD = ((20., 20.), (80., 20.), (80., 80.), (20., 80.))


@unittest.skipIf(cv2 is None, "OpenCV unavailable")
class QrPayloadRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((120, 120, 3), 60, np.uint8)

    def backend(self, *, isolated_texts=("Target",), full_text="Target", single=False):
        counts = {"native": 0, "wechat": 0, "isolated_shapes": []}

        class Native:
            def detectAndDecodeMulti(self, frame):
                return (False, (), None if single else np.array([QUAD]), ())

            def detectAndDecode(self, frame):
                return "", np.array([QUAD]), None

        class Wechat:
            def detectAndDecode(self, frame):
                h, w = frame.shape[:2]
                bounds = np.array([((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1))])
                if (h, w) == (120, 120):
                    return (full_text,), bounds
                counts["isolated_shapes"].append((h, w))
                return (() if h == 256 else isolated_texts), bounds

        class Cv2:
            def QRCodeDetector(self):
                counts["native"] += 1
                return Native()

            def wechat_qrcode_WeChatQRCode(self):
                counts["wechat"] += 1
                return Wechat()

            def __getattr__(self, name):
                return getattr(cv2, name)

        return Cv2(), counts

    def decode_one_variant(self, backend, diagnostics):
        with patch("scripts.aufgabe04.qr_scanning.opencv_qr_detector._qr_decode_candidates_with_geometry",
                   return_value=((self.frame, 1., 0),)):
            return detect_qr_observations_bgr(self.frame, backend, diagnostics=diagnostics)

    def test_margin_recovers_same_payload_with_original_native_quad(self):
        backend, counts = self.backend()
        trace = {}
        observations = self.decode_one_variant(backend, trace)
        self.assertEqual(tuple(o.text for o in observations), ("Target",))
        self.assertEqual(observations[0].corners, QUAD)
        self.assertEqual(counts["isolated_shapes"], [(256, 256), (204, 204)])
        self.assertEqual((counts["native"], counts["wechat"]), (1, 1))
        self.assertEqual(trace["events"][0]["corner_validation"][0]["reason"], "full_input_extent")
        isolated = [event for event in trace["events"] if "isolated" in event]
        self.assertEqual(isolated[-1]["isolated"]["selected_view"], "source_quiet_margin")

    def test_isolated_neighbor_payload_cannot_acquire_full_crop_identity(self):
        backend, _ = self.backend(isolated_texts=("Neighbor",))
        observations = self.decode_one_variant(backend, {})
        self.assertEqual(tuple(o.text for o in observations), ("Target", "Neighbor"))
        self.assertIsNone(observations[0].corners)
        self.assertEqual(observations[1].corners, QUAD)

    def test_single_only_quad_is_recovered_after_regular_variants_exhausted(self):
        backend, counts = self.backend(single=True)
        trace = {}
        observations = self.decode_one_variant(backend, trace)
        self.assertEqual(tuple(o.text for o in observations), ("Target",))
        self.assertEqual(observations[0].corners, QUAD)
        self.assertEqual(counts["isolated_shapes"], [(256, 256), (204, 204)])
        self.assertEqual(trace["events"][-1]["stage"], "opencv_single_isolated_deferred")

    def test_single_isolation_does_not_delay_later_successful_multi_variant(self):
        backend, counts = self.backend(single=True)
        native = backend.QRCodeDetector()
        multi_results = iter(((False, (), None, ()), (True, ("Target",), np.array([QUAD]), ())))
        native.detectAndDecodeMulti = lambda _frame: next(multi_results)
        backend.QRCodeDetector = lambda: native
        trace = {}
        with patch("scripts.aufgabe04.qr_scanning.opencv_qr_detector._qr_decode_candidates_with_geometry",
                   return_value=((self.frame, 1., 0), (self.frame.copy(), 1., 0))):
            observations = detect_qr_observations_bgr(self.frame, backend, diagnostics=trace)
        self.assertEqual(observations[0].corners, QUAD)
        self.assertEqual(counts["isolated_shapes"], [])
        self.assertFalse(any(e["stage"] == "opencv_single_isolated_deferred" for e in trace["events"]))

    def test_deferred_single_recovery_retains_at_most_four_current_variants(self):
        backend, counts = self.backend(single=True, isolated_texts=())
        with patch("scripts.aufgabe04.qr_scanning.opencv_qr_detector._qr_decode_candidates_with_geometry",
                   return_value=tuple((self.frame.copy(), 1., 0) for _ in range(8))):
            observations = detect_qr_observations_bgr(self.frame, backend)
        self.assertEqual(tuple(o.text for o in observations), ("Target",))
        self.assertIsNone(observations[0].corners)
        self.assertEqual(len(counts["isolated_shapes"]), 8)

    def test_undecoded_native_multiplicity_blocks_single_recovery_across_variants(self):
        neighbor = ((85., 20.), (115., 20.), (115., 50.), (85., 50.))
        multiple = (False, (), np.array([QUAD, neighbor]), ())
        none = (False, (), None, ())
        unique = (True, ("Target",), np.array([QUAD]), ())
        # Both queued-single-before-multi2 and multi2-before-single must keep
        # this ROI unbound, including a later native unique decoded payload.
        for results in ((multiple,), (none, multiple), (multiple, none, unique)):
            with self.subTest(variants=len(results)):
                backend, counts = self.backend(single=True)
                native = backend.QRCodeDetector()
                multi_results = iter(results)
                native.detectAndDecodeMulti = lambda _frame: next(multi_results)
                backend.QRCodeDetector = lambda: native
                trace = {}
                with patch("scripts.aufgabe04.qr_scanning.opencv_qr_detector._qr_decode_candidates_with_geometry",
                           return_value=tuple((self.frame.copy(), 1., 0) for _ in results)):
                    observations = detect_qr_observations_bgr(self.frame, backend, diagnostics=trace)
                self.assertEqual(tuple(o.text for o in observations), ("Target",))
                self.assertIsNone(observations[0].corners)
                self.assertEqual(counts["isolated_shapes"], [])
                self.assertEqual(trace["result"]["native_symbol_count"], 2)
                self.assertEqual(trace["result"]["geometry_suppressed_reason"], "multiple_native_quads")

    def test_native_quad_count_requires_individually_valid_original_crop_geometry(self):
        from scripts.aufgabe04.qr_scanning.qr_decoder_runtime import QrDecoderRuntime
        runtime = QrDecoderRuntime(cv2)
        outside = ((-2., 10.), (10., 10.), (10., 20.), (-2., 20.))
        malformed = ((1., 1.), (1., 1.), (1., 1.), (1., 1.))
        runtime.observe_native_multi((QUAD, outside, malformed), image_shape=self.frame.shape,
                                     scale=1., border_px=0)
        self.assertEqual(runtime.native_symbol_count, 1)

    def test_multiple_isolated_payloads_preserve_conflict_including_same_text(self):
        for texts in (("Target", "Neighbor"), ("Target", "Target")):
            with self.subTest(texts=texts):
                backend, _ = self.backend(isolated_texts=texts)
                observations = self.decode_one_variant(backend, {})
                self.assertEqual(tuple(o.text for o in observations), texts)
                self.assertTrue(all(o.corners is None for o in observations))

    def test_failed_isolation_is_bounded_and_not_repeated_for_identical_single_quad(self):
        backend, counts = self.backend(isolated_texts=())
        trace = {}
        observations = self.decode_one_variant(backend, trace)
        self.assertEqual(tuple(o.text for o in observations), ("Target",))
        self.assertIsNone(observations[0].corners)
        self.assertEqual(counts["isolated_shapes"], [(256, 256), (204, 204)])
        self.assertEqual(trace["events"][-1]["stage"], "opencv_single")

    def test_instances_are_reused_across_variants_but_never_across_images(self):
        backend, counts = self.backend(isolated_texts=())
        with patch("scripts.aufgabe04.qr_scanning.opencv_qr_detector._qr_decode_candidates_with_geometry",
                   side_effect=lambda *_: iter(((self.frame, 1., 0),) * 4)):
            for _ in range(2):
                trace = {}
                detect_qr_observations_bgr(self.frame, backend, diagnostics=trace)
                self.assertEqual(trace["decoder_instances"], {"wechat": 1, "native": 1})
                self.assertLessEqual(len(trace["events"]), 32)
        self.assertEqual((counts["native"], counts["wechat"]), (2, 2))
        self.assertEqual(len(counts["isolated_shapes"]), 16)

    def test_invalid_or_multiple_native_quads_never_run_isolated_decode(self):
        backend, counts = self.backend()
        for points in (None, ((1, 1), (1, 1), (2, 2), (3, 3)),
                       ((-1, 20), (80, 20), (80, 80), (-1, 80)), (QUAD, QUAD)):
            with self.subTest(points=points):
                result = decode_isolated_native_quad(self.frame, points, backend,
                    image_shape=self.frame.shape, scale=1., border_px=0)
                self.assertIsNone(result)
        self.assertEqual(counts["wechat"], 0)

    def test_recorded_quads_keep_exact_original_coordinates_after_margin_recovery(self):
        """Use recorded native output; deployed real decoding is checked separately."""
        fixture = json.loads((FIXTURES / "inputs.json").read_text())
        for entry in fixture["frames"]:
            raw = (FIXTURES / entry["image_file"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["image_sha256"])
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            rect = rectify_bgr_frame(image, SimpleNamespace(**entry["camera_info"]), cv2, np)
            for roi_entry in entry["rois"]:
                roi = roi_entry["roi"]
                crop = rect[roi["y0"]:roi["y1"], roi["x0"]:roi["x1"]]
                variant = roi_entry["native_variants"][1]
                candidate, scale, border = list(_qr_decode_candidates_with_geometry(crop, cv2))[1]
                self.assertEqual((scale, border), (variant["scale"], variant["border_px"]))
                class RecordedFailureDecoder:
                    def detectAndDecode(self, isolated):
                        return (() if isolated.shape[0] == 256 else ("QR_003",)), None
                result = decode_isolated_native_quad(candidate, variant["multi_points"], cv2,
                    image_shape=crop.shape, scale=scale, border_px=border,
                    wechat_decoder=RecordedFailureDecoder())
                self.assertEqual(result.text, "QR_003")
                self.assertEqual(result.corners, validated_qr_corners(variant["multi_points"],
                    image_shape=crop.shape, scale=scale, border_px=border))

    @unittest.skipUnless(cv2 is not None and hasattr(cv2, "wechat_qrcode_WeChatQRCode"),
                         "Real deployed WeChat backend required")
    def test_recorded_original_pixels_recover_real_payload_with_real_backends(self):
        fixture = json.loads((FIXTURES / "inputs.json").read_text())
        for entry in fixture["frames"]:
            image = cv2.imread(str(FIXTURES / entry["image_file"]))
            rect = rectify_bgr_frame(image, SimpleNamespace(**entry["camera_info"]), cv2, np)
            for roi_entry in entry["rois"]:
                roi = roi_entry["roi"]
                crop = rect[roi["y0"]:roi["y1"], roi["x0"]:roi["x1"]]
                observations = detect_qr_observations_bgr(crop, cv2)
                with self.subTest(frame=entry["frame_index"], source=roi_entry["source"]):
                    self.assertEqual(tuple(o.text for o in observations), ("QR_003",))
                    self.assertIsNotNone(observations[0].corners)


if __name__ == "__main__":
    unittest.main()
