"""Cooperative QR work limits preserve positive evidence at stage boundaries."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr
from scripts.aufgabe04.qr_scanning.isolated_qr_identity import decode_isolated_native_quad
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation


MODULE = "scripts.aufgabe04.qr_scanning.opencv_qr_detector."


class QrDecoderBudgetTests(unittest.TestCase):
    def decode(self, stages, *, limit=.12):
        now, calls, variants = [10.], [], []
        frame = SimpleNamespace(shape=(100, 100, 3))
        def candidates(*_):
            for index in range(4):
                variants.append(index)
                yield frame, float(index + 1), 0
        def observations(*_, scale, **_kwargs):
            for index, (elapsed, value) in enumerate(stages):
                calls.append((scale, index))
                now[0] += elapsed
                yield value
        diagnostics = {}
        with patch(MODULE + "monotonic", side_effect=lambda: now[0]), \
             patch(MODULE + "_qr_decode_candidates_with_geometry", side_effect=candidates), \
             patch(MODULE + "_candidate_observations", side_effect=observations):
            result = detect_qr_observations_bgr(
                frame, SimpleNamespace(), diagnostics=diagnostics, max_elapsed_sec=limit,
            )
        return result, diagnostics, calls, variants

    def test_empty_recovery_stops_between_backends_without_expanded_pyramid(self):
        result, diagnostics, calls, variants = self.decode([(.08, ()), (.08, ()), (.08, ())])
        self.assertEqual(result, ())
        self.assertEqual(calls, [(1., 0), (1., 1)])
        self.assertEqual(variants, [0])
        self.assertTrue(diagnostics["processing_budget"]["exhausted"])

    def test_budget_retains_current_text_only_evidence(self):
        text = (DecodedQrObservation("QR_003", None, "test", 1.),)
        result, diagnostics, calls, _ = self.decode([(.04, text), (.10, ()), (.01, ())])
        self.assertIs(result, text)
        self.assertEqual(len(calls), 2)
        self.assertTrue(diagnostics["processing_budget"]["exhausted"])

    def test_conflicting_evidence_is_not_hidden_when_budget_expires(self):
        first = (DecodedQrObservation("QR_003", None, "test", 1.),)
        second = (DecodedQrObservation("QR_004", None, "test", 1.),)
        result, diagnostics, _, _ = self.decode([(.04, first), (.10, second)])
        self.assertEqual([item.text for item in result], ["QR_003", "QR_004"])
        self.assertTrue(diagnostics["processing_budget"]["exhausted"])

    def test_atomic_backend_overrun_is_reported_but_not_claimed_preempted(self):
        result, diagnostics, calls, variants = self.decode([(.4, ()), (.01, ())])
        self.assertEqual(result, ())
        self.assertEqual(calls, [(1., 0)])
        self.assertEqual(variants, [0])
        self.assertAlmostEqual(diagnostics["processing_budget"]["elapsed_sec"], .4)
        self.assertTrue(diagnostics["processing_budget"]["cooperative_between_backend_calls"])

    def test_default_decoder_still_searches_all_variants(self):
        _, diagnostics, calls, variants = self.decode([(.08, ()), (.08, ())], limit=None)
        self.assertEqual(len(calls), 8)
        self.assertEqual(variants, [0, 1, 2, 3])
        self.assertNotIn("processing_budget", diagnostics)

    def test_invalid_budget_rejected_before_backend_work(self):
        for value in (0., -1., float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                detect_qr_observations_bgr(object(), object(), max_elapsed_sec=value)

    def test_slow_backend_exception_cannot_start_next_decoder(self):
        now, calls = [10.], []
        def failed_decode(_frame):
            calls.append("wechat")
            now[0] += .4
            raise RuntimeError("backend failure")
        backend = SimpleNamespace(
            wechat_qrcode_WeChatQRCode=lambda: SimpleNamespace(detectAndDecode=failed_decode),
            QRCodeDetector=lambda: calls.append("native"),
        )
        frame, diagnostics = SimpleNamespace(shape=(100, 100, 3)), {}
        with patch(MODULE + "monotonic", side_effect=lambda: now[0]), \
             patch(MODULE + "_qr_decode_candidates_with_geometry", return_value=((frame, 1., 0),)):
            result = detect_qr_observations_bgr(frame, backend, diagnostics=diagnostics,
                                               max_elapsed_sec=.12)
        self.assertEqual(result, ())
        self.assertEqual(calls, ["wechat"])
        self.assertTrue(diagnostics["processing_budget"]["exhausted"])

    def test_isolated_recovery_does_not_start_second_view_after_expiry(self):
        now, calls = [0.], []
        frame = SimpleNamespace(shape=(100, 100, 3))
        def decode(_frame):
            calls.append("decode")
            now[0] += .13
            return (), None
        diagnostics = {}
        with patch("scripts.aufgabe04.qr_scanning.isolated_qr_identity.rectify_isolated_qr_view",
                   return_value=frame):
            result = decode_isolated_native_quad(
                frame, ((20., 20.), (80., 20.), (80., 80.), (20., 80.)),
                SimpleNamespace(), image_shape=frame.shape, scale=1., border_px=0,
                wechat_decoder=SimpleNamespace(detectAndDecode=decode), diagnostics=diagnostics,
                budget_exhausted=lambda: now[0] >= .12,
            )
        self.assertIsNone(result)
        self.assertEqual(calls, ["decode"])
        self.assertEqual(diagnostics["reason"], "processing_budget_exhausted")


if __name__ == "__main__":
    unittest.main()


def test_preferred_scale_has_a_pixel_cap_and_obeys_work_denial():
    from unittest.mock import Mock
    from scripts.aufgabe04.qr_scanning.opencv_qr_detector import _qr_decode_candidates_with_geometry
    small, large = SimpleNamespace(shape=(200, 200, 3)), SimpleNamespace(shape=(600, 800, 3))
    resized = SimpleNamespace(shape=(800, 800, 3))
    with patch(MODULE + '_resize_for_qr', return_value=resized) as resize:
        assert next(_qr_decode_candidates_with_geometry(small, object(), preferred_scale=4)) == (resized, 4., 0)
        resize.assert_called_once()
        resize.reset_mock()
        assert next(_qr_decode_candidates_with_geometry(large, object(), preferred_scale=4)) == (large, 1., 0)
        denied = SimpleNamespace(allow=Mock(return_value=False))
        assert next(_qr_decode_candidates_with_geometry(small, object(), preferred_scale=4, work_budget=denied)) == (small, 1., 0)
        resize.assert_not_called()
