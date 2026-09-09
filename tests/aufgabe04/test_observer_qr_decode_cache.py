import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.observer.qr_decode_cache import RoiQrDecodeCache


ROI = (10, 20, 110, 120)
QUAD = ((1.0, 2.0), (7.0, 2.0), (7.0, 8.0), (1.0, 8.0))


class ObserverQrDecodeCacheTest(unittest.TestCase):
    def test_exact_crop_and_mode_decode_once_including_empty_and_unavailable(self):
        for observations in ((), None):
            with self.subTest(observations=observations):
                frame = object()
                decoder = Mock(return_value=observations)
                cache = RoiQrDecodeCache()

                first = cache.decode(roi=ROI, mode="full", frame=frame, decoder=decoder)
                second = cache.decode(roi=ROI, mode="full", frame=frame, decoder=decoder)

                decoder.assert_called_once_with(frame)
                self.assertIs(first.observations, observations)
                self.assertIs(second.observations, observations)
                self.assertFalse(first.cache_hit)
                self.assertTrue(second.cache_hit)

    def test_multisymbol_text_only_and_corner_provenance_remain_unchanged(self):
        observations = (
            DecodedQrObservation("neighbor", QUAD, "opencv_multi", 2.0),
            DecodedQrObservation("target", None, "wechat", 1.0),
            DecodedQrObservation("neighbor", QUAD, "opencv_single", 1.0),
        )
        decoder = Mock(return_value=observations)
        cache = RoiQrDecodeCache()

        for _ in range(2):
            result = cache.decode(roi=ROI, mode="full", frame=None, decoder=decoder)
            self.assertIs(result.observations, observations)
            self.assertIs(result.observations[0].corners, QUAD)
            self.assertEqual(tuple(item.text for item in result.observations),
                             ("neighbor", "target", "neighbor"))
        decoder.assert_called_once_with(None)

    def test_different_crop_or_mode_requires_its_own_decode(self):
        calls = (
            (ROI, "full", object()),
            ((11, 20, 111, 120), "full", object()),
            (ROI, "native", object()),
        )
        outputs = tuple((DecodedQrObservation(str(index), QUAD, mode, 1.0),)
                        for index, (_roi, mode, _frame) in enumerate(calls))
        decoder = Mock(side_effect=outputs)
        cache = RoiQrDecodeCache()

        for (roi, mode, frame), expected in zip(calls, outputs):
            result = cache.decode(roi=roi, mode=mode, frame=frame, decoder=decoder)
            self.assertFalse(result.cache_hit)
            self.assertIs(result.observations, expected)
        self.assertEqual([call.args[0] for call in decoder.call_args_list],
                         [frame for _roi, _mode, frame in calls])
        for (roi, mode, frame), expected in zip(calls, outputs):
            result = cache.decode(roi=roi, mode=mode, frame=frame, decoder=decoder)
            self.assertTrue(result.cache_hit)
            self.assertIs(result.observations, expected)
        self.assertEqual(decoder.call_count, 3)

    def test_new_image_cache_redecodes_identical_bounds(self):
        decoder = Mock(return_value=())
        for frame in (object(), object()):
            result = RoiQrDecodeCache().decode(
                roi=ROI, mode="full", frame=frame, decoder=decoder,
            )
            self.assertFalse(result.cache_hit)
        self.assertEqual(decoder.call_count, 2)

    def test_timing_metadata_separates_current_call_and_original_decoder_work(self):
        cache = RoiQrDecodeCache()
        decoder = Mock(return_value=())
        with patch(
            "scripts.aufgabe04.real_robot.observer.qr_decode_cache.perf_counter",
            side_effect=(1.0, 1.025, 2.0, 2.0001),
        ):
            first = cache.decode(roi=ROI, mode="native", frame=None, decoder=decoder)
            second = cache.decode(roi=ROI, mode="native", frame=None, decoder=decoder)

        self.assertAlmostEqual(first.elapsed_ms, 25.0)
        self.assertAlmostEqual(first.decoder_elapsed_ms, 25.0)
        self.assertAlmostEqual(second.elapsed_ms, 0.1)
        self.assertAlmostEqual(second.decoder_elapsed_ms, 25.0)
        metadata = second.metadata()
        self.assertEqual(metadata["roi"], list(ROI))
        self.assertEqual(metadata["mode"], "native")
        self.assertIs(metadata["cache_hit"], True)
        self.assertAlmostEqual(metadata["elapsed_ms"], 0.1)
        self.assertAlmostEqual(metadata["decoder_elapsed_ms"], 25.0)

    def test_capacity_does_not_suppress_uncached_decoder_work(self):
        cache = RoiQrDecodeCache()
        decoder = Mock(return_value=())
        for index in range(3):
            cache.decode(roi=(index, 0, index + 10, 10), mode="full",
                         frame=None, decoder=decoder)
        for _ in range(2):
            result = cache.decode(roi=(4, 0, 14, 10), mode="full",
                                  frame=None, decoder=decoder)
            self.assertFalse(result.cache_hit)
        retained = cache.decode(roi=(0, 0, 10, 10), mode="full",
                                frame=None, decoder=decoder)
        self.assertTrue(retained.cache_hit)
        self.assertEqual(decoder.call_count, 5)

    def test_decoder_exception_propagates_and_is_not_cached(self):
        cache = RoiQrDecodeCache()
        decoder = Mock(side_effect=(RuntimeError("decode failed"), ()))
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            cache.decode(roi=ROI, mode="full", frame=None, decoder=decoder)
        result = cache.decode(roi=ROI, mode="full", frame=None, decoder=decoder)
        self.assertEqual(result.observations, ())
        self.assertFalse(result.cache_hit)
        self.assertEqual(decoder.call_count, 2)

    def test_nonexact_or_invalid_key_is_rejected_before_decoding(self):
        decoder = Mock()
        for roi, mode in (
            ((10.1, 20, 110, 120), "full"),
            ((True, 20, 110, 120), "full"),
            ((10, 20, 10, 120), "full"),
            ((-1, 20, 110, 120), "full"),
            (ROI, "other"),
        ):
            with self.subTest(roi=roi, mode=mode), self.assertRaises(ValueError):
                RoiQrDecodeCache().decode(roi=roi, mode=mode, frame=None, decoder=decoder)
        decoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
