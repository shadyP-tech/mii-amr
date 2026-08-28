import inspect
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.localization.stationary_map_odom_capture import (  # noqa: E402
    StationaryMapOdomEpochBaseline,
    StationaryMapOdomEpochCapture,
    evaluate_stationary_map_odom_amcl_window_binding,
    evaluate_stationary_map_odom_candidate,
)


def sample(
    stamp_nanoseconds: object,
    receipt_time_nanoseconds: object,
    capture_time_nanoseconds: object,
    *,
    amcl_sample_index: int = 99,
    x_m: float = 1.0,
):
    return {
        "source": "direct_dynamic_tf",
        "stamp_nanoseconds": stamp_nanoseconds,
        "receipt_time_nanoseconds": receipt_time_nanoseconds,
        "capture_time_nanoseconds": capture_time_nanoseconds,
        "amcl_sample_index": amcl_sample_index,
        "x_m": x_m,
        "y_m": 2.0,
        "yaw_rad": 0.2,
    }


def capture(required_count: int = 3) -> StationaryMapOdomEpochCapture:
    return StationaryMapOdomEpochCapture(
        epoch_start_baseline=StationaryMapOdomEpochBaseline(
            stamp_nanoseconds=100,
            receipt_time_nanoseconds=200,
        ),
        required_count=required_count,
    )


class StationaryMapOdomCapturePolicyTest(unittest.TestCase):
    def test_policy_is_independent_of_amcl_callback_order(self):
        parameters = inspect.signature(
            evaluate_stationary_map_odom_candidate
        ).parameters
        self.assertFalse(any("amcl" in name for name in parameters))

        collector = capture(required_count=2)
        first = collector.consider(
            sample(101, 201, 301, amcl_sample_index=900)
        )
        second = collector.consider(
            sample(102, 202, 302, amcl_sample_index=0)
        )

        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertTrue(collector.window_result().complete)

    def test_epoch_baseline_rejects_cached_stamp_or_receipt(self):
        collector = capture(required_count=1)

        reused_stamp = collector.consider(sample(100, 201, 301))
        reused_receipt = collector.consider(sample(101, 200, 302))
        fresh = collector.consider(sample(101, 201, 303))

        self.assertFalse(reused_stamp.accepted)
        self.assertIn("stamp_not_after_epoch_start", reused_stamp.reasons)
        self.assertFalse(reused_receipt.accepted)
        self.assertIn("receipt_not_after_epoch_start", reused_receipt.reasons)
        self.assertTrue(fresh.accepted)
        self.assertEqual(len(collector.rejections), 2)

    def test_out_of_order_candidate_never_replaces_accepted_head(self):
        collector = capture(required_count=3)
        self.assertTrue(collector.consider(sample(110, 210, 310)).accepted)

        old_receipt = collector.consider(sample(111, 209, 311))
        old_stamp = collector.consider(sample(109, 211, 312))

        self.assertFalse(old_receipt.accepted)
        self.assertIn(
            "receipt_not_strictly_increasing",
            old_receipt.reasons,
        )
        self.assertFalse(old_stamp.accepted)
        self.assertIn("stamp_not_strictly_increasing", old_stamp.reasons)
        self.assertEqual(
            collector.accepted_head["stamp_nanoseconds"],
            110,
        )
        self.assertEqual(
            collector.accepted_head["receipt_time_nanoseconds"],
            210,
        )

        self.assertTrue(collector.consider(sample(112, 212, 313)).accepted)
        self.assertEqual(
            [entry["stamp_nanoseconds"] for entry in collector.retained_samples],
            [110, 112],
        )

    def test_newest_required_samples_are_retained_and_reindexed(self):
        collector = capture(required_count=3)
        for index in range(5):
            decision = collector.consider(
                sample(
                    101 + index,
                    201 + index,
                    301 + index,
                    amcl_sample_index=40 + index,
                    x_m=float(index),
                )
            )
            self.assertTrue(decision.accepted)

        result = collector.window_result()

        self.assertTrue(result.complete)
        self.assertEqual(result.retained_sample_count, 3)
        self.assertEqual(
            [entry["x_m"] for entry in result.samples],
            [2.0, 3.0, 4.0],
        )
        self.assertEqual(
            [entry["amcl_sample_index"] for entry in result.samples],
            [0, 1, 2],
        )
        self.assertEqual(
            [
                entry["stationary_epoch_sample_index"]
                for entry in result.samples
            ],
            [0, 1, 2],
        )
        self.assertEqual(
            [
                entry["amcl_sample_index"]
                for entry in result.retained_samples
            ],
            [42, 43, 44],
        )
        json.dumps(result.to_log_dict(), allow_nan=False)

    def test_direct_tf_window_may_lead_callbacks_but_not_amcl_window(self):
        samples = (
            sample(101, 201, 201),
            sample(102, 202, 202),
            sample(103, 203, 203),
        )

        accepted = evaluate_stationary_map_odom_amcl_window_binding(
            samples,
            amcl_receipt_nanoseconds=(202, 204, 206),
        )
        predates = evaluate_stationary_map_odom_amcl_window_binding(
            samples,
            amcl_receipt_nanoseconds=(204, 205, 206),
        )

        self.assertTrue(accepted.accepted)
        self.assertEqual(
            accepted.reason,
            "direct_tf_window_overlaps_amcl_window",
        )
        self.assertFalse(predates.accepted)
        self.assertIn(
            "direct_tf_window_predates_amcl_window",
            predates.reasons,
        )
        json.dumps(accepted.to_log_dict(), allow_nan=False)
        json.dumps(predates.to_log_dict(), allow_nan=False)

    def test_amcl_window_binding_rejects_malformed_or_nonmonotonic_amcl(self):
        samples = (sample(101, 201, 301), sample(102, 202, 302))

        malformed = evaluate_stationary_map_odom_amcl_window_binding(
            samples,
            amcl_receipt_nanoseconds=(201, True),
        )
        nonmonotonic = evaluate_stationary_map_odom_amcl_window_binding(
            samples,
            amcl_receipt_nanoseconds=(201, 201),
        )

        self.assertFalse(malformed.accepted)
        self.assertIn(
            "amcl_1_receipt_time_nanoseconds_not_nonnegative_integer",
            malformed.reasons,
        )
        self.assertFalse(nonmonotonic.accepted)
        self.assertIn(
            "amcl_receipts_not_strictly_increasing",
            nonmonotonic.reasons,
        )

    def test_amcl_window_binding_rejects_sample_count_mismatch(self):
        decision = evaluate_stationary_map_odom_amcl_window_binding(
            (sample(101, 201, 301), sample(102, 202, 302)),
            amcl_receipt_nanoseconds=(201,),
        )

        self.assertFalse(decision.accepted)
        self.assertIn(
            "stationary_window_sample_count_mismatch",
            decision.reasons,
        )

    def test_malformed_and_duplicate_candidates_are_audit_rejections(self):
        collector = capture(required_count=2)
        accepted = collector.consider(sample(101, 201, 301))
        self.assertTrue(accepted.accepted)

        malformed_cases = (
            {"stamp_nanoseconds": 102},
            sample(True, 202, 302),
            sample(102, -1, 302),
            sample(102.0, 202, 302),
            sample(101, 201, 301),
        )
        for candidate in malformed_cases:
            self.assertFalse(collector.consider(candidate).accepted)

        self.assertEqual(len(collector.rejections), len(malformed_cases))
        self.assertEqual(
            collector.accepted_head["capture_time_nanoseconds"],
            301,
        )
        self.assertTrue(collector.consider(sample(102, 202, 302)).accepted)
        self.assertTrue(collector.window_result().complete)
        json.dumps(
            [entry.to_log_dict() for entry in collector.rejections],
            allow_nan=False,
        )

    def test_incomplete_window_never_exposes_claimable_samples(self):
        collector = capture(required_count=3)
        collector.consider(sample(101, 201, 301))
        collector.consider(sample(102, 202, 302))

        result = collector.window_result()

        self.assertFalse(result.complete)
        self.assertEqual(result.retained_sample_count, 2)
        self.assertEqual(result.samples, ())
        self.assertEqual(len(result.retained_samples), 2)

    def test_baseline_and_configuration_are_strictly_validated(self):
        for invalid in (-1, 1.0, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    StationaryMapOdomEpochBaseline(
                        stamp_nanoseconds=invalid,
                        receipt_time_nanoseconds=0,
                    )
        with self.assertRaises(ValueError):
            capture(required_count=0)


if __name__ == "__main__":
    unittest.main()
