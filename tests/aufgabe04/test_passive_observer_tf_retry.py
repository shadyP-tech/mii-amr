import dataclasses
import unittest

from scripts.aufgabe04.real_robot.observer.tf_retry import (
    PassiveObserverTfRetryError,
    PassiveObserverTfRetryScheduler,
)


class PassiveObserverTfRetrySchedulerTests(unittest.TestCase):
    def test_future_extrapolation_retries_same_exact_frame(self) -> None:
        scheduler = PassiveObserverTfRetryScheduler[object]()
        frame = object()
        image_stamp = 1787663986.244712
        latest_tf_stamp = 1787663986.242854

        self.assertTrue(scheduler.offer(frame, stamp_sec=image_stamp))
        held = scheduler.pending_frame
        evidence = scheduler.mark_transform_unavailable(
            stamp_sec=image_stamp,
            observed_sec=1787663986.457761,
            reason=(
                "future extrapolation: latest TF trails image by "
                f"{image_stamp - latest_tf_stamp:.6f}s"
            ),
        )

        self.assertIs(held, scheduler.pending_frame)
        self.assertIs(frame, scheduler.pending_frame.frame)
        self.assertEqual(scheduler.pending_frame.stamp_sec, image_stamp)
        self.assertAlmostEqual(image_stamp - latest_tf_stamp, 0.001858)
        self.assertEqual(evidence.retry_count, 1)
        self.assertEqual(evidence.pending_stamp_sec, image_stamp)
        self.assertEqual(
            evidence.first_failure_time_sec,
            1787663986.457761,
        )
        self.assertEqual(
            evidence.last_failure_reason,
            evidence.first_failure_reason,
        )

        second = scheduler.mark_transform_unavailable(
            stamp_sec=image_stamp,
            observed_sec=1787663986.500000,
            reason="exact-time transform still unavailable",
        )
        self.assertEqual(second.retry_count, 2)
        self.assertEqual(second.first_failure_time_sec, 1787663986.457761)
        self.assertEqual(
            second.last_failure_reason,
            "exact-time transform still unavailable",
        )

    def test_newer_frame_does_not_replace_pending_frame(self) -> None:
        scheduler = PassiveObserverTfRetryScheduler[str]()
        self.assertTrue(scheduler.offer("original", stamp_sec=10.0))

        self.assertFalse(scheduler.offer("newer", stamp_sec=10.1))
        self.assertEqual(scheduler.pending_frame.frame, "original")
        self.assertEqual(scheduler.pending_frame.stamp_sec, 10.0)

    def test_transform_ready_frame_is_released_once(self) -> None:
        scheduler = PassiveObserverTfRetryScheduler[dict[str, int]]()
        frame = {"sequence": 7}
        scheduler.offer(frame, stamp_sec=20.0)
        scheduler.mark_transform_unavailable(
            stamp_sec=20.0,
            observed_sec=20.01,
            reason="TF listener is catching up",
        )

        scheduler.mark_transform_ready(stamp_sec=20.0)
        self.assertIsNone(scheduler.pending_frame)
        consumed = scheduler.consume(stamp_sec=20.0)

        self.assertIs(consumed.frame, frame)
        self.assertEqual(consumed.stamp_sec, 20.0)
        self.assertEqual(scheduler.evidence.state, "idle")
        self.assertEqual(scheduler.evidence.last_consumed_stamp_sec, 20.0)
        with self.assertRaisesRegex(
            PassiveObserverTfRetryError,
            "no stamped frame is pending",
        ):
            scheduler.consume(stamp_sec=20.0)

    def test_stale_frame_can_be_discarded_without_emission(self) -> None:
        scheduler = PassiveObserverTfRetryScheduler[str]()
        scheduler.offer("stale", stamp_sec=30.0)
        scheduler.mark_transform_unavailable(
            stamp_sec=30.0,
            observed_sec=31.0,
            reason="future extrapolation",
        )

        self.assertIsNone(
            scheduler.discard(
                stamp_sec=30.0,
                reason="sensor tuple exceeded maximum age",
            )
        )
        self.assertIsNone(scheduler.pending_frame)
        self.assertEqual(scheduler.evidence.last_discarded_stamp_sec, 30.0)
        self.assertEqual(
            scheduler.evidence.last_discard_reason,
            "sensor tuple exceeded maximum age",
        )
        self.assertTrue(scheduler.offer("fresh", stamp_sec=30.1))

    def test_duplicate_out_of_order_and_nonfinite_stamps_are_rejected(self) -> None:
        scheduler = PassiveObserverTfRetryScheduler[str]()
        scheduler.offer("pending", stamp_sec=40.0)

        for stamp in (40.0, 39.9):
            with self.subTest(stamp=stamp):
                with self.assertRaisesRegex(
                    PassiveObserverTfRetryError,
                    "strictly newer than the pending",
                ):
                    scheduler.offer("invalid", stamp_sec=stamp)
        for stamp in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(stamp=stamp):
                with self.assertRaisesRegex(
                    PassiveObserverTfRetryError,
                    "finite and positive",
                ):
                    scheduler.offer("invalid", stamp_sec=stamp)

        scheduler.discard(stamp_sec=40.0, reason="test terminal state")
        for stamp in (40.0, 39.9):
            with self.subTest(terminal_stamp=stamp):
                with self.assertRaisesRegex(
                    PassiveObserverTfRetryError,
                    "strictly newer than the last terminal",
                ):
                    scheduler.offer("invalid", stamp_sec=stamp)

    def test_mismatched_or_illegal_operations_fail_closed(self) -> None:
        scheduler = PassiveObserverTfRetryScheduler[str]()
        scheduler.offer("pending", stamp_sec=50.0)

        operations = (
            lambda: scheduler.mark_transform_unavailable(
                stamp_sec=50.1,
                observed_sec=50.2,
                reason="wrong tuple",
            ),
            lambda: scheduler.mark_transform_ready(stamp_sec=50.1),
            lambda: scheduler.consume(stamp_sec=50.0),
            lambda: scheduler.discard(stamp_sec=50.1, reason="wrong tuple"),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(PassiveObserverTfRetryError):
                    operation()

        scheduler.mark_transform_unavailable(
            stamp_sec=50.0,
            observed_sec=50.2,
            reason="first",
        )
        with self.assertRaisesRegex(
            PassiveObserverTfRetryError,
            "must not precede",
        ):
            scheduler.mark_transform_unavailable(
                stamp_sec=50.0,
                observed_sec=50.1,
                reason="clock moved backward",
            )
        scheduler.mark_transform_ready(stamp_sec=50.0)
        with self.assertRaises(PassiveObserverTfRetryError):
            scheduler.mark_transform_ready(stamp_sec=50.0)
        with self.assertRaises(PassiveObserverTfRetryError):
            scheduler.mark_transform_unavailable(
                stamp_sec=50.0,
                observed_sec=50.3,
                reason="ready frames are not retryable",
            )

    def test_evidence_is_frozen_and_excludes_frame_payload(self) -> None:
        scheduler = PassiveObserverTfRetryScheduler[object]()
        scheduler.offer(object(), stamp_sec=60.0)
        evidence = scheduler.evidence

        self.assertFalse(hasattr(evidence, "frame"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            evidence.retry_count = 99


if __name__ == "__main__":
    unittest.main()
