"""ROS-free executing-buffer tests; they do not simulate DDS transport."""

import json
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.real_robot.observer.tf_delivery_trace import (
    ObserverTfDeliveryTrace, create_observer_traced_buffer, traced_observer_lookup,
)


def transform(parent="map", child="odom", stamp=100.):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=parent, stamp=SimpleNamespace(sec=stamp, nanosec=0)),
        child_frame_id=child,
    )


class ObserverTfDeliveryTraceTests(unittest.TestCase):
    def setUp(self):
        self.ros_sec, self.monotonic_sec = 100.2, 10.
        self.trace = ObserverTfDeliveryTrace(
            ros_now=lambda: self.ros_sec, monotonic=lambda: self.monotonic_sec,
            max_edges=2, history_limit=2, lookup_history_limit=2,
        )
        self.failure = None
        self.result = object()
        self.during_ingestion = []
        case = self

        class FakeBuffer:
            def __init__(self, **kwargs):
                self.kwargs, self.dynamic, self.static = kwargs, [], []

            def set_transform(self, value, authority):
                case.during_ingestion.append(case.trace.snapshot())
                self.dynamic.append((value, authority))
                case.monotonic_sec += .01
                if case.failure is not None:
                    raise case.failure
                return case.result

            def set_transform_static(self, value, authority):
                self.static.append((value, authority))
                if case.failure is not None:
                    raise case.failure
                return case.result

        self.buffer = create_observer_traced_buffer(FakeBuffer, trace=self.trace, test_option=3)

    def test_receipt_brackets_actual_ingestion_without_changing_call(self):
        value = transform(stamp=101.)  # Future-dated validity is legitimate diagnostic data.
        self.assertIs(self.buffer.set_transform(value, authority="authority"), self.result)
        self.assertEqual(self.buffer.kwargs, {"test_option": 3})
        self.assertEqual(self.buffer.dynamic, [(value, "authority")])
        entered = self.during_ingestion[0]["edges"][0]["last_receipts"][0]
        self.assertEqual(entered["ingestion_call_state"], "entered")
        receipt = self.trace.snapshot()["edges"][0]["last_receipts"][0]
        self.assertEqual(receipt["ingestion_call_state"], "returned")
        self.assertEqual(receipt["received_ros_sec"], 100.2)
        self.assertEqual(receipt["received_monotonic_sec"], 10.)
        self.assertAlmostEqual(receipt["finished_monotonic_sec"], 10.01)
        self.assertAlmostEqual(receipt["source_to_receipt_sec"], -.8)
        self.assertFalse(self.trace.snapshot()["insertion_acceptance_proven"])
        self.assertFalse(self.trace.snapshot()["motion_authorized"])

    def test_static_receipts_are_timeless_and_separate_from_dynamic_receipts(self):
        value = transform(stamp=0.)
        self.assertIs(self.buffer.set_transform_static(value, "static_authority"), self.result)
        self.assertIs(self.buffer.set_transform(value, "dynamic_authority"), self.result)
        self.monotonic_sec += .5
        static, dynamic = self.trace.snapshot()["edges"]
        self.assertTrue(static["static"])
        self.assertTrue(static["timeless"])
        self.assertIsNone(static["last_receipts"][0]["source_to_receipt_sec"])
        self.assertEqual(static["last_receipts"][0]["source_stamp_sec"], 0.)
        self.assertFalse(dynamic["timeless"])
        self.assertEqual(len(self.buffer.static), 1)
        self.assertEqual(len(self.buffer.dynamic), 1)
        self.assertGreater(static["last_receipt_age_sec"], .5)

    def test_core_exception_and_interrupt_identity_are_preserved(self):
        for error in (RuntimeError("core failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.failure = error
                with self.assertRaises(type(error)) as caught:
                    self.buffer.set_transform(transform(), "authority")
                self.assertIs(caught.exception, error)
        edge = self.trace.snapshot()["edges"][0]
        self.assertEqual(edge["ingestion_raised_count"], 2)
        self.assertEqual(edge["last_receipts"][-1]["exception_type"], "KeyboardInterrupt")

    def test_exact_lookup_request_outcome_and_return_are_preserved(self):
        request = dict(target_frame="map", source_frame="base_footprint",
                       query_kind="exact_sensor_time", query_stamp_sec=100.125)
        value = transform("map", "base_footprint", 100.125)
        self.assertIs(traced_observer_lookup(self.trace, request, lambda: value), value)
        error = LookupError("future extrapolation")

        def failed_lookup():
            raise error

        with self.assertRaises(LookupError) as caught:
            traced_observer_lookup(self.trace, request, failed_lookup)
        self.assertIs(caught.exception, error)
        first, second = self.trace.snapshot()["recent_lookups"]
        self.assertEqual(first["query_stamp_sec"], 100.125)
        self.assertEqual(first["returned_stamp_sec"], 100.125)
        self.assertEqual(first["requested_monotonic_sec"], 10.)
        self.assertEqual(first["outcome"], "returned")
        self.assertEqual(second["outcome"], "raised")
        self.assertEqual(second["exception_type"], "LookupError")
        self.assertEqual(second["receipt_count_at_completion"], 0)

    def test_histories_edges_text_and_snapshots_remain_bounded_and_detached(self):
        for index in range(12):
            self.buffer.set_transform(transform(stamp=100 + index), "authority")
            self.trace.lookup(dict(target_frame="map", source_frame="odom",
                                   query_kind="exact_sensor_time", query_stamp_sec=100 + index),
                              lambda: transform())
        snapshot = self.trace.snapshot()
        self.assertEqual(snapshot["counts"]["receipts"], 12)
        self.assertEqual(snapshot["edges"][0]["received_count"], 12)
        self.assertEqual(len(snapshot["edges"][0]["last_receipts"]), 2)
        self.assertEqual(len(snapshot["recent_lookups"]), 2)
        snapshot["edges"][0]["last_receipts"][0]["source_stamp_sec"] = -1
        self.assertEqual(self.trace.snapshot()["edges"][0]["last_receipts"][0]["source_stamp_sec"], 110.)
        for index in range(5):
            self.buffer.set_transform(transform(child=f"other_{index}"), "authority")
        self.assertEqual(len(self.trace.snapshot()["edges"]), 2)
        self.assertEqual(self.trace.snapshot()["counts"]["edge_evictions"], 4)
        self.buffer.set_transform(transform(parent="x" * 1024), "authority")
        self.assertEqual(self.trace.snapshot()["counts"]["ignored_receipts"], 1)

    def test_bad_diagnostic_data_or_clocks_cannot_prevent_real_ingestion(self):
        for stamp in (float("nan"), float("inf"), -float("inf")):
            self.assertIs(self.buffer.set_transform(transform(stamp=stamp), "authority"), self.result)
        self.ros_sec = float("nan")
        self.assertIs(self.buffer.set_transform(transform(), "authority"), self.result)
        self.ros_sec = 1e308
        self.assertIs(self.buffer.set_transform(transform(stamp=-1e308), "authority"), self.result)
        self.assertEqual(len(self.buffer.dynamic), 5)
        json.dumps(self.trace.snapshot(), allow_nan=False)

    def test_no_trace_or_invalid_lookup_metadata_never_changes_lookup_behavior(self):
        self.assertIs(traced_observer_lookup(None, {}, lambda: self.result), self.result)
        self.assertIs(traced_observer_lookup(self.trace, {}, lambda: self.result), self.result)
        self.assertEqual(self.trace.snapshot()["recent_lookups"], [])


if __name__ == "__main__":
    unittest.main()
