"""Offline ingestion tests; fake TF core does not validate ROS/DDS delivery."""

from types import SimpleNamespace
import json
import unittest

from scripts.aufgabe04.navigation.waypoint_follower.tf_receipts import create_receipt_traced_buffer


def transform(target="map", source="odom", *, stamp=100):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=target, stamp=SimpleNamespace(sec=stamp, nanosec=0)),
        child_frame_id=source,
    )


class FollowerTfReceiptsTest(unittest.TestCase):
    def setUp(self):
        self.ros_sec, self.monotonic_sec = 101.0, 20.0
        self.failure, self.reject_insertion = None, False
        self.snapshots_during_core_calls = []
        self.result = object()
        case = self

        class Core:
            def __init__(self, *, node):
                self.latest, self.calls, self.static_calls = None, [], []

            def set_transform(self, value, authority):
                case.snapshots_during_core_calls.append(self.tf_receipt_snapshot())
                self.calls.append((value, authority))
                case.monotonic_sec += 0.01
                if case.failure is not None:
                    raise case.failure
                if not case.reject_insertion:
                    self.latest = value
                return case.result

            def set_transform_static(self, value, authority):
                self.static_calls.append((value, authority))

            def lookup_transform_core(self, target, source, at_time):
                case.snapshots_during_core_calls.append(self.tf_receipt_snapshot())
                case.monotonic_sec += 0.02
                if self.latest is None:
                    raise LookupError("buffer has no sample")
                return self.latest

        self.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=int(self.ros_sec * 1e9))))
        self.buffer = create_receipt_traced_buffer(
            Core, node=self.node, edges={"execution_pose": ("odom", "base_footprint"),
                                        "global_consistency": ("map", "odom")},
            latest_time_factory=lambda: 0, monotonic=lambda: self.monotonic_sec,
        )

    def edge(self, role="global_consistency"):
        return self.buffer.tf_receipt_snapshot()["edges"][role]

    def test_receipts_bracket_same_buffer_ingestion_and_preserve_return(self):
        value = transform()
        self.assertIs(self.buffer.set_transform(value, "authority"), self.result)
        self.assertEqual(self.buffer.calls, [(value, "authority")])
        receipt = self.edge()["last_receipts"][0]
        self.assertEqual(receipt["received_ros_sec"], 101.0)
        self.assertEqual(receipt["received_monotonic_sec"], 20.0)
        self.assertAlmostEqual(receipt["ingestion_finished_monotonic_sec"], 20.01)
        self.assertEqual(receipt["header_stamp_sec"], 100.0)
        self.assertEqual(receipt["age_at_receipt_sec"], 1.0)
        self.assertEqual(receipt["newest_buffer_stamp_sec"], 100.0)
        snapshot = self.buffer.tf_receipt_snapshot()
        self.assertEqual(snapshot["created_monotonic_sec"], 20.0)
        self.assertEqual(snapshot["created_ros_sec"], 101.0)
        self.assertEqual(len(self.snapshots_during_core_calls), 2)
        self.assertAlmostEqual(self.monotonic_sec, 20.03)

    def test_counts_are_cumulative_histories_bounded_and_snapshots_detached(self):
        for index in range(12):
            self.buffer.set_transform(transform(stamp=100 + index), "authority")
        edge = self.edge()
        self.assertEqual(edge["received_count"], 12)
        self.assertEqual(edge["ingestion_returned_count"], 12)
        self.assertEqual(len(edge["last_receipts"]), 8)
        self.assertEqual([item["receipt_index"] for item in edge["last_receipts"]], list(range(5, 13)))
        edge["last_receipts"][0]["header_stamp_sec"] = -1
        self.assertEqual(self.edge()["last_receipts"][0]["header_stamp_sec"], 104.0)

    def test_static_and_unrelated_edges_do_not_claim_required_dynamic_receipt(self):
        self.buffer.set_transform_static(transform(), "authority")
        self.buffer.set_transform(transform("map", "camera"), "authority")
        self.buffer.set_transform(transform("/map", "odom"), "authority")
        self.assertEqual(self.edge()["received_count"], 0)
        self.buffer.set_transform(transform("odom", "base_footprint"), "authority")
        self.assertEqual(self.edge("execution_pose")["received_count"], 1)
        self.assertEqual(len(self.buffer.calls), 3)
        self.assertEqual(len(self.buffer.static_calls), 1)

    def test_ingestion_exception_preserved_with_diagnostic_receipt(self):
        self.failure = RuntimeError("actual core failure")
        with self.assertRaises(RuntimeError) as caught:
            self.buffer.set_transform(transform(), "authority")
        self.assertIs(caught.exception, self.failure)
        self.assertEqual(self.edge()["ingestion_exception_count"], 1)
        self.assertEqual(self.edge()["last_receipts"][0]["ingestion_call_state"], "raised")

    def test_silent_core_rejection_and_clock_failure_cannot_become_readiness(self):
        self.reject_insertion = True
        self.assertIs(self.buffer.set_transform(transform(), "authority"), self.result)
        receipt = self.edge()["last_receipts"][0]
        self.assertEqual(receipt["ingestion_call_state"], "returned")
        self.assertIsNone(receipt["newest_buffer_stamp_sec"])
        self.assertEqual(receipt["newest_buffer_lookup_error"], "LookupError")
        self.assertFalse(self.buffer.tf_receipt_snapshot()["insertion_acceptance_proven"])
        self.node.get_clock = lambda: (_ for _ in ()).throw(RuntimeError("diagnostic clock failure"))
        self.assertIs(self.buffer.set_transform(transform(), "authority"), self.result)
        self.assertEqual(len(self.buffer.calls), 2)

    def test_nonfinite_receipt_fields_never_poison_json_or_prevent_ingestion(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.assertIs(self.buffer.set_transform(transform(stamp=value), "authority"), self.result)
                self.ros_sec = value
                self.assertIs(self.buffer.set_transform(transform(), "authority"), self.result)
                self.ros_sec = 101.0
        self.assertEqual(self.edge()["received_count"], 0)
        self.assertEqual(len(self.buffer.calls), 6)
        json.dumps(self.buffer.tf_receipt_snapshot(), allow_nan=False)

    def test_newer_received_header_and_unchanged_buffer_stamp_do_not_prove_insertion(self):
        self.buffer.set_transform(transform(stamp=100), "authority")
        self.reject_insertion = True
        self.buffer.set_transform(transform(stamp=101), "authority")
        receipt = self.edge()["last_receipts"][-1]
        self.assertEqual(receipt["header_stamp_sec"], 101.0)
        self.assertEqual(receipt["newest_buffer_stamp_sec"], 100.0)
        self.assertEqual(receipt["ingestion_call_state"], "returned")
        self.assertFalse(self.buffer.tf_receipt_snapshot()["insertion_acceptance_proven"])


if __name__ == "__main__":
    unittest.main()
