from collections import deque
from types import SimpleNamespace
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception import stand_explorer_node  # noqa: E402


class StandExplorerTfTest(unittest.TestCase):
    def test_paused_observer_ignores_scans_before_readiness(self):
        fake_node = type(
            "FakeNode",
            (),
            {
                "observation_enabled": False,
            },
        )()

        stand_explorer_node.StandExplorerNode._scan_callback(fake_node, object())

    def test_lookup_uses_exact_laser_scan_timestamp(self):
        marker = object()
        scan_stamp = object()

        class FakeTime:
            @classmethod
            def from_msg(cls, stamp):
                self.assertIs(stamp, scan_stamp)
                return marker

        original_time = stand_explorer_node.Time
        try:
            stand_explorer_node.Time = FakeTime
            self.assertIs(
                stand_explorer_node._transform_time_for_scan_stamp(scan_stamp),
                marker,
            )
        finally:
            stand_explorer_node.Time = original_time

    def test_timing_validator_preserves_small_future_age(self):
        timing = stand_explorer_node.validated_observation_timing(
            observer_clock_sec=100.0,
            scan_stamp_sec=100.1,
            tf_stamp_sec=100.1,
            max_scan_age_sec=1.0,
            max_future_timestamp_sec=0.25,
            max_tf_age_sec=1.0,
            max_tf_scan_skew_sec=0.02,
        )

        self.assertAlmostEqual(timing.scan_age_sec, -0.1)
        self.assertAlmostEqual(timing.tf_age_sec, -0.1)
        self.assertEqual(timing.tf_scan_skew_sec, 0.0)

    def test_timing_validator_fails_closed_for_invalid_or_unaligned_stamps(self):
        common = {
            "observer_clock_sec": 100.0,
            "scan_stamp_sec": 99.9,
            "tf_stamp_sec": 99.9,
            "max_scan_age_sec": 1.0,
            "max_future_timestamp_sec": 0.25,
            "max_tf_age_sec": 1.0,
            "max_tf_scan_skew_sec": 0.02,
        }
        cases = (
            ({"observer_clock_sec": 0.0}, "observer clock"),
            ({"scan_stamp_sec": 0.0}, "scan timestamp"),
            ({"tf_stamp_sec": 0.0}, "TF timestamp"),
            ({"scan_stamp_sec": 98.9, "tf_stamp_sec": 98.9}, "scan timestamp is stale"),
            (
                {"scan_stamp_sec": 100.3, "tf_stamp_sec": 100.3},
                "scan timestamp is in the future",
            ),
            (
                {
                    "scan_stamp_sec": 99.9,
                    "tf_stamp_sec": 98.9,
                    "max_tf_scan_skew_sec": 2.0,
                },
                "TF timestamp is stale",
            ),
            ({"tf_stamp_sec": 99.87}, "TF/scan timestamp skew"),
        )
        for changes, expected in cases:
            with self.subTest(changes=changes):
                kwargs = {**common, **changes}
                with self.assertRaisesRegex(ValueError, expected):
                    stand_explorer_node.validated_observation_timing(**kwargs)

    def test_pending_queue_checks_and_looks_up_tf_without_waiting(self):
        marker_transform = object()
        calls = []

        class FakeDuration:
            def __init__(self, *, seconds):
                self.seconds = seconds

        class FakeBuffer:
            def can_transform(self, target, source, query_time, *, timeout):
                calls.append(("can", target, source, query_time, timeout.seconds))
                return True

            def lookup_transform(self, target, source, query_time, *, timeout):
                calls.append(("lookup", target, source, query_time, timeout.seconds))
                return marker_transform

        query_time = object()
        pending = stand_explorer_node._PendingScan(
            message=object(),
            scan_frame="base_scan",
            scan_stamp_sec=10.0,
            query_time=query_time,
            deadline_monotonic_sec=stand_explorer_node.time.monotonic() + 10.0,
        )
        processed = []
        fake_node = SimpleNamespace(
            pending_scans=deque((pending,)),
            tf_buffer=FakeBuffer(),
            runtime=SimpleNamespace(map_frame="map"),
            get_logger=lambda: SimpleNamespace(warn=lambda _message: None),
            _process_scan_with_transform=lambda item, transform: processed.append(
                (item, transform)
            ),
        )
        original_duration = stand_explorer_node.Duration
        try:
            stand_explorer_node.Duration = FakeDuration
            stand_explorer_node.StandExplorerNode._drain_pending_scans(fake_node)
        finally:
            stand_explorer_node.Duration = original_duration

        self.assertEqual(
            calls,
            [
                ("can", "map", "base_scan", query_time, 0.0),
                ("lookup", "map", "base_scan", query_time, 0.0),
            ],
        )
        self.assertEqual(processed, [(pending, marker_transform)])
        self.assertEqual(tuple(fake_node.pending_scans), ())

    def test_pending_queue_retains_scan_until_tf_arrives(self):
        class FakeDuration:
            def __init__(self, *, seconds):
                self.seconds = seconds

        pending = stand_explorer_node._PendingScan(
            message=object(),
            scan_frame="base_scan",
            scan_stamp_sec=10.0,
            query_time=object(),
            deadline_monotonic_sec=stand_explorer_node.time.monotonic() + 10.0,
        )
        fake_node = SimpleNamespace(
            pending_scans=deque((pending,)),
            tf_buffer=SimpleNamespace(can_transform=lambda *_args, **_kwargs: False),
            runtime=SimpleNamespace(map_frame="map"),
            get_logger=lambda: SimpleNamespace(warn=lambda _message: None),
        )
        original_duration = stand_explorer_node.Duration
        try:
            stand_explorer_node.Duration = FakeDuration
            stand_explorer_node.StandExplorerNode._drain_pending_scans(fake_node)
        finally:
            stand_explorer_node.Duration = original_duration

        self.assertEqual(tuple(fake_node.pending_scans), (pending,))

    def test_parser_exposes_shared_timing_defaults_and_legacy_alias(self):
        defaults = stand_explorer_node.build_parser().parse_args([])
        alias = stand_explorer_node.build_parser().parse_args(
            ["--max-scan-future-skew-sec", "0.1"]
        )

        self.assertEqual(defaults.max_scan_age_sec, 1.0)
        self.assertEqual(defaults.max_future_timestamp_sec, 0.25)
        self.assertEqual(defaults.max_tf_scan_skew_sec, 0.02)
        self.assertEqual(defaults.tf_timeout_sec, 0.5)
        self.assertEqual(defaults.pending_scan_limit, 8)
        self.assertEqual(alias.max_future_timestamp_sec, 0.1)

    def test_sim_time_override_sets_ros_node_clock_parameter(self):
        class FakeParameter:
            class Type:
                BOOL = "bool"

            def __init__(self, name, parameter_type, value):
                self.name = name
                self.parameter_type = parameter_type
                self.value = value

        original_parameter = stand_explorer_node.Parameter
        try:
            stand_explorer_node.Parameter = FakeParameter
            overrides = stand_explorer_node._node_parameter_overrides(True)
            self.assertEqual(len(overrides), 1)
            self.assertEqual(overrides[0].name, "use_sim_time")
            self.assertEqual(overrides[0].parameter_type, "bool")
            self.assertTrue(overrides[0].value)
            self.assertEqual(stand_explorer_node._node_parameter_overrides(False), [])
        finally:
            stand_explorer_node.Parameter = original_parameter


if __name__ == "__main__":
    unittest.main()
