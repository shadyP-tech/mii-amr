"""Replay callback/queue outcomes into observer input and latency evidence."""

from collections import deque
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.perception import stand_explorer_node as explorer
from scripts.aufgabe04.navigation.foundation.ros_runtime_config import RuntimeConfig, resolve_runtime_config
from scripts.aufgabe04.perception.lidar_observer_runtime import LidarObserverRuntime


def stamp(value):
    ns = round(value * 1e9)
    return SimpleNamespace(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def scan(*, frame="base_scan", stamp_sec=99.9):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame, stamp=stamp(stamp_sec)),
        ranges=[5.0, 5.0, 5.0], angle_min=0.0, angle_increment=.02,
        angle_max=.04, range_min=.1, range_max=10.0,
    )


class StandExplorerInputCountersTest(unittest.TestCase):
    def setUp(self):
        self.monotonic = 10.0
        self.node = object.__new__(explorer.StandExplorerNode)
        self.node.observation_enabled = True
        self.node.pending_scans = deque()
        self.node.args = SimpleNamespace(pending_scan_limit=1, tf_timeout_sec=.5,
                                         scan_topology_profile="linear")
        self.node.runtime = resolve_runtime_config(RuntimeConfig())
        self.node.observer_runtime = LidarObserverRuntime(self.node.runtime)
        self.node.timing_limits = explorer.DEFAULT_OBSERVATION_TIMING_LIMITS
        self.node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: stamp(100.0)))
        self.node.get_logger = lambda: SimpleNamespace(warn=Mock(), info=Mock())
        self.transform = SimpleNamespace(
            header=SimpleNamespace(frame_id="map", stamp=stamp(99.9)), child_frame_id="base_scan",
            transform=SimpleNamespace(translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                                      rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)),
        )
        self.node.tf_buffer = SimpleNamespace(can_transform=Mock(return_value=False),
                                              lookup_transform=Mock(return_value=self.transform))
        self.node.processed_scan_count = 0
        self.node.detected_candidate_count = 0
        self.node.accepted_observation_count = 0
        self.node.last_confirmed_stand_count = 0
        self.node.started_unix_sec = 90.0
        self.node.output_jsonl = Path("unused_observations.jsonl")
        self.node.map_bundle = None
        self.node.last_scan_pose_map = None
        self.node.last_processed_scan_stamp_sec = None
        self.node.visibility_session = SimpleNamespace(
            enabled=False, summary_fields=lambda **kwargs: explorer.disabled_visibility_summary_fields(),
        )
        self.node.detector_config = explorer.LidarStandDetectorConfig()
        self.contexts = [
            patch.object(explorer.time, "monotonic", side_effect=lambda: self.monotonic),
            patch.object(explorer, "_transform_time_for_scan_stamp", return_value=object()),
            patch.object(explorer, "Duration", side_effect=lambda **kwargs: kwargs),
        ]
        for context in self.contexts:
            context.start()
            self.addCleanup(context.stop)

    def summary(self):
        return explorer.observer_summary_payload(self.node)["scan_input_diagnostics"]

    def test_active_receipts_and_rejected_input_reasons_are_counted(self):
        self.node._scan_callback(scan(frame=""))
        self.node._scan_callback(scan(stamp_sec=98.0))
        with patch.object(explorer, "_transform_time_for_scan_stamp", side_effect=ValueError("invalid stamp")):
            self.node._scan_callback(scan())
        result = self.summary()
        self.assertEqual(result["received_scan_count"], 3)
        self.assertEqual(result["dropped_scan_count"], 3)
        self.assertEqual(result["rejected_scan_counts"], {
            "missing_scan_frame": 1, "invalid_scan_timing": 1, "invalid_scan_timestamp": 1,
        })
        self.assertEqual(result["pending_scan_count"], 0)
        self.assertEqual(result["latency_sec"], {})

    def test_queue_full_and_exact_tf_timeout_are_distinct_terminal_outcomes(self):
        self.node._scan_callback(scan())
        self.monotonic = 10.1
        self.node._scan_callback(scan(stamp_sec=99.95))
        self.monotonic = 10.7
        self.node._drain_pending_scans()
        result = self.summary()
        self.assertEqual(result["received_scan_count"], 2)
        self.assertEqual(result["dropped_scan_count"], 2)
        self.assertEqual(result["rejected_scan_counts"], {"pending_queue_full": 1, "exact_time_tf_timeout": 1})
        self.assertEqual(result["queue_drop_count"], 1)
        self.assertEqual(result["exact_tf_timeout_count"], 1)
        self.assertEqual(result["pending_scan_count"], 0)
        self.node.tf_buffer.lookup_transform.assert_not_called()

    def test_success_records_measured_queue_processing_and_scan_age_latency(self):
        self.node._scan_callback(scan())
        self.monotonic = 10.2
        self.node.tf_buffer.can_transform.return_value = True

        def detect(*args, **kwargs):
            self.monotonic += .03
            return []

        with patch.object(explorer, "detect_stand_candidates_from_scan", side_effect=detect):
            self.node._drain_pending_scans()
        result = self.summary()
        self.assertEqual(result["received_scan_count"], 1)
        self.assertEqual(self.node.processed_scan_count, 1)
        self.assertEqual(result["dropped_scan_count"], 0)
        self.assertEqual(result["pending_scan_count"], 0)
        for name, expected in (("tf_queue_wait", .2), ("processing", .03), ("scan_age_at_processing", .1)):
            series = result["latency_sec"][name]
            self.assertEqual(series["sample_count"], 1)
            self.assertAlmostEqual(series["mean_sec"], expected)
            self.assertAlmostEqual(series["maximum_sec"], expected)
        self.assertEqual(explorer.observer_summary_payload(self.node)["scan_topology_profile"], "linear")

    def test_invalid_exact_transform_is_counted_without_processed_scan(self):
        self.node.tf_buffer.can_transform.return_value = True
        self.transform.header.frame_id = "wrong_map"
        self.node._scan_callback(scan())
        self.assertEqual(self.summary()["rejected_scan_counts"], {"invalid_exact_time_tf": 1})
        self.assertEqual(self.node.processed_scan_count, 0)

    def test_detection_uses_captured_profile_after_arguments_change(self):
        self.node.observer_runtime = LidarObserverRuntime(self.node.runtime, "full_rotation")
        self.node.args.scan_topology_profile = "linear"
        self.node.tf_buffer.can_transform.return_value = True
        with patch.object(
            explorer, "detect_stand_candidates_from_scan",
            wraps=explorer.detect_stand_candidates_from_scan,
        ) as detect:
            self.node._scan_callback(scan())

        self.assertEqual(self.node.processed_scan_count, 1)
        self.assertEqual(detect.call_args.kwargs["scan_topology_profile"], "full_rotation")
        summary = explorer.observer_summary_payload(self.node)
        self.assertEqual(summary["scan_topology_profile"], "full_rotation")
        self.assertEqual(summary["runtime_config"]["scan_topology_profile"], "full_rotation")

    def test_malformed_wire_timestamp_is_counted_before_tf(self):
        message = scan()
        message.header.stamp.sec = "invalid"
        self.node._scan_callback(message)
        self.assertEqual(self.summary()["rejected_scan_counts"], {"invalid_scan_timestamp": 1})
        self.node.tf_buffer.can_transform.assert_not_called()

    def test_pause_accounts_pending_drop_and_separates_disabled_callbacks(self):
        self.node._scan_callback(scan())
        self.node.set_observation_enabled(False)
        self.node._scan_callback(object())
        result = self.summary()
        self.assertEqual(result["received_scan_count"], 1)
        self.assertEqual(result["ignored_disabled_scan_count"], 1)
        self.assertEqual(result["rejected_scan_counts"], {"observation_disabled": 1})
        self.assertEqual(result["pending_scan_count"], 0)

    def test_parser_defaults_linear_and_requires_explicit_profile_selection(self):
        parser = explorer.build_parser()
        self.assertEqual(parser.parse_args([]).scan_topology_profile, "linear")
        self.assertEqual(parser.parse_args(["--scan-topology-profile", "full_rotation"]).scan_topology_profile, "full_rotation")


if __name__ == "__main__":
    unittest.main()
