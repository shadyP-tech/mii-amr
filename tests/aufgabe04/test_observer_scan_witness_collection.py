"""Scan arrival can seed witnesses without a camera result or camera-time pose."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from scripts.aufgabe04.real_robot.observer.node import _StampedMessage
from tests.aufgabe04 import test_camera_observer_processing as fixtures

transform = fixtures.transform


class ObserverScanWitnessCollectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CameraObserverProcessingTest()
        self.adapter = self.fixture.make_adapter()
        self.adapter.scans = deque(maxlen=20)
        self.adapter._scan_target_persistence = Mock()
        self.lookups = []

        def lookup(target, source, stamp=None):
            self.lookups.append((target, source, stamp))
            # Historical scan-time poses differ from adapter.last_pose.
            if (target, source) == ("scan", "map"):
                return transform((-.003, 0., 0.))
            if (target, source) == ("map", "base"):
                return transform((.003, 0., 0.))
            return transform((0., 0., .45), (.5, -.5, .5, -.5))

        self.adapter._lookup_scan_witness = lookup
        self.patch = patch(
            "scripts.aufgabe04.real_robot.observer.scan_witness_collection.transform_mismatches",
            return_value=())
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def sample(self, stamp=100.):
        whole = int(stamp)
        header = SimpleNamespace(frame_id="scan", stamp=SimpleNamespace(
            sec=whole, nanosec=round((stamp - whole) * 1e9)))
        return SimpleNamespace(header=header, ranges=(.6,) * 5, angle_min=-.02,
            angle_increment=.01, range_min=.01, range_max=10., angle_max=.02)

    def test_callback_ingests_exact_scan_time_without_running_camera(self):
        message = self.sample()
        self.adapter._process_latest = Mock()
        self.adapter._on_scan(message)
        self.adapter._process_latest.assert_not_called()
        call = self.adapter._scan_target_persistence.ingest_scan.call_args
        self.assertIsNotNone(call)
        self.assertEqual(call.args[0].receipt_sec, 100.1)
        context = call.kwargs["context"]
        self.assertAlmostEqual(context.robot_pose.x_m, .003)
        self.assertAlmostEqual(context.scan_pose_map.x_m, .003)
        self.assertEqual(context.image_stamp_sec, 100.)
        self.assertIs(self.lookups[0][2], message.header.stamp)
        self.assertIs(self.lookups[1][2], message.header.stamp)
        self.assertEqual(context.epoch_key, "0")
        self.assertEqual(len(self.adapter._pending_scan_witnesses), 0)

    def test_exact_tf_delay_preserves_scan_order_then_timer_collects(self):
        original = self.adapter._lookup_scan_witness
        self.adapter._lookup_scan_witness = Mock(side_effect=RuntimeError("TF pending"))
        self.adapter._on_scan(self.sample(99.9))
        self.adapter._on_scan(self.sample(100.))
        self.adapter._scan_target_persistence.ingest_scan.assert_not_called()
        self.assertEqual([s.stamp_sec for s in self.adapter._pending_scan_witnesses], [99.9, 100.])
        self.adapter._lookup_scan_witness = original
        self.adapter._collect_scan_witnesses()
        stamps = [call.args[0].scan_stamp_sec for call in
                  self.adapter._scan_target_persistence.ingest_scan.call_args_list]
        self.assertEqual(stamps, [99.9, 100.])

    def test_stale_tf_pending_scan_cannot_seed_history(self):
        self.adapter._lookup_scan_witness = Mock(side_effect=RuntimeError("TF pending"))
        self.adapter._on_scan(self.sample())
        self.fixture.clock_sec = 100.6
        self.adapter._collect_scan_witnesses()
        self.adapter._scan_target_persistence.ingest_scan.assert_not_called()
        self.assertEqual(len(self.adapter._pending_scan_witnesses), 0)
        self.assertEqual(self.adapter._camera_pipeline_counters["scan_witness_expired_before_tf"], 1)
        self.adapter._scan_target_persistence.reset.assert_called_once()

    def test_contract_reset_clears_pending_sources_and_head_context(self):
        self.adapter._pending_scan_witnesses = deque([_StampedMessage(100., self.sample())])
        tracking = self.adapter._candidate_head_tracking = Mock()
        self.adapter._reset_observation_evidence()
        self.assertEqual(len(self.adapter._pending_scan_witnesses), 0)
        tracking.reset.assert_called_once_with("observation_evidence_reset")

    def test_witness_lookup_preserves_selected_camera_tf_capture(self):
        adapter = self.adapter
        # Exercise the real lookup, not the per-test fixture's lookup stub.
        del adapter._lookup_scan_witness
        adapter.Time = Mock(return_value="static", from_msg=Mock(return_value="exact"))
        adapter.Duration = Mock(return_value="nonblocking")
        adapter.tf_buffer = SimpleNamespace(lookup_transform=Mock(return_value=transform()))
        adapter._active_tf_request = {"selected": "camera"}
        adapter._capture_pending = {"tf_samples": ["camera_transform"]}
        adapter._lookup_scan_witness("scan", "map", self.sample().header.stamp)
        self.assertEqual(adapter._active_tf_request, {"selected": "camera"})
        self.assertEqual(adapter._capture_pending, {"tf_samples": ["camera_transform"]})
        self.assertEqual(adapter.tf_buffer.lookup_transform.call_args.args, ("scan", "map", "exact"))


if __name__ == "__main__":
    unittest.main()
