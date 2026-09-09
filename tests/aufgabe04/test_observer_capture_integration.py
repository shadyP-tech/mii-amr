"""Capture the processed tuple, including failures, without creating authority."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode, _StampedMessage


class ObserverCaptureIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.node = PassiveRealViewpointNode.__new__(PassiveRealViewpointNode)
        self.node.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=101_000_000_000)))
        self.node.capture_history = Mock()
        self.node.profile = object()
        self.node.calibration = object()
        self.node.stand_model_profile = SimpleNamespace(sha256="c" * 64)
        self.node.completed = False
        self.image = _StampedMessage(100., SimpleNamespace(data=b"original compressed pixels", format="jpeg"), 100.1, 10.1)
        self.scan = _StampedMessage(100.01, object(), 100.12, 10.12)
        self.info = _StampedMessage(100., object(), 100.09, 10.09)
        for name, result in (("real_robot_profile_sha256", "a" * 64),
                             ("camera_calibration_sha256", "b" * 64),
                             ("sensor_capture_metadata", {"sensors": "unchanged"})):
            patcher = patch(f"scripts.aufgabe04.real_robot.observer.node.{name}", return_value=result)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_pending_tf_defers_capture_until_same_tuple_failed_or_accepted_outcome(self):
        for outcome in ("obsolete_detector_result", "recommendation_committed"):
            with self.subTest(outcome=outcome):
                self.node.capture_history.reset_mock()
                self.node._stage_camera_capture(self.image, self.scan, self.info)
                self.node._capture_camera_outcome("tf_pending_exact_time", {"reason": "warming"})
                self.node.capture_history.submit.assert_not_called()
                self.node._capture_pending["detector_metadata"] = {"attempts": ["nominal", "strict"]}
                self.node._capture_camera_outcome(outcome, {"reason": "final"})
                self.node._capture_camera_outcome(outcome, {"reason": "duplicate status"})
                self.node.capture_history.submit.assert_called_once()
                call = self.node.capture_history.submit.call_args
                self.assertEqual(call.args, (b"original compressed pixels",))
                self.assertEqual(call.kwargs["compressed_format"], "jpeg")
                metadata = call.kwargs["metadata"]
                self.assertEqual(metadata["observer_state"], outcome)
                self.assertEqual(metadata["image_stamp_sec"], 100.)
                self.assertEqual(metadata["image_received_ros_sec"], 100.1)
                self.assertEqual(metadata["detector_metadata"]["attempts"], ["nominal", "strict"])
                self.assertFalse(self.node.completed)

    def test_unpaired_or_malformed_metadata_keeps_original_failed_image(self):
        self.node._stage_camera_capture(self.image, None, None)
        with patch("scripts.aufgabe04.real_robot.observer.node.sensor_capture_metadata",
                   side_effect=AttributeError("no paired scan")):
            self.node._capture_camera_outcome("awaiting_synchronized_sensors", {})
        metadata = self.node.capture_history.submit.call_args.kwargs["metadata"]
        self.assertIsNone(metadata["scan_stamp_sec"])
        self.assertIn("no paired scan", metadata["sensor_metadata_error"])
        self.assertEqual(self.node.capture_history.submit.call_args.args[0], self.image.value.data)

    def test_static_tf_stamp_zero_and_exact_query_identity_are_preserved(self):
        self.node._stage_camera_capture(self.image, self.scan, self.info)
        self.node._active_tf_request = {"target_frame": "base", "source_frame": "camera",
                                        "query_kind": "time_invariant_camera_extrinsic", "query_stamp_sec": None}
        transform = SimpleNamespace(
            header=SimpleNamespace(frame_id="base", stamp=SimpleNamespace(sec=0, nanosec=0)),
            child_frame_id="camera", transform=SimpleNamespace(
                translation=SimpleNamespace(x=.1, y=.2, z=.3),
                rotation=SimpleNamespace(x=0., y=0., z=0., w=1.)))
        self.node._capture_tf_sample(transform)
        self.node._capture_camera_outcome("collecting_consensus", {})
        sample = self.node.capture_history.submit.call_args.kwargs["metadata"]["tf_samples"][0]
        self.assertEqual(sample["returned_stamp_sec"], 0.)
        self.assertEqual(sample["translation_xyz_m"], (.1, .2, .3))
        self.assertEqual(sample["query_kind"], "time_invariant_camera_extrinsic")

    def test_capture_errors_are_diagnostics_only(self):
        self.node._stage_camera_capture(self.image, self.scan, self.info)
        self.node.capture_history.submit.side_effect = OSError("disk unavailable")
        self.node._capture_camera_outcome("collecting_consensus", {})
        self.assertEqual(self.node._camera_pipeline_counters["capture_metadata_failures"], 1)
        self.assertIsNone(self.node._capture_pending)
        self.assertFalse(self.node.completed)
        self.node.capture_history.snapshot.side_effect = RuntimeError("diagnostic snapshot")
        self.assertTrue(self.node._capture_snapshot()["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
