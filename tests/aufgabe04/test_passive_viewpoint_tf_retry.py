import json
import math
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.perception.stand_axis.real_camera_profile import (
    RealCameraStandAxisProfile,
)
from scripts.aufgabe04.perception.stand_axis_consensus import (
    AxisConsensusAccumulator,
)
from scripts.aufgabe04.real_robot.passive_observer_tf_retry import (
    PassiveObserverTfRetryScheduler,
)
from scripts.aufgabe04.real_robot.passive_viewpoint_node import (
    PassiveRealViewpointNode,
    _StampedMessage,
    _SynchronizedSensorTuple,
)


class _ResetCountingConsensus:
    required_samples = 7
    sample_count = 3

    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def _message(name: str) -> object:
    return SimpleNamespace(name=name)


def _sensor_tuple(stamp_sec: float = 1787663986.244712):
    return _SynchronizedSensorTuple(
        image=_StampedMessage(stamp_sec, _message("image")),
        scan=_StampedMessage(stamp_sec - 0.002, _message("scan")),
        camera_info=_StampedMessage(stamp_sec - 0.01, _message("camera_info")),
    )


class PassiveViewpointTfRetryIntegrationTests(unittest.TestCase):
    def _adapter(self) -> PassiveRealViewpointNode:
        adapter = PassiveRealViewpointNode.__new__(PassiveRealViewpointNode)
        adapter.images = deque(maxlen=8)
        adapter.scans = deque(maxlen=20)
        adapter.camera_infos = deque(maxlen=8)
        adapter.last_processed_image_stamp = -math.inf
        adapter.tf_retry_scheduler = (
            PassiveObserverTfRetryScheduler[_SynchronizedSensorTuple]()
        )
        adapter._active_tf_request = None
        adapter.args = SimpleNamespace(
            sync_tolerance_sec=0.10,
            camera_info_tolerance_sec=1.0,
            tf_timeout_sec=0.15,
        )
        adapter.consensus = _ResetCountingConsensus()
        adapter._write_status = lambda state, **details: None
        return adapter

    def test_complete_synchronized_tuple_is_frozen_while_tf_catches_up(self):
        adapter = self._adapter()
        first = _sensor_tuple()
        adapter.images.append(first.image)
        adapter.scans.append(first.scan)
        adapter.camera_infos.append(first.camera_info)

        selected = adapter._next_sensor_tuple()
        newer = _sensor_tuple(first.image.stamp_sec + 0.05)
        adapter.images.append(newer.image)
        adapter.scans.append(newer.scan)
        adapter.camera_infos.append(newer.camera_info)

        self.assertIs(selected, adapter._next_sensor_tuple())
        self.assertIs(selected.image.value, first.image.value)
        self.assertIs(selected.scan.value, first.scan.value)
        self.assertIs(selected.camera_info.value, first.camera_info.value)
        self.assertEqual(adapter.last_processed_image_stamp, -math.inf)

    def test_millisecond_tf_lag_defers_without_resetting_consensus(self):
        adapter = self._adapter()
        sensor_tuple = _sensor_tuple()
        adapter.tf_retry_scheduler.offer(
            sensor_tuple,
            stamp_sec=sensor_tuple.image.stamp_sec,
        )
        adapter._active_tf_request = {
            "target_frame": "map",
            "source_frame": "base_footprint",
            "query_stamp_sec": sensor_tuple.image.stamp_sec,
        }
        statuses = []
        adapter._write_status = lambda state, **details: statuses.append(
            (state, details)
        )

        with patch(
            "scripts.aufgabe04.real_robot.passive_viewpoint_node.time.time",
            return_value=1787663986.457761,
        ):
            adapter._defer_for_exact_tf(
                sensor_tuple,
                reason=(
                    "Requested time 1787663986.244712 but latest data is "
                    "1787663986.242854"
                ),
            )

        self.assertIs(
            adapter.tf_retry_scheduler.pending_frame.frame,
            sensor_tuple,
        )
        self.assertEqual(adapter.consensus.reset_count, 0)
        self.assertEqual(adapter.last_processed_image_stamp, -math.inf)
        self.assertEqual(statuses[0][0], "tf_pending_exact_time")
        self.assertEqual(statuses[0][1]["tf_retry_attempt"]["retry_count"], 1)
        self.assertFalse(statuses[0][1]["retry_exhausted"])

    def test_retry_budget_exhaustion_discards_only_transport_tuple(self):
        adapter = self._adapter()
        sensor_tuple = _sensor_tuple()
        adapter.tf_retry_scheduler.offer(
            sensor_tuple,
            stamp_sec=sensor_tuple.image.stamp_sec,
        )
        statuses = []
        adapter._write_status = lambda state, **details: statuses.append(
            (state, details)
        )

        with patch(
            "scripts.aufgabe04.real_robot.passive_viewpoint_node.time.time",
            side_effect=(100.0, 100.05, 100.16),
        ):
            adapter._defer_for_exact_tf(sensor_tuple, reason="first lag")
            adapter._defer_for_exact_tf(sensor_tuple, reason="lag persisted")
            adapter._defer_for_exact_tf(sensor_tuple, reason="lag persisted")

        self.assertIsNone(adapter.tf_retry_scheduler.pending_frame)
        self.assertEqual(
            adapter.last_processed_image_stamp,
            sensor_tuple.image.stamp_sec,
        )
        self.assertEqual(adapter.consensus.reset_count, 0)
        self.assertEqual(len(statuses), 2)
        self.assertEqual(statuses[-1][0], "tf_retry_exhausted")
        self.assertEqual(statuses[-1][1]["tf_retry_attempt"]["retry_count"], 3)
        self.assertTrue(statuses[-1][1]["retry_exhausted"])

    def test_lookup_is_nonblocking_and_records_exact_request(self):
        adapter = self._adapter()
        calls = []

        class _Time:
            @staticmethod
            def from_msg(stamp):
                return (stamp.sec, stamp.nanosec)

        adapter.Time = _Time
        adapter.Duration = lambda *, seconds: ("duration", seconds)
        adapter.tf_buffer = SimpleNamespace(
            lookup_transform=lambda *args, **kwargs: calls.append(
                (args, kwargs)
            )
            or "transform"
        )
        stamp = SimpleNamespace(sec=1787663986, nanosec=244712000)

        result = adapter._lookup("map", "base_footprint", stamp)

        self.assertEqual(result, "transform")
        self.assertEqual(calls[0][1]["timeout"], ("duration", 0.0))
        self.assertEqual(
            adapter._active_tf_request,
            {
                "target_frame": "map",
                "source_frame": "base_footprint",
                "query_kind": "exact_sensor_time",
                "query_stamp_sec": 1787663986.244712,
            },
        )

    def test_static_camera_extrinsic_lookup_is_also_nonblocking(self):
        adapter = self._adapter()
        calls = []

        class _Time:
            def __init__(self):
                pass

        adapter.Time = _Time
        adapter.Duration = lambda *, seconds: ("duration", seconds)
        adapter.tf_buffer = SimpleNamespace(
            lookup_transform=lambda *args, **kwargs: calls.append(
                (args, kwargs)
            )
            or "transform"
        )

        result = adapter._lookup_static_transform(
            "base_footprint",
            "camera_rgb_optical_frame",
        )

        self.assertEqual(result, "transform")
        self.assertEqual(calls[0][1]["timeout"], ("duration", 0.0))
        self.assertEqual(
            adapter._active_tf_request,
            {
                "target_frame": "base_footprint",
                "source_frame": "camera_rgb_optical_frame",
                "query_kind": "time_invariant_camera_extrinsic",
                "query_stamp_sec": None,
            },
        )

    def test_status_is_latest_snapshot_plus_append_only_history(self):
        adapter = self._adapter()
        adapter.consensus = AxisConsensusAccumulator(
            required_samples=3,
            max_deviation_rad=math.radians(5.0),
        )
        adapter.consensus.add(
            yaw_rad=0.1,
            source="edge",
            side="axis_only",
            qr_texts=(),
        )
        adapter.stand_axis_profile = RealCameraStandAxisProfile.from_cli(
            edge_preprocess="channel_union",
            canny_low=20,
            canny_high=60,
        )
        adapter._write_status = PassiveRealViewpointNode._write_status.__get__(
            adapter,
            PassiveRealViewpointNode,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter.args.status_json = root / "observer_status.json"
            adapter.args.status_events_jsonl = root / "observer_events.jsonl"

            adapter._write_status("collecting_consensus")
            adapter._write_status("tf_pending_exact_time", reason="catching up")

            status = json.loads(adapter.args.status_json.read_text())
            events = [
                json.loads(line)
                for line in adapter.args.status_events_jsonl.read_text().splitlines()
            ]

        self.assertEqual(status["schema_version"], 2)
        self.assertEqual(status["state"], "tf_pending_exact_time")
        self.assertEqual(status["axis_consensus"]["sample_count"], 1)
        self.assertEqual(status["axis_consensus"]["required_sample_count"], 3)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["state"], "collecting_consensus")
        self.assertEqual(events[1]["state"], "tf_pending_exact_time")


if __name__ == "__main__":
    unittest.main()
