from collections import deque
from types import SimpleNamespace
import json
import math
import sys
import tempfile
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
        self.assertIsNone(defaults.summary_json)
        self.assertIsNone(defaults.survey_candidate_radius_m)
        self.assertEqual(defaults.duration_sec, 0.0)
        self.assertEqual(alias.max_future_timestamp_sec, 0.1)

    def test_observer_summary_records_negative_scan_epoch(self):
        fake_node = SimpleNamespace(
            started_unix_sec=10.0,
            output_jsonl=Path("empty_observations.jsonl"),
            map_bundle=SimpleNamespace(bundle_sha256="a" * 64),
            runtime=SimpleNamespace(
                map_frame="map",
                as_log_dict=lambda: {"map_frame": "map"},
            ),
            timing_limits=SimpleNamespace(
                as_dict=lambda: {"max_scan_age_sec": 1.0}
            ),
            last_scan_pose_map={"x_m": 1.0, "y_m": 2.0, "yaw_rad": 0.0},
            last_processed_scan_stamp_sec=9.9,
            processed_scan_count=5,
            detected_candidate_count=0,
            accepted_observation_count=0,
            last_confirmed_stand_count=0,
        )

        payload = stand_explorer_node.observer_summary_payload(fake_node)

        self.assertFalse(payload["motion_published"])
        self.assertEqual(payload["processed_scan_count"], 5)
        self.assertEqual(payload["accepted_observation_count"], 0)
        self.assertEqual(
            payload["scan_frame_pose_in_planning_frame"]["x_m"],
            1.0,
        )
        self.assertEqual(
            payload[
                stand_explorer_node.PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY
            ]["max_width_m"],
            0.45,
        )

    def test_summary_binds_morphology_without_narrowing_proposals(self):
        profile = stand_explorer_node.stand_width_profile_from_radius(0.06)
        fake_node = SimpleNamespace(
            started_unix_sec=10.0,
            output_jsonl=Path("observations.jsonl"),
            map_bundle=SimpleNamespace(bundle_sha256="a" * 64),
            runtime=SimpleNamespace(
                map_frame="map",
                as_log_dict=lambda: {"map_frame": "map"},
            ),
            timing_limits=SimpleNamespace(as_dict=lambda: {}),
            last_scan_pose_map=None,
            last_processed_scan_stamp_sec=None,
            processed_scan_count=0,
            detected_candidate_count=0,
            accepted_observation_count=0,
            last_confirmed_stand_count=0,
            detector_config=stand_explorer_node.LidarStandDetectorConfig(),
            morphology_profile=profile,
        )

        payload = stand_explorer_node.observer_summary_payload(fake_node)

        self.assertEqual(
            payload[
                stand_explorer_node.PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY
            ]["max_width_m"],
            0.45,
        )
        self.assertEqual(
            payload[stand_explorer_node.MORPHOLOGY_PROFILE_EVIDENCE_KEY],
            profile.to_evidence_dict(),
        )
        self.assertEqual(
            len(payload[stand_explorer_node.MORPHOLOGY_PROFILE_SHA256_KEY]),
            64,
        )

    def test_visibility_receipts_are_flushed_once_and_bound_to_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "visibility.jsonl"
            profile = stand_explorer_node.stand_width_profile_from_radius(0.06)
            detector_config = stand_explorer_node.LidarStandDetectorConfig()
            visibility_session = stand_explorer_node.LidarVisibilitySession.create(
                output_path=receipt_path,
                survey_id="survey_01",
                viewpoint_id="viewpoint_01",
                runtime_config={"map_frame": "map", "scan_topic": "/scan"},
                timing_limits={},
                map_bundle_sha256="a" * 64,
                observation_geometry_mode=(
                    stand_explorer_node.FROZEN_ODOM_OBSERVATION_GEOMETRY
                ),
                proposal_detector_config=(
                    stand_explorer_node.proposal_detector_config_evidence(
                        detector_config
                    )
                ),
                morphology_profile=profile.to_evidence_dict(),
            )
            receipt = stand_explorer_node.lidar_visibility_receipt_from_scan(
                receipt_id="viewpoint_01_000001",
                survey_id="survey_01",
                viewpoint_id="viewpoint_01",
                planning_frame="map",
                scan_frame="base_scan",
                scan_topic="/scan",
                map_bundle_sha256="a" * 64,
                observer_config_sha256=(
                    visibility_session.observer_config_sha256
                ),
                scan_stamp_sec=1.0,
                pose_stamp_sec=1.0,
                observer_clock_sec=1.01,
                scan_pose_map=stand_explorer_node.Pose2D(0.0, 0.0, 0.0),
                angle_min_rad=-1.0,
                angle_increment_rad=1.0,
                range_min_m=0.08,
                range_max_m=3.5,
                ranges_m=(1.0, math.inf, 2.0),
            )
            visibility_session.buffer_receipt(receipt)
            fake_node = SimpleNamespace(
                started_unix_sec=1.0,
                output_jsonl=root / "observations.jsonl",
                map_bundle=SimpleNamespace(bundle_sha256="a" * 64),
                runtime=SimpleNamespace(
                    map_frame="map",
                    as_log_dict=lambda: {"map_frame": "map"},
                ),
                timing_limits=SimpleNamespace(as_dict=lambda: {}),
                last_scan_pose_map={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                last_processed_scan_stamp_sec=1.0,
                processed_scan_count=1,
                detected_candidate_count=0,
                accepted_observation_count=0,
                last_confirmed_stand_count=0,
                detector_config=detector_config,
                morphology_profile=profile,
                visibility_session=visibility_session,
            )
            summary_path = root / "observer_summary.json"

            stand_explorer_node.write_observer_summary(
                summary_path,
                fake_node,
            )
            payload = json.loads(summary_path.read_text())

            self.assertEqual(
                payload[stand_explorer_node.VISIBILITY_RECEIPT_COUNT_KEY],
                1,
            )
            self.assertEqual(
                len(
                    payload[
                        stand_explorer_node.VISIBILITY_RECEIPTS_FILE_SHA256_KEY
                    ]
                ),
                64,
            )
            self.assertEqual(len(receipt_path.read_text().splitlines()), 1)

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
