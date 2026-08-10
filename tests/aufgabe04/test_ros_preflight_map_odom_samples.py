import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation import ros_preflight  # noqa: E402


def _transform(
    *,
    map_frame: str = "map",
    odom_frame: str = "odom",
    stamp_sec: float = 10.0,
    x_m: float = 1.25,
    y_m: float = -0.50,
    yaw_rad: float = 0.30,
    quaternion=None,
):
    integer_sec = math.floor(stamp_sec)
    nanosec = int(round((stamp_sec - integer_sec) * 1_000_000_000.0))
    if quaternion is None:
        quaternion = (
            0.0,
            0.0,
            math.sin(0.5 * yaw_rad),
            math.cos(0.5 * yaw_rad),
        )
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=map_frame,
            stamp=SimpleNamespace(sec=integer_sec, nanosec=nanosec),
        ),
        child_frame_id=odom_frame,
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=x_m, y=y_m, z=0.0),
            rotation=SimpleNamespace(
                x=quaternion[0],
                y=quaternion[1],
                z=quaternion[2],
                w=quaternion[3],
            ),
        ),
    )


def _build(transform, **overrides):
    kwargs = {
        "expected_map_frame": "map",
        "expected_odom_frame": "odom",
        "receipt_time_nanoseconds": 10_100_000_000,
        "capture_time_nanoseconds": 10_200_000_000,
        "max_age_sec": 1.0,
        "max_future_sec": 0.25,
        "amcl_sample_index": 0,
    }
    kwargs.update(overrides)
    return ros_preflight.build_stationary_map_from_odom_sample(
        transform,
        **kwargs,
    )


class RosPreflightMapFromOdomSamplesTest(unittest.TestCase):
    def test_dynamic_callback_counts_only_configured_direct_transform(self):
        node = object.__new__(ros_preflight.RosPreflightNode)
        node.config = SimpleNamespace(map_frame="map", odom_frame="odom")
        node.latest_dynamic_map_to_odom = None
        node.latest_dynamic_map_to_odom_receipt = None
        node.dynamic_map_to_odom_message_count = 0
        node.get_clock = lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=10_100_000_000)
        )

        node._dynamic_tf_callback(
            SimpleNamespace(transforms=[_transform(map_frame="other")])
        )
        self.assertEqual(node.dynamic_map_to_odom_message_count, 0)

        direct = _transform()
        node._dynamic_tf_callback(SimpleNamespace(transforms=[direct]))
        self.assertEqual(node.dynamic_map_to_odom_message_count, 1)
        self.assertIs(node.latest_dynamic_map_to_odom, direct)

    def test_valid_direct_sample_has_policy_and_capture_fields(self):
        sample, failure = _build(_transform())

        self.assertEqual(failure, "")
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample["source"], "direct_dynamic_tf")
        self.assertEqual(sample["target_frame"], "map")
        self.assertEqual(sample["source_frame"], "odom")
        self.assertEqual(sample["observed_target_frame"], "map")
        self.assertEqual(sample["observed_source_frame"], "odom")
        self.assertEqual(sample["amcl_sample_index"], 0)
        self.assertAlmostEqual(sample["x_m"], 1.25)
        self.assertAlmostEqual(sample["y_m"], -0.50)
        self.assertAlmostEqual(sample["yaw_rad"], 0.30)
        self.assertAlmostEqual(sample["stamp_sec"], 10.0)
        self.assertEqual(sample["stamp_nanoseconds"], 10_000_000_000)
        self.assertEqual(
            sample["receipt_time_nanoseconds"],
            10_100_000_000,
        )
        self.assertEqual(
            sample["capture_time_nanoseconds"],
            10_200_000_000,
        )
        self.assertAlmostEqual(sample["receipt_time_sec"], 10.10)
        self.assertAlmostEqual(sample["capture_time_sec"], 10.20)
        self.assertAlmostEqual(sample["header_age_sec"], 0.20)
        self.assertAlmostEqual(sample["receipt_age_sec"], 0.10)

    def test_malformed_or_nonfresh_direct_samples_are_rejected(self):
        cases = (
            (
                _transform(map_frame="wrong_map"),
                {},
                "frame identity mismatch",
            ),
            (
                _transform(quaternion=(0.0, 0.0, 0.0, 0.0)),
                {},
                "invalid quaternion",
            ),
            (
                _transform(x_m=math.nan),
                {},
                "non-finite values",
            ),
            (
                _transform(stamp_sec=10.50),
                {},
                "future-dated",
            ),
            (
                _transform(stamp_sec=8.0),
                {},
                "stale",
            ),
            (
                _transform(),
                {"receipt_time_nanoseconds": "not-an-integer"},
                "non-negative integer",
            ),
        )
        for transform, overrides, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                sample, failure = _build(transform, **overrides)
                self.assertIsNone(sample)
                self.assertIn(expected_reason, failure)

    def test_only_captures_from_latest_amcl_window_are_retained(self):
        old_sample, _ = _build(_transform(x_m=0.0), amcl_sample_index=0)
        current_sample, _ = _build(
            _transform(x_m=1.0),
            amcl_sample_index=2,
        )
        samples, failures = (
            ros_preflight._latest_stationary_map_from_odom_capture_window(
                [old_sample, None, current_sample, None],
                [None, "old failure", None, "current failure"],
                amcl_window_size=2,
            )
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["amcl_sample_index"], 0)
        self.assertAlmostEqual(samples[0]["x_m"], 1.0)
        self.assertEqual(
            failures,
            [{"amcl_sample_index": 1, "reason": "current failure"}],
        )

    def test_pairing_rejects_cached_or_not_strictly_ordered_samples(self):
        candidate, failure = _build(
            _transform(stamp_sec=11.0),
            receipt_time_nanoseconds=11_100_000_000,
            capture_time_nanoseconds=11_200_000_000,
        )
        self.assertEqual(failure, "")
        assert candidate is not None

        accepted = ros_preflight._stationary_map_from_odom_pairing_failure(
            candidate,
            baseline_identity=(10_000_000_000, 10_100_000_000),
            previous_sample=None,
            paired_amcl_receipt_nanoseconds=11_000_000_000,
        )
        reused_stamp = ros_preflight._stationary_map_from_odom_pairing_failure(
            candidate,
            baseline_identity=(11_000_000_000, 10_100_000_000),
            previous_sample=None,
            paired_amcl_receipt_nanoseconds=11_000_000_000,
        )
        reused_receipt = (
            ros_preflight._stationary_map_from_odom_pairing_failure(
                candidate,
                baseline_identity=(10_000_000_000, 11_100_000_000),
                previous_sample=None,
                paired_amcl_receipt_nanoseconds=11_000_000_000,
            )
        )
        before_amcl = ros_preflight._stationary_map_from_odom_pairing_failure(
            candidate,
            baseline_identity=(10_000_000_000, 10_100_000_000),
            previous_sample=None,
            paired_amcl_receipt_nanoseconds=11_100_000_000,
        )

        self.assertEqual(accepted, "")
        self.assertIn("stamp did not advance", reused_stamp)
        self.assertIn("receipt did not advance", reused_receipt)
        self.assertIn("did not follow paired AMCL", before_amcl)

    def test_odom_capture_gate_requires_every_configured_pair(self):
        node = object.__new__(ros_preflight.RosPreflightNode)
        node.stationary_amcl_sample_count = 5
        node.stationary_amcl_samples = [object() for _ in range(5)]
        samples = []
        for index in range(5):
            sample, failure = _build(
                _transform(
                    stamp_sec=10.0 + 0.1 * index,
                    x_m=0.001 * index,
                ),
                receipt_time_nanoseconds=(
                    10_100_000_000 + 100_000_000 * index
                ),
                capture_time_nanoseconds=(
                    10_200_000_000 + 100_000_000 * index
                ),
                amcl_sample_index=index,
            )
            self.assertEqual(failure, "")
            assert sample is not None
            samples.append(sample)
        node.stationary_map_from_odom_samples = samples[:4]
        node.stationary_map_from_odom_capture_failures = []
        node.stationary_map_from_odom_capture_failure_history = []

        insufficient = node._observe_stationary_map_from_odom_samples()

        self.assertFalse(insufficient.ok)
        self.assertEqual(insufficient.data["sample_count"], 4)
        self.assertEqual(insufficient.data["minimum_sample_count"], 2)
        self.assertEqual(insufficient.data["required_pair_count"], 5)
        self.assertFalse(insufficient.data["complete_transform_window"])

        node.stationary_map_from_odom_samples = samples
        complete = node._observe_stationary_map_from_odom_samples()

        self.assertTrue(complete.ok)
        self.assertEqual(complete.data["sample_order"], "oldest_to_newest")
        self.assertTrue(complete.data["direct_dynamic_tf_required"])

        node.stationary_map_from_odom_capture_failures = [
            {"amcl_sample_index": 2, "reason": "missing pair"}
        ]
        failed_pair = node._observe_stationary_map_from_odom_samples()
        self.assertFalse(failed_pair.ok)
        self.assertEqual(len(failed_pair.data["capture_failures"]), 1)

    def test_result_json_field_is_optional_and_persists_order(self):
        legacy_constructor = ros_preflight.RosPreflightResult(
            ok=True,
            failures=[],
            observations=[],
            runtime_config={},
        )
        samples = [
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            {"x_m": 0.001, "y_m": 0.0, "yaw_rad": 0.002},
        ]
        with_samples = ros_preflight.RosPreflightResult(
            ok=True,
            failures=[],
            observations=[],
            runtime_config={},
            stationary_map_from_odom_samples=samples,
        )

        self.assertEqual(
            legacy_constructor.to_json_dict()[
                "stationary_map_from_odom_samples"
            ],
            [],
        )
        self.assertEqual(
            with_samples.to_json_dict()[
                "stationary_map_from_odom_samples"
            ],
            samples,
        )


if __name__ == "__main__":
    unittest.main()
