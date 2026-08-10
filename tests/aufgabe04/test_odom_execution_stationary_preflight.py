import unittest

from scripts.aufgabe04.navigation.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.navigation.ros_preflight import RosPreflightResult
from scripts.aufgabe04.navigation.run_single_station_segment import (
    _admit_stationary_map_from_odom_window,
)


def _sample(
    index: int,
    *,
    x_m: float = 0.0,
    stamp_sec: float | None = None,
    receipt_time_sec: float | None = None,
) -> dict[str, object]:
    stamp = 10.0 + index if stamp_sec is None else stamp_sec
    receipt = 20.0 + index if receipt_time_sec is None else receipt_time_sec
    stamp_nanoseconds = round(stamp * 1_000_000_000)
    receipt_nanoseconds = round(receipt * 1_000_000_000)
    capture_nanoseconds = receipt_nanoseconds + 10_000_000
    return {
        "amcl_sample_index": index,
        "source": "direct_dynamic_tf",
        "target_frame": "map",
        "source_frame": "odom",
        "observed_target_frame": "map",
        "observed_source_frame": "odom",
        "stamp_sec": stamp,
        "stamp_nanoseconds": stamp_nanoseconds,
        "receipt_time_sec": receipt,
        "receipt_time_nanoseconds": receipt_nanoseconds,
        "capture_time_sec": capture_nanoseconds / 1_000_000_000.0,
        "capture_time_nanoseconds": capture_nanoseconds,
        "x_m": x_m,
        "y_m": 0.0,
        "yaw_rad": 0.0,
    }


def _preflight(samples: list[dict[str, object]]) -> RosPreflightResult:
    return RosPreflightResult(
        ok=True,
        failures=[],
        observations=[],
        runtime_config={},
        stationary_map_from_odom_samples=samples,
    )


class OdomExecutionStationaryPreflightTests(unittest.TestCase):
    def test_final_lookup_is_the_exact_admitted_certificate_transform(self):
        final = PlanarTransform2D(0.02, 0.0, 0.01)

        admitted, evidence = _admit_stationary_map_from_odom_window(
            _preflight([_sample(0), _sample(1, x_m=0.01)]),
            map_frame="map",
            odom_frame="odom",
            final_map_from_odom=final,
            final_stamp_sec=12.0,
            final_capture_time_sec=22.0,
            max_translation_drift_m=0.05,
            max_yaw_drift_rad=0.02,
        )

        self.assertEqual(admitted, final)
        self.assertEqual(evidence["frozen_map_from_odom"]["x_m"], 0.02)
        self.assertEqual(len(evidence["sample_comparisons"]), 3)
        self.assertEqual(
            evidence["sample_provenance"][-1]["source"],
            "final_preflight_tf_lookup",
        )

    def test_duplicate_direct_tf_receipt_or_stamp_is_rejected(self):
        cases = (
            (
                [_sample(0), _sample(1, receipt_time_sec=20.0)],
                "strictly newer direct-TF receipts",
            ),
            (
                [_sample(0), _sample(1, stamp_sec=10.0)],
                "strictly newer direct-TF stamps",
            ),
        )
        for samples, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(
                ValueError,
                reason,
            ):
                _admit_stationary_map_from_odom_window(
                    _preflight(samples),
                    map_frame="map",
                    odom_frame="odom",
                    final_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
                    final_stamp_sec=12.0,
                    final_capture_time_sec=22.0,
                    max_translation_drift_m=0.05,
                    max_yaw_drift_rad=0.1,
                )

    def test_missing_window_and_unstable_final_lookup_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            _admit_stationary_map_from_odom_window(
                _preflight([]),
                map_frame="map",
                odom_frame="odom",
                final_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
                final_stamp_sec=12.0,
                final_capture_time_sec=22.0,
                max_translation_drift_m=0.05,
                max_yaw_drift_rad=0.1,
            )
        with self.assertRaisesRegex(ValueError, "window rejected"):
            _admit_stationary_map_from_odom_window(
                _preflight([_sample(0), _sample(1, x_m=0.01)]),
                map_frame="map",
                odom_frame="odom",
                final_map_from_odom=PlanarTransform2D(0.20, 0.0, 0.0),
                final_stamp_sec=12.0,
                final_capture_time_sec=22.0,
                max_translation_drift_m=0.05,
                max_yaw_drift_rad=0.1,
            )

    def test_final_lookup_may_not_predate_stationary_window(self):
        with self.assertRaisesRegex(ValueError, "predates"):
            _admit_stationary_map_from_odom_window(
                _preflight([_sample(0), _sample(1)]),
                map_frame="map",
                odom_frame="odom",
                final_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
                final_stamp_sec=10.5,
                final_capture_time_sec=20.5,
                max_translation_drift_m=0.05,
                max_yaw_drift_rad=0.1,
            )


if __name__ == "__main__":
    unittest.main()
