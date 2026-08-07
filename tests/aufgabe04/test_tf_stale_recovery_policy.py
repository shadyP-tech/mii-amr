import json
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.tf_stale_recovery_policy import (  # noqa: E402
    OdomStationaritySample,
    StationarityLimits,
    TfEdgeSample,
    evaluate_recovery_acceptance,
    evaluate_recovery_eligibility,
    evaluate_stationarity,
)


NOW_SEC = 10.0


def edge(parent: str, child: str, stamp_sec: float | None) -> TfEdgeSample:
    return TfEdgeSample(parent, child, stamp_sec)


def odom_sample(
    callback_count: int,
    stamp_sec: float,
    *,
    x_m: float = 1.0,
    y_m: float = 2.0,
    yaw_rad: float = 0.2,
    linear_x_mps: float = 0.0,
    angular_z_radps: float = 0.0,
) -> OdomStationaritySample:
    return OdomStationaritySample(
        callback_count=callback_count,
        stamp_sec=stamp_sec,
        x_m=x_m,
        y_m=y_m,
        yaw_rad=yaw_rad,
        linear_x_mps=linear_x_mps,
        angular_z_radps=angular_z_radps,
    )


def stationary_decision():
    return evaluate_stationarity(
        odom_sample(40, 9.80),
        odom_sample(41, 9.90, x_m=1.001, yaw_rad=0.201),
        now_sec=NOW_SEC,
    )


def eligibility(
    *,
    localization_source: str = "amcl",
    use_sim_time: bool = False,
    composed_before_stamp: float | None = 8.70,
    composed_retry_stamp: float | None = 8.70,
    map_before_stamp: float | None = 9.90,
    map_retry_stamp: float | None = 9.90,
    odom_retry_stamp: float | None = 9.95,
):
    return evaluate_recovery_eligibility(
        localization_source=localization_source,
        use_sim_time=use_sim_time,
        composed_before=edge("map", "base_footprint", composed_before_stamp),
        composed_retry=edge("map", "base_footprint", composed_retry_stamp),
        map_to_odom_before=edge("map", "odom", map_before_stamp),
        map_to_odom_retry=edge("map", "odom", map_retry_stamp),
        odom_to_base_retry=edge("odom", "base_footprint", odom_retry_stamp),
        now_sec=NOW_SEC,
        max_tf_age_sec=1.0,
    )


class StationarityPolicyTest(unittest.TestCase):
    def test_two_fresh_distinct_stationary_samples_are_accepted(self):
        decision = stationary_decision()

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "odom_stationary")
        self.assertTrue(decision.callback_advanced)
        self.assertTrue(decision.stamp_advanced)
        json.dumps(decision.to_log_dict(), allow_nan=False)

    def test_callback_stamp_pose_and_twist_are_independent_gates(self):
        first = odom_sample(40, 9.80)
        cases = {
            "callback": (
                odom_sample(40, 9.90),
                "odom_callback_not_advanced",
            ),
            "stamp": (
                odom_sample(41, 9.80),
                "odom_stamp_not_advanced",
            ),
            "separation": (
                odom_sample(41, 9.85),
                "odom_sample_separation_too_short",
            ),
            "translation": (
                odom_sample(41, 9.90, x_m=1.02),
                "odom_translation_not_stationary",
            ),
            "yaw": (
                odom_sample(41, 9.90, yaw_rad=0.25),
                "odom_yaw_not_stationary",
            ),
            "linear_twist": (
                odom_sample(41, 9.90, linear_x_mps=0.011),
                "odom_linear_twist_not_stationary",
            ),
            "angular_twist": (
                odom_sample(41, 9.90, angular_z_radps=0.051),
                "odom_angular_twist_not_stationary",
            ),
        }

        for name, (second, expected_reason) in cases.items():
            with self.subTest(name=name):
                decision = evaluate_stationarity(first, second, now_sec=NOW_SEC)
                self.assertFalse(decision.accepted)
                self.assertIn(expected_reason, decision.reasons)

    def test_stale_and_future_stationarity_samples_fail_closed(self):
        stale = evaluate_stationarity(
            odom_sample(40, 9.30),
            odom_sample(41, 9.90),
            now_sec=NOW_SEC,
        )
        future = evaluate_stationarity(
            odom_sample(40, 10.10),
            odom_sample(41, 10.20),
            now_sec=NOW_SEC,
        )

        self.assertIn("first_odom_sample_stale", stale.reasons)
        self.assertIn("first_odom_sample_future", future.reasons)
        self.assertIn("second_odom_sample_future", future.reasons)

    def test_evidence_and_limits_are_frozen_finite_and_json_safe(self):
        sample = odom_sample(1, 1.0)
        limits = StationarityLimits()
        with self.assertRaises(FrozenInstanceError):
            sample.x_m = 3.0
        with self.assertRaises(FrozenInstanceError):
            limits.max_translation_m = 0.2
        with self.assertRaises(ValueError):
            odom_sample(1, 1.0, x_m=float("nan"))
        with self.assertRaises(ValueError):
            TfEdgeSample("map", "map", 1.0)
        with self.assertRaises(ValueError):
            StationarityLimits(min_sample_separation_sec=0.6)
        json.dumps(sample.to_log_dict(), allow_nan=False)
        json.dumps(limits.to_log_dict(), allow_nan=False)
        json.dumps(edge("map", "odom", None).to_log_dict(now_sec=1.0))


class RecoveryEligibilityTest(unittest.TestCase):
    def test_real_amcl_stale_composed_and_nonadvancing_map_edge_is_eligible(self):
        decision = eligibility()

        self.assertTrue(decision.accepted)
        self.assertEqual(
            decision.reason, "real_amcl_stale_edge_recovery_eligible"
        )
        self.assertEqual(decision.composed_retry_status, "stale")
        self.assertEqual(decision.map_to_odom_retry_status, "fresh")
        self.assertFalse(decision.map_to_odom_advanced)
        json.dumps(decision.to_log_dict(), allow_nan=False)

    def test_recovery_is_amcl_only_and_real_time_only(self):
        for source, use_sim_time, expected in (
            ("odom", False, "localization_source_not_amcl"),
            ("slam_toolbox", False, "localization_source_not_amcl"),
            ("amcl", True, "sim_time_recovery_forbidden"),
        ):
            with self.subTest(source=source, use_sim_time=use_sim_time):
                decision = eligibility(
                    localization_source=source, use_sim_time=use_sim_time
                )
                self.assertFalse(decision.accepted)
                self.assertIn(expected, decision.reasons)

    def test_stale_odom_edge_is_terminal(self):
        decision = eligibility(odom_retry_stamp=8.5)

        self.assertFalse(decision.accepted)
        self.assertIn("odom_to_base_retry_not_fresh:stale", decision.reasons)

    def test_stale_or_nonadvancing_map_edge_is_required(self):
        stale_advancing = eligibility(map_before_stamp=8.5, map_retry_stamp=8.8)
        fresh_advancing = eligibility(map_before_stamp=9.8, map_retry_stamp=9.9)

        self.assertTrue(stale_advancing.accepted)
        self.assertFalse(fresh_advancing.accepted)
        self.assertIn(
            "map_to_odom_not_stale_or_nonadvancing",
            fresh_advancing.reasons,
        )

    def test_regressing_transform_stamps_are_terminal(self):
        composed = eligibility(
            composed_before_stamp=8.8,
            composed_retry_stamp=8.7,
        )
        map_edge = eligibility(
            map_before_stamp=9.9,
            map_retry_stamp=9.8,
        )

        self.assertIn("composed_retry_stamp_regressed", composed.reasons)
        self.assertIn("map_to_odom_retry_stamp_regressed", map_edge.reasons)

    def test_amcl_direct_future_edge_has_a_separate_tolerance(self):
        admitted = eligibility(map_before_stamp=10.8, map_retry_stamp=10.8)
        too_future = eligibility(map_before_stamp=11.2, map_retry_stamp=11.2)

        self.assertTrue(admitted.accepted)
        self.assertEqual(admitted.map_to_odom_retry_status, "fresh")
        self.assertFalse(too_future.accepted)
        self.assertIn("map_to_odom_retry_future", too_future.reasons)

    def test_service_is_not_admitted_without_a_stale_composed_retry(self):
        fresh = eligibility(composed_retry_stamp=9.9)
        future = eligibility(composed_retry_stamp=10.1)
        unavailable = eligibility(composed_retry_stamp=None)

        self.assertIn("composed_retry_not_stale:fresh", fresh.reasons)
        self.assertIn("composed_retry_not_stale:future", future.reasons)
        self.assertIn(
            "composed_retry_not_stale:unavailable", unavailable.reasons
        )

    def test_missing_baseline_or_inconsistent_chain_is_not_evidence(self):
        missing_composed = eligibility(composed_before_stamp=None)
        missing_map = eligibility(map_before_stamp=None)
        inconsistent_chain = evaluate_recovery_eligibility(
            localization_source="amcl",
            use_sim_time=False,
            composed_before=edge("map", "base_footprint", 8.7),
            composed_retry=edge("map", "base_footprint", 8.7),
            map_to_odom_before=edge("map", "wrong_odom", 9.9),
            map_to_odom_retry=edge("map", "wrong_odom", 9.9),
            odom_to_base_retry=edge("odom", "base_footprint", 9.95),
            now_sec=NOW_SEC,
            max_tf_age_sec=1.0,
        )

        self.assertIn("composed_before_unavailable", missing_composed.reasons)
        self.assertIn("map_to_odom_before_unavailable", missing_map.reasons)
        self.assertIn(
            "tf_edge_topology_inconsistent", inconsistent_chain.reasons
        )


class RecoveryAcceptanceTest(unittest.TestCase):
    def accept(self, **overrides):
        values = {
            "eligibility": eligibility(
                map_before_stamp=10.70,
                map_retry_stamp=10.70,
            ),
            "composed_before": edge("map", "base_footprint", 8.70),
            "composed_recovered": edge("map", "base_footprint", 9.95),
            "map_to_odom_before": edge("map", "odom", 10.70),
            "map_to_odom_recovered": edge("map", "odom", 10.80),
            "odom_to_base_recovered": edge(
                "odom", "base_footprint", 9.98
            ),
            "stationarity": stationary_decision(),
            "scan_fresh": True,
            "odom_fresh": True,
            "exclusive_cmd_vel_owner": True,
            "now_sec": NOW_SEC,
            "max_tf_age_sec": 1.0,
        }
        values.update(overrides)
        return evaluate_recovery_acceptance(**values)

    def test_new_composed_and_advancing_future_skewed_amcl_edge_are_accepted(self):
        decision = self.accept()

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "stale_tf_recovery_accepted")
        self.assertTrue(decision.composed_strictly_newer)
        self.assertTrue(decision.map_to_odom_strictly_newer)
        self.assertEqual(decision.map_to_odom_recovered_status, "fresh")
        json.dumps(decision.to_log_dict(), allow_nan=False)

    def test_stale_future_and_nonadvancing_composed_are_rejected(self):
        for name, sample, expected in (
            (
                "stale",
                edge("map", "base_footprint", 8.80),
                "composed_recovered_not_fresh:stale",
            ),
            (
                "future",
                edge("map", "base_footprint", 10.03),
                "composed_recovered_not_fresh:future",
            ),
            (
                "nonadvancing",
                edge("map", "base_footprint", 8.70),
                "composed_transform_not_strictly_newer",
            ),
        ):
            with self.subTest(name=name):
                decision = self.accept(composed_recovered=sample)
                self.assertFalse(decision.accepted)
                self.assertIn(expected, decision.reasons)

    def test_advancing_amcl_and_fresh_odom_edges_are_mandatory(self):
        nonadvancing_map = self.accept(
            map_to_odom_recovered=edge("map", "odom", 10.70)
        )
        stale_odom = self.accept(
            odom_to_base_recovered=edge("odom", "base_footprint", 8.5)
        )

        self.assertIn(
            "map_to_odom_transform_not_strictly_newer",
            nonadvancing_map.reasons,
        )
        self.assertIn(
            "odom_to_base_recovered_not_fresh:stale", stale_odom.reasons
        )

    def test_sensor_owner_and_stationarity_gates_fail_independently(self):
        moving = evaluate_stationarity(
            odom_sample(40, 9.8),
            odom_sample(41, 9.9, linear_x_mps=0.02),
            now_sec=NOW_SEC,
        )
        cases = {
            "scan": ({"scan_fresh": False}, "scan_not_fresh"),
            "odom": ({"odom_fresh": False}, "odom_not_fresh"),
            "owner": (
                {"exclusive_cmd_vel_owner": False},
                "cmd_vel_owner_not_exclusive",
            ),
            "stationarity": (
                {"stationarity": moving},
                "stationarity_not_confirmed",
            ),
        }

        for name, (override, expected) in cases.items():
            with self.subTest(name=name):
                decision = self.accept(**override)
                self.assertFalse(decision.accepted)
                self.assertIn(expected, decision.reasons)

    def test_rejected_eligibility_cannot_be_bypassed(self):
        decision = self.accept(eligibility=eligibility(use_sim_time=True))

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "recovery_not_eligible")


if __name__ == "__main__":
    unittest.main()
