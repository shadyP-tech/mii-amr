import ast
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.prestart_localization_reseal import (
    FRESH_LOCALIZATION_RESEAL,
    TF_WARMUP_RETRY,
    evaluate_prestart_localization_reseal,
)


_CERTIFICATE_SHA256 = "a" * 64


def _continuity(*, reason: str) -> dict[str, object]:
    missing = reason == "map_from_odom_missing"
    return {
        "schema_version": 1,
        "accepted": False,
        "decision": "force_zero_reseal",
        "reason": reason,
        "fail_closed": True,
        "requires_zero_cycle": True,
        "requires_reseal": True,
        "threshold_semantics": (
            "accept_if_observed_less_than_or_equal_to_limit"
        ),
        "certificate_sha256": _CERTIFICATE_SHA256,
        "map_frame": "map",
        "odom_frame": "odom",
        "base_frame": "base_footprint",
        "frozen_map_from_odom": {
            "x_m": 0.0,
            "y_m": 0.0,
            "yaw_rad": 0.0,
        },
        "live_map_from_odom": (
            None
            if missing
            else {"x_m": 0.20, "y_m": 0.0, "yaw_rad": 0.04}
        ),
        "relative_translation_x_m": None if missing else 0.20,
        "relative_translation_y_m": None if missing else 0.0,
        "translation_drift_m": None if missing else 0.20,
        "relative_yaw_rad": None if missing else 0.04,
        "absolute_yaw_drift_rad": None if missing else 0.04,
        "max_translation_drift_m": 0.10,
        "max_yaw_drift_rad": 0.03,
        "validation_error": (
            "live map_from_odom is missing" if missing else None
        ),
    }


def _stop_details(
    *,
    reason: str = "map_from_odom_translation_and_yaw_drift",
    warning: str = "",
) -> dict[str, object]:
    return {
        "reason": "global localization consistency requires zero and reseal",
        "fault_code": "localization_reseal_required",
        "source": "global_consistency_monitor",
        "execution_phase": "before_motion",
        "phase": "initial_runtime_input_wait",
        "execution_pose_owner": "odom",
        "global_consistency_monitor": "amcl",
        "monitor_action": "FORCE_ZERO_RESEAL",
        "monitor_reason": "reseal_required",
        "monitor_warning": warning,
        "motion_published": False,
        "continuity": _continuity(reason=reason),
        "fail_closed": True,
    }


def _decision(details: object, *, status: object = "stopped", motion: object = False):
    return evaluate_prestart_localization_reseal(
        status=status,
        motion_published=motion,
        stop_details=details,
    )


class PrestartLocalizationResealTest(unittest.TestCase):
    def test_exact_translation_and_yaw_drift_is_eligible_without_motion_authority(self):
        decision = _decision(_stop_details())

        self.assertTrue(decision.eligible)
        self.assertEqual(decision.recovery_action, FRESH_LOCALIZATION_RESEAL)
        self.assertEqual(decision.execution_phase, "before_motion")
        self.assertFalse(decision.motion_published)
        self.assertEqual(
            decision.continuity_reason,
            "map_from_odom_translation_and_yaw_drift",
        )
        self.assertEqual(decision.monitor_warning, "")
        self.assertTrue(decision.requires_fresh_localization)
        self.assertTrue(decision.requires_new_route_certificate)
        self.assertFalse(decision.automatic_motion_authorized)
        self.assertFalse(decision.to_evidence()["automatic_motion_authorized"])
        self.assertEqual(decision.to_evidence()["schema_version"], 1)
        with self.assertRaises(FrozenInstanceError):
            decision.eligible = False  # type: ignore[misc]

    def test_each_exact_drift_reason_is_classified_from_observed_limits(self):
        cases = {
            "map_from_odom_translation_drift": (0.20, 0.02),
            "map_from_odom_yaw_drift": (0.08, 0.04),
            "map_from_odom_translation_and_yaw_drift": (0.20, 0.04),
        }
        for reason, (translation, yaw) in cases.items():
            with self.subTest(reason=reason):
                details = _stop_details(reason=reason)
                continuity = details["continuity"]
                assert isinstance(continuity, dict)
                continuity["relative_translation_x_m"] = translation
                continuity["translation_drift_m"] = translation
                continuity["relative_yaw_rad"] = yaw
                continuity["absolute_yaw_drift_rad"] = yaw

                decision = _decision(details)

                self.assertTrue(decision.eligible)
                self.assertEqual(
                    decision.recovery_action,
                    FRESH_LOCALIZATION_RESEAL,
                )
                self.assertEqual(decision.continuity_reason, reason)

    def test_each_allowed_tf_warning_is_distinct_warmup_class(self):
        warnings = (
            "stale_map_from_odom",
            "future_map_from_odom",
            "map_from_odom_lookup_failed: transform buffer warming",
        )
        for warning in warnings:
            with self.subTest(warning=warning):
                decision = _decision(
                    _stop_details(
                        reason="map_from_odom_missing",
                        warning=warning,
                    )
                )

                self.assertTrue(decision.eligible)
                self.assertEqual(decision.recovery_action, TF_WARMUP_RETRY)
                self.assertEqual(
                    decision.reason,
                    "prestart_tf_warmup_retry_required",
                )
                self.assertEqual(
                    decision.continuity_reason,
                    "map_from_odom_missing",
                )
                self.assertEqual(decision.monitor_warning, warning)
                self.assertTrue(decision.requires_fresh_localization)
                self.assertTrue(decision.requires_new_route_certificate)
                self.assertFalse(decision.automatic_motion_authorized)

    def test_wrong_status_motion_and_top_level_types_fail_closed(self):
        cases = (
            ("completed", False, _stop_details(), "outcome_not_stopped"),
            ("stopped", True, _stop_details(), "motion_already_published"),
            (
                "stopped",
                0,
                _stop_details(),
                "motion_published_not_boolean",
            ),
            ("stopped", False, None, "stop_details_not_mapping"),
        )
        for status, motion, details, reason in cases:
            with self.subTest(reason=reason):
                decision = _decision(details, status=status, motion=motion)
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.reason, reason)
                self.assertEqual(decision.execution_phase, "not_admitted")
                self.assertFalse(decision.automatic_motion_authorized)

    def test_exact_before_motion_phase_markers_are_required(self):
        for field, replacement in (
            ("execution_phase", "after_motion"),
            ("execution_phase", ""),
            ("phase", "before_motion_global_consistency_monitor"),
            ("phase", ""),
        ):
            with self.subTest(field=field, replacement=replacement):
                details = _stop_details()
                details[field] = replacement
                decision = _decision(details)
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.reason, f"invalid_{field}")

    def test_every_global_consistency_field_is_exact_and_conflicts_are_terminal(self):
        cases = {
            "reason": "some localization error",
            "fault_code": "odom_execution_admission_failed",
            "source": "front_sector",
            "execution_pose_owner": "amcl",
            "global_consistency_monitor": "none",
            "monitor_action": "LOG",
            "monitor_reason": "uncertainty_budget_exhausted",
            "fail_closed": 1,
        }
        for field, replacement in cases.items():
            with self.subTest(field=field):
                details = _stop_details()
                details[field] = replacement
                decision = _decision(details)
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.reason, f"invalid_{field}")

    def test_nested_motion_claim_cannot_conflict_with_outcome(self):
        missing = _stop_details()
        missing.pop("motion_published")
        decision = _decision(missing)
        self.assertFalse(decision.eligible)
        self.assertEqual(
            decision.reason,
            "conflicting_stop_details_motion_published",
        )

        for replacement in (True, 0, "false"):
            with self.subTest(replacement=replacement):
                details = _stop_details()
                details["motion_published"] = replacement
                decision = _decision(details)
                self.assertFalse(decision.eligible)
                self.assertEqual(
                    decision.reason,
                    "conflicting_stop_details_motion_published",
                )

    def test_warning_must_match_the_missing_transform_trigger_exactly(self):
        cases = (
            ("", "missing_tf_warmup_warning"),
            ("map_from_odom_missing", "unsupported_monitor_warning"),
            (
                "map_from_odom_lookup_failed:",
                "invalid_map_from_odom_lookup_warning",
            ),
            (
                "map_from_odom_lookup_failed:   ",
                "invalid_monitor_warning",
            ),
            ("stale_map_from_odom ", "invalid_monitor_warning"),
        )
        for warning, expected_reason in cases:
            with self.subTest(warning=warning):
                decision = _decision(
                    _stop_details(
                        reason="map_from_odom_missing",
                        warning=warning,
                    )
                )
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.reason, expected_reason)

    def test_warning_on_geometric_drift_is_rejected_instead_of_conflated(self):
        decision = _decision(
            _stop_details(warning="stale_map_from_odom")
        )
        self.assertFalse(decision.eligible)
        self.assertEqual(
            decision.reason,
            "unexpected_monitor_warning_for_drift",
        )

    def test_continuity_must_be_mapping_with_exact_fail_closed_contract(self):
        details = _stop_details()
        details["continuity"] = []
        decision = _decision(details)
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.reason, "continuity_not_mapping")

        replacements = {
            "schema_version": True,
            "accepted": 0,
            "requires_zero_cycle": 1,
            "requires_reseal": False,
            "decision": "force_zero_and_reseal",
            "fail_closed": 1,
            "threshold_semantics": "accept_if_less_than_limit",
        }
        for field, replacement in replacements.items():
            with self.subTest(field=field):
                details = _stop_details()
                continuity = details["continuity"]
                assert isinstance(continuity, dict)
                continuity[field] = replacement
                decision = _decision(details)
                self.assertFalse(decision.eligible)
                self.assertEqual(
                    decision.reason,
                    f"invalid_continuity_{field}",
                )

    def test_continuity_identity_and_static_geometry_are_complete(self):
        mutations = (
            ("certificate_sha256", "A" * 64, "invalid_continuity_certificate_sha256"),
            ("map_frame", "/map", "invalid_continuity_map_frame"),
            ("odom_frame", "map", "conflicting_continuity_frames"),
            (
                "frozen_map_from_odom",
                {"x_m": 0.0, "y_m": 0.0},
                "invalid_continuity_frozen_map_from_odom",
            ),
            (
                "max_translation_drift_m",
                -0.1,
                "invalid_continuity_max_translation_drift_m",
            ),
            (
                "max_yaw_drift_rad",
                float("nan"),
                "invalid_continuity_max_yaw_drift_rad",
            ),
        )
        for field, replacement, expected_reason in mutations:
            with self.subTest(field=field):
                details = _stop_details()
                continuity = details["continuity"]
                assert isinstance(continuity, dict)
                continuity[field] = replacement
                decision = _decision(details)
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.reason, expected_reason)

    def test_missing_transform_contract_cannot_carry_live_or_drift_values(self):
        mutations = (
            (
                "live_map_from_odom",
                {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                "invalid_missing_continuity_live_map_from_odom",
            ),
            (
                "translation_drift_m",
                0.0,
                "invalid_missing_continuity_translation_drift_m",
            ),
            (
                "validation_error",
                "",
                "invalid_missing_continuity_validation_error",
            ),
        )
        for field, replacement, expected_reason in mutations:
            with self.subTest(field=field):
                details = _stop_details(
                    reason="map_from_odom_missing",
                    warning="stale_map_from_odom",
                )
                continuity = details["continuity"]
                assert isinstance(continuity, dict)
                continuity[field] = replacement
                decision = _decision(details)
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.reason, expected_reason)

    def test_drift_contract_rejects_missing_malformed_and_inconsistent_values(self):
        cases = (
            (
                "live_map_from_odom",
                None,
                "invalid_drift_continuity_live_map_from_odom",
            ),
            (
                "validation_error",
                "live transform invalid",
                "invalid_drift_continuity_validation_error",
            ),
            (
                "translation_drift_m",
                float("inf"),
                "invalid_drift_continuity_translation_drift_m",
            ),
            (
                "relative_translation_y_m",
                0.2,
                "inconsistent_drift_continuity_translation",
            ),
            (
                "absolute_yaw_drift_rad",
                0.08,
                "inconsistent_drift_continuity_yaw",
            ),
        )
        for field, replacement, expected_reason in cases:
            with self.subTest(field=field):
                details = _stop_details()
                continuity = details["continuity"]
                assert isinstance(continuity, dict)
                continuity[field] = replacement
                decision = _decision(details)
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.reason, expected_reason)

    def test_claimed_drift_reason_must_match_exact_limit_comparison(self):
        details = _stop_details(reason="map_from_odom_translation_drift")
        continuity = details["continuity"]
        assert isinstance(continuity, dict)
        # Both components exceed their limits, contradicting translation-only.
        decision = _decision(details)
        self.assertFalse(decision.eligible)
        self.assertEqual(
            decision.reason,
            "inconsistent_drift_continuity_reason",
        )

        details = _stop_details(reason="map_from_odom_yaw_drift")
        continuity = details["continuity"]
        assert isinstance(continuity, dict)
        continuity["relative_translation_x_m"] = 0.10
        continuity["translation_drift_m"] = 0.10
        continuity["relative_yaw_rad"] = 0.03
        continuity["absolute_yaw_drift_rad"] = 0.03
        decision = _decision(details)
        self.assertFalse(decision.eligible)
        self.assertEqual(
            decision.reason,
            "inconsistent_drift_continuity_reason",
        )

    def test_empty_unknown_and_malformed_continuity_reasons_are_terminal(self):
        for replacement, expected_reason in (
            ("", "invalid_continuity_reason"),
            (None, "invalid_continuity_reason"),
            ("map_from_odom_malformed", "unsupported_continuity_reason"),
            ("temporary_tf_issue", "unsupported_continuity_reason"),
        ):
            with self.subTest(replacement=replacement):
                details = _stop_details()
                continuity = details["continuity"]
                assert isinstance(continuity, dict)
                continuity["reason"] = replacement
                decision = _decision(details)
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.reason, expected_reason)

    def test_module_remains_ros_and_motion_edge_free(self):
        module_path = (
            Path(__file__).parents[2]
            / "scripts"
            / "aufgabe04"
            / "navigation"
            / "prestart_localization_reseal.py"
        )
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_roots.update(
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        self.assertTrue(
            {"rclpy", "geometry_msgs", "tf2_ros", "subprocess"}.isdisjoint(
                imported_roots
            )
        )
        self.assertNotIn("cmd_vel", source)
        self.assertNotIn("input(", source)


if __name__ == "__main__":
    unittest.main()
