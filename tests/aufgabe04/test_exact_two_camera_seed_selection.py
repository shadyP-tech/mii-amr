from dataclasses import FrozenInstanceError, replace
import json
import unittest

from scripts.aufgabe04.navigation.coverage.exact_two_camera_seed_selection import (
    SELECTION_MODE_BOUNDED_INSPECTION_POOL,
    SELECTION_MODE_NOT_READY,
    SELECTION_MODE_STRICT_EXACT,
    SELECTION_MODE_STRICT_EXACT_BOUNDARY_AUDIT_ONLY,
    select_exact_two_camera_seed_candidates,
)


class ExactTwoCameraSeedSelectionTest(unittest.TestCase):
    def _select(
        self,
        *,
        expected=5,
        strict=(),
        boundary=(),
    ):
        return select_exact_two_camera_seed_candidates(
            expected_stand_count=expected,
            static_map_admitted_candidate_uids=strict,
            boundary_provisional_candidate_uids=boundary,
        )

    def test_goal_sized_strict_population_still_includes_all_boundary_candidates(self):
        decision = self._select(
            strict=tuple(f"candidate_{i:02d}" for i in range(5, 0, -1)),
            boundary=("candidate_07", "candidate_06"),
        )
        self.assertTrue(decision.ready)
        self.assertEqual(decision.selection_mode, SELECTION_MODE_BOUNDED_INSPECTION_POOL)
        self.assertEqual(decision.selected_candidate_uids,
                         tuple(f"candidate_{i:02d}" for i in range(1, 8)))
        self.assertEqual(decision.boundary_fill_candidate_uids, ("candidate_06", "candidate_07"))
        self.assertEqual(decision.boundary_audit_only_candidate_uids, ())
        self.assertEqual(decision.excluded_candidate_uids, ())
        self.assertEqual(decision.expected_stand_count, 5)
        self.assertEqual(decision.inspection_pool_limit, 10)
        self.assertFalse(decision.motion_authorized)

    def test_exact_strict_population_without_boundary_uses_strict_mode(self):
        decision = self._select(
            expected=2,
            strict=("candidate_02", "candidate_01"),
        )

        self.assertTrue(decision.ready)
        self.assertEqual(decision.selection_mode, SELECTION_MODE_BOUNDED_INSPECTION_POOL)
        self.assertEqual(
            decision.selected_candidate_uids,
            ("candidate_01", "candidate_02"),
        )
        self.assertEqual(decision.excluded_candidate_uids, ())

    def test_boundary_fills_only_the_exact_strict_deficit(self):
        decision = self._select(
            strict=("candidate_03", "candidate_01", "candidate_02"),
            boundary=("candidate_05", "candidate_04"),
        )

        self.assertTrue(decision.ready)
        self.assertEqual(
            decision.selection_mode,
            SELECTION_MODE_BOUNDED_INSPECTION_POOL,
        )
        self.assertEqual(
            decision.selected_candidate_uids,
            (
                "candidate_01",
                "candidate_02",
                "candidate_03",
                "candidate_04",
                "candidate_05",
            ),
        )
        self.assertEqual(
            decision.boundary_fill_candidate_uids,
            ("candidate_04", "candidate_05"),
        )
        self.assertEqual(decision.boundary_audit_only_candidate_uids, ())
        self.assertEqual(decision.excluded_candidate_uids, ())
        self.assertTrue(
            set(decision.strict_static_map_admitted_candidate_uids).issubset(
                decision.selected_candidate_uids
            )
        )

    def test_deficit_fill_selection_uses_global_canonical_uid_order(self):
        decision = self._select(
            expected=3,
            strict=("candidate_30", "candidate_20"),
            boundary=("candidate_10",),
        )

        self.assertTrue(decision.ready)
        self.assertEqual(
            decision.strict_static_map_admitted_candidate_uids,
            ("candidate_20", "candidate_30"),
        )
        self.assertEqual(
            decision.boundary_fill_candidate_uids,
            ("candidate_10",),
        )
        self.assertEqual(
            decision.selected_candidate_uids,
            ("candidate_10", "candidate_20", "candidate_30"),
        )

    def test_pool_cap_overflow_fails_closed_without_truncation(self):
        decision = self._select(
            expected=2,
            strict=("candidate_01", "candidate_02", "candidate_03"),
            boundary=("candidate_04", "candidate_05"),
        )

        self.assertFalse(decision.ready)
        self.assertEqual(
            decision.reasons,
            ("inspection_pool_candidate_count_exceeds_limit",),
        )
        self.assertEqual(decision.selection_mode, SELECTION_MODE_NOT_READY)
        self.assertEqual(decision.selected_candidate_uids, ())
        self.assertEqual(decision.boundary_fill_candidate_uids, ())
        self.assertEqual(
            decision.excluded_candidate_uids,
            ("candidate_01", "candidate_02", "candidate_03", "candidate_04", "candidate_05"),
        )

    def test_usable_population_below_expected_fails_closed(self):
        decision = self._select(
            expected=5,
            strict=("candidate_01", "candidate_02"),
            boundary=("candidate_03", "candidate_04"),
        )

        self.assertFalse(decision.ready)
        self.assertEqual(
            decision.reasons,
            ("usable_candidate_count_below_expected",),
        )
        self.assertEqual(decision.usable_candidate_count, 4)
        self.assertEqual(decision.selected_candidate_uids, ())

    def test_strict_and_boundary_surplus_within_cap_remain_inspectable(self):
        for strict_count in (3, 6, 10):
            with self.subTest(strict_count=strict_count):
                total = max(6, strict_count)
                decision = self._select(
                    strict=tuple(f"candidate_{i:02d}" for i in range(strict_count)),
                    boundary=tuple(f"candidate_{i:02d}" for i in range(strict_count, total)),
                )
                self.assertTrue(decision.ready)
                self.assertEqual(decision.selected_candidate_count, total)
                self.assertEqual(decision.excluded_candidate_uids, ())

    def test_pool_policy_and_cap_forgery_are_rejected(self):
        decision = self._select(expected=1, strict=("candidate_01",))
        for changes in ({"inspection_pool_limit": 3},
                        {"inspection_pool_policy_id": "take_best_five"},
                        {"inspection_pool_limit": True}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "pool policy"):
                replace(decision, **changes)

    def test_missing_noninteger_and_nonpositive_expected_counts_fail_closed(self):
        cases = (
            (None, "expected_stand_count_missing"),
            (True, "expected_stand_count_not_integer"),
            (5.0, "expected_stand_count_not_integer"),
            ("5", "expected_stand_count_not_integer"),
            (0, "expected_stand_count_not_positive"),
            (-1, "expected_stand_count_not_positive"),
        )
        for expected, reason in cases:
            with self.subTest(expected=expected):
                decision = self._select(
                    expected=expected,
                    strict=("candidate_01",),
                )
                self.assertFalse(decision.ready)
                self.assertIn(reason, decision.reasons)
                self.assertEqual(decision.selected_candidate_uids, ())
                self.assertFalse(decision.motion_authorized)

    def test_malformed_uid_partitions_fail_closed_with_raw_text_evidence(self):
        cases = (
            (
                "candidate_01",
                (),
                "strict_candidate_uids_not_sequence",
            ),
            (
                None,
                (),
                "strict_candidate_uids_missing",
            ),
            (
                ("candidate_01", 2),
                (),
                "malformed_strict_candidate_uid",
            ),
            (
                ("candidate_01", "bad/candidate"),
                (),
                "malformed_strict_candidate_uid",
            ),
            (
                ("candidate_01",),
                "candidate_02",
                "boundary_candidate_uids_not_sequence",
            ),
        )
        for strict, boundary, reason in cases:
            with self.subTest(reason=reason):
                decision = self._select(
                    expected=1,
                    strict=strict,
                    boundary=boundary,
                )
                self.assertFalse(decision.ready)
                self.assertIn(reason, decision.reasons)
                self.assertEqual(decision.selected_candidate_uids, ())
                json.dumps(decision.to_evidence_dict(), sort_keys=True)

    def test_duplicate_and_overlapping_uids_fail_closed(self):
        cases = (
            (
                ("candidate_01", "candidate_01"),
                (),
                "duplicate_strict_candidate_uid",
            ),
            (
                ("candidate_01",),
                ("candidate_02", "candidate_02"),
                "duplicate_boundary_candidate_uid",
            ),
            (
                ("candidate_01",),
                ("candidate_01",),
                "candidate_uid_partition_overlap",
            ),
        )
        for strict, boundary, reason in cases:
            with self.subTest(reason=reason):
                decision = self._select(
                    expected=1,
                    strict=strict,
                    boundary=boundary,
                )
                self.assertFalse(decision.ready)
                self.assertIn(reason, decision.reasons)
                self.assertEqual(decision.selected_candidate_uids, ())

    def test_order_is_canonical_evidence_is_json_ready_and_decision_is_frozen(self):
        forward = self._select(
            expected=3,
            strict=("candidate_02", "candidate_01"),
            boundary=("candidate_03",),
        )
        reversed_input = self._select(
            expected=3,
            strict=("candidate_01", "candidate_02"),
            boundary=("candidate_03",),
        )

        self.assertEqual(forward, reversed_input)
        payload = forward.to_evidence()
        self.assertEqual(payload["selected_candidate_uids"], [
            "candidate_01",
            "candidate_02",
            "candidate_03",
        ])
        self.assertEqual(payload["counts"]["selected"], 3)
        self.assertFalse(payload["motion_authorized"])
        json.dumps(payload, sort_keys=True)
        with self.assertRaises(FrozenInstanceError):
            forward.ready = False

    def test_manual_mode_and_duplicate_partition_forgery_are_rejected(self):
        decision = self._select(
            expected=1,
            strict=("candidate_01",),
            boundary=("candidate_02",),
        )

        with self.assertRaisesRegex(ValueError, "mode differs"):
            replace(decision, selection_mode=SELECTION_MODE_STRICT_EXACT)
        with self.assertRaisesRegex(ValueError, "canonical and unique"):
            replace(
                decision,
                boundary_audit_only_candidate_uids=(
                    "candidate_02",
                    "candidate_02",
                ),
            )

    def test_selector_is_identity_and_count_only_support_remains_caller_owned(self):
        decision = self._select(
            expected=2,
            strict=("caller_candidate_02", "caller_candidate_01"),
        )

        self.assertTrue(decision.ready)
        payload_text = json.dumps(decision.to_evidence_dict(), sort_keys=True)
        self.assertNotIn("support", payload_text)
        self.assertEqual(
            decision.selected_candidate_uids,
            ("caller_candidate_01", "caller_candidate_02"),
        )


if __name__ == "__main__":
    unittest.main()
