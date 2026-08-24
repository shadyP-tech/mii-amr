import math
import unittest
from dataclasses import FrozenInstanceError

from scripts.aufgabe04.navigation.approach.camera_candidate_selection import (
    CameraCandidateRouteOption,
    CameraCandidateSelectionConfig,
    CameraCandidateSelectionError,
    NoFeasibleCameraCandidateError,
    select_camera_candidate,
)


MULTI_VIEW = "multi_view"
SINGLE_VIEW = "single_view_requires_camera_validation"


def _option(
    candidate_uid: str,
    *,
    feasible: bool = True,
    failure_reason: str | None = None,
    route_length_m: float | None = 0.50,
    turn_burden_rad: float | None = 0.20,
    initial_turn_rad: float | None = 0.10,
    inside_requested_standoff: bool = False,
    support_class: str = MULTI_VIEW,
    confidence: float = 0.80,
    hit_count: int = 5,
) -> CameraCandidateRouteOption:
    return CameraCandidateRouteOption(
        candidate_uid=candidate_uid,
        feasible=feasible,
        failure_reason=failure_reason,
        route_length_m=route_length_m,
        turn_burden_rad=turn_burden_rad,
        initial_turn_rad=initial_turn_rad,
        inside_requested_standoff=inside_requested_standoff,
        support_class=support_class,
        confidence=confidence,
        hit_count=hit_count,
    )


def _blocked(candidate_uid: str, reason: str = "route_blocked") -> CameraCandidateRouteOption:
    return _option(
        candidate_uid,
        feasible=False,
        failure_reason=reason,
        route_length_m=None,
        turn_burden_rad=None,
        initial_turn_rad=None,
    )


class CameraCandidateSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CameraCandidateSelectionConfig(
            linear_speed_mps=0.10,
            angular_speed_radps=0.20,
        )

    def test_blocked_near_candidate_is_skipped_before_motion(self):
        blocked_near = _option(
            "stand-001",
            feasible=False,
            failure_reason="goal_cell_blocked",
            route_length_m=None,
            turn_burden_rad=None,
            initial_turn_rad=None,
            inside_requested_standoff=True,
        )
        reachable_farther = _option("stand-002", route_length_m=0.85)

        selection = select_camera_candidate(
            (blocked_near, reachable_farther), self.config
        )

        self.assertEqual(selection.selected_candidate_uid, "stand-002")
        self.assertEqual(
            tuple(item.candidate_uid for item in selection.rejected_candidates),
            ("stand-001",),
        )
        self.assertFalse(selection.motion_authorized)

    def test_turn_aware_cost_prefers_farther_candidate(self):
        near_with_reversal = _option(
            "stand-near",
            route_length_m=0.20,
            turn_burden_rad=math.pi,
            initial_turn_rad=math.radians(171.0),
            inside_requested_standoff=True,
        )
        farther_with_small_turn = _option(
            "stand-far",
            route_length_m=0.70,
            turn_burden_rad=0.15,
            initial_turn_rad=0.10,
        )

        selection = select_camera_candidate(
            (near_with_reversal, farther_with_small_turn), self.config
        )

        self.assertEqual(selection.selected_candidate_uid, "stand-far")
        self.assertTrue(selection.ranked_candidates[1].large_initial_turn)
        self.assertEqual(selection.ranked_candidates[1].risk_tier, 1)

    def test_single_view_candidate_remains_admitted_for_camera_validation(self):
        provisional = _option(
            "stand-single",
            route_length_m=0.30,
            support_class=SINGLE_VIEW,
            confidence=0.72,
            hit_count=3,
        )
        supported_but_slower = _option(
            "stand-multi",
            route_length_m=1.30,
            support_class=MULTI_VIEW,
            confidence=0.95,
            hit_count=12,
        )

        selection = select_camera_candidate(
            (supported_but_slower, provisional), self.config
        )

        self.assertEqual(selection.selected_candidate_uid, "stand-single")
        self.assertEqual(
            selection.ranked_candidates[0].option.support_class, SINGLE_VIEW
        )
        self.assertFalse(selection.to_evidence()["motion_authorized"])

    def test_large_initial_turn_risk_tier_is_lexicographic(self):
        fast_but_large_turn = _option(
            "stand-fast",
            route_length_m=0.05,
            turn_burden_rad=math.pi,
            initial_turn_rad=math.pi,
        )
        slower_without_large_turn = _option(
            "stand-steady",
            route_length_m=0.80,
            turn_burden_rad=0.10,
            initial_turn_rad=0.10,
        )
        high_angular_speed = CameraCandidateSelectionConfig(
            linear_speed_mps=0.10,
            angular_speed_radps=100.0,
        )

        selection = select_camera_candidate(
            (fast_but_large_turn, slower_without_large_turn), high_angular_speed
        )

        self.assertLess(
            selection.ranked_candidates[1].estimated_duration_sec,
            selection.ranked_candidates[0].estimated_duration_sec,
        )
        self.assertEqual(selection.selected_candidate_uid, "stand-steady")

    def test_equal_route_cost_uses_evidence_then_uid_as_stable_ties(self):
        weaker = _option(
            "stand-a",
            support_class=SINGLE_VIEW,
            confidence=0.99,
            hit_count=12,
        )
        stronger = _option(
            "stand-z", support_class=MULTI_VIEW, confidence=0.70, hit_count=2
        )

        evidence_selection = select_camera_candidate(
            (weaker, stronger), self.config
        )
        uid_selection = select_camera_candidate(
            (_option("stand-b"), _option("stand-a")), self.config
        )

        self.assertEqual(evidence_selection.selected_candidate_uid, "stand-z")
        self.assertEqual(uid_selection.selected_candidate_uid, "stand-a")

    def test_all_infeasible_fails_closed_with_rejection_evidence(self):
        with self.assertRaises(NoFeasibleCameraCandidateError) as raised:
            select_camera_candidate(
                (_blocked("stand-b"), _blocked("stand-a", "no_path")),
                self.config,
            )

        error = raised.exception
        self.assertEqual(error.code, "no_feasible_camera_candidate")
        self.assertEqual(
            tuple(item.candidate_uid for item in error.rejected_candidates),
            ("stand-a", "stand-b"),
        )
        evidence = error.to_evidence()
        self.assertFalse(evidence["motion_authorized"])
        self.assertEqual(evidence["error_code"], error.code)

    def test_result_is_immutable_and_json_ready(self):
        selection = select_camera_candidate(
            (_option("stand-b"), _blocked("stand-a")), self.config
        )

        with self.assertRaises(FrozenInstanceError):
            selection.selected_candidate_uid = "stand-a"  # type: ignore[misc]
        payload = selection.to_dict()
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["selected_candidate_uid"], "stand-b")
        self.assertEqual(payload["ranked_candidates"][0]["rank"], 1)
        self.assertEqual(
            payload["selection_policy"]["linear_speed_mps"], 0.10
        )

    def test_invalid_config_and_options_fail_validation(self):
        for kwargs in (
            {"linear_speed_mps": 0.0, "angular_speed_radps": 0.2},
            {"linear_speed_mps": 0.1, "angular_speed_radps": math.nan},
            {
                "linear_speed_mps": 0.1,
                "angular_speed_radps": 0.2,
                "large_initial_turn_threshold_rad": math.pi + 0.01,
            },
        ):
            with self.subTest(config=kwargs):
                with self.assertRaises(CameraCandidateSelectionError):
                    CameraCandidateSelectionConfig(**kwargs)

        invalid_builders = (
            lambda: _option("", feasible=True),
            lambda: _option("stand", route_length_m=-0.1),
            lambda: _option("stand", route_length_m=None),
            lambda: _option("stand", confidence=1.1),
            lambda: _option("stand", hit_count=-1),
            lambda: _option("stand", initial_turn_rad=math.pi + 0.01),
            lambda: _option("stand", feasible=True, failure_reason="blocked"),
            lambda: _option(
                "stand",
                feasible=False,
                failure_reason=None,
                route_length_m=None,
                turn_burden_rad=None,
                initial_turn_rad=None,
            ),
        )
        for build in invalid_builders:
            with self.subTest(builder=build):
                with self.assertRaises(CameraCandidateSelectionError):
                    build()

        with self.assertRaisesRegex(CameraCandidateSelectionError, "at least one"):
            select_camera_candidate((), self.config)
        with self.assertRaisesRegex(CameraCandidateSelectionError, "unique"):
            select_camera_candidate(
                (_option("stand"), _option("stand")), self.config
            )


if __name__ == "__main__":
    unittest.main()
