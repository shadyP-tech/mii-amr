from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.approach.camera_candidate_selection import (
    CameraCandidateSelectionConfig,
    NoFeasibleCameraCandidateError,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_compute import (
    compute_candidate_preapproach_plan,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_selection import (
    plan_and_select_camera_candidate,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.stations.candidate_snapshot import new_candidate_snapshot
from tests.aufgabe04 import test_candidate_preapproach_planning as fixtures
from tests.aufgabe04 import test_lidar_inspection_planning as lidar_fixtures
from tests.aufgabe04.test_candidate_route_uncertainty_selection import _plan


MODULE = "scripts.aufgabe04.navigation.approach.candidate_preapproach_selection"


class CameraCandidateTimeBudgetTest(unittest.TestCase):
    def _select(self, plans, config, *, support=None):
        helper = fixtures.CandidatePreapproachPlanningTest()
        snapshot = new_candidate_snapshot(
            snapshot_id="snapshot",
            created_unix_sec=3.0,
            planning_frame="map",
            map_bundle_sha256="c" * 64,
            candidates=tuple(
                helper._candidate(plan.candidate_uid, 0.50, 0.45)
                for plan in plans
            ),
        )
        plans_by_uid = {plan.candidate_uid: plan for plan in plans}
        with (
            patch(f"{MODULE}.load_candidate_planning_context", return_value=object()),
            patch(
                f"{MODULE}.compute_candidate_preapproach_plan",
                side_effect=lambda **kwargs: plans_by_uid[kwargs["candidate_uid"]],
            ),
        ):
            return plan_and_select_camera_candidate(
                map_yaml=Path("unused.yaml"),
                semantic_map_id="arena",
                plan=helper._plan(snapshot.map_bundle_sha256),
                snapshot=snapshot,
                current_pose=plans[0].start,
                unresolved=set(snapshot.candidate_uids),
                approach_offset_m=0.70,
                inflation_radius_m=0.25,
                candidate_transit_radius_m=0.31,
                physical_clearance=fixtures.PHYSICAL_CLEARANCE,
                selection_config=config,
                support_class_by_uid=support,
            )

    def test_slow_over_cap_candidate_is_rejected_before_ranking(self):
        start = Pose2D(0.0, 0.0, 0.0)
        slow = _plan("candidate_slow", (start, Pose2D(1.5, 0.0, 0.0)))
        short = _plan("candidate_short", (start, Pose2D(0.3, 0.0, 0.0)))
        support = {
            "candidate_slow": "multi_view",
            "candidate_short": "single_view_requires_camera_validation",
        }
        config = CameraCandidateSelectionConfig(0.01, 0.18, route_time_budget_enabled=True)

        selected = self._select((slow, short), config, support=support)

        self.assertEqual(selected.selected_candidate_uid, "candidate_short")
        rejected = selected.to_evidence()["rejected_candidates"][0]
        self.assertEqual(rejected["candidate_uid"], "candidate_slow")
        self.assertIn("waypoint timeout cap", rejected["failure_reason"])
        self.assertFalse(rejected["route_time_budget"]["accepted"])
        self.assertEqual(rejected["route_time_budget"]["controller"]["max_linear_mps"], 0.01)
        # Existing nominal/support ranking remains unchanged when admission is disabled.
        legacy = self._select(
            (slow, short), replace(config, route_time_budget_enabled=False), support=support
        )
        self.assertEqual(legacy.selected_candidate_uid, "candidate_slow")

    def test_all_over_cap_candidates_preserve_rejection_evidence(self):
        route = _plan("candidate_slow", (Pose2D(0.0, 0.0, 0.0), Pose2D(1.5, 0.0, 0.0)))
        with self.assertRaises(NoFeasibleCameraCandidateError) as raised:
            self._select(
                (route,),
                CameraCandidateSelectionConfig(0.01, 0.18, route_time_budget_enabled=True),
            )

        evidence = raised.exception.to_evidence()
        self.assertFalse(evidence["motion_authorized"])
        budget = evidence["rejected_candidates"][0]["route_time_budget"]
        self.assertFalse(budget["accepted"])
        self.assertEqual(budget["policy"]["maximum_timeout_sec"], 120.0)

    def test_audited_geometry_uses_configured_speeds_and_finite_budget(self):
        distance, alignment = 1.725129519, 2.2006
        route = _plan(
            "candidate_audit",
            (Pose2D(0.0, 0.0, -alignment), Pose2D(distance, 0.0, 0.0)),
        )
        timeouts = []
        for linear, angular in ((0.055, 0.18), (0.04, 0.12)):
            with self.subTest(linear=linear, angular=angular):
                selected = self._select(
                    (route,),
                    CameraCandidateSelectionConfig(linear, angular, route_time_budget_enabled=True),
                )
                evidence = selected.to_evidence()["ranked_candidates"][0]["route_time_budget"]
                controller = evidence["controller"]
                budget = evidence["budgets"][0]
                self.assertTrue(evidence["accepted"])
                self.assertTrue(controller["exact_vertex_pursuit"])
                self.assertEqual(controller["goal_tolerance_m"], 0.02)
                self.assertEqual(controller["terminal_goal_tolerance_m"], 0.03)
                self.assertAlmostEqual(controller["heading_tolerance_rad"], math.radians(3.0))
                self.assertAlmostEqual(budget["nominal_motion_sec"], distance / linear + alignment / angular)
                self.assertGreater(budget["timeout_sec"], 45.0)
                self.assertLessEqual(budget["timeout_sec"], 120.0)
                timeouts.append(budget["timeout_sec"])
        self.assertGreater(timeouts[1], timeouts[0])

    def test_over_cap_lidar_view_does_not_hide_feasible_other_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs, _ = lidar_fixtures.LidarInspectionPlanningTest().fixture(Path(tmp))
            kwargs["selection_config"] = CameraCandidateSelectionConfig(
                0.055, 0.18, route_time_budget_enabled=True
            )

            def preview(**arguments):
                prepared = compute_candidate_preapproach_plan(**arguments)
                if arguments.get("inspection_view_normal_rad", 0.0) > 0:
                    detour = _plan(
                        prepared.candidate_uid,
                        (prepared.start, Pose2D(10.0, 0.0, 0.0), prepared.selected_approach_pose),
                    )
                    # Make this side attractive to nominal ranking; only its exact
                    # segment geometry reveals that it exceeds the execution cap.
                    return replace(
                        prepared, result=detour.result, route_length_m=0.01,
                        turn_burden_rad=0.0, initial_turn_rad=0.0,
                    )
                return prepared

            with patch(f"{MODULE}.compute_candidate_preapproach_plan", side_effect=preview):
                selected = plan_and_select_camera_candidate(**kwargs)

        self.assertLess(selected.selected_plan.selected_approach_pose.y_m, 0.0)
        views = selected.to_evidence()["lidar_inspection_hints"]["candidate_views"]["candidate_1"]
        self.assertFalse(views["fallback"])
        self.assertFalse(views["views"][0]["accepted"])
        self.assertIn("waypoint timeout cap", views["views"][0]["reason"])
        self.assertTrue(views["views"][1]["accepted"])


if __name__ == "__main__":
    unittest.main()
