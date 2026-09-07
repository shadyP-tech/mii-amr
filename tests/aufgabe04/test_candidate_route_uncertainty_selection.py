from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.approach.camera_candidate_selection import (
    CameraCandidateRouteOption,
    CameraCandidateSelectionConfig,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_models import (
    CandidatePreapproachPlan,
)
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import (
    CandidateRouteUncertaintyContext,
    NoUncertaintyAdmittedCameraCandidateError,
    select_uncertainty_admitted_camera_candidate,
    validate_candidate_route_uncertainty_selection_binding,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
    route_uncertainty_admission_evidence_sha256,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.models import (
    GridCell,
    PlanningDiagnostics,
    Pose2D,
    Route,
    RoutePoint,
)
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.global_planner import PlanRouteResult
from scripts.aufgabe04.navigation.planning.map_io import (
    CELL_FREE,
    MapMetadata,
    OccupancyGrid,
)


MULTI_VIEW = "multi_view"
SINGLE_VIEW = "single_view_requires_camera_validation"
START = Pose2D(-1.0, 0.0, 0.0)


def _costmap(*, arena_width_m: float = 2.0) -> Costmap:
    metadata = MapMetadata(
        yaml_path=Path("map.yaml"),
        image_path=Path("map.pgm"),
        resolution=0.05,
        origin=(-2.0, -1.0, 0.0),
        negate=0,
        occupied_thresh=0.65,
        free_thresh=0.20,
        mode="trinary",
    )
    grid = OccupancyGrid(
        metadata=metadata,
        width=80,
        height=40,
        cells=tuple(tuple([CELL_FREE] * 80) for _ in range(40)),
    )
    return Costmap.from_occupancy_grid(grid).with_arena_bounds(
        ArenaBounds(length_m=3.0, width_m=arena_width_m)
    )


def _uncertainty() -> CandidateRouteUncertaintyContext:
    config = RouteUncertaintyAdmissionConfig(
        robot_radius_m=0.20,
        collision_margin_m=0.10,
        fixed_odom_tracking_bound_m=0.05,
        empirical_odom_drift_bound_m=0.02,
        braking_latency_distance_m=0.02,
        localization_sigma_multiplier=2.0,
        heading_sigma_rad=0.0,
        heading_lever_arm_m=0.20,
        sampling_spacing_m=0.005,
        heading_reference_x_m=START.x_m,
        heading_reference_y_m=START.y_m,
    )
    return CandidateRouteUncertaintyContext(
        covariance=PlanarCovariance(0.0, 0.0, 0.0),
        admission_config=config,
        source_evidence={
            "source_preplanning_localization_sha256": "a" * 64,
            "selection_only": True,
            "motion_authorized": False,
        },
    )


def _route(poses: tuple[Pose2D, ...]) -> Route:
    points = []
    cumulative_m = 0.0
    previous = None
    for index, pose in enumerate(poses):
        segment_m = (
            0.0
            if previous is None
            else math.hypot(
                pose.x_m - previous.x_m,
                pose.y_m - previous.y_m,
            )
        )
        cumulative_m += segment_m
        points.append(
            RoutePoint(
                index=index,
                cell=GridCell(index, 0),
                pose=pose,
                segment_length_m=segment_m,
                cumulative_length_m=cumulative_m,
            )
        )
        previous = pose
    return Route(
        points=tuple(points),
        requested_start=poses[0],
        requested_goal=poses[-1],
        snapped_start=poses[0],
        snapped_goal=poses[-1],
        length_m=cumulative_m,
    )


def _plan(
    candidate_uid: str,
    poses: tuple[Pose2D, ...],
) -> CandidatePreapproachPlan:
    route = _route(poses)
    result = PlanRouteResult(
        route=route,
        diagnostics=PlanningDiagnostics(
            status="ok",
            route_length_m=route.length_m,
        ),
    )
    return CandidatePreapproachPlan(
        candidate_uid=candidate_uid,
        candidate_snapshot_sha256="b" * 64,
        start=poses[0],
        approach_offset_m=0.70,
        inflation_radius_m=0.25,
        candidate_transit_radius_m=0.31,
        approach_bearing_rad=0.0,
        approach_bearing_mode="robot_to_stand",
        dry_run=SimpleNamespace(metadata={"map_bundle_sha256": "c" * 64}),
        result=result,
        connector=SimpleNamespace(),
        smoothing=SimpleNamespace(),
        selected_approach_pose=poses[-1],
        terminal_yaw_rad=poses[-1].yaw_rad,
        route_length_m=route.length_m,
        initial_turn_rad=0.0,
        turn_burden_rad=0.0,
        distance_to_stand_m=0.70,
        endpoint_standoff_m=0.70,
        inside_requested_standoff=False,
        minimum_active_standoff_m=0.32,
        minimum_candidate_transit_radius_m=0.31,
        minimum_static_inflation_m=0.25,
        goal_cell_selection=None,
    )


def _option(
    candidate_uid: str,
    *,
    route_length_m: float,
    support_class: str = MULTI_VIEW,
    confidence: float = 0.90,
) -> CameraCandidateRouteOption:
    return CameraCandidateRouteOption(
        candidate_uid=candidate_uid,
        feasible=True,
        failure_reason=None,
        route_length_m=route_length_m,
        turn_burden_rad=0.10,
        initial_turn_rad=0.10,
        inside_requested_standoff=False,
        support_class=support_class,
        confidence=confidence,
        hit_count=5,
    )


def _select(
    plans: tuple[CandidatePreapproachPlan, ...],
    options: tuple[CameraCandidateRouteOption, ...],
    *,
    costmap: Costmap | None = None,
):
    return select_uncertainty_admitted_camera_candidate(
        base_costmap=_costmap() if costmap is None else costmap,
        options=options,
        plans_by_uid={plan.candidate_uid: plan for plan in plans},
        selection_config=CameraCandidateSelectionConfig(
            linear_speed_mps=0.10,
            angular_speed_radps=0.20,
        ),
        uncertainty=_uncertainty(),
    )


class CandidateRouteUncertaintySelectionTest(unittest.TestCase):
    def test_negative_geometric_winner_is_skipped_for_admitted_alternative(self):
        near_wall = _plan(
            "candidate-top",
            (
                START,
                Pose2D(-0.9, 0.75, 0.0),
                Pose2D(-0.5, 0.75, 0.0),
            ),
        )
        center = _plan(
            "candidate-alternative",
            (START, Pose2D(0.5, 0.0, 0.0)),
        )

        selected = _select(
            (near_wall, center),
            (
                _option("candidate-top", route_length_m=0.10),
                _option("candidate-alternative", route_length_m=2.0),
            ),
        )

        evidence = selected.to_evidence()
        uncertainty = evidence["route_uncertainty_selection"]
        self.assertEqual(
            uncertainty["geometric_candidate_selection"][
                "selected_candidate_uid"
            ],
            "candidate-top",
        )
        self.assertEqual(
            selected.selected_candidate_uid,
            "candidate-alternative",
        )
        self.assertIs(selected.selected_plan, center)
        self.assertEqual(
            tuple(
                item.candidate_uid
                for item in selected.selection.rejected_candidates
            ),
            ("candidate-top",),
        )
        self.assertFalse(selected.motion_authorized)
        self.assertFalse(evidence["motion_authorized"])

    def test_existing_ranking_is_preserved_among_admitted_routes(self):
        faster_single = _plan(
            "candidate-single",
            (START, Pose2D(0.0, 0.20, 0.0)),
        )
        slower_multi = _plan(
            "candidate-multi",
            (START, Pose2D(0.50, 0.0, 0.0)),
        )

        selected = _select(
            (faster_single, slower_multi),
            (
                _option(
                    "candidate-single",
                    route_length_m=0.10,
                    support_class=SINGLE_VIEW,
                    confidence=0.99,
                ),
                _option(
                    "candidate-multi",
                    route_length_m=2.0,
                    support_class=MULTI_VIEW,
                    confidence=0.60,
                ),
            ),
        )

        self.assertEqual(selected.selected_candidate_uid, "candidate-multi")
        self.assertEqual(
            tuple(
                row.candidate_uid
                for row in selected.selection.ranked_candidates
            ),
            ("candidate-multi", "candidate-single"),
        )
        self.assertTrue(all(item.accepted for item in selected.evaluations))

    def test_admission_binds_the_terminal_yaw_written_to_route_csv(self):
        raw = _plan(
            "candidate-terminal-yaw",
            (START, Pose2D(0.5, 0.0, float("nan"))),
        )
        prepared = replace(
            raw,
            terminal_yaw_rad=0.75,
            selected_approach_pose=Pose2D(0.5, 0.0, 0.75),
        )

        selected = _select(
            (prepared,),
            (_option("candidate-terminal-yaw", route_length_m=1.5),),
        )

        route_evidence = selected.evaluations[0].admission.to_evidence_dict()[
            "route"
        ]
        self.assertEqual(route_evidence["poses"][-1]["yaw_rad"], 0.75)
        self.assertEqual(route_evidence["poses"][-1]["yaw_mode"], "constrained")
        validate_candidate_route_uncertainty_selection_binding(
            selected.to_evidence(),
            prepared,
        )

    def test_all_uncertainty_rejected_routes_fail_closed_with_hashed_evidence(self):
        first = _plan("candidate-a", (START, Pose2D(0.0, 0.0, 0.0)))
        second = _plan("candidate-b", (START, Pose2D(0.5, 0.0, 0.0)))

        with self.assertRaises(
            NoUncertaintyAdmittedCameraCandidateError
        ) as captured:
            _select(
                (first, second),
                (
                    _option("candidate-a", route_length_m=1.0),
                    _option("candidate-b", route_length_m=1.5),
                ),
                costmap=_costmap(arena_width_m=0.30),
            )

        error_evidence = captured.exception.to_evidence()
        selection = error_evidence["route_uncertainty_selection"]
        self.assertEqual(
            error_evidence["error_code"],
            "no_uncertainty_admitted_camera_candidate",
        )
        self.assertIsNone(error_evidence["selected_candidate_uid"])
        self.assertFalse(error_evidence["motion_authorized"])
        self.assertFalse(selection["decision"]["ready"])
        self.assertTrue(selection["decision"]["fail_closed"])
        self.assertEqual(selection["uncertainty_admitted_candidate_count"], 0)
        self.assertTrue(
            all(
                not row["accepted"]
                for row in selection["route_uncertainty_evaluations"]
            )
        )
        self.assertEqual(
            error_evidence["route_uncertainty_selection_sha256"],
            payload_sha256(selection),
        )

    def test_selection_binding_rejects_hash_and_route_tampering(self):
        selected_plan = _plan(
            "candidate-selected",
            (START, Pose2D(0.5, 0.0, 0.0)),
        )
        selected = _select(
            (selected_plan,),
            (_option("candidate-selected", route_length_m=1.5),),
        )
        evidence = selected.to_evidence()
        validate_candidate_route_uncertainty_selection_binding(
            evidence,
            selected_plan,
        )

        outer_hash_tamper = deepcopy(evidence)
        outer_hash_tamper["route_uncertainty_selection"]["decision"][
            "selected_candidate_uid"
        ] = "candidate-forged"
        with self.assertRaisesRegex(ValueError, "selection hash mismatch"):
            validate_candidate_route_uncertainty_selection_binding(
                outer_hash_tamper,
                selected_plan,
            )

        admission_hash_tamper = deepcopy(evidence)
        evaluations = admission_hash_tamper[
            "route_uncertainty_selection"
        ]["route_uncertainty_evaluations"]
        evaluations[0]["admission_evidence"]["route"]["poses"][1][
            "x_m"
        ] += 0.01
        admission_hash_tamper[
            "route_uncertainty_selection_sha256"
        ] = payload_sha256(
            admission_hash_tamper["route_uncertainty_selection"]
        )
        with self.assertRaisesRegex(ValueError, "admission hash mismatch"):
            validate_candidate_route_uncertainty_selection_binding(
                admission_hash_tamper,
                selected_plan,
            )

        forged_acceptance = deepcopy(evidence)
        forged_evaluation = forged_acceptance[
            "route_uncertainty_selection"
        ]["route_uncertainty_evaluations"][0]
        forged_evaluation["admission_evidence"]["decision"]["decision"][
            "accepted"
        ] = False
        forged_evaluation["admission_evidence_sha256"] = (
            route_uncertainty_admission_evidence_sha256(
                forged_evaluation["admission_evidence"]
            )
        )
        forged_acceptance["route_uncertainty_selection_sha256"] = (
            payload_sha256(forged_acceptance["route_uncertainty_selection"])
        )
        with self.assertRaisesRegex(ValueError, "not an accepted pure decision"):
            validate_candidate_route_uncertainty_selection_binding(
                forged_acceptance,
                selected_plan,
            )

        changed_route = _plan(
            "candidate-selected",
            (START, Pose2D(0.25, 0.0, 0.0)),
        )
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_candidate_route_uncertainty_selection_binding(
                evidence,
                changed_route,
            )


if __name__ == "__main__":
    unittest.main()
