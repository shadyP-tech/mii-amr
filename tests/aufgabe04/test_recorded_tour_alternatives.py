"""Replay Oct 2 geometry alternatives from tracked, frozen sensor evidence.

The companion fixture retains every projected temporary obstacle cell and uses
the identical static occupancy already stored in the Sep 23 regression fixture.
It needs no run artifacts, map files, ROS, or robot connection.
"""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.approach.admitted_pose_route import (
    STORED_POSE_TOUR_ROUTE_PURPOSE,
    _plan_admitted_pose_geometry,
)
from scripts.aufgabe04.navigation.approach.admitted_return_uncertainty import (
    ReturnUncertaintyExhausted,
    select_admitted_return_prefix,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_compute import (
    validate_physical_clearance,
)
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import (
    CandidateRouteUncertaintyContext,
)
from scripts.aufgabe04.navigation.approach.stored_pose_route_alternatives import (
    AlternativeGeometryRejected,
    select_stored_pose_route_alternative,
    uncertainty_context_evidence,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
    _costmap_evidence,
    evaluate_route_uncertainty_admission,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import PlanarCovariance
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import MapMetadata, OccupancyGrid
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    candidate_snapshot_sha256,
    new_candidate_snapshot,
)


FIXTURE = Path(__file__).with_name("fixtures") / "station_tour_uncertainty_20261002.json"
TARGET_SHA256 = "a" * 64


def recorded_tour_case():
    """Reconstruct raw occupancy, all frozen stand keepouts and one stopped budget."""
    fixture = json.loads(FIXTURE.read_text())
    static_fixture = json.loads(FIXTURE.with_name(fixture["static_map_fixture"]).read_text())
    for key in ("map_yaml_sha256", "map_image_sha256"):
        if fixture[key] != static_fixture["provenance"][key]:
            raise AssertionError("recorded tours do not share identical static map bytes")
    if fixture["map_bundle_sha256"] != static_fixture["recorded_candidate_pool"]["map_bundle_sha256"]:
        raise AssertionError("recorded tours do not share an identical map bundle")
    source_map = static_fixture["map"]
    cells = tuple(value for value, count in source_map["cells_rle"] for _ in range(count))
    width, height = source_map["width"], source_map["height"]
    if len(cells) != width * height:
        raise AssertionError("recorded occupancy dimensions differ")
    metadata = MapMetadata(
        yaml_path=Path("recorded-map.yaml"), image_path=Path("recorded-map.pgm"),
        resolution=source_map["resolution"], origin=tuple(source_map["origin"]),
        negate=source_map["negate"], occupied_thresh=source_map["occupied_thresh"],
        free_thresh=source_map["free_thresh"], mode=source_map["mode"],
    )
    grid = OccupancyGrid(
        metadata=metadata, width=width, height=height,
        cells=tuple(cells[y * width:(y + 1) * width] for y in range(height)),
    )
    base = Costmap.from_occupancy_grid(grid).with_arena_bounds(ArenaBounds(**source_map["arena"]))
    base = base.with_blocked_cells(
        (GridCell(x, y) for x, y in fixture["temporary_obstacle_cells"]),
        source="temporary_obstacle",
    )
    context = CandidateRouteUncertaintyContext(
        covariance=PlanarCovariance(**fixture["covariance_m2"]),
        admission_config=RouteUncertaintyAdmissionConfig(**fixture["admission_config"]),
        source_evidence={
            "recorded_preflight_sha256": fixture["provenance"]["preflight_file_sha256"],
            "selection_only": True, "motion_authorized": False,
        },
    )
    # This minimal snapshot carries the real geometry with explicit test-only
    # observation ancestry. It does not reconstruct operational authorization.
    snapshot = new_candidate_snapshot(
        snapshot_id="recorded_tour_geometry", created_unix_sec=0., planning_frame="map",
        map_bundle_sha256=fixture["map_bundle_sha256"],
        candidates=(FrozenCandidate(
            candidate_uid=entry["candidate_uid"],
            geometry=CandidateGeometry(**entry["geometry"]),
            source=CandidateSource(
                "recorded_geometry_fixture",
                fixture["provenance"]["projected_candidate_snapshot_file_sha256"],
                "0" * 64, (entry["candidate_uid"],),
            ),
            confidence=1., hit_count=1, first_seen_sec=0., last_seen_sec=0.,
        ) for entry in fixture["candidate_geometries"]),
    )
    return fixture, base, snapshot, context


def recorded_geometry_builder(fixture, base, snapshot):
    """Use the production A*, keepouts, exact endpoints and certified smoother."""
    physical = fixture["physical_clearance"]
    radius = physical["minimum_candidate_transit_radius_m"]
    active, _, _ = validate_physical_clearance(
        physical, inflation_radius_m=physical["minimum_static_inflation_m"],
        candidate_transit_radius_m=radius,
    )

    def build(inflation):
        try:
            geometry, goal_connector = _plan_admitted_pose_geometry(
                base=base, snapshot=snapshot, radius=radius, inflation=inflation,
                collision=physical["minimum_collision_standoff_m"],
                candidate_uid=fixture["candidate_uid"], evidence={},
                start=Pose2D(**fixture["start"]), target=Pose2D(**fixture["target"]),
                plan=SimpleNamespace(config=SimpleNamespace(snap_radius_m=fixture["snap_radius_m"])),
                active=active, purpose=STORED_POSE_TOUR_ROUTE_PURPOSE, stationary_turn=False,
            )
        except ValueError as exc:
            raise AlternativeGeometryRejected(str(exc)) from exc
        if goal_connector is not None:
            raise AssertionError("stored-pose tour must preserve its exact goal without a Start connector")
        return geometry

    return build


class RecordedTourAlternativesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture, cls.base, cls.snapshot, cls.context = recorded_tour_case()

    def test_recorded_minimum_inflation_exhausts_full_route_and_prefixes(self):
        self.assertEqual(_costmap_evidence(self.base), self.fixture["expected_augmented_costmap"])
        case = self.fixture["cases"][0]
        geometry = recorded_geometry_builder(self.fixture, self.base, self.snapshot)(case["inflation_radius_m"])
        self.assertEqual([asdict(p) for p in geometry.full_poses], case["full_poses"])
        admission = evaluate_route_uncertainty_admission(
            self.base, geometry.full_poses, self.context.covariance, self.context.admission_config,
        )
        self.assertFalse(admission.decision.accepted)
        self.assertAlmostEqual(admission.decision.remaining_margin_m, case["expected_full_minimum_margin_m"], places=12)
        with self.assertRaisesRegex(ReturnUncertaintyExhausted, case["expected_selection"]["error"]):
            select_admitted_return_prefix(
                full_poses=geometry.full_poses, base_costmap=self.base, uncertainty=self.context,
                target_evidence_sha256=TARGET_SHA256,
                minimum_prefix_vertex_index=case["minimum_prefix_vertex_index"],
            )

    def test_real_alternatives_find_first_admitted_prefix_without_changing_raw_inputs(self):
        before_map = _costmap_evidence(self.base)
        before_context = uncertainty_context_evidence(self.context)
        before_candidates = candidate_snapshot_sha256(self.snapshot)
        built = {}
        build_geometry = recorded_geometry_builder(self.fixture, self.base, self.snapshot)

        def build(radius):
            built[radius] = build_geometry(radius)
            return built[radius]

        geometry, stage, radius, evidence = select_stored_pose_route_alternative(
            build, base_radius_m=self.fixture["physical_clearance"]["minimum_static_inflation_m"],
            base_costmap=self.base, uncertainty=self.context, target_evidence_sha256=TARGET_SHA256,
            identity={"tour_id": self.fixture["tour_id"], "candidate_uid": self.fixture["candidate_uid"]},
        )
        case = self.fixture["cases"][1]
        self.assertEqual(list(built), [.25, .275, .30, .325])
        self.assertEqual(radius, case["inflation_radius_m"])
        self.assertIs(geometry, built[radius])
        self.assertEqual([asdict(p) for p in geometry.full_poses], case["full_poses"])
        self.assertEqual(geometry.full_poses[0], Pose2D(**self.fixture["start"]))
        self.assertEqual(geometry.full_poses[-1], Pose2D(**self.fixture["target"]))
        self.assertFalse(stage.is_final_stage)
        self.assertEqual(asdict(stage.stage_target_pose), case["expected_selection"]["stage_target_pose"])
        full = evaluate_route_uncertainty_admission(
            self.base, geometry.full_poses, self.context.covariance, self.context.admission_config,
        )
        self.assertFalse(full.decision.accepted)
        self.assertAlmostEqual(full.decision.remaining_margin_m, case["expected_full_minimum_margin_m"], places=12)
        prefix = evaluate_route_uncertainty_admission(
            self.base, stage.poses, self.context.covariance, self.context.admission_config,
        )
        self.assertTrue(prefix.decision.accepted)
        self.assertAlmostEqual(prefix.decision.remaining_margin_m, case["expected_selected_minimum_margin_m"], places=12)
        self.assertEqual([a["status"] for a in evidence["attempts"]], ["uncertainty_rejected"] * 3 + ["accepted"])
        self.assertEqual(evidence["selected_attempt_index"], 3)
        self.assertFalse(evidence["motion_authorized"])
        self.assertFalse(stage.evidence["motion_authorized"])
        self.assertEqual(evidence["uncertainty_context_sha256"], payload_sha256(before_context))
        self.assertEqual(uncertainty_context_evidence(self.context), before_context)
        self.assertEqual(_costmap_evidence(self.base), before_map)
        self.assertEqual(candidate_snapshot_sha256(self.snapshot), before_candidates)


if __name__ == "__main__":
    unittest.main()
