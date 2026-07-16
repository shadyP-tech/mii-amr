from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.dynamic_approach_planner import (
    DynamicApproachConfig,
    plan_dynamic_approach,
)
from scripts.aufgabe04.navigation.dynamic_replan_policy import (
    DynamicReplanPolicy,
    DynamicReplanState,
)
from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    DynamicRouteSource,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.map_io import CELL_FREE, MapMetadata, OccupancyGrid
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.plan_synchronized_viewpoint import (
    _diagnostics_payload,
    _route_csv_text,
)
from scripts.aufgabe04.navigation.route_revision_store import RouteRevisionStore
from scripts.aufgabe04.navigation.run_events import build_event
from scripts.aufgabe04.navigation.viewpoint_recommendation import (
    FaceCandidate,
    MaterialTarget,
    SideEvidence,
    StandGeometry,
    SynchronizedViewpointRecommendation,
)


def open_costmap() -> Costmap:
    metadata = MapMetadata(
        yaml_path=Path("map.yaml"),
        image_path=Path("map.pgm"),
        resolution=0.1,
        origin=(0.0, 0.0, 0.0),
        negate=0,
        occupied_thresh=0.65,
        free_thresh=0.2,
        mode="trinary",
    )
    cells = tuple(tuple(CELL_FREE for _ in range(40)) for _ in range(30))
    return Costmap.from_occupancy_grid(
        OccupancyGrid(metadata=metadata, width=40, height=30, cells=cells)
    )


def recommendation(
    robot: Pose2D,
    observed_at: float,
    *,
    face_normal_rad: float = 0.0,
) -> SynchronizedViewpointRecommendation:
    stand = Pose2D(1.5, 1.5)
    face_a = Pose2D(
        stand.x_m + 0.35 * math.cos(face_normal_rad),
        stand.y_m + 0.35 * math.sin(face_normal_rad),
        face_normal_rad + math.pi,
    )
    opposite = face_normal_rad + math.pi
    face_b = Pose2D(
        stand.x_m + 0.35 * math.cos(opposite),
        stand.y_m + 0.35 * math.sin(opposite),
        face_normal_rad,
    )
    return SynchronizedViewpointRecommendation(
        schema_version=1,
        simulation_only=True,
        stream_id="integration-stream",
        stand_id="A",
        planning_frame="odom",
        source="synchronized_lidar_camera_viewpoint",
        observation_unix_sec=observed_at,
        sensor_stamp_sec=observed_at - 90.0,
        stand=StandGeometry(stand, 0.06, 0.02, "lidar_cluster"),
        robot_pose=robot,
        axis_confidence=0.9,
        axis_state="resolved",
        face_candidates=(
            FaceCandidate("face_a", face_normal_rad, face_a, True),
            FaceCandidate("face_b", opposite, face_b, True),
        ),
        side_evidence=SideEvidence(
            "qr_registry", 0.98, True, True, "face_a", "sim_qr_consensus"
        ),
        material_target=MaterialTarget("face_a", face_a, "hard_qr"),
    )


class DynamicApproachIntegrationTest(unittest.TestCase):
    def test_revision_adoption_acknowledgement_withdrawal_and_planner_death(self):
        costmap = open_costmap()
        config = DynamicApproachConfig(standoff_distance_m=0.35)
        clock = [100.0]
        policy = DynamicReplanPolicy(refresh_timeout_sec=20.0)
        state = DynamicReplanState()

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "route.manifest.json"
            store = RouteRevisionStore(
                manifest,
                stream_id="integration-stream",
                writer_id="planner",
                now_fn=lambda: clock[0],
            )
            rec = recommendation(Pose2D(0.4, 1.5), clock[0])
            state, decision = policy.evaluate(
                state,
                target=rec.material_target,
                robot_pose=rec.robot_pose,
                now_sec=clock[0],
            )
            self.assertTrue(decision.should_replan)
            result = plan_dynamic_approach(
                costmap,
                rec.robot_pose,
                rec.stand.center,
                -math.pi / 2.0,
                hard_face_id=0,
                config=config,
            )
            self.assertIsNotNone(result.plan)
            plan = result.plan
            assert plan is not None
            diagnostics = _diagnostics_payload(result, rec, decision.target_revision)
            first = store.publish_active(
                _route_csv_text(
                    costmap,
                    plan,
                    stream_id=rec.stream_id,
                    target_revision=decision.target_revision,
                ),
                diagnostics,
                target_revision=decision.target_revision,
                observation_unix_sec=rec.observation_unix_sec,
                source_robot_pose=asdict(rec.robot_pose),
                target={**asdict(plan.target), "face_id": "face_a"},
                evidence=asdict(rec.side_evidence),
                previous_route_length_m=0.0,
                new_route_length_m=plan.length_m,
                safety_diagnostics=asdict(result.diagnostics),
            )
            state = policy.mark_route_planned(
                state,
                planned_start=rec.robot_pose,
                now_sec=clock[0],
                target_revision=decision.target_revision,
            )

            source = DynamicRouteSource(
                manifest,
                stream_id=rec.stream_id,
                expected_writer_id="planner",
                max_manifest_age_sec=2.0,
                max_observation_age_sec=2.0,
                max_join_distance_m=0.20,
            )
            adopted = source.poll(rec.robot_pose, clock[0])
            self.assertEqual(adopted.kind, RouteUpdateKind.ADOPT)
            self.assertTrue(adopted.requires_zero_cycle)
            acknowledgement = build_event("route_reloaded", **dict(adopted.event_fields))
            self.assertEqual(acknowledgement["route_revision"], first.route_revision)
            self.assertEqual(acknowledgement["route_sha256"], first.route_hash)

            # A 20-degree physical axis refinement is material at this
            # standoff.  It must create both a new target revision and a new
            # immutable route that the moving-side consumer adopts.
            clock[0] = 100.5
            moved_robot = Pose2D(0.55, 1.5)
            refined = recommendation(
                moved_robot,
                clock[0],
                face_normal_rad=math.radians(20.0),
            )
            state, refined_decision = policy.evaluate(
                state,
                target=refined.material_target,
                robot_pose=refined.robot_pose,
                now_sec=clock[0],
            )
            self.assertTrue(refined_decision.should_replan)
            self.assertTrue(refined_decision.target_changed)
            self.assertEqual(refined_decision.target_revision, decision.target_revision + 1)
            refined_result = plan_dynamic_approach(
                costmap,
                refined.robot_pose,
                refined.stand.center,
                math.radians(20.0) - math.pi / 2.0,
                hard_face_id=0,
                config=config,
            )
            self.assertIsNotNone(refined_result.plan)
            refined_plan = refined_result.plan
            assert refined_plan is not None
            second = store.publish_active(
                _route_csv_text(
                    costmap,
                    refined_plan,
                    stream_id=refined.stream_id,
                    target_revision=refined_decision.target_revision,
                ),
                _diagnostics_payload(
                    refined_result, refined, refined_decision.target_revision
                ),
                target_revision=refined_decision.target_revision,
                observation_unix_sec=refined.observation_unix_sec,
                source_robot_pose=asdict(refined.robot_pose),
                target={**asdict(refined_plan.target), "face_id": "face_a"},
                evidence=asdict(refined.side_evidence),
                previous_route_length_m=plan.length_m,
                new_route_length_m=refined_plan.length_m,
                safety_diagnostics=asdict(refined_result.diagnostics),
            )
            state = policy.mark_route_planned(
                state,
                planned_start=refined.robot_pose,
                now_sec=clock[0],
                target_revision=refined_decision.target_revision,
            )
            adopted_refinement = source.poll(refined.robot_pose, clock[0])
            self.assertEqual(adopted_refinement.kind, RouteUpdateKind.ADOPT)
            self.assertEqual(second.route_revision, first.route_revision + 1)
            self.assertNotEqual(second.route_hash, first.route_hash)
            self.assertEqual(
                adopted_refinement.event_fields["target_revision"],
                refined_decision.target_revision,
            )
            self.assertAlmostEqual(
                adopted_refinement.event_fields["previous_route_length_m"],
                plan.length_m,
            )
            self.assertAlmostEqual(
                adopted_refinement.event_fields["new_route_length_m"],
                refined_plan.length_m,
            )

            clock[0] = 101.0
            store.withdraw(
                "recommendation stale",
                target_revision=refined_decision.target_revision,
                observation_unix_sec=clock[0],
            )
            stopped = source.poll(rec.robot_pose, clock[0])
            self.assertEqual(stopped.kind, RouteUpdateKind.STOP)
            self.assertEqual(stopped.event_name, "dynamic_route_withdrawn")

            # A separate active stream proves planner/manifest liveness is
            # bounded independently of the observer timestamp.
            death_manifest = Path(tmp) / "death.manifest.json"
            clock[0] = 200.0
            death_store = RouteRevisionStore(
                death_manifest,
                stream_id="death-stream",
                writer_id="planner",
                now_fn=lambda: clock[0],
            )
            death_store.publish_active(
                _route_csv_text(
                    costmap,
                    plan,
                    stream_id="death-stream",
                    target_revision=1,
                ),
                diagnostics,
                target_revision=1,
                observation_unix_sec=clock[0],
                source_robot_pose=asdict(rec.robot_pose),
                target=asdict(plan.target),
                evidence={},
                previous_route_length_m=0.0,
                new_route_length_m=plan.length_m,
                safety_diagnostics=asdict(result.diagnostics),
            )
            death_source = DynamicRouteSource(
                death_manifest,
                stream_id="death-stream",
                max_manifest_age_sec=1.0,
                max_observation_age_sec=10.0,
                max_join_distance_m=0.20,
            )
            self.assertEqual(
                death_source.poll(rec.robot_pose, clock[0]).kind,
                RouteUpdateKind.ADOPT,
            )
            clock[0] = 202.0
            planner_dead = death_source.poll(rec.robot_pose, clock[0])
            self.assertEqual(planner_dead.kind, RouteUpdateKind.STOP)
            self.assertEqual(planner_dead.event_fields["fault_code"], "manifest_stale")


if __name__ == "__main__":
    unittest.main()
