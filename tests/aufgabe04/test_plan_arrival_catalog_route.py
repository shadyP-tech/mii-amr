import csv
import hashlib
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.aufgabe04.navigation.plan_arrival_catalog_route import main
from scripts.aufgabe04.navigation.safety_checks import (
    validate_catalog_route_binding_json,
    validate_route_diagnostics_json,
)
from scripts.aufgabe04.navigation.waypoint_csv import load_route_leg
from scripts.aufgabe04.stations.arrival_pose_catalog import (
    load_arrival_pose_catalog,
    new_arrival_pose_catalog,
    upsert_arrival_pose,
    write_arrival_pose_catalog,
)
from scripts.aufgabe04.stations.arrival_pose_models import (
    ArrivalPoseRecord,
    ArrivalPoseValidation,
    AxisEstimate,
    CatalogPose2D,
    CatalogProvenance,
    FaceSelection,
    StandEstimate,
)


def record(candidate_uid: str, x_m: float, y_m: float) -> ArrivalPoseRecord:
    return ArrivalPoseRecord(
        candidate_uid=candidate_uid,
        stand_id=candidate_uid,
        stand=StandEstimate(x_m, y_m, 0.06, 0.02),
        axis=AxisEstimate(0.0, 0.95, 8, "silhouette/head_rectangle", 101.0),
        face=FaceSelection(
            "face_0",
            math.pi / 2.0,
            True,
            "robot_facing_axis",
            0.95,
            False,
            True,
            "synchronized/lidar_camera",
        ),
        arrival_pose=CatalogPose2D(x_m, y_m + 0.32, -math.pi / 2.0),
        corridor_entry_pose=CatalogPose2D(x_m, y_m + 0.72, -math.pi / 2.0),
        standoff_m=0.32,
        corridor_length_m=0.40,
        validation=ArrivalPoseValidation(True, True, True, "", 102.0),
        source_observation_ids=(f"obs/{candidate_uid}",),
        sensor_stamp_sec=80.0,
        source="simulation/synchronized_viewpoint",
    )


class PlanArrivalCatalogRouteTest(unittest.TestCase):
    def test_freezes_catalog_and_writes_exact_optimized_route(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "map.pgm").write_text(
                "P2\n60 50\n255\n" + " ".join(["254"] * 3000) + "\n"
            )
            map_yaml = root / "map.yaml"
            map_yaml.write_text(
                "image: map.pgm\nresolution: 0.1\norigin: [0.0, 0.0, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.2\nmode: trinary\n"
            )
            map_hash = hashlib.sha256(map_yaml.read_bytes()).hexdigest()
            world = root / "world_001.world"
            world.write_text("test world\n")
            world_hash = hashlib.sha256(world.read_bytes()).hexdigest()
            first = record("candidate_a", 1.5, 1.5)
            second = record("candidate_b", 3.0, 2.5)
            first = first.__class__(
                **{
                    **first.__dict__,
                    "validation": first.validation.__class__(
                        True, True, True, map_hash, 102.0
                    ),
                }
            )
            second = second.__class__(
                **{
                    **second.__dict__,
                    "validation": second.validation.__class__(
                        True, True, True, map_hash, 102.0
                    ),
                }
            )
            catalog = new_arrival_pose_catalog(
                catalog_id="survey_001",
                provenance=CatalogProvenance(
                    "odom",
                    map_hash,
                    "world_001",
                    world_hash,
                    "session_001",
                    "simulation",
                ),
                expected_candidate_uids=("candidate_a", "candidate_b"),
                created_unix_sec=100.0,
            )
            catalog = upsert_arrival_pose(catalog, first, updated_unix_sec=103.0)
            catalog = upsert_arrival_pose(catalog, second, updated_unix_sec=104.0)
            catalog_path = root / "catalog.json"
            write_arrival_pose_catalog(catalog_path, catalog)
            route = root / "route.csv"
            diagnostics = root / "diagnostics.json"
            visits = root / "visits.json"
            costs = root / "costs.json"
            snapshot = root / "snapshot.json"

            with redirect_stdout(StringIO()):
                status = main(
                    [
                        "--catalog", str(catalog_path),
                        "--map", str(map_yaml),
                        "--world", str(world),
                        "--session-id", "session_001",
                        "--start-x", "0.4",
                        "--start-y", "0.4",
                        "--route-csv", str(route),
                        "--diagnostics-json", str(diagnostics),
                        "--visit-order-json", str(visits),
                        "--pairwise-costs-json", str(costs),
                        "--catalog-snapshot-json", str(snapshot),
                    ]
                )
            frozen = load_arrival_pose_catalog(catalog_path)
            with route.open() as handle:
                rows = list(csv.DictReader(handle))
            diagnostics_payload = json.loads(diagnostics.read_text())
            visits_payload = json.loads(visits.read_text())
            costs_payload = json.loads(costs.read_text())
            validation_results = []
            for leg_index, _leg in enumerate(diagnostics_payload["legs"]):
                selected_leg = load_route_leg(route, leg_index)
                count = sum(
                    1 for row in rows if int(row["leg_index"]) == leg_index
                )
                validation_results.append(
                    validate_route_diagnostics_json(
                        diagnostics,
                        leg_index,
                        csv_point_count=count,
                    ).ok
                    and validate_catalog_route_binding_json(
                        diagnostics,
                        selected_leg,
                    ).ok
                )
            # Even a syntactically harmless content change must invalidate the
            # route/diagnostics binding before motion.
            route.write_text(route.read_text() + "\n")
            tampered_binding = validate_catalog_route_binding_json(
                diagnostics,
                load_route_leg(route, 0),
            )

        self.assertEqual(status, 0)
        self.assertTrue(frozen.frozen)
        self.assertTrue(visits_payload["optimal"])
        self.assertEqual(len(visits_payload["candidate_order"]), 2)
        self.assertEqual(len(diagnostics_payload["legs"]), 2)
        self.assertEqual({row["leg_index"] for row in rows}, {"0", "1"})
        self.assertTrue(all(row["route_kind"] == "catalog_face_approach" for row in rows))
        self.assertTrue(all(row["catalog_sha256"] for row in rows))
        self.assertEqual(len(costs_payload["edges"]), 4)
        self.assertEqual(
            diagnostics_payload["metadata"]["non_target_stand_keepout_policy"],
            "max(body_uncertainty_collision,lidar_minimum_standoff)",
        )
        self.assertTrue(
            all(
                clearance["radius_m"] >= 0.26
                and clearance["minimum_route_clearance_m"]
                > clearance["radius_m"]
                for leg in diagnostics_payload["legs"]
                for clearance in leg["non_target_stand_clearances"]
            )
        )
        self.assertTrue(
            all(
                "non_target_stand_clearances" in edge
                for edge in costs_payload["edges"]
            )
        )
        self.assertTrue(
            all(
                "non_target_keepout_overlay" in edge
                for edge in costs_payload["edges"]
            )
        )
        self.assertTrue(
            all(
                "non_target_keepout_overlay" in leg
                for leg in diagnostics_payload["legs"]
            )
        )
        self.assertTrue(all(validation_results))
        self.assertFalse(tampered_binding.ok)
        self.assertIn(
            "catalog route CSV SHA-256 does not match diagnostics",
            tampered_binding.failures,
        )

    def test_catalog_binding_rejects_a_mixed_route_and_diagnostics_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            route = root / "route.csv"
            diagnostics = root / "diagnostics.json"
            route.write_text(
                "leg_index,point_index,world_x_m,world_y_m,yaw_rad,"
                "cumulative_length_m,protected,corridor,route_kind,"
                "source_arrival_id,target_arrival_id,catalog_sha256\n"
                f"0,0,0.0,0.0,,0.0,false,false,catalog_face_approach,"
                f"mission_start,candidate_a::face_0,{'a' * 64}\n"
                f"0,1,0.4,0.0,0.0,0.4,true,true,catalog_face_approach,"
                f"mission_start,candidate_a::face_0,{'a' * 64}\n"
            )
            diagnostics.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "route_kind": "catalog_face_approach",
                            "catalog_sha256": "b" * 64,
                        },
                        "legs": [
                            {
                                "source_arrival_id": "mission_start",
                                "target_arrival_id": "candidate_a::face_0",
                                "exact_target": {
                                    "x_m": 0.4,
                                    "y_m": 0.0,
                                    "yaw_rad": 0.0,
                                },
                                "corridor_entry": {
                                    "x_m": 0.4,
                                    "y_m": 0.0,
                                    "yaw_rad": 0.0,
                                },
                            }
                        ],
                    }
                )
            )
            leg = load_route_leg(route, 0)

            status = validate_catalog_route_binding_json(diagnostics, leg)

        self.assertFalse(status.ok)
        self.assertIn("catalog SHA-256 does not match diagnostics", status.failures)


if __name__ == "__main__":
    unittest.main()
