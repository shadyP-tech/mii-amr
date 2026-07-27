import csv
import hashlib
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

from scripts.aufgabe04.artifacts import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    SurveyManifest,
    artifact_reference,
    load_mission_plan_manifest,
    mission_plan_manifest_sha256,
    survey_manifest_sha256,
    write_survey_manifest,
)
from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.logistics.server_validation.artifacts import (
    validated_task_snapshot_sha256,
    write_validated_task_snapshot,
)
from scripts.aufgabe04.logistics.server_validation.models import (
    ValidatedServerTask,
    server_order_sha256,
)
from scripts.aufgabe04.navigation import (
    plan_arrival_catalog_route as planner_module,
)
from scripts.aufgabe04.navigation.plan_arrival_catalog_route import main
from scripts.aufgabe04.navigation.safety_checks import (
    validate_catalog_route_binding_json,
    validate_route_diagnostics_json,
)
from scripts.aufgabe04.navigation.waypoint_csv import load_route_leg
from scripts.aufgabe04.navigation.execution_route_certificate import (
    execution_route_certificate_sha256,
    load_execution_route_certificate,
    validate_execution_route_identity,
)
from scripts.aufgabe04.navigation.map_io import (
    freeze_map_bundle,
    write_frozen_map_bundle,
)
from scripts.aufgabe04.navigation.mission_execution_gate import (
    validate_logistics_execution_bundle,
)
from scripts.aufgabe04.stations.arrival_pose_catalog import (
    arrival_pose_catalog_sha256,
    freeze_arrival_pose_catalog,
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
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    candidate_snapshot_sha256,
    new_candidate_snapshot,
    write_candidate_snapshot,
)
from scripts.aufgabe04.stations.station_identity_registry import (
    StationIdentity,
    new_station_identity_registry,
    station_identity_registry_sha256,
    write_station_identity_registry,
)


SYNTHETIC_ARENA_ARGS = [
    "--arena-length-m",
    "20.0",
    "--arena-width-m",
    "20.0",
]


def record(
    candidate_uid: str,
    x_m: float,
    y_m: float,
    *,
    stand_id: Optional[str] = None,
) -> ArrivalPoseRecord:
    return ArrivalPoseRecord(
        candidate_uid=candidate_uid,
        stand_id=stand_id or candidate_uid,
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
    def test_route_node_conservatively_covers_frozen_geometry_and_tracking_tube(self):
        frozen = FrozenCandidate(
            candidate_uid="candidate_a",
            geometry=CandidateGeometry(2.02, 2.0, 0.10, 0.04, 0.40),
            source=CandidateSource(
                "lidar/stand_confirmation",
                "1" * 64,
                "2" * 64,
                ("observation_a",),
            ),
            confidence=0.95,
            hit_count=8,
            first_seen_sec=90.0,
            last_seen_sec=99.0,
        )
        args = SimpleNamespace(
            robot_radius_m=0.105,
            collision_margin_m=0.02,
            tracking_margin_m=0.03,
            corridor_sample_spacing_m=0.05,
            lidar_stop_distance_m=0.18,
            scan_origin_to_base_offset_m=0.0,
            lidar_clearance_margin_m=0.02,
        )

        node = planner_module._route_node(
            record("candidate_a", 2.0, 2.0),
            args,
            frozen_candidate=frozen,
        )

        self.assertEqual(node.config.stand_radius_m, 0.10)
        self.assertAlmostEqual(node.config.stand_position_uncertainty_m, 0.06)
        self.assertAlmostEqual(node.config.tracking_margin_m, 0.03)
        self.assertAlmostEqual(node.config.stand_keepout_radius_m, 0.315)
        self.assertAlmostEqual(node.config.non_target_stand_keepout_radius_m, 0.43)

    def test_validated_task_drives_nonlexical_revisit_route_and_linked_manifest(self):
        """Survey + task order C,A,C remains intact through every artifact."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            width = 90
            height = 60
            (root / "map.pgm").write_text(
                f"P2\n{width} {height}\n255\n"
                + " ".join(["254"] * (width * height))
                + "\n"
            )
            map_yaml = root / "temporary_map.yaml"
            map_yaml.write_text(
                "image: map.pgm\nresolution: 0.1\norigin: [0.0, 0.0, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.2\n"
                "mode: trinary\n"
            )
            map_hash = hashlib.sha256(map_yaml.read_bytes()).hexdigest()
            map_bundle = freeze_map_bundle(
                map_yaml,
                semantic_map_id="temporary_arena_001",
                planning_frame="map",
            )
            map_bundle_path = root / "map_bundle.json"
            write_frozen_map_bundle(map_bundle_path, map_bundle)

            world = root / "world_logistics.world"
            world.write_text("test logistics world\n")
            world_hash = hashlib.sha256(world.read_bytes()).hexdigest()

            records = (
                record("candidate_a", 2.0, 2.0, stand_id="A"),
                record("candidate_b", 4.0, 2.0, stand_id="B"),
                record("candidate_c", 6.0, 2.0, stand_id="C"),
            )
            records = tuple(
                replace(
                    item,
                    validation=replace(
                        item.validation,
                        validated_map_yaml_sha256=map_hash,
                    ),
                )
                for item in records
            )
            catalog = new_arrival_pose_catalog(
                catalog_id="survey_logistics_001",
                provenance=CatalogProvenance(
                    "map",
                    map_hash,
                    world.stem,
                    world_hash,
                    "session_logistics_001",
                    "simulation",
                ),
                expected_candidate_uids=tuple(
                    item.candidate_uid for item in records
                ),
                created_unix_sec=100.0,
            )
            for index, item in enumerate(records, start=1):
                catalog = upsert_arrival_pose(
                    catalog,
                    item,
                    updated_unix_sec=100.0 + index,
                )
            catalog = freeze_arrival_pose_catalog(
                catalog, frozen_unix_sec=104.0
            )
            catalog_path = root / "catalog.json"
            write_arrival_pose_catalog(catalog_path, catalog)
            catalog_sha256 = arrival_pose_catalog_sha256(catalog)

            candidate_snapshot = new_candidate_snapshot(
                snapshot_id="candidate_snapshot_logistics_001",
                created_unix_sec=103.0,
                planning_frame="map",
                map_bundle_sha256=map_bundle.bundle_sha256,
                candidates=tuple(
                    FrozenCandidate(
                        candidate_uid=item.candidate_uid,
                        geometry=CandidateGeometry(
                            x_m=item.stand.x_m,
                            y_m=item.stand.y_m,
                            radius_m=item.stand.radius_m,
                            uncertainty_m=item.stand.uncertainty_m,
                            keepout_radius_m=0.26,
                        ),
                        source=CandidateSource(
                            source_kind="lidar/stand_confirmation",
                            source_artifact_sha256="1" * 64,
                            detector_config_sha256="2" * 64,
                            observation_ids=(f"observation_{item.candidate_uid}",),
                        ),
                        confidence=0.95,
                        hit_count=8,
                        first_seen_sec=90.0,
                        last_seen_sec=99.0,
                    )
                    for item in records
                ),
            )
            candidate_snapshot_path = root / "candidate_snapshot.json"
            write_candidate_snapshot(candidate_snapshot_path, candidate_snapshot)

            identity_registry = new_station_identity_registry(
                registry_id="station_identities_logistics_001",
                created_unix_sec=105.0,
                candidate_snapshot_sha256=candidate_snapshot_sha256(
                    candidate_snapshot
                ),
                source_artifact_sha256="3" * 64,
                expected_candidate_uids=candidate_snapshot.candidate_uids,
                mappings=(
                    StationIdentity("candidate_a", "A", "station_A"),
                    StationIdentity("candidate_b", "B", "station_B"),
                    StationIdentity("candidate_c", "C", "station_C"),
                ),
            )
            identity_registry_path = root / "station_identities.json"
            write_station_identity_registry(
                identity_registry_path, identity_registry
            )

            survey_config_payload = {
                "arena_bounds": {
                    "length_m": 20.0,
                    "width_m": 20.0,
                    "center_x_m": 0.0,
                    "center_y_m": 0.0,
                    "yaw_deg": 0.0,
                    "margin_m": 0.0,
                }
            }
            survey_config_sha256 = payload_sha256(survey_config_payload)
            survey_config_path = root / (
                f"survey_config_{survey_config_sha256}.json"
            )
            write_content_hashed_json(
                survey_config_path,
                survey_config_payload,
                hash_field="survey_config_sha256",
            )
            catalog = replace(
                catalog,
                provenance=replace(
                    catalog.provenance,
                    map_bundle_sha256=map_bundle.bundle_sha256,
                    candidate_snapshot_sha256=candidate_snapshot_sha256(
                        candidate_snapshot
                    ),
                    station_identity_registry_sha256=(
                        station_identity_registry_sha256(identity_registry)
                    ),
                    survey_config_sha256=survey_config_sha256,
                    calibration_profile_sha256="5" * 64,
                    survey_input_binding_sha256="7" * 64,
                ),
            )
            write_arrival_pose_catalog(catalog_path, catalog)
            catalog_sha256 = arrival_pose_catalog_sha256(catalog)

            survey_manifest = SurveyManifest(
                schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
                manifest_id="survey_manifest_logistics_001",
                created_unix_sec=104.0,
                session_id="session_logistics_001",
                environment="simulation",
                planning_frame="map",
                map_bundle=artifact_reference(
                    "map_bundle",
                    map_bundle.semantic_map_id,
                    map_bundle.bundle_sha256,
                ),
                candidate_snapshot=artifact_reference(
                    "candidate_snapshot",
                    candidate_snapshot.snapshot_id,
                    candidate_snapshot_sha256(candidate_snapshot),
                ),
                environment_descriptor=artifact_reference(
                    "simulation_world", world.stem, world_hash
                ),
                survey_config=artifact_reference(
                    "survey_config",
                    f"survey_config_{survey_config_sha256[:16]}",
                    survey_config_sha256,
                ),
                calibration_profile=artifact_reference(
                    "calibration_profile", "calibration_001", "5" * 64
                ),
                arrival_pose_catalog=artifact_reference(
                    "arrival_pose_catalog", catalog.catalog_id, catalog_sha256
                ),
            )
            survey_manifest_path = root / "survey_manifest.json"
            write_survey_manifest(survey_manifest_path, survey_manifest)

            required_station_order = (
                "station_C",
                "station_A",
                "station_C",
            )
            order_sha256 = server_order_sha256(
                robot_id="tb3_1",
                mission_id="mission_042",
                target_station="station_C",
                plan_step_index=2,
                ordered_station_ids=required_station_order,
                plan_generated_at_sec=140.0,
            )
            validated_task = ValidatedServerTask(
                robot_id="tb3_1",
                mission_id="mission_042",
                state="RUNNING",
                last_qr="C",
                resolved_current_station="station_C",
                target_station="station_C",
                cargo="puck_01",
                plan_step_index=2,
                evidence={"validation": "server_status_and_plan"},
                ordered_station_ids=required_station_order,
                status_observed_at_sec=145.0,
                plan_generated_at_sec=140.0,
                validated_at_sec=150.0,
                order_sha256=order_sha256,
                source_plan_sha256="6" * 64,
            )
            task_snapshot_path = root / "validated_task.json"
            write_validated_task_snapshot(task_snapshot_path, validated_task)

            route = root / "logistics_route.csv"
            diagnostics = root / "logistics_diagnostics.json"
            visits = root / "logistics_visits.json"
            costs = root / "logistics_required_costs.json"
            catalog_snapshot = root / "logistics_catalog_snapshot.json"
            certificate_path = root / "logistics_route_certificate.json"
            mission_plan_path = root / "mission_plan.json"
            planner_args = [
                "--route-purpose", "logistics",
                "--catalog", str(catalog_path),
                "--map", str(map_yaml),
                "--map-frame", "map",
                "--semantic-map-id", "temporary_arena_001",
                "--map-bundle-json", str(map_bundle_path),
                "--candidate-snapshot", str(candidate_snapshot_path),
                "--station-identity-registry", str(identity_registry_path),
                "--survey-manifest", str(survey_manifest_path),
                "--task-snapshot", str(task_snapshot_path),
                "--robot-id", "tb3_1",
                "--max-task-snapshot-age-sec", "5.0",
                "--max-task-future-skew-sec", "2.0",
                "--world", str(world),
                "--session-id", "session_logistics_001",
                "--start-x", "0.5",
                "--start-y", "0.5",
                "--route-csv", str(route),
                "--diagnostics-json", str(diagnostics),
                "--visit-order-json", str(visits),
                "--pairwise-costs-json", str(costs),
                "--catalog-snapshot-json", str(catalog_snapshot),
                "--route-certificate-json", str(certificate_path),
                "--mission-plan-manifest", str(mission_plan_path),
                *SYNTHETIC_ARENA_ARGS,
            ]
            catalog_bytes_before = catalog_path.read_bytes()
            catalog_stat_before = catalog_path.stat()
            mismatched_bounds_args = list(planner_args)
            arena_length_index = mismatched_bounds_args.index(
                "--arena-length-m"
            )
            mismatched_bounds_args[arena_length_index + 1] = "19.0"
            mismatched_bounds_stderr = StringIO()
            with patch(
                "scripts.aufgabe04.navigation.plan_arrival_catalog_route.time.time",
                return_value=151.0,
            ), redirect_stderr(mismatched_bounds_stderr):
                with self.assertRaises(SystemExit) as mismatched_bounds_exit:
                    main(mismatched_bounds_args)

            other_survey_config_payload = {
                **survey_config_payload,
                "fixture_variant": "different-content",
            }
            other_survey_config_path = root / "other_survey_config.json"
            write_content_hashed_json(
                other_survey_config_path,
                other_survey_config_payload,
                hash_field="survey_config_sha256",
            )
            wrong_config_args = [
                *planner_args,
                "--survey-config-json",
                str(other_survey_config_path),
            ]
            wrong_config_stderr = StringIO()
            with patch(
                "scripts.aufgabe04.navigation.plan_arrival_catalog_route.time.time",
                return_value=151.0,
            ), redirect_stderr(wrong_config_stderr):
                with self.assertRaises(SystemExit) as wrong_config_exit:
                    main(wrong_config_args)

            self.assertFalse(route.exists())
            self.assertFalse(diagnostics.exists())
            output = StringIO()
            with patch(
                "scripts.aufgabe04.navigation.plan_arrival_catalog_route.time.time",
                return_value=151.0,
            ), redirect_stdout(output):
                status = main(planner_args)
            catalog_bytes_after = catalog_path.read_bytes()
            catalog_stat_after = catalog_path.stat()

            mission_plan = load_mission_plan_manifest(
                mission_plan_path, parent_survey=survey_manifest
            )
            certificate = load_execution_route_certificate(certificate_path)
            diagnostics_payload = json.loads(diagnostics.read_text())
            visit_payload = json.loads(visits.read_text())
            costs_payload = json.loads(costs.read_text())
            with route.open() as route_handle:
                rows = list(csv.DictReader(route_handle))
            stdout_payload = json.loads(output.getvalue())
            planner_config_path = Path(stdout_payload["planner_config_json"])
            route_bundle_path = Path(stdout_payload["route_bundle_json"])
            planner_config_payload = load_content_hashed_json(
                planner_config_path,
                hash_field="artifact_sha256",
            )
            route_bundle_payload = load_content_hashed_json(
                route_bundle_path,
                hash_field="artifact_sha256",
            )
            route_sha256 = hashlib.sha256(route.read_bytes()).hexdigest()
            execution_binding = validate_logistics_execution_bundle(
                route_leg=load_route_leg(route, 0),
                diagnostics_path=diagnostics,
                route_certificate_path=certificate_path,
                mission_plan_path=mission_plan_path,
                survey_manifest_path=survey_manifest_path,
                route_bundle_path=route_bundle_path,
                planner_config_path=planner_config_path,
                runtime_map_bundle_path=map_bundle_path,
                runtime_environment_path=world,
                candidate_snapshot_path=candidate_snapshot_path,
                station_identity_registry_path=identity_registry_path,
                arrival_pose_catalog_path=catalog_path,
                task_snapshot_path=task_snapshot_path,
                robot_id="tb3_1",
                runtime_planning_frame="map",
                now_sec=151.0,
            )
            with self.assertRaisesRegex(ValueError, "mission plan robot"):
                validate_logistics_execution_bundle(
                    route_leg=load_route_leg(route, 0),
                    diagnostics_path=diagnostics,
                    route_certificate_path=certificate_path,
                    mission_plan_path=mission_plan_path,
                    survey_manifest_path=survey_manifest_path,
                    route_bundle_path=route_bundle_path,
                    planner_config_path=planner_config_path,
                    runtime_map_bundle_path=map_bundle_path,
                    runtime_environment_path=world,
                    candidate_snapshot_path=candidate_snapshot_path,
                    station_identity_registry_path=identity_registry_path,
                    arrival_pose_catalog_path=catalog_path,
                    task_snapshot_path=task_snapshot_path,
                    robot_id="another_robot",
                    runtime_planning_frame="map",
                    now_sec=151.0,
                )
            with self.assertRaisesRegex(ValueError, "stale"):
                validate_logistics_execution_bundle(
                    route_leg=load_route_leg(route, 0),
                    diagnostics_path=diagnostics,
                    route_certificate_path=certificate_path,
                    mission_plan_path=mission_plan_path,
                    survey_manifest_path=survey_manifest_path,
                    route_bundle_path=route_bundle_path,
                    planner_config_path=planner_config_path,
                    runtime_map_bundle_path=map_bundle_path,
                    runtime_environment_path=world,
                    candidate_snapshot_path=candidate_snapshot_path,
                    station_identity_registry_path=identity_registry_path,
                    arrival_pose_catalog_path=catalog_path,
                    task_snapshot_path=task_snapshot_path,
                    robot_id="tb3_1",
                    runtime_planning_frame="map",
                    now_sec=200.0,
                )
            target_arrival_by_leg = tuple(
                next(
                    row["target_arrival_id"]
                    for row in rows
                    if int(row["leg_index"]) == leg_index
                )
                for leg_index in range(3)
            )
            expected_route_bundle_payload = {
                "schema_version": 1,
                "artifact_kind": "route_bundle",
                "route_csv_sha256": route_sha256,
                "diagnostics_sha256": hashlib.sha256(
                    diagnostics.read_bytes()
                ).hexdigest(),
                "visit_order_sha256": hashlib.sha256(
                    visits.read_bytes()
                ).hexdigest(),
                "required_edge_costs_sha256": hashlib.sha256(
                    costs.read_bytes()
                ).hexdigest(),
                "catalog_snapshot_sha256": hashlib.sha256(
                    catalog_snapshot.read_bytes()
                ).hexdigest(),
                "route_certificate_sha256": (
                    execution_route_certificate_sha256(certificate)
                ),
            }
            expected_route_bundle_sha256 = payload_sha256(
                expected_route_bundle_payload
            )
            validate_execution_route_identity(
                certificate,
                route_path=route,
                planning_frame="map",
                route_kind="catalog_face_approach",
                waypoint_count=len(rows),
                command_owner="/aufgabe04_simple_waypoint_follower",
                map_bundle_sha256=map_bundle.bundle_sha256,
                candidate_snapshot_sha256=candidate_snapshot_sha256(
                    candidate_snapshot
                ),
            )
            route_binding_results = []
            for leg_index in range(3):
                selected_leg = load_route_leg(route, leg_index)
                csv_point_count = sum(
                    int(row["leg_index"]) == leg_index for row in rows
                )
                route_binding_results.append(
                    validate_route_diagnostics_json(
                        diagnostics,
                        leg_index,
                        csv_point_count=csv_point_count,
                    ).ok
                    and validate_catalog_route_binding_json(
                        diagnostics,
                        selected_leg,
                    ).ok
                )

            stale_task_path = root / "stale_validated_task.json"
            write_validated_task_snapshot(
                stale_task_path,
                replace(validated_task, validated_at_sec=100.0),
            )
            stale_args = list(planner_args)
            stale_args[stale_args.index("--task-snapshot") + 1] = str(
                stale_task_path
            )
            stale_stderr = StringIO()
            with patch(
                "scripts.aufgabe04.navigation.plan_arrival_catalog_route.time.time",
                return_value=200.0,
            ), redirect_stderr(stale_stderr):
                with self.assertRaises(SystemExit) as stale_exit:
                    main(stale_args)

            future_task_path = root / "future_validated_task.json"
            write_validated_task_snapshot(
                future_task_path,
                replace(validated_task, validated_at_sec=205.0),
            )
            future_args = list(planner_args)
            future_args[future_args.index("--task-snapshot") + 1] = str(
                future_task_path
            )
            future_stderr = StringIO()
            with patch(
                "scripts.aufgabe04.navigation.plan_arrival_catalog_route.time.time",
                return_value=200.0,
            ), redirect_stderr(future_stderr):
                with self.assertRaises(SystemExit) as future_exit:
                    main(future_args)

            open_catalog_path = root / "open_catalog.json"
            write_arrival_pose_catalog(
                open_catalog_path,
                replace(catalog, frozen=False, frozen_unix_sec=None),
            )
            open_catalog_args = list(planner_args)
            open_catalog_args[open_catalog_args.index("--catalog") + 1] = str(
                open_catalog_path
            )
            open_catalog_stderr = StringIO()
            with patch(
                "scripts.aufgabe04.navigation.plan_arrival_catalog_route.time.time",
                return_value=151.0,
            ), redirect_stderr(open_catalog_stderr):
                with self.assertRaises(SystemExit) as open_catalog_exit:
                    main(open_catalog_args)

            real_graph_builder = (
                planner_module.build_required_arrival_route_graph
            )

            def mutate_catalog_after_graph(*builder_args, **builder_kwargs):
                graph = real_graph_builder(*builder_args, **builder_kwargs)
                write_arrival_pose_catalog(
                    catalog_path,
                    replace(catalog, catalog_id="concurrent_catalog"),
                )
                return graph

            concurrent_stderr = StringIO()
            with patch(
                "scripts.aufgabe04.navigation.plan_arrival_catalog_route."
                "build_required_arrival_route_graph",
                side_effect=mutate_catalog_after_graph,
            ), patch(
                "scripts.aufgabe04.navigation.plan_arrival_catalog_route.time.time",
                return_value=151.0,
            ), redirect_stderr(concurrent_stderr):
                with self.assertRaises(SystemExit) as concurrent_exit:
                    main(planner_args)

        self.assertEqual(status, 0)
        self.assertEqual(mismatched_bounds_exit.exception.code, 2)
        self.assertIn(
            "planner arena bounds differ from bound survey configuration",
            mismatched_bounds_stderr.getvalue(),
        )
        self.assertEqual(wrong_config_exit.exception.code, 2)
        self.assertIn(
            "survey configuration artifact differs from survey manifest",
            wrong_config_stderr.getvalue(),
        )
        self.assertEqual(
            execution_binding.expected_candidate_uid, "candidate_c"
        )
        self.assertEqual(catalog_bytes_after, catalog_bytes_before)
        self.assertEqual(catalog_stat_after.st_ino, catalog_stat_before.st_ino)
        self.assertEqual(
            catalog_stat_after.st_mtime_ns,
            catalog_stat_before.st_mtime_ns,
        )
        self.assertEqual(
            mission_plan.required_station_order, required_station_order
        )
        self.assertEqual(
            mission_plan.ordered_candidate_uids,
            ("candidate_c", "candidate_a", "candidate_c"),
        )
        self.assertEqual(
            mission_plan.task_snapshot.sha256,
            validated_task_snapshot_sha256(validated_task),
        )
        self.assertEqual(
            mission_plan.station_identity_registry.sha256,
            station_identity_registry_sha256(identity_registry),
        )
        self.assertEqual(
            mission_plan.route_bundle.sha256, expected_route_bundle_sha256
        )
        self.assertEqual(route_bundle_payload, expected_route_bundle_payload)
        self.assertEqual(
            mission_plan.planner_config.sha256,
            payload_sha256(planner_config_payload),
        )
        self.assertEqual(
            planner_config_payload["artifact_kind"], "planner_config"
        )
        self.assertEqual(
            planner_config_payload["start_pose"],
            {"x_m": 0.5, "y_m": 0.5, "yaw_rad": 0.0},
        )
        self.assertEqual(
            planner_config_path.name,
            f"{mission_plan.planner_config.artifact_id}.json",
        )
        self.assertEqual(
            route_bundle_path.name,
            f"{mission_plan.route_bundle.artifact_id}.json",
        )
        self.assertEqual(
            diagnostics_payload["metadata"]["optimization"]["station_order"],
            list(required_station_order),
        )
        self.assertEqual(
            diagnostics_payload["metadata"]["task_snapshot_sha256"],
            validated_task_snapshot_sha256(validated_task),
        )
        self.assertEqual(
            diagnostics_payload["metadata"]["server_order_sha256"],
            validated_task.order_sha256,
        )
        self.assertEqual(
            diagnostics_payload["metadata"]["max_task_snapshot_age_sec"],
            5.0,
        )
        self.assertEqual(
            diagnostics_payload["metadata"]["max_task_future_skew_sec"],
            2.0,
        )
        self.assertEqual(visit_payload["station_order"], list(required_station_order))
        self.assertEqual(
            visit_payload["candidate_order"],
            ["candidate_c", "candidate_a", "candidate_c"],
        )
        self.assertEqual(len(diagnostics_payload["legs"]), 3)
        self.assertEqual(
            {
                (edge["source_id"], edge["target_id"])
                for edge in costs_payload["edges"]
            },
            {
                ("mission_start", "candidate_c::face_0"),
                ("candidate_c::face_0", "candidate_a::face_0"),
                ("candidate_a::face_0", "candidate_c::face_0"),
            },
        )
        self.assertEqual(
            target_arrival_by_leg,
            (
                "candidate_c::face_0",
                "candidate_a::face_0",
                "candidate_c::face_0",
            ),
        )
        self.assertEqual(
            certificate.route_sha256,
            route_sha256,
        )
        self.assertEqual(certificate.map_bundle_sha256, map_bundle.bundle_sha256)
        self.assertEqual(
            certificate.candidate_snapshot_sha256,
            candidate_snapshot_sha256(candidate_snapshot),
        )
        self.assertEqual(certificate.waypoint_count, len(rows))
        self.assertTrue(certificate.exact_vertex_pursuit)
        self.assertTrue(all(route_binding_results))
        self.assertEqual(
            stdout_payload["mission_plan_manifest_sha256"],
            mission_plan_manifest_sha256(mission_plan),
        )
        self.assertEqual(
            stdout_payload["planner_config_sha256"],
            mission_plan.planner_config.sha256,
        )
        self.assertEqual(
            stdout_payload["route_bundle_sha256"],
            mission_plan.route_bundle.sha256,
        )
        self.assertEqual(stale_exit.exception.code, 2)
        self.assertIn("validated task snapshot is stale", stale_stderr.getvalue())
        self.assertEqual(future_exit.exception.code, 2)
        self.assertIn("validated_at_sec is in the future", future_stderr.getvalue())
        self.assertEqual(open_catalog_exit.exception.code, 2)
        self.assertIn(
            "logistics arrival-pose catalog must already be frozen",
            open_catalog_stderr.getvalue(),
        )
        self.assertEqual(concurrent_exit.exception.code, 2)
        self.assertIn("changed concurrently", concurrent_stderr.getvalue())

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
                        *SYNTHETIC_ARENA_ARGS,
                    ]
                )
            frozen = load_arrival_pose_catalog(catalog_path)
            map_bundle = freeze_map_bundle(
                map_yaml,
                semantic_map_id="map",
                planning_frame="odom",
            )
            map_bundle_path = root / "sealed_map_bundle.json"
            write_frozen_map_bundle(map_bundle_path, map_bundle)
            survey_config_payload = {
                "arena_bounds": {
                    "length_m": 20.0,
                    "width_m": 20.0,
                    "center_x_m": 0.0,
                    "center_y_m": 0.0,
                    "yaw_deg": 0.0,
                    "margin_m": 0.0,
                }
            }
            survey_config_sha256 = payload_sha256(survey_config_payload)
            survey_config_path = root / (
                f"survey_config_{survey_config_sha256}.json"
            )
            write_content_hashed_json(
                survey_config_path,
                survey_config_payload,
                hash_field="survey_config_sha256",
            )
            survey_manifest = SurveyManifest(
                schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
                manifest_id="survey_manifest_optimized_001",
                created_unix_sec=frozen.updated_unix_sec,
                session_id="session_001",
                environment="simulation",
                planning_frame="odom",
                map_bundle=artifact_reference(
                    "map_bundle", "map", map_bundle.bundle_sha256
                ),
                candidate_snapshot=artifact_reference(
                    "candidate_snapshot", "candidate_snapshot_001", "7" * 64
                ),
                environment_descriptor=artifact_reference(
                    "simulation_world", world.stem, world_hash
                ),
                survey_config=artifact_reference(
                    "survey_config",
                    f"survey_config_{survey_config_sha256[:16]}",
                    survey_config_sha256,
                ),
                calibration_profile=artifact_reference(
                    "calibration_profile", "calibration_optimized_001", "9" * 64
                ),
                arrival_pose_catalog=artifact_reference(
                    "arrival_pose_catalog",
                    frozen.catalog_id,
                    arrival_pose_catalog_sha256(frozen),
                ),
            )
            survey_manifest_path = root / "sealed_survey_manifest.json"
            write_survey_manifest(survey_manifest_path, survey_manifest)
            sealed_diagnostics = root / "sealed_diagnostics.json"
            with redirect_stdout(StringIO()):
                sealed_status = main(
                    [
                        "--catalog", str(catalog_path),
                        "--map", str(map_yaml),
                        "--world", str(world),
                        "--session-id", "session_001",
                        "--map-frame", "odom",
                        "--semantic-map-id", "map",
                        "--map-bundle-json", str(map_bundle_path),
                        "--survey-manifest", str(survey_manifest_path),
                        "--route-purpose", "survey",
                        "--start-x", "0.4",
                        "--start-y", "0.4",
                        "--route-csv", str(root / "sealed_route.csv"),
                        "--diagnostics-json", str(sealed_diagnostics),
                        "--visit-order-json", str(root / "sealed_visits.json"),
                        "--pairwise-costs-json", str(root / "sealed_costs.json"),
                        "--catalog-snapshot-json", str(root / "sealed_catalog.json"),
                        *SYNTHETIC_ARENA_ARGS,
                    ]
                )
            sealed_metadata = json.loads(sealed_diagnostics.read_text())["metadata"]

            wrong_session_manifest = replace(
                survey_manifest,
                manifest_id="survey_manifest_wrong_session",
                session_id="another_session",
            )
            wrong_session_path = root / "wrong_session_manifest.json"
            write_survey_manifest(wrong_session_path, wrong_session_manifest)
            wrong_session_stderr = StringIO()
            with redirect_stderr(wrong_session_stderr):
                with self.assertRaises(SystemExit) as wrong_session_exit:
                    main(
                        [
                            "--catalog", str(catalog_path),
                            "--map", str(map_yaml),
                            "--world", str(world),
                            "--session-id", "session_001",
                            "--map-frame", "odom",
                            "--semantic-map-id", "map",
                            "--map-bundle-json", str(map_bundle_path),
                            "--survey-manifest", str(wrong_session_path),
                            "--route-purpose", "survey",
                            "--start-x", "0.4",
                            "--start-y", "0.4",
                        ]
                    )

            wrong_world_manifest = replace(
                survey_manifest,
                manifest_id="survey_manifest_wrong_world",
                environment_descriptor=artifact_reference(
                    "simulation_world", "another_world", world_hash
                ),
            )
            wrong_world_path = root / "wrong_world_manifest.json"
            write_survey_manifest(wrong_world_path, wrong_world_manifest)
            wrong_world_stderr = StringIO()
            with redirect_stderr(wrong_world_stderr):
                with self.assertRaises(SystemExit) as wrong_world_exit:
                    main(
                        [
                            "--catalog", str(catalog_path),
                            "--map", str(map_yaml),
                            "--world", str(world),
                            "--session-id", "session_001",
                            "--map-frame", "odom",
                            "--semantic-map-id", "map",
                            "--map-bundle-json", str(map_bundle_path),
                            "--survey-manifest", str(wrong_world_path),
                            "--route-purpose", "survey",
                            "--start-x", "0.4",
                            "--start-y", "0.4",
                        ]
                    )
            with route.open() as handle:
                rows = list(csv.DictReader(handle))
            diagnostics_payload = json.loads(diagnostics.read_text())
            certificate = load_execution_route_certificate(
                Path(diagnostics_payload["metadata"]["route_certificate_path"])
            )
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
        self.assertEqual(sealed_status, 0)
        self.assertEqual(
            sealed_metadata["survey_manifest_sha256"],
            survey_manifest_sha256(survey_manifest),
        )
        self.assertEqual(wrong_session_exit.exception.code, 2)
        self.assertIn(
            "survey manifest session differs from catalog",
            wrong_session_stderr.getvalue(),
        )
        self.assertEqual(wrong_world_exit.exception.code, 2)
        self.assertIn(
            "survey manifest environment descriptor differs from catalog",
            wrong_world_stderr.getvalue(),
        )
        self.assertTrue(frozen.frozen)
        self.assertTrue(visits_payload["optimal"])
        self.assertEqual(len(visits_payload["candidate_order"]), 2)
        self.assertEqual(len(diagnostics_payload["legs"]), 2)
        self.assertEqual({row["leg_index"] for row in rows}, {"0", "1"})
        self.assertTrue(all(row["route_kind"] == "catalog_face_approach" for row in rows))
        self.assertTrue(all(row["catalog_sha256"] for row in rows))
        self.assertEqual(len(costs_payload["edges"]), 4)
        self.assertEqual(certificate.route_sha256, diagnostics_payload["metadata"]["route_csv_sha256"])
        self.assertTrue(certificate.exact_vertex_pursuit)
        self.assertEqual(
            diagnostics_payload["metadata"]["non_target_stand_keepout_policy"],
            "max(body_uncertainty_collision,lidar_minimum_standoff,"
            "frozen_candidate_keepout)+certified_tracking_tube",
        )
        self.assertTrue(
            all(
                clearance["radius_m"] >= 0.29
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
