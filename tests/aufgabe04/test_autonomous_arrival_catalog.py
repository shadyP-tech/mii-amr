import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.aufgabe04.artifacts import load_survey_manifest
from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, payload_sha256, write_content_hashed_json
from scripts.aufgabe04.logistics.server_validation.artifacts import write_validated_task_snapshot
from scripts.aufgabe04.logistics.server_validation.validators import build_server_task_snapshot, validate_server_task
from scripts.aufgabe04.task_client.server_response_decoder import decode_robot_plans, decode_robot_statuses
from datetime import datetime, timezone

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame, project_candidate_snapshot_to_planning_frame
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import CandidateFrameProvenance, CandidatePoint2D
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import recommendation_to_dict
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyConfig, CoverageSurveyPlan, SurveyViewpoint, SURVEY_PLAN_SCHEMA_VERSION,
    coverage_survey_plan_sha256, stand_survey_registry_sha256, write_coverage_survey_plan, write_stand_survey_registry,
)
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.planning.map_io import freeze_map_bundle
from scripts.aufgabe04.real_robot.configuration.profile import camera_calibration_sha256, real_robot_profile_sha256, write_camera_calibration, write_real_robot_profile
from scripts.aufgabe04.real_robot.configuration.recommendation import build_real_viewpoint_recommendation
from scripts.aufgabe04.stations.arrival_pose_catalog import load_arrival_pose_catalog
from scripts.aufgabe04.stations.autonomous_arrival_catalog import AutonomousCatalogInputs, promote_autonomous_arrival_catalog
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256, new_candidate_snapshot, write_candidate_snapshot
from scripts.aufgabe04.stations.server_identity_binding import seal_server_qr_mapping_evidence, write_observed_identities, write_server_qr_mapping_evidence
from tests.aufgabe04.test_candidate_frame_projection import _registry, _frozen_candidate
from tests.aufgabe04.test_real_robot_pipeline import calibration, robot_profile
from tests.aufgabe04.test_server_identity_binding import mapping_payload

ROOT = Path(__file__).resolve().parents[2]


def fixture(root, *, obstacle=(4.0, 4.0), now=None):
    now = time.time() if now is None else now
    (root / "map.pgm").write_bytes(b"P5\n60 60\n255\n" + bytes([255]) * 3600)
    (root / "map.yaml").write_text("image: map.pgm\nresolution: 0.1\norigin: [0, 0, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.2\n")
    bundle = freeze_map_bundle(root / "map.yaml", semantic_map_id="arena", planning_frame="map")
    cell = GridCell(1, 1)
    plan = CoverageSurveyPlan(SURVEY_PLAN_SCHEMA_VERSION, "survey", "map", bundle.bundle_sha256, ArenaBounds(length_m=6.0, width_m=6.0, center_x_m=3.0, center_y_m=3.0, margin_m=0.1), CoverageSurveyConfig(), (SurveyViewpoint("survey_vp_001", Pose2D(2.0, 3.4), cell, (cell,)),), (cell,), (cell,), 1.0)
    write_coverage_survey_plan(root / "plan.json", plan)
    candidates = []
    for uid, point in (("survey_candidate_0003", (2.0, 2.0)), ("survey_candidate_0004", obstacle)):
        provenance = CandidateFrameProvenance.from_frozen_map_observation(map_frame="map", odom_frame="odom", frozen_map_point=CandidatePoint2D(*point), frozen_map_from_odom=PlanarTransform2D(0, 0, 0), source_evidence_id=("a" if uid.endswith("3") else "b") * 64)
        item = _registry(*point, provenance).candidates[0]
        candidates.append(replace(item, candidate_uid=uid, source_observation_ids=(f"obs_{uid}",)))
    registry = replace(_registry(2, 2, candidates[0].frame_provenance), candidates=tuple(candidates), map_bundle_sha256=bundle.bundle_sha256)
    reg_sha = stand_survey_registry_sha256(registry)
    write_stand_survey_registry(root / "registry.json", registry)
    frozen = []
    for item in candidates:
        candidate = _frozen_candidate(item.x_m, item.y_m, source_registry_sha256=reg_sha)
        frozen.append(replace(candidate, candidate_uid=item.candidate_uid, source=replace(candidate.source, observation_ids=item.source_observation_ids)))
    source = new_candidate_snapshot(snapshot_id="source", created_unix_sec=now-20, planning_frame="map", map_bundle_sha256=bundle.bundle_sha256, candidates=frozen)
    confirmed = replace(source, candidates=source.candidates[:1])
    write_candidate_snapshot(root / "source.json", source)
    write_candidate_snapshot(root / "confirmed.json", confirmed)
    frames = []
    for name, transform in (("observed", PlanarTransform2D(0, 0, 0)), ("target", PlanarTransform2D(0.1, 0.0, 0.0))):
        frame = CandidatePlanningFrame(Pose2D(2.1, 3.4), transform)
        projection = project_candidate_snapshot_to_planning_frame(source, registry, frame)
        snapshot_path = root / f"{name}_snapshot.json"
        snapshot_sha = write_candidate_snapshot(snapshot_path, projection.projected_snapshot)
        projection_path = root / f"{name}_projection.json"
        projection_sha = write_content_hashed_json(projection_path, {**projection.to_evidence(), "source_candidate_snapshot_path": str(root / "source.json"), "projected_candidate_snapshot_path": str(snapshot_path)}, hash_field="candidate_frame_projection_sha256")
        frames.append({"camera_candidate_snapshot_path": str(snapshot_path), "camera_candidate_snapshot_sha256": snapshot_sha, "candidate_frame_projection_path": str(projection_path), "candidate_frame_projection_sha256": projection_sha})
    (root / "arena_real.json").write_text('{"test_site":true}')
    cal = calibration()
    cal_sha = write_camera_calibration(root / "calibration.json", cal)
    profile = robot_profile(cal_sha, hashlib.sha256((root / "arena_real.json").read_bytes()).hexdigest())
    profile_sha = write_real_robot_profile(root / "robot.json", profile)
    recommendation = build_real_viewpoint_recommendation(stream_id="session_camera", stand_id=confirmed.candidate_uids[0], planning_frame="map", stand_center=Pose2D(2, 2), stand_radius_m=0.06, stand_uncertainty_m=0.02, robot_pose=Pose2D(2.0, 3.4), stand_axis_rad=0, axis_confidence=0.95, axis_sample_count=7, sensor_stamp_sec=now-5, expected_qr_id="qr_Mixed_a", observed_qr_ids=("qr_Mixed_a",), target_distance_m=0.50, observation_unix_sec=now-5)
    (root / "recommendation.json").write_text(json.dumps(recommendation_to_dict(recommendation)))
    observed_sha = write_observed_identities(root / "observed_identities.json", candidate_snapshot=confirmed, observed_qr_by_candidate={confirmed.candidate_uids[0]: "qr_Mixed_a"}, session_id="session", observed_unix_sec=now-3)
    _, plans = mapping_payload(now)
    evidence = seal_server_qr_mapping_evidence(plans, robot_id="Robot_Test_01", captured_unix_sec=now-1)
    write_server_qr_mapping_evidence(root / "server.json", evidence)
    facing = {
        "schema_version": 1, "catalog_kind": "real_autonomous_stand_facing_poses", "session_id": "session", "planning_frame": "map",
        "map_bundle_sha256": bundle.bundle_sha256, "coverage_plan_sha256": coverage_survey_plan_sha256(plan),
        "candidate_snapshot_sha256": candidate_snapshot_sha256(source), "confirmed_candidate_snapshot_sha256": candidate_snapshot_sha256(confirmed),
        "source_registry_sha256": reg_sha, "station_identity_registry_sha256": None,
        "observed_station_identities_sha256": observed_sha, "calibration_profile_sha256": cal_sha, "robot_profile_sha256": profile_sha,
        "expected_stand_count": 1, "stand_count": 1, "candidate_pool_count": 2,
        "records": [{"candidate_uid": confirmed.candidate_uids[0], "qr_id": "qr_Mixed_a", "recommendation_json": str(root / "recommendation.json"), "camera_recommendation_sha256": hashlib.sha256((root / "recommendation.json").read_bytes()).hexdigest(), "calibration_profile_sha256": cal_sha, "robot_profile_sha256": profile_sha, "active_stand_clearance": {"minimum_active_standoff_m": 0.4}, **frames[0]}],
    }
    write_content_hashed_json(root / "facing.json", facing, hash_field="stand_facing_catalog_sha256")
    return AutonomousCatalogInputs(
        facing_catalog=root / "facing.json", candidate_snapshot=root / "source.json", confirmed_candidate_snapshot=root / "confirmed.json",
        observed_identities=root / "observed_identities.json", source_stand_registry=root / "registry.json", coverage_plan=root / "plan.json",
        target_frame_projection=root / "target_projection.json", map_yaml=root / "map.yaml", semantic_map_id="arena",
        robot_profile=root / "robot.json", camera_calibration=root / "calibration.json", physical_site=root / "arena_real.json",
        server_qr_mapping_evidence=root / "server.json", server_robot_id="Robot_Test_01", output_dir=root / "promoted",
    ), now


class AutonomousArrivalCatalogTests(unittest.TestCase):
    def test_real_validators_transform_freeze_and_preserve_full_obstacle_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, now = fixture(Path(directory))
            result = promote_autonomous_arrival_catalog(inputs, now_sec=now)
            catalog = load_arrival_pose_catalog(inputs.output_dir / "arrival_pose_catalog.json")
            self.assertTrue(catalog.frozen)
            self.assertEqual(result["stand_count"], 1)
            self.assertEqual(result["obstacle_count"], 2)
            self.assertEqual(catalog.records[0].stand_id, "station_Mixed_a")
            self.assertAlmostEqual(catalog.records[0].stand.x_m, 2.1)
            self.assertAlmostEqual(catalog.records[0].arrival_pose.x_m, 2.1)
            self.assertNotEqual(catalog.provenance.obstacle_candidate_snapshot_sha256, catalog.provenance.candidate_snapshot_sha256)
            manifest = load_survey_manifest(inputs.output_dir / "survey_manifest.json")
            self.assertEqual(manifest.arrival_pose_catalog.sha256, result["catalog_sha256"])

    def test_actual_offline_promotion_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, _ = fixture(Path(directory))
            argv = [sys.executable, "-m", "scripts.aufgabe04.stations.promote_autonomous_arrival_catalog"]
            for name, value in vars(inputs).items():
                if value is not None:
                    argv.extend(["--" + name.replace("_", "-"), str(value)])
            result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["frozen"])

    def test_actual_logistics_route_requires_and_uses_the_bound_full_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, now = fixture(root)
            promote_autonomous_arrival_catalog(inputs, now_sec=now)
            status, plans = mapping_payload(now)
            task = validate_server_task(build_server_task_snapshot(
                robot_id="Robot_Test_01", scanned_qr_id="qr_Mixed_a",
                statuses=decode_robot_statuses(status), plans=decode_robot_plans(plans),
            ), local_station_ids=("station_Mixed_a",), now=datetime.fromtimestamp(now, timezone.utc))
            write_validated_task_snapshot(root / "task.json", task)
            output = inputs.output_dir
            argv = [sys.executable, "-m", "scripts.aufgabe04.navigation.missions.plan_arrival_catalog_route",
                "--route-purpose", "logistics", "--catalog", str(output / "arrival_pose_catalog.json"),
                "--map", str(inputs.map_yaml), "--map-frame", "map", "--semantic-map-id", "arena",
                "--map-bundle-json", str(output / "map_bundle.json"),
                "--candidate-snapshot", str(output / "candidate_snapshot.json"),
                "--station-identity-registry", str(output / "station_identity_registry.json"),
                "--survey-manifest", str(output / "survey_manifest.json"),
                "--task-snapshot", str(root / "task.json"), "--robot-id", "Robot_Test_01",
                "--arena-length-m", "6", "--arena-width-m", "6", "--arena-center-x-m", "3", "--arena-center-y-m", "3", "--arena-margin-m", "0.1",
                "--start-x", "4.1", "--start-y", "5.0", "--start-yaw", "0",
                "--route-csv", str(root / "route.csv"), "--diagnostics-json", str(root / "route_diagnostics.json")]
            missing = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(missing.returncode, 2, missing.stderr)
            self.assertIn("--obstacle-candidate-snapshot", missing.stderr)
            self.assertFalse((root / "route.csv").exists())
            wrong = subprocess.run(argv + ["--obstacle-candidate-snapshot", str(output / "candidate_snapshot.json")], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(wrong.returncode, 2, wrong.stderr)
            self.assertIn("hash differs", wrong.stderr)
            successful = subprocess.run(argv + ["--obstacle-candidate-snapshot", str(output / "obstacle_candidate_snapshot.json")], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(successful.returncode, 0, successful.stderr)
            diagnostics = json.loads((root / "route_diagnostics.json").read_text())
            self.assertEqual(diagnostics["metadata"]["unconfirmed_obstacle_count"], 1)
            self.assertEqual(diagnostics["metadata"]["obstacle_candidate_snapshot_sha256"], load_arrival_pose_catalog(output / "arrival_pose_catalog.json").provenance.obstacle_candidate_snapshot_sha256)

    def test_unconfirmed_obstacle_blocks_exact_terminal_corridor(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, now = fixture(Path(directory), obstacle=(2.0, 2.8))
            with self.assertRaisesRegex(ValueError, "fixed target/corridor"):
                promote_autonomous_arrival_catalog(inputs, now_sec=now)
            self.assertFalse(inputs.output_dir.exists())

    def test_resealed_non_qr_stale_sensor_and_foreign_frame_evidence_rejected(self):
        for mutation in ("manual_side", "stale_sensor", "future_sensor", "refreshed_observation", "observed_odom", "target_odom"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs, now = fixture(root)
                facing = load_content_hashed_json(root / "facing.json", hash_field="stand_facing_catalog_sha256")
                if mutation.endswith("odom"):
                    path = root / ("observed_projection.json" if mutation == "observed_odom" else "target_projection.json")
                    projection = load_content_hashed_json(path, hash_field="candidate_frame_projection_sha256")
                    projection["planning_frame_admission"]["odom_frame"] = "other_robot_odom"
                    path.unlink()
                    digest = write_content_hashed_json(path, projection, hash_field="candidate_frame_projection_sha256")
                    if mutation == "observed_odom":
                        facing["records"][0]["candidate_frame_projection_sha256"] = digest
                else:
                    path = root / "recommendation.json"
                    recommendation = json.loads(path.read_text())
                    if mutation == "manual_side":
                        recommendation["side_evidence"].update(kind="manual", provenance="manual/operator")
                    elif mutation == "stale_sensor":
                        recommendation["sensor_stamp_sec"] = now - 10000
                    elif mutation == "future_sensor":
                        recommendation["sensor_stamp_sec"] = now + 5
                    else:
                        recommendation["observation_unix_sec"] = now - 1
                    path.write_text(json.dumps(recommendation))
                    facing["records"][0]["camera_recommendation_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                (root / "facing.json").unlink()
                write_content_hashed_json(root / "facing.json", facing, hash_field="stand_facing_catalog_sha256")
                with self.assertRaisesRegex(ValueError, "onboard QR|sensor stamp|frame identities"):
                    promote_autonomous_arrival_catalog(inputs, now_sec=now)
                self.assertFalse(inputs.output_dir.exists())

    def test_source_mutations_and_missing_binding_never_publish(self):
        for mutation in ("map", "recommendation", "projection", "calibration", "registry", "observations", "missing_projection", "stale"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                inputs, now = fixture(Path(directory))
                root = Path(directory)
                if mutation == "map":
                    (root / "map.pgm").write_bytes(b"P5\n60 60\n255\n" + bytes([0]) * 3600)
                elif mutation == "recommendation":
                    data = json.loads((root / "recommendation.json").read_text())
                    data["stand_id"] = "other"
                    (root / "recommendation.json").write_text(json.dumps(data))
                elif mutation in ("calibration", "projection", "registry"):
                    path = {"calibration": root / "calibration.json", "projection": root / "observed_projection.json", "registry": root / "registry.json"}[mutation]
                    data = json.loads(path.read_text())
                    data["schema_version"] = 99
                    path.write_text(json.dumps(data))
                elif mutation == "observations":
                    data = load_content_hashed_json(root / "observed_identities.json", hash_field="observed_station_identities_sha256")
                    data["observed_qr_by_candidate"]["survey_candidate_0003"] = "another_qr"
                    (root / "observed_identities.json").unlink()
                    write_content_hashed_json(root / "observed_identities.json", data, hash_field="observed_station_identities_sha256")
                elif mutation == "missing_projection":
                    data = load_content_hashed_json(root / "facing.json", hash_field="stand_facing_catalog_sha256")
                    del data["records"][0]["candidate_frame_projection_path"]
                    (root / "facing.json").unlink()
                    write_content_hashed_json(root / "facing.json", data, hash_field="stand_facing_catalog_sha256")
                else:
                    now += 301
                with self.assertRaises((ValueError, KeyError)):
                    promote_autonomous_arrival_catalog(inputs, now_sec=now)
                self.assertFalse(inputs.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
