import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.artifacts.bounded_orientation import (
    BOUNDED_ORIENTATION_POLICY, validated_bounded_orientation, validate_bounded_endpoint,
)
from scripts.aufgabe04.artifacts.backside_axis_observation import validated_backside_axis_observation
from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import BacksideAxisFrameProjection
from scripts.aufgabe04.navigation.approach.candidate_preapproach_planning import (
    compute_candidate_preapproach_plan, materialize_candidate_preapproach_plan,
)
from scripts.aufgabe04.navigation.approach.detected_stand_preapproach import validate_detected_stand_preapproach_binding
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import load_recommendation, recommendation_to_payload
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.real_robot.configuration.recommendation import build_real_viewpoint_recommendation
from scripts.aufgabe04.real_robot.candidate.approach import FacingValidationRequest, validate_facing_pose
from scripts.aufgabe04.perception.arrival_pose_estimator import arrival_pose_record_from_recommendation
from scripts.aufgabe04.stations.candidate_snapshot import new_candidate_snapshot, write_candidate_snapshot
from tests.aufgabe04.backside_axis_fixture import backside_axis_payload
from tests.aufgabe04.test_detected_station_exploration import write_free_map
from tests.aufgabe04 import test_candidate_preapproach_planning as fixtures
from tests.aufgabe04 import test_autonomous_candidate_approach as candidate_fixtures

PHYSICAL_CLEARANCE = fixtures.PHYSICAL_CLEARANCE


def bounds(center=0.0, half_degrees=8.0, count=7):
    return {"policy": BOUNDED_ORIENTATION_POLICY, "center_rad": center,
            "half_width_rad": math.radians(half_degrees), "sample_count": count}


class BoundedOrientationPlanningTest(unittest.TestCase):
    def test_contract_rejects_unbounded_malformed_or_insufficient_evidence(self):
        for changed in ({"half_width_rad": float("nan")}, {"half_width_rad": math.radians(16)},
                        {"sample_count": 6}, {"sample_count": True}, {"policy": "ignore_ambiguity"},
                        {"motion_authorized": True}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validated_bounded_orientation({**bounds(), **changed})
        with self.assertRaisesRegex(ValueError, "center differs"):
            validated_bounded_orientation(bounds(), expected_axis_rad=0.2)

    def test_one_actual_endpoint_must_cover_every_angle_and_arrival_reserve(self):
        result = validate_bounded_endpoint(bounds(), selected_normal_rad=math.pi / 2,
                                           stand_x_m=0, stand_y_m=0, stand_uncertainty_m=.02,
                                           target_x_m=0, target_y_m=.5)
        self.assertTrue(result["all_plausible_angles_supported"])
        # Each nominal interval endpoint could have a feasible dedicated route;
        # this one rasterized route target nevertheless has excessive incidence.
        target_normal = math.pi / 2 + math.radians(12)
        with self.assertRaisesRegex(ValueError, "viewing obliquity"):
            validate_bounded_endpoint(bounds(), selected_normal_rad=math.pi / 2,
                                      stand_x_m=0, stand_y_m=0, stand_uncertainty_m=.02,
                                      target_x_m=.5 * math.cos(target_normal),
                                      target_y_m=.5 * math.sin(target_normal))

    def test_backside_low_confidence_is_only_diagnostic_with_valid_bound(self):
        payload = backside_axis_payload()
        payload["axis_confidence"] = .435
        with self.assertRaisesRegex(ValueError, "axis_confidence"):
            validated_backside_axis_observation(payload)
        payload["bounded_orientation"] = bounds()
        observation = validated_backside_axis_observation(payload)
        self.assertAlmostEqual(observation.opposite_face_normal_rad, -math.pi / 2)
        payload["sample_gate_evidence"]["all_samples_lidar_associated"] = False
        with self.assertRaisesRegex(ValueError, "all_samples_lidar_associated"):
            validated_backside_axis_observation(payload)

    def test_candidate_center_uncertainty_is_not_dropped_from_viewing_budget(self):
        args = dict(selected_normal_rad=math.pi / 2, stand_x_m=0, stand_y_m=0,
                    target_x_m=0, target_y_m=.5)
        validate_bounded_endpoint(bounds(half_degrees=15), stand_uncertainty_m=0, **args)
        with self.assertRaisesRegex(ValueError, "viewing obliquity"):
            validate_bounded_endpoint(bounds(half_degrees=15), stand_uncertainty_m=.03, **args)

    def test_backside_all_angles_must_resolve_the_same_opposite_side(self):
        payload = backside_axis_payload(robot_x_m=math.cos(math.radians(35)),
                                        robot_y_m=math.sin(math.radians(35)))
        payload["bounded_orientation"] = bounds(half_degrees=10)
        with self.assertRaisesRegex(ValueError, "every angle"):
            _ = validated_backside_axis_observation(payload).opposite_face_normal_rad

    def test_projection_rotates_interval_without_shrinking_it(self):
        payload = backside_axis_payload()
        payload["bounded_orientation"] = bounds()
        source = validated_backside_axis_observation(payload)
        projected = BacksideAxisFrameProjection(
            stand_id=source.stand_id, planning_frame="map", stand_axis_rad=.4,
            stand_x_m=0, stand_y_m=0, robot_x_m=-.7 * math.sin(.4),
            robot_y_m=.7 * math.cos(.4), robot_yaw_rad=0, source_observation=source,
            source_axis_observation_path=Path("axis.json"), source_axis_observation_sha256="a" * 64,
            source_candidate_projection_path=Path("source.json"), source_candidate_projection_sha256="b" * 64,
            target_candidate_projection_path=Path("target.json"), target_candidate_projection_sha256="c" * 64,
            projection_sha256="d" * 64,
        )
        self.assertAlmostEqual(projected.bounded_orientation["center_rad"], .4)
        self.assertEqual(projected.bounded_orientation["half_width_rad"], bounds()["half_width_rad"])
        self.assertAlmostEqual(projected.opposite_face_normal_rad, .4 - math.pi / 2)

    def test_front_roundtrip_keeps_interval_and_requires_qr_binding(self):
        recommendation = build_real_viewpoint_recommendation(
            stream_id="stream", stand_id="stand", planning_frame="map",
            stand_center=Pose2D(0, 0, 0), stand_radius_m=.06, stand_uncertainty_m=.02,
            robot_pose=Pose2D(.5, 0, math.pi), stand_axis_rad=math.pi / 2,
            axis_confidence=.435, axis_sample_count=7, sensor_stamp_sec=123,
            expected_qr_id="QR_002", observed_qr_ids=("QR_002",), target_distance_m=.35,
            bounded_orientation=bounds(math.pi / 2),
        )
        payload = recommendation_to_payload(recommendation)
        self.assertEqual(payload["schema_version"], 2)
        loaded = load_recommendation(payload)
        self.assertEqual(loaded.bounded_orientation, recommendation.bounded_orientation)
        with self.assertRaisesRegex(ValueError, "schema-1.*cannot contain"):
            load_recommendation({**payload, "schema_version": 1})
        with self.assertRaisesRegex(ValueError, "schema-1.*cannot contain"):
            load_recommendation({**payload, "schema_version": 1, "bounded_orientation": None})
        stripped = dict(payload)
        stripped.pop("bounded_orientation")
        with self.assertRaisesRegex(ValueError, "schema-2.*requires"):
            load_recommendation(stripped)
        with self.assertRaisesRegex(ValueError, "schema-2.*requires"):
            load_recommendation({**payload, "bounded_orientation": None})
        with self.assertRaisesRegex(ValueError, "cannot discard bounded orientation"):
            arrival_pose_record_from_recommendation(
                loaded, candidate_uid="stand", map_yaml_sha256="a" * 64,
                corridor_length_m=.2, validated_unix_sec=124)
        payload["side_evidence"]["hard"] = False
        with self.assertRaisesRegex(ValueError, "onboard QR"):
            load_recommendation(payload)

    def test_bounded_front_identity_keeps_one_collision_checked_facing_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root, width=80, height=40, resolution=.05)
            _, bundle = load_occupancy_grid_with_bundle(map_yaml, semantic_map_id="arena", planning_frame="map")
            factory = candidate_fixtures.AutonomousCandidateApproachTest()
            candidate = factory._candidate("candidate_1", 0., 0.)
            config = factory._config(root, (candidate,))
            config = replace(config, map_yaml=map_yaml,
                             plan=replace(config.plan, map_bundle_sha256=bundle.bundle_sha256),
                             physical_clearance={"minimum_active_standoff_m": .33,
                                 "minimum_collision_standoff_m": .285,
                                 "minimum_candidate_transit_radius_m": .34,
                                 "minimum_static_inflation_m": .25})
            recommendation = build_real_viewpoint_recommendation(
                stream_id="stream", stand_id=candidate.candidate_uid, planning_frame="map",
                stand_center=Pose2D(0, 0, 0), stand_radius_m=.06, stand_uncertainty_m=.02,
                robot_pose=Pose2D(-.5, 0, 0), stand_axis_rad=math.pi / 2,
                axis_confidence=0., axis_sample_count=7, sensor_stamp_sec=123,
                expected_qr_id="QR_002", observed_qr_ids=("QR_002",), target_distance_m=.35,
                bounded_orientation=bounds(math.pi / 2),
            )
            recommendation_path = root / "recommendation.json"
            recommendation_path.write_text(json.dumps(recommendation_to_payload(recommendation)))
            result = validate_facing_pose(FacingValidationRequest(
                config=config, candidate=candidate, recommendation_path=recommendation_path,
                current_pose=Pose2D(-.5, 0, 0), output_dir=root / "facing"))
            self.assertEqual(result["bounded_orientation"], bounds(math.pi / 2))
            self.assertTrue(result["bounded_orientation_view"]["all_plausible_angles_supported"])
            self.assertTrue(result["active_stand_clearance"]["continuous_centerline_validated"])
            self.assertFalse(result["motion_to_facing_pose_authorized"])

    def test_opposite_route_materialization_and_preflight_bind_interval_to_actual_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            _, bundle = load_occupancy_grid_with_bundle(map_yaml, semantic_map_id="arena", planning_frame="map")
            candidate = fixtures.CandidatePreapproachPlanningTest._candidate("candidate_1", .5, 0.0)
            snapshot = new_candidate_snapshot(snapshot_id="snapshot", created_unix_sec=3.0,
                                             planning_frame="map", map_bundle_sha256=bundle.bundle_sha256,
                                             candidates=(candidate,))
            snapshot_path = root / "snapshot.json"
            write_candidate_snapshot(snapshot_path, snapshot)
            payload = backside_axis_payload(stand_x_m=.5, robot_x_m=.5, robot_y_m=-.7)
            payload["bounded_orientation"] = bounds(half_degrees=6)
            payload["axis_confidence"] = .435
            axis_path = root / "axis.json"
            axis_path.write_text(json.dumps(payload))
            prepared = compute_candidate_preapproach_plan(
                map_yaml=map_yaml, semantic_map_id="arena", plan=fixtures.CandidatePreapproachPlanningTest._plan(bundle.bundle_sha256),
                snapshot=snapshot, candidate_uid=candidate.candidate_uid, start=Pose2D(-.4, 0, 0),
                approach_offset_m=.45, inflation_radius_m=.25, candidate_transit_radius_m=.31,
                physical_clearance=PHYSICAL_CLEARANCE, approach_normal_rad=math.pi / 2,
            )
            outputs = materialize_candidate_preapproach_plan(
                prepared, snapshot=snapshot, snapshot_path=snapshot_path, output_dir=root / "route",
                physical_clearance=PHYSICAL_CLEARANCE, axis_observation_path=axis_path,
                approach_normal_rad=math.pi / 2,
            )
            diagnostics_path = Path(outputs["diagnostics_json"])
            diagnostics = json.loads(diagnostics_path.read_text())
            self.assertTrue(diagnostics["metadata"]["bounded_orientation_view"]["actual_endpoint_checked"])
            leg = load_route_leg(Path(outputs["route_csv"]), leg_index=0)
            status = validate_detected_stand_preapproach_binding(diagnostics_path, leg, candidate_snapshot_path=snapshot_path)
            self.assertTrue(status.ok, status.failures)
            altered = copy.deepcopy(diagnostics)
            altered["metadata"]["bounded_orientation_view"]["bounded_orientation"]["half_width_rad"] = 0
            status = validate_detected_stand_preapproach_binding(diagnostics_path, leg, candidate_snapshot_path=snapshot_path,
                                                                diagnostics_payload=altered)
            self.assertFalse(status.ok)
            self.assertTrue(any("bounded orientation" in item for item in status.failures))
