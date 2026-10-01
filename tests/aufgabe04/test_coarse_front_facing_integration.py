"""Coarse front viewing still needs one real collision-checked candidate route."""

from dataclasses import replace
import json
import math

import pytest

from scripts.aufgabe04.artifacts.bounded_orientation import COARSE_FRONT_ORIENTATION_POLICY
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    load_recommendation,
    recommendation_to_payload,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.perception.arrival_pose_estimator import arrival_pose_record_from_recommendation
from scripts.aufgabe04.real_robot.candidate.approach import FacingValidationRequest, validate_facing_pose
from scripts.aufgabe04.real_robot.configuration.recommendation import build_real_viewpoint_recommendation
from tests.aufgabe04 import test_autonomous_candidate_approach as candidate_fixtures
from tests.aufgabe04.test_bounded_orientation_planning import bounds


@pytest.fixture
def coarse_front(tmp_path):
    factory = candidate_fixtures.AutonomousCandidateApproachTest()
    candidate = factory._candidate("candidate_1", 0., 0.)
    config = factory._config(tmp_path, (candidate,))
    # Isolate actual endpoint clearance: a blocked final pose cannot snap to
    # a different cell in these integration controls.
    config = replace(config, plan=replace(config.plan,
        config=replace(config.plan.config, snap_radius_m=0.)))
    interval = {**bounds(-math.pi / 2., half_degrees=19.9),
                "policy": COARSE_FRONT_ORIENTATION_POLICY}
    recommendation = build_real_viewpoint_recommendation(
        stream_id="stream", stand_id=candidate.candidate_uid, planning_frame="map",
        stand_center=Pose2D(0., 0.), stand_radius_m=candidate.geometry.radius_m,
        stand_uncertainty_m=candidate.geometry.uncertainty_m,
        robot_pose=Pose2D(-.9, 0.), stand_axis_rad=math.pi / 2.,
        axis_confidence=0., axis_sample_count=7, sensor_stamp_sec=123.,
        expected_qr_id="QR_001", observed_qr_ids=("QR_001",),
        target_distance_m=.35, bounded_orientation=interval,
        observation_unix_sec=123.)
    path = tmp_path / "coarse_front_recommendation.json"
    path.write_text(json.dumps(recommendation_to_payload(recommendation)))
    request = FacingValidationRequest(config, candidate, path, Pose2D(-.9, 0.), tmp_path / "facing")
    return factory, request, recommendation, interval


def test_real_facing_route_preserves_coarse_interval_and_all_clearance_gates(coarse_front):
    _, request, recommendation, interval = coarse_front
    result = validate_facing_pose(request)
    view = result["bounded_orientation_view"]
    assert recommendation.axis_sample_count == 7 and recommendation.axis_confidence == 0.
    assert result["bounded_orientation"] == interval == view["bounded_orientation"]
    assert view["all_plausible_angles_supported"] and view["actual_endpoint_checked"]
    assert view["maximum_view_obliquity_rad"] == pytest.approx(math.radians(30.))
    # The accepted budget includes the original 2 cm target uncertainty and
    # 3 cm terminal reserve, rather than testing the 19.9 degree half-width alone.
    expected_worst = math.radians(19.9) + math.asin(.05 / .35)
    assert view["worst_case_view_obliquity_rad"] == pytest.approx(expected_worst)
    assert view["stand_center_uncertainty_m"] == .02
    assert view["terminal_position_reserve_m"] == .03
    assert result["active_stand_clearance"]["active_stand_in_planning_costmap"]
    assert result["active_stand_clearance"]["continuous_centerline_validated"]
    assert not result["motion_to_facing_pose_authorized"] and not view["motion_authorized"]
    diagnostics = json.loads((request.output_dir / "facing_pose_validation_diagnostics.json").read_text())
    assert diagnostics["metadata"]["motion_authorized"] is False
    assert diagnostics["metadata"]["bounded_orientation_view"] == view
    assert (request.output_dir / "facing_pose_validation_route.csv").is_file()


def test_coarse_front_cannot_override_static_map_obstacle(coarse_front):
    _, request, _, _ = coarse_front
    image = request.config.map_yaml.with_name("map.pgm")
    tokens = image.read_text().split()
    width, height = int(tokens[1]), int(tokens[2])
    cells = tokens[4:]
    # The fixture map has origin (-5,-5), resolution .1. An occupied column
    # crosses x=-.35, the exact facing target; current robot x=-.9 stays free.
    for row in range(height):
        cells[row * width + 46] = "0"
    image.write_text(f"P2\n{width} {height}\n255\n" + " ".join(cells) + "\n")
    _, bundle = load_occupancy_grid_with_bundle(request.config.map_yaml,
        semantic_map_id=request.config.semantic_map_id, planning_frame=request.config.planning_frame)
    config = replace(request.config, plan=replace(request.config.plan, map_bundle_sha256=bundle.bundle_sha256))
    with pytest.raises(ValueError, match="not A\\*-reachable"):
        validate_facing_pose(replace(request, config=config))
    assert not (request.output_dir / "facing_pose_validation_route.csv").exists()


def test_coarse_front_cannot_override_neighbor_keepout(coarse_front):
    factory, request, recommendation, _ = coarse_front
    target = recommendation.material_target.pose
    neighbor = factory._candidate("neighbor", target.x_m, target.y_m)
    config = replace(request.config, snapshot=replace(request.config.snapshot,
        candidates=(*request.config.snapshot.candidates, neighbor)))
    with pytest.raises(ValueError, match="not A\\*-reachable"):
        validate_facing_pose(replace(request, config=config))
    assert len(config.snapshot.candidates) == 2
    assert not (request.output_dir / "facing_pose_validation_route.csv").exists()


def test_coarse_candidate_facing_does_not_enter_legacy_arrival_catalog(coarse_front):
    _, request, recommendation, _ = coarse_front
    assert validate_facing_pose(request)["bounded_orientation_view"]["all_plausible_angles_supported"]
    with pytest.raises(ValueError, match="cannot discard bounded orientation"):
        arrival_pose_record_from_recommendation(recommendation,
            candidate_uid=request.candidate.candidate_uid, map_yaml_sha256="a" * 64,
            corridor_length_m=.2, validated_unix_sec=124., axis_sample_count=7)


@pytest.mark.parametrize("schema_version", (4, 5))
def test_coarse_front_cannot_masquerade_as_retained_backside(coarse_front, schema_version):
    _, _, recommendation, _ = coarse_front
    payload = recommendation_to_payload(recommendation)
    payload["schema_version"] = schema_version
    payload["axis_measurement"] = {
        "policy": ("retained_backside_current_qr_facing" if schema_version == 4
                   else "projected_retained_backside_current_qr_facing"),
        "current_angle_refit": False,
    }
    with pytest.raises(ValueError, match="coarse front orientation requires a schema-2"):
        load_recommendation(payload)
