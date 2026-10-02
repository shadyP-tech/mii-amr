"""Fresh LiDAR surface estimates remain bound through route seal and preflight."""

import json
import math
from pathlib import Path

import pytest

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.candidate_inspection_view import (
    INSPECTION_VIEW_BEARING_MODE, write_candidate_inspection_view,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_planning import (
    compute_candidate_preapproach_plan, materialize_candidate_preapproach_plan,
)
from scripts.aufgabe04.navigation.approach.detected_stand_preapproach import (
    validate_detected_stand_preapproach_binding,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.real_robot.candidate.current_lidar_targets import capture_current_lidar_targets
from tests.aufgabe04 import test_candidate_preapproach_planning as planning_fixtures
from tests.aufgabe04.test_lidar_mount_cohort_integration import production_capture, recovery


@pytest.fixture
def current_route(tmp_path, recovery):
    fixture = planning_fixtures.CandidatePreapproachPlanningTest()
    candidate, snapshot, snapshot_path, _ = fixture._materialization_fixture(tmp_path)
    plan = fixture._plan(snapshot.map_bundle_sha256)
    recovery.source.snapshot = snapshot
    recovery.source.plan = plan
    # The original center is (0.5, 0); actual scans support (0.64, 0).
    # Keep the source capture pose consistent with the admitted planning pose.
    frame = CandidatePlanningFrame(Pose2D(.04, 0., 0.), PlanarTransform2D(.64, 0., 0.))
    production_capture(recovery)
    estimates, support = capture_current_lidar_targets(
        recovery.source, recovery.effects, frame, (candidate.candidate_uid,), tmp_path / "support",
    )
    estimate = estimates[candidate.candidate_uid]
    assert math.dist((estimate["x_m"], estimate["y_m"]),
                     (candidate.geometry.x_m, candidate.geometry.y_m)) > .10
    prepared = compute_candidate_preapproach_plan(
        map_yaml=tmp_path / "map.yaml", semantic_map_id="arena", plan=plan,
        snapshot=snapshot, candidate_uid=candidate.candidate_uid, start=frame.current_pose,
        approach_offset_m=.70, inflation_radius_m=.25, candidate_transit_radius_m=.31,
        physical_clearance=planning_fixtures.PHYSICAL_CLEARANCE,
        inspection_view_normal_rad=math.pi, validated_target_center=estimate,
    )
    view_path = tmp_path / "inspection.json"
    write_candidate_inspection_view(
        view_path, snapshot=snapshot, candidate_uid=candidate.candidate_uid,
        start=frame.current_pose, view_normal_rad=math.pi, purpose="current_lidar_target",
        view_index=0, validated_target_center=estimate,
        current_lidar_targets_path=Path(support["evidence_path"]),
    )
    outputs = materialize_candidate_preapproach_plan(
        prepared, snapshot=snapshot, snapshot_path=snapshot_path, output_dir=tmp_path / "route",
        physical_clearance=planning_fixtures.PHYSICAL_CLEARANCE, inspection_view_path=view_path,
    )
    path = Path(outputs["diagnostics_json"])
    return dict(path=path, payload=json.loads(path.read_text()),
        leg=load_route_leg(Path(outputs["route_csv"]), 0), snapshot_path=snapshot_path,
        estimate=estimate, support_path=Path(support["evidence_path"]), outputs=outputs)


def validate(route):
    return validate_detected_stand_preapproach_binding(
        route["path"], route["leg"], candidate_snapshot_path=route["snapshot_path"],
        diagnostics_payload=route["payload"],
    )


def test_current_surface_route_seals_and_passes_real_preflight_without_axis(current_route):
    result = validate(current_route)
    assert result.ok, result.failures
    metadata = current_route["payload"]["metadata"]
    assert metadata["validated_target_center"] == current_route["estimate"]
    assert metadata["approach_bearing_mode"] == INSPECTION_VIEW_BEARING_MODE
    assert metadata["head_alignment_verified"] is False
    assert "axis_observation_json" not in metadata
    assert Path(current_route["outputs"]["route_certificate_json"]).is_file()


@pytest.mark.parametrize("change", ["omit", "replace", "axis"])
def test_preflight_rejects_unbound_target_or_claimed_axis(current_route, change):
    metadata = current_route["payload"]["metadata"]
    if change == "omit":
        metadata.pop("validated_target_center")
    elif change == "replace":
        metadata["validated_target_center"]["x_m"] += .01
    else:
        metadata["axis_observation_json"] = metadata["inspection_view_json"]
    result = validate(current_route)
    assert not result.ok
    assert any("inspection view validation failed" in failure for failure in result.failures)


def test_preflight_rechecks_pinned_current_scan_evidence(current_route):
    path = current_route["support_path"]
    data = json.loads(path.read_text())
    data["assessed_at_unix_sec"] += .01
    path.write_text(json.dumps(data))
    result = validate(current_route)
    assert not result.ok
    assert any("inspection view validation failed" in failure for failure in result.failures)
