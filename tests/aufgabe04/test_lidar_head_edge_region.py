"""LiDAR priors reduce locator clutter without manufacturing border evidence."""

from dataclasses import replace
import math
from types import SimpleNamespace
from unittest.mock import Mock, patch
import hashlib
import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.stand_axis.candidate_head_search import CandidateHeadSearch
from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import acquire_cold_head_proposal
from scripts.aufgabe04.perception.stand_axis.head_geometry_acquisition import estimate_current_head_geometry
from scripts.aufgabe04.perception.stand_axis.lidar_head_edge_region import (
    LidarHeadEdgeRegion, project_lidar_candidate_head_region,
)
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis_handoff.models import RigidTransform
from tests.aufgabe04.test_physical_head_pipeline import profile, head_image, estimate
from tests.aufgabe04.test_head_cold_acquisition import frame_with_heads


def project(profile, **changes):
    # Map +x is forward, +z up; camera optical +z forward, +y down.
    args = dict(candidate_xy=(.6, 0.),
        camera_from_map=RigidTransform("camera", "map", (0., .12, -.05), (.5, -.5, .5, .5)),
        model_profile=profile, intrinsics=(640., 640., 400., 300.), image_shape=(600, 800),
        position_uncertainty_m=.02, surface_center_margin_m=.06,
        association=SimpleNamespace(associated=True, eligible_cluster_count=1, scan_stamp_sec=10.),
        image_stamp_sec=10., now_sec=10.1, max_sensor_age_sec=.5, sync_tolerance_sec=.15)
    args.update(changes)
    return project_lidar_candidate_head_region(**args)


def test_projection_covers_unknown_yaw_and_uncertainty_in_current_camera(profile):
    region, info = project(profile)
    assert region is not None, info
    assert info["applied"] and not info["pixel_depth_measured"]
    assert not info["supplies_corners"] and info["raw_edges_unchanged"]
    for yaw in np.linspace(-math.pi, math.pi, 33):
        for dx in (-.02, .02):
            for side in (-1., 1.):
                x = .6 + dx + side * .039 * math.cos(yaw)
                y = side * .039 * math.sin(yaw)
                for z in (.132, .21):
                    p = ImagePoint(400.-640.*y/(x-.05), 300.+640.*(.12-z)/(x-.05))
                    assert region.contains((p,))
    # Same saved candidate, new robot viewpoint: mask must move in current image.
    shifted, _ = project(profile,
        camera_from_map=RigidTransform("camera", "map", (.08, .12, -.05), (.5, -.5, .5, .5)))
    assert shifted.bounds[0] > region.bounds[0]
    wider, _ = project(profile, position_uncertainty_m=.06)
    assert wider.bounds[0] <= region.bounds[0] and wider.bounds[2] >= region.bounds[2]
    assert wider.bounds[1] <= region.bounds[1] and wider.bounds[3] >= region.bounds[3]


@pytest.mark.parametrize("changes", [
    {"association": None},
    {"association": SimpleNamespace(associated=False, eligible_cluster_count=1)},
    {"association": SimpleNamespace(associated=True, eligible_cluster_count=2)},
    {"now_sec": 11.}, {"now_sec": 9.9}, {"image_stamp_sec": 9.8},
    {"image_stamp_sec": math.nan}, {"sync_tolerance_sec": 0.},
    {"candidate_xy": (-.6, 0.)}, {"candidate_xy": (.06, 0.)},
    {"candidate_xy": (.6, 10.)}, {"candidate_xy": (math.nan, 0.)},
    {"position_uncertainty_m": -1.}, {"intrinsics": (0., 640., 400., 300.)},
])
def test_invalid_or_uncorroborated_context_never_masks(profile, changes):
    region, info = project(profile, **changes)
    assert region is None and not info["applied"]
    assert info["reason"]


def test_locator_mask_preserves_exact_raw_evidence_and_coordinates():
    region = LidarHeadEdgeRegion((100, 140), (20, 30, 80, 90))
    raw = np.full(region.shape, 255, np.uint8)
    before = raw.copy()
    locator = region.locator_edges(raw)
    assert np.count_nonzero(locator) == 60*60
    np.testing.assert_array_equal(locator[30:90, 20:80], raw[30:90, 20:80])
    np.testing.assert_array_equal(raw, before)
    assert not np.shares_memory(locator, raw)
    with pytest.raises(ValueError, match="exact processing image"):
        region.locator_edges(np.zeros((50, 70), np.uint8))


def test_remote_clutter_is_removed_before_contour_budget_and_raw_fit_is_unchanged():
    frame = frame_with_heads()
    for y in range(250, 335, 40):
        for x in range(20, 485, 40):
            cv2.rectangle(frame, (x, y), (x+28, y+28), (255, 255, 255), -1)
    raw = cv2.Canny(frame, 20, 60)
    original = raw.copy()
    region = LidarHeadEdgeRegion(raw.shape, (115, 55, 265, 205))
    search = CandidateHeadSearch((190., 130.), 100., edge_region=region)
    module = "scripts.aufgabe04.perception.stand_axis.head_cold_acquisition"
    from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
    with patch(module+".MAX_CONTOURS", 20):
        unfiltered = acquire_cold_head_proposal(cv2, frame, raw_edges=raw)
        with patch(module+".refine_projected_head_border", wraps=refine_projected_head_border) as refine:
            filtered = acquire_cold_head_proposal(cv2, frame, raw_edges=raw, candidate_search=search)
    assert unfiltered.reason == "head_cold_acquisition_contour_budget_exceeded"
    assert filtered.proposal is not None, filtered.reason
    assert abs(filtered.proposal.center_u_px-190.) < 3.
    assert refine.called
    assert all(call.args[1] is raw for call in refine.call_args_list)
    info = filtered.joint_border_diagnostics["lidar_edge_region"]
    assert info["retained_edge_pixels"] < info["input_edge_pixels"] / 3
    np.testing.assert_array_equal(raw, original)


def test_mask_does_not_close_missing_border_or_resolve_in_region_ambiguity():
    frame = np.zeros((480, 640, 3), np.uint8)
    cv2.line(frame, (140, 80), (240, 80), (255, 255, 255), 2)
    cv2.line(frame, (140, 80), (140, 180), (255, 255, 255), 2)
    cv2.line(frame, (240, 80), (240, 180), (255, 255, 255), 2)
    region = LidarHeadEdgeRegion(frame.shape[:2], (120, 60, 260, 181))
    assert acquire_cold_head_proposal(cv2, frame,
        candidate_search=CandidateHeadSearch((190., 130.), 100., edge_region=region)).proposal is None
    frame = frame_with_heads((((190, 130), (100, 100), 0), ((340, 130), (100, 100), 0)))
    region = LidarHeadEdgeRegion(frame.shape[:2], (120, 60, 410, 200))
    result = acquire_cold_head_proposal(cv2, frame,
        candidate_search=CandidateHeadSearch((265., 130.), 100., edge_region=region))
    assert result.proposal is None and result.reason == "head_proposal_ambiguous"


def test_tracked_fit_cannot_bypass_candidate_region(profile):
    frame, _ = head_image(profile, angle=35.)
    fitted, debug = estimate(profile, frame)
    assert fitted.usable
    region = LidarHeadEdgeRegion(frame.shape[:2], (10, 10, 150, 150))
    result, artifacts = estimate_current_head_geometry(cv2, frame, model_profile=profile,
        candidate_search=CandidateHeadSearch((80., 80.), 100., edge_region=region),
        pose_hint=debug.model_pose, estimator=Mock(return_value=(fitted, debug)),
        camera_fx_px=640., camera_fy_px=640., camera_cx_px=400., camera_cy_px=300.)
    assert not result.usable and result.corners is None
    assert result.reason == "head_border_outside_lidar_candidate_region"
    assert artifacts.model_pose is None and artifacts.head_outer_recovery is None


@pytest.mark.parametrize("metric_size", (False, True))
def test_recorded_displaced_head_survives_early_background_filtering(profile, metric_size):
    from scripts.aufgabe04.perception.camera_calibration import CameraCalibration, rectify_bgr_frame
    from scripts.aufgabe04.perception.stand_axis_handoff.geometry import transform_point
    from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
    from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
    from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics
    from scripts.aufgabe04.real_robot.observer.current_scan_head_proposal_filter import CurrentScanHeadProposalFilter

    root = Path(__file__).parent / "fixtures/lidar_head_region"
    data = json.loads((root / "frame_000025.json").read_text())
    image = (root / "frame_000025.jpg").read_bytes()
    assert hashlib.sha256(image).hexdigest() == data["image_sha256"]
    c = data["sensors"]["camera_info"]
    cal = CameraCalibration(c["width"], c["height"], "camera", tuple(c["k"]),
                           tuple(c["d"]), tuple(c["r"]), tuple(c["p"]))
    frame = rectify_bgr_frame(cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR), cal, cv2, np)
    def tf(parent, child):
        t = next(t for t in data["tf_samples"] if t["target_frame"] == parent and t["source_frame"] == child)
        return RigidTransform(parent, child, tuple(t["translation_xyz_m"]), tuple(t["rotation_xyzw"]))
    k = (cal.fx_px, cal.fy_px, cal.cx_px, cal.cy_px)
    proj = data["target_projection"]
    z = proj["depth_m"]
    # Reconstruct only the saved map prior, never measured head corners/pose.
    mapped = transform_point(((proj["u_px"]-k[2])*z/k[0], (proj["v_px"]-k[3])*z/k[1], z),
                             tf("map", "camera"))
    s = data["sensors"]["scan"]
    scan = PlainLaserScan(tuple(math.inf if r is None else float(r) for r in s["ranges"]),
        s["angle_min"], s["angle_increment"], s["range_min"], s["range_max"], "base_scan",
        data["scan_stamp_sec"], data["scan_received_ros_sec"], s["angle_max"], "full_rotation")
    a = data["preliminary_association"]
    assert not a["associated"]  # Drift outside the original three-degree cone.
    common = dict(map_bearing_rad=a["map_bearing_rad"], accepted_range_m=tuple(a["accepted_range_m"]),
                  now_sec=data["now_sec"], max_scan_age_sec=.5)
    association = associate_candidate_lidar_target(scan,
        cone_half_angle_rad=a["cone_half_angle_rad"]+math.radians(12), **common)
    region, info = project(profile, candidate_xy=mapped[:2], camera_from_map=tf("camera", "map"),
        intrinsics=k, image_shape=frame.shape, association=association,
        image_stamp_sec=data["image_stamp_sec"], now_sec=data["now_sec"])
    assert region is not None, info
    filtering = CurrentScanHeadProposalFilter(intrinsics=CameraIntrinsics(c["width"], c["height"], *k),
        scan_from_camera=tf("base_scan", "camera"), scan=scan,
        cone_half_angle_rad=a["cone_half_angle_rad"], min_cluster_sample_count=1,
        max_camera_map_bearing_delta_rad=math.radians(12), **common)
    search = CandidateHeadSearch((proj["u_px"], proj["v_px"]),
        data["profile"]["expected_head_size_px"], edge_region=region)
    if metric_size:
        from scripts.aufgabe04.perception.stand_axis.metric_head_search import metric_head_search
        from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector
        search = metric_head_search(model_profile=profile, depth_m=z,
            fx=k[0], fy=k[1], cx=k[2], cy=k[3], image_shape=frame.shape, center=search.center,
            depth_uncertainty_m=.08, edge_region=region,
            camera_vertical=rotate_vector((0., 0., 1.), tf("camera", "map").rotation_xyzw))
    estimate, debug = estimate_current_head_geometry(cv2, frame, model_profile=profile,
        camera_fx_px=k[0], camera_fy_px=k[1], camera_cx_px=k[2], camera_cy_px=k[3],
        candidate_search=search, proposal_filter=filtering)
    assert estimate.usable, estimate.reason
    # Broad image-location check: no fitted yaw/corners supplied to the solver.
    assert 250 < sum(p.u_px for p in estimate.corners)/4 < 290
    counts = debug.head_acquisition_diagnostics["acquisition"]["joint_border_diagnostics"]["lidar_edge_region"]
    assert counts["retained_edge_pixels"] < .4 * counts["input_edge_pixels"]
    assert counts["rejected_line_segments"] > 0
