"""Recorded partial-envelope recovery centers an angle-ambiguous current head.

The image and scan are the same stopped frame, with no supplied corners, pose,
QR result, or model yaw. This checks one frame and a motion-neutral centering
advisory, not seven-frame backside consensus or physical route execution.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.camera_calibration import (
    CameraCalibration, rectify_bgr_frame, rectified_source_support,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis.image_source_support import ImageSourceSupport
from scripts.aufgabe04.perception.stand_axis.lidar_head_edge_region import project_lidar_candidate_head_region
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector
from scripts.aufgabe04.qr_scanning.native_qr_observations import detect_native_qr_observations_bgr
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, OpticalProjection
from scripts.aufgabe04.real_robot.observer.candidate_centering import (
    build_camera_centering_advisory, validate_camera_centering_advisory,
)
from scripts.aufgabe04.real_robot.observer.candidate_centering_receipt import prepare_candidate_centering
from scripts.aufgabe04.real_robot.observer.candidate_position_epoch import recovered_search, RECOVERY_STEP_RAD
from scripts.aufgabe04.real_robot.observer.current_head_association import associate_current_measured_head
from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose
from scripts.aufgabe04.real_robot.observer.inspection_framing import review_centering_destination
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer.qr_acquisition_policy import QrAcquisitionPolicy
from scripts.aufgabe04.real_robot.observer.qr_decode_cache import RoiQrDecodeCache
from scripts.aufgabe04.real_robot.observer.target_reconciliation import StoppedTargetReconciliation
from scripts.aufgabe04.real_robot.observer.tracked_head_registration import (
    tracked_head_selection, register_current_tracked_head, review_current_tracked_head_crop,
)
from scripts.aufgabe04.real_robot.observer.viewer_head_acquisition import evaluate_viewer_head, classify_viewer_head
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot
from tests.aufgabe04.test_candidate_position_epoch_clipped import transform, pose
from tests.aufgabe04 import test_camera_observer_processing as processing
from scripts.aufgabe04.real_robot.observer.qr_target_binding import bind_qr_observations_to_target
from scripts.aufgabe04.real_robot.observer.current_head_qr_binding import bind_qr_to_current_head
from scripts.aufgabe04.real_robot.observer.target_reconciliation import validate_reconciliation

ROOT = Path(__file__).parent / "fixtures/candidate_position_epoch_partial_20260930"
UID = "survey_candidate_0005"


def metadata():
    return json.loads((ROOT / "observations.json").read_text())


def recorded_rows(indices=(29, 30, 31)):
    """Rebuild three original stopped observations from the self-contained fixture."""
    geometry = load_candidate_snapshot(ROOT / "candidate_snapshot.json").candidate_for(UID).geometry
    for row in metadata()["rows"]:
        if row["capture_index"] not in indices:
            continue
        raw = row["sensors"]["scan"]
        scan = PlainLaserScan(
            ranges=tuple(math.nan if value is None else float(value) for value in raw["ranges"]),
            **{key: raw[key] for key in ("angle_min", "angle_max", "angle_increment", "range_min", "range_max")},
            scan_frame_id=raw["header"]["frame_id"], scan_stamp_sec=row["scan_stamp_sec"],
            receipt_sec=row["scan_received_ros_sec"], scan_topology_profile="full_rotation")
        options = dict(row["original_options"])
        options["accepted_range_m"] = tuple(options["accepted_range_m"])
        yield dict(snapshot_path=ROOT / "candidate_snapshot.json", candidate_uid=UID,
            planning_frame="map", stand_center=(geometry.x_m, geometry.y_m),
            target_key="recorded-partial-target", epoch=0, scan=scan,
            scan_from_map=transform(row, "base_scan", "map"), robot_pose=pose(row, "map"),
            image_stamp_sec=row["image_stamp_sec"], now_sec=row["now_sec"], options=options,
            position_epoch_path=ROOT / "candidate_frame_projection.json")


def test_recorded_two_three_two_beam_sequence_reconciles_at_third_frame():
    rows = list(recorded_rows((1, 2, 3)))
    ordinary = [associate_candidate_lidar_target(row["scan"],
        map_bearing_rad=row["options"]["map_bearing_rad"], cone_half_angle_rad=math.radians(15),
        accepted_range_m=row["options"]["accepted_range_m"],
        now_sec=row["now_sec"], max_scan_age_sec=.5) for row in rows]
    assert [a.selected_cluster_sample_count for a in ordinary] == [2, 3, 2]
    tracker = StoppedTargetReconciliation()
    assert tracker.observe(**rows[0]) is None
    assert tracker.observe(**rows[1]) is None
    assert tracker.metadata["sample_count"] == 2
    proof = tracker.observe(**rows[2])
    assert proof is not None, tracker.metadata
    _, envelope, _, _ = validate_reconciliation(proof)
    assert envelope.selected_cluster_source_indices == (6, 7, 8, 9)
    assert len(proof["entries"]) == 3
    assert proof["candidate_geometry_updated"] is False
    assert proof["motion_authorized"] is False


def test_recorded_partial_cluster_centers_without_admitting_ambiguous_angle(tmp_path, monkeypatch):
    preserved = {name: (ROOT / name).read_bytes() for name in (
        "candidate_snapshot.json", "candidate_frame_projection.json")}
    data = metadata()
    for name, original in preserved.items():
        assert hashlib.sha256(original).hexdigest() == data["source_sha256"][name]
    rows = list(recorded_rows())
    fixture = processing.CameraObserverProcessingTest()
    adapter = fixture.make_adapter()
    adapter.args.stand_id = UID
    adapter.args.stream_id = "recorded_partial_cluster"
    adapter.args.stand_x, adapter.args.stand_y = rows[-1]["stand_center"]
    adapter.args.candidate_centering_json = tmp_path / "centering.json"
    adapter.args.recommended_pose_json = tmp_path / "recommendation.json"
    adapter.args.status_json = tmp_path / "status.json"
    for item in rows:
        item["target_key"] = adapter._target_evidence_key()
    tracker = StoppedTargetReconciliation()
    proofs = [tracker.observe(**row) for row in rows]
    proof, row = proofs[-1], rows[-1]
    assert proof is not None, tracker.metadata
    current = data["rows"][-1]
    assert current["capture_index"] == 31
    c = current["sensors"]["camera_info"]
    calibration = CameraCalibration(c["width"], c["height"], "camera", tuple(c["k"]),
        tuple(c["d"]), tuple(c["r"]), tuple(c["p"]))
    intrinsics = CameraIntrinsics(c["width"], c["height"], calibration.fx_px,
        calibration.fy_px, calibration.cx_px, calibration.cy_px)
    tf = lambda parent, child: transform(current, parent, child)
    model = load_measured_physical_stand_model(Path(__file__).resolve().parents[2] /
        "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
    projection = OpticalProjection(**current["projection"])
    hint, envelope = recovered_search(proof, scan=row["scan"], original_projection=projection,
        camera_from_map=tf("camera", "map"), intrinsics=intrinsics, model_profile=model)
    assert 385 < projection.u_px < 395
    assert 185 < hint.projection.u_px < 200
    image_bytes = (ROOT / "frame_000031.jpg").read_bytes()
    assert hashlib.sha256(image_bytes).hexdigest() == data["image_sha256"]
    frame = rectify_bgr_frame(cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR),
        calibration, cv2, np)
    region, region_info = project_lidar_candidate_head_region(candidate_xy=hint.search_xy,
        camera_from_map=tf("camera", "map"), model_profile=model,
        intrinsics=(intrinsics.fx_px, intrinsics.fy_px, intrinsics.cx_px, intrinsics.cy_px),
        image_shape=frame.shape, position_uncertainty_m=.02, surface_center_margin_m=.06,
        association=envelope, image_stamp_sec=row["image_stamp_sec"], now_sec=row["now_sec"],
        max_sensor_age_sec=.5, sync_tolerance_sec=.1)
    assert region is not None, region_info
    # Wall-clock throughput is independent of this geometric regression. Keep
    # the actual historical image/scan timestamps and all source-age guards.
    budget = QrAcquisitionPolicy().begin_frame(target_key=row["target_key"],
        image_stamp_sec=row["image_stamp_sec"], started_ros_sec=row["now_sec"],
        started_monotonic_sec=10., max_sensor_age_sec=.5)
    evaluation = evaluate_viewer_head(cv2, frame, model_profile=model, intrinsics=intrinsics,
        pose_hint=None, projection=hint.projection, expected_head_height_px=projection.expected_size_px,
        fallback_attempt=None, cache=RoiQrDecodeCache(), budget=budget,
        native_decoder=lambda crop: detect_native_qr_observations_bgr(crop, cv2),
        full_decoder=lambda crop, limit, provenance: detect_qr_observations_bgr(crop, cv2,
            diagnostics=provenance, max_elapsed_sec=limit, prefer_native_geometry=True),
        deadline_monotonic_sec=None, now=lambda: 10., lidar_edge_region=region,
        lidar_edge_region_diagnostics=region_info, depth_uncertainty_m=.08, position_uncertainty_m=.08,
        camera_vertical=rotate_vector((0., 0., 1.), tf("camera", "map").rotation_xyzw),
        source_support=ImageSourceSupport(cv2, rectified_source_support(calibration, cv2, np)),
        nearest_context=dict(scan=row["scan"], image_stamp_sec=row["image_stamp_sec"],
            now_sec=row["now_sec"], max_scan_age_sec=.5,
            scan_from_camera=tf("base_scan", "camera"), base_from_camera=tf("base_footprint", "camera"),
            accepted_range_m=row["options"]["accepted_range_m"], sync_tolerance_sec=.1))
    evaluation = classify_viewer_head(evaluation, model_profile=model, intrinsics=intrinsics,
        expected_head_height_px=projection.expected_size_px)
    assert not evaluation.estimate.usable
    assert evaluation.estimate.reason == "head_model_planar_axis_ambiguous"
    assert evaluation.estimate.yaw_deg is None
    assert evaluation.debug.head_orientation_bounds.accepted
    assert len(evaluation.debug.head_orientation_bounds.hypotheses) >= 2
    assert evaluation.estimate.corners is not None
    association_options = dict(estimate=evaluation.estimate, debug=evaluation.debug,
        attempt=evaluation.attempt, projection=projection, expected_head_height_px=projection.expected_size_px,
        profile_sha256=model.sha256, intrinsics=intrinsics, scan_from_camera=tf("base_scan", "camera"),
        scan=row["scan"], now_sec=row["now_sec"], max_scan_age_sec=.5, min_cluster_sample_count=1,
        max_center_offset_ratio=1.5, **row["options"])
    # Pixels alone do not bypass candidate identity association.
    original = associate_current_measured_head(**association_options)
    assert not original.accepted
    association = associate_current_measured_head(**association_options,
        search_reconciliation=hint, target_reconciliation=proof)
    assert association.accepted, association.reason
    assert 195 < association.full_image_center_px[0] < 210
    assert not association.head_admission.accepted
    assert association.head_orientation_bounds is not None
    selection = register_current_tracked_head(tracked_head_selection(evaluation), association=association,
        observed_at_sec=row["image_stamp_sec"], now_sec=row["now_sec"], max_age_sec=.5,
        expected_model_sha256=model.sha256)
    review = review_current_tracked_head_crop(selection)
    assert review.accepted, review.reason
    # Keep the actual decoded symbol and its independent current-scan binding.
    # Centering does not require marker absence or a usable planar angle.
    qr_binding = bind_qr_observations_to_target(evaluation.qr_observations,
        roi=evaluation.attempt.roi, intrinsics=intrinsics,
        scan_from_camera=tf("base_scan", "camera"), scan=row["scan"],
        now_sec=row["now_sec"], max_scan_age_sec=.5, min_cluster_sample_count=1,
        camera_registration_accepted=association.accepted,
        target_reconciliation=proof, **row["options"])
    qr_binding = bind_qr_to_current_head(qr_binding, evaluation.qr_observations,
        head_corners=evaluation.estimate.corners, head_association=association)
    assert qr_binding.symbol_count <= 1
    if evaluation.qr_observations:
        assert qr_binding.accepted, qr_binding.reason

    advice = build_camera_centering_advisory(association=association, intrinsics=intrinsics,
        scan_from_camera=tf("base_scan", "camera"), base_from_camera=tf("base_footprint", "camera"),
        candidate_uid=UID, target_key=row["target_key"], stream_id="recorded_partial_cluster",
        planning_frame="map", motion_epoch=0, anchor_pose=EvidencePose(*row["robot_pose"]),
        anchor_odom_pose=EvidencePose(*pose(current, "odom")), odom_stamp_sec=row["image_stamp_sec"],
        image_stamp_sec=row["image_stamp_sec"], now_sec=row["now_sec"],
        robot_profile_sha256=current["robot_profile_sha256"],
        calibration_profile_sha256=current["calibration_profile_sha256"],
        stand_model_profile_sha256=current["stand_model_profile_sha256"])
    assert advice is not None
    assert advice.arrival_recovery
    assert math.radians(14) < advice.required_yaw_rad < math.radians(17)
    assert 0 < advice.requested_yaw_rad < advice.required_yaw_rad <= RECOVERY_STEP_RAD
    search = association.lidar_association.search_association
    assert review_centering_destination(advice, search_association=search).allowed
    # The coarse step must stop before centering would cross the scan seam.
    assert not review_centering_destination(replace(advice, requested_yaw_rad=advice.required_yaw_rad),
        search_association=search).allowed
    serialized = json.loads(json.dumps(advice.metadata()))
    assert validate_camera_centering_advisory(serialized).arrival_recovery
    assert serialized["motion_authorized"] is False
    assert serialized["completion_authorized"] is False
    assert proof["candidate_geometry_updated"] is False
    assert proof["motion_authorized"] is False

    # Carry the same recovered pixels and raw-scan identity proof through the
    # real stopped-frame admission and status handoff. No axis/centering policy
    # is mocked; only profile hashes and the ROS clock come from the recording.
    fixture.clock_sec = row["now_sec"]
    adapter.stand_model_profile = model
    adapter.profile.base_frame = "base_footprint"
    adapter.profile.scan_frame = "base_scan"
    adapter.profile.odom_frame = "odom"
    adapter.last_pose = robot_pose = Pose2D(*row["robot_pose"])
    receipt_module = "scripts.aufgabe04.real_robot.observer.candidate_centering_receipt."
    monkeypatch.setattr(receipt_module + "real_robot_profile_sha256",
        lambda _: current["robot_profile_sha256"])
    monkeypatch.setattr(receipt_module + "camera_calibration_sha256",
        lambda _: current["calibration_profile_sha256"])
    frame_metadata = {}
    adapter._pending_candidate_centering = prepare_candidate_centering(
        crop=review, association=association, image_stamp_sec=row["image_stamp_sec"],
        scan_stamp_sec=row["scan"].scan_stamp_sec, target_key=row["target_key"],
        robot_pose=robot_pose, odom_pose=Pose2D(*pose(current, "odom")), intrinsics=intrinsics,
        scan_from_camera=tf("base_scan", "camera"),
        base_from_camera=tf("base_footprint", "camera"), metadata=frame_metadata)
    assert adapter._pending_candidate_centering is not None
    update = adapter._record_observation_frame(robot_pose=robot_pose,
        image_stamp_sec=row["image_stamp_sec"], scan_stamp_sec=row["scan"].scan_stamp_sec,
        observed_at_sec=row["now_sec"], lidar_associated=association.accepted,
        axis_yaw_rad=None,
        axis_source=None, qr_texts=qr_binding.qr_texts_for_evidence,
        qr_symbol_count=qr_binding.symbol_count)
    assert update.frame_accepted
    assert update.reason == "associated_frame_recorded"
    assert not update.axis_sample_accepted
    assert update.snapshot.current_axis_sample_count == 0
    assert update.axis_consensus is None
    assert adapter._candidate_centering_ready is not None
    PassiveRealViewpointNode._write_status(adapter, "metric_model_measurement_unavailable")
    committed = json.loads(adapter.args.candidate_centering_json.read_text())
    assert validate_camera_centering_advisory(committed).arrival_recovery
    assert committed["requested_yaw_rad"] == pytest.approx(advice.requested_yaw_rad)
    assert committed["motion_authorized"] is False
    assert committed["completion_authorized"] is False
    assert adapter.completed
    assert not adapter.args.recommended_pose_json.exists()
    status = json.loads(adapter.args.status_json.read_text())
    assert status["state"] == "candidate_centering_committed"
    assert status["axis_consensus"]["sample_count"] == 0
    assert status["camera_centering"]["reason"] == "fresh_current_head_off_center"
    assert all((ROOT / name).read_bytes() == original for name, original in preserved.items())
