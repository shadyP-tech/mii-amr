"""Portable historical-image replay through the production ROI/fit policies."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.camera_calibration import rectify_bgr_frame
from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
from scripts.aufgabe04.perception.stand_axis.model_input_cache import MetricModelInputCache
from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform, rectified_pixel_bearing_in_scan
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr
from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
from scripts.aufgabe04.real_robot.observer.backside_proposal_reuse import BacksideProposalContext
from scripts.aufgabe04.real_robot.observer.camera_target_registration import HeadRoiEvaluation
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt
from scripts.aufgabe04.real_robot.observer.qr_decode_cache import RoiQrDecodeCache


REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path(__file__).with_name("fixtures") / "recorded_backside_20260910"


class RecordedBacksideFixture:
    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self.inputs = json.loads((FIXTURE_ROOT / "inputs.json").read_text())
        self.model = load_measured_physical_stand_model(
            REPOSITORY / self.inputs["stand_model_path"]
        )

    def evaluate(self, name, reuse, *, cache_inputs=True, test_stamp=None):
        """Decode and refit every invocation; never reuse a measurement.

        ``test_stamp`` is only for repeated-content deterministic unit tests.
        Historical source timestamps are retained in the fixture and no
        freshness/consensus/receipt or motion code is called here.
        """
        record = self.inputs["frames"][name]
        raw = (FIXTURE_ROOT / record["image"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["image_sha256"]
        assert self.model.sha256 == record["stand_model_profile_sha256"]
        info = record["camera_info"]
        camera = info["p"]
        started = perf_counter()
        frame = self.cv2.imdecode(self.np.frombuffer(raw, dtype=self.np.uint8), self.cv2.IMREAD_COLOR)
        frame = rectify_bgr_frame(frame, SimpleNamespace(**info), self.cv2, self.np)
        input_cache = MetricModelInputCache(frame) if cache_inputs else None
        qr_cache = RoiQrDecodeCache()
        attempts = tuple(
            HeadRoiAttempt(**{**item, "roi": ImageRoi(**item["roi"])})
            for item in record["attempts"]
        )
        calls = []

        def evaluate(attempt, pose_hint):
            assert pose_hint is None  # Backside has no directed metric-pose seed.
            roi = attempt.roi
            bounds = (roi.x0, roi.y0, roi.x1, roi.y1)
            crop = frame[roi.y0:roi.y1, roi.x0:roi.x1]
            decoded = qr_cache.decode(
                roi=bounds, mode="full", frame=crop,
                decoder=lambda image: detect_qr_observations_bgr(image, self.cv2),
            )
            profile = record["profile"]
            estimate, debug = estimate_stand_axis_from_metric_model(
                self.cv2, crop, model_profile=self.model,
                camera_fx_px=camera[0], camera_fy_px=camera[5],
                camera_cx_px=camera[2] - roi.x0, camera_cy_px=camera[6] - roi.y0,
                pose_hint=None, qr_observations=decoded.observations,
                edge_preprocess=profile["edge_preprocess"],
                canny_low=profile["canny_low"], canny_high=profile["canny_high"],
                min_edge_height_px=profile["min_edge_height_px"],
                expected_head_center_u_px=attempt.expected_center_u_px - roi.x0,
                expected_head_center_v_px=attempt.expected_center_v_px - roi.y0,
                expected_head_height_px=attempt.expected_head_height_px,
                backside_target_crop_horizontal_half_width_ratio=attempt.backside_target_crop_half_width_ratio,
                input_cache=input_cache,
                input_cache_roi=bounds if input_cache is not None else None,
            )
            calls.append({
                "source": attempt.source, "reason": estimate.reason,
                "qr_decode": decoded.metadata(), "stage_timings_ms": debug.stage_timings_ms,
                "input_cache": None if input_cache is None else deepcopy(input_cache.last_metadata),
            })
            return HeadRoiEvaluation(attempt, crop, estimate, debug,
                                     decoded.observations, decoded.metadata())

        transform = record["map_from_base"]
        x, y, _ = transform["translation_xyz_m"]
        qx, qy, qz, qw = transform["rotation_xyzw"]
        pose = Pose2D(x, y, math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        selection = reuse.select(
            attempts,
            context=BacksideProposalContext(
                self.inputs["candidate_uid"], self.model.sha256,
                (camera[0], camera[5], camera[2], camera[6]), tuple(frame.shape),
            ),
            observed_at_sec=(record["source_clocks"]["image_stamp_sec"]
                             if test_stamp is None else test_stamp),
            robot_pose=pose, marker_seen_in_stationary_epoch=False,
            tracked_pose=None, evaluate=evaluate, enable_reacquisition=True,
            max_center_offset_ratio=1.5,
        )
        return selection, {
            "elapsed_ms": (perf_counter() - started) * 1000.0,
            "calls": calls, "proposal_reuse": deepcopy(reuse.last_metadata),
        }

    def registered_binding(self, name, selection):
        """Use original scan time only to isolate geometric association."""
        record = self.inputs["frames"][name]
        camera = record["camera_info"]["p"]
        selected = selection.selected
        corners = selected.estimate.corners
        center_u = sum(point.u_px for point in corners) / 4 + selected.attempt.roi.x0
        center_v = sum(point.v_px for point in corners) / 4 + selected.attempt.roi.y0
        tf = record["scan_from_camera"]
        bearing = rectified_pixel_bearing_in_scan(
            u_px=center_u, v_px=center_v, fx_px=camera[0], fy_px=camera[5],
            cx_px=camera[2], cy_px=camera[6],
            scan_from_camera=RigidTransform(
                parent_frame="base_scan", child_frame="camera",
                translation_xyz_m=tuple(tf["translation_xyz_m"]),
                rotation_xyzw=tuple(tf["rotation_xyzw"]),
            ),
        )
        scan = record["scan"]
        stamp = record["source_clocks"]["scan_stamp_sec"]
        plain = PlainLaserScan(
            ranges=tuple(map(float, scan["ranges"])),
            **{key: scan[key] for key in ("angle_min", "angle_max", "angle_increment", "range_min", "range_max")},
            scan_stamp_sec=stamp, receipt_sec=stamp, scan_frame_id="base_scan",
            scan_topology_profile="full_rotation",
        )
        prior = record["preliminary_lidar_association"]
        return associate_camera_registered_candidate_lidar_target(
            plain, map_bearing_rad=prior["map_bearing_rad"],
            observed_camera_bearing_rad=bearing, cone_half_angle_rad=prior["cone_half_angle_rad"],
            accepted_range_m=tuple(prior["accepted_range_m"]), now_sec=stamp,
            max_scan_age_sec=0.5, min_cluster_sample_count=1,
        )
