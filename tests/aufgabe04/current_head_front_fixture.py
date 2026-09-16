"""Synthetic validated current-head receipt for contract tests, not hardware proof."""

from dataclasses import asdict
import math

from scripts.aufgabe04.artifacts.current_head_front_observation import (
    CURRENT_HEAD_FRONT_GATES, build_current_head_front_evidence,
)
from scripts.aufgabe04.perception.stand_axis.head_model_admission import HeadModelAdmission
from tests.aufgabe04.test_head_model_admission import quality


def current_head_front_evidence(*, stamp=100., qr_id="QR_003", target_key="stream:stand:0.0:0.0",
                                stand_axis_rad=math.pi/2, **changes):
    model = quality()
    cluster = {"associated": True, "eligible_cluster_count": 1,
               "scan_stamp_sec": stamp, "scan_frame_id": "base_scan",
               "selected_cluster_source_indices": [3, 4, 5]}
    qr_binding = {"accepted": True, "reason": "decoded_qr_target_associated", "symbol_count": 1,
                  "qr_texts_for_evidence": [qr_id], "association": dict(cluster),
                  "current_head_binding": {"accepted": True,
                    "reason": "qr_inside_current_head_same_lidar_cluster",
                    "shared_scan_source_indices": [3, 4, 5]}}
    fields = dict(
        head_model_quality=asdict(model),
        head_admission=HeadModelAdmission(True, "measured_head_geometry_quality_accepted", model.yaw_std_deg).metadata(),
        head_corners_px=((100., 100.), (200., 100.), (200., 200.), (100., 200.)),
        qr_corners_px=((125., 125.), (175., 125.), (175., 175.), (125., 175.)),
        image_shape=(600, 800), model_profile_sha256=model.profile_sha256,
        sensor_stamp_sec=stamp, scan_stamp_sec=stamp, checked_at_sec=stamp+.1,
        qr_sensor_stamp_sec=stamp, qr_scan_stamp_sec=stamp, qr_checked_at_sec=stamp+.1,
        qr_id=qr_id, qr_binding=qr_binding, head_lidar_association=dict(cluster),
        camera_yaw_rad=.1, camera_heading_rad=stand_axis_rad + math.pi/2-.1,
        stand_axis_rad=stand_axis_rad, target_key=target_key, motion_epoch=0,
        camera_signature=(640., 640., 400., 300.),
        sample_gate_evidence={key: True for key in CURRENT_HEAD_FRONT_GATES},
    )
    return build_current_head_front_evidence(**{**fields, **changes})
