"""Build a real-robot recommendation from sealed passive sensor evidence."""

from __future__ import annotations

import math
import time

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    FaceCandidate,
    MaterialTarget,
    SideEvidence,
    StandGeometry,
    SynchronizedViewpointRecommendation,
    REAL_VIEWPOINT_SOURCE,
    angular_distance,
    normalize_angle,
    validate_recommendation,
)


def build_real_viewpoint_recommendation(
    *,
    stream_id: str,
    stand_id: str,
    planning_frame: str,
    stand_center: Pose2D,
    stand_radius_m: float,
    stand_uncertainty_m: float,
    robot_pose: Pose2D,
    stand_axis_rad: float,
    axis_confidence: float,
    axis_sample_count: int,
    sensor_stamp_sec: float,
    expected_qr_id: str,
    observed_qr_ids: tuple[str, ...],
    target_distance_m: float,
    observation_unix_sec: float | None = None,
) -> SynchronizedViewpointRecommendation:
    """Create one committed, robot-facing real arrival recommendation.

    A hard QR match binds the currently visible physical face.  Without that
    match the 180-degree planar head-axis ambiguity is intentionally not
    committed.
    """

    if expected_qr_id not in observed_qr_ids:
        raise ValueError("expected QR ID is absent from the passive observation")
    if len(set(observed_qr_ids)) != 1:
        raise ValueError("passive QR consensus must contain exactly one identity")
    if not math.isfinite(stand_axis_rad):
        raise ValueError("stand_axis_rad must be finite")
    if not 0.0 <= axis_confidence <= 1.0:
        raise ValueError("axis_confidence must be in [0, 1]")
    if type(axis_sample_count) is not int or axis_sample_count < 2:
        raise ValueError("axis_sample_count must be at least two")
    if not math.isfinite(target_distance_m) or target_distance_m <= 0.0:
        raise ValueError("target_distance_m must be finite and positive")

    outward_normals = (
        normalize_angle(stand_axis_rad + math.pi / 2.0),
        normalize_angle(stand_axis_rad - math.pi / 2.0),
    )
    robot_normal = math.atan2(
        robot_pose.y_m - stand_center.y_m,
        robot_pose.x_m - stand_center.x_m,
    )
    selected_index = min(
        range(2),
        key=lambda index: angular_distance(outward_normals[index], robot_normal),
    )
    faces = []
    for index, normal in enumerate(outward_normals):
        face_id = "qr_face" if index == selected_index else "opposite_face"
        pose = Pose2D(
            stand_center.x_m + target_distance_m * math.cos(normal),
            stand_center.y_m + target_distance_m * math.sin(normal),
            normalize_angle(normal + math.pi),
        )
        faces.append(FaceCandidate(face_id, normal, pose, True))
    selected = faces[selected_index]
    recommendation = SynchronizedViewpointRecommendation(
        schema_version=1,
        simulation_only=False,
        stream_id=stream_id,
        stand_id=stand_id,
        planning_frame=planning_frame,
        source=REAL_VIEWPOINT_SOURCE,
        observation_unix_sec=(
            time.time() if observation_unix_sec is None else observation_unix_sec
        ),
        sensor_stamp_sec=sensor_stamp_sec,
        stand=StandGeometry(
            center=stand_center,
            radius_m=stand_radius_m,
            uncertainty_m=stand_uncertainty_m,
            provenance="real/lidar_candidate_snapshot",
        ),
        robot_pose=robot_pose,
        axis_confidence=axis_confidence,
        axis_state="target_committed",
        face_candidates=(faces[0], faces[1]),
        side_evidence=SideEvidence(
            kind="qr_consensus",
            confidence=1.0,
            hard=True,
            valid=True,
            face_id=selected.face_id,
            provenance="real/onboard_camera_qr_consensus",
        ),
        material_target=MaterialTarget(
            face_id=selected.face_id,
            pose=selected.pose,
            evidence_state="hard_qr",
        ),
        axis_sample_count=axis_sample_count,
    )
    validate_recommendation(
        recommendation,
        required_simulation_only=False,
        required_source=REAL_VIEWPOINT_SOURCE,
    )
    return recommendation
