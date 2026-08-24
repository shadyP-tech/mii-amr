"""Convert synchronized stand evidence into a durable arrival-pose record.

The synchronized recommendation remains the sensor/perception contract.  This
adapter extracts its committed perpendicular pose into the semantic catalog
contract without importing route execution or ROS.
"""

from __future__ import annotations

import math

from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    SynchronizedViewpointRecommendation,
    angular_distance,
    validate_recommendation,
)
from scripts.aufgabe04.stations.arrival_pose_geometry import canonical_axial_angle
from scripts.aufgabe04.stations.arrival_pose_models import (
    ArrivalPoseRecord,
    ArrivalPoseValidation,
    AxisEstimate,
    CatalogPose2D,
    FaceSelection,
    StandEstimate,
)


COMMITTED_AXIS_STATES = frozenset({"target_committed", "resolved"})


def arrival_pose_record_from_recommendation(
    recommendation: SynchronizedViewpointRecommendation,
    *,
    candidate_uid: str,
    map_yaml_sha256: str,
    corridor_length_m: float,
    validated_unix_sec: float,
    axis_sample_count: int = 1,
    estimator: str = "simulation/silhouette_head_rectangle",
    source: str = "simulation/synchronized_viewpoint",
    source_observation_ids: tuple[str, ...] | None = None,
) -> ArrivalPoseRecord:
    """Build one explicit record from a committed synchronized estimate."""

    validate_recommendation(recommendation)
    if recommendation.axis_state not in COMMITTED_AXIS_STATES:
        raise ValueError(
            "arrival pose can only be recorded from a committed stand axis"
        )
    if not candidate_uid.strip():
        raise ValueError("candidate_uid must be non-empty")
    if not math.isfinite(corridor_length_m) or corridor_length_m <= 0.0:
        raise ValueError("corridor_length_m must be finite and positive")
    if type(axis_sample_count) is not int or axis_sample_count < 1:
        raise ValueError("axis_sample_count must be a positive integer")

    selected = next(
        (
            face
            for face in recommendation.face_candidates
            if face.face_id == recommendation.material_target.face_id
        ),
        None,
    )
    if selected is None:
        raise ValueError("material target does not reference a face candidate")
    if angular_distance(
        selected.outward_normal_rad,
        math.atan2(
            selected.pose.y_m - recommendation.stand.center.y_m,
            selected.pose.x_m - recommendation.stand.center.x_m,
        ),
    ) > 1.0e-6:
        raise ValueError("selected target is not on the selected face-normal ray")

    standoff_m = math.hypot(
        selected.pose.x_m - recommendation.stand.center.x_m,
        selected.pose.y_m - recommendation.stand.center.y_m,
    )
    normal = selected.outward_normal_rad
    entry_distance = standoff_m + corridor_length_m
    entry = CatalogPose2D(
        recommendation.stand.center.x_m + entry_distance * math.cos(normal),
        recommendation.stand.center.y_m + entry_distance * math.sin(normal),
        selected.pose.yaw_rad,
    )

    evidence = recommendation.side_evidence
    selected_has_side_evidence = evidence.valid and (
        evidence.face_id is None or evidence.face_id == selected.face_id
    )
    evidence_kind = evidence.kind if selected_has_side_evidence else "robot_facing_axis"
    evidence_confidence = (
        evidence.confidence
        if selected_has_side_evidence
        else recommendation.axis_confidence
    )
    evidence_hard = evidence.hard if selected_has_side_evidence else False
    evidence_provenance = (
        evidence.provenance if selected_has_side_evidence else recommendation.source
    )

    observation_ids = source_observation_ids
    if observation_ids is None:
        observation_ids = (
            f"{recommendation.stream_id}:{recommendation.stand_id}:"
            f"{recommendation.sensor_stamp_sec:.9f}",
        )
    axis = canonical_axial_angle(
        recommendation.face_candidates[0].outward_normal_rad - math.pi / 2.0
    )
    return ArrivalPoseRecord(
        candidate_uid=candidate_uid,
        stand_id=recommendation.stand_id,
        stand=StandEstimate(
            x_m=recommendation.stand.center.x_m,
            y_m=recommendation.stand.center.y_m,
            radius_m=recommendation.stand.radius_m,
            uncertainty_m=recommendation.stand.uncertainty_m,
        ),
        axis=AxisEstimate(
            axis_rad=axis,
            confidence=recommendation.axis_confidence,
            sample_count=axis_sample_count,
            estimator=estimator,
            observation_unix_sec=recommendation.observation_unix_sec,
        ),
        face=FaceSelection(
            face_id=selected.face_id,
            outward_normal_rad=normal,
            identity_resolved=selected.identity_resolved,
            evidence_kind=evidence_kind,
            evidence_confidence=evidence_confidence,
            evidence_hard=evidence_hard,
            evidence_valid=True,
            evidence_provenance=evidence_provenance,
        ),
        arrival_pose=CatalogPose2D(
            selected.pose.x_m,
            selected.pose.y_m,
            selected.pose.yaw_rad,
        ),
        corridor_entry_pose=entry,
        standoff_m=standoff_m,
        corridor_length_m=corridor_length_m,
        validation=ArrivalPoseValidation(
            target_in_bounds=True,
            target_collision_free=True,
            corridor_collision_free=True,
            validated_map_yaml_sha256=map_yaml_sha256,
            validated_unix_sec=validated_unix_sec,
        ),
        source_observation_ids=tuple(sorted(observation_ids)),
        sensor_stamp_sec=recommendation.sensor_stamp_sec,
        source=source,
    )
