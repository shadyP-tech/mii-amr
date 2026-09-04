"""Build receipt evidence from accepted camera/LiDAR registration objects.

This ROS-free adapter keeps the observer node from assembling a motion-facing
receipt out of unrelated scalar values.  It accepts only an associated legacy
LiDAR result, and for the exception path also requires the accepted bounded
image-registration decision and its exact camera-centred LiDAR wrapper.
"""

from __future__ import annotations

import math

from scripts.aufgabe04.perception.candidate_lidar_association import (
    CameraRegisteredCandidateLidarAssociation,
    CandidateLidarAssociation,
)
from scripts.aufgabe04.real_robot.observer.contract import (
    MAXIMUM_HEAD_CENTER_ERROR_RATIO,
    MAXIMUM_REGISTRATION_BEARING_DELTA_RAD,
    MAXIMUM_REGISTRATION_HEAD_CENTER_OFFSET_RATIO,
    TARGET_REGISTRATION_LIDAR_SOURCE_CAMERA,
    TARGET_REGISTRATION_LIDAR_SOURCE_MAP,
    TARGET_REGISTRATION_MODE_BOUNDED_CAMERA_LIDAR,
    TARGET_REGISTRATION_MODE_MAP_PROJECTION,
)
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    HeadRoiRegistrationDecision,
    REGISTERED_BACKSIDE_REACQUISITION_SOURCE,
)


def build_backside_target_registration_evidence(
    *,
    final_head_center_error_ratio: float,
    candidate_lidar_association: CandidateLidarAssociation,
    registration_decision: HeadRoiRegistrationDecision | None = None,
    registered_lidar_association: (
        CameraRegisteredCandidateLidarAssociation | None
    ) = None,
) -> dict[str, object]:
    """Return the exact schema-v3 registration block or fail closed."""

    final_error = _finite_nonnegative(
        final_head_center_error_ratio,
        "final_head_center_error_ratio",
    )
    if final_error > MAXIMUM_HEAD_CENTER_ERROR_RATIO:
        raise ValueError("final backside measurement is not strictly centred")
    if not isinstance(candidate_lidar_association, CandidateLidarAssociation):
        raise TypeError("candidate_lidar_association has an unexpected type")
    if not candidate_lidar_association.associated:
        raise ValueError("candidate LiDAR association is not accepted")
    if candidate_lidar_association.eligible_cluster_count < 1:
        raise ValueError("candidate LiDAR association has no eligible cluster")

    if registered_lidar_association is None:
        if registration_decision is not None:
            raise ValueError(
                "nominal target evidence cannot carry a registration decision"
            )
        camera_delta = (
            candidate_lidar_association.observed_camera_bearing_delta_from_map_rad
        )
        if camera_delta is None:
            raise ValueError("nominal LiDAR evidence has no camera/map bearing delta")
        camera_delta = _finite_nonnegative(
            camera_delta,
            "camera_map_bearing_delta_rad",
        )
        bearing_limit = _finite_nonnegative(
            candidate_lidar_association.cone_half_angle_rad,
            "bearing_delta_limit_rad",
        )
        if camera_delta > bearing_limit:
            raise ValueError("nominal camera bearing is outside the map cone")
        return {
            "mode": TARGET_REGISTRATION_MODE_MAP_PROJECTION,
            "original_head_center_error_ratio": final_error,
            "center_offset_limit_ratio": MAXIMUM_HEAD_CENTER_ERROR_RATIO,
            "final_strict_head_center_error_ratio": final_error,
            "map_bearing_rad": candidate_lidar_association.map_bearing_rad,
            "lidar_search_bearing_rad": candidate_lidar_association.map_bearing_rad,
            "camera_map_bearing_delta_rad": camera_delta,
            "bearing_delta_limit_rad": bearing_limit,
            "lidar_search_bearing_source": TARGET_REGISTRATION_LIDAR_SOURCE_MAP,
            "unique_eligible_lidar_cluster_required": False,
            "eligible_lidar_cluster_count": (
                candidate_lidar_association.eligible_cluster_count
            ),
        }

    if not isinstance(
        registered_lidar_association,
        CameraRegisteredCandidateLidarAssociation,
    ):
        raise TypeError("registered_lidar_association has an unexpected type")
    if registration_decision is None or not registration_decision.accepted:
        raise ValueError("registered target evidence has no accepted image decision")
    if registration_decision.attempt is None or (
        registration_decision.attempt.source
        != REGISTERED_BACKSIDE_REACQUISITION_SOURCE
    ):
        raise ValueError("registered target evidence has no strict retry attempt")
    if registration_decision.center_offset_ratio is None:
        raise ValueError("registered target evidence has no original centre offset")
    original_error = _finite_nonnegative(
        registration_decision.center_offset_ratio,
        "original_head_center_error_ratio",
    )
    center_limit = _finite_nonnegative(
        registration_decision.max_center_offset_ratio,
        "center_offset_limit_ratio",
    )
    if (
        center_limit > MAXIMUM_REGISTRATION_HEAD_CENTER_OFFSET_RATIO
        or original_error > center_limit
    ):
        raise ValueError("registered image displacement exceeds its hard bound")
    if not registered_lidar_association.associated:
        raise ValueError("registered LiDAR association is not accepted")
    if not registered_lidar_association.unique_eligible_cluster_required:
        raise ValueError("registered LiDAR association did not require uniqueness")
    if registered_lidar_association.search_bearing_source != (
        TARGET_REGISTRATION_LIDAR_SOURCE_CAMERA
    ):
        raise ValueError("registered LiDAR association used the wrong bearing source")
    if registered_lidar_association.search_association != (
        candidate_lidar_association
    ):
        raise ValueError("registered wrapper is not bound to the accepted scan result")
    if candidate_lidar_association.eligible_cluster_count != 1:
        raise ValueError("registered LiDAR association is not unique")

    camera_delta = _finite_nonnegative(
        registered_lidar_association.camera_map_bearing_delta_rad,
        "camera_map_bearing_delta_rad",
    )
    bearing_limit = _finite_nonnegative(
        registered_lidar_association.max_camera_map_bearing_delta_rad,
        "bearing_delta_limit_rad",
    )
    if (
        bearing_limit > MAXIMUM_REGISTRATION_BEARING_DELTA_RAD
        or camera_delta > bearing_limit
    ):
        raise ValueError("registered camera/map bearing exceeds its hard bound")
    return {
        "mode": TARGET_REGISTRATION_MODE_BOUNDED_CAMERA_LIDAR,
        "original_head_center_error_ratio": original_error,
        "center_offset_limit_ratio": center_limit,
        "final_strict_head_center_error_ratio": final_error,
        "map_bearing_rad": registered_lidar_association.map_bearing_rad,
        "lidar_search_bearing_rad": (
            registered_lidar_association.registered_search_bearing_rad
        ),
        "camera_map_bearing_delta_rad": camera_delta,
        "bearing_delta_limit_rad": bearing_limit,
        "lidar_search_bearing_source": TARGET_REGISTRATION_LIDAR_SOURCE_CAMERA,
        "unique_eligible_lidar_cluster_required": True,
        "eligible_lidar_cluster_count": (
            candidate_lidar_association.eligible_cluster_count
        ),
    }


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


__all__ = ["build_backside_target_registration_evidence"]
