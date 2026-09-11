"""Reconstruct a candidate route anchor from admitted, captured localization.

Planning and replacement-permit validation use this same ROS-free derivation.
The direct map/odom transform projects both the robot's observed odom pose and
the frozen candidates.  A chained map/base lookup remains diagnostic only.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Mapping

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
    odom_pose_to_map,
)


def _frame(value: object) -> str:
    return value.strip().strip("/") if isinstance(value, str) else ""


def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"candidate planning frame {name} is not finite")
    return float(value)


def _pose(data: Mapping[str, object], name: str) -> tuple[float, float, float]:
    return tuple(_finite(data.get(key), f"{name} {key}") for key in (
        "x_m", "y_m", "yaw_rad",
    ))


def _observation(evidence: Mapping[str, object], name: str) -> Mapping[str, object]:
    observations = evidence.get("observations")
    matches = [
        item for item in observations
        if isinstance(item, Mapping) and item.get("name") == name
    ] if isinstance(observations, list) else []
    if len(matches) != 1 or matches[0].get("ok") is not True:
        raise ValueError(f"candidate planning frame lacks successful {name} evidence")
    data = matches[0].get("data")
    if not isinstance(data, Mapping):
        raise ValueError(f"candidate planning frame {name} capture is malformed")
    return data


def candidate_tf_capture_pose(
    data: Mapping[str, object], target: str, source: str,
) -> tuple[float, float, float]:
    """Read finite captured coordinates with matching TF frame identities."""
    target, source = _frame(target), _frame(source)
    if (
        not isinstance(data, Mapping) or not target or not source
        or target == source or data.get("available") is not True
    ) or any(
        _frame(data.get(key)) != expected
        for key, expected in (
            ("target_frame", target), ("source_frame", source),
            ("observed_target_frame", target), ("observed_source_frame", source),
        )
    ):
        raise ValueError("candidate planning frame TF capture identity mismatch")
    for name in ("stamp_sec", "capture_time_sec"):
        if _finite(data.get(name), f"TF capture {name}") < 0.0:
            raise ValueError("candidate planning frame TF capture timestamp is invalid")
    return _pose(data, f"{target}<-{source}")


def candidate_pose_from_captures(
    direct: Mapping[str, object], odom: Mapping[str, object], *,
    map_frame: str, odom_frame: str,
) -> tuple[Pose2D, PlanarTransform2D]:
    """Validate and compose saved captures, without refreshing their admission."""

    if not isinstance(direct, Mapping) or not isinstance(odom, Mapping):
        raise ValueError("candidate planning frame TF capture is malformed")
    base_frame = _frame(odom.get("source_frame"))
    map_frame, odom_frame = _frame(map_frame), _frame(odom_frame)
    if not all((map_frame, odom_frame, base_frame)) or len({
        map_frame, odom_frame, base_frame,
    }) != 3:
        raise ValueError("candidate planning frame requires distinct frame identities")
    transform = PlanarTransform2D(*candidate_tf_capture_pose(direct, map_frame, odom_frame))
    odom_pose = Pose2D(*candidate_tf_capture_pose(odom, odom_frame, base_frame))
    return odom_pose_to_map(odom_pose, transform), transform


def admitted_candidate_planning_pose(
    evidence: Mapping[str, object], *, map_frame: str, odom_frame: str,
) -> tuple[Pose2D, dict[str, object]]:
    """Return the coherent map pose and complete capture provenance.

    This reconstructs already admitted evidence; callers must additionally
    validate their required stopped localization window and route certificate.
    It cannot refresh evidence or authorize motion.
    """

    if evidence.get("ok") is not True or evidence.get("failures") != []:
        raise ValueError("candidate planning frame preflight was not admitted")
    config = evidence.get("runtime_config")
    base_frame = _frame(config.get("base_frame")) if isinstance(config, Mapping) else ""
    if not base_frame:
        raise ValueError("candidate planning frame lacks base frame identity")
    map_frame, odom_frame = _frame(map_frame), _frame(odom_frame)
    if not map_frame or not odom_frame or len({map_frame, odom_frame, base_frame}) != 3:
        raise ValueError("candidate planning frame requires distinct frame identities")
    direct = _observation(evidence, f"tf {map_frame}->{odom_frame}")
    candidate_tf_capture_pose(direct, map_frame, odom_frame)
    transform = evidence.get("map_from_odom")
    if not isinstance(transform, Mapping) or transform != direct:
        raise ValueError("candidate planning frame direct TF capture mismatch")
    _observation(evidence, "odom freshness")
    odom_capture = _observation(evidence, f"tf {odom_frame}->{base_frame}")
    observed_odom_values = candidate_tf_capture_pose(odom_capture, odom_frame, base_frame)
    odom_pose = evidence.get("odom_pose")
    if not isinstance(odom_pose, Mapping):
        raise ValueError("candidate planning frame returned no odom pose")
    if (
        _frame(odom_pose.get("frame_id")) != odom_frame
        or _frame(odom_pose.get("child_frame_id")) != base_frame
    ):
        raise ValueError("candidate planning frame odom pose frame identity mismatch")
    odom_values = _pose(odom_pose, "odom pose")
    if odom_values != observed_odom_values:
        raise ValueError("candidate planning frame odom pose capture mismatch")
    pose, _ = candidate_pose_from_captures(
        direct, odom_capture, map_frame=map_frame, odom_frame=odom_frame,
    )
    return (
        pose,
        {
            "pose_basis": "direct_map_from_odom_times_observed_odom_pose",
            "map_from_odom_capture": deepcopy(dict(direct)),
            "odom_pose_capture": deepcopy(dict(odom_capture)),
        },
    )


__all__ = [
    "admitted_candidate_planning_pose", "candidate_pose_from_captures",
    "candidate_tf_capture_pose",
]
