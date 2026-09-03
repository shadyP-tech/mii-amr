"""ROS-free JSON contract shared by preflight producers and consumers.

Keeping the exact field sets here makes serializer drift fail where evidence is
created instead of later, after a recovery route has already been prepared.
"""

from __future__ import annotations

from typing import Dict, Mapping


ROS_PREFLIGHT_EVIDENCE_FIELDS = frozenset(
    {
        "ok",
        "failures",
        "observations",
        "runtime_config",
        "preflight_requirements",
        "route_pose",
        "odom_pose",
        "map_from_odom",
        "stationary_amcl_samples",
        "stationary_map_from_odom_samples",
    }
)
ROS_PREFLIGHT_REQUIREMENTS_FIELDS = frozenset(
    {
        "stationary_map_from_odom_pairing_requested",
        "stationary_map_from_odom_pairing_required",
    }
)


def ros_preflight_requirements_evidence(
    *,
    stationary_map_from_odom_pairing_requested: bool,
    stationary_map_from_odom_pairing_required: bool,
) -> Dict[str, object]:
    """Return canonical evidence for optional ROS preflight requirements."""

    if type(stationary_map_from_odom_pairing_requested) is not bool:
        raise TypeError(
            "stationary_map_from_odom_pairing_requested must be a bool"
        )
    if type(stationary_map_from_odom_pairing_required) is not bool:
        raise TypeError(
            "stationary_map_from_odom_pairing_required must be a bool"
        )
    if (
        stationary_map_from_odom_pairing_requested
        and not stationary_map_from_odom_pairing_required
    ):
        raise ValueError(
            "stationary map-from-odom pairing cannot be requested without "
            "being required"
        )
    return {
        "stationary_map_from_odom_pairing_requested": (
            stationary_map_from_odom_pairing_requested
        ),
        "stationary_map_from_odom_pairing_required": (
            stationary_map_from_odom_pairing_required
        ),
    }


def validate_ros_preflight_requirements_evidence(
    evidence: object,
    *,
    require_explicit_stationary_map_from_odom_pairing: bool = False,
    context: str = "ROS preflight evidence",
) -> Mapping[str, object]:
    """Validate the nested requirement schema and optional explicit policy.

    Odom-owned execution can make pairing effective without an explicit caller
    request.  Candidate planning is stricter: it must record both that pairing
    was requested and that it became required.
    """

    if type(require_explicit_stationary_map_from_odom_pairing) is not bool:
        raise TypeError(
            "require_explicit_stationary_map_from_odom_pairing must be a bool"
        )
    if not isinstance(evidence, Mapping):
        raise ValueError(f"{context} preflight_requirements is malformed")
    if frozenset(evidence) != ROS_PREFLIGHT_REQUIREMENTS_FIELDS:
        raise ValueError(
            f"{context} preflight_requirements fields mismatch"
        )
    requested = evidence.get("stationary_map_from_odom_pairing_requested")
    required = evidence.get("stationary_map_from_odom_pairing_required")
    if type(requested) is not bool or type(required) is not bool:
        raise ValueError(
            f"{context} preflight_requirements flags must be booleans"
        )
    if requested and not required:
        raise ValueError(
            f"{context} preflight_requirements flags are inconsistent"
        )
    if require_explicit_stationary_map_from_odom_pairing and (
        requested is not True or required is not True
    ):
        raise ValueError(
            f"{context} did not require stationary map-from-odom pairing"
        )
    return evidence


def validate_ros_preflight_evidence_fields(
    evidence: object,
    *,
    context: str = "ROS preflight evidence",
) -> Mapping[str, object]:
    if not isinstance(evidence, Mapping):
        raise ValueError(f"{context} is malformed")
    if frozenset(evidence) != ROS_PREFLIGHT_EVIDENCE_FIELDS:
        raise ValueError(f"{context} fields mismatch")
    return evidence


__all__ = [
    "ROS_PREFLIGHT_EVIDENCE_FIELDS",
    "ROS_PREFLIGHT_REQUIREMENTS_FIELDS",
    "ros_preflight_requirements_evidence",
    "validate_ros_preflight_evidence_fields",
    "validate_ros_preflight_requirements_evidence",
]
