"""Immutable frozen-frame provenance for exact-time LiDAR scan receipts.

The observation certificate identifies the epoch's ``map <- odom`` transform.
Keeping the exact odom scan pose lets consumers verify the recorded map pose
and compare candidate geometry across localization corrections without ROS.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
    normalize_yaw,
)


_FRAME = re.compile(r"^/?[A-Za-z][A-Za-z0-9_/.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_POSE_FIELDS = frozenset({"x_m", "y_m", "yaw_rad"})
_FIELDS = frozenset(
    {
        "map_frame",
        "odom_frame",
        "map_from_odom",
        "canonical_scan_pose_odom",
        "source_evidence_id",
    }
)
_POSE_CONSISTENCY_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class LidarVisibilityFrameProvenance:
    """Bind an exact odom scan pose to one certified frozen map transform."""

    map_frame: str
    odom_frame: str
    map_from_odom: PlanarTransform2D
    canonical_scan_pose_odom: Pose2D
    source_evidence_id: str

    def __post_init__(self) -> None:
        for name in ("map_frame", "odom_frame"):
            value = getattr(self, name)
            if not isinstance(value, str) or _FRAME.fullmatch(value) is None:
                raise ValueError(f"visibility frame provenance {name} is invalid")
        if self.map_frame.lstrip("/") == self.odom_frame.lstrip("/"):
            raise ValueError("visibility map and odom frames must differ")
        if not isinstance(self.map_from_odom, PlanarTransform2D):
            raise ValueError("map_from_odom must be a PlanarTransform2D")
        _validated_pose(self.canonical_scan_pose_odom, "canonical_scan_pose_odom")
        if (
            not isinstance(self.source_evidence_id, str)
            or _SHA256.fullmatch(self.source_evidence_id) is None
        ):
            raise ValueError(
                "source_evidence_id must be a lowercase certificate SHA-256"
            )

    def validate_scan_pose_map(self, scan_pose_map: Pose2D) -> None:
        """Reject inconsistent map and canonical odom geometry."""

        _validated_pose(scan_pose_map, "scan_pose_map")
        transform = self.map_from_odom
        pose = self.canonical_scan_pose_odom
        cosine, sine = math.cos(transform.yaw_rad), math.sin(transform.yaw_rad)
        expected_x = cosine * pose.x_m - sine * pose.y_m + transform.x_m
        expected_y = sine * pose.x_m + cosine * pose.y_m + transform.y_m
        position_error = math.hypot(
            scan_pose_map.x_m - expected_x, scan_pose_map.y_m - expected_y
        )
        yaw_error = abs(
            normalize_yaw(
                scan_pose_map.yaw_rad - pose.yaw_rad - transform.yaw_rad
            )
        )
        if (
            not math.isfinite(position_error)
            or max(position_error, yaw_error) > _POSE_CONSISTENCY_TOLERANCE
        ):
            raise ValueError("scan_pose_map disagrees with frozen-frame provenance")

    def to_mapping(self) -> dict[str, object]:
        return {
            "map_frame": self.map_frame,
            "odom_frame": self.odom_frame,
            "map_from_odom": _pose_mapping(self.map_from_odom),
            "canonical_scan_pose_odom": _pose_mapping(
                self.canonical_scan_pose_odom
            ),
            "source_evidence_id": self.source_evidence_id,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "LidarVisibilityFrameProvenance":
        if not isinstance(value, Mapping) or frozenset(value) != _FIELDS:
            raise ValueError("visibility frame provenance fields do not match schema")
        transform = _pose_from_mapping(value["map_from_odom"], "map_from_odom")
        return cls(
            map_frame=value["map_frame"],
            odom_frame=value["odom_frame"],
            map_from_odom=PlanarTransform2D(
                transform.x_m, transform.y_m, transform.yaw_rad
            ),
            canonical_scan_pose_odom=_pose_from_mapping(
                value["canonical_scan_pose_odom"], "canonical_scan_pose_odom"
            ),
            source_evidence_id=value["source_evidence_id"],
        )


def _validated_pose(value: Pose2D, name: str) -> None:
    if not isinstance(value, Pose2D):
        raise ValueError(f"{name} must be a finite Pose2D")
    for field in _POSE_FIELDS:
        _finite_number(getattr(value, field), f"{name}.{field}")


def _pose_mapping(value: Pose2D | PlanarTransform2D) -> dict[str, float]:
    return {name: float(getattr(value, name)) for name in sorted(_POSE_FIELDS)}


def _pose_from_mapping(value: object, name: str) -> Pose2D:
    if not isinstance(value, Mapping) or frozenset(value) != _POSE_FIELDS:
        raise ValueError(f"{name} fields do not match schema")
    return Pose2D(
        *(
            _finite_number(value[field], f"{name}.{field}")
            for field in ("x_m", "y_m", "yaw_rad")
        )
    )


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


__all__ = ["LidarVisibilityFrameProvenance"]
