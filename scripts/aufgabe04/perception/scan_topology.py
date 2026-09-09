"""Conservative endpoint adjacency for explicitly full-rotation LaserScans.

Near-360-degree coverage alone does not declare a scan circular. The caller
must supply its scanner/profile contract and the original unfiltered geometry.
Wider boundary gaps remain linear, including the two-step gaps in real LDS
receipts. This helper never changes a caller's spatial or range gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


SCAN_TOPOLOGY_PROFILES = ("linear", "full_rotation")
MAX_SEAM_STEP_ERROR = 0.10
MAX_ENDPOINT_METADATA_ERROR_STEPS = 0.25
_EPSILON = 1.0e-9


@dataclass(frozen=True)
class ScanTopology:
    sample_count: int
    angle_min_rad: float
    angle_increment_rad: float
    angle_max_rad: float | None = None
    profile: str = "linear"

    def evidence(self) -> dict[str, object]:
        result = {
            "profile": self.profile,
            "sample_count": self.sample_count if type(self.sample_count) is int else None,
            "circular_adjacency_enabled": False,
            "reason": "linear_profile",
            "indexed_seam_gap_steps": None,
            "reported_seam_gap_steps": None,
            "endpoint_metadata_error_steps": None,
            "maximum_seam_step_error": MAX_SEAM_STEP_ERROR,
            "maximum_endpoint_metadata_error_steps": MAX_ENDPOINT_METADATA_ERROR_STEPS,
        }
        for name in ("angle_min_rad", "angle_max_rad", "angle_increment_rad"):
            value = getattr(self, name)
            result[name] = value if type(value) in (int, float) and math.isfinite(value) else None
        if self.profile != "full_rotation":
            if self.profile != "linear":
                result["reason"] = "unknown_scan_topology_profile"
            return result
        if (
            type(self.sample_count) is not int or self.sample_count < 3
            or not all(type(value) in (int, float) and math.isfinite(value)
                       for value in (self.angle_min_rad, self.angle_max_rad, self.angle_increment_rad))
            or self.angle_increment_rad == 0.0
        ):
            result["reason"] = "invalid_original_scan_geometry"
            return result
        step = abs(self.angle_increment_rad)
        direction = 1.0 if self.angle_increment_rad > 0.0 else -1.0
        indexed_span = (self.sample_count - 1) * step
        reported_span = direction * (self.angle_max_rad - self.angle_min_rad)
        if not (0.0 < indexed_span < math.tau and 0.0 < reported_span < math.tau):
            result["reason"] = "partial_or_overlapping_scan_geometry"
            return result
        indexed_gap = (math.tau - indexed_span) / step
        reported_gap = (math.tau - reported_span) / step
        endpoint_error = abs(reported_span - indexed_span) / step
        result.update(indexed_seam_gap_steps=indexed_gap,
                      reported_seam_gap_steps=reported_gap,
                      endpoint_metadata_error_steps=endpoint_error)
        if endpoint_error > MAX_ENDPOINT_METADATA_ERROR_STEPS + _EPSILON:
            result["reason"] = "inconsistent_scan_endpoint_metadata"
        elif any(abs(gap - 1.0) > MAX_SEAM_STEP_ERROR + _EPSILON
                 for gap in (indexed_gap, reported_gap)):
            result["reason"] = "seam_is_not_one_sampling_step"
        else:
            result.update(circular_adjacency_enabled=True, reason="validated_full_rotation")
        return result

    def joins_endpoints(self, left_index: int, right_index: int) -> bool:
        return (
            self.evidence()["circular_adjacency_enabled"] is True
            and type(left_index) is int and type(right_index) is int
            and left_index == self.sample_count - 1 and right_index == 0
        )


def wraps_scan_seam(source_indices) -> bool:
    """Wrapped groups are represented in acquisition order across N-1 -> 0."""

    return any(right < left for left, right in zip(source_indices, source_indices[1:]))
