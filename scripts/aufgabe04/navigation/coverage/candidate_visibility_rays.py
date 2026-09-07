"""Select negative-visibility evidence without discarding neighboring returns.

Each forward scan ray is intersected with the candidate's radius-plus-uncertainty
disk.  A finite matching or occluding return on any intersecting ray vetoes
clearance, even when the nearest-center ray is clear or invalid.  In the absence
of that veto, the existing nearest-center clearance/dropout rule is preserved.
Invalid side rays are recorded, never treated as finite clearance evidence.

Geometry here is scan-relative; callers must first express the candidate and
sensor in a common frame.  This module neither confirms a stand nor admits motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    LidarVisibilityReceipt,
    validate_lidar_visibility_receipt,
)


_EPSILON = 1.0e-12


@dataclass(frozen=True)
class CandidateVisibilityRaySelection:
    classification: str
    reason: str
    selected_ray_index: int | None = None
    selected_ray_bearing_rad: float | None = None
    selected_ray_offset_rad: float | None = None
    selected_range_m: float | None = None
    intersecting_ray_indices: tuple[int, ...] = ()
    supporting_ray_indices: tuple[int, ...] = ()
    occluding_ray_indices: tuple[int, ...] = ()
    invalid_ray_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _IntersectingRay:
    index: int
    bearing_rad: float
    offset_rad: float
    near_distance_m: float
    far_distance_m: float
    range_m: float | None

    @property
    def selection_key(self) -> tuple[float, int]:
        return abs(self.offset_rad), self.index


def select_candidate_visibility_ray(
    receipt: LidarVisibilityReceipt,
    *,
    target_bearing_rad: float,
    candidate_distance_m: float,
    envelope_radius_m: float,
    far_edge_clearance_margin_m: float,
    matching_range_tolerance_m: float,
) -> CandidateVisibilityRaySelection:
    """Choose one deterministic scan classification and preserve veto evidence.

    Off-center support uses the ray's actual near/far disk intersections, not
    an angular cone or the center-distance interval.  Matching takes precedence
    over occlusion; within each class the nearest bearing and then scan index
    selects the representative.  Clearance still requires the nearest-center
    ray to exceed the conservative radial far edge plus the existing margins.
    """

    validate_lidar_visibility_receipt(receipt)
    target_bearing = _finite(target_bearing_rad, "target_bearing_rad")
    distance = _nonnegative(candidate_distance_m, "candidate_distance_m")
    radius = _nonnegative(envelope_radius_m, "envelope_radius_m")
    margin = _nonnegative(
        far_edge_clearance_margin_m, "far_edge_clearance_margin_m"
    )
    tolerance = _nonnegative(
        matching_range_tolerance_m, "matching_range_tolerance_m"
    )
    rays = _intersecting_rays(receipt, target_bearing, distance, radius)
    if not rays:
        return CandidateVisibilityRaySelection(
            classification="no_intersection",
            reason="no_scan_ray_intersects_candidate_envelope",
        )

    supporting: list[_IntersectingRay] = []
    occluding: list[_IntersectingRay] = []
    invalid: list[_IntersectingRay] = []
    for ray in rays:
        if ray.range_m is None:
            invalid.append(ray)
        elif ray.range_m < ray.near_distance_m - tolerance:
            occluding.append(ray)
        elif ray.range_m <= ray.far_distance_m + margin + tolerance:
            supporting.append(ray)

    if supporting:
        selected = min(supporting, key=lambda ray: ray.selection_key)
        classification = "matching"
        reason = "matching_return_supports_candidate"
    elif occluding:
        selected = min(occluding, key=lambda ray: ray.selection_key)
        classification = "nearer"
        reason = "nearer_return_occludes_candidate"
    else:
        selected = min(rays, key=lambda ray: ray.selection_key)
        if selected.range_m is None:
            classification = "invalid"
            reason = "selected_scan_ray_invalid"
        elif selected.range_m <= distance + radius + margin + tolerance:
            # Preserve the old, more conservative radial clearance threshold
            # even when the selected off-axis ray exits the disk earlier.
            classification = "matching"
            reason = "matching_return_supports_candidate"
            supporting.append(selected)
        else:
            classification = "clear"
            reason = "finite_ray_clears_candidate_far_edge"

    return CandidateVisibilityRaySelection(
        classification=classification,
        reason=reason,
        selected_ray_index=selected.index,
        selected_ray_bearing_rad=selected.bearing_rad,
        selected_ray_offset_rad=selected.offset_rad,
        selected_range_m=selected.range_m,
        intersecting_ray_indices=tuple(ray.index for ray in rays),
        supporting_ray_indices=tuple(ray.index for ray in supporting),
        occluding_ray_indices=tuple(ray.index for ray in occluding),
        invalid_ray_indices=tuple(ray.index for ray in invalid),
    )


def _intersecting_rays(
    receipt: LidarVisibilityReceipt,
    target_bearing_rad: float,
    distance_m: float,
    radius_m: float,
) -> tuple[_IntersectingRay, ...]:
    result: list[_IntersectingRay] = []
    for index, range_m in enumerate(receipt.ranges_m):
        bearing = receipt.angle_min_rad + index * receipt.angle_increment_rad
        offset = math.atan2(
            math.sin(bearing - target_bearing_rad),
            math.cos(bearing - target_bearing_rad),
        )
        perpendicular = distance_m * abs(math.sin(offset))
        if perpendicular > radius_m + _EPSILON:
            continue
        projection = distance_m * math.cos(offset)
        half_chord = math.sqrt(max(0.0, radius_m**2 - perpendicular**2))
        far_distance = projection + half_chord
        if far_distance < 0.0:
            continue
        result.append(
            _IntersectingRay(
                index=index,
                bearing_rad=bearing,
                offset_rad=offset,
                near_distance_m=max(0.0, projection - half_chord),
                far_distance_m=far_distance,
                range_m=range_m,
            )
        )
    return tuple(result)


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _nonnegative(value: float, name: str) -> float:
    parsed = _finite(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


__all__ = ["CandidateVisibilityRaySelection", "select_candidate_visibility_ray"]
