"""ROS-free LiDAR association for a known stand candidate.

The passive camera observer already has a map-derived target bearing and a
fail-closed accepted surface-range interval.  This module keeps both gates
intact, but avoids treating unrelated background returns as part of the target:
samples are range-gated first and only then aggregated into contiguous scan
clusters.

The optional camera bearing must agree with the map cone and can then rank
clusters that already passed the map cone and range gates.  It never expands
either gate and this module has no motion or ROS side effects.

When a separately validated camera-registration stage has found that the
candidate is displaced from its map projection, the registered helper below
can move the *same narrow cone* to the observed camera bearing.  That path is
deliberately separate from :func:`associate_candidate_lidar_target`: it has a
bounded camera/map delta, retains the strict candidate range gate, and requires
exactly one eligible cluster.  Its wrapper records both bearings so a
camera-centred search can never be mistaken for the legacy map-centred search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan


CANDIDATE_LIDAR_ASSOCIATION_SCHEMA_VERSION = 1
CAMERA_REGISTERED_LIDAR_ASSOCIATION_SCHEMA_VERSION = 1
MAX_CAMERA_MAP_BEARING_DELTA_DEG = 12.0
MAX_CAMERA_MAP_BEARING_DELTA_RAD = math.radians(
    MAX_CAMERA_MAP_BEARING_DELTA_DEG
)
DEFAULT_MAX_CAMERA_MAP_BEARING_DELTA_RAD = MAX_CAMERA_MAP_BEARING_DELTA_RAD
CAMERA_MAP_BEARING_DELTA_BOUND_TOLERANCE_RAD = 1.0e-12


@dataclass(frozen=True)
class CandidateLidarAssociation:
    """Auditable result of associating one scan with one mapped candidate."""

    schema_version: int
    associated: bool
    distance_m: float | None
    rejection_reason: str
    map_bearing_rad: float
    cone_half_angle_rad: float
    accepted_range_m: tuple[float, float]
    scan_frame_id: str
    scan_stamp_sec: float | None
    scan_age_sec: float | None
    cone_valid_sample_count: int
    in_range_sample_count: int
    candidate_cluster_count: int
    eligible_cluster_count: int
    min_cluster_sample_count: int
    max_range_jump_m: float
    max_point_gap_m: float
    selected_cluster_sample_count: int
    selected_cluster_start_index: int | None
    selected_cluster_end_index: int | None
    selected_cluster_bearing_rad: float | None
    selected_cluster_bearing_delta_from_map_rad: float | None
    observed_camera_bearing_rad: float | None
    observed_camera_bearing_delta_from_map_rad: float | None
    selected_cluster_bearing_delta_from_camera_rad: float | None
    selection_source: str
    nearest_cone_distance_m: float | None
    nearest_range_delta_m: float | None


@dataclass(frozen=True)
class CameraRegisteredCandidateLidarAssociation:
    """Association evidence for a bounded camera-centred cone search.

    ``search_association.map_bearing_rad`` is the centre actually supplied to
    the unchanged legacy association engine.  The explicitly named fields on
    this wrapper retain the original map bearing and make that reinterpretation
    auditable without changing the legacy result schema.
    """

    schema_version: int
    associated: bool
    distance_m: float | None
    rejection_reason: str
    map_bearing_rad: float
    registered_search_bearing_rad: float
    camera_map_bearing_delta_rad: float
    max_camera_map_bearing_delta_rad: float
    search_bearing_source: str
    unique_eligible_cluster_required: bool
    search_association: CandidateLidarAssociation | None


@dataclass(frozen=True)
class _ScanSample:
    index: int
    bearing_rad: float
    distance_m: float


@dataclass(frozen=True)
class _ContiguousCluster:
    samples: tuple[_ScanSample, ...]
    distance_m: float
    bearing_rad: float

    @property
    def start_index(self) -> int:
        return self.samples[0].index

    @property
    def end_index(self) -> int:
        return self.samples[-1].index


def associate_candidate_lidar_target(
    scan: PlainLaserScan | None,
    *,
    map_bearing_rad: float,
    cone_half_angle_rad: float,
    accepted_range_m: tuple[float, float],
    now_sec: float | None = None,
    max_scan_age_sec: float = 0.0,
    min_cluster_sample_count: int = 1,
    max_range_jump_m: float = 0.05,
    max_point_gap_m: float = 0.04,
    observed_camera_bearing_rad: float | None = None,
) -> CandidateLidarAssociation:
    """Associate a scan cluster with a mapped stand candidate.

    Only finite scan samples inside the original map-centred cone are
    considered.  Those samples are filtered by ``accepted_range_m`` before
    contiguous clusters and their medians are computed.  Consequently a wall
    or other background return cannot pull a valid thin stand return outside
    the accepted interval.

    ``observed_camera_bearing_rad`` must itself agree with the original map
    cone and then changes only cluster ranking.  It cannot admit a sample
    outside the original map cone or range interval.
    """

    map_bearing_rad = _require_finite(map_bearing_rad, "map_bearing_rad")
    cone_half_angle_rad = _require_nonnegative(
        cone_half_angle_rad,
        "cone_half_angle_rad",
    )
    lower_range_m, upper_range_m = _validated_range(accepted_range_m)
    max_scan_age_sec = _require_nonnegative(max_scan_age_sec, "max_scan_age_sec")
    min_cluster_sample_count = max(1, int(min_cluster_sample_count))
    max_range_jump_m = _require_nonnegative(max_range_jump_m, "max_range_jump_m")
    max_point_gap_m = _require_nonnegative(max_point_gap_m, "max_point_gap_m")

    camera_bearing_delta_from_map_rad = None
    if observed_camera_bearing_rad is not None:
        observed_camera_bearing_rad = _require_finite(
            observed_camera_bearing_rad,
            "observed_camera_bearing_rad",
        )
        camera_bearing_delta_from_map_rad = abs(
            _angle_delta(observed_camera_bearing_rad, map_bearing_rad)
        )

    common = {
        "schema_version": CANDIDATE_LIDAR_ASSOCIATION_SCHEMA_VERSION,
        "map_bearing_rad": map_bearing_rad,
        "cone_half_angle_rad": cone_half_angle_rad,
        "accepted_range_m": (lower_range_m, upper_range_m),
        "min_cluster_sample_count": min_cluster_sample_count,
        "max_range_jump_m": max_range_jump_m,
        "max_point_gap_m": max_point_gap_m,
        "observed_camera_bearing_rad": observed_camera_bearing_rad,
        "observed_camera_bearing_delta_from_map_rad": (
            camera_bearing_delta_from_map_rad
        ),
    }
    if (
        camera_bearing_delta_from_map_rad is not None
        and camera_bearing_delta_from_map_rad > cone_half_angle_rad
    ):
        return _rejected(common, rejection_reason="camera_bearing_outside_map_cone")
    if scan is None:
        return _rejected(common, rejection_reason="no_scan")

    scan_age_sec = _scan_age(scan, now_sec=now_sec)
    scan_common = {
        **common,
        "scan_frame_id": scan.scan_frame_id,
        "scan_stamp_sec": scan.scan_stamp_sec,
        "scan_age_sec": scan_age_sec,
    }
    if not _valid_scan_geometry(scan):
        return _rejected(scan_common, rejection_reason="invalid_scan_geometry")
    if max_scan_age_sec > 0.0 and scan_age_sec is not None and scan_age_sec > max_scan_age_sec:
        return _rejected(scan_common, rejection_reason="stale_scan")

    cone_samples = _valid_samples_in_map_cone(
        scan,
        map_bearing_rad=map_bearing_rad,
        cone_half_angle_rad=cone_half_angle_rad,
    )
    nearest_cone_distance_m, nearest_range_delta_m = _nearest_range_evidence(
        cone_samples,
        lower_range_m=lower_range_m,
        upper_range_m=upper_range_m,
    )
    evidence = {
        **scan_common,
        "cone_valid_sample_count": len(cone_samples),
        "nearest_cone_distance_m": nearest_cone_distance_m,
        "nearest_range_delta_m": nearest_range_delta_m,
    }
    if not cone_samples:
        return _rejected(evidence, rejection_reason="no_valid_samples_in_map_cone")

    # The safety-critical ordering is intentional: unrelated cone returns are
    # discarded before any median or cluster aggregation can influence the
    # candidate distance.
    in_range_samples = tuple(
        sample
        for sample in cone_samples
        if lower_range_m <= sample.distance_m <= upper_range_m
    )
    clusters = _contiguous_clusters(
        in_range_samples,
        max_range_jump_m=max_range_jump_m,
        max_point_gap_m=max_point_gap_m,
    )
    eligible = tuple(
        cluster for cluster in clusters if len(cluster.samples) >= min_cluster_sample_count
    )
    clustered_evidence = {
        **evidence,
        "in_range_sample_count": len(in_range_samples),
        "candidate_cluster_count": len(clusters),
        "eligible_cluster_count": len(eligible),
    }
    if not in_range_samples:
        return _rejected(clustered_evidence, rejection_reason="no_samples_in_accepted_range")
    if not eligible:
        return _rejected(
            clustered_evidence,
            rejection_reason="no_contiguous_cluster_meets_minimum",
        )

    score_bearing_rad = (
        observed_camera_bearing_rad
        if observed_camera_bearing_rad is not None
        else map_bearing_rad
    )
    selected = min(
        eligible,
        key=lambda cluster: (
            abs(_angle_delta(cluster.bearing_rad, score_bearing_rad)),
            -len(cluster.samples),
            abs(_angle_delta(cluster.bearing_rad, map_bearing_rad)),
            cluster.start_index,
        ),
    )
    camera_delta = None
    if observed_camera_bearing_rad is not None:
        camera_delta = abs(_angle_delta(selected.bearing_rad, observed_camera_bearing_rad))

    return CandidateLidarAssociation(
        **clustered_evidence,
        associated=True,
        distance_m=selected.distance_m,
        rejection_reason="",
        selected_cluster_sample_count=len(selected.samples),
        selected_cluster_start_index=selected.start_index,
        selected_cluster_end_index=selected.end_index,
        selected_cluster_bearing_rad=selected.bearing_rad,
        selected_cluster_bearing_delta_from_map_rad=abs(
            _angle_delta(selected.bearing_rad, map_bearing_rad)
        ),
        selected_cluster_bearing_delta_from_camera_rad=camera_delta,
        selection_source=(
            "camera_bearing" if observed_camera_bearing_rad is not None else "map_bearing"
        ),
    )


def associate_camera_registered_candidate_lidar_target(
    scan: PlainLaserScan | None,
    *,
    map_bearing_rad: float,
    observed_camera_bearing_rad: float,
    cone_half_angle_rad: float,
    accepted_range_m: tuple[float, float],
    now_sec: float | None = None,
    max_scan_age_sec: float = 0.0,
    min_cluster_sample_count: int = 1,
    max_range_jump_m: float = 0.05,
    max_point_gap_m: float = 0.04,
    max_camera_map_bearing_delta_rad: float = (
        DEFAULT_MAX_CAMERA_MAP_BEARING_DELTA_RAD
    ),
) -> CameraRegisteredCandidateLidarAssociation:
    """Associate a candidate in a bounded camera-registered LiDAR cone.

    This is an explicit exception path for a target registration that has
    already been established from camera geometry.  It does not widen the
    normal map-centred cone.  Instead, it first requires the wrapped difference
    between the finite map and camera bearings to be within
    ``max_camera_map_bearing_delta_rad``, then evaluates the unchanged narrow
    cone and strict surface-range gate at the camera bearing.

    Unlike the legacy ranking path, camera registration must leave exactly one
    eligible contiguous cluster.  Multiple clusters therefore fail closed
    instead of selecting whichever happens to be closest to the registered
    bearing.
    """

    map_bearing_rad = _require_finite(map_bearing_rad, "map_bearing_rad")
    observed_camera_bearing_rad = _require_finite(
        observed_camera_bearing_rad,
        "observed_camera_bearing_rad",
    )
    max_camera_map_bearing_delta_rad = (
        normalize_certified_camera_map_bearing_limit(
            max_camera_map_bearing_delta_rad
        )
    )
    camera_map_bearing_delta_rad = abs(
        _angle_delta(observed_camera_bearing_rad, map_bearing_rad)
    )
    common = {
        "schema_version": CAMERA_REGISTERED_LIDAR_ASSOCIATION_SCHEMA_VERSION,
        "map_bearing_rad": map_bearing_rad,
        "registered_search_bearing_rad": observed_camera_bearing_rad,
        "camera_map_bearing_delta_rad": camera_map_bearing_delta_rad,
        "max_camera_map_bearing_delta_rad": max_camera_map_bearing_delta_rad,
        "search_bearing_source": "registered_camera_bearing",
        "unique_eligible_cluster_required": True,
    }
    if camera_map_bearing_delta_rad > max_camera_map_bearing_delta_rad:
        return CameraRegisteredCandidateLidarAssociation(
            **common,
            associated=False,
            distance_m=None,
            rejection_reason="camera_map_bearing_delta_exceeds_limit",
            search_association=None,
        )

    # Reuse the legacy association without altering its behaviour.  Supplying
    # the registered camera bearing as its map-centred search bearing moves the
    # narrow cone while retaining scan validation, range-first filtering,
    # clustering, staleness checks, and deterministic evidence generation.
    search_association = associate_candidate_lidar_target(
        scan,
        map_bearing_rad=observed_camera_bearing_rad,
        cone_half_angle_rad=cone_half_angle_rad,
        accepted_range_m=accepted_range_m,
        now_sec=now_sec,
        max_scan_age_sec=max_scan_age_sec,
        min_cluster_sample_count=min_cluster_sample_count,
        max_range_jump_m=max_range_jump_m,
        max_point_gap_m=max_point_gap_m,
    )
    if search_association.eligible_cluster_count > 1:
        return CameraRegisteredCandidateLidarAssociation(
            **common,
            associated=False,
            distance_m=None,
            rejection_reason="ambiguous_registered_camera_clusters",
            search_association=_association_rejected_as_ambiguous(
                search_association
            ),
        )
    return CameraRegisteredCandidateLidarAssociation(
        **common,
        associated=search_association.associated,
        distance_m=search_association.distance_m,
        rejection_reason=search_association.rejection_reason,
        search_association=search_association,
    )


def _association_rejected_as_ambiguous(
    association: CandidateLidarAssociation,
) -> CandidateLidarAssociation:
    """Clear any selected cluster while retaining ambiguity evidence."""

    return replace(
        association,
        associated=False,
        distance_m=None,
        rejection_reason="ambiguous_registered_camera_clusters",
        selected_cluster_sample_count=0,
        selected_cluster_start_index=None,
        selected_cluster_end_index=None,
        selected_cluster_bearing_rad=None,
        selected_cluster_bearing_delta_from_map_rad=None,
        selected_cluster_bearing_delta_from_camera_rad=None,
        selection_source="none",
    )


def normalize_certified_camera_map_bearing_limit(value: float) -> float:
    """Return a limit at or below the certified camera-registration cap.

    CLI values cross a degrees-to-radians boundary.  A mathematically exact
    12 degree default can therefore arrive a few floating-point ulps above the
    radian constant.  Values inside a numerical-only tolerance are clamped to
    the certified value; a meaningful increase still fails closed.
    """

    limit = _require_nonnegative(value, "max_camera_map_bearing_delta_rad")
    if (
        limit
        > MAX_CAMERA_MAP_BEARING_DELTA_RAD
        + CAMERA_MAP_BEARING_DELTA_BOUND_TOLERANCE_RAD
    ):
        raise ValueError(
            "max_camera_map_bearing_delta_rad cannot exceed the certified "
            f"{MAX_CAMERA_MAP_BEARING_DELTA_DEG:g} degree registration bound"
        )
    return min(limit, MAX_CAMERA_MAP_BEARING_DELTA_RAD)


def _rejected(
    evidence: dict[str, object],
    *,
    rejection_reason: str,
) -> CandidateLidarAssociation:
    defaults = {
        "scan_frame_id": "",
        "scan_stamp_sec": None,
        "scan_age_sec": None,
        "cone_valid_sample_count": 0,
        "in_range_sample_count": 0,
        "candidate_cluster_count": 0,
        "eligible_cluster_count": 0,
        "selected_cluster_sample_count": 0,
        "selected_cluster_start_index": None,
        "selected_cluster_end_index": None,
        "selected_cluster_bearing_rad": None,
        "selected_cluster_bearing_delta_from_map_rad": None,
        "selected_cluster_bearing_delta_from_camera_rad": None,
        "selection_source": "none",
        "nearest_cone_distance_m": None,
        "nearest_range_delta_m": None,
    }
    defaults.update(evidence)
    return CandidateLidarAssociation(
        **defaults,
        associated=False,
        distance_m=None,
        rejection_reason=rejection_reason,
    )


def _valid_samples_in_map_cone(
    scan: PlainLaserScan,
    *,
    map_bearing_rad: float,
    cone_half_angle_rad: float,
) -> tuple[_ScanSample, ...]:
    samples = []
    for index, raw_range in enumerate(scan.ranges):
        try:
            distance_m = float(raw_range)
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(distance_m)
            or distance_m < scan.range_min
            or distance_m > scan.range_max
        ):
            continue
        bearing_rad = _normalize_angle(scan.angle_min + index * scan.angle_increment)
        if abs(_angle_delta(bearing_rad, map_bearing_rad)) <= cone_half_angle_rad:
            samples.append(
                _ScanSample(
                    index=index,
                    bearing_rad=bearing_rad,
                    distance_m=distance_m,
                )
            )
    return tuple(samples)


def _contiguous_clusters(
    samples: tuple[_ScanSample, ...],
    *,
    max_range_jump_m: float,
    max_point_gap_m: float,
) -> tuple[_ContiguousCluster, ...]:
    if not samples:
        return ()
    groups: list[list[_ScanSample]] = []
    for sample in samples:
        previous = groups[-1][-1] if groups else None
        if previous is None or not _samples_are_contiguous(
            previous,
            sample,
            max_range_jump_m=max_range_jump_m,
            max_point_gap_m=max_point_gap_m,
        ):
            groups.append([sample])
        else:
            groups[-1].append(sample)
    return tuple(_build_cluster(tuple(group)) for group in groups)


def _samples_are_contiguous(
    left: _ScanSample,
    right: _ScanSample,
    *,
    max_range_jump_m: float,
    max_point_gap_m: float,
) -> bool:
    if right.index != left.index + 1:
        return False
    if abs(right.distance_m - left.distance_m) > max_range_jump_m:
        return False
    bearing_delta = abs(_angle_delta(right.bearing_rad, left.bearing_rad))
    point_gap_m = math.sqrt(
        max(
            0.0,
            left.distance_m**2
            + right.distance_m**2
            - 2.0 * left.distance_m * right.distance_m * math.cos(bearing_delta),
        )
    )
    return point_gap_m <= max_point_gap_m


def _build_cluster(samples: tuple[_ScanSample, ...]) -> _ContiguousCluster:
    return _ContiguousCluster(
        samples=samples,
        distance_m=_median(tuple(sample.distance_m for sample in samples)),
        bearing_rad=_circular_mean(tuple(sample.bearing_rad for sample in samples)),
    )


def _nearest_range_evidence(
    samples: tuple[_ScanSample, ...],
    *,
    lower_range_m: float,
    upper_range_m: float,
) -> tuple[float | None, float | None]:
    if not samples:
        return None, None

    def delta(distance_m: float) -> float:
        if distance_m < lower_range_m:
            return lower_range_m - distance_m
        if distance_m > upper_range_m:
            return distance_m - upper_range_m
        return 0.0

    nearest = min(samples, key=lambda sample: (delta(sample.distance_m), sample.index))
    return nearest.distance_m, delta(nearest.distance_m)


def _scan_age(scan: PlainLaserScan, *, now_sec: float | None) -> float | None:
    if now_sec is None or scan.receipt_sec is None:
        return None
    try:
        now = float(now_sec)
        receipt = float(scan.receipt_sec)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(now) or not math.isfinite(receipt):
        return None
    return max(0.0, now - receipt)


def _valid_scan_geometry(scan: PlainLaserScan) -> bool:
    return (
        math.isfinite(scan.angle_min)
        and math.isfinite(scan.angle_increment)
        and scan.angle_increment != 0.0
        and math.isfinite(scan.range_min)
        and math.isfinite(scan.range_max)
        and 0.0 <= scan.range_min < scan.range_max
    )


def _validated_range(accepted_range_m: tuple[float, float]) -> tuple[float, float]:
    if len(accepted_range_m) != 2:
        raise ValueError("accepted_range_m must contain exactly two bounds")
    lower = _require_nonnegative(accepted_range_m[0], "accepted_range_m[0]")
    upper = _require_nonnegative(accepted_range_m[1], "accepted_range_m[1]")
    if lower > upper:
        raise ValueError("accepted_range_m lower bound must not exceed upper bound")
    return lower, upper


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _circular_mean(angles: tuple[float, ...]) -> float:
    sine = sum(math.sin(angle) for angle in angles)
    cosine = sum(math.cos(angle) for angle in angles)
    return _normalize_angle(math.atan2(sine, cosine))


def _angle_delta(left: float, right: float) -> float:
    return _normalize_angle(left - right)


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _require_finite(value: float, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _require_nonnegative(value: float, name: str) -> float:
    numeric = _require_finite(value, name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return numeric
