"""ROS-free data contracts for surveyed stand arrival poses.

The occupancy map remains the geometric source of truth.  These models form a
small semantic layer over that map: one validated perpendicular arrival pose
for every surveyed stand candidate, plus enough provenance to reject reuse in
another map, world, or simulation session.

Mutation, validation, hashing, and JSON persistence live in
``arrival_pose_catalog``.  Keeping these dataclasses dependency-free lets the
perception, planning, and mission layers share the same contract without
importing ROS.
"""

from __future__ import annotations

from dataclasses import dataclass


ARRIVAL_POSE_CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CatalogPose2D:
    """An exact map-frame pose; no grid-cell snapping is implied."""

    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class CatalogProvenance:
    """Identity of the coordinate system and environment behind a catalog."""

    planning_frame: str
    map_yaml_sha256: str
    world_id: str
    world_sha256: str
    session_id: str
    environment: str


@dataclass(frozen=True)
class StandEstimate:
    """LiDAR-derived stand geometry used by the arrival-pose calculation."""

    x_m: float
    y_m: float
    radius_m: float
    uncertainty_m: float


@dataclass(frozen=True)
class AxisEstimate:
    """A 180-degree-symmetric head-silhouette axis estimate."""

    axis_rad: float
    confidence: float
    sample_count: int
    estimator: str
    observation_unix_sec: float


@dataclass(frozen=True)
class FaceSelection:
    """The outward stand face normal selected for the future approach."""

    face_id: str
    outward_normal_rad: float
    identity_resolved: bool
    evidence_kind: str
    evidence_confidence: float
    evidence_hard: bool
    evidence_valid: bool
    evidence_provenance: str


@dataclass(frozen=True)
class ArrivalPoseValidation:
    """Map checks that must all pass before an estimate may enter the catalog."""

    target_in_bounds: bool
    target_collision_free: bool
    corridor_collision_free: bool
    validated_map_yaml_sha256: str
    validated_unix_sec: float


@dataclass(frozen=True)
class ArrivalPoseRecord:
    """One immutable, validated result of surveying a stand candidate."""

    candidate_uid: str
    stand_id: str
    stand: StandEstimate
    axis: AxisEstimate
    face: FaceSelection
    arrival_pose: CatalogPose2D
    corridor_entry_pose: CatalogPose2D
    standoff_m: float
    corridor_length_m: float
    validation: ArrivalPoseValidation
    source_observation_ids: tuple[str, ...]
    sensor_stamp_sec: float
    source: str


@dataclass(frozen=True)
class CandidateRejection:
    """A terminal survey decision for a candidate that cannot yield a pose."""

    candidate_uid: str
    reason: str
    source_observation_ids: tuple[str, ...]
    rejected_unix_sec: float


@dataclass(frozen=True)
class ArrivalPoseCatalog:
    """An immutable semantic-map snapshot.

    ``revision`` changes for every material catalog mutation.  The content
    SHA-256 is deliberately derived rather than stored in memory so callers
    cannot accidentally carry a stale hash after constructing a new snapshot.
    """

    schema_version: int
    catalog_id: str
    provenance: CatalogProvenance
    revision: int
    frozen: bool
    created_unix_sec: float
    updated_unix_sec: float
    frozen_unix_sec: float | None
    expected_candidate_uids: tuple[str, ...]
    records: tuple[ArrivalPoseRecord, ...]
    rejections: tuple[CandidateRejection, ...]

    @property
    def ready_candidate_uids(self) -> frozenset[str]:
        return frozenset(record.candidate_uid for record in self.records)

    @property
    def rejected_candidate_uids(self) -> frozenset[str]:
        return frozenset(rejection.candidate_uid for rejection in self.rejections)

    @property
    def resolved_candidate_uids(self) -> frozenset[str]:
        return self.ready_candidate_uids | self.rejected_candidate_uids

    @property
    def complete(self) -> bool:
        expected = frozenset(self.expected_candidate_uids)
        return bool(expected) and self.resolved_candidate_uids == expected

    def record_for(self, candidate_uid: str) -> ArrivalPoseRecord | None:
        return next(
            (record for record in self.records if record.candidate_uid == candidate_uid),
            None,
        )

