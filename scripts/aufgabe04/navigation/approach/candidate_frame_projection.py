"""Bind frozen LiDAR candidates to one current camera-planning frame.

The immutable detector snapshot remains the source artifact.  This module
creates a derived, in-memory snapshot whose x/y geometry is expressed under a
fresh stationary ``map <- odom`` transform.  It is ROS-free, writes nothing,
and never authorizes motion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameReprojectionResult,
    reproject_candidate_point,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    StandSurveyRegistry,
    stand_survey_registry_sha256,
    validate_stand_survey_registry,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    candidate_snapshot_sha256,
    validate_candidate_snapshot,
)


CANDIDATE_FRAME_PROJECTION_SCHEMA_VERSION = 1


class CandidateFrameProjectionError(ValueError):
    """Execution-frame projection failure with a stable error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CandidatePlanningFrame:
    """One stationary planning pose and its simultaneous map/odom frame."""

    current_pose: Pose2D
    map_from_odom: PlanarTransform2D
    map_frame: str = "map"
    odom_frame: str = "odom"

    def __post_init__(self) -> None:
        if not isinstance(self.current_pose, Pose2D):
            raise CandidateFrameProjectionError(
                "invalid_planning_frame", "current_pose must be a Pose2D"
            )
        if not isinstance(self.map_from_odom, PlanarTransform2D):
            raise CandidateFrameProjectionError(
                "invalid_planning_frame",
                "map_from_odom must be a PlanarTransform2D",
            )
        # Reconstruct both values to run their finite/canonical validation.
        pose_values = (
            float(self.current_pose.x_m),
            float(self.current_pose.y_m),
            float(self.current_pose.yaw_rad),
        )
        if not all(math.isfinite(value) for value in pose_values):
            raise CandidateFrameProjectionError(
                "invalid_planning_frame", "current_pose must be finite"
            )
        Pose2D(*pose_values)
        PlanarTransform2D(
            self.map_from_odom.x_m,
            self.map_from_odom.y_m,
            self.map_from_odom.yaw_rad,
        )
        map_frame = _frame_id(self.map_frame, "map_frame")
        odom_frame = _frame_id(self.odom_frame, "odom_frame")
        if map_frame == odom_frame:
            raise CandidateFrameProjectionError(
                "invalid_planning_frame",
                "map_frame and odom_frame must be distinct",
            )

    def to_evidence(self) -> dict[str, object]:
        return {
            "current_pose": {
                "x_m": self.current_pose.x_m,
                "y_m": self.current_pose.y_m,
                "yaw_rad": self.current_pose.yaw_rad,
            },
            "map_from_odom": {
                "x_m": self.map_from_odom.x_m,
                "y_m": self.map_from_odom.y_m,
                "yaw_rad": self.map_from_odom.yaw_rad,
            },
            "map_frame": self.map_frame,
            "odom_frame": self.odom_frame,
        }


@dataclass(frozen=True)
class CandidateSnapshotFrameProjection:
    """A derived execution snapshot plus complete reprojection evidence."""

    source_snapshot_sha256: str
    source_registry_sha256: str
    planning_frame: CandidatePlanningFrame
    projected_snapshot: CandidateSnapshot
    candidate_results: tuple[
        tuple[str, CandidateFrameReprojectionResult], ...
    ]
    schema_version: int = CANDIDATE_FRAME_PROJECTION_SCHEMA_VERSION
    motion_authorized: bool = False

    def result_for(
        self, candidate_uid: str
    ) -> CandidateFrameReprojectionResult | None:
        return next(
            (
                result
                for uid, result in self.candidate_results
                if uid == candidate_uid
            ),
            None,
        )

    def to_evidence(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_candidate_snapshot_sha256": self.source_snapshot_sha256,
            "source_registry_sha256": self.source_registry_sha256,
            "projected_candidate_snapshot_sha256": candidate_snapshot_sha256(
                self.projected_snapshot
            ),
            "planning_frame_admission": self.planning_frame.to_evidence(),
            "candidate_reprojections": {
                uid: result.to_mapping()
                for uid, result in self.candidate_results
            },
            "motion_authorized": False,
        }


def project_candidate_snapshot_to_planning_frame(
    snapshot: CandidateSnapshot,
    registry: StandSurveyRegistry,
    planning_frame: CandidatePlanningFrame,
) -> CandidateSnapshotFrameProjection:
    """Re-express every snapshot candidate using canonical odom geometry."""

    validate_candidate_snapshot(snapshot)
    validate_stand_survey_registry(registry)
    if not isinstance(planning_frame, CandidatePlanningFrame):
        raise CandidateFrameProjectionError(
            "invalid_planning_frame",
            "planning_frame must be a CandidatePlanningFrame",
        )
    registry_by_uid = {
        candidate.candidate_uid: candidate for candidate in registry.candidates
    }
    missing = sorted(set(snapshot.candidate_uids).difference(registry_by_uid))
    if missing:
        raise CandidateFrameProjectionError(
            "registry_candidate_missing",
            "candidate registry is missing snapshot candidates: "
            + ", ".join(missing),
        )
    if snapshot.planning_frame.strip("/") != registry.planning_frame.strip("/"):
        raise CandidateFrameProjectionError(
            "frame_mismatch", "snapshot and registry planning frames differ"
        )
    if snapshot.planning_frame.strip("/") != planning_frame.map_frame:
        raise CandidateFrameProjectionError(
            "frame_mismatch",
            "stationary planning-frame admission uses another map frame",
        )
    if snapshot.map_bundle_sha256 != registry.map_bundle_sha256:
        raise CandidateFrameProjectionError(
            "map_mismatch", "snapshot and registry map bundles differ"
        )

    registry_sha256 = stand_survey_registry_sha256(registry)
    provenance_mismatches = tuple(
        candidate.candidate_uid
        for candidate in snapshot.candidates
        if candidate.source.source_artifact_sha256 != registry_sha256
    )
    if provenance_mismatches:
        raise CandidateFrameProjectionError(
            "source_registry_mismatch",
            "candidate snapshot is not bound to the registry used for "
            "frame reprojection: " + ", ".join(provenance_mismatches),
        )

    projected_candidates = []
    results: list[tuple[str, CandidateFrameReprojectionResult]] = []
    for frozen_candidate in snapshot.candidates:
        registry_candidate = registry_by_uid[frozen_candidate.candidate_uid]
        provenance = registry_candidate.frame_provenance
        if provenance is None:
            raise CandidateFrameProjectionError(
                "frame_provenance_missing",
                "camera execution requires frozen frame provenance for "
                f"{frozen_candidate.candidate_uid}",
            )
        if provenance.map_frame.strip("/") != snapshot.planning_frame.strip("/"):
            raise CandidateFrameProjectionError(
                "frame_mismatch",
                "candidate observation map frame differs from planning frame",
            )
        if provenance.odom_frame.strip("/") != planning_frame.odom_frame:
            raise CandidateFrameProjectionError(
                "frame_mismatch",
                "candidate observation odom frame differs from current frame",
            )
        result = reproject_candidate_point(
            provenance,
            planning_frame.map_from_odom,
        )
        projected_candidates.append(
            replace(
                frozen_candidate,
                geometry=replace(
                    frozen_candidate.geometry,
                    x_m=result.current_map_point.x_m,
                    y_m=result.current_map_point.y_m,
                ),
            )
        )
        results.append((frozen_candidate.candidate_uid, result))

    projected_snapshot = replace(
        snapshot,
        candidates=tuple(projected_candidates),
    )
    validate_candidate_snapshot(projected_snapshot)
    return CandidateSnapshotFrameProjection(
        source_snapshot_sha256=candidate_snapshot_sha256(snapshot),
        source_registry_sha256=registry_sha256,
        planning_frame=planning_frame,
        projected_snapshot=projected_snapshot,
        candidate_results=tuple(results),
    )


def projection_candidate_points(
    projection: CandidateSnapshotFrameProjection,
) -> Mapping[str, tuple[float, float]]:
    """Return a detached UID-to-current-map point mapping for diagnostics."""

    return {
        uid: (result.current_map_point.x_m, result.current_map_point.y_m)
        for uid, result in projection.candidate_results
    }


def _frame_id(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise CandidateFrameProjectionError(
            "invalid_planning_frame", f"{name} must be a frame identifier"
        )
    normalized = value.strip("/")
    if not normalized or normalized != value or any(
        character.isspace() for character in normalized
    ):
        raise CandidateFrameProjectionError(
            "invalid_planning_frame",
            f"{name} must be a non-prefixed frame identifier",
        )
    return normalized


__all__ = [
    "CANDIDATE_FRAME_PROJECTION_SCHEMA_VERSION",
    "CandidateFrameProjectionError",
    "CandidatePlanningFrame",
    "CandidateSnapshotFrameProjection",
    "project_candidate_snapshot_to_planning_frame",
    "projection_candidate_points",
]
