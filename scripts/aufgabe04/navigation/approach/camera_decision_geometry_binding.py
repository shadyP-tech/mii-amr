"""Validate camera-decision geometry against authenticated frame evidence.

The exact-two handoff snapshot remains the immutable source/registry contract.
Camera observation may happen after ``map <- odom`` changes, so its geometry is
bound separately to a derived snapshot plus the projection artifact that
created it.  This module is ROS-free, writes nothing, and never authorizes
motion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import re
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    payload_sha256,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CANDIDATE_FRAME_PROJECTION_SCHEMA_VERSION,
    CandidatePlanningFrame,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameReprojectionResult,
    reproject_candidate_point,
)
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    REAL_VIEWPOINT_SOURCE,
    load_recommendation,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    StandSurveyRegistry,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
    normalize_yaw,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    FrozenCandidate,
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)


CAMERA_DECISION_PROJECTED_RECEIPT_SCHEMA_VERSION = 3
CANDIDATE_FRAME_PROJECTION_HASH_FIELD = (
    "candidate_frame_projection_sha256"
)
CAMERA_FRAME_BINDING_RECEIPT_FIELDS = frozenset(
    {
        "camera_candidate_snapshot_path",
        "camera_candidate_snapshot_sha256",
        "candidate_frame_projection_path",
        "candidate_frame_projection_sha256",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GEOMETRY_ABS_TOL_M = 1.0e-6
_PROJECTION_FIELDS = frozenset(
    {
        "schema_version",
        "source_candidate_snapshot_sha256",
        "source_registry_sha256",
        "projected_candidate_snapshot_sha256",
        "planning_frame_admission",
        "candidate_reprojections",
        "motion_authorized",
        "source_candidate_snapshot_path",
        "projected_candidate_snapshot_path",
    }
)
_PLANNING_FRAME_FIELDS = frozenset(
    {"current_pose", "map_from_odom", "map_frame", "odom_frame"}
)
_POSE_FIELDS = frozenset({"x_m", "y_m", "yaw_rad"})


@dataclass(frozen=True)
class CameraCandidateFrameBinding:
    """Paths and content hashes for the geometry used by one observation."""

    camera_snapshot_path: Path
    camera_snapshot_sha256: str
    projection_path: Path
    projection_sha256: str

    def __post_init__(self) -> None:
        for name in ("camera_snapshot_path", "projection_path"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise TypeError(f"{name} must be a Path")
        _require_sha256(
            self.camera_snapshot_sha256,
            "camera_snapshot_sha256",
        )
        _require_sha256(self.projection_sha256, "projection_sha256")

    def to_receipt_fields(self) -> dict[str, str]:
        return {
            "camera_candidate_snapshot_path": str(
                self.camera_snapshot_path
            ),
            "camera_candidate_snapshot_sha256": (
                self.camera_snapshot_sha256
            ),
            "candidate_frame_projection_path": str(self.projection_path),
            "candidate_frame_projection_sha256": self.projection_sha256,
        }


def require_projected_camera_candidate_binding(
    receipt: Mapping[str, object],
    *,
    canonical_snapshot_path: Path,
    canonical_snapshot: CandidateSnapshot,
    registry: StandSurveyRegistry,
    source_registry_sha256: str,
    camera_snapshot_path: Path,
    projection_path: Path,
    candidate_uid: str,
) -> FrozenCandidate:
    """Return the authenticated projected candidate named by a v3 receipt."""

    binding = _binding_from_receipt(receipt)
    if not _paths_match(binding.camera_snapshot_path, camera_snapshot_path):
        raise ValueError("receipt camera candidate snapshot path mismatch")
    if not _paths_match(binding.projection_path, projection_path):
        raise ValueError("receipt candidate frame projection path mismatch")

    projection = load_content_hashed_json(
        projection_path,
        hash_field=CANDIDATE_FRAME_PROJECTION_HASH_FIELD,
    )
    _strict_mapping(projection, _PROJECTION_FIELDS, "frame projection")
    if payload_sha256(projection) != binding.projection_sha256:
        raise ValueError("candidate frame projection SHA-256 mismatch")
    if (
        projection["schema_version"]
        != CANDIDATE_FRAME_PROJECTION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported candidate frame projection schema")
    if projection["motion_authorized"] is not False:
        raise ValueError("candidate frame projection must not authorize motion")
    if not _paths_match(
        projection["source_candidate_snapshot_path"],
        canonical_snapshot_path,
    ):
        raise ValueError("frame projection source snapshot path mismatch")
    if not _paths_match(
        projection["projected_candidate_snapshot_path"],
        camera_snapshot_path,
    ):
        raise ValueError("frame projection projected snapshot path mismatch")

    canonical_sha256 = candidate_snapshot_sha256(canonical_snapshot)
    if projection["source_candidate_snapshot_sha256"] != canonical_sha256:
        raise ValueError("frame projection source snapshot SHA-256 mismatch")
    sealed_registry_sha256 = _require_sha256(
        source_registry_sha256,
        "source_registry_sha256",
    )
    if projection["source_registry_sha256"] != sealed_registry_sha256:
        raise ValueError("frame projection source registry SHA-256 mismatch")

    camera_snapshot = load_candidate_snapshot(
        camera_snapshot_path,
        required_map_bundle_sha256=canonical_snapshot.map_bundle_sha256,
    )
    camera_sha256 = candidate_snapshot_sha256(camera_snapshot)
    if camera_sha256 != binding.camera_snapshot_sha256:
        raise ValueError("camera candidate snapshot SHA-256 mismatch")
    if projection["projected_candidate_snapshot_sha256"] != camera_sha256:
        raise ValueError("frame projection projected snapshot SHA-256 mismatch")

    planning_frame = _planning_frame_from_mapping(
        projection["planning_frame_admission"]
    )
    expected_map_frame = canonical_snapshot.planning_frame.strip("/")
    if planning_frame.map_frame != expected_map_frame:
        raise ValueError("frame projection planning frame mismatch")

    reprojections = projection["candidate_reprojections"]
    if not isinstance(reprojections, Mapping):
        raise ValueError("candidate_reprojections must be an object")
    expected_uids = set(canonical_snapshot.candidate_uids)
    if set(reprojections) != expected_uids:
        raise ValueError("frame projection candidate population mismatch")

    registry_by_uid = {
        candidate.candidate_uid: candidate
        for candidate in registry.candidates
    }
    if set(registry_by_uid) != expected_uids:
        raise ValueError("frame projection registry population mismatch")

    expected_candidates: list[FrozenCandidate] = []
    for canonical_candidate in canonical_snapshot.candidates:
        uid = canonical_candidate.candidate_uid
        raw_result = reprojections[uid]
        if not isinstance(raw_result, Mapping):
            raise ValueError(f"frame projection result for {uid!r} is invalid")
        result = CandidateFrameReprojectionResult.from_mapping(raw_result)
        live = registry_by_uid[uid]
        if live.frame_provenance is None:
            raise ValueError(
                f"frame projection registry provenance missing for {uid!r}"
            )
        if result.provenance != live.frame_provenance:
            raise ValueError(
                f"frame projection provenance differs from registry for {uid!r}"
            )
        expected_result = reproject_candidate_point(
            live.frame_provenance,
            planning_frame.map_from_odom,
        )
        if result != expected_result:
            raise ValueError(
                f"frame projection result differs from registry for {uid!r}"
            )
        expected_candidates.append(
            replace(
                canonical_candidate,
                geometry=replace(
                    canonical_candidate.geometry,
                    x_m=result.current_map_point.x_m,
                    y_m=result.current_map_point.y_m,
                ),
            )
        )

    expected_snapshot = replace(
        canonical_snapshot,
        candidates=tuple(expected_candidates),
    )
    if camera_snapshot != expected_snapshot:
        raise ValueError(
            "camera candidate snapshot differs from authenticated projection"
        )
    candidate = camera_snapshot.candidate_for(candidate_uid)
    if candidate is None:
        raise ValueError(
            "camera candidate snapshot does not contain receipt candidate"
        )
    return candidate


def require_camera_recommendation_binding(
    receipt: Mapping[str, object],
    *,
    candidate: FrozenCandidate,
    planning_frame: str,
) -> None:
    """Bind real camera evidence to one canonical or projected candidate."""

    recommendation_path = Path(str(receipt["camera_evidence_path"]))
    if _file_sha256(recommendation_path) != receipt[
        "camera_recommendation_sha256"
    ]:
        raise ValueError("camera recommendation SHA-256 mismatch")
    recommendation = load_recommendation(
        recommendation_path,
        expected_frame=planning_frame,
        expected_source=REAL_VIEWPOINT_SOURCE,
        expected_simulation_only=False,
    )
    if recommendation.stand_id != candidate.candidate_uid:
        raise ValueError(
            "camera recommendation candidate UID mismatch: "
            f"expected {candidate.candidate_uid!r}, "
            f"got {recommendation.stand_id!r}"
        )
    expected = candidate.geometry
    observed = recommendation.stand
    geometry_pairs = (
        (observed.center.x_m, expected.x_m),
        (observed.center.y_m, expected.y_m),
        (observed.radius_m, expected.radius_m),
        (observed.uncertainty_m, expected.uncertainty_m),
    )
    if any(
        not math.isclose(
            first,
            second,
            rel_tol=0.0,
            abs_tol=_GEOMETRY_ABS_TOL_M,
        )
        for first, second in geometry_pairs
    ):
        raise ValueError(
            "camera recommendation geometry differs from bound candidate "
            "snapshot"
        )


def _binding_from_receipt(
    receipt: Mapping[str, object],
) -> CameraCandidateFrameBinding:
    missing = CAMERA_FRAME_BINDING_RECEIPT_FIELDS.difference(receipt)
    if missing:
        raise ValueError(
            "candidate decision receipt is missing camera frame binding: "
            + ", ".join(sorted(missing))
        )
    for name in (
        "camera_candidate_snapshot_path",
        "candidate_frame_projection_path",
    ):
        value = receipt[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"candidate decision receipt requires {name}")
    return CameraCandidateFrameBinding(
        camera_snapshot_path=Path(
            str(receipt["camera_candidate_snapshot_path"])
        ),
        camera_snapshot_sha256=_require_sha256(
            receipt["camera_candidate_snapshot_sha256"],
            "camera_candidate_snapshot_sha256",
        ),
        projection_path=Path(str(receipt["candidate_frame_projection_path"])),
        projection_sha256=_require_sha256(
            receipt["candidate_frame_projection_sha256"],
            "candidate_frame_projection_sha256",
        ),
    )


def _planning_frame_from_mapping(value: object) -> CandidatePlanningFrame:
    payload = _strict_mapping(
        value,
        _PLANNING_FRAME_FIELDS,
        "planning_frame_admission",
    )
    pose = _pose_values(payload["current_pose"], "current_pose")
    transform = _pose_values(payload["map_from_odom"], "map_from_odom")
    raw_yaw = transform[2]
    if raw_yaw != normalize_yaw(raw_yaw):
        raise ValueError("map_from_odom.yaw_rad must be normalized")
    return CandidatePlanningFrame(
        current_pose=Pose2D(*pose),
        map_from_odom=PlanarTransform2D(*transform),
        map_frame=_frame_id(payload["map_frame"], "map_frame"),
        odom_frame=_frame_id(payload["odom_frame"], "odom_frame"),
    )


def _pose_values(value: object, name: str) -> tuple[float, float, float]:
    payload = _strict_mapping(value, _POSE_FIELDS, name)
    return (
        _finite(payload["x_m"], f"{name}.x_m"),
        _finite(payload["y_m"], f"{name}.y_m"),
        _finite(payload["yaw_rad"], f"{name}.yaw_rad"),
    )


def _strict_mapping(
    value: object,
    expected_fields: frozenset[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"{name} fields mismatch")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _frame_id(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a frame identifier")
    normalized = value.strip("/")
    if not normalized or normalized != value or any(
        character.isspace() for character in normalized
    ):
        raise ValueError(f"{name} must be a non-prefixed frame identifier")
    return normalized


def _paths_match(first: object, second: Path) -> bool:
    if isinstance(first, Path):
        first_path = first
    elif isinstance(first, str) and first.strip():
        first_path = Path(first)
    else:
        return False
    return first_path.resolve() == Path(second).resolve()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CAMERA_DECISION_PROJECTED_RECEIPT_SCHEMA_VERSION",
    "CAMERA_FRAME_BINDING_RECEIPT_FIELDS",
    "CANDIDATE_FRAME_PROJECTION_HASH_FIELD",
    "CameraCandidateFrameBinding",
    "require_camera_recommendation_binding",
    "require_projected_camera_candidate_binding",
]
