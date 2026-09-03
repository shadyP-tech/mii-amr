"""Strict, motion-neutral frame projection of a backside-axis receipt.

The passive observer receipt remains immutable.  This derived artifact binds
that receipt to the content-hashed candidate-frame projections used at camera
capture and at route planning, then re-expresses the observed stand, robot,
and axial heading in the latter frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Mapping, Union

from scripts.aufgabe04.artifacts.backside_axis_observation import (
    BacksideAxisObservation,
    load_backside_axis_observation,
)
from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CANDIDATE_FRAME_PROJECTION_SCHEMA_VERSION,
    CandidatePlanningFrame,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameReprojectionResult,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
    normalize_yaw,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)


BACKSIDE_AXIS_FRAME_PROJECTION_SCHEMA_VERSION = 1
BACKSIDE_AXIS_FRAME_PROJECTION_KIND = (
    "real_stand_backside_axis_frame_projection"
)
BACKSIDE_AXIS_FRAME_PROJECTION_HASH_FIELD = (
    "backside_axis_frame_projection_sha256"
)
CANDIDATE_FRAME_PROJECTION_HASH_FIELD = "candidate_frame_projection_sha256"

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "projection_kind",
        "motion_capability",
        "stand_id",
        "planning_frame",
        "stand_center",
        "robot_pose",
        "stand_axis_rad",
        "source_axis_observation",
        "source_candidate_frame_projection",
        "target_candidate_frame_projection",
        "frame_transform_evidence",
    }
)
_REFERENCE_FIELDS = frozenset({"path", "sha256"})
_POINT_FIELDS = frozenset({"x_m", "y_m"})
_POSE_FIELDS = frozenset({"x_m", "y_m", "yaw_rad"})
_TRANSFORM_FIELDS = _POSE_FIELDS
_TRANSFORM_EVIDENCE_FIELDS = frozenset(
    {
        "map_frame",
        "odom_frame",
        "source_map_from_odom",
        "target_map_from_odom",
        "yaw_delta_rad",
    }
)
_CANDIDATE_PROJECTION_FIELDS = frozenset(
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


@dataclass(frozen=True)
class _ProjectionBinding:
    path: Path
    sha256: str
    planning_frame: CandidatePlanningFrame
    result: CandidateFrameReprojectionResult
    source_snapshot_path: Path
    source_snapshot: CandidateSnapshot
    projected_snapshot_path: Path
    projected_snapshot: CandidateSnapshot
    canonical_odom_x_m: float
    canonical_odom_y_m: float
    current_map_x_m: float
    current_map_y_m: float
    source_snapshot_sha256: str
    source_registry_sha256: str
    map_bundle_sha256: str

    @property
    def map_frame(self) -> str:
        return self.planning_frame.map_frame

    @property
    def odom_frame(self) -> str:
        return self.planning_frame.odom_frame

    @property
    def map_from_odom(self) -> PlanarTransform2D:
        return self.planning_frame.map_from_odom


@dataclass(frozen=True)
class BacksideAxisFrameProjection:
    """Validated planning geometry plus its immutable observer provenance."""

    stand_id: str
    planning_frame: str
    stand_axis_rad: float
    stand_x_m: float
    stand_y_m: float
    robot_x_m: float
    robot_y_m: float
    robot_yaw_rad: float
    source_observation: BacksideAxisObservation
    source_axis_observation_path: Path
    source_axis_observation_sha256: str
    source_candidate_projection_path: Path
    source_candidate_projection_sha256: str
    target_candidate_projection_path: Path
    target_candidate_projection_sha256: str
    projection_sha256: str

    @property
    def visible_face_confidence(self) -> float:
        return self.source_observation.visible_face_confidence

    @property
    def axis_confidence(self) -> float:
        return self.source_observation.axis_confidence

    @property
    def axis_sample_count(self) -> int:
        return self.source_observation.axis_sample_count

    @property
    def stand_model_profile_sha256(self) -> str:
        return self.source_observation.stand_model_profile_sha256

    @property
    def opposite_face_normal_rad(self) -> float:
        projected = BacksideAxisObservation(
            stand_id=self.stand_id,
            planning_frame=self.planning_frame,
            stand_axis_rad=self.stand_axis_rad,
            stand_x_m=self.stand_x_m,
            stand_y_m=self.stand_y_m,
            robot_x_m=self.robot_x_m,
            robot_y_m=self.robot_y_m,
            visible_face_confidence=self.visible_face_confidence,
            axis_confidence=self.axis_confidence,
            axis_sample_count=self.axis_sample_count,
            stand_model_profile_sha256=self.stand_model_profile_sha256,
        )
        return projected.opposite_face_normal_rad


BacksideAxisPlanningObservation = Union[
    BacksideAxisObservation, BacksideAxisFrameProjection
]


def write_backside_axis_frame_projection(
    output_path: Path,
    *,
    axis_evidence_path: Path,
    target_candidate_projection_path: Path,
    target_candidate_projection_sha256: str,
    target_candidate_x_m: float,
    target_candidate_y_m: float,
    source_candidate_projection_path: Path | None = None,
    source_candidate_projection_sha256: str | None = None,
) -> str:
    """Project one immutable observer receipt into a fresh planning frame.

    ``axis_evidence_path`` may be the original observer receipt or an earlier
    derived projection.  Reprojection always restarts from the original
    receipt and its capture-frame proof, avoiding chained numeric drift.
    """

    source = load_backside_axis_planning_observation(axis_evidence_path)
    if isinstance(source, BacksideAxisFrameProjection):
        source_axis_path = source.source_axis_observation_path
        source_projection_path = source.source_candidate_projection_path
        source_projection_sha256 = source.source_candidate_projection_sha256
    else:
        source_axis_path = _absolute_existing_file(axis_evidence_path)
        if (
            source_candidate_projection_path is None
            or source_candidate_projection_sha256 is None
        ):
            raise ValueError(
                "native backside receipt projection requires its capture-frame "
                "candidate projection path and SHA-256"
            )
        source_projection_path = source_candidate_projection_path
        source_projection_sha256 = source_candidate_projection_sha256

    source_observation = _load_native_source(source_axis_path)
    source_axis_sha256 = _file_sha256(source_axis_path)
    source_binding = _load_projection_binding(
        source_projection_path,
        expected_sha256=source_projection_sha256,
        candidate_uid=source_observation.stand_id,
    )
    target_binding = _load_projection_binding(
        target_candidate_projection_path,
        expected_sha256=target_candidate_projection_sha256,
        candidate_uid=source_observation.stand_id,
    )
    _validate_projection_pair(source_binding, target_binding)
    if source_observation.planning_frame != source_binding.map_frame:
        raise ValueError(
            "source axis observation planning frame differs from capture proof"
        )
    if math.hypot(
        source_observation.stand_x_m - source_binding.current_map_x_m,
        source_observation.stand_y_m - source_binding.current_map_y_m,
    ) > 1.0e-6:
        raise ValueError(
            "source axis observation center differs from capture projection"
        )

    projected_stand = _reproject_point(
        source_observation.stand_x_m,
        source_observation.stand_y_m,
        source_binding.map_from_odom,
        target_binding.map_from_odom,
    )
    if math.hypot(
        projected_stand[0] - target_binding.current_map_x_m,
        projected_stand[1] - target_binding.current_map_y_m,
    ) > 1.0e-6:
        raise ValueError(
            "projected axis observation center differs from target projection"
        )
    target_x_m = _finite(target_candidate_x_m, "target_candidate_x_m")
    target_y_m = _finite(target_candidate_y_m, "target_candidate_y_m")
    if math.hypot(
        target_x_m - target_binding.current_map_x_m,
        target_y_m - target_binding.current_map_y_m,
    ) > 1.0e-6:
        raise ValueError(
            "target snapshot center differs from target projection evidence"
        )

    projected_robot = _reproject_point(
        source_observation.robot_x_m,
        source_observation.robot_y_m,
        source_binding.map_from_odom,
        target_binding.map_from_odom,
    )
    yaw_delta = _normalize_angle(
        target_binding.map_from_odom.yaw_rad
        - source_binding.map_from_odom.yaw_rad
    )
    # The observer contract validates the original robot yaw even though its
    # typed subset does not expose it.  Recover it from the immutable source.
    source_payload = _load_json_mapping(source_axis_path)
    source_robot_pose = _strict_mapping(
        source_payload.get("robot_pose"), _POSE_FIELDS, "source robot_pose"
    )
    source_robot_yaw = _finite(
        source_robot_pose["yaw_rad"], "source robot_pose.yaw_rad"
    )
    payload = {
        "schema_version": BACKSIDE_AXIS_FRAME_PROJECTION_SCHEMA_VERSION,
        "projection_kind": BACKSIDE_AXIS_FRAME_PROJECTION_KIND,
        "motion_capability": "none",
        "stand_id": source_observation.stand_id,
        "planning_frame": target_binding.map_frame,
        "stand_center": {"x_m": target_x_m, "y_m": target_y_m},
        "robot_pose": {
            "x_m": projected_robot[0],
            "y_m": projected_robot[1],
            "yaw_rad": _normalize_angle(source_robot_yaw + yaw_delta),
        },
        "stand_axis_rad": _normalize_angle(
            source_observation.stand_axis_rad + yaw_delta
        ),
        "source_axis_observation": {
            "path": str(source_axis_path),
            "sha256": source_axis_sha256,
        },
        "source_candidate_frame_projection": {
            "path": str(source_binding.path),
            "sha256": source_binding.sha256,
        },
        "target_candidate_frame_projection": {
            "path": str(target_binding.path),
            "sha256": target_binding.sha256,
        },
        "frame_transform_evidence": {
            "map_frame": target_binding.map_frame,
            "odom_frame": target_binding.odom_frame,
            "source_map_from_odom": _transform_mapping(
                source_binding.map_from_odom
            ),
            "target_map_from_odom": _transform_mapping(
                target_binding.map_from_odom
            ),
            "yaw_delta_rad": yaw_delta,
        },
    }
    digest = write_content_hashed_json(
        Path(output_path),
        payload,
        hash_field=BACKSIDE_AXIS_FRAME_PROJECTION_HASH_FIELD,
    )
    load_backside_axis_frame_projection(output_path)
    return digest


def load_backside_axis_frame_projection(
    path: Path,
) -> BacksideAxisFrameProjection:
    """Recursively authenticate and validate one derived frame artifact."""

    payload = load_content_hashed_json(
        Path(path), hash_field=BACKSIDE_AXIS_FRAME_PROJECTION_HASH_FIELD
    )
    projection_sha256 = payload_sha256(payload)
    payload = _strict_mapping(payload, _TOP_LEVEL_FIELDS, "axis projection")
    schema_version = payload["schema_version"]
    if (
        type(schema_version) is not int
        or schema_version != BACKSIDE_AXIS_FRAME_PROJECTION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported backside-axis frame projection schema")
    if payload["projection_kind"] != BACKSIDE_AXIS_FRAME_PROJECTION_KIND:
        raise ValueError("unexpected backside-axis frame projection kind")
    if payload["motion_capability"] != "none":
        raise ValueError("backside-axis frame projection must not authorize motion")

    stand_id = _nonempty_string(payload["stand_id"], "stand_id")
    planning_frame = _frame_id(payload["planning_frame"], "planning_frame")
    source_ref = _reference(
        payload["source_axis_observation"], "source_axis_observation"
    )
    source_observation = _load_native_source(source_ref[0])
    if _file_sha256(source_ref[0]) != source_ref[1]:
        raise ValueError("source axis observation SHA-256 mismatch")
    if source_observation.stand_id != stand_id:
        raise ValueError("projected stand ID differs from source observation")

    source_projection_ref = _reference(
        payload["source_candidate_frame_projection"],
        "source_candidate_frame_projection",
    )
    target_projection_ref = _reference(
        payload["target_candidate_frame_projection"],
        "target_candidate_frame_projection",
    )
    source_binding = _load_projection_binding(
        source_projection_ref[0],
        expected_sha256=source_projection_ref[1],
        candidate_uid=stand_id,
    )
    target_binding = _load_projection_binding(
        target_projection_ref[0],
        expected_sha256=target_projection_ref[1],
        candidate_uid=stand_id,
    )
    _validate_projection_pair(source_binding, target_binding)
    if source_observation.planning_frame != source_binding.map_frame:
        raise ValueError("source observation frame differs from capture proof")
    if planning_frame != target_binding.map_frame:
        raise ValueError("projected observation frame differs from target proof")

    stand = _strict_mapping(payload["stand_center"], _POINT_FIELDS, "stand_center")
    robot = _strict_mapping(payload["robot_pose"], _POSE_FIELDS, "robot_pose")
    stand_x_m = _finite(stand["x_m"], "stand_center.x_m")
    stand_y_m = _finite(stand["y_m"], "stand_center.y_m")
    robot_x_m = _finite(robot["x_m"], "robot_pose.x_m")
    robot_y_m = _finite(robot["y_m"], "robot_pose.y_m")
    robot_yaw_rad = _finite(robot["yaw_rad"], "robot_pose.yaw_rad")
    stand_axis_rad = _finite(payload["stand_axis_rad"], "stand_axis_rad")

    transforms = _strict_mapping(
        payload["frame_transform_evidence"],
        _TRANSFORM_EVIDENCE_FIELDS,
        "frame_transform_evidence",
    )
    if (
        transforms["map_frame"] != target_binding.map_frame
        or transforms["odom_frame"] != target_binding.odom_frame
    ):
        raise ValueError("frame-transform identifiers differ from projection proof")
    recorded_source_transform = _transform(
        transforms["source_map_from_odom"], "source_map_from_odom"
    )
    recorded_target_transform = _transform(
        transforms["target_map_from_odom"], "target_map_from_odom"
    )
    if (
        recorded_source_transform != source_binding.map_from_odom
        or recorded_target_transform != target_binding.map_from_odom
    ):
        raise ValueError("recorded frame transforms differ from projection proof")
    yaw_delta = _normalize_angle(
        target_binding.map_from_odom.yaw_rad
        - source_binding.map_from_odom.yaw_rad
    )
    if not _angle_close(
        _finite(transforms["yaw_delta_rad"], "yaw_delta_rad"), yaw_delta
    ):
        raise ValueError("recorded yaw delta differs from projection proof")

    expected_stand = _reproject_point(
        source_observation.stand_x_m,
        source_observation.stand_y_m,
        source_binding.map_from_odom,
        target_binding.map_from_odom,
    )
    source_payload = _load_json_mapping(source_ref[0])
    source_robot = _strict_mapping(
        source_payload.get("robot_pose"), _POSE_FIELDS, "source robot_pose"
    )
    expected_robot = _reproject_point(
        source_observation.robot_x_m,
        source_observation.robot_y_m,
        source_binding.map_from_odom,
        target_binding.map_from_odom,
    )
    if math.hypot(
        source_observation.stand_x_m - source_binding.current_map_x_m,
        source_observation.stand_y_m - source_binding.current_map_y_m,
    ) > 1.0e-6:
        raise ValueError("source observation center differs from capture proof")
    if math.hypot(
        stand_x_m - target_binding.current_map_x_m,
        stand_y_m - target_binding.current_map_y_m,
    ) > 1.0e-6 or math.hypot(
        stand_x_m - expected_stand[0], stand_y_m - expected_stand[1]
    ) > 1.0e-6:
        raise ValueError("projected stand center is inconsistent")
    if math.hypot(
        robot_x_m - expected_robot[0], robot_y_m - expected_robot[1]
    ) > 1.0e-9:
        raise ValueError("projected observing robot position is inconsistent")
    if not _angle_close(
        robot_yaw_rad,
        _normalize_angle(
            _finite(source_robot["yaw_rad"], "source robot_pose.yaw_rad")
            + yaw_delta
        ),
    ):
        raise ValueError("projected observing robot yaw is inconsistent")
    if not _angle_close(
        stand_axis_rad,
        _normalize_angle(source_observation.stand_axis_rad + yaw_delta),
    ):
        raise ValueError("projected stand axis is inconsistent")

    result = BacksideAxisFrameProjection(
        stand_id=stand_id,
        planning_frame=planning_frame,
        stand_axis_rad=stand_axis_rad,
        stand_x_m=stand_x_m,
        stand_y_m=stand_y_m,
        robot_x_m=robot_x_m,
        robot_y_m=robot_y_m,
        robot_yaw_rad=robot_yaw_rad,
        source_observation=source_observation,
        source_axis_observation_path=source_ref[0],
        source_axis_observation_sha256=source_ref[1],
        source_candidate_projection_path=source_binding.path,
        source_candidate_projection_sha256=source_binding.sha256,
        target_candidate_projection_path=target_binding.path,
        target_candidate_projection_sha256=target_binding.sha256,
        projection_sha256=projection_sha256,
    )
    # Reusing the shared opposite-face geometry also rejects degenerate
    # projected stand/robot configurations.
    projected_normal = result.opposite_face_normal_rad
    expected_normal = _normalize_angle(
        source_observation.opposite_face_normal_rad + yaw_delta
    )
    if not _angle_close(projected_normal, expected_normal):
        raise ValueError(
            "projected opposite-face normal is not the rotated source normal"
        )
    return result


def load_backside_axis_planning_observation(
    path: Path,
) -> BacksideAxisPlanningObservation:
    """Load either a native observer receipt or the distinct derived kind."""

    payload = _load_json_mapping(path)
    if payload.get("projection_kind") == BACKSIDE_AXIS_FRAME_PROJECTION_KIND:
        return load_backside_axis_frame_projection(path)
    return load_backside_axis_observation(path)


def _load_native_source(path: Path) -> BacksideAxisObservation:
    path = _absolute_existing_file(path)
    return load_backside_axis_observation(path)


def _load_projection_binding(
    path: Path,
    *,
    expected_sha256: str,
    candidate_uid: str,
) -> _ProjectionBinding:
    path = _absolute_existing_file(path)
    expected_sha256 = _sha256(expected_sha256, "candidate projection SHA-256")
    payload = load_content_hashed_json(
        path, hash_field=CANDIDATE_FRAME_PROJECTION_HASH_FIELD
    )
    payload = _strict_mapping(
        payload,
        _CANDIDATE_PROJECTION_FIELDS,
        "candidate frame projection",
    )
    actual_sha256 = payload_sha256(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError("candidate frame projection SHA-256 mismatch")
    schema_version = payload["schema_version"]
    if (
        type(schema_version) is not int
        or schema_version != CANDIDATE_FRAME_PROJECTION_SCHEMA_VERSION
    ):
        raise ValueError("candidate frame projection contract is unsupported")
    if payload["motion_authorized"] is not False:
        raise ValueError("candidate frame projection must not authorize motion")

    source_snapshot_path = _artifact_path(
        payload["source_candidate_snapshot_path"],
        "source_candidate_snapshot_path",
    )
    source_snapshot = load_candidate_snapshot(source_snapshot_path)
    source_snapshot_sha256 = candidate_snapshot_sha256(source_snapshot)
    if payload["source_candidate_snapshot_sha256"] != source_snapshot_sha256:
        raise ValueError("candidate projection source snapshot SHA-256 mismatch")
    source_registry_sha256 = _sha256(
        payload["source_registry_sha256"], "source_registry_sha256"
    )
    if any(
        candidate.source.source_artifact_sha256 != source_registry_sha256
        for candidate in source_snapshot.candidates
    ):
        raise ValueError(
            "candidate projection source snapshot has another registry binding"
        )

    projected_snapshot_path = _artifact_path(
        payload["projected_candidate_snapshot_path"],
        "projected_candidate_snapshot_path",
    )
    projected_snapshot = load_candidate_snapshot(
        projected_snapshot_path,
        required_map_bundle_sha256=source_snapshot.map_bundle_sha256,
    )
    projected_snapshot_sha256 = candidate_snapshot_sha256(projected_snapshot)
    if (
        payload["projected_candidate_snapshot_sha256"]
        != projected_snapshot_sha256
    ):
        raise ValueError("candidate projection target snapshot SHA-256 mismatch")
    if source_snapshot.planning_frame != projected_snapshot.planning_frame:
        raise ValueError("candidate projection snapshot planning frames differ")

    planning_frame = _planning_frame_from_mapping(
        payload["planning_frame_admission"]
    )
    if planning_frame.map_frame != source_snapshot.planning_frame.strip("/"):
        raise ValueError("candidate projection planning frame differs from snapshot")

    reprojections = _strict_population_mapping(
        payload["candidate_reprojections"],
        expected_uids=set(source_snapshot.candidate_uids),
    )
    projected_candidates = []
    selected_result = None
    for source_candidate in source_snapshot.candidates:
        uid = source_candidate.candidate_uid
        raw_result = reprojections[uid]
        if not isinstance(raw_result, Mapping):
            raise ValueError(
                f"candidate frame projection result for {uid!r} is invalid"
            )
        try:
            result = CandidateFrameReprojectionResult.from_mapping(raw_result)
        except ValueError as exc:
            raise ValueError(
                f"candidate frame projection result for {uid!r} is invalid: {exc}"
            ) from exc
        if (
            result.current_map_from_odom != planning_frame.map_from_odom
            or result.provenance.map_frame != planning_frame.map_frame
            or result.provenance.odom_frame != planning_frame.odom_frame
        ):
            raise ValueError(
                f"candidate frame projection result for {uid!r} differs "
                "from planning-frame admission"
            )
        projected_candidates.append(
            replace(
                source_candidate,
                geometry=replace(
                    source_candidate.geometry,
                    x_m=result.current_map_point.x_m,
                    y_m=result.current_map_point.y_m,
                ),
            )
        )
        if uid == candidate_uid:
            selected_result = result

    expected_snapshot = replace(
        source_snapshot, candidates=tuple(projected_candidates)
    )
    if expected_snapshot != projected_snapshot:
        raise ValueError(
            "candidate projected snapshot differs from authoritative "
            "reprojection results"
        )
    if selected_result is None:
        raise ValueError("candidate projection does not contain the stand ID")

    return _ProjectionBinding(
        path=path,
        sha256=actual_sha256,
        planning_frame=planning_frame,
        result=selected_result,
        source_snapshot_path=source_snapshot_path,
        source_snapshot=source_snapshot,
        projected_snapshot_path=projected_snapshot_path,
        projected_snapshot=projected_snapshot,
        canonical_odom_x_m=selected_result.canonical_odom_point.x_m,
        canonical_odom_y_m=selected_result.canonical_odom_point.y_m,
        current_map_x_m=selected_result.current_map_point.x_m,
        current_map_y_m=selected_result.current_map_point.y_m,
        source_snapshot_sha256=source_snapshot_sha256,
        source_registry_sha256=source_registry_sha256,
        map_bundle_sha256=source_snapshot.map_bundle_sha256,
    )


def _validate_projection_pair(
    source: _ProjectionBinding, target: _ProjectionBinding
) -> None:
    if source.map_frame != target.map_frame or source.odom_frame != target.odom_frame:
        raise ValueError("source and target candidate projection frames differ")
    if (
        source.source_snapshot_path != target.source_snapshot_path
        or source.source_snapshot != target.source_snapshot
        or source.map_bundle_sha256 != target.map_bundle_sha256
        or source.source_snapshot_sha256 != target.source_snapshot_sha256
        or source.source_registry_sha256 != target.source_registry_sha256
    ):
        raise ValueError("source and target candidate projections have different lineage")
    if not _point_close(
        source.canonical_odom_x_m,
        source.canonical_odom_y_m,
        target.canonical_odom_x_m,
        target.canonical_odom_y_m,
    ):
        raise ValueError("source and target projections bind different odom geometry")


def _reproject_point(
    x_m: float,
    y_m: float,
    source: PlanarTransform2D,
    target: PlanarTransform2D,
) -> tuple[float, float]:
    delta_x = x_m - source.x_m
    delta_y = y_m - source.y_m
    cosine = math.cos(source.yaw_rad)
    sine = math.sin(source.yaw_rad)
    odom_x = cosine * delta_x + sine * delta_y
    odom_y = -sine * delta_x + cosine * delta_y
    return _odom_to_map(odom_x, odom_y, target)


def _odom_to_map(
    x_m: float, y_m: float, transform: PlanarTransform2D
) -> tuple[float, float]:
    cosine = math.cos(transform.yaw_rad)
    sine = math.sin(transform.yaw_rad)
    return (
        cosine * x_m - sine * y_m + transform.x_m,
        sine * x_m + cosine * y_m + transform.y_m,
    )


def _transform(value: object, name: str) -> PlanarTransform2D:
    payload = _strict_mapping(value, _TRANSFORM_FIELDS, name)
    yaw_rad = _finite(payload["yaw_rad"], f"{name}.yaw_rad")
    if yaw_rad != normalize_yaw(yaw_rad):
        raise ValueError(f"{name}.yaw_rad must be normalized")
    return PlanarTransform2D(
        _finite(payload["x_m"], f"{name}.x_m"),
        _finite(payload["y_m"], f"{name}.y_m"),
        yaw_rad,
    )


def _transform_mapping(value: PlanarTransform2D) -> dict[str, float]:
    return {"x_m": value.x_m, "y_m": value.y_m, "yaw_rad": value.yaw_rad}


def _planning_frame_from_mapping(value: object) -> CandidatePlanningFrame:
    payload = _strict_mapping(
        value, _PLANNING_FRAME_FIELDS, "planning_frame_admission"
    )
    pose = _pose_values(payload["current_pose"], "current_pose")
    transform = _transform(payload["map_from_odom"], "map_from_odom")
    return CandidatePlanningFrame(
        current_pose=Pose2D(*pose),
        map_from_odom=transform,
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


def _strict_population_mapping(
    value: object, *, expected_uids: set[str]
) -> Mapping[str, object]:
    payload = _mapping(value, "candidate_reprojections")
    if set(payload) != expected_uids:
        raise ValueError("candidate frame projection population mismatch")
    return payload


def _artifact_path(value: object, name: str) -> Path:
    path_text = _nonempty_string(value, name)
    return _absolute_existing_file(Path(path_text).resolve())


def _reference(value: object, name: str) -> tuple[Path, str]:
    payload = _strict_mapping(value, _REFERENCE_FIELDS, name)
    path_text = _nonempty_string(payload["path"], f"{name}.path")
    path = Path(path_text)
    if not path.is_absolute():
        raise ValueError(f"{name}.path must be absolute")
    return _absolute_existing_file(path), _sha256(
        payload["sha256"], f"{name}.sha256"
    )


def _absolute_existing_file(path: Path) -> Path:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a non-symlink file: {path}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    import json

    path = _absolute_existing_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load axis evidence: {exc}") from exc
    return _mapping(payload, "axis evidence")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _strict_mapping(
    value: object, expected_fields: frozenset[str], name: str
) -> Mapping[str, object]:
    payload = _mapping(value, name)
    if set(payload) != expected_fields:
        raise ValueError(f"{name} has unexpected fields")
    return payload


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _frame_id(value: object, name: str) -> str:
    result = _nonempty_string(value, name)
    if result.strip("/") != result or any(character.isspace() for character in result):
        raise ValueError(f"{name} must be a canonical frame identifier")
    return result


def _sha256(value: object, name: str) -> str:
    import re

    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _angle_close(first: float, second: float) -> bool:
    return abs(_normalize_angle(first - second)) <= 1.0e-9


def _point_close(
    first_x: float, first_y: float, second_x: float, second_y: float
) -> bool:
    return math.hypot(first_x - second_x, first_y - second_y) <= 1.0e-9


__all__ = [
    "BACKSIDE_AXIS_FRAME_PROJECTION_HASH_FIELD",
    "BACKSIDE_AXIS_FRAME_PROJECTION_KIND",
    "BACKSIDE_AXIS_FRAME_PROJECTION_SCHEMA_VERSION",
    "BacksideAxisFrameProjection",
    "BacksideAxisPlanningObservation",
    "load_backside_axis_frame_projection",
    "load_backside_axis_planning_observation",
    "write_backside_axis_frame_projection",
]
