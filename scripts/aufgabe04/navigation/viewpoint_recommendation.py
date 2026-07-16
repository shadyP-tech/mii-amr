"""ROS-free contract and identity state for synchronized stand viewpoints.

The simulation observer and route planner run in different processes.  This
module gives their JSON hand-off a strict, versioned shape and keeps the two
180-degree-opposed stand faces stable while unordered silhouette estimates
arrive.  It deliberately contains no ROS, camera, or QR-decoder dependencies.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Collection, Mapping, Sequence

from scripts.aufgabe04.navigation.models import Pose2D


RECOMMENDATION_SCHEMA_VERSION = 1

_FACE_GEOMETRY_TOLERANCE_RAD = 1.0e-6
_FACE_POSE_MIN_RADIUS_M = 1.0e-9

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_FRAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_/.-]{0,127}$")
_SAFE_SOURCE_RE = re.compile(r"^[A-Za-z0-9/][A-Za-z0-9_.:/+@-]{0,255}$")


@dataclass(frozen=True)
class StandGeometry:
    center: Pose2D
    radius_m: float
    uncertainty_m: float
    provenance: str


@dataclass(frozen=True)
class FaceCandidate:
    face_id: str
    outward_normal_rad: float
    pose: Pose2D
    identity_resolved: bool


@dataclass(frozen=True)
class SideEvidence:
    kind: str
    confidence: float
    hard: bool
    valid: bool
    face_id: str | None
    provenance: str


@dataclass(frozen=True)
class MaterialTarget:
    """The small subset whose material changes may trigger route replanning."""

    face_id: str
    pose: Pose2D
    evidence_state: str


@dataclass(frozen=True)
class SynchronizedViewpointRecommendation:
    schema_version: int
    simulation_only: bool
    stream_id: str
    stand_id: str
    planning_frame: str
    source: str
    observation_unix_sec: float
    sensor_stamp_sec: float
    stand: StandGeometry
    robot_pose: Pose2D
    axis_confidence: float
    axis_state: str
    face_candidates: tuple[FaceCandidate, FaceCandidate]
    side_evidence: SideEvidence
    material_target: MaterialTarget


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def angular_distance(first_rad: float, second_rad: float) -> float:
    return abs(normalize_angle(first_rad - second_rad))


def validate_recommendation(
    recommendation: SynchronizedViewpointRecommendation,
    *,
    required_planning_frame: str | None = None,
    required_source: str | None = None,
) -> None:
    """Validate structure and provenance, but intentionally not wall-clock age."""

    if recommendation.schema_version != RECOMMENDATION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported viewpoint recommendation schema_version: "
            f"{recommendation.schema_version!r}"
        )
    if recommendation.simulation_only is not True:
        raise ValueError("viewpoint recommendation must be marked simulation_only=true")
    _validate_safe_id(recommendation.stream_id, "stream_id")
    _validate_safe_id(recommendation.stand_id, "stand_id")
    _validate_frame(recommendation.planning_frame, "planning_frame")
    _validate_source(recommendation.source, "source")
    if required_planning_frame is not None:
        _validate_frame(required_planning_frame, "required_planning_frame")
        if recommendation.planning_frame != required_planning_frame:
            raise ValueError("viewpoint recommendation planning_frame mismatch")
    if required_source is not None and recommendation.source != required_source:
        raise ValueError("viewpoint recommendation source mismatch")

    _finite_nonnegative(recommendation.observation_unix_sec, "observation_unix_sec")
    _finite_nonnegative(recommendation.sensor_stamp_sec, "sensor_stamp_sec")
    _validate_pose(recommendation.stand.center, "stand.center")
    radius_m = _finite_number(recommendation.stand.radius_m, "stand.radius_m")
    if radius_m <= 0.0:
        raise ValueError("stand.radius_m must be positive")
    _finite_nonnegative(recommendation.stand.uncertainty_m, "stand.uncertainty_m")
    _validate_source(recommendation.stand.provenance, "stand.provenance")
    _validate_pose(recommendation.robot_pose, "robot_pose")

    confidence = _finite_number(recommendation.axis_confidence, "axis_confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("axis_confidence must be in [0, 1]")
    _validate_safe_id(recommendation.axis_state, "axis_state")

    faces = tuple(recommendation.face_candidates)
    if len(faces) != 2:
        raise ValueError("viewpoint recommendation must contain exactly two face candidates")
    face_ids: set[str] = set()
    normals: list[float] = []
    for index, face in enumerate(faces):
        _validate_safe_id(face.face_id, f"face_candidates[{index}].face_id")
        if face.face_id in face_ids:
            raise ValueError("face candidate IDs must be distinct")
        face_ids.add(face.face_id)
        normals.append(
            _finite_number(
                face.outward_normal_rad,
                f"face_candidates[{index}].outward_normal_rad",
            )
        )
        _validate_pose(face.pose, f"face_candidates[{index}].pose")
        if type(face.identity_resolved) is not bool:
            raise ValueError(
                f"face_candidates[{index}].identity_resolved must be boolean"
            )
    normal_separation = angular_distance(normals[0], normals[1])
    if abs(normal_separation - math.pi) > _FACE_GEOMETRY_TOLERANCE_RAD:
        raise ValueError("face candidate normals must be antipodal")
    for index, (face, normal) in enumerate(zip(faces, normals)):
        radial_x = face.pose.x_m - recommendation.stand.center.x_m
        radial_y = face.pose.y_m - recommendation.stand.center.y_m
        radial_distance = math.hypot(radial_x, radial_y)
        if radial_distance <= _FACE_POSE_MIN_RADIUS_M:
            raise ValueError(
                f"face_candidates[{index}].pose must be outside the stand center"
            )
        radial_normal = math.atan2(radial_y, radial_x)
        if angular_distance(radial_normal, normal) > _FACE_GEOMETRY_TOLERANCE_RAD:
            raise ValueError(
                f"face_candidates[{index}].pose must lie on its outward-normal ray"
            )
        expected_yaw = normalize_angle(normal + math.pi)
        if angular_distance(face.pose.yaw_rad, expected_yaw) > _FACE_GEOMETRY_TOLERANCE_RAD:
            raise ValueError(
                f"face_candidates[{index}].pose yaw must face the stand"
            )

    _validate_side_evidence(recommendation.side_evidence, face_ids)
    target = recommendation.material_target
    _validate_safe_id(target.face_id, "material_target.face_id")
    if target.face_id not in face_ids:
        raise ValueError("material_target.face_id does not reference a face candidate")
    _validate_pose(target.pose, "material_target.pose")
    _validate_safe_id(target.evidence_state, "material_target.evidence_state")

    matching_face = next(face for face in faces if face.face_id == target.face_id)
    if not _poses_close(target.pose, matching_face.pose):
        raise ValueError("material_target.pose must match its referenced face candidate")
    if recommendation.side_evidence.hard and recommendation.side_evidence.valid:
        if recommendation.side_evidence.face_id != target.face_id:
            raise ValueError(
                "hard side evidence and material_target must reference the same face"
            )
        if not matching_face.identity_resolved:
            raise ValueError("hard side evidence requires a resolved physical face identity")


def validate_recommendation_freshness(
    recommendation: SynchronizedViewpointRecommendation,
    *,
    now_unix_sec: float,
    max_age_sec: float,
    max_future_skew_sec: float = 0.0,
) -> None:
    """Apply wall-clock freshness separately from structural JSON validation."""

    now = _finite_nonnegative(now_unix_sec, "now_unix_sec")
    max_age = _finite_nonnegative(max_age_sec, "max_age_sec")
    future_skew = _finite_nonnegative(max_future_skew_sec, "max_future_skew_sec")
    age_sec = now - recommendation.observation_unix_sec
    if age_sec < -future_skew:
        raise ValueError("viewpoint recommendation timestamp is in the future")
    if age_sec > max_age:
        raise ValueError("viewpoint recommendation is stale")


def recommendation_from_payload(
    payload: Mapping[str, object],
) -> SynchronizedViewpointRecommendation:
    try:
        stand_payload = _require_mapping(payload, "stand")
        axis_payload = _require_mapping(payload, "axis")
        raw_faces = payload.get("face_candidates")
        if not isinstance(raw_faces, (list, tuple)):
            raise ValueError("face_candidates must be an array")
        faces = tuple(_face_from_payload(item, index) for index, item in enumerate(raw_faces))
        if len(faces) != 2:
            raise ValueError("face_candidates must contain exactly two entries")
        evidence_payload = _require_mapping(payload, "side_evidence")
        target_payload = _require_mapping(payload, "material_target")
        recommendation = SynchronizedViewpointRecommendation(
            schema_version=_require_int(payload, "schema_version"),
            simulation_only=_require_bool(payload, "simulation_only"),
            stream_id=_require_string(payload, "stream_id"),
            stand_id=_require_string(payload, "stand_id"),
            planning_frame=_require_string(payload, "planning_frame"),
            source=_require_string(payload, "source"),
            observation_unix_sec=_require_number(payload, "observation_unix_sec"),
            sensor_stamp_sec=_require_number(payload, "sensor_stamp_sec"),
            stand=StandGeometry(
                center=_pose_from_payload(_require_mapping(stand_payload, "center"), "stand.center"),
                radius_m=_require_number(stand_payload, "radius_m"),
                uncertainty_m=_require_number(stand_payload, "uncertainty_m"),
                provenance=_require_string(stand_payload, "provenance"),
            ),
            robot_pose=_pose_from_payload(_require_mapping(payload, "robot_pose"), "robot_pose"),
            axis_confidence=_require_number(axis_payload, "confidence"),
            axis_state=_require_string(axis_payload, "state"),
            face_candidates=(faces[0], faces[1]),
            side_evidence=SideEvidence(
                kind=_require_string(evidence_payload, "kind"),
                confidence=_require_number(evidence_payload, "confidence"),
                hard=_require_bool(evidence_payload, "hard"),
                valid=_require_bool(evidence_payload, "valid"),
                face_id=_optional_string(evidence_payload, "face_id"),
                provenance=_require_string(evidence_payload, "provenance"),
            ),
            material_target=MaterialTarget(
                face_id=_require_string(target_payload, "face_id"),
                pose=_pose_from_payload(
                    _require_mapping(target_payload, "pose"), "material_target.pose"
                ),
                evidence_state=_require_string(target_payload, "evidence_state"),
            ),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed viewpoint recommendation: {exc}") from exc
    validate_recommendation(recommendation)
    return recommendation


def recommendation_to_payload(
    recommendation: SynchronizedViewpointRecommendation,
) -> dict[str, object]:
    validate_recommendation(recommendation)
    payload = asdict(recommendation)
    payload["axis"] = {
        "confidence": payload.pop("axis_confidence"),
        "state": payload.pop("axis_state"),
    }
    return payload


def recommendation_to_dict(
    recommendation: SynchronizedViewpointRecommendation,
) -> dict[str, object]:
    """Public spelling used by producers that do not write a file themselves."""

    return recommendation_to_payload(recommendation)


def load_viewpoint_recommendation(
    source: Path | Mapping[str, object],
    *,
    required_planning_frame: str | None = None,
    required_source: str | None = None,
) -> SynchronizedViewpointRecommendation:
    if isinstance(source, Mapping):
        payload = source
    else:
        try:
            payload = json.loads(Path(source).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load viewpoint recommendation: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("viewpoint recommendation JSON root must be an object")
    recommendation = recommendation_from_payload(payload)
    validate_recommendation(
        recommendation,
        required_planning_frame=required_planning_frame,
        required_source=required_source,
    )
    return recommendation


def load_recommendation(
    source: Path | Mapping[str, object],
    *,
    expected_frame: str | None = None,
    expected_source: str | None = None,
    now_unix_sec: float | None = None,
    max_age_sec: float | None = None,
) -> SynchronizedViewpointRecommendation:
    """Load, validate, and optionally freshness-check a recommendation.

    Structural validation always runs.  Freshness remains opt-in and requires
    both ``now_unix_sec`` and ``max_age_sec`` so a caller cannot accidentally
    compare simulation sensor time with wall time.
    """

    recommendation = load_viewpoint_recommendation(
        source,
        required_planning_frame=expected_frame,
        required_source=expected_source,
    )
    if (now_unix_sec is None) != (max_age_sec is None):
        raise ValueError("now_unix_sec and max_age_sec must be provided together")
    if now_unix_sec is not None and max_age_sec is not None:
        validate_recommendation_freshness(
            recommendation,
            now_unix_sec=now_unix_sec,
            max_age_sec=max_age_sec,
        )
    return recommendation


@dataclass(frozen=True)
class StableFace:
    face_id: str
    outward_normal_rad: float
    identity_resolved: bool


@dataclass(frozen=True)
class StableFaceResolution:
    stream_id: str
    faces: tuple[StableFace, StableFace]
    identity_resolved: bool


class StableFaceResolver:
    """Assign persistent IDs to two unordered, opposing face normals."""

    def __init__(
        self,
        *,
        face_ids: tuple[str, str] = ("face_a", "face_b"),
        max_continuity_step_rad: float = math.radians(60.0),
        ambiguity_margin_rad: float = math.radians(5.0),
        min_face_separation_rad: float = math.radians(30.0),
    ) -> None:
        if len(face_ids) != 2 or face_ids[0] == face_ids[1]:
            raise ValueError("face_ids must contain two distinct IDs")
        for index, face_id in enumerate(face_ids):
            _validate_safe_id(face_id, f"face_ids[{index}]")
        self._face_ids = face_ids
        self._max_step = _finite_nonnegative(
            max_continuity_step_rad, "max_continuity_step_rad"
        )
        self._ambiguity_margin = _finite_nonnegative(
            ambiguity_margin_rad, "ambiguity_margin_rad"
        )
        self._min_separation = _finite_nonnegative(
            min_face_separation_rad, "min_face_separation_rad"
        )
        self._stream_id: str | None = None
        self._normals: tuple[float, float] | None = None
        self._resolved = False

    def reset(self, stream_id: str | None = None) -> None:
        if stream_id is not None:
            _validate_safe_id(stream_id, "stream_id")
        self._stream_id = stream_id
        self._normals = None
        self._resolved = False

    def update(
        self, *, stream_id: str, outward_normals_rad: Sequence[float]
    ) -> StableFaceResolution:
        _validate_safe_id(stream_id, "stream_id")
        if len(outward_normals_rad) != 2:
            raise ValueError("exactly two outward face normals are required")
        normals = tuple(
            normalize_angle(_finite_number(value, f"outward_normals_rad[{index}]"))
            for index, value in enumerate(outward_normals_rad)
        )
        if angular_distance(normals[0], normals[1]) < self._min_separation:
            raise ValueError("outward face normals are not sufficiently distinct")
        if self._stream_id != stream_id:
            self.reset(stream_id)

        if self._normals is None:
            ordered = tuple(sorted(normals))
            self._normals = (ordered[0], ordered[1])
            self._resolved = False
        else:
            previous = self._normals
            direct_cost = angular_distance(previous[0], normals[0]) + angular_distance(
                previous[1], normals[1]
            )
            swapped_cost = angular_distance(previous[0], normals[1]) + angular_distance(
                previous[1], normals[0]
            )
            assigned = normals if direct_cost <= swapped_cost else (normals[1], normals[0])
            winning_cost = min(direct_cost, swapped_cost)
            losing_cost = max(direct_cost, swapped_cost)
            continuous = all(
                angular_distance(old, new) <= self._max_step
                for old, new in zip(previous, assigned)
            )
            unambiguous = losing_cost - winning_cost >= self._ambiguity_margin
            self._resolved = continuous and unambiguous
            if self._resolved:
                self._normals = (assigned[0], assigned[1])
            # Do not let an unresolved frame become the identity baseline.
            # Otherwise two repeated axial outliers can first poison the
            # baseline and then appear self-consistent, swapping the stable
            # physical face IDs underneath an already latched QR result.
            # Returning the last trusted pair with ``identity_resolved=false``
            # makes the producer fail closed while allowing a subsequent
            # in-family observation to recover the original identities.

        faces = (
            StableFace(self._face_ids[0], self._normals[0], self._resolved),
            StableFace(self._face_ids[1], self._normals[1], self._resolved),
        )
        return StableFaceResolution(stream_id, faces, self._resolved)


@dataclass(frozen=True)
class QrBindingObservation:
    face_id: str
    confidence: float
    provenance: str
    registry_match: bool
    inside_target_roi: bool
    distinct_fresh_frame_consensus: bool
    visibility_margin_rad: float
    identity_resolved: bool = True
    contradiction: bool = False
    kind: str = "qr_registry"


@dataclass(frozen=True)
class QrBindingResult:
    evidence: SideEvidence
    accepted: bool
    reason: str


class QrFaceLatch:
    """Fail-closed hard QR-to-face binding with explicit stream reset semantics."""

    def __init__(
        self,
        *,
        min_visibility_margin_rad: float = math.radians(8.0),
    ) -> None:
        self._min_visibility_margin = _finite_nonnegative(
            min_visibility_margin_rad, "min_visibility_margin_rad"
        )
        self._stream_id: str | None = None
        self._latched: SideEvidence | None = None
        self._invalid_reason: str | None = None

    @property
    def latched_evidence(self) -> SideEvidence | None:
        return self._latched

    def reset(self, stream_id: str | None = None) -> None:
        if stream_id is not None:
            _validate_safe_id(stream_id, "stream_id")
        self._stream_id = stream_id
        self._latched = None
        self._invalid_reason = None

    def invalidate(self, *, stream_id: str, reason: str) -> None:
        """Poison the current stream until an explicit different-stream reset."""

        _validate_safe_id(stream_id, "stream_id")
        _validate_safe_id(reason, "reason")
        if self._stream_id != stream_id:
            self.reset(stream_id)
        # Preserve the first fault—especially a QR contradiction—so a later
        # secondary sensor fault cannot erase its audit meaning.
        if self._invalid_reason is None:
            self._invalid_reason = reason
        self._latched = None

    def update(
        self,
        *,
        stream_id: str,
        observation: QrBindingObservation | None,
        known_face_ids: Collection[str] | None = None,
    ) -> QrBindingResult:
        _validate_safe_id(stream_id, "stream_id")
        if self._stream_id != stream_id:
            self.reset(stream_id)
        if self._invalid_reason is not None:
            return QrBindingResult(
                _empty_side_evidence("contradictory_qr_evidence"),
                False,
                self._invalid_reason,
            )
        if observation is None:
            if self._latched is not None:
                return QrBindingResult(self._latched, False, "dropout_latch_retained")
            return QrBindingResult(_empty_side_evidence("no_observation"), False, "no_observation")

        invalid = self._validate_binding_observation(observation, known_face_ids)
        if invalid is not None:
            return self._retained_or_rejected(observation, invalid)
        if self._latched is not None:
            if observation.contradiction or observation.face_id != self._latched.face_id:
                self._invalid_reason = "contradicts_latch"
                return QrBindingResult(
                    _empty_side_evidence("contradictory_qr_evidence"),
                    False,
                    self._invalid_reason,
                )
            return QrBindingResult(self._latched, False, "already_latched")
        if observation.contradiction:
            return self._retained_or_rejected(observation, "contradiction")

        self._latched = SideEvidence(
            kind=observation.kind,
            confidence=observation.confidence,
            hard=True,
            valid=True,
            face_id=observation.face_id,
            provenance=observation.provenance,
        )
        return QrBindingResult(self._latched, True, "hard_binding_accepted")

    def _validate_binding_observation(
        self,
        observation: QrBindingObservation,
        known_face_ids: Collection[str] | None,
    ) -> str | None:
        _validate_safe_id(observation.face_id, "qr.face_id")
        _validate_safe_id(observation.kind, "qr.kind")
        _validate_source(observation.provenance, "qr.provenance")
        confidence = _finite_number(observation.confidence, "qr.confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("qr.confidence must be in [0, 1]")
        margin = _finite_number(observation.visibility_margin_rad, "qr.visibility_margin_rad")
        for field_name in (
            "registry_match",
            "inside_target_roi",
            "distinct_fresh_frame_consensus",
            "identity_resolved",
            "contradiction",
        ):
            if type(getattr(observation, field_name)) is not bool:
                raise ValueError(f"qr.{field_name} must be boolean")
        if known_face_ids is not None and observation.face_id not in known_face_ids:
            return "unknown_face_id"
        if not observation.registry_match:
            return "registry_mismatch"
        if not observation.inside_target_roi:
            return "outside_target_roi"
        if not observation.distinct_fresh_frame_consensus:
            return "insufficient_fresh_frame_consensus"
        if not observation.identity_resolved:
            return "face_identity_unresolved"
        if margin < self._min_visibility_margin:
            return "visibility_near_tangent"
        return None

    def _retained_or_rejected(
        self, observation: QrBindingObservation, reason: str
    ) -> QrBindingResult:
        if self._latched is not None:
            return QrBindingResult(self._latched, False, reason + "_latch_retained")
        evidence = SideEvidence(
            kind=observation.kind,
            confidence=observation.confidence,
            hard=False,
            valid=False,
            face_id=None,
            provenance=observation.provenance,
        )
        return QrBindingResult(evidence, False, reason)


def _empty_side_evidence(provenance: str) -> SideEvidence:
    return SideEvidence(
        kind="none",
        confidence=0.0,
        hard=False,
        valid=False,
        face_id=None,
        provenance=provenance,
    )


def _validate_side_evidence(evidence: SideEvidence, face_ids: set[str]) -> None:
    _validate_safe_id(evidence.kind, "side_evidence.kind")
    confidence = _finite_number(evidence.confidence, "side_evidence.confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("side_evidence.confidence must be in [0, 1]")
    if type(evidence.hard) is not bool or type(evidence.valid) is not bool:
        raise ValueError("side_evidence hard/valid fields must be boolean")
    if evidence.hard and not evidence.valid:
        raise ValueError("hard side evidence must also be valid")
    if evidence.face_id is not None:
        _validate_safe_id(evidence.face_id, "side_evidence.face_id")
        if evidence.face_id not in face_ids:
            raise ValueError("side_evidence.face_id does not reference a face candidate")
    if evidence.valid and evidence.face_id is None:
        raise ValueError("valid side evidence must reference a face candidate")
    _validate_source(evidence.provenance, "side_evidence.provenance")


def _face_from_payload(payload: object, index: int) -> FaceCandidate:
    if not isinstance(payload, Mapping):
        raise ValueError(f"face_candidates[{index}] must be an object")
    return FaceCandidate(
        face_id=_require_string(payload, "face_id"),
        outward_normal_rad=_require_number(payload, "outward_normal_rad"),
        pose=_pose_from_payload(
            _require_mapping(payload, "pose"), f"face_candidates[{index}].pose"
        ),
        identity_resolved=_require_bool(payload, "identity_resolved"),
    )


def _pose_from_payload(payload: Mapping[str, object], name: str) -> Pose2D:
    return Pose2D(
        x_m=_require_number(payload, "x_m"),
        y_m=_require_number(payload, "y_m"),
        yaw_rad=_require_number(payload, "yaw_rad"),
    )


def _require_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _require_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _require_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be boolean")
    return value


def _require_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _require_number(payload: Mapping[str, object], key: str) -> float:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    return _finite_number(payload[key], key)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_pose(pose: Pose2D, name: str) -> None:
    _finite_number(pose.x_m, f"{name}.x_m")
    _finite_number(pose.y_m, f"{name}.y_m")
    _finite_number(pose.yaw_rad, f"{name}.yaw_rad")


def _validate_safe_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"{name} is not a safe identifier")


def _validate_frame(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not _SAFE_FRAME_RE.fullmatch(value)
        or value.startswith("/")
        or "//" in value
        or ".." in value
    ):
        raise ValueError(f"{name} is not a valid relative ROS frame")


def _validate_source(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not _SAFE_SOURCE_RE.fullmatch(value)
        or ".." in value
        or "//" in value
    ):
        raise ValueError(f"{name} is not a valid provenance source")


def _poses_close(first: Pose2D, second: Pose2D, tolerance: float = 1e-9) -> bool:
    return (
        math.hypot(first.x_m - second.x_m, first.y_m - second.y_m) <= tolerance
        and angular_distance(first.yaw_rad, second.yaw_rad) <= tolerance
    )
