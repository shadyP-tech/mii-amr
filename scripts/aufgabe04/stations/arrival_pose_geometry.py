"""Pure geometry for persistent, perpendicular stand-arrival poses.

Stand orientation is axial: an estimate and the same estimate rotated by
``pi`` describe the same physical stand.  This module canonicalizes that
ambiguity before assigning face IDs, so face 0/1 remain stable for equivalent
axis representations.  It deliberately contains no ROS, map, or planner
dependencies; collision and map validation belong to a separate layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast

_EPSILON = 1.0e-10


class Pose2DLike(Protocol):
    """Structural input accepted from perception, navigation, or catalog code."""

    x_m: float
    y_m: float
    yaw_rad: float


_PoseT = TypeVar("_PoseT", bound=Pose2DLike)


@dataclass(frozen=True)
class ArrivalGeometryConfig:
    """Distances used to derive a stand target and its terminal entry."""

    standoff_distance_m: float = 0.32
    terminal_corridor_length_m: float = 0.40

    def __post_init__(self) -> None:
        for name, value in (
            ("standoff_distance_m", self.standoff_distance_m),
            ("terminal_corridor_length_m", self.terminal_corridor_length_m),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ArrivalFaceGeometry(Generic[_PoseT]):
    """One stable antipodal face and its eventual physical arrival geometry."""

    face_id: int
    stand_axis_rad: float
    outward_normal_rad: float
    target_pose: _PoseT
    corridor_entry_pose: _PoseT


@dataclass(frozen=True)
class ArrivalGeometryValidation:
    """Geometric diagnostics for a proposed stored arrival face."""

    valid: bool
    standoff_distance_m: float
    standoff_error_m: float
    perpendicular_error_rad: float
    face_normal_error_rad: float
    yaw_error_rad: float
    corridor_entry_position_error_m: float
    corridor_entry_yaw_error_rad: float
    violations: tuple[str, ...]


def _require_finite(value: float, *, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_finite_pose(pose: Pose2DLike, *, name: str) -> None:
    for field_name, value in (
        ("x_m", pose.x_m),
        ("y_m", pose.y_m),
        ("yaw_rad", pose.yaw_rad),
    ):
        _require_finite(value, name=f"{name}.{field_name}")


def _pose_from_template(
    template: _PoseT, *, x_m: float, y_m: float, yaw_rad: float
) -> _PoseT:
    """Construct a pose in the caller's dependency layer.

    All Aufgabe 04 pose values use the same three-field value-object contract.
    Reconstructing the concrete input type keeps this station-domain module
    independent of navigation while allowing navigation callers to receive
    their own ``Pose2D`` and catalog callers to receive ``CatalogPose2D``.
    """

    try:
        pose = type(template)(x_m=x_m, y_m=y_m, yaw_rad=yaw_rad)
    except TypeError as exc:
        raise ValueError(
            "pose type must support x_m, y_m, and yaw_rad construction"
        ) from exc
    _require_finite_pose(pose, name="constructed pose")
    return cast(_PoseT, pose)


def normalize_angle(angle_rad: float) -> float:
    """Normalize a directed angle to ``[-pi, pi]``."""

    _require_finite(angle_rad, name="angle_rad")
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def canonical_axial_angle(axis_rad: float) -> float:
    """Canonicalize a 180-degree-symmetric axis to ``[-pi/2, pi/2)``.

    In particular, ``axis_rad`` and ``axis_rad + k*pi`` always produce the
    same canonical value and therefore the same face IDs.
    """

    _require_finite(axis_rad, name="stand axis")
    canonical = (axis_rad + math.pi / 2.0) % math.pi - math.pi / 2.0
    # Floating-point arithmetic can represent the upper endpoint after a
    # modulo operation.  Keep the interval half-open so IDs cannot change at
    # equivalent +pi/2/-pi/2 inputs.
    if canonical >= math.pi / 2.0 - _EPSILON:
        canonical = -math.pi / 2.0
    return 0.0 if abs(canonical) <= _EPSILON else canonical


def angular_distance_rad(first_rad: float, second_rad: float) -> float:
    """Return the unsigned distance between two directed angles."""

    return abs(normalize_angle(first_rad - second_rad))


def axial_distance_rad(first_rad: float, second_rad: float) -> float:
    """Return the unsigned distance between two undirected axial angles."""

    _require_finite(first_rad, name="first axial angle")
    _require_finite(second_rad, name="second axial angle")
    return 0.5 * abs(normalize_angle(2.0 * (first_rad - second_rad)))


def face_normal_rad(stand_axis_rad: float, face_id: int) -> float:
    """Return the stable outward normal for face 0 or face 1."""

    if type(face_id) is not int or face_id not in (0, 1):
        raise ValueError("face_id must be 0 or 1")
    axis = canonical_axial_angle(stand_axis_rad)
    offset = math.pi / 2.0 if face_id == 0 else -math.pi / 2.0
    return normalize_angle(axis + offset)


def arrival_face_candidates(
    stand: _PoseT,
    stand_axis_rad: float,
    config: ArrivalGeometryConfig = ArrivalGeometryConfig(),
) -> tuple[ArrivalFaceGeometry[_PoseT], ArrivalFaceGeometry[_PoseT]]:
    """Derive both explicit antipodal arrival poses for a stand axis."""

    _require_finite_pose(stand, name="stand")
    axis = canonical_axial_angle(stand_axis_rad)
    candidates: list[ArrivalFaceGeometry[_PoseT]] = []
    for face_id in (0, 1):
        normal = face_normal_rad(axis, face_id)
        yaw = normalize_angle(normal + math.pi)
        target_radius = config.standoff_distance_m
        entry_radius = target_radius + config.terminal_corridor_length_m
        candidates.append(
            ArrivalFaceGeometry(
                face_id=face_id,
                stand_axis_rad=axis,
                outward_normal_rad=normal,
                target_pose=_pose_from_template(
                    stand,
                    x_m=stand.x_m + target_radius * math.cos(normal),
                    y_m=stand.y_m + target_radius * math.sin(normal),
                    yaw_rad=yaw,
                ),
                corridor_entry_pose=_pose_from_template(
                    stand,
                    x_m=stand.x_m + entry_radius * math.cos(normal),
                    y_m=stand.y_m + entry_radius * math.sin(normal),
                    yaw_rad=yaw,
                ),
            )
        )
    return candidates[0], candidates[1]


def observer_facing_arrival_face(
    stand: _PoseT,
    stand_axis_rad: float,
    observer: Pose2DLike,
    config: ArrivalGeometryConfig = ArrivalGeometryConfig(),
) -> ArrivalFaceGeometry[_PoseT]:
    """Select the stand face whose outward normal points toward the observer.

    Equal angular distances use ``face_id`` as a deterministic tie-breaker.
    An observer at the stand center has no defined facing side and is rejected.
    """

    _require_finite_pose(stand, name="stand")
    _require_finite_pose(observer, name="observer")
    dx = observer.x_m - stand.x_m
    dy = observer.y_m - stand.y_m
    if math.hypot(dx, dy) <= _EPSILON:
        raise ValueError("observer must not coincide with stand center")
    observer_bearing = math.atan2(dy, dx)
    candidates = arrival_face_candidates(stand, stand_axis_rad, config)
    return min(
        candidates,
        key=lambda item: (
            angular_distance_rad(item.outward_normal_rad, observer_bearing),
            item.face_id,
        ),
    )


def validate_arrival_face_geometry(
    stand: Pose2DLike,
    stand_axis_rad: float,
    face: ArrivalFaceGeometry,
    config: ArrivalGeometryConfig = ArrivalGeometryConfig(),
    *,
    position_tolerance_m: float = 1.0e-6,
    angle_tolerance_rad: float = 1.0e-6,
) -> ArrivalGeometryValidation:
    """Validate standoff, perpendicularity, face identity, yaw, and entry.

    Non-finite inputs and invalid tolerances are programming errors and raise
    ``ValueError``.  Finite but geometrically inconsistent proposals return a
    diagnostic result with ``valid=False``.
    """

    _require_finite_pose(stand, name="stand")
    _require_finite_pose(face.target_pose, name="target_pose")
    _require_finite_pose(face.corridor_entry_pose, name="corridor_entry_pose")
    _require_finite(face.stand_axis_rad, name="face.stand_axis_rad")
    _require_finite(face.outward_normal_rad, name="face.outward_normal_rad")
    canonical_axis = canonical_axial_angle(stand_axis_rad)
    expected_normal = face_normal_rad(canonical_axis, face.face_id)
    for name, tolerance in (
        ("position_tolerance_m", position_tolerance_m),
        ("angle_tolerance_rad", angle_tolerance_rad),
    ):
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    target_dx = face.target_pose.x_m - stand.x_m
    target_dy = face.target_pose.y_m - stand.y_m
    standoff_distance = math.hypot(target_dx, target_dy)
    standoff_error = abs(standoff_distance - config.standoff_distance_m)

    if standoff_distance <= _EPSILON:
        target_bearing = expected_normal
        perpendicular_error = math.pi / 2.0
        face_normal_error = math.pi
    else:
        target_bearing = math.atan2(target_dy, target_dx)
        perpendicular_error = abs(
            math.pi / 2.0 - axial_distance_rad(target_bearing, canonical_axis)
        )
        face_normal_error = angular_distance_rad(target_bearing, expected_normal)

    expected_yaw = normalize_angle(expected_normal + math.pi)
    yaw_error = angular_distance_rad(face.target_pose.yaw_rad, expected_yaw)
    expected_entry = _pose_from_template(
        stand,
        x_m=stand.x_m
        + (config.standoff_distance_m + config.terminal_corridor_length_m)
        * math.cos(expected_normal),
        y_m=stand.y_m
        + (config.standoff_distance_m + config.terminal_corridor_length_m)
        * math.sin(expected_normal),
        yaw_rad=expected_yaw,
    )
    entry_position_error = math.hypot(
        face.corridor_entry_pose.x_m - expected_entry.x_m,
        face.corridor_entry_pose.y_m - expected_entry.y_m,
    )
    entry_yaw_error = angular_distance_rad(
        face.corridor_entry_pose.yaw_rad, expected_entry.yaw_rad
    )

    violations: list[str] = []
    if angular_distance_rad(face.stand_axis_rad, canonical_axis) > angle_tolerance_rad:
        violations.append("stand_axis_not_canonical")
    if angular_distance_rad(face.outward_normal_rad, expected_normal) > angle_tolerance_rad:
        violations.append("outward_normal_mismatch")
    if standoff_error > position_tolerance_m:
        violations.append("arrival_standoff_mismatch")
    if perpendicular_error > angle_tolerance_rad:
        violations.append("arrival_not_perpendicular_to_axis")
    if face_normal_error > angle_tolerance_rad:
        violations.append("arrival_on_wrong_face")
    if yaw_error > angle_tolerance_rad:
        violations.append("arrival_yaw_not_facing_stand")
    if entry_position_error > position_tolerance_m:
        violations.append("corridor_entry_position_mismatch")
    if entry_yaw_error > angle_tolerance_rad:
        violations.append("corridor_entry_yaw_mismatch")

    return ArrivalGeometryValidation(
        valid=not violations,
        standoff_distance_m=standoff_distance,
        standoff_error_m=standoff_error,
        perpendicular_error_rad=perpendicular_error,
        face_normal_error_rad=face_normal_error,
        yaw_error_rad=yaw_error,
        corridor_entry_position_error_m=entry_position_error,
        corridor_entry_yaw_error_rad=entry_yaw_error,
        violations=tuple(violations),
    )
