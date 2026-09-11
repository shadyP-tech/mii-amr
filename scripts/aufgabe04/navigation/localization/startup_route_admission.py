"""Recomputable evidence for a no-motion odom startup corridor rejection.

This evidence admits only a bounded replan attempt. It never authorizes motion
and does not replace the fresh localization, collision or certificate gates.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from scripts.aufgabe04.navigation.foundation.content_hashed_evidence import payload_sha256
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D, odom_pose_to_map, pose_route_sha256,
    transform_map_route_to_odom,
)
from scripts.aufgabe04.navigation.waypoint_follower.startup import certified_static_startup_decision

ODOM_STARTUP_ROUTE_MISMATCH = "odom_startup_route_mismatch"
ODOM_STARTUP_ROUTE_REJECTION_REASON = (
    "odom execution admission failed: odom pose is outside the transformed "
    "certified startup segment: pose left certified route tube"
)
_PHASE = "before_follower_and_motion_authorization"
_EVIDENCE_KEY = "startup_route_admission"
_HASH_KEY = "startup_route_admission_sha256"


@dataclass(frozen=True)
class OdomStartupRouteRejectionDecision:
    eligible: bool
    reason: str


class OdomStartupRouteAdmissionRejected(ValueError):
    """A validated corridor mismatch raised before follower/permit startup."""

    def __init__(self, *, evidence: Mapping[str, object], dry_run: bool) -> None:
        details = {
            "reason": ODOM_STARTUP_ROUTE_REJECTION_REASON,
            "source": ODOM_STARTUP_ROUTE_MISMATCH,
            "fault_code": ODOM_STARTUP_ROUTE_MISMATCH,
            "phase": _PHASE,
            "dry_run": dry_run,
            "execution_pose_owner": "odom",
            "global_consistency_monitor": "amcl",
            "motion_published": False,
            "follower_started": False,
            "motion_authorization_consumed": False,
            "fail_closed": True,
            _EVIDENCE_KEY: deepcopy(dict(evidence)),
            _HASH_KEY: payload_sha256(evidence),
        }
        decision = evaluate_odom_startup_route_rejection(
            status="preflight_failed", motion_published=False,
            stop_reason=ODOM_STARTUP_ROUTE_REJECTION_REASON, stop_details=details,
        )
        if not decision.eligible:
            raise ValueError(f"invalid odom startup rejection evidence: {decision.reason}")
        self._details = details
        # Reporting prepends this common admission prefix for generic errors.
        super().__init__(ODOM_STARTUP_ROUTE_REJECTION_REASON.removeprefix(
            "odom execution admission failed: "
        ))

    def to_stop_details(self) -> dict[str, object]:
        return deepcopy(self._details)


def build_odom_startup_route_admission_evidence(
    *, map_route: Sequence[Pose2D], odom_pose: Mapping[str, object],
    chained_map_pose: Mapping[str, object], map_from_odom: Mapping[str, object],
    pose_tf_observations: Mapping[str, object], map_frame: str, odom_frame: str,
    base_frame: str, tracking_tube_radius_m: float, max_tf_age_sec: float,
    max_composition_yaw_error_rad: float,
    source_preflight_sha256: str, source_map_execution_certificate_sha256: str,
) -> dict[str, object]:
    """Freeze the exact projection inputs; typed construction validates them."""

    transform = PlanarTransform2D(**_coordinates(map_from_odom))
    transformed_route = transform_map_route_to_odom(map_route, transform)
    pose = Pose2D(**_coordinates(odom_pose))
    decision = certified_static_startup_decision(
        pose, transformed_route, tracking_tube_radius_m=tracking_tube_radius_m,
    )
    composed = odom_pose_to_map(pose, transform)
    chained = Pose2D(**_coordinates(chained_map_pose))
    return {
        "schema_version": 1,
        "pose_basis": "latest_direct_map_from_odom_composed_with_odom_base",
        "map_frame": map_frame, "odom_frame": odom_frame, "base_frame": base_frame,
        "source_preflight_sha256": source_preflight_sha256,
        "source_map_execution_certificate_sha256": source_map_execution_certificate_sha256,
        "source_map_route_sha256": pose_route_sha256(map_route),
        "transformed_odom_route_sha256": pose_route_sha256(transformed_route),
        "map_from_odom_sha256": payload_sha256(map_from_odom),
        "map_route": [_route_pose_dict(item) for item in map_route],
        "transformed_odom_route": [_route_pose_dict(item) for item in transformed_route],
        "first_transformed_segment": [_route_pose_dict(item) for item in transformed_route[:2]],
        "odom_pose": dict(odom_pose), "chained_map_pose": dict(chained_map_pose),
        "composed_map_pose": _pose_dict(composed),
        "map_from_odom": dict(map_from_odom),
        "pose_tf_observations": deepcopy(dict(pose_tf_observations)),
        "tracking_tube_radius_m": tracking_tube_radius_m,
        "max_tf_age_sec": max_tf_age_sec,
        "composition_position_error_m": math.hypot(composed.x_m - chained.x_m, composed.y_m - chained.y_m),
        "composition_yaw_error_rad": _yaw_error(composed.yaw_rad, chained.yaw_rad),
        "max_composition_yaw_error_rad": max_composition_yaw_error_rad,
        "route_check": decision.route_check.to_log_dict(),
    }


def evaluate_odom_startup_route_rejection(
    *, status: object, motion_published: object, stop_reason: object,
    stop_details: object,
) -> OdomStartupRouteRejectionDecision:
    """Require a typed, finite, internally consistent no-motion rejection."""

    if status != "preflight_failed":
        return OdomStartupRouteRejectionDecision(False, "outcome_not_preflight_failed")
    if motion_published is not False:
        return OdomStartupRouteRejectionDecision(False, "motion_published_or_unknown")
    if stop_reason != ODOM_STARTUP_ROUTE_REJECTION_REASON:
        return OdomStartupRouteRejectionDecision(False, "failure_not_odom_startup_route_mismatch")
    try:
        details = _mapping(stop_details)
        for key, expected in {
            "reason": ODOM_STARTUP_ROUTE_REJECTION_REASON,
            "source": ODOM_STARTUP_ROUTE_MISMATCH,
            "fault_code": ODOM_STARTUP_ROUTE_MISMATCH, "phase": _PHASE,
            "execution_pose_owner": "odom", "global_consistency_monitor": "amcl",
        }.items():
            _require(details.get(key) == expected, f"invalid_{key}")
        for key in ("motion_published", "follower_started", "motion_authorization_consumed"):
            _require(details.get(key) is False, f"{key}_or_unknown")
        _require(details.get("fail_closed") is True, "not_fail_closed")
        _require(details.get("motion_history_uncertain", False) is False, "motion_history_uncertain")
        _require(type(details.get("dry_run")) is bool, "invalid_dry_run")
        evidence = _mapping(details.get(_EVIDENCE_KEY))
        _require(details.get(_HASH_KEY) == payload_sha256(evidence), "evidence_hash_mismatch")
        _validate_evidence(evidence)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return OdomStartupRouteRejectionDecision(False, str(exc))
    return OdomStartupRouteRejectionDecision(True, ODOM_STARTUP_ROUTE_MISMATCH)


def _validate_evidence(evidence: Mapping[str, object]) -> None:
    _require(type(evidence.get("schema_version")) is int and evidence["schema_version"] == 1,
             "unsupported_schema")
    _require(evidence.get("pose_basis") == "latest_direct_map_from_odom_composed_with_odom_base",
             "invalid_pose_basis")
    map_frame, odom_frame, base_frame = (
        evidence.get(name) for name in ("map_frame", "odom_frame", "base_frame")
    )
    _require(all(isinstance(frame, str) and frame and frame.strip() == frame
                 for frame in (map_frame, odom_frame, base_frame)), "invalid_frames")
    _require(len({map_frame, odom_frame, base_frame}) == 3, "frame_identity_mismatch")
    for key in ("source_preflight_sha256", "source_map_execution_certificate_sha256",
                "source_map_route_sha256", "transformed_odom_route_sha256", "map_from_odom_sha256"):
        digest = evidence.get(key)
        _require(isinstance(digest, str) and len(digest) == 64
                 and all(char in "0123456789abcdef" for char in digest), f"invalid_{key}")
    radius = _number(evidence.get("tracking_tube_radius_m"))
    _require(radius > 0, "invalid_tracking_tube_radius")
    max_tf_age_sec = _number(evidence.get("max_tf_age_sec"))
    _require(max_tf_age_sec > 0, "invalid_max_tf_age")
    raw_transform = _mapping(evidence.get("map_from_odom"))
    _validate_tf(raw_transform, target=map_frame, source=odom_frame, max_age_sec=max_tf_age_sec)
    _require(evidence["map_from_odom_sha256"] == payload_sha256(raw_transform),
             "transform_hash_mismatch")
    transform = PlanarTransform2D(**_coordinates(raw_transform))
    odom_pose = _pose_evidence(evidence.get("odom_pose"), odom_frame, base_frame)
    chained_pose = _pose_evidence(evidence.get("chained_map_pose"), map_frame, base_frame)
    observations = _mapping(evidence.get("pose_tf_observations"))
    for name, frame, pose in (("odom", odom_frame, odom_pose), ("map", map_frame, chained_pose)):
        observed = _mapping(observations.get(name))
        _validate_tf(observed, target=frame, source=base_frame, max_age_sec=max_tf_age_sec)
        _require(_pose_dict(pose) == _coordinates(observed), f"{name}_pose_provenance_mismatch")
    map_route = _route(evidence.get("map_route"))
    odom_route = _route(evidence.get("transformed_odom_route"))
    expected_route = transform_map_route_to_odom(map_route, transform)
    _require(pose_route_sha256(odom_route) == pose_route_sha256(expected_route),
             "transformed_route_mismatch")
    _require(evidence["source_map_route_sha256"] == pose_route_sha256(map_route), "map_route_hash_mismatch")
    _require(evidence["transformed_odom_route_sha256"] == pose_route_sha256(odom_route), "odom_route_hash_mismatch")
    _require(evidence.get("first_transformed_segment") == [_route_pose_dict(pose) for pose in odom_route[:2]],
             "first_segment_mismatch")
    _require(_coordinates(_mapping(evidence.get("composed_map_pose"))) ==
             _pose_dict(odom_pose_to_map(odom_pose, transform)), "composed_pose_mismatch")
    composed = odom_pose_to_map(odom_pose, transform)
    position_error = math.hypot(composed.x_m - chained_pose.x_m, composed.y_m - chained_pose.y_m)
    yaw_error = _yaw_error(composed.yaw_rad, chained_pose.yaw_rad)
    yaw_limit = _number(evidence.get("max_composition_yaw_error_rad"))
    _require(yaw_limit > 0 and position_error <= radius and yaw_error <= yaw_limit,
             "pose_composition_outside_admission_bounds")
    _require(_number(evidence.get("composition_position_error_m")) == position_error
             and _number(evidence.get("composition_yaw_error_rad")) == yaw_error,
             "pose_composition_diagnostics_mismatch")
    decision = certified_static_startup_decision(
        odom_pose, odom_route, tracking_tube_radius_m=radius,
    )
    _require(not decision.ok and decision.route_check.reason == "pose left certified route tube",
             "geometry_not_startup_corridor_mismatch")
    route_check = _mapping(evidence.get("route_check"))
    _require(route_check == decision.route_check.to_log_dict(), "route_check_recomputation_mismatch")
    # Equality alone would accept bools as integer indices or NaN diagnostics.
    for key in ("pose_distance_to_segment_m", "maximum_chord_distance_to_segment_m", "tracking_tube_radius_m"):
        _number(route_check.get(key))
    for key in ("active_segment_start_index", "active_segment_end_index", "target_index", "pursuit_index"):
        _require(type(route_check.get(key)) is int, "invalid_route_index")
    _require(route_check.get("fail_closed") is True, "invalid_route_failure")


def _validate_tf(raw: Mapping[str, object], *, target: str, source: str, max_age_sec: float) -> None:
    _require(raw.get("available") is True, "tf_unavailable")
    for key, expected in (("target_frame", target), ("observed_target_frame", target),
                          ("source_frame", source), ("observed_source_frame", source)):
        _require(raw.get(key) == expected, "tf_frame_identity_mismatch")
    for key in ("stamp_sec", "capture_time_sec"):
        _require(_number(raw.get(key)) >= 0, "invalid_tf_timestamp")
    age = _number(raw.get("capture_time_sec")) - _number(raw.get("stamp_sec"))
    reported_age = _number(raw.get("age_sec"))
    # Float second timestamps lose nanosecond precision at epoch-scale values.
    _require(math.isclose(age, reported_age, rel_tol=0, abs_tol=1e-6), "tf_age_inconsistent")
    future = _number(raw.get("max_future_sec"))
    _require(future >= 0 and -future <= reported_age <= max_age_sec, "tf_not_fresh")
    _coordinates(raw)


def _pose_evidence(raw: object, frame: str, child: str) -> Pose2D:
    value = _mapping(raw)
    _require(value.get("frame_id") == frame and value.get("child_frame_id") == child,
             "pose_frame_identity_mismatch")
    return Pose2D(**_coordinates(value))


def _route(raw: object) -> tuple[Pose2D, ...]:
    _require(isinstance(raw, list) and len(raw) >= 2, "invalid_route")
    poses = []
    for item in raw:
        value = _mapping(item)
        _require("yaw_rad" in value, "missing_route_yaw")
        # Only explicit route null means an unconstrained heading. Actual
        # robot poses and transforms still pass the strictly finite parser.
        poses.append(Pose2D(
            x_m=_number(value.get("x_m")), y_m=_number(value.get("y_m")),
            yaw_rad=math.nan if value["yaw_rad"] is None else _number(value["yaw_rad"]),
        ))
    return tuple(poses)


def _route_pose_dict(pose: Pose2D) -> dict[str, float | None]:
    # Match pose_route_sha256's portable JSON spelling for CSV blank yaw.
    _require(type(pose.yaw_rad) in (int, float), "invalid_route_yaw")
    return {
        "x_m": _number(pose.x_m), "y_m": _number(pose.y_m),
        "yaw_rad": None if math.isnan(pose.yaw_rad) else _number(pose.yaw_rad),
    }


def _yaw_error(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def _pose_dict(pose: Pose2D) -> dict[str, float]:
    return {"x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad}


def _coordinates(raw: Mapping[str, object]) -> dict[str, float]:
    return {name: _number(raw.get(name)) for name in ("x_m", "y_m", "yaw_rad")}


def _number(raw: object) -> float:
    _require(type(raw) in (int, float) and math.isfinite(raw), "nonfinite_or_invalid_number")
    return float(raw)


def _mapping(raw: object) -> Mapping[str, object]:
    _require(isinstance(raw, Mapping), "missing_or_invalid_mapping")
    return raw


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)
