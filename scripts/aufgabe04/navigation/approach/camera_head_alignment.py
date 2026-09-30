"""Calibrated planar camera poses and conservative head-view error budgets.

A plan is an unsigned viewing suggestion. It is never an observed face label or
an arrival proof; those require fresh sensor evidence after execution.
"""
from __future__ import annotations

import math
from typing import Mapping

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.configuration.geometry import rotate_vector
from scripts.aufgabe04.real_robot.configuration.profile import camera_calibration_sha256


def _finite(value, field, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"camera alignment {field} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"camera alignment {field} must be nonnegative")
    return float(value)


def make_camera_alignment(*, hint, snapshot, candidate_uid, normal_rad,
                          standoff_m, calibration, uncertainty):
    """Require an observed geometry bound and calibrated extrinsics; never guess."""
    geometry = hint.fit_geometry(snapshot, candidate_uid)
    if geometry is None:
        raise ValueError("fitted_head_center_unavailable")
    if hint.evidence.get("independent_view_requirement_met") is not True:
        raise ValueError("independent_head_view_support_unavailable")
    if calibration is None:
        raise ValueError("camera_calibration_unavailable")
    if not isinstance(uncertainty, Mapping):
        raise ValueError("camera_alignment_localization_uncertainty_unavailable")
    calibration_sha = camera_calibration_sha256(calibration)
    forward = rotate_vector((0., 0., 1.), calibration.base_to_camera.rotation_xyzw)
    if math.hypot(forward[0], forward[1]) < .5:
        raise ValueError("camera_optical_axis_planar_projection_insufficient")
    payload = {
        "schema_version": 1,
        "candidate_uid": candidate_uid,
        "candidate_snapshot_sha256": hint.snapshot_sha256,
        **geometry,
        "view_normal_rad": math.remainder(normal_rad, 2 * math.pi),
        "camera_standoff_m": standoff_m,
        "camera_translation_x_m": calibration.base_to_camera.translation_xyz_m[0],
        "camera_translation_y_m": calibration.base_to_camera.translation_xyz_m[1],
        "camera_optical_yaw_rad": math.atan2(forward[1], forward[0]),
        "camera_calibration_sha256": calibration_sha,
        "localization_position_m": uncertainty.get("localization_position_m"),
        "localization_yaw_rad": uncertainty.get("localization_yaw_rad"),
        "endpoint_tracking_m": uncertainty.get("endpoint_tracking_m", .03),
        "terminal_yaw_rad": uncertainty.get("terminal_yaw_rad", math.radians(3)),
        "maximum_alignment_rad": uncertainty.get("maximum_alignment_rad", math.radians(20)),
        "head_alignment_verified": False,
        "independent_view_requirement_met": True,
        "arrival_verification_required": True,
        "stand_axis_authorized": False,
        "motion_authorized": False,
    }
    validate_camera_alignment(payload)
    return payload


def validate_camera_alignment(payload):
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("unsupported camera alignment payload")
    for field in ("center_x_m", "center_y_m", "view_normal_rad", "camera_translation_x_m",
                  "camera_translation_y_m", "camera_optical_yaw_rad"):
        _finite(payload.get(field), field)
    for field in ("center_uncertainty_m", "angle_uncertainty_rad", "camera_standoff_m",
                  "localization_position_m", "localization_yaw_rad", "endpoint_tracking_m",
                  "terminal_yaw_rad", "maximum_alignment_rad"):
        _finite(payload.get(field), field, nonnegative=True)
    if payload["camera_standoff_m"] <= 0 or not 0 < payload["maximum_alignment_rad"] < math.pi / 2:
        raise ValueError("camera alignment standoff/angular budget invalid")
    for field in ("camera_calibration_sha256", "candidate_snapshot_sha256"):
        digest = payload.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"camera alignment {field} invalid")
    if (payload.get("head_alignment_verified") is not False
            or payload.get("independent_view_requirement_met") is not True
            or payload.get("arrival_verification_required") is not True
            or payload.get("stand_axis_authorized") is not False
            or payload.get("motion_authorized") is not False):
        raise ValueError("camera alignment must remain an unverified advisory")


def requested_camera_base_pose(payload):
    validate_camera_alignment(payload)
    normal = payload["view_normal_rad"]
    yaw = math.remainder(normal + math.pi - payload["camera_optical_yaw_rad"], 2 * math.pi)
    tx, ty = payload["camera_translation_x_m"], payload["camera_translation_y_m"]
    return Pose2D(
        payload["center_x_m"] + payload["camera_standoff_m"] * math.cos(normal) - math.cos(yaw)*tx + math.sin(yaw)*ty,
        payload["center_y_m"] + payload["camera_standoff_m"] * math.sin(normal) - math.sin(yaw)*tx - math.cos(yaw)*ty,
        yaw,
    )


def camera_facing_base_yaw(payload, x_m, y_m):
    """Aim the optical axis at the center after rasterization, with lever arm."""
    dx, dy = payload["center_x_m"]-x_m, payload["center_y_m"]-y_m
    distance = math.hypot(dx, dy)
    alpha = payload["camera_optical_yaw_rad"]
    lateral = -math.sin(alpha)*payload["camera_translation_x_m"] + math.cos(alpha)*payload["camera_translation_y_m"]
    if distance <= abs(lateral):
        raise ValueError("camera alignment target inside camera lever arm")
    return math.remainder(math.atan2(dy, dx)-alpha-math.asin(lateral/distance), 2*math.pi)


def camera_alignment_endpoint(payload, pose):
    """Budget fit, localization, tracking, terminal yaw and snapped position."""
    validate_camera_alignment(payload)
    tx, ty = payload["camera_translation_x_m"], payload["camera_translation_y_m"]
    x = pose.x_m + math.cos(pose.yaw_rad)*tx-math.sin(pose.yaw_rad)*ty
    y = pose.y_m + math.sin(pose.yaw_rad)*tx+math.cos(pose.yaw_rad)*ty
    dx, dy = x-payload["center_x_m"], y-payload["center_y_m"]
    distance = math.hypot(dx, dy)
    side_error = abs(math.remainder(math.atan2(dy, dx)-payload["view_normal_rad"], 2*math.pi))
    optical_error = abs(math.remainder(pose.yaw_rad+payload["camera_optical_yaw_rad"]-payload["view_normal_rad"]-math.pi, 2*math.pi))
    yaw_bound = payload["localization_yaw_rad"]+payload["terminal_yaw_rad"]
    position_bound = (payload["center_uncertainty_m"]+payload["localization_position_m"]
                      +payload["endpoint_tracking_m"]+2*math.hypot(tx, ty)*math.sin(min(math.pi, yaw_bound)/2))
    position_angle = math.asin(position_bound/distance) if distance > position_bound else math.pi/2
    total = max(side_error, optical_error)+payload["angle_uncertainty_rad"]+position_angle+yaw_bound
    requested = requested_camera_base_pose(payload)
    endpoint_displacement = math.hypot(pose.x_m-requested.x_m, pose.y_m-requested.y_m)
    return {
        "accepted": total <= payload["maximum_alignment_rad"] and endpoint_displacement <= .06,
        "camera_x_m": x, "camera_y_m": y, "camera_distance_m": distance,
        "position_side_error_rad": side_error, "optical_normal_error_rad": optical_error,
        "position_uncertainty_angle_rad": position_angle,
        "endpoint_displacement_m": endpoint_displacement,
        "total_alignment_bound_rad": total,
        "maximum_alignment_rad": payload["maximum_alignment_rad"],
        "head_alignment_verified": False,
        "arrival_verification_required": True,
    }


def reproject_camera_alignment(payload, *, snapshot, candidate_uid,
                               source_map_from_odom, target_map_from_odom,
                               uncertainty):
    """Carry the fitted point through its odom frame and bind a fresh admission."""
    from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256
    validate_camera_alignment(payload)
    if payload["candidate_uid"] != candidate_uid or snapshot.candidate_for(candidate_uid) is None:
        raise ValueError("camera alignment reprojection candidate mismatch")
    if not isinstance(uncertainty, Mapping):
        raise ValueError("camera alignment reprojection requires fresh localization bounds")
    if (source_map_from_odom is None) != (target_map_from_odom is None):
        raise ValueError("camera alignment reprojection requires both frame transforms")
    result = dict(payload)
    if source_map_from_odom is not None:
        source, target = source_map_from_odom, target_map_from_odom
        dx, dy = payload["center_x_m"]-source.x_m, payload["center_y_m"]-source.y_m
        ox = math.cos(source.yaw_rad)*dx+math.sin(source.yaw_rad)*dy
        oy = -math.sin(source.yaw_rad)*dx+math.cos(source.yaw_rad)*dy
        result["center_x_m"] = target.x_m+math.cos(target.yaw_rad)*ox-math.sin(target.yaw_rad)*oy
        result["center_y_m"] = target.y_m+math.sin(target.yaw_rad)*ox+math.cos(target.yaw_rad)*oy
        result["view_normal_rad"] = math.remainder(payload["view_normal_rad"]+target.yaw_rad-source.yaw_rad, 2*math.pi)
    result["candidate_snapshot_sha256"] = candidate_snapshot_sha256(snapshot)
    for name in ("localization_position_m", "localization_yaw_rad"):
        result[name] = uncertainty.get(name)
    validate_camera_alignment(result)
    return result
