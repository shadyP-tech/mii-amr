"""Verify an executed camera view against fresh, stopped LiDAR head geometry.

This is a passive geometry check, never a face label or motion permit. Actual
endpoint and yaw errors are measured here; planning tracking/yaw allowances
must not be added a second time. Fit bounds remain conditional engineering
bounds, not empirically calibrated accuracy claims.
"""

from __future__ import annotations

from dataclasses import asdict
import math

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.lidar_visibility_evidence import validate_lidar_visibility_receipt
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import axial_difference_rad
from scripts.aufgabe04.real_robot.configuration.geometry import rotate_vector
from scripts.aufgabe04.real_robot.configuration.profile import camera_calibration_sha256


def _finite(value, name, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value) or nonnegative and value < 0:
        raise ValueError(f"invalid_{name}")
    return float(value)


def _pose(value, name):
    if not isinstance(value, Pose2D):
        raise ValueError(f"invalid_{name}")
    for key in ("x_m", "y_m", "yaw_rad"):
        _finite(getattr(value, key), f"{name}_{key}")
    return value


def verify_lidar_alignment_arrival(
    *, hint, current_fit, current_receipts, snapshot, candidate_uid, planning_frame,
    calibration, now_sec, not_before_sec, current_base_pose_odom=None, pose_stamp_sec=None,
    localization_position_bound_m=None, localization_yaw_bound_rad=None,
    maximum_alignment_rad=math.radians(20.), maximum_centering_rad=math.radians(3.),
    max_source_age_sec=.5, max_history_sec=1.5,
):
    """Return explicit unverified evidence on missing, stale or conflicting input.

    ``current_fit`` must be fitted only from ``current_receipts`` with
    ``fit_current_lidar_view``. The capture boundary supplies the exact-time
    base pose from the same stopped scan cohort. The hint and current fit are
    expressed in the supplied planning frame, whose pose is recomposed from
    that base pose. A previous survey fit alone can never verify an arrival.
    """
    result = dict(schema_version=1, policy="fresh_lidar_camera_alignment_arrival",
                  candidate_uid=candidate_uid, accepted=False,
                  head_alignment_verified=False, camera_centered_verified=False,
                  front_back_identity_resolved=False, stand_axis_authorized=False,
                  motion_authorized=False, angle_accuracy_calibrated=False,
                  fallback="ordinary_camera_acquisition_unverified")

    def reject(reason):
        return {**result, "reason": reason}

    try:
        now = _finite(now_sec, "now_sec")
        not_before = _finite(not_before_sec, "not_before_sec")
        age_limit = _finite(max_source_age_sec, "max_source_age_sec")
        history_limit = _finite(max_history_sec, "max_history_sec")
        angle_limit = _finite(maximum_alignment_rad, "maximum_alignment_rad")
        center_limit = _finite(maximum_centering_rad, "maximum_centering_rad")
        if not 0 < age_limit <= .5 or not 0 < history_limit <= 1.5:
            return reject("arrival_freshness_bounds_exceeded")
        if not 0 < center_limit <= angle_limit < math.pi / 2:
            return reject("arrival_angular_limits_invalid")
        position_bound = _finite(localization_position_bound_m, "localization_position_bound_m", nonnegative=True)
        yaw_bound = _finite(localization_yaw_bound_rad, "localization_yaw_bound_rad", nonnegative=True)
        if yaw_bound >= math.pi / 2:
            return reject("arrival_localization_yaw_bound_excessive")
        if hint is None or current_fit is None:
            return reject("fresh_head_fit_unavailable")
        historical = hint.fit_geometry(snapshot, candidate_uid)
        current = current_fit.fit_geometry(snapshot, candidate_uid)
        if historical is None or current is None:
            return reject("fitted_head_geometry_unavailable")
        history_views = hint.evidence.get("viewpoint_ids", ())
        if len(set(history_views)) < 2 or hint.evidence.get("independent_view_requirement_met") is False:
            return reject("independent_survey_hint_required")
        expected_frame = planning_frame.to_evidence()
        for fitted in (hint, current_fit):
            source_frame = fitted.evidence.get("planning_frame", {})
            if any(source_frame.get(key) != expected_frame[key] for key in
                   ("map_frame", "odom_frame", "map_from_odom")):
                return reject("fitted_geometry_planning_frame_mismatch")
        receipts = tuple(current_receipts)
        if len(receipts) < 3:
            return reject("three_current_scans_required")
        for receipt in receipts:
            validate_lidar_visibility_receipt(receipt)
        stamps = [r.scan_stamp_sec for r in receipts]
        if any(a >= b for a, b in zip(stamps, stamps[1:])):
            return reject("current_scan_order_or_identity_invalid")
        if (stamps[0] < not_before or stamps[-1] - stamps[0] > history_limit
                or not 0 <= now - stamps[-1] <= age_limit):
            return reject("arrival_scan_sources_not_fresh")
        if any(not 0 <= r.observer_clock_sec - r.scan_stamp_sec <= age_limit for r in receipts):
            return reject("arrival_scan_not_fresh_when_captured")
        hashes = [r.receipt_sha256 for r in receipts]
        evidence = current_fit.evidence
        examined = evidence.get("examined_receipt_sha256s")
        accepted = evidence.get("source_receipt_sha256s")
        if (not isinstance(examined, (list, tuple)) or set(examined) != set(hashes)
                or len(examined) != len(hashes)
                or not isinstance(accepted, (list, tuple)) or len(accepted) < 3
                or len(set(accepted)) != len(accepted) or not set(accepted).issubset(hashes)
                or evidence.get("independent_view_requirement_met") is not False):
            return reject("fresh_fit_scan_binding_invalid")
        last_support_stamp = max(r.scan_stamp_sec for r in receipts if r.receipt_sha256 in accepted)
        if not 0 <= now - last_support_stamp <= age_limit:
            return reject("arrival_fitted_support_not_fresh")
        if len({(r.survey_id, r.viewpoint_id, r.scan_frame, r.map_bundle_sha256) for r in receipts}) != 1:
            return reject("arrival_scan_context_changed")
        if any(r.map_bundle_sha256 != snapshot.map_bundle_sha256 or r.frame_provenance is None
               or r.planning_frame != planning_frame.map_frame
               or r.frame_provenance.odom_frame != planning_frame.odom_frame for r in receipts):
            return reject("arrival_scan_frame_or_map_mismatch")
        anchor = receipts[0].frame_provenance.canonical_scan_pose_odom
        if any(math.hypot(r.frame_provenance.canonical_scan_pose_odom.x_m-anchor.x_m,
                         r.frame_provenance.canonical_scan_pose_odom.y_m-anchor.y_m) > .02
               or abs(math.remainder(r.frame_provenance.canonical_scan_pose_odom.yaw_rad-anchor.yaw_rad,
                                     math.tau)) > math.radians(2.) for r in receipts):
            return reject("arrival_scan_cohort_not_stationary")
        base = _pose(current_base_pose_odom, "current_base_pose_odom")
        pose_stamp = _finite(pose_stamp_sec, "pose_stamp_sec")
        if not 0 <= now - pose_stamp <= age_limit or abs(pose_stamp-stamps[-1]) > .1:
            return reject("arrival_base_pose_not_current")
        robot = _pose(planning_frame.current_pose, "current_robot_pose")
        tf = planning_frame.map_from_odom
        c, s = math.cos(tf.yaw_rad), math.sin(tf.yaw_rad)
        mapped = Pose2D(tf.x_m+c*base.x_m-s*base.y_m,
                        tf.y_m+s*base.x_m+c*base.y_m, base.yaw_rad+tf.yaw_rad)
        if (math.hypot(mapped.x_m-robot.x_m, mapped.y_m-robot.y_m) > 1e-8
                or abs(math.remainder(mapped.yaw_rad-robot.yaw_rad, math.tau)) > 1e-8):
            return reject("arrival_base_pose_planning_frame_mismatch")
        center_difference = math.hypot(current["center_x_m"]-historical["center_x_m"],
                                       current["center_y_m"]-historical["center_y_m"])
        axis_difference = axial_difference_rad(current_fit.tangent_rad, hint.tangent_rad)
        result.update(source_receipt_sha256s=hashes, accepted_fit_receipt_sha256s=list(accepted),
                      current_scan_stamp_sec=stamps[-1], checked_at_sec=now, pose_stamp_sec=pose_stamp,
                      current_base_pose_odom=asdict(base), robot_pose=asdict(robot),
                      planning_frame=planning_frame.to_evidence(),
                      fresh_fit=dict(current), historical_fit=dict(historical),
                      center_disagreement_m=center_difference, normal_disagreement_rad=axis_difference)
        if center_difference > current["center_uncertainty_m"] + historical["center_uncertainty_m"]:
            return reject("fresh_head_center_conflicts_with_hint")
        if axis_difference > current["angle_uncertainty_rad"] + historical["angle_uncertainty_rad"]:
            return reject("fresh_head_normal_conflicts_with_hint")
        calibration_sha = camera_calibration_sha256(calibration)
        forward = rotate_vector((0., 0., 1.), calibration.base_to_camera.rotation_xyzw)
        if math.hypot(forward[0], forward[1]) < .5:
            return reject("camera_optical_axis_planar_projection_insufficient")
        tx, ty = calibration.base_to_camera.translation_xyz_m[:2]
        c, s = math.cos(robot.yaw_rad), math.sin(robot.yaw_rad)
        camera_x, camera_y = robot.x_m+c*tx-s*ty, robot.y_m+s*tx+c*ty
        dx, dy = current["center_x_m"]-camera_x, current["center_y_m"]-camera_y
        distance = math.hypot(dx, dy)
        target_bearing = math.atan2(dy, dx)
        optical_yaw = robot.yaw_rad + math.atan2(forward[1], forward[0])
        normal = current_fit.tangent_rad + math.pi / 2
        side_error = axial_difference_rad(target_bearing, normal)
        optical_error = axial_difference_rad(optical_yaw, normal)
        center_error = abs(math.remainder(optical_yaw-target_bearing, math.tau))
        lever_arm_bound = 2*math.hypot(tx, ty)*math.sin(yaw_bound/2)
        total_position = current["center_uncertainty_m"] + position_bound + lever_arm_bound
        if distance <= total_position:
            return reject("arrival_target_range_inside_uncertainty")
        position_angle = math.asin(total_position/distance)
        normal_bound = max(side_error, optical_error) + current["angle_uncertainty_rad"] + position_angle + yaw_bound
        center_bound = center_error + position_angle + yaw_bound
        result.update(camera_calibration_sha256=calibration_sha,
                      camera_x_m=camera_x, camera_y_m=camera_y, camera_optical_yaw_rad=optical_yaw,
                      camera_distance_m=distance, position_side_error_rad=side_error,
                      optical_normal_error_rad=optical_error, camera_center_error_rad=center_error,
                      fit_angle_uncertainty_rad=current["angle_uncertainty_rad"],
                      center_uncertainty_m=current["center_uncertainty_m"],
                      localization_position_bound_m=position_bound, localization_yaw_bound_rad=yaw_bound,
                      camera_lever_arm_uncertainty_m=lever_arm_bound,
                      position_uncertainty_angle_rad=position_angle,
                      total_alignment_bound_rad=normal_bound, maximum_alignment_rad=angle_limit,
                      total_centering_bound_rad=center_bound, maximum_centering_rad=center_limit,
                      actual_endpoint_and_yaw_used=True, planning_execution_tolerances_added=False)
        # Axial agreement intentionally cannot distinguish front from back.
        # Directed centering additionally ensures the target is in front.
        if center_error >= math.pi/2:
            return reject("arrival_target_behind_camera")
        if normal_bound > angle_limit:
            return reject("arrival_alignment_budget_exceeded")
        result.update(accepted=True, head_alignment_verified=True,
                      camera_centered_verified=center_bound <= center_limit,
                      fallback=None, reason="fresh_lidar_camera_alignment_verified")
        return result
    except (AttributeError, KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return reject(str(exc) or "invalid_arrival_alignment_input")
