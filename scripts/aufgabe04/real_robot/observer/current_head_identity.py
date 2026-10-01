"""Bind payload-only identity to an exclusive current measured head region.

Head pixels and their synchronized LiDAR support own geometry. A decoded
payload has no corners, scale, bearing, range, or angle authority.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
import math

from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import bind_crop_text


POLICY = "current_measured_head_identity_only"
CONTEXT_FIELDS = frozenset({"policy", "candidate_uid", "stream_id", "planning_frame",
    "stand_center", "robot_pose", "target_key", "motion_epoch", "camera_signature",
    "image_shape", "sensor_stamp_sec", "scan_stamp_sec", "checked_at_sec",
    "robot_profile_sha256", "calibration_profile_sha256", "stand_model_profile_sha256",
    "localization_provenance"})


def validate_identity_startup(model_profile, *, candidate_crop_snapshot):
    """Check required ownership configuration before starting a ROS observer.

    Snapshot contents are validated by the existing artifact loaders. This
    pure configuration check prevents silently disabling the physical ID path.
    """
    if (getattr(model_profile, "committable", False)
            and getattr(model_profile, "environment", None) == "physical"
            and candidate_crop_snapshot is None):
        raise ValueError("--candidate-crop-snapshot is required for a measured physical stand profile "
                         "to bind decoded identity and exclude neighboring candidates")


def is_current_head_identity(binding):
    crop = binding.get("current_head_binding") if isinstance(binding, Mapping) else getattr(binding, "current_head_binding", None)
    return (isinstance(crop, Mapping) and crop.get("sampling") == "masked_current_head_region"
            and isinstance(crop.get("ordinary_context"), Mapping)
            and crop["ordinary_context"].get("policy") == POLICY)


def _same(left, right):
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    return left == right


def validate_current_head_identity_binding(binding, *, image_stamp_sec=None,
        scan_stamp_sec=None, image_shape=None, target_key=None, camera_signature=None,
        model_profile_sha256=None, head_corners=None, expected_context=None):
    """Replay crop/source bindings without deriving anything from QR geometry."""
    if (not isinstance(binding, Mapping) or binding.get("accepted") is not True
            or binding.get("reason") != "decoded_qr_target_associated"
            or type(binding.get("symbol_count")) is not int or binding["symbol_count"] != 1
            or len(binding.get("qr_texts_for_evidence") or ()) != 1
            or not is_current_head_identity(binding)):
        raise ValueError("current head identity needs one exclusive decoded payload")
    crop = binding["current_head_binding"]
    context = crop["ordinary_context"]
    if set(context) != CONTEXT_FIELDS:
        raise ValueError("current head identity context is incomplete")
    for field in ("candidate_uid", "stream_id", "planning_frame", "target_key"):
        if not isinstance(context[field], str) or not context[field].strip():
            raise ValueError("current head identity target context missing")
    for field in ("robot_profile_sha256", "calibration_profile_sha256", "stand_model_profile_sha256"):
        value = context[field]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("current head identity profile context invalid")
    if type(context["motion_epoch"]) is not int or context["motion_epoch"] < 0:
        raise ValueError("current head identity motion epoch invalid")
    stamps = [context[field] for field in ("sensor_stamp_sec", "scan_stamp_sec", "checked_at_sec")]
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in stamps)
            or abs(stamps[0]-stamps[1]) > .1+1e-9
            or any(not -.05 <= stamps[2]-value <= .5 for value in stamps[:2])):
        raise ValueError("current head identity source tuple is stale or unsynchronized")
    for field, expected in (("sensor_stamp_sec", image_stamp_sec), ("scan_stamp_sec", scan_stamp_sec),
            ("image_shape", image_shape), ("target_key", target_key),
            ("camera_signature", camera_signature), ("stand_model_profile_sha256", model_profile_sha256)):
        if expected is not None and not _same(context[field], expected):
            raise ValueError("current head identity differs from current " + field)
    if expected_context is not None:
        for field in CONTEXT_FIELDS - {"policy", "checked_at_sec"}:
            if field in expected_context and not _same(context[field], expected_context[field]):
                raise ValueError("current head identity differs from receipt " + field)
    if any(binding.get(key) is not None for key in
           ("camera_bearing_rad", "finite_bearing", "range_resolution", "independent_registration", "target_reconciliation")):
        raise ValueError("decoded payload cannot supply geometric evidence")
    support = crop.get("target_support") or {}
    from scripts.aufgabe04.real_robot.observer.opposite_head_support import HEAD_POLICY
    if support.get("policy") != HEAD_POLICY:
        raise ValueError("current head identity requires measured head support")
    if head_corners is not None and not _same(support.get("corners_px"), head_corners):
        raise ValueError("current head identity belongs to another measured border")
    from scripts.aufgabe04.artifacts.qr_verified_observation_pose import _validate_opposite_crop
    data = {**context, "qr_binding": binding, "qr_id": binding["qr_texts_for_evidence"][0]}
    _validate_opposite_crop(crop, data, stamps[0], stamps[1], context["image_shape"])
    return deepcopy(dict(context))


def bind_current_head_identity(observations, crop, *, context):
    """Promote only one payload decoded from the region described by this crop."""
    observations = tuple(observations or ())
    if any(observation.corners is not None for observation in observations):
        raise ValueError("current head identity decoder must not return QR geometry")
    crop = {**crop, "ordinary_context": {**context, "policy": POLICY}}
    binding = bind_crop_text(observations, crop)
    if not binding.accepted:
        return binding
    binding = replace(binding, reason="decoded_qr_target_associated")
    validate_current_head_identity_binding(binding.metadata())
    return binding


def acquire_current_head_identity(adapter, *, frame, estimate, association, crop_review,
        selected_roi, intrinsics, scan, scan_from_camera, scan_from_map, camera_from_map,
        map_bearing_rad, accepted_range_m, image_stamp_sec, robot_pose, camera_signature,
        target_reconciliation=None, fragmentation=None, metadata):
    """Decode only after current head, target association and neighbor review."""
    from pathlib import Path
    from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr
    from scripts.aufgabe04.real_robot.configuration.profile import camera_calibration_sha256, real_robot_profile_sha256
    from scripts.aufgabe04.real_robot.observer.opposite_head_support import head_region_from_corners, support_opposite_head_region
    from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import exclusive_identity_crop, masked_identity_pixels
    from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search
    from scripts.aufgabe04.real_robot.observer.qr_target_binding import QrTargetBinding
    from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot

    details = dict(policy=POLICY, attempted=False, accepted_crop=False, decoded_texts=[],
        image_stamp_sec=image_stamp_sec, scan_stamp_sec=scan.scan_stamp_sec,
        qr_geometry_used=False)
    metadata["current_head_identity"] = details

    def miss(reason):
        details["reason"] = reason
        return (), QrTargetBinding(False, reason)

    if association is None or not association.accepted or not crop_review.accepted:
        return miss("current_complete_associated_head_required")
    snapshot_path = getattr(adapter.args, "candidate_crop_snapshot", None)
    if snapshot_path is None:
        return miss("current_head_identity_snapshot_required")
    now = lambda: adapter.node.get_clock().now().nanoseconds / 1e9
    options = dict(scan=scan, scan_from_map=scan_from_map, camera_from_map=camera_from_map,
        intrinsics=intrinsics, model_profile=adapter.stand_model_profile,
        image_stamp_sec=image_stamp_sec, sync_tolerance_sec=adapter.args.sync_tolerance_sec,
        map_bearing_rad=map_bearing_rad,
        cone_half_angle_rad=math.radians(adapter.args.lidar_cone_half_angle_deg),
        max_camera_map_bearing_delta_rad=math.radians(adapter.args.backside_registration_max_bearing_delta_deg),
        accepted_range_m=accepted_range_m, now_sec=now(), max_scan_age_sec=adapter.args.max_sensor_age_sec,
        target_reconciliation=target_reconciliation, fragmentation=fragmentation)
    try:
        snapshot = load_candidate_snapshot(Path(snapshot_path))
        search = current_scan_qr_search(**options)
        details["search"] = search[1]
        corners = tuple((point.u_px+selected_roi.x0, point.v_px+selected_roi.y0)
                        for point in estimate.corners or ())
        region = head_region_from_corners(corners, image_shape=frame.shape[:2],
            image_stamp_sec=image_stamp_sec, attempt=search[0], model_profile=adapter.stand_model_profile)
        if region is None:
            return miss("current_head_identity_region_unavailable")
        support = support_opposite_head_region(region, attempt=search[0], image_shape=frame.shape[:2],
            intrinsics=intrinsics, model_profile=adapter.stand_model_profile,
            scan_from_camera=scan_from_camera, scan=scan, image_stamp_sec=image_stamp_sec,
            now_sec=now(), map_bearing_rad=map_bearing_rad,
            cone_half_angle_rad=options["cone_half_angle_rad"], accepted_range_m=accepted_range_m,
            max_scan_age_sec=adapter.args.max_sensor_age_sec,
            max_camera_map_bearing_delta_rad=options["max_camera_map_bearing_delta_rad"],
            target_reconciliation=target_reconciliation, fragmentation=fragmentation,
            diagnostics=details.setdefault("support", {}))
        if support is None:
            return miss("current_head_identity_support_unavailable")
        previous = association.lidar_association.search_association
        current = support.lidar_association.search_association
        if (previous.scan_stamp_sec != current.scan_stamp_sec or previous.scan_frame_id != current.scan_frame_id
                or not set(previous.selected_cluster_source_indices).intersection(current.selected_cluster_source_indices)):
            return miss("current_head_identity_cluster_mismatch")
        attempt, crop = exclusive_identity_crop(candidate_uid=adapter.args.stand_id, snapshot=snapshot,
            support=support, search_result=search, **options)
        details["crop"] = crop
        if attempt is None or crop.get("sampling") != "masked_current_head_region":
            return miss(crop.get("reason", "current_head_identity_crop_unavailable"))
        details["accepted_crop"] = True
        remaining = min(.12, adapter.args.max_sensor_age_sec-max(0., now()-min(image_stamp_sec, scan.scan_stamp_sec))-.05)
        if remaining <= 0:
            return miss("current_head_identity_source_budget_exhausted")
        pixels = masked_identity_pixels(frame, adapter.cv2, roi=attempt.roi, corners_px=support.corners_px)
        decoder = details.setdefault("decoder", {})
        observations = detect_qr_observations_bgr(pixels, adapter.cv2, identity_only=True,
            preferred_scale=2, max_elapsed_sec=remaining, diagnostics=decoder,
            **getattr(adapter, "_qr_decoder_options", {}))
        details["attempted"] = any(event.get("stage") in {"wechat", "opencv_multi", "opencv_single"}
            and event.get("reason") != "decoder_error" for event in decoder.get("events", ()))
        details["decoded_texts"] = [observation.text for observation in observations]
        args, profile = adapter.args, adapter.profile
        epoch = 0 if adapter.observation_evidence is None else adapter.observation_evidence.snapshot().motion_epoch
        context = dict(candidate_uid=args.stand_id, stream_id=args.stream_id, planning_frame=profile.map_frame,
            stand_center=dict(x_m=args.stand_x, y_m=args.stand_y), robot_pose=asdict(robot_pose),
            target_key=adapter._target_evidence_key(), motion_epoch=epoch,
            camera_signature=tuple(camera_signature), image_shape=tuple(frame.shape[:2]),
            sensor_stamp_sec=image_stamp_sec, scan_stamp_sec=scan.scan_stamp_sec, checked_at_sec=now(),
            robot_profile_sha256=real_robot_profile_sha256(profile),
            calibration_profile_sha256=camera_calibration_sha256(adapter.calibration),
            stand_model_profile_sha256=adapter.stand_model_profile.sha256,
            localization_provenance=dict(map_frame=profile.map_frame, base_frame=profile.base_frame,
                scan_frame=profile.scan_frame, camera_frame=profile.camera_optical_frame,
                exact_image_transform_stamp_sec=image_stamp_sec, exact_scan_transform_stamp_sec=scan.scan_stamp_sec))
        binding = bind_current_head_identity(observations, crop, context=context)
        details["reason"] = binding.reason
        return observations, binding
    except (OSError, TypeError, ValueError, KeyError, AttributeError, ArithmeticError) as exc:
        details["detail"] = str(exc)
        return miss("current_head_identity_rejected")
