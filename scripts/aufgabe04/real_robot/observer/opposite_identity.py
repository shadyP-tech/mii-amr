"""Stopped opposite-side identity acquisition without another head-angle fit."""
import math
from types import SimpleNamespace

from scripts.aufgabe04.artifacts.retained_backside_orientation import opposite_view_matches
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr
from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import (
    exclusive_identity_crop, bind_crop_text, masked_identity_pixels, decoded_symbol_identity_crop,
)
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import prepare_qr_observation_pose
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import (
    current_scan_qr_search, retained_opposite_qr_search,
)
from scripts.aufgabe04.real_robot.observer.opposite_head_support import (
    detect_opposite_head_support, detect_opposite_head_region, support_opposite_head_region,
)
from scripts.aufgabe04.real_robot.observer.candidate_centering_receipt import (
    prepare_candidate_centering, centering_observation_requested,
)
from scripts.aufgabe04.real_robot.configuration.geometry import pose2d_from_transform
from scripts.aufgabe04.real_robot.observer.opposite_endpoint_confirmation import opposite_endpoint_validation_scope


@opposite_endpoint_validation_scope()
def process_opposite_identity(adapter, *, context, frame, intrinsics, robot_pose,
        camera_signature, image_stamp_sec, scan, scan_from_map, camera_from_map,
        map_bearing_rad, accepted_range_m, scan_from_camera, base_from_camera, image_stamp,
        target_reconciliation=None, fragmentation=None, require_target_reconciliation=False,
        persistence_context=None, reconcile_target=None, raw_frame=None, camera_calibration=None):
    """The caller supplies the ordinary stopped, exact-TF sensor tuple.

    A newly decoded payload may finish this branch immediately. No historical
    identity, measured front corners, or synthetic current axis samples enter.
    """
    now = adapter.node.get_clock().now().nanoseconds / 1e9
    orientation = context.orientation
    if not opposite_view_matches(orientation, robot_pose):
        adapter._note_observation_soft_miss('retained_backside_wrong_side', stamp_sec=image_stamp_sec, pose=robot_pose)
        adapter._write_status('opposite_identity_unavailable', reason='retained_backside_wrong_side')
        return
    endpoint_diagnostics = {}
    endpoint_region = None
    if (require_target_reconciliation and target_reconciliation is None
            and persistence_context is not None and reconcile_target is not None):
        from scripts.aufgabe04.real_robot.observer.opposite_endpoint_confirmation import (
            build_opposite_endpoint_hint, confirm_opposite_endpoint,
        )
        try:
            hint, seed = build_opposite_endpoint_hint(scan=scan,
                persistence_context=persistence_context,
                snapshot_path=adapter.args.candidate_crop_snapshot,
                intrinsics=intrinsics, scan_from_camera=scan_from_camera,
                camera_from_map=camera_from_map, scan_from_map=scan_from_map,
                model_profile=adapter.stand_model_profile,
                model_path=adapter.args.stand_model_profile, now_sec=now,
                max_scan_age_sec=adapter.args.max_sensor_age_sec,
                map_bearing_rad=map_bearing_rad, accepted_range_m=accepted_range_m,
                cone_half_angle_rad=math.radians(adapter.args.lidar_cone_half_angle_deg),
                max_camera_map_bearing_delta_rad=math.radians(adapter.args.backside_registration_max_bearing_delta_deg))
            now = adapter.node.get_clock().now().nanoseconds/1e9
            endpoint_region = detect_opposite_head_region(frame, adapter.cv2, attempt=hint,
                model_profile=adapter.stand_model_profile, image_stamp_sec=image_stamp_sec,
                now_sec=now, max_scan_age_sec=adapter.args.max_sensor_age_sec,
                resources=getattr(adapter, '_qr_decoder_options', {}).get('resources'),
                max_elapsed_sec=min(.06, adapter.args.max_sensor_age_sec
                    -max(0., now-min(image_stamp_sec, scan.scan_stamp_sec))-.08),
                diagnostics=endpoint_diagnostics)
            if endpoint_region is not None:
                confirmed = confirm_opposite_endpoint(seed, endpoint_region,
                    now_sec=adapter.node.get_clock().now().nanoseconds/1e9)
                target_reconciliation = reconcile_target(confirmed)
                if target_reconciliation is not None:
                    fragmentation = confirmed
                    adapter._current_position_epoch_proof = target_reconciliation
                    endpoint_diagnostics.update(accepted=True, reason='current_head_confirms_endpoint_target')
                else:
                    endpoint_region = None
                    endpoint_diagnostics.update(accepted=False, reason='endpoint_target_reconciliation_rejected')
        except (ValueError, TypeError, KeyError, ArithmeticError, OSError) as exc:
            endpoint_region = None
            endpoint_diagnostics.update(accepted=False, reason=str(exc))
        now = adapter.node.get_clock().now().nanoseconds/1e9
    options = dict(scan=scan, scan_from_map=scan_from_map,
        target_reconciliation=target_reconciliation,
        camera_from_map=camera_from_map, intrinsics=intrinsics,
        model_profile=adapter.stand_model_profile, image_stamp_sec=image_stamp_sec,
        sync_tolerance_sec=adapter.args.sync_tolerance_sec, fragmentation=fragmentation,
        map_bearing_rad=map_bearing_rad,
        cone_half_angle_rad=math.radians(adapter.args.lidar_cone_half_angle_deg),
        max_camera_map_bearing_delta_rad=math.radians(adapter.args.backside_registration_max_bearing_delta_deg),
        accepted_range_m=accepted_range_m, now_sec=now, max_scan_age_sec=adapter.args.max_sensor_age_sec)
    search = current_scan_qr_search(**options)
    search_attempt = search[0]
    if require_target_reconciliation and target_reconciliation is None:
        search = (None, {**search[1], 'accepted': False,
            'reason': 'retained_target_reconciliation_pending'})
    support_diagnostics = {}
    support_options = dict(attempt=search[0],
        intrinsics=intrinsics, model_profile=adapter.stand_model_profile,
        scan_from_camera=scan_from_camera, scan=scan, image_stamp_sec=image_stamp_sec,
        now_sec=now, map_bearing_rad=map_bearing_rad,
        cone_half_angle_rad=options['cone_half_angle_rad'], accepted_range_m=accepted_range_m,
        max_scan_age_sec=adapter.args.max_sensor_age_sec,
        max_camera_map_bearing_delta_rad=options['max_camera_map_bearing_delta_rad'],
        diagnostics=support_diagnostics,
        target_reconciliation=target_reconciliation, fragmentation=fragmentation)
    support = (support_opposite_head_region(endpoint_region,
            image_shape=frame.shape[:2], **support_options)
        if endpoint_region is not None else
        detect_opposite_head_support(frame, adapter.cv2, **support_options,
            resources=getattr(adapter, '_qr_decoder_options', {}).get('resources'),
            max_elapsed_sec=min(.06, adapter.args.max_sensor_age_sec-max(0., now-min(image_stamp_sec, scan.scan_stamp_sec))-.08)))
    if endpoint_region is not None and support is None:
        # This branch was certified by a particular current head. Do not
        # replace a failed ray binding with a larger rectangular identity crop.
        search = (None, {**search[1], 'accepted': False,
            'reason': 'endpoint_region_support_unavailable'})
    attempt, crop = exclusive_identity_crop(candidate_uid=adapter.args.stand_id,
        snapshot=context.snapshot, support=support, search_result=search, **options)
    metadata = dict(policy='opposite_identity_only', retained_backside_orientation=orientation,
                    identity_crop=crop, current_angle_refit=False,
                    target_support_diagnostics=support_diagnostics,
                    endpoint_confirmation=endpoint_diagnostics,
                    target_reconciliation_status=getattr(getattr(adapter, '_target_reconciliation', None), 'metadata', {}))
    observations = ()
    if attempt is not None:
        roi = attempt.roi
        # Leave the same publication reserve as ordinary acquisition. The
        # decoder's cooperative budget is followed by fresh source admission.
        now = adapter.node.get_clock().now().nanoseconds / 1e9
        remaining = min(.12, adapter.args.max_sensor_age_sec-max(0., now-min(image_stamp_sec, scan.scan_stamp_sec))-.05)
        if remaining > 0:
            pixels = (masked_identity_pixels(frame, adapter.cv2, roi=roi,
                        corners_px=support.corners_px)
                      if crop.get('sampling') == 'masked_current_head_region'
                      else frame[roi.y0:roi.y1, roi.x0:roi.x1])
            observations = detect_qr_observations_bgr(pixels, adapter.cv2,
                max_elapsed_sec=remaining, diagnostics=metadata.setdefault('decoder', {}),
                **{**getattr(adapter, '_qr_decoder_options', {}), 'identity_only': True},
                preferred_scale=4)
    else:
        # Observation is allowed before current head/target proof is complete.
        # A payload from this search remains provisional until its own actual
        # symbol pixels pass the ordinary current ray and neighbor checks.
        now = adapter.node.get_clock().now().nanoseconds / 1e9
        if search_attempt is None:
            search_attempt = retained_opposite_qr_search(orientation=orientation,
                camera_from_map=camera_from_map, intrinsics=intrinsics,
                model_profile=adapter.stand_model_profile,
                image_stamp_sec=image_stamp_sec, now_sec=now,
                max_scan_age_sec=adapter.args.max_sensor_age_sec)
        provisional = metadata['provisional_qr_search'] = dict(
            attempted=False, accepted=False, supplies_angle=False,
            reason='search_unavailable', motion_authorized=False)
        remaining = min(.12, adapter.args.max_sensor_age_sec-max(0., now-image_stamp_sec)-.05)
        if search_attempt is not None and remaining > 0:
            roi = search_attempt.roi
            provisional.update(attempted=True, search=search_attempt.metadata())
            observations = detect_qr_observations_bgr(
                frame[roi.y0:roi.y1, roi.x0:roi.x1], adapter.cv2,
                max_elapsed_sec=remaining, diagnostics=metadata.setdefault('decoder', {}),
                **{**getattr(adapter, '_qr_decoder_options', {}), 'identity_only': False},
                preferred_scale=4)
            options['now_sec'] = adapter.node.get_clock().now().nanoseconds / 1e9
            decoded_attempt, decoded_crop, decoded_support = decoded_symbol_identity_crop(
                observations, search_attempt=search_attempt, image_shape=frame.shape[:2],
                candidate_uid=adapter.args.stand_id, snapshot=context.snapshot,
                scan_from_camera=scan_from_camera, diagnostics=provisional, **options)
            if decoded_attempt is not None:
                attempt, crop, support = decoded_attempt, decoded_crop, decoded_support
                metadata['identity_crop'] = crop
    binding = bind_crop_text(observations, crop)
    texts = tuple(sorted({o.text for o in observations}))
    now = adapter.node.get_clock().now().nanoseconds / 1e9
    metadata['decoded_qr_target_binding'] = binding.metadata()
    if getattr(adapter, '_capture_pending', None) is not None:
        adapter._capture_pending['detector_metadata'] = metadata
    if attempt is not None:
        adapter._pending_qr_observation_pose = prepare_qr_observation_pose(
            qr_binding=binding, qr_observations=observations, observed_qr_texts=texts,
            image_stamp_sec=image_stamp_sec, scan_stamp_sec=scan.scan_stamp_sec,
            robot_pose=robot_pose, target_key=adapter._target_evidence_key(),
            camera_signature=camera_signature, image_shape=frame.shape, roi=attempt.roi,
            model_profile_sha256=adapter.stand_model_profile.sha256, metadata=metadata,
            retained_backside_orientation=orientation, arrival_target_reconciliation=target_reconciliation)
    if (support is not None and centering_observation_requested(adapter.args)
            and (not binding.accepted
                 or getattr(adapter.args, 'observation_not_before_sec', None) is not None)):
        try:
            odom_pose = pose2d_from_transform(adapter._lookup(
                adapter.profile.odom_frame, adapter.profile.base_frame, image_stamp))
        except adapter.TransformException:
            odom_pose = None
        adapter._pending_candidate_centering = prepare_candidate_centering(
            crop=SimpleNamespace(accepted=True), association=support,
            image_stamp_sec=image_stamp_sec, scan_stamp_sec=scan.scan_stamp_sec,
            target_key=adapter._target_evidence_key(), robot_pose=robot_pose, odom_pose=odom_pose,
            intrinsics=intrinsics, scan_from_camera=scan_from_camera,
            base_from_camera=base_from_camera, metadata=metadata,
            allow_advisory=not binding.accepted)
    update = adapter._record_observation_frame(robot_pose=robot_pose, image_stamp_sec=image_stamp_sec,
        scan_stamp_sec=scan.scan_stamp_sec, observed_at_sec=now,
        lidar_associated=(support is not None or crop.get('accepted') is True
            or target_reconciliation is not None), axis_yaw_rad=None, axis_source=None,
        qr_texts=binding.qr_texts_for_evidence, qr_symbol_count=binding.symbol_count)
    state = 'opposite_identity_collecting' if crop.get('accepted') else 'opposite_identity_crop_conflict'
    if (metadata.get('candidate_centering') or {}).get('reason') == 'centering_budget_exceeded':
        state = 'candidate_centering_budget_exceeded'
    from scripts.aufgabe04.real_robot.observer.opposite_identity_opportunity import OppositeIdentityOpportunity
    if not hasattr(adapter, '_opposite_identity_opportunity'):
        adapter._opposite_identity_opportunity = OppositeIdentityOpportunity()
    adapter._opposite_identity_failure = adapter._opposite_identity_opportunity.observe(
        target_key=adapter._target_evidence_key(), epoch=getattr(update.snapshot, 'motion_epoch', 0),
        pose=(robot_pose.x_m, robot_pose.y_m, robot_pose.yaw_rad),
        image_stamp_sec=image_stamp_sec, scan_stamp_sec=scan.scan_stamp_sec, now_sec=now,
        poisoned=getattr(update.snapshot, 'poisoned', True),
        motion_epoch_reset=getattr(update, 'motion_epoch_reset', False),
        conflict=not crop.get('accepted') and not binding.accepted)
    adapter._write_status(state, qr_texts=list(texts),
        stand_axis_debug=metadata, observation_evidence=update.snapshot.as_dict())
