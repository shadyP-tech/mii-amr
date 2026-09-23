"""Stopped opposite-side identity acquisition without another head-angle fit."""
import math
from types import SimpleNamespace

from scripts.aufgabe04.artifacts.retained_backside_orientation import opposite_view_matches
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr
from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import exclusive_identity_crop, bind_crop_text
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import prepare_qr_observation_pose
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search
from scripts.aufgabe04.real_robot.observer.opposite_target_support import detect_opposite_target_support
from scripts.aufgabe04.real_robot.observer.candidate_centering_receipt import prepare_candidate_centering
from scripts.aufgabe04.qr_scanning.isolated_qr_views import rectify_isolated_qr_view, ISOLATED_QR_VIEWS
from scripts.aufgabe04.real_robot.configuration.geometry import pose2d_from_transform


def process_opposite_identity(adapter, *, context, frame, intrinsics, robot_pose,
        camera_signature, image_stamp_sec, scan, scan_from_map, camera_from_map,
        map_bearing_rad, accepted_range_m, scan_from_camera, base_from_camera, image_stamp,
        target_reconciliation=None):
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
    options = dict(scan=scan, scan_from_map=scan_from_map,
        camera_from_map=camera_from_map, intrinsics=intrinsics,
        model_profile=adapter.stand_model_profile, image_stamp_sec=image_stamp_sec,
        sync_tolerance_sec=adapter.args.sync_tolerance_sec,
        map_bearing_rad=map_bearing_rad,
        cone_half_angle_rad=math.radians(adapter.args.lidar_cone_half_angle_deg),
        max_camera_map_bearing_delta_rad=math.radians(adapter.args.backside_registration_max_bearing_delta_deg),
        accepted_range_m=accepted_range_m, now_sec=now, max_scan_age_sec=adapter.args.max_sensor_age_sec)
    search = current_scan_qr_search(**options)
    support = detect_opposite_target_support(frame, adapter.cv2, attempt=search[0],
        intrinsics=intrinsics, model_profile=adapter.stand_model_profile,
        scan_from_camera=scan_from_camera, scan=scan, image_stamp_sec=image_stamp_sec,
        now_sec=now, map_bearing_rad=map_bearing_rad,
        cone_half_angle_rad=options['cone_half_angle_rad'], accepted_range_m=accepted_range_m,
        max_scan_age_sec=adapter.args.max_sensor_age_sec,
        max_camera_map_bearing_delta_rad=options['max_camera_map_bearing_delta_rad'],
        resources=getattr(adapter, '_qr_decoder_options', {}).get('resources'),
        target_reconciliation=target_reconciliation,
        max_elapsed_sec=min(.06, adapter.args.max_sensor_age_sec-max(0., now-min(image_stamp_sec, scan.scan_stamp_sec))-.08))
    attempt, crop = exclusive_identity_crop(candidate_uid=adapter.args.stand_id,
        snapshot=context.snapshot, support=support, search_result=search, **options)
    metadata = dict(policy='opposite_identity_only', retained_backside_orientation=orientation,
                    identity_crop=crop, current_angle_refit=False)
    observations = ()
    if attempt is not None:
        roi = attempt.roi
        # Leave the same publication reserve as ordinary acquisition. The
        # decoder's cooperative budget is followed by fresh source admission.
        now = adapter.node.get_clock().now().nanoseconds / 1e9
        remaining = min(.12, adapter.args.max_sensor_age_sec-max(0., now-min(image_stamp_sec, scan.scan_stamp_sec))-.05)
        if remaining > 0:
            pixels = (rectify_isolated_qr_view(frame, support.corners_px, adapter.cv2, ISOLATED_QR_VIEWS[0])
                      if crop.get('sampling') == 'isolated_current_qr_quad' else frame[roi.y0:roi.y1, roi.x0:roi.x1])
            observations = detect_qr_observations_bgr(pixels, adapter.cv2,
                max_elapsed_sec=remaining, diagnostics=metadata.setdefault('decoder', {}),
                **getattr(adapter, '_qr_decoder_options', {}),
                preferred_scale=None if crop.get('sampling') == 'isolated_current_qr_quad' else 4)
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
            retained_backside_orientation=orientation)
    if (support is not None and not binding.accepted
            and getattr(adapter.args, 'candidate_centering_json', None) is not None):
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
            base_from_camera=base_from_camera, metadata=metadata)
    update = adapter._record_observation_frame(robot_pose=robot_pose, image_stamp_sec=image_stamp_sec,
        scan_stamp_sec=scan.scan_stamp_sec, observed_at_sec=now,
        lidar_associated=(support is not None or crop.get('accepted') is True
            or target_reconciliation is not None), axis_yaw_rad=None, axis_source=None,
        qr_texts=binding.qr_texts_for_evidence, qr_symbol_count=binding.symbol_count)
    state = 'opposite_identity_collecting' if crop.get('accepted') else 'opposite_identity_crop_conflict'
    if (metadata.get('candidate_centering') or {}).get('reason') == 'centering_budget_exceeded':
        state = 'candidate_centering_budget_exceeded'
    adapter._write_status(state, qr_texts=list(texts),
        stand_axis_debug=metadata, observation_evidence=update.snapshot.as_dict())
