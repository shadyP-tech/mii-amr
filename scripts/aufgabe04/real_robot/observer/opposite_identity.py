"""Stopped opposite-side identity acquisition without another head-angle fit."""
import math

from scripts.aufgabe04.artifacts.retained_backside_orientation import opposite_view_matches
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr
from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import exclusive_identity_crop, bind_crop_text
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import prepare_qr_observation_pose


def process_opposite_identity(adapter, *, context, frame, intrinsics, robot_pose,
        camera_signature, image_stamp_sec, scan, scan_from_map, camera_from_map,
        map_bearing_rad, accepted_range_m):
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
    attempt, crop = exclusive_identity_crop(candidate_uid=adapter.args.stand_id,
        snapshot=context.snapshot, scan=scan, scan_from_map=scan_from_map,
        camera_from_map=camera_from_map, intrinsics=intrinsics,
        model_profile=adapter.stand_model_profile, image_stamp_sec=image_stamp_sec,
        sync_tolerance_sec=adapter.args.sync_tolerance_sec,
        map_bearing_rad=map_bearing_rad,
        cone_half_angle_rad=math.radians(adapter.args.lidar_cone_half_angle_deg),
        max_camera_map_bearing_delta_rad=math.radians(adapter.args.backside_registration_max_bearing_delta_deg),
        accepted_range_m=accepted_range_m, now_sec=now, max_scan_age_sec=adapter.args.max_sensor_age_sec)
    metadata = dict(policy='opposite_identity_only', retained_backside_orientation=orientation,
                    identity_crop=crop, current_angle_refit=False)
    observations = ()
    if attempt is not None:
        roi = attempt.roi
        # Leave the same publication reserve as ordinary acquisition. The
        # decoder's cooperative budget is followed by fresh source admission.
        remaining = min(.12, adapter.args.max_sensor_age_sec-max(0., now-min(image_stamp_sec, scan.scan_stamp_sec))-.05)
        if remaining > 0:
            observations = detect_qr_observations_bgr(frame[roi.y0:roi.y1, roi.x0:roi.x1], adapter.cv2,
                max_elapsed_sec=remaining, diagnostics=metadata.setdefault('decoder', {}),
                **getattr(adapter, '_qr_decoder_options', {}), preferred_scale=4)
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
    update = adapter._record_observation_frame(robot_pose=robot_pose, image_stamp_sec=image_stamp_sec,
        scan_stamp_sec=scan.scan_stamp_sec, observed_at_sec=now,
        lidar_associated=crop.get('accepted') is True, axis_yaw_rad=None, axis_source=None,
        qr_texts=binding.qr_texts_for_evidence, qr_symbol_count=binding.symbol_count)
    adapter._write_status('opposite_identity_collecting', qr_texts=list(texts),
        stand_axis_debug=metadata, observation_evidence=update.snapshot.as_dict())
