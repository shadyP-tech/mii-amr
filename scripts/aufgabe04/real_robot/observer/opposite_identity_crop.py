"""Current scan crop for text-only identity, with neighboring stand exclusion.

A broad search ROI is not enough. The identity crop is reduced to the current
projected head vicinity and clipped away from every other candidate's projected
head volume. If that leaves no useful target area, no identity is admitted.
"""
from dataclasses import replace
import math

from scripts.aufgabe04.perception.stand_axis_handoff.geometry import transform_point
from scripts.aufgabe04.real_robot.configuration.geometry import project_optical_point
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search
from scripts.aufgabe04.real_robot.observer.qr_target_binding import QrTargetBinding

POLICY = "opposite_current_scan_exclusive_identity_crop"


def exclusive_identity_crop(*, candidate_uid, snapshot, camera_from_map, intrinsics,
                            model_profile, **search_options):
    attempt, search = current_scan_qr_search(camera_from_map=camera_from_map,
        intrinsics=intrinsics, model_profile=model_profile, **search_options)
    info = dict(policy=POLICY, accepted=False, search=search, motion_authorized=False,
                candidate_uid=candidate_uid, image_stamp_sec=search_options['image_stamp_sec'])
    if attempt is None:
        return None, {**info, 'reason': search['reason']}
    # The search box is intentionally wider than an identity box. Keep a
    # bounded 1.6-head-size region, then exclude projected neighboring heads.
    cx, cy = attempt.expected_center_u_px, attempt.expected_center_v_px
    half = .8*attempt.expected_head_height_px
    box = [max(0, math.floor(cx-half)), max(0, math.floor(cy-half)),
           min(intrinsics.width_px, math.ceil(cx+half)), min(intrinsics.height_px, math.ceil(cy+half))]
    competitors = []
    for candidate in snapshot.candidates:
        if candidate.candidate_uid == candidate_uid:
            continue
        g = candidate.geometry
        point = transform_point((g.x_m, g.y_m, model_profile.head_center_height_m), camera_from_map)
        if point[2] <= 0:
            continue
        projected = project_optical_point(point, intrinsics, physical_size_m=model_profile.head_width_m)
        # Include mapped uncertainty and a raw-pixel border margin. A nearer
        # neighboring stand cannot be ignored merely because its QR is absent.
        margin = max(4., intrinsics.fx_px*g.uncertainty_m/point[2])
        hx = intrinsics.fx_px*model_profile.head_width_m/(2*point[2])+margin
        hy = intrinsics.fy_px*model_profile.head_height_m/(2*point[2])+margin
        other = [math.floor(projected.u_px-hx), math.floor(projected.v_px-hy),
                 math.ceil(projected.u_px+hx), math.ceil(projected.v_px+hy)]
        competitors.append(dict(candidate_uid=candidate.candidate_uid, bounds_xyxy=other))
        if box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]:
            choices = ([box[0], box[1], min(box[2], other[0]), box[3]],
                       [max(box[0], other[2]), box[1], box[2], box[3]],
                       [box[0], box[1], box[2], min(box[3], other[1])],
                       [box[0], max(box[1], other[3]), box[2], box[3]])
            choices = [b for b in choices if b[0] < cx < b[2] and b[1] < cy < b[3]]
            if not choices:
                return None, {**info, 'reason': 'neighbor_overlaps_target_center', 'competitors': competitors}
            box = max(choices, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    if min(box[2]-box[0], box[3]-box[1]) < .6*attempt.expected_head_height_px:
        return None, {**info, 'reason': 'exclusive_identity_crop_too_small', 'competitors': competitors}
    roi = replace(attempt.roi, x0=box[0], y0=box[1], x1=box[2], y1=box[3])
    attempt = replace(attempt, roi=roi, source=POLICY, padding_scale=1.6)
    return attempt, {**info, 'accepted': True, 'reason': 'exclusive_current_target_crop',
        'bounds_xyxy': box, 'competitors': competitors, 'target_center_px': [cx, cy],
        'snapshot_id': snapshot.snapshot_id, 'scan_stamp_sec': search_options['scan'].scan_stamp_sec}


def bind_crop_text(observations, crop_evidence):
    observations = tuple(observations or ())
    if len(observations) != 1:
        return QrTargetBinding(False, 'multiple_qr_symbols' if observations else 'no_decoded_qr_identity',
                               symbol_count=len(observations))
    text = observations[0].text
    if not isinstance(text, str) or not text.strip() or crop_evidence.get('accepted') is not True:
        return QrTargetBinding(False, 'exclusive_target_crop_required', symbol_count=1)
    return QrTargetBinding(True, 'decoded_qr_exclusive_opposite_crop', (text,), 1,
        association=crop_evidence['search']['envelope'], current_head_binding=crop_evidence)
