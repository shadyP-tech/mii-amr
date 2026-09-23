"""Current scan crop for text-only identity, with neighboring stand exclusion.

A broad search ROI is not enough. The identity crop is reduced to the current
projected head vicinity. Overlap requires a complete, current foreground QR
outline and separated depth bounds; the decoder then samples that symbol only.
"""
from dataclasses import replace
import math

from scripts.aufgabe04.perception.stand_axis_handoff.geometry import transform_point
from scripts.aufgabe04.real_robot.configuration.geometry import project_optical_point
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search
from scripts.aufgabe04.real_robot.observer.qr_target_binding import QrTargetBinding

POLICY = "opposite_current_scan_exclusive_identity_crop"


def exclusive_identity_crop(*, candidate_uid, snapshot, camera_from_map, intrinsics,
                            model_profile, support=None, search_result=None, **search_options):
    attempt, search = (current_scan_qr_search(camera_from_map=camera_from_map,
        intrinsics=intrinsics, model_profile=model_profile, **search_options)
        if search_result is None else search_result)
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
    sampling = 'rectangular_crop'
    target_depth_interval = None
    if support is not None:
        from scripts.aufgabe04.real_robot.observer.opposite_target_support import validate_target_support
        validate_target_support(support.metadata())
        proof = support.target_reconciliation
        if proof is not None:
            from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256
            if proof['candidate_uid'] != candidate_uid or proof['snapshot_sha256'] != candidate_snapshot_sha256(snapshot):
                return None, {**info, 'reason': 'outline_candidate_snapshot_mismatch'}
        indices = support.lidar_association.search_association.selected_cluster_source_indices
        if not set(indices).issubset(search['envelope']['selected_cluster_source_indices']):
            return None, {**info, 'reason': 'outline_search_cluster_mismatch'}
        if support.image_stamp_sec != search_options['image_stamp_sec']:
            return None, {**info, 'reason': 'outline_image_stamp_mismatch'}
        cx, cy = support.full_image_center_px
        # Never truncate the target to make a neighbor exclusion rectangle fit.
        box = [math.floor(min(p[0] for p in support.corners_px)),
               math.floor(min(p[1] for p in support.corners_px)),
               math.ceil(max(p[0] for p in support.corners_px))+1,
               math.ceil(max(p[1] for p in support.corners_px))+1]
        target = snapshot.candidate_for(candidate_uid)
        if target is None:
            return None, {**info, 'reason': 'target_missing_from_snapshot'}
        uncertainty = target.geometry.radius_m + target.geometry.uncertainty_m
        target_depth_interval = [max(0., support.depth_m-uncertainty), support.depth_m+uncertainty]
        sampling = 'isolated_current_qr_quad'
    competitors = []
    for candidate in snapshot.candidates:
        if candidate.candidate_uid == candidate_uid:
            continue
        g = candidate.geometry
        point = transform_point((g.x_m, g.y_m, model_profile.head_center_height_m), camera_from_map)
        if point[2] <= 0:
            continue
        projected = project_optical_point(point, intrinsics, physical_size_m=model_profile.head_width_m)
        margin = max(4., intrinsics.fx_px*g.uncertainty_m/point[2])
        hx = intrinsics.fx_px*model_profile.head_width_m/(2*point[2])+margin
        hy = intrinsics.fy_px*model_profile.head_height_m/(2*point[2])+margin
        other = [math.floor(projected.u_px-hx), math.floor(projected.v_px-hy),
                 math.ceil(projected.u_px+hx), math.ceil(projected.v_px+hy)]
        neighbor_depth = [point[2]-g.uncertainty_m-model_profile.head_width_m/2,
                          point[2]+g.uncertainty_m+model_profile.head_width_m/2]
        overlap = box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]
        occluded = bool(overlap and target_depth_interval is not None
                        and neighbor_depth[0] > target_depth_interval[1])
        competitors.append(dict(candidate_uid=candidate.candidate_uid, bounds_xyxy=other,
            depth_interval_m=neighbor_depth, occluded_by_target_symbol=occluded))
        if overlap and not occluded:
            return None, {**info, 'reason': 'target_crop_overlap_unresolved', 'competitors': competitors}
    if not 0 <= box[0] < box[2] <= intrinsics.width_px or not 0 <= box[1] < box[3] <= intrinsics.height_px:
        return None, {**info, 'reason': 'complete_target_outside_image'}
    roi = replace(attempt.roi, x0=box[0], y0=box[1], x1=box[2], y1=box[3])
    attempt = replace(attempt, roi=roi, source=POLICY, padding_scale=1.6)
    return attempt, {**info, 'accepted': True, 'reason': 'exclusive_current_target_crop',
        'bounds_xyxy': box, 'competitors': competitors, 'target_center_px': [cx, cy],
        'snapshot_id': snapshot.snapshot_id, 'scan_stamp_sec': search_options['scan'].scan_stamp_sec,
        'sampling': sampling, 'target_support': None if support is None else support.metadata(),
        'target_depth_interval_m': target_depth_interval}


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
