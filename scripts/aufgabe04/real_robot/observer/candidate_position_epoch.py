"""Bounded recovery of an older survey position from current stopped scans.

A localization displacement supplies a search hypothesis, never a replacement
landmark. Normal association remains unchanged. Every receipt replays its raw
scan and excludes competitors at both recorded position hypotheses.
"""
from dataclasses import asdict, replace
import math
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, payload_sha256
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import candidate_frame_reprojection_result_from_mapping
from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector, transform_point
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256

MAX_DISPLACEMENT_M = .35
MAX_RECOVERY_BEARING_RAD = math.radians(35.)
RECOVERY_STEP_RAD = math.radians(30.)
RECOVERY_TRAVEL_RAD = math.radians(42.)  # one coarse arrival, then two fine turns
RECOVERY_TURNS = 3


def epoch_reference(path):
    data = load_content_hashed_json(Path(path), hash_field='candidate_frame_projection_sha256')
    return dict(path=str(Path(path).resolve()), sha256=payload_sha256(data))


def load_epoch(reference, snapshot, uid, *, now, robot_pose):
    data = load_content_hashed_json(Path(reference['path']), hash_field='candidate_frame_projection_sha256')
    if (payload_sha256(data) != reference['sha256']
            or data['projected_candidate_snapshot_sha256'] != candidate_snapshot_sha256(snapshot)
            or data.get('motion_authorized') is not False):
        raise ValueError('candidate position epoch binding changed')
    frame = data['planning_frame_admission']
    if frame['map_frame'] != snapshot.planning_frame:
        raise ValueError('candidate position epoch frame changed')
    captured = frame['pose_provenance']['odom_pose_capture']['capture_time_sec']
    if not math.isfinite(captured) or not 0 <= now-captured <= 30.:
        raise ValueError('candidate position epoch expired')
    pose = frame['current_pose']
    if (math.dist(robot_pose[:2], (pose['x_m'], pose['y_m'])) > .03
            or abs(math.remainder(robot_pose[2]-pose['yaw_rad'], math.tau)) > math.radians(2.)):
        raise ValueError('robot moved since candidate position epoch')
    rows = {}
    for candidate in snapshot.candidates:
        row = candidate_frame_reprojection_result_from_mapping(data['candidate_reprojections'][candidate.candidate_uid])
        if (asdict(row.current_map_from_odom) != frame["map_from_odom"]
                or row.provenance.map_frame != frame["map_frame"]
                or row.provenance.odom_frame != frame["odom_frame"]):
            raise ValueError("candidate projection uses another localization epoch")
        point = row.current_map_point
        if math.dist((point.x_m, point.y_m), (candidate.geometry.x_m, candidate.geometry.y_m)) > 1e-6:
            raise ValueError('candidate position epoch snapshot geometry changed')
        rows[candidate.candidate_uid] = row
    row = rows[uid]
    old = row.provenance.frozen_map_point
    current = row.current_map_point
    displacement = math.dist((old.x_m, old.y_m), (current.x_m, current.y_m))
    if not .08 <= displacement <= MAX_DISPLACEMENT_M:
        raise ValueError('candidate epoch displacement outside recovery bound')
    return rows, displacement


def epoch_cluster(entry, snapshot, uid, scan, scan_from_map):
    rows, displacement = load_epoch(entry['position_epoch'], snapshot, uid,
                                    now=entry['checked_at_sec'], robot_pose=entry['robot_pose'])
    g = snapshot.candidate_for(uid).geometry
    options = entry['options']
    common = dict(map_bearing_rad=options['map_bearing_rad'],
        accepted_range_m=options['accepted_range_m'], now_sec=entry['checked_at_sec'],
        max_scan_age_sec=.5, min_cluster_sample_count=1)
    ordinary = associate_candidate_lidar_target(scan, cone_half_angle_rad=math.radians(15.), **common)
    if ordinary.eligible_cluster_count:
        raise ValueError('ordinary candidate envelope must be empty for epoch recovery')
    # Reproject both authenticated hypotheses into this exact scan frame.
    # Preserve the original surface offsets rather than inventing a larger
    # global tolerance. Searching their enclosing interval also counts returns
    # in any gap between them, so separate hypotheses cannot hide a competitor.
    old = rows[uid].provenance.frozen_map_point
    current_point = transform_point((g.x_m, g.y_m, 0.), scan_from_map)
    frozen_point = transform_point((old.x_m, old.y_m, 0.), scan_from_map)
    delta = math.hypot(*frozen_point[:2])-math.hypot(*current_point[:2])
    lower, upper = options['accepted_range_m']
    recovery_range = (max(0., min(lower, lower+delta)), max(upper, upper+delta))
    envelope = associate_candidate_lidar_target(scan, cone_half_angle_rad=MAX_RECOVERY_BEARING_RAD,
        **{**common, 'accepted_range_m': recovery_range})
    if not envelope.eligible_cluster_count:
        raise ValueError('epoch recovery has no cluster in position hypotheses')
    if envelope.eligible_cluster_count > 1:
        raise ValueError('epoch recovery has competing clusters in position hypotheses')
    if not envelope.associated or envelope.selected_cluster_sample_count < 3:
        raise ValueError('epoch recovery requires one unique three-beam cluster')
    points = [(scan.ranges[i]*math.cos(scan.angle_min+i*scan.angle_increment),
               scan.ranges[i]*math.sin(scan.angle_min+i*scan.angle_increment), 0.)
              for i in envelope.selected_cluster_source_indices]
    compact_limit = min(.16, 2*(g.radius_m+g.uncertainty_m))
    if max(math.dist(a,b) for a in points for b in points) > compact_limit:
        raise ValueError('epoch recovery cluster is not compact')
    mean = tuple(sum(p[k] for p in points)/len(points) for k in range(3))
    q = scan_from_map.rotation_xyzw
    world = rotate_vector(tuple(a-b for a,b in zip(mean,scan_from_map.translation_xyz_m)),
                          (-q[0],-q[1],-q[2],q[3]))[:2]
    # Require agreement with the old map hypothesis AND a capped displacement
    # from the current one. Localization change alone never grants identity.
    if (math.dist(world,(old.x_m,old.y_m)) > g.radius_m+g.uncertainty_m+.05
            or math.dist(world,(g.x_m,g.y_m)) > min(MAX_DISPLACEMENT_M,displacement+g.radius_m+g.uncertainty_m)):
        raise ValueError('cluster outside candidate epoch hypotheses')
    for candidate in snapshot.candidates:
        if candidate.candidate_uid == uid:
            continue
        other = candidate.geometry
        old_other = rows[candidate.candidate_uid].provenance.frozen_map_point
        exclusion = MAX_DISPLACEMENT_M + 2*(other.radius_m+other.uncertainty_m)
        if min(math.dist(world,p) for p in ((other.x_m,other.y_m),(old_other.x_m,old_other.y_m))) <= exclusion:
            raise ValueError('another candidate can explain epoch recovery target')
    # The broad union has established uniqueness. Keep the surviving model's
    # tighter interval for calibrated finite-distance rays; using the entire
    # localization displacement as optical depth uncertainty would regress
    # valid angular-only recovery. Never trim a cluster to fit a hypothesis.
    distances = tuple(scan.ranges[i] for i in envelope.selected_cluster_source_indices)
    frozen_range = (max(0., lower+delta), upper+delta)
    for interval in (tuple(options['accepted_range_m']), frozen_range):
        if all(interval[0] <= distance <= interval[1] for distance in distances):
            envelope = associate_candidate_lidar_target(scan,
                cone_half_angle_rad=MAX_RECOVERY_BEARING_RAD,
                **{**common, 'accepted_range_m': interval})
            break
    else:
        raise ValueError('epoch recovery cluster spans incompatible range hypotheses')
    return scan, envelope, world, math.atan2(mean[1],mean[0])


def is_epoch_recovery(proof):
    return isinstance(proof,dict) and len(proof.get('entries',())) == 3 and all(
        isinstance(e.get('position_epoch'),dict) for e in proof['entries'])


def check_current_scan(proof_scan, scan):
    """A shared timestamp is insufficient to bind a replayed scan."""
    fields = ('scan_frame_id', 'scan_stamp_sec', 'receipt_sec', 'angle_min', 'angle_max',
              'angle_increment', 'range_min', 'range_max', 'scan_topology_profile')
    if (any(getattr(proof_scan, k) != getattr(scan, k) for k in fields)
            or len(proof_scan.ranges) != len(scan.ranges)
            or any(a != b and not (not math.isfinite(a) and not math.isfinite(b))
                   for a, b in zip(proof_scan.ranges, scan.ranges))):
        raise ValueError('reconciliation scan differs from current observation')


def validated_reconciliation_envelope(proof, *, scan, map_bearing_rad, accepted_range_m,
                                      image_stamp_sec=None):
    """Replay recovery while binding its unmodified source candidate options.

    The returned range is recomputed from authenticated position hypotheses;
    callers must never overwrite the source options to make a proof match.
    The fourth result is the measured cluster bearing for the narrow ray only.
    """
    from scripts.aufgabe04.real_robot.observer.target_reconciliation import validate_reconciliation
    result = validate_reconciliation(proof, image_stamp_sec=image_stamp_sec,
                                     scan_stamp_sec=scan.scan_stamp_sec)
    proof_scan, envelope, _, _ = result
    check_current_scan(proof_scan, scan)
    original = proof['entries'][-1]['options']
    if (tuple(original['accepted_range_m']) != tuple(accepted_range_m)
            or abs(original['map_bearing_rad']-map_bearing_rad) > 1e-9
            or abs(envelope.map_bearing_rad-map_bearing_rad) > 1e-9
            or not is_epoch_recovery(proof)
                and tuple(envelope.accepted_range_m) != tuple(accepted_range_m)):
        raise ValueError('reconciliation differs from original candidate envelope')
    return result


def recovered_search(proof, *, scan, original_projection, camera_from_map, intrinsics, model_profile):
    from scripts.aufgabe04.real_robot.observer.target_reconciliation import validate_reconciliation
    from scripts.aufgabe04.real_robot.observer.stopped_target_search import StoppedTargetSearch
    from scripts.aufgabe04.real_robot.configuration.geometry import project_optical_point
    proof_scan, envelope, world, _ = validate_reconciliation(proof,scan_stamp_sec=scan.scan_stamp_sec)
    check_current_scan(proof_scan, scan)
    point = transform_point((*world,model_profile.head_top_height_m-model_profile.head_height_m/2),camera_from_map)
    projection = project_optical_point(point,intrinsics,physical_size_m=max(model_profile.head_height_m,model_profile.head_width_m))
    if not projection.inside_image or projection.depth_m <= 0:
        raise ValueError('recovered candidate head projects outside camera')
    return StoppedTargetSearch(original_projection,projection,tuple(proof['stand_center']),tuple(world),
        envelope.map_bearing_rad,scan,tuple(envelope.selected_cluster_source_indices),
        proof['entries'][-1]['image_stamp_sec'],math.dist(world,proof['stand_center'])), envelope


def epoch_requested_turn(advisory, search_association):
    """Largest bounded coarse step that preserves the existing scan-edge veto.

    Stop short of the optical center if centering would put the target on a
    malformed scan seam. A new stopped capture must decide the next action.
    """
    from scripts.aufgabe04.real_robot.observer.inspection_framing import review_centering_destination
    size=min(abs(advisory.required_yaw_rad),RECOVERY_STEP_RAD)
    while size > math.radians(.3):
        turn=math.copysign(size,advisory.required_yaw_rad)
        if review_centering_destination(replace(advisory,requested_yaw_rad=turn),
                                       search_association=search_association).allowed:
            return turn
        size-=math.radians(1.)
    raise ValueError('epoch arrival correction has no scan-safe destination')
