"""Center evidence may finish at arrival without replacing the retained angle."""
from scripts.aufgabe04.artifacts.current_target_estimate import current_target_estimate


def validate_arrival_center(qr):
    from scripts.aufgabe04.real_robot.observer.target_reconciliation import validate_reconciliation
    proof = qr['arrival_target_reconciliation']
    orientation = qr.get('retained_backside_orientation')
    validate_reconciliation(proof, candidate_uid=qr['candidate_uid'],
        stand_center=tuple(qr['stand_center'][k] for k in ('x_m','y_m')),
        image_stamp_sec=qr['sensor_stamp_sec'], scan_stamp_sec=qr['scan_stamp_sec'])
    if (orientation is None or proof.get('retained_orientation') != orientation
            or proof['target_key'] != qr['target_key'] or proof['epoch'] != qr['motion_epoch']
            or proof['planning_frame'] != qr['planning_frame']
            or tuple(proof['entries'][-1]['robot_pose']) != tuple(qr['robot_pose'][k] for k in ('x_m','y_m','yaw_rad'))):
        raise ValueError('arrival center differs from retained candidate, pose or epoch')
    cluster = qr['qr_binding']['association']
    cluster = cluster.get('search_association', cluster)
    _, envelope, _, _ = validate_reconciliation(proof)
    if not set(cluster['selected_cluster_source_indices']).issubset(envelope.selected_cluster_source_indices):
        raise ValueError('arrival center differs from current QR cluster')
    return current_target_estimate(proof)


def retained_facing_center(qr):
    orientation = qr.get('retained_backside_orientation') or {}
    center = orientation.get('validated_target_center')
    if center is not None:
        return center
    if qr.get('arrival_target_reconciliation') is not None:
        return validate_arrival_center(qr)
    raise ValueError('retained facing requires a validated source or arrival center')
