"""Bounded stopped scan-to-candidate reconciliation; never rewrites survey geometry.

Three independently fresh, unique current clusters must agree in the admitted
map frame. The original range/envelope bounds and all competing candidates are
rechecked when the proof crosses the discovery/centering process boundary.
"""
from dataclasses import asdict
from pathlib import Path
import math

from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import qr_registration_envelope
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot, candidate_snapshot_sha256

POLICY = 'three_stopped_scans_candidate_reconciliation'
MAX_HISTORY_SEC = 1.5


def load_reconciliation_snapshot(path, *, candidate_uid, planning_frame, center):
    snapshot = load_candidate_snapshot(Path(path))
    target = snapshot.candidate_for(candidate_uid)
    if (snapshot.planning_frame != planning_frame or target is None
            or math.dist(center, (target.geometry.x_m, target.geometry.y_m)) > 1e-6):
        raise ValueError('reconciliation snapshot differs from selected candidate frame')
    return snapshot


def _entry_result(entry, snapshot, uid):
    raw = entry['scan']
    scan = PlainLaserScan(**{**raw, 'ranges': tuple(math.nan if v is None else v for v in raw['ranges'])})
    tf = RigidTransform(**entry['scan_from_map'])
    image, now = entry['image_stamp_sec'], entry['checked_at_sec']
    if (not all(math.isfinite(v) for v in (image, now, scan.scan_stamp_sec, scan.receipt_sec, *entry['robot_pose']))
            or any(not 0 <= now-v <= .5 for v in (image, scan.scan_stamp_sec, scan.receipt_sec))
            or abs(image-scan.scan_stamp_sec) > .1
            or tf.parent_frame != scan.scan_frame_id or tf.child_frame != snapshot.planning_frame):
        raise ValueError('reconciliation sources are stale or frame mismatched')
    g = snapshot.candidate_for(uid).geometry
    options = entry['options']
    # Original scan-origin candidate bearing and range interval cannot be replaced.
    from scripts.aufgabe04.perception.stand_axis_handoff.geometry import transform_point
    point = transform_point((g.x_m, g.y_m, 0.), tf)
    if abs(math.remainder(math.atan2(point[1],point[0])-options['map_bearing_rad'], math.tau)) > 1e-6:
        raise ValueError('reconciliation original bearing differs from candidate')
    if (not 0 < options['cone_half_angle_rad'] <= math.radians(3)+1e-9
            or not 0 < options['max_camera_map_bearing_delta_rad'] <= math.radians(12)+1e-9):
        raise ValueError('reconciliation cone exceeds bound')
    lo, hi = options['accepted_range_m']
    distance = math.hypot(*point[:2])
    tolerance = hi-distance
    if (not 0 < lo < hi or not 0 <= tolerance <= .05+1e-9
            or abs(lo-(distance-2*g.radius_m-g.uncertainty_m-tolerance)) > 1e-6):
        raise ValueError('reconciliation range is outside candidate surface envelope')
    envelope = qr_registration_envelope(scan, **options, now_sec=now, max_scan_age_sec=.5)
    ids = envelope.selected_cluster_source_indices
    if not envelope.associated or envelope.eligible_cluster_count != 1 or len(ids) < 3:
        raise ValueError('reconciliation requires one compact three-beam cluster')
    points = [(scan.ranges[i]*math.cos(scan.angle_min+i*scan.angle_increment),
               scan.ranges[i]*math.sin(scan.angle_min+i*scan.angle_increment)) for i in ids]
    x,y = (sum(p[k] for p in points)/len(points) for k in (0,1))
    limit = min(.16, 2*(g.radius_m+g.uncertainty_m))
    if (max(math.dist(a,b) for a in points for b in points) > limit
            or abs(math.remainder(math.atan2(y,x)-options['map_bearing_rad'],math.tau)) > options['max_camera_map_bearing_delta_rad']):
        raise ValueError('reconciliation cluster exceeds original registration bounds')
    qx,qy,qz,qw = tf.rotation_xyzw
    world = rotate_vector(tuple(a-b for a,b in zip((x,y,0.),tf.translation_xyz_m)),(-qx,-qy,-qz,qw))[:2]
    residual = math.dist(world,(g.x_m,g.y_m))
    if residual > limit:
        raise ValueError('reconciliation candidate displacement exceeds bound')
    for candidate in snapshot.candidates:
        if candidate.candidate_uid == uid:
            continue
        other = candidate.geometry
        # Both geometry envelopes must remain disjoint after reconciliation.
        exclusion = limit + 2*(other.radius_m+other.uncertainty_m)
        if math.dist(world,(other.x_m,other.y_m)) <= exclusion:
            raise ValueError('another candidate can explain current target')
    return scan, envelope, world, math.atan2(y,x)


def validate_reconciliation(proof, *, candidate_uid=None, stand_center=None, image_stamp_sec=None, scan_stamp_sec=None):
    if (not isinstance(proof,dict) or proof.get('policy') != POLICY
            or proof.get('candidate_geometry_updated') is not False or proof.get('motion_authorized') is not False):
        raise ValueError('invalid target reconciliation proof')
    uid = proof['candidate_uid']
    snapshot = load_reconciliation_snapshot(proof['snapshot_path'], candidate_uid=uid,
        planning_frame=proof['planning_frame'], center=proof['stand_center'])
    if candidate_snapshot_sha256(snapshot) != proof['snapshot_sha256'] or candidate_uid not in (None,uid):
        raise ValueError('reconciliation snapshot/candidate changed')
    if stand_center is not None and math.dist(stand_center,proof['stand_center']) > 1e-6:
        raise ValueError('reconciliation stand center changed')
    entries = proof['entries']
    if len(entries) != 3:
        raise ValueError('three independent stopped observations required')
    results = [_entry_result(e,snapshot,uid) for e in entries]
    for old,new in zip(entries,entries[1:]):
        if not old['image_stamp_sec'] < new['image_stamp_sec'] or not old['scan']['scan_stamp_sec'] < new['scan']['scan_stamp_sec']:
            raise ValueError('reconciliation reuses or regresses sensor samples')
    if entries[-1]['image_stamp_sec']-entries[0]['image_stamp_sec'] > MAX_HISTORY_SEC:
        raise ValueError('reconciliation history expired')
    anchor = entries[0]['robot_pose']
    for e, result in zip(entries,results):
        if (math.dist(anchor[:2],e['robot_pose'][:2]) > .02
                or abs(math.remainder(anchor[2]-e['robot_pose'][2],math.tau)) > math.radians(2)
                or math.dist(results[0][2],result[2]) > .03):
            raise ValueError('reconciliation target or robot moved')
    if image_stamp_sec not in (None, entries[-1]['image_stamp_sec']) or scan_stamp_sec not in (None, results[-1][0].scan_stamp_sec):
        raise ValueError('reconciliation does not describe current tuple')
    return results[-1]


class StoppedTargetReconciliation:
    def __init__(self):
        self.entries = []
        self.context = None
        self.metadata = {}

    def observe(self, *, snapshot_path, candidate_uid, planning_frame, stand_center,
                target_key, epoch, scan, scan_from_map, robot_pose, image_stamp_sec, now_sec, options):
        context = (str(snapshot_path),candidate_uid,planning_frame,tuple(stand_center),target_key,epoch)
        if context != self.context:
            self.entries = []
            self.context = context
        try:
            snapshot = load_reconciliation_snapshot(snapshot_path,candidate_uid=candidate_uid,
                planning_frame=planning_frame,center=stand_center)
            raw = asdict(scan)
            raw['ranges'] = [v if math.isfinite(v) else None for v in scan.ranges]
            entry = dict(scan=raw,scan_from_map=asdict(scan_from_map),robot_pose=list(robot_pose),
                image_stamp_sec=image_stamp_sec,checked_at_sec=now_sec,options=options)
            _entry_result(entry,snapshot,candidate_uid)
            if self.entries and (image_stamp_sec <= self.entries[-1]['image_stamp_sec'] or scan.scan_stamp_sec <= self.entries[-1]['scan']['scan_stamp_sec']):
                self.entries = []
                raise ValueError('duplicate or regressed reconciliation tuple')
            self.entries = [e for e in self.entries if image_stamp_sec-e['image_stamp_sec'] <= MAX_HISTORY_SEC][-2:]+[entry]
            proof = dict(policy=POLICY,candidate_uid=candidate_uid,planning_frame=planning_frame,
                stand_center=list(stand_center),snapshot_path=str(Path(snapshot_path).resolve()),
                snapshot_sha256=candidate_snapshot_sha256(snapshot),target_key=target_key,epoch=epoch,
                entries=self.entries.copy(),candidate_geometry_updated=False,motion_authorized=False)
            if len(self.entries) < 3:
                self.metadata = dict(ready=False,reason='collecting_stopped_target',sample_count=len(self.entries))
                return None
            validate_reconciliation(proof)
            self.metadata = dict(ready=True,reason='current_target_reconciled',sample_count=3)
            return proof
        except (ValueError,TypeError,KeyError,AttributeError,OSError) as exc:
            self.entries = []
            self.metadata = dict(ready=False,reason=str(exc),sample_count=0)
            return None
