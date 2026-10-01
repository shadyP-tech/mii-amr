"""Current visual confirmation of a certified opposite target at the scan seam.

The search hint has no association authority. Only one complete current QR
outline spanning both bounded gap endpoints can confirm the retained target.
Original rays, topology and two raw cluster counts remain unchanged.
"""
from dataclasses import asdict, replace
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import copy
import hashlib
import json
import math

from scripts.aufgabe04.artifacts.content_store import content_hashed_payload, payload_sha256
from scripts.aufgabe04.artifacts.retained_backside_orientation import (
    opposite_view_matches, validate_retained_orientation,
)
from scripts.aufgabe04.perception import candidate_lidar_association as lidar
from scripts.aufgabe04.perception.scan_endpoint_fragments import bounded_endpoint_fragments
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import transform_point, rotate_vector
from scripts.aufgabe04.real_robot.configuration.geometry import (
    CameraIntrinsics, project_optical_point, roi_from_projection, validate_intrinsics,
)
from scripts.aufgabe04.real_robot.observer.finite_target_bearing import (
    finite_target_bearing, point_on_scan_range,
)
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot, candidate_snapshot_sha256

KIND = 'retained_opposite_current_qr_endpoint_confirmation'
HINT_KIND = 'retained_opposite_endpoint_search_hint'
HASH_FIELD = 'opposite_endpoint_sha256'
_VALIDATION_SCOPE = ContextVar('opposite_endpoint_validation_scope', default=None)
_FIELDS = {'schema_version', 'kind', 'current', 'snapshot_path', 'snapshot_sha256',
           'model_path', 'model_sha256', 'robot_profile_sha256', 'calibration_profile_sha256',
           'intrinsics', 'scan_from_camera', 'scan_from_map', 'camera_from_map',
           'motion_authorized', 'candidate_geometry_updated', 'supplies_identity', 'supplies_angle'}


def _hashed(value):
    return content_hashed_payload(value, hash_field=HASH_FIELD)


@contextmanager
def opposite_endpoint_validation_scope():
    """Reuse exact proof replay only within one synchronous sensor callback.

    Every use still checks the full proof hash and every external source file.
    The caller's fresh clock gates remain authoritative; no cache survives this
    scope, grants identity, or changes a stored observation's checked time.
    """
    token = _VALIDATION_SCOPE.set({'proofs': {}, 'snapshots': {}})
    try:
        yield
    finally:
        _VALIDATION_SCOPE.reset(token)


def _proof_files(proof):
    from scripts.aufgabe04.artifacts.retained_orientation_cache import _dependencies
    retained = proof['current']['context']['retained_orientation']
    return tuple(sorted(set(_dependencies(Path(retained['path']))) | {
        Path(proof['snapshot_path']), Path(proof['model_path'])}))


def _fingerprints(paths):
    if any(path.is_symlink() for path in paths):
        raise ValueError('opposite endpoint source must not become a symlink')
    return tuple((str(path), hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths)


def load_endpoint_snapshot(path):
    """Read an immutable snapshot once per callback, rechecking its bytes."""
    path = Path(path).absolute()
    if path.is_symlink():
        raise ValueError('opposite endpoint snapshot must not be a symlink')
    path = path.resolve()
    scope = _VALIDATION_SCOPE.get()
    cache = None if scope is None else scope['snapshots']
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    if cache is not None and path in cache:
        expected, snapshot, digest = cache[path]
        if before != expected:
            raise ValueError('opposite endpoint snapshot changed during the current callback')
        return copy.deepcopy(snapshot), digest
    snapshot = load_candidate_snapshot(path)
    digest = candidate_snapshot_sha256(snapshot)
    if hashlib.sha256(path.read_bytes()).hexdigest() != before:
        raise ValueError('opposite endpoint snapshot changed during validation')
    if cache is not None:
        if len(cache) >= 4:
            cache.pop(next(iter(cache)))
        cache[path] = before, copy.deepcopy(snapshot), digest
    return snapshot, digest


def _check_hash(value, *, confirmed):
    expected = _FIELDS | {HASH_FIELD} | ({'outline'} if confirmed else set())
    if (not isinstance(value, dict) or set(value) != expected
            or type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or value.get('kind') != (KIND if confirmed else HINT_KIND)
            or any(value.get(k) is not False for k in
                   ('motion_authorized', 'candidate_geometry_updated', 'supplies_identity', 'supplies_angle'))
            or value.get(HASH_FIELD) != payload_sha256({k: v for k, v in value.items() if k != HASH_FIELD})):
        raise ValueError('invalid opposite endpoint confirmation proof')


def _geometry(value):
    # This function deliberately never calls target reconciliation: the final
    # reconciliation receipt consumes this independently replayable proof.
    from scripts.aufgabe04.real_robot.observer.scan_target_persistence import (
        _read_entry, _xy, _stationary, scan_pose_in_map,
    )
    current = value['current']
    scan, robot, pose, association, clusters = _read_entry(current)
    context, parameters = current['context'], current['parameters']
    now = current['now_sec']
    if (not 0 < current['max_scan_age_sec'] <= .5
            or any(not 0 <= now-stamp <= current['max_scan_age_sec'] for stamp in
                   (context['image_stamp_sec'], scan.scan_stamp_sec, scan.receipt_sec))
            or abs(context['image_stamp_sec']-scan.scan_stamp_sec) > .1):
        raise ValueError('opposite endpoint sources are stale or unsynchronized')
    snapshot, snapshot_digest = load_endpoint_snapshot(value['snapshot_path'])
    if snapshot_digest != value['snapshot_sha256']:
        raise ValueError('opposite endpoint snapshot changed')
    retained = context.get('retained_orientation')
    if not isinstance(retained, dict):
        raise ValueError('opposite endpoint requires a certified retained target')
    candidate = snapshot.candidate_for(retained['candidate_uid'])
    if candidate is None:
        raise ValueError('opposite endpoint candidate is missing')
    g = candidate.geometry
    validate_retained_orientation(retained, candidate_uid=candidate.candidate_uid,
        planning_frame=snapshot.planning_frame, stand_center=dict(x_m=g.x_m, y_m=g.y_m),
        model_sha256=value['model_sha256'])
    center = retained.get('validated_target_center')
    if (center is None or center.get('policy') != 'reconciled_metric_head_position_engineering_bound'
            or not opposite_view_matches(retained, robot)
            or any(context[k] != v for k, v in (('candidate_x_m', g.x_m),
                ('candidate_y_m', g.y_m), ('stand_radius_m', g.radius_m),
                ('stand_uncertainty_m', g.uncertainty_m)))
            or any(value[k] != retained[k] for k in ('robot_profile_sha256', 'calibration_profile_sha256'))):
        raise ValueError('opposite endpoint differs from the certified opposite target')
    model = load_measured_physical_stand_model(Path(value['model_path']))
    if model.sha256 != value['model_sha256']:
        raise ValueError('opposite endpoint stand model changed')
    intrinsics = CameraIntrinsics(**value['intrinsics'])
    validate_intrinsics(intrinsics)
    # validate_retained_orientation above already replayed (or content-checked)
    # this immutable chain. Read its authenticated source bytes without doing
    # a second full metric pose reconstruction inside the live freshness budget.
    projection_bytes = Path(retained['path']).read_bytes()
    if hashlib.sha256(projection_bytes).hexdigest() != retained['file_sha256']:
        raise ValueError('retained projection changed during endpoint confirmation')
    source_ref = json.loads(projection_bytes)['source_axis_observation']
    source_bytes = Path(source_ref['path']).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != source_ref['sha256']:
        raise ValueError('retained source changed during endpoint confirmation')
    head_evidence = json.loads(source_bytes)['head_position_evidence']
    bounds = head_evidence['head_bounds']
    matrix = (intrinsics.fx_px, intrinsics.fy_px, intrinsics.cx_px, intrinsics.cy_px)
    if (tuple(bounds['frame_shape']) != (intrinsics.height_px, intrinsics.width_px)
            or len(bounds['camera_matrix']) != 4
            or any(abs(a-b) > 1e-6 for a, b in zip(matrix, bounds['camera_matrix']))):
        raise ValueError('opposite endpoint intrinsics differ from the certified calibration')
    scan_from_map = RigidTransform(**value['scan_from_map'])
    camera_from_map = RigidTransform(**value['camera_from_map'])
    scan_from_camera = RigidTransform(**value['scan_from_camera'])
    calibrated_camera = RigidTransform(**head_evidence['scan_from_camera'])
    old_q, current_q = calibrated_camera.rotation_xyzw, scan_from_camera.rotation_xyzw
    norms = math.sqrt(sum(v*v for v in old_q))*math.sqrt(sum(v*v for v in current_q))
    rotation_error = math.inf if norms <= 0 else 2*math.acos(min(1., abs(sum(
        a*b for a, b in zip(old_q, current_q))/norms)))
    if (calibrated_camera.parent_frame != scan_from_camera.parent_frame
            or calibrated_camera.child_frame != scan_from_camera.child_frame
            or math.dist(calibrated_camera.translation_xyz_m, scan_from_camera.translation_xyz_m) > .005
            or rotation_error > math.radians(1.)):
        raise ValueError('opposite endpoint camera extrinsics differ from certified calibration')
    if (scan_from_map.parent_frame != scan.scan_frame_id
            or scan_from_map.child_frame != snapshot.planning_frame
            or camera_from_map.child_frame != snapshot.planning_frame
            or scan_from_camera.parent_frame != scan.scan_frame_id
            or scan_from_camera.child_frame != camera_from_map.parent_frame
            or asdict(scan_pose_in_map(scan_from_map.translation_xyz_m,
                scan_from_map.rotation_xyzw)) != context['scan_pose_map']):
        raise ValueError('opposite endpoint transform frames or exact scan pose differ')
    # Camera<-map is at image time; scan<-map is at scan time. Compare their
    # relative planar poses using the existing stopped-tuple contract rather
    # than demanding numerical equality across these different timestamps.
    origin = transform_point(transform_point((0., 0., 0.), camera_from_map), scan_from_camera)
    axes = tuple(tuple(v-t for v, t in zip(transform_point(transform_point(p, camera_from_map),
        scan_from_camera), origin)) for p in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    if (max(abs(axes[0][2]), abs(axes[1][2]), abs(axes[2][0]), abs(axes[2][1])) > 1e-6
            or axes[2][2] < 0):
        raise ValueError('opposite endpoint camera-to-map composition must be planar')
    image_yaw = math.atan2(axes[0][1], axes[0][0])
    image_scan_pose = scan_pose_in_map(origin, (0., 0., math.sin(image_yaw/2), math.cos(image_yaw/2)))
    if not _stationary(pose, image_scan_pose):
        raise ValueError('opposite endpoint camera and scan transforms exceed stopped motion')
    limit = parameters['max_camera_map_bearing_delta_rad']
    if (not 0 < limit <= math.radians(12)+1e-9
            or not 0 < parameters['cone_half_angle_rad']-limit <= math.radians(3)+1e-9
            or parameters['observed_camera_bearing_rad'] != parameters['map_bearing_rad']
            or parameters['min_cluster_sample_count'] != 1
            or association.rejection_reason != 'ambiguous_registered_camera_clusters'
            or len(clusters) != 2
            or not bounded_endpoint_fragments(scan, tuple(tuple(s.index for s in c.samples) for c in clusters))):
        raise ValueError('opposite endpoint requires exactly two bounded raw endpoint groups')
    left, right = sorted(clusters, key=lambda c: c.start_index)
    samples = (*right.samples, *left.samples)
    a, b = right.samples[-1], left.samples[0]
    points = tuple(_xy(s, pose) for s in samples)
    diameter_limit = min(.16, 2*(g.radius_m+g.uncertainty_m))
    scan_center = tuple(sum(s.distance_m*f(s.bearing_rad) for s in samples)/len(samples)
                        for f in (math.cos, math.sin))
    tx, ty, tz = scan_from_map.translation_xyz_m
    qx, qy, qz, qw = scan_from_map.rotation_xyzw
    world = rotate_vector((scan_center[0]-tx, scan_center[1]-ty, -tz), (-qx, -qy, -qz, qw))[:2]
    retained_point = (center['x_m'], center['y_m'])
    if (len(samples) < 3 or abs(a.distance_m-b.distance_m) > parameters['max_range_jump_m']
            or math.dist(_xy(a, pose), _xy(b, pose)) > parameters['max_point_gap_m']
            or max(math.dist(p, q) for p in points for q in points) > diameter_limit
            or any(math.dist(p, retained_point) > g.radius_m+center['uncertainty_m'] for p in points)
            or math.dist(world, (g.x_m, g.y_m)) > diameter_limit):
        raise ValueError('opposite endpoint fragments exceed certified target geometry')
    for other in snapshot.candidates:
        if other.candidate_uid != candidate.candidate_uid and math.dist(
                world, (other.geometry.x_m, other.geometry.y_m)) <= (
                    diameter_limit+2*(other.geometry.radius_m+other.geometry.uncertainty_m)):
            raise ValueError('another candidate can explain opposite endpoint fragments')
    aggregate = lidar._build_cluster(samples)
    search = replace(association.search_association, associated=True, rejection_reason='',
        distance_m=aggregate.distance_m, selected_cluster_sample_count=len(samples),
        selected_cluster_start_index=samples[0].index, selected_cluster_end_index=samples[-1].index,
        selected_cluster_source_indices=tuple(s.index for s in samples),
        selected_cluster_wraps_scan_seam=True, selected_cluster_bearing_rad=aggregate.bearing_rad,
        selected_cluster_bearing_delta_from_map_rad=abs(math.remainder(
            aggregate.bearing_rad-parameters['map_bearing_rad'], math.tau)), selection_source=KIND)
    point = transform_point((*world, model.head_center_height_m), camera_from_map)
    if point[2] <= 0:
        raise ValueError('opposite endpoint hint is behind the camera')
    projection = project_optical_point(point, intrinsics,
        physical_size_m=max(model.head_width_m, model.head_height_m))
    roi = roi_from_projection(projection, intrinsics, padding_scale=2.4)
    if roi is None:
        raise ValueError('opposite endpoint hint is outside the image')
    attempt = HeadRoiAttempt(roi, HINT_KIND, 2.4, projection.u_px, projection.v_px,
                           intrinsics.fy_px*model.head_height_m/point[2])
    return (scan, association, search, attempt, intrinsics, scan_from_camera, model, (a, b))


def build_opposite_endpoint_hint(*, scan, persistence_context, snapshot_path, intrinsics,
        scan_from_camera, camera_from_map, scan_from_map, model_profile, model_path,
        now_sec, max_scan_age_sec, cone_half_angle_rad, max_camera_map_bearing_delta_rad,
        map_bearing_rad, accepted_range_m):
    from scripts.aufgabe04.real_robot.observer.scan_target_persistence import _entry
    retained = persistence_context.retained_orientation
    if not isinstance(retained, dict):
        raise ValueError('opposite endpoint requires a certified retained target')
    association = lidar.associate_camera_registered_candidate_lidar_target(scan,
        map_bearing_rad=map_bearing_rad, observed_camera_bearing_rad=map_bearing_rad,
        cone_half_angle_rad=cone_half_angle_rad+max_camera_map_bearing_delta_rad,
        accepted_range_m=accepted_range_m, now_sec=now_sec, max_scan_age_sec=max_scan_age_sec,
        min_cluster_sample_count=1, max_camera_map_bearing_delta_rad=max_camera_map_bearing_delta_rad)
    snapshot, snapshot_digest = load_endpoint_snapshot(snapshot_path)
    seed = _hashed(dict(schema_version=1, kind=HINT_KIND,
        current=_entry(association, scan, persistence_context, now_sec, max_scan_age_sec),
        snapshot_path=str(Path(snapshot_path).resolve()), snapshot_sha256=snapshot_digest,
        model_path=str(Path(model_path).resolve()), model_sha256=model_profile.sha256,
        robot_profile_sha256=retained['robot_profile_sha256'],
        calibration_profile_sha256=retained['calibration_profile_sha256'],
        intrinsics=asdict(intrinsics), scan_from_camera=asdict(scan_from_camera),
        scan_from_map=asdict(scan_from_map), camera_from_map=asdict(camera_from_map),
        motion_authorized=False, candidate_geometry_updated=False, supplies_identity=False, supplies_angle=False))
    _check_hash(seed, confirmed=False)
    return _geometry(seed)[3], seed


def confirm_opposite_endpoint(seedproof, outline, *, now_sec):
    _check_hash(seedproof, confirmed=False)
    if type(now_sec) not in (int, float) or not math.isfinite(now_sec) or now_sec < seedproof['current']['now_sec']:
        raise ValueError('opposite endpoint confirmation clock regressed')
    value = copy.deepcopy(seedproof)
    value.pop(HASH_FIELD)
    value['kind'] = KIND
    value['current']['now_sec'] = now_sec
    value['outline'] = copy.deepcopy(outline.metadata() if hasattr(outline, 'metadata') else outline)
    proof = _hashed(value)
    validate_opposite_endpoint(proof)
    return proof


def validate_opposite_endpoint(proof):
    _check_hash(proof, confirmed=True)
    state = _VALIDATION_SCOPE.get()
    scope = None if state is None else state['proofs']
    key = proof[HASH_FIELD]
    if scope is not None and key in scope:
        paths, expected, result = scope[key]
        if _fingerprints(paths) != expected:
            raise ValueError('opposite endpoint source changed during the current callback')
        return copy.deepcopy(result)
    paths = _proof_files(proof) if scope is not None else ()
    before = _fingerprints(paths)
    result = _validate_current_outline(proof)
    if scope is not None:
        if _fingerprints(paths) != before:
            raise ValueError('opposite endpoint source changed during proof validation')
        if len(scope) >= 4:
            scope.pop(next(iter(scope)))
        scope[key] = paths, before, copy.deepcopy(result)
    return result


def _validate_current_outline(proof):
    from scripts.aufgabe04.real_robot.observer.opposite_target_support import validate_opposite_qr_outline
    scan, association, search, attempt, intrinsics, camera, model, gap = _geometry(proof)
    outline = proof['outline']
    validate_opposite_qr_outline(outline)
    expected = attempt.expected_head_height_px*model.qr_symbol_height_m/model.head_height_m
    center = tuple(outline['center_px'])
    if (outline['image_stamp_sec'] != proof['current']['context']['image_stamp_sec']
            or tuple(outline['image_shape']) != (intrinsics.height_px, intrinsics.width_px)
            or not math.isclose(outline['expected_symbol_height_px'], expected, rel_tol=1e-9, abs_tol=1e-9)
            or math.dist(center, (attempt.expected_center_u_px, attempt.expected_center_v_px)) > .75*attempt.expected_head_height_px
            or any(not attempt.roi.x0 <= p[0] < attempt.roi.x1 or not attempt.roi.y0 <= p[1] < attempt.roi.y1
                   for p in outline['corners_px'])):
        raise ValueError('opposite endpoint outline differs from its current calibrated hint')
    common = dict(intrinsics=intrinsics, scan_from_camera=camera)
    bearing, uncertainty, _ = finite_target_bearing(center_px=center, **common,
        distance_m=search.distance_m, range_interval_m=search.accepted_range_m)
    if abs(math.remainder(bearing-search.selected_cluster_bearing_rad, math.tau))+uncertainty > math.radians(3)+1e-9:
        raise ValueError('opposite endpoint outline misses the current target ray')
    # At either accepted depth, the same complete symbol must span BOTH
    # missing-interval endpoints. A center-only ray cannot bridge two objects.
    endpoints = tuple(math.remainder(s.bearing_rad-bearing, math.tau) for s in gap)
    for distance in search.accepted_range_m:
        rays = [point_on_scan_range(center_px=p, **common, distance_m=distance)[0]
                for p in outline['corners_px']]
        extent = [math.remainder(math.atan2(p[1], p[0])-bearing, math.tau) for p in rays]
        if min(endpoints) < min(extent)-1e-9 or max(endpoints) > max(extent)+1e-9:
            raise ValueError('current QR outline does not span both scan gap endpoints')
    return replace(association, associated=True, distance_m=search.distance_m, rejection_reason='',
        search_association=search, unique_eligible_cluster_required=False, witnessed_fragmentation=proof)
