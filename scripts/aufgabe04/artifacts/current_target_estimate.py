"""Current metric head position, cross-checked against stopped candidate scans.

The scan centroid is a surface measurement, not an exact stand center. Its
radius, survey uncertainty and observed scatter remain explicit uncertainty.
The immutable survey center continues to bind identity and frame projection.
"""
import math


def current_target_estimate(proof):
    from scripts.aufgabe04.real_robot.observer.target_reconciliation import validate_reconciliation
    from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot
    from pathlib import Path
    _, _, world, _ = validate_reconciliation(proof)
    geometry = load_candidate_snapshot(Path(proof['snapshot_path'])).candidate_for(proof['candidate_uid']).geometry
    return dict(x_m=world[0], y_m=world[1],
        uncertainty_m=geometry.radius_m+geometry.uncertainty_m+.03,
        policy='reconciled_scan_surface_with_center_uncertainty')


def project_current_target(estimate, *, source_center, target_center, yaw_delta):
    if estimate is None:
        return None
    dx, dy = estimate['x_m']-source_center[0], estimate['y_m']-source_center[1]
    c, s = math.cos(yaw_delta), math.sin(yaw_delta)
    return {**estimate, 'x_m': target_center[0]+c*dx-s*dy,
            'y_m': target_center[1]+s*dx+c*dy}


def planning_target_geometry(candidate, estimate):
    """Return a local planning view; never change the candidate snapshot."""
    from dataclasses import replace
    if estimate is None:
        return candidate.geometry
    if (estimate.get('policy') not in ('reconciled_scan_surface_with_center_uncertainty', 'reconciled_metric_head_position_engineering_bound')
            or any(type(estimate.get(k)) not in (int,float) or not math.isfinite(estimate[k])
                   for k in ('x_m','y_m','uncertainty_m'))
            or not 0 < estimate['uncertainty_m'] <= .3
            or math.dist((candidate.geometry.x_m,candidate.geometry.y_m),
                         (estimate['x_m'],estimate['y_m'])) > .16+1e-9):
        raise ValueError('invalid reconciled planning target')
    return replace(candidate.geometry,x_m=estimate['x_m'],y_m=estimate['y_m'],
                   uncertainty_m=estimate['uncertainty_m'])


def measured_target_estimate(proof, evidence, *, model_sha256):
    """Bound all current metric head positions without selecting a new angle."""
    from pathlib import Path
    from scripts.aufgabe04.perception.stand_axis.head_orientation_bounds import (
        CurrentHeadOrientationBounds, HeadOrientationHypothesis, validated_current_head_orientation_bounds)
    from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
    from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
    from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
    from scripts.aufgabe04.perception.stand_axis_handoff.geometry import transform_point, rotate_vector
    from scripts.aufgabe04.real_robot.observer.target_reconciliation import validate_reconciliation
    from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot
    scan,_,surface,reference = validate_reconciliation(proof)
    model=load_measured_physical_stand_model(Path(evidence['model_path']))
    data=dict(evidence['head_bounds'])
    data['hypotheses']=tuple(HeadOrientationHypothesis(**p) for p in data['hypotheses'])
    data['corners']=tuple(ImagePoint(**p) for p in data['corners'])
    bounds=CurrentHeadOrientationBounds(**data)
    if (model.sha256 != model_sha256 or tuple(bounds.head_size_m) != (model.head_width_m,model.head_height_m)
            or not validated_current_head_orientation_bounds(bounds,profile_sha256=model_sha256)):
        raise ValueError('current target lacks validated metric head geometry')
    camera=RigidTransform(**evidence['scan_from_camera'])
    if camera.parent_frame != scan.scan_frame_id:
        raise ValueError('current target camera and scan frames differ')
    tf=RigidTransform(**proof['entries'][-1]['scan_from_map']);qx,qy,qz,qw=tf.rotation_xyzw
    points=[];errors=[]
    for hypothesis in bounds.hypotheses:
        point=transform_point(tuple(v+model.head_depth_m*n/2 for v,n in zip(
            hypothesis.translation_xyz_m,hypothesis.face_normal_xyz)),camera)
        if abs(math.remainder(math.atan2(point[1],point[0])-reference,math.tau))>math.radians(3):
            raise ValueError('metric head center misses reconciled scan')
        world=rotate_vector(tuple(v-t for v,t in zip(point,tf.translation_xyz_m)),(-qx,-qy,-qz,qw))[:2]
        points.append(world)
        # Range uncertainty from pixel scale and metrology; engineering bound,
        # not a claim of calibrated statistical coverage.
        depth=hypothesis.translation_xyz_m[2]
        errors.append(3*hypothesis.corner_sigma_px*depth/bounds.minimum_edge_length_px
            + model.tolerance_m*depth/min(bounds.head_size_m)+model.head_depth_m/2)
    center=tuple(sum(p[k] for p in points)/len(points) for k in (0,1))
    snapshot=load_candidate_snapshot(Path(proof['snapshot_path']));g=snapshot.candidate_for(proof['candidate_uid']).geometry
    uncertainty=max(.02,max(error+math.dist(p,center) for p,error in zip(points,errors)))
    if (uncertainty>.08 or math.dist(center,surface)>g.radius_m+g.uncertainty_m+uncertainty
            or math.dist(center,(g.x_m,g.y_m))>min(.16,2*(g.radius_m+g.uncertainty_m))):
        raise ValueError('metric head center exceeds candidate position bounds')
    for other in snapshot.candidates:
        if other.candidate_uid != proof['candidate_uid'] and math.dist(center,(other.geometry.x_m,other.geometry.y_m)) <= uncertainty+2*(other.geometry.radius_m+other.geometry.uncertainty_m):
            raise ValueError('another candidate explains metric head center')
    return dict(x_m=center[0],y_m=center[1],uncertainty_m=uncertainty,
                policy='reconciled_metric_head_position_engineering_bound')
