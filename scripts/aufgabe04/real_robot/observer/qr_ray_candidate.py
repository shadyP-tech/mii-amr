"""Bind a ray-resolved target to one frozen candidate, excluding every neighbor."""
import math
from pathlib import Path

from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector, transform_point
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot, candidate_snapshot_sha256


def bind_ray_candidate(range_proof, context):
    if context is None:
        raise ValueError('QR ray resolution requires the frozen candidate population')
    snapshot=load_candidate_snapshot(Path(context['snapshot_path']))
    proof={**context,'snapshot_path':str(Path(context['snapshot_path']).resolve()),
           'snapshot_sha256':candidate_snapshot_sha256(snapshot)}
    validate_ray_candidate(range_proof,proof)
    return proof


def validate_ray_candidate(range_proof, proof):
    snapshot=load_candidate_snapshot(Path(proof['snapshot_path']))
    candidate=snapshot.candidate_for(proof['candidate_uid'])
    if candidate is None or candidate_snapshot_sha256(snapshot)!=proof['snapshot_sha256']:
        raise ValueError('QR ray candidate snapshot changed')
    tf=RigidTransform(**proof['scan_from_map'])
    scan=range_proof['scan'];parameters=range_proof['parameters'];g=candidate.geometry
    if tf.parent_frame!=scan['scan_frame_id'] or tf.child_frame!=snapshot.planning_frame:
        raise ValueError('QR ray candidate frames differ')
    point=transform_point((g.x_m,g.y_m,0.),tf)
    distance=math.hypot(*point[:2]);lo,hi=parameters['accepted_range_m'];tolerance=hi-distance
    if (abs(math.remainder(math.atan2(point[1],point[0])-parameters['map_bearing_rad'],math.tau))>1e-6
            or not 0<=tolerance<=.05+1e-9
            or abs(lo-(distance-2*g.radius_m-g.uncertainty_m-tolerance))>1e-6):
        raise ValueError('QR ray differs from original candidate bearing/range')
    indices=range_proof['support']['search_association']['selected_cluster_source_indices']
    points=[(scan['ranges'][i]*math.cos(scan['angle_min']+i*scan['angle_increment']),
             scan['ranges'][i]*math.sin(scan['angle_min']+i*scan['angle_increment'])) for i in indices]
    x,y=(sum(p[k] for p in points)/len(points) for k in (0,1))
    limit=min(.16,2*(g.radius_m+g.uncertainty_m))
    if max(math.dist(a,b) for a in points for b in points)>limit:
        raise ValueError('QR ray target is not a compact stand')
    qx,qy,qz,qw=tf.rotation_xyzw
    world=rotate_vector(tuple(a-b for a,b in zip((x,y,0.),tf.translation_xyz_m)),(-qx,-qy,-qz,qw))[:2]
    if math.dist(world,(g.x_m,g.y_m))>limit:
        raise ValueError('QR ray target exceeds candidate displacement bound')
    for other in snapshot.candidates:
        if other.candidate_uid!=candidate.candidate_uid and math.dist(world,(other.geometry.x_m,other.geometry.y_m))<=limit+2*(other.geometry.radius_m+other.geometry.uncertainty_m):
            raise ValueError('another candidate can explain the QR ray target')
    return candidate
