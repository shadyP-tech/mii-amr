"""Authenticated common-frame derivation; original retained receipts stay intact."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import math

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, payload_sha256
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import map_pose_to_odom, odom_pose_to_map
from scripts.aufgabe04.artifacts.bounded_orientation import validated_bounded_orientation

POLICY='projected_retained_backside_current_qr_facing'
SCHEMA_VERSION=5


def _sources(evidence):
    from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import load_recommendation
    from scripts.aufgabe04.artifacts.retained_facing import validate_retained_facing
    if not isinstance(evidence,dict) or evidence.get('policy')!=POLICY:
        raise ValueError('invalid retained projection policy')
    path=Path(evidence['source_recommendation_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=evidence['source_recommendation_sha256']:
        raise ValueError('retained projection source recommendation changed')
    if json.loads(path.read_text()).get("schema_version") != 4:
        raise ValueError("retained projection must reference an original schema-4 receipt")
    original=load_recommendation(path)
    if original.schema_version!=4:
        raise ValueError('retained projection must reference an original schema-4 receipt')
    qr=validate_retained_facing(original)
    from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import load_backside_axis_frame_projection
    retained_projection=load_backside_axis_frame_projection(Path(qr['retained_backside_orientation']['path']))
    if retained_projection.target_candidate_projection_sha256 != evidence['source_projection_sha256']:
        raise ValueError('retained source frame differs from certified QR arrival frame')
    frames=[]
    certificates=[]
    candidates=[]
    from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot, candidate_snapshot_sha256
    from scripts.aufgabe04.navigation.foundation.models import Pose2D
    for prefix in ('source','target'):
        payload=load_content_hashed_json(Path(evidence[prefix+'_projection_path']),hash_field='candidate_frame_projection_sha256')
        if payload_sha256(payload)!=evidence[prefix+'_projection_sha256']:
            raise ValueError('retained projection certificate changed')
        if payload.get('motion_authorized') is not False:
            raise ValueError('projection cannot authorize motion')
        snapshot=load_candidate_snapshot(Path(payload['projected_candidate_snapshot_path']))
        if candidate_snapshot_sha256(snapshot)!=payload['projected_candidate_snapshot_sha256']:
            raise ValueError('projected candidate snapshot changed')
        candidate=snapshot.candidate_for(qr['candidate_uid'])
        if candidate is None:
            raise ValueError('retained candidate absent from projection')
        candidates.append(candidate)
        certificates.append(payload)
        frames.append(CandidatePlanningFrame.from_evidence(payload['planning_frame_admission']))
    if frames[0].map_frame!=frames[1].map_frame or frames[0].odom_frame!=frames[1].odom_frame:
        raise ValueError('retained projection frame identities differ')
    if any(certificates[0][key]!=certificates[1][key] for key in
           ('source_candidate_snapshot_sha256','source_registry_sha256')):
        raise ValueError('retained projections have different ancestry')
    anchor=candidates[0].geometry
    if (qr['stand_center']!=dict(x_m=anchor.x_m,y_m=anchor.y_m)
            or original.stand.radius_m!=anchor.radius_m):
        raise ValueError('retained source differs from projected candidate anchor')
    target_anchor=odom_pose_to_map(map_pose_to_odom(Pose2D(anchor.x_m,anchor.y_m),
        frames[0].map_from_odom),frames[1].map_from_odom)
    target_geometry=candidates[1].geometry
    if (math.hypot(target_anchor.x_m-target_geometry.x_m,target_anchor.y_m-target_geometry.y_m)>1e-6
            or anchor.radius_m!=target_geometry.radius_m
            or anchor.uncertainty_m!=target_geometry.uncertainty_m
            or anchor.keepout_radius_m!=target_geometry.keepout_radius_m):
        raise ValueError('projected candidate anchor differs from frame transform')
    return original,qr,*frames


def projected_retained_from_evidence(evidence):
    original,qr,source,target=_sources(evidence)
    from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import normalize_angle
    def pose(p):
        return odom_pose_to_map(map_pose_to_odom(p,source.map_from_odom),target.map_from_odom)
    rotation=target.map_from_odom.yaw_rad-source.map_from_odom.yaw_rad
    return replace(original,schema_version=SCHEMA_VERSION,
        stand=replace(original.stand,center=pose(original.stand.center)),robot_pose=pose(original.robot_pose),
        face_candidates=tuple(replace(f,pose=pose(f.pose),outward_normal_rad=normalize_angle(f.outward_normal_rad+rotation)) for f in original.face_candidates),
        material_target=replace(original.material_target,pose=pose(original.material_target.pose)),
        bounded_orientation=validated_bounded_orientation(original.bounded_orientation).rotated(rotation).payload(),
        axis_measurement=dict(evidence))


def validate_projected_retained(recommendation):
    expected=projected_retained_from_evidence(recommendation.axis_measurement)
    if recommendation != expected:
        raise ValueError('retained geometry differs from authenticated frame projection')
    return _sources(recommendation.axis_measurement)[1]


def build_projected_retained(source_path,source_projection_path,target_projection_path):
    source_path=Path(source_path).resolve()
    evidence=dict(policy=POLICY,source_recommendation_path=str(source_path),
        source_recommendation_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
    for prefix,path in (('source',source_projection_path),('target',target_projection_path)):
        path=Path(path).resolve()
        payload=load_content_hashed_json(path,hash_field='candidate_frame_projection_sha256')
        evidence[prefix+'_projection_path']=str(path)
        evidence[prefix+'_projection_sha256']=payload_sha256(payload)
    result=projected_retained_from_evidence(evidence)
    from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import validate_recommendation
    validate_recommendation(result)
    return result


def projected_retained_candidate(evidence):
    """Load the frozen anchor only after validating both projections."""
    _,qr,_,_=_sources(evidence)
    from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot
    payload=load_content_hashed_json(Path(evidence['target_projection_path']),
        hash_field='candidate_frame_projection_sha256')
    return load_candidate_snapshot(Path(payload['projected_candidate_snapshot_path'])).candidate_for(qr['candidate_uid'])
