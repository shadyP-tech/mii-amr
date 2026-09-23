"""Real retained receipts survive common-frame projection and catalog persistence."""
from dataclasses import replace
import json
import math
from pathlib import Path
import pytest
from tests.aufgabe04.test_retained_facing import retained_qr
from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, write_content_hashed_json
from scripts.aufgabe04.artifacts.retained_facing import build_retained_facing
from scripts.aufgabe04.artifacts.projected_retained_facing import build_projected_retained, projected_retained_candidate
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import recommendation_to_dict, load_recommendation, validate_recommendation, recommendation_axis_estimator
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D, map_pose_to_odom, odom_pose_to_map
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot, write_candidate_snapshot
from scripts.aufgabe04.perception.arrival_pose_estimator import arrival_pose_record_from_recommendation
from scripts.aufgabe04.stations.arrival_pose_catalog import (
    new_arrival_pose_catalog, upsert_arrival_pose, freeze_arrival_pose_catalog,
    write_arrival_pose_catalog, load_arrival_pose_catalog, validate_arrival_pose_record)
from scripts.aufgabe04.stations.arrival_pose_models import CatalogProvenance
from scripts.aufgabe04.stations.retained_catalog_geometry import require_retained_candidate_anchor


@pytest.fixture
def projected(retained_qr,tmp_path):
    path,qr,snapshot=retained_qr
    rec=build_retained_facing(path,stand_radius_m=snapshot.candidate_for(qr['candidate_uid']).geometry.radius_m,target_distance_m=.4)
    source=tmp_path/'retained.json';source.write_text(json.dumps(recommendation_to_dict(rec)))
    # The opposite receipt is in the existing target.json frame.
    old_path=tmp_path/'target.json'
    certificate=load_content_hashed_json(old_path,hash_field='candidate_frame_projection_sha256')
    frame=CandidatePlanningFrame.from_evidence(certificate['planning_frame_admission'])
    new_transform=PlanarTransform2D(.25,-.12,frame.map_from_odom.yaw_rad+.2)
    def project(p):return odom_pose_to_map(map_pose_to_odom(p,frame.map_from_odom),new_transform)
    moved=[]
    for c in snapshot.candidates:
        p=project(Pose2D(c.geometry.x_m,c.geometry.y_m))
        moved.append(replace(c,geometry=replace(c.geometry,x_m=p.x_m,y_m=p.y_m)))
    moved_snapshot=replace(snapshot,candidates=tuple(moved))
    new_snapshot=tmp_path/'new_snapshot.json';sha=write_candidate_snapshot(new_snapshot,moved_snapshot)
    new_path=tmp_path/'new_projection.json'
    write_content_hashed_json(new_path,{**certificate,
        'planning_frame_admission':CandidatePlanningFrame(project(frame.current_pose),new_transform).to_evidence(),
        'projected_candidate_snapshot_path':str(new_snapshot),'projected_candidate_snapshot_sha256':sha},
        hash_field='candidate_frame_projection_sha256')
    result=build_projected_retained(source,old_path,new_path)
    return result,rec,qr,moved_snapshot


def test_projection_retains_interval_and_authenticates_original_sources(projected,tmp_path):
    rec,original,qr,_=projected
    assert rec.schema_version==5 and rec.axis_sample_count==original.axis_sample_count
    assert rec.bounded_orientation['half_width_rad']==original.bounded_orientation['half_width_rad']
    assert rec.stand.center!=original.stand.center
    assert rec.axis_confidence==0.
    path=tmp_path/'projected.json';path.write_text(json.dumps(recommendation_to_dict(rec)))
    assert load_recommendation(path)==rec
    for changed in (replace(rec,stand=original.stand),replace(rec,axis_sample_count=1),replace(rec,schema_version=4)):
        with pytest.raises(ValueError):validate_recommendation(changed)
    Path(rec.axis_measurement['source_recommendation_path']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):validate_recommendation(rec)


def test_catalog_roundtrip_preserves_bounds_and_rejects_geometry_or_anchor_changes(projected,tmp_path):
    rec,_,qr,snapshot=projected
    record=arrival_pose_record_from_recommendation(rec,candidate_uid=rec.stand_id,
        map_yaml_sha256='a'*64,corridor_length_m=.35,validated_unix_sec=qr['checked_at_sec'],
        axis_sample_count=rec.axis_sample_count,estimator=recommendation_axis_estimator(rec),source='real/test')
    validate_arrival_pose_record(record)
    from scripts.aufgabe04.stations.retained_catalog_geometry import retained_catalog_keepouts
    from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import DynamicApproachConfig
    active=snapshot.candidate_for(rec.stand_id)
    clearance={'minimum_active_standoff_m':.30,'minimum_collision_standoff_m':.25}
    keepouts=retained_catalog_keepouts(record,active,DynamicApproachConfig(),clearance)
    assert len(keepouts)==2 and keepouts[0][:2]!=(keepouts[1][:2])
    assert keepouts[0][2]>=active.geometry.keepout_radius_m
    for invalid in ({}, {**clearance,'minimum_active_standoff_m':.6},
                    {**clearance,'minimum_collision_standoff_m':.5}):
        with pytest.raises(ValueError):retained_catalog_keepouts(record,active,DynamicApproachConfig(),invalid)
    candidate=snapshot.candidate_for(rec.stand_id)
    require_retained_candidate_anchor(record,candidate)
    from types import SimpleNamespace
    from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import DynamicApproachConfig
    from scripts.aufgabe04.navigation.missions.plan_arrival_catalog_route import _route_node
    defaults=DynamicApproachConfig()
    args=SimpleNamespace(**{k:getattr(defaults,k) for k in (
        'robot_radius_m','collision_margin_m','tracking_margin_m','corridor_sample_spacing_m',
        'lidar_stop_distance_m','scan_origin_to_base_offset_m','lidar_clearance_margin_m')})
    node=_route_node(record,args,frozen_candidate=candidate)
    delta=math.hypot(record.stand.x_m-candidate.geometry.x_m,record.stand.y_m-candidate.geometry.y_m)
    assert node.config.stand_keepout_radius_m+1e-9 >= delta+candidate.geometry.keepout_radius_m+args.tracking_margin_m
    catalog=new_arrival_pose_catalog(catalog_id='retained',provenance=CatalogProvenance('map','a'*64,'lab','b'*64,'run','real'),
        expected_candidate_uids=(rec.stand_id,),created_unix_sec=qr['sensor_stamp_sec'])
    catalog=upsert_arrival_pose(catalog,record,updated_unix_sec=qr['checked_at_sec'])
    catalog=freeze_arrival_pose_catalog(catalog,frozen_unix_sec=qr['checked_at_sec'])
    output=tmp_path/'catalog.json';write_arrival_pose_catalog(output,catalog)
    assert catalog.schema_version==2
    assert load_arrival_pose_catalog(output)==catalog
    for changed in (replace(record,retained_facing=None),replace(record,axis=replace(record.axis,sample_count=1)),
                    replace(record,stand=replace(record.stand,x_m=record.stand.x_m+.01))):
        with pytest.raises(ValueError):validate_arrival_pose_record(changed)
    with pytest.raises(ValueError,match='anchor changed'):
        require_retained_candidate_anchor(record,replace(candidate,geometry=replace(candidate.geometry,x_m=candidate.geometry.x_m+.01)))


def test_recursive_source_is_rejected_before_loading(projected):
    rec,_,_,_=projected
    import hashlib
    path=Path(rec.axis_measurement['source_recommendation_path'])
    path.write_text(json.dumps({'schema_version':5}))
    evidence={**rec.axis_measurement,'source_recommendation_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError,match='original schema-4'):
        validate_recommendation(replace(rec,axis_measurement=evidence))
