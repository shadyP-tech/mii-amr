"""One current bounded front fit retained independently of QR admission.

This is inspection evidence, not a seven-frame recommendation or motion
permission. Consumers cannot obtain facing readiness by changing a flag.
"""
from dataclasses import asdict
from pathlib import Path
from scripts.aufgabe04.navigation.foundation.models import Pose2D

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, write_content_hashed_json
from scripts.aufgabe04.perception.stand_axis.head_orientation_bounds import (
    CurrentHeadOrientationBounds, HeadOrientationHypothesis, validated_current_head_orientation_bounds)
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.artifacts.qr_verified_observation_pose import load_qr_verified_observation_pose

POLICY = 'current_bounded_front_inspection_only'
HASH_FIELD = 'bounded_front_geometry_sha256'


def _bounds(payload):
    data=dict(payload)
    data['hypotheses']=tuple(HeadOrientationHypothesis(**p) for p in data['hypotheses'])
    data['corners']=tuple(ImagePoint(**p) for p in data['corners'])
    result=CurrentHeadOrientationBounds(**data)
    if not validated_current_head_orientation_bounds(result):
        raise ValueError('invalid current bounded front geometry')
    return result


def validate_bounded_front_geometry(payload):
    if (payload.get('policy')!=POLICY or payload.get('schema_version')!=1
            or type(payload.get('axis_sample_count')) is not int
            or payload.get('axis_sample_count')!=1 or payload.get('facing_ready') is not False
            or payload.get('motion_authorized') is not False):
        raise ValueError('bounded front evidence cannot authorize facing or consensus')
    qr=load_qr_verified_observation_pose(Path(payload['qr_observation_path']))
    bounds=_bounds(payload['head_orientation_bounds'])
    if (qr['qr_verified_observation_pose_sha256']!=payload['qr_observation_sha256']
            or qr.get('retained_backside_orientation') is not None
            or any(payload[key]!=qr[key] for key in ('candidate_uid','stream_id','qr_id',
                'target_key','motion_epoch','sensor_stamp_sec','scan_stamp_sec','stand_model_profile_sha256'))
            or bounds.profile_sha256!=qr['stand_model_profile_sha256']):
        raise ValueError('bounded front geometry differs from current QR source')
    return payload


def save_bounded_front_geometry(current, qr_path, qr):
    """Only the already associated current front sample may accompany this QR."""
    if current is None or qr.get('retained_backside_orientation') is not None:
        return None
    sample=current.sample
    if (sample.face!='front' or sample.qr_id!=qr['qr_id']
            or tuple(sample.camera_signature)!=tuple(qr['camera_signature'])
            or current.robot_pose!=Pose2D(**qr['robot_pose'])
            or sample.model_sha256!=qr['stand_model_profile_sha256']
            or sample.half_width_rad!=getattr(current.proof,"half_width_rad",None)
            or sample.stamp_sec!=qr['sensor_stamp_sec'] or current.scan_stamp_sec!=qr['scan_stamp_sec']
            or not validated_current_head_orientation_bounds(current.proof,
                debug=current.debug,profile_sha256=qr['stand_model_profile_sha256'])):
        return None
    payload=dict(schema_version=1,policy=POLICY,axis_sample_count=1,facing_ready=False,
        motion_authorized=False,qr_observation_path=str(Path(qr_path).resolve()),
        qr_observation_sha256=qr['qr_verified_observation_pose_sha256'],
        head_orientation_bounds=asdict(current.proof),
        **{k:qr[k] for k in ('candidate_uid','stream_id','qr_id','target_key','motion_epoch',
                           'sensor_stamp_sec','scan_stamp_sec','stand_model_profile_sha256')})
    validate_bounded_front_geometry(payload)
    path=Path(qr_path).with_name('qr_bounded_front_geometry.json')
    write_content_hashed_json(path,payload,hash_field=HASH_FIELD)
    return path


def load_bounded_front_geometry(path):
    return validate_bounded_front_geometry(load_content_hashed_json(Path(path),hash_field=HASH_FIELD))
