"""Facing geometry from a certified historical angle and current QR identity.

The QR receipt remains discovery-only. This separate recommendation retains its
source chain and uncertainty, and still requires route/clearance validation.
"""
from dataclasses import replace
import hashlib
import math
from pathlib import Path

from scripts.aufgabe04.artifacts.qr_verified_observation_pose import load_qr_verified_observation_pose
from scripts.aufgabe04.artifacts.retained_backside_orientation import opposite_view_matches
from scripts.aufgabe04.navigation.foundation.models import Pose2D

POLICY = "retained_backside_current_qr_facing"
SCHEMA_VERSION = 4


def retained_facing_source(recommendation):
    evidence = recommendation.axis_measurement
    if not isinstance(evidence, dict) or evidence.get("policy") != POLICY:
        raise ValueError("retained facing requires its source receipt")
    source_path = evidence.get("qr_observation_path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("retained facing requires a QR source path")
    path = Path(source_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != evidence.get("qr_observation_sha256"):
        raise ValueError("retained facing QR source changed")
    qr = load_qr_verified_observation_pose(path)
    orientation = qr.get("retained_backside_orientation")
    if not orientation or not orientation.get("validated_target_center") or not orientation.get("bounded_orientation"):
        raise ValueError("retained facing requires validated center and bounded orientation")
    return qr, orientation


def validate_retained_facing(recommendation):
    qr, orientation = retained_facing_source(recommendation)
    center = orientation["validated_target_center"]
    target = next((f for f in recommendation.face_candidates
                   if f.face_id == recommendation.material_target.face_id), None)
    if target is None:
        raise ValueError("retained facing requires a selected face")
    if (recommendation.simulation_only
            or recommendation.stand_id != qr['candidate_uid']
            or recommendation.stream_id != qr['stream_id']
            or recommendation.planning_frame != qr['planning_frame']
            or recommendation.robot_pose != Pose2D(**qr['robot_pose'])
            or recommendation.sensor_stamp_sec != qr['sensor_stamp_sec']
            or recommendation.observation_unix_sec != qr['sensor_stamp_sec']
            or recommendation.bounded_orientation != orientation['bounded_orientation']
            or recommendation.axis_sample_count != orientation['axis_sample_count']
            or recommendation.axis_confidence != 0.
            or recommendation.stand.center != Pose2D(center['x_m'], center['y_m'])
            or recommendation.stand.uncertainty_m != center['uncertainty_m']
            or not opposite_view_matches(orientation, recommendation.robot_pose)
            or abs(math.remainder(target.outward_normal_rad-orientation['opposite_face_normal_rad'], math.tau)) > 1e-9
            or recommendation.side_evidence.kind != 'qr_observation'
            or recommendation.side_evidence.provenance != 'real/onboard_camera_qr_observation'
            or recommendation.axis_measurement.get('current_angle_refit') is not False):
        raise ValueError("retained facing differs from certified angle, center or current QR")
    return qr


def build_retained_facing(qr_path, *, stand_radius_m, target_distance_m):
    from scripts.aufgabe04.real_robot.configuration.recommendation import build_real_viewpoint_recommendation
    from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import validate_recommendation
    path = Path(qr_path).resolve()
    qr = load_qr_verified_observation_pose(path)
    orientation = qr.get('retained_backside_orientation')
    if not orientation or not orientation.get('validated_target_center') or not orientation.get('bounded_orientation'):
        raise ValueError('retained facing requires validated center and bounded orientation')
    center = orientation['validated_target_center']
    result = build_real_viewpoint_recommendation(
        stream_id=qr['stream_id'], stand_id=qr['candidate_uid'], planning_frame=qr['planning_frame'],
        stand_center=Pose2D(center['x_m'], center['y_m']), stand_radius_m=stand_radius_m,
        stand_uncertainty_m=center['uncertainty_m'], robot_pose=Pose2D(**qr['robot_pose']),
        stand_axis_rad=orientation['stand_axis_rad'], axis_confidence=0.,
        axis_sample_count=orientation['axis_sample_count'], sensor_stamp_sec=qr['sensor_stamp_sec'],
        expected_qr_id=qr['qr_id'], observed_qr_ids=(qr['qr_id'],), target_distance_m=target_distance_m,
        observation_unix_sec=qr['sensor_stamp_sec'], bounded_orientation=orientation['bounded_orientation'])
    result = replace(result, schema_version=SCHEMA_VERSION,
        side_evidence=replace(result.side_evidence, kind='qr_observation',
                              provenance='real/onboard_camera_qr_observation'),
        axis_measurement=dict(policy=POLICY, source=POLICY, current_angle_refit=False,
            qr_observation_path=str(path), qr_observation_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    validate_recommendation(result)
    return result
