"""Retain certified orientation across an opposite-side identity inspection.

The source remains historical angle evidence, re-expressed in the arrival
planning frame. It never substitutes for fresh image/scan/robot-pose evidence.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import load_backside_axis_frame_projection
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot


def orientation_record(path):
    path = Path(path).resolve()
    axis = load_backside_axis_frame_projection(path)
    source = json.loads(axis.source_axis_observation_path.read_text())
    return dict(policy="certified_backside_orientation_retained", path=str(path),
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        projection_sha256=axis.projection_sha256,
        candidate_uid=axis.stand_id, planning_frame=axis.planning_frame,
        stand_center=dict(x_m=axis.stand_x_m, y_m=axis.stand_y_m),
        stand_model_profile_sha256=axis.stand_model_profile_sha256,
        robot_profile_sha256=source['robot_profile_sha256'],
        calibration_profile_sha256=source['calibration_profile_sha256'],
        stand_axis_rad=axis.stand_axis_rad, bounded_orientation=axis.bounded_orientation,
        opposite_face_normal_rad=axis.opposite_face_normal_rad,
        axis_sample_count=axis.axis_sample_count, current_angle_refit=False,
        motion_authorized=False)


def validate_retained_orientation(record, *, candidate_uid, planning_frame, stand_center, model_sha256):
    if not isinstance(record, dict):
        raise ValueError("retained backside orientation missing")
    expected = orientation_record(record.get("path", ""))
    if record != expected:
        raise ValueError("retained backside orientation differs from its certified source")
    if (record['candidate_uid'] != candidate_uid or record['planning_frame'] != planning_frame
            or record['stand_model_profile_sha256'] != model_sha256
            or any(not math.isclose(record['stand_center'][key], stand_center[key], abs_tol=1e-6,
                                    rel_tol=0.) for key in ('x_m', 'y_m'))):
        raise ValueError("retained orientation is not bound to this candidate frame")
    return record


def opposite_view_matches(record, pose):
    side = math.atan2(pose.y_m-record['stand_center']['y_m'], pose.x_m-record['stand_center']['x_m'])
    interval = record['bounded_orientation']
    half_width = 0. if interval is None else interval['half_width_rad']
    return abs(math.remainder(side-record['opposite_face_normal_rad'], math.tau))+half_width < math.pi/2


@dataclass(frozen=True)
class OppositeIdentityContext:
    orientation: dict
    snapshot: object


def load_opposite_identity_context(axis_path, snapshot_path, *, candidate_uid, planning_frame,
                                   stand_center, model_sha256):
    record = validate_retained_orientation(orientation_record(axis_path), candidate_uid=candidate_uid,
        planning_frame=planning_frame, stand_center=stand_center, model_sha256=model_sha256)
    axis = load_backside_axis_frame_projection(axis_path)
    projection = json.loads(axis.target_candidate_projection_path.read_text())
    if Path(projection['projected_candidate_snapshot_path']).resolve() != Path(snapshot_path).resolve():
        raise ValueError("opposite identity crop snapshot differs from certified arrival frame")
    snapshot = load_candidate_snapshot(Path(snapshot_path))
    return OppositeIdentityContext(record, snapshot)
