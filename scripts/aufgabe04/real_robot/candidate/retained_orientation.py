"""Carry certified orientation across centering turns and bounded view recovery."""
from dataclasses import replace

from scripts.aufgabe04.artifacts.current_target_estimate import planning_target_geometry
from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import (
    load_backside_axis_planning_observation, write_backside_axis_frame_projection,
)
from scripts.aufgabe04.real_robot.candidate.target_admission import retain_camera_target_geometry


def retain_orientation_after_arrival(source, arrival, root):
    evidence = getattr(source, 'retained_backside_axis_path', None)
    if evidence is None:
        return retain_camera_target_geometry(source, arrival,
            evidence_path=root / 'retained_camera_target_geometry.json')
    binding = arrival.decision_binding
    if binding is None:
        raise ValueError('retained orientation requires an admitted arrival candidate frame')
    path = root / 'retained_backside_orientation.json'
    write_backside_axis_frame_projection(path, axis_evidence_path=evidence,
        target_candidate_projection_path=binding.projection_path,
        target_candidate_projection_sha256=binding.projection_sha256,
        target_candidate_x_m=arrival.candidate.geometry.x_m,
        target_candidate_y_m=arrival.candidate.geometry.y_m)
    geometry = getattr(source, 'camera_target_geometry', None)
    if (geometry is not None and getattr(source, 'retained_lidar_target', None) is None
            and getattr(arrival, 'camera_target_geometry', None) is None):
        original = load_backside_axis_planning_observation(evidence)
        if geometry == planning_target_geometry(source.candidate, original.validated_target_center):
            # A certified reconciled center has its own position bound. Replay
            # that proof instead of treating it as an unbound camera fit with
            # the smaller survey-envelope bound.
            projected = load_backside_axis_planning_observation(path)
            return replace(arrival, retained_backside_axis_path=path,
                camera_target_geometry=planning_target_geometry(arrival.candidate, projected.validated_target_center),
                retained_lidar_target=None, current_lidar_target_path=None,
                camera_alignment=None, camera_target_geometry_evidence_path=None)
    # The certified axis and the previously supported target point are
    # independent receipts. Refreshing the axis must not discard the point
    # used by the completed route or demand another scan at this stopped view.
    return retain_camera_target_geometry(
        source, replace(arrival, retained_backside_axis_path=path),
        evidence_path=root / 'retained_camera_target_geometry.json')
