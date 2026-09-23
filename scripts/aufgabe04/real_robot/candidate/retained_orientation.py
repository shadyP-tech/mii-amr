"""Carry certified orientation across centering turns and bounded view recovery."""
from dataclasses import replace

from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import write_backside_axis_frame_projection


def retain_orientation_after_arrival(source, arrival, root):
    evidence = getattr(source, 'retained_backside_axis_path', None)
    if evidence is None:
        return arrival
    binding = arrival.decision_binding
    if binding is None:
        raise ValueError('retained orientation requires an admitted arrival candidate frame')
    path = root / 'retained_backside_orientation.json'
    write_backside_axis_frame_projection(path, axis_evidence_path=evidence,
        target_candidate_projection_path=binding.projection_path,
        target_candidate_projection_sha256=binding.projection_sha256,
        target_candidate_x_m=arrival.candidate.geometry.x_m,
        target_candidate_y_m=arrival.candidate.geometry.y_m)
    return replace(arrival, retained_backside_axis_path=path)
