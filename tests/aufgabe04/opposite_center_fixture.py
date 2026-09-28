"""Exact second-candidate sources from run 20260928T142407Z on mii002."""
import json
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.artifacts.retained_backside_orientation import orientation_record
from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import write_backside_axis_frame_projection
from scripts.aufgabe04.real_robot.observer.opposite_target_geometry import retained_scan_target
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import transform_point
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot
from tests.aufgabe04.test_target_reconciliation import recorded_inputs

ROOT = Path(__file__).parent/'fixtures/opposite_center_20260928'


def recorded_center(root):
    data, rows, intrinsics, camera = recorded_inputs(json.loads((ROOT/'inputs.json').read_text()),
        root=ROOT, candidate_uid='survey_candidate_0001')
    raw = json.loads((ROOT/'backside_observation.json').read_text())
    raw['target_reconciliation']['snapshot_path'] = str((ROOT/'backside_snapshot.json').resolve())
    model = Path(__file__).resolve().parents[2]/'configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json'
    raw['head_position_evidence']['model_path'] = str(model)
    axis = root/'axis.json'; axis.write_text(json.dumps(raw))
    digests = []
    for label, name in [('source', 'backside_snapshot.json'), ('target', 'candidate_snapshot.json')]:
        projection = json.loads((ROOT/f'{label}_projection.json').read_text())
        projection.pop('candidate_frame_projection_sha256')
        projection['source_candidate_snapshot_path'] = str((ROOT/'canonical_snapshot.json').resolve())
        projection['projected_candidate_snapshot_path'] = str((ROOT/name).resolve())
        digests.append(write_content_hashed_json(root/f'{label}.json', projection,
            hash_field='candidate_frame_projection_sha256'))
    snapshot = load_candidate_snapshot(ROOT/'candidate_snapshot.json')
    g = snapshot.candidate_for('survey_candidate_0001').geometry
    write_backside_axis_frame_projection(root/'orientation.json', axis_evidence_path=axis,
        source_candidate_projection_path=root/'source.json', source_candidate_projection_sha256=digests[0],
        target_candidate_projection_path=root/'target.json', target_candidate_projection_sha256=digests[1],
        target_candidate_x_m=g.x_m, target_candidate_y_m=g.y_m)
    orientation = orientation_record(root/'orientation.json')
    center = orientation['validated_target_center']
    import math
    for row in rows:
        old = transform_point((g.x_m,g.y_m,0.),row['scan_from_map'])
        tolerance = row['options']['accepted_range_m'][1]-math.hypot(*old[:2])
        point = transform_point((center['x_m'],center['y_m'],0.),row['scan_from_map'])
        target = retained_scan_target(point, center=center, stand_radius_m=g.radius_m,
            stand_uncertainty_m=g.uncertainty_m, lidar_range_tolerance_m=tolerance)
        row['options'] = {**row['options'], 'map_bearing_rad':target.bearing_rad,
                          'accepted_range_m':target.accepted_range_m}
        row.update(retained_orientation=orientation, use_retained_target=True)
    return data, rows, intrinsics, camera, snapshot, orientation, model
