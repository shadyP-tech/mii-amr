"""Seal and validate an exact prefix of a certified opposite-face route.

The parent retains the full stand endpoint, angle interval and center receipt.
The checkpoint endpoint is a stopped transit pose, never camera arrival.
"""
import csv
import json
import math
from pathlib import Path
import shutil

from scripts.aufgabe04.navigation.control.safety_checks import PreflightStatus
from scripts.aufgabe04.navigation.execution.route_context import file_sha256
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg

CHECKPOINT_BEARING_MODE = "opposite-localization-checkpoint"


def _parent(checkpoint):
    route = Path(checkpoint["parent_route_csv"])
    diagnostics = Path(checkpoint["parent_diagnostics_json"])
    for path, key in ((route, "parent_route_sha256"), (diagnostics, "parent_diagnostics_sha256")):
        if file_sha256(path) != checkpoint[key]:
            raise ValueError("checkpoint parent artifact hash mismatch")
    payload = json.loads(diagnostics.read_text())
    if payload["metadata"]["approach_bearing_mode"] != "camera-axis-face":
        raise ValueError("checkpoint requires a full certified opposite route; nesting forbidden")
    return route, diagnostics, payload


def validate_opposite_checkpoint_binding(metadata, leg, snapshot_path):
    from scripts.aufgabe04.navigation.approach.detected_stand_preapproach import validate_detected_stand_preapproach_binding
    try:
        checkpoint = metadata["opposite_localization_checkpoint"]
        if type(checkpoint["maximum_checkpoint_count"]) is not int or checkpoint["maximum_checkpoint_count"] != 1 or checkpoint["camera_arrival"] is not False:
            raise ValueError("invalid opposite checkpoint bounds")
        route, diagnostics, parent = _parent(checkpoint)
        parent_leg = load_route_leg(route, 0, thinning_min_spacing_m=0.)
        status = validate_detected_stand_preapproach_binding(
            diagnostics, parent_leg, candidate_snapshot_path=snapshot_path, diagnostics_payload=parent,
        )
        if not status.ok:
            raise ValueError("invalid checkpoint parent: " + "; ".join(status.failures))
        index = checkpoint["vertex_index"]
        if type(index) is not int or not 0 < index < len(parent_leg.raw_waypoints)-1:
            raise ValueError("checkpoint must be an interior parent waypoint")
        if len(leg.raw_waypoints) != index+1:
            raise ValueError("checkpoint prefix length mismatch")
        for actual, expected in zip(leg.raw_waypoints, parent_leg.raw_waypoints[:index+1]):
            if (actual.pose.x_m, actual.pose.y_m, actual.point_index, actual.cumulative_length_m) != (
                    expected.pose.x_m, expected.pose.y_m, expected.point_index, expected.cumulative_length_m):
                raise ValueError("checkpoint differs from exact parent prefix")
        a, b = parent_leg.raw_waypoints[index-1:index+1]
        start = parent_leg.raw_waypoints[0].pose
        if (math.hypot(b.pose.x_m-start.x_m, b.pose.y_m-start.y_m) < .20
                or parent_leg.route_length_m-b.cumulative_length_m < .15):
            raise ValueError("checkpoint must leave meaningful prefix and suffix travel")
        expected_yaw = math.atan2(b.pose.y_m-a.pose.y_m, b.pose.x_m-a.pose.x_m)
        if abs(math.remainder(leg.raw_waypoints[-1].pose.yaw_rad-expected_yaw, 2*math.pi)) > 1e-9:
            raise ValueError("checkpoint must stop aligned with the incoming segment")
        for point in leg.raw_waypoints[:-1]:
            if math.isfinite(point.pose.yaw_rad) or point.protected or point.corridor:
                raise ValueError("checkpoint transit waypoint changed")
        changed = {"approach_bearing_mode", "selected_approach_pose", "route_csv_sha256",
                   "route_certificate_path", "route_certificate_sha256", "candidate_snapshot_json",
                   "source_route_sha256", "source_diagnostics_sha256", "source_pipeline_summary_sha256"}
        if set(metadata) != set(parent["metadata"]) | {"opposite_localization_checkpoint"}:
            raise ValueError("checkpoint metadata fields differ from parent contract")
        for key, value in parent["metadata"].items():
            if key not in changed and metadata.get(key) != value:
                raise ValueError(f"checkpoint changed parent {key}")
        if metadata["route_csv_sha256"] != leg.source_sha256:
            raise ValueError("checkpoint route digest mismatch")
        if metadata["selected_approach_pose"] != dict(x_m=b.pose.x_m,y_m=b.pose.y_m,yaw_rad=expected_yaw):
            raise ValueError("checkpoint endpoint metadata mismatch")
        return PreflightStatus(ok=True, failures=[])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return PreflightStatus(ok=False, failures=[f"opposite checkpoint binding: {exc}"])


def materialize_opposite_checkpoint(*, sealed_parent, snapshot_path, choice, output_dir):
    from scripts.aufgabe04.navigation.approach.detected_stand_preapproach import seal_detected_stand_preapproach
    route = Path(sealed_parent["route_csv"]).resolve()
    diagnostics = Path(sealed_parent["diagnostics_json"]).resolve()
    if file_sha256(route) != choice.evidence["source_route_sha256"]:
        raise ValueError("checkpoint selection route changed")
    parent = json.loads(diagnostics.read_text())
    output_dir.mkdir(parents=True, exist_ok=False)
    with route.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows = rows[:choice.vertex_index+1]
    for row in rows:
        row["yaw_rad"] = ""
        row["protected"] = row["corridor"] = "false"
    rows[-1]["yaw_rad"] = repr(choice.poses[-1].yaw_rad)
    with (output_dir/"route.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(snapshot_path, output_dir/"candidate_snapshot.json")
    checkpoint = dict(parent_route_csv=str(route), parent_route_sha256=file_sha256(route),
        parent_diagnostics_json=str(diagnostics), parent_diagnostics_sha256=file_sha256(diagnostics),
        vertex_index=choice.vertex_index, maximum_checkpoint_count=1, camera_arrival=False)
    end = choice.poses[-1]
    metadata = {**parent["metadata"], "approach_bearing_mode": CHECKPOINT_BEARING_MODE,
        "selected_approach_pose": dict(x_m=end.x_m,y_m=end.y_m,yaw_rad=end.yaw_rad),
        "opposite_localization_checkpoint": checkpoint}
    (output_dir/"route_diagnostics.json").write_text(json.dumps({**parent,"metadata":metadata},indent=2)+"\n")
    (output_dir/"pipeline_summary.json").write_text(json.dumps({"status":"observe_and_plan_complete",
        "motion_published":False,"camera_arrival":False})+"\n")
    (output_dir/"checkpoint_selection.json").write_text(json.dumps(choice.evidence,indent=2)+"\n")
    return seal_detected_stand_preapproach(pipeline_root=output_dir)
