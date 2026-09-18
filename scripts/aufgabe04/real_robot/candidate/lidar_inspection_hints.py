"""Read the completed survey's bound scans for optional first-view hints."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.aufgabe04.navigation.approach.lidar_inspection_hint import derive_lidar_inspection_hints
from scripts.aufgabe04.navigation.coverage.coverage_stop_perception_admission import load_stopped_observer_summary
from scripts.aufgabe04.navigation.coverage.coverage_visibility_reporting import (
    coverage_visibility_epoch_fields, validate_coverage_visibility_evidence,
)


def load_camera_lidar_hints(*, survey_root, plan, snapshot, registry, planning_frame):
    if registry is None or planning_frame is None:
        return {}, {"reason": "survey_planning_frame_unavailable", "motion_authorized": False}
    receipts, failures = [], {}
    viewpoint_ids = sorted({v for c in registry.candidates
                            if c.candidate_uid in snapshot.candidate_uids for v in c.viewpoint_ids})
    for viewpoint_id in viewpoint_ids:
        try:
            path = Path(survey_root) / "epochs" / f"{viewpoint_id}.json"
            if path.is_symlink():
                raise ValueError("survey epoch must not be a symlink")
            epoch = json.loads(path.read_text())
            if (not isinstance(epoch, dict) or epoch.get("survey_id") != plan.survey_id
                    or epoch.get("viewpoint_id") != viewpoint_id):
                raise ValueError("survey epoch identity mismatch")
            summary = load_stopped_observer_summary(Path(epoch["observer_summary_json"]))
            evidence = validate_coverage_visibility_evidence(summary, plan, viewpoint_id, True)
            if any(epoch.get(k) != v for k, v in coverage_visibility_epoch_fields(evidence).items()):
                raise ValueError("survey epoch visibility binding mismatch")
            receipts.extend(evidence.receipts)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Hints are optional. Unavailable or corrupt evidence cannot steer a
            # view; the ordinary candidate route and all its checks remain.
            failures[viewpoint_id] = str(exc)
    hints, candidates = derive_lidar_inspection_hints(
        snapshot=snapshot, registry=registry, planning_frame=planning_frame, receipts=receipts,
    )
    return hints, {"candidates": candidates, "unavailable_epochs": failures,
                   "motion_authorized": False, "stand_axis_authorized": False}
