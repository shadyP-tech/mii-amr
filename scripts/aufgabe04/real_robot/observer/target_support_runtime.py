"""Bind a stopped target-support failure to the current observer tuple.

Negative scan evidence never enters accepted-frame or identity consensus. Its
only result is a fresh, bound receipt asking the parent to reconcile the target.
"""

from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import content_hashed_payload
from scripts.aufgabe04.real_robot.configuration.profile import (
    camera_calibration_sha256, real_robot_profile_sha256,
)
from scripts.aufgabe04.real_robot.observer.target_reconciliation import (
    load_reconciliation_snapshot, validate_reconciliation,
)
from scripts.aufgabe04.real_robot.observer.target_support_failure import (
    STATE, TargetSupportFailureWindow, validate_target_support_failure,
)
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256


HASH_FIELD = "target_support_failure_sha256"
RESULT_STATES = frozenset({
    "metric_model_measurement_unavailable", "evidence_not_committable",
    "axis_observation_not_committable", "collecting_consensus",
    "backside_center_collecting", "opposite_identity_collecting",
    "opposite_identity_crop_conflict", "candidate_centering_budget_exceeded",
})


def target_support_binding(adapter):
    """Load the selected snapshot; never infer its identity from a status hint."""
    args = adapter.args
    path = getattr(args, "candidate_crop_snapshot", None)
    if path is None:
        raise ValueError("target-support deferral requires a candidate snapshot")
    snapshot = load_reconciliation_snapshot(path, candidate_uid=args.stand_id,
        planning_frame=adapter.profile.map_frame, center=(args.stand_x, args.stand_y))
    return dict(candidate_uid=args.stand_id, stream_id=args.stream_id,
        target_key=adapter._target_evidence_key(), planning_frame=adapter.profile.map_frame,
        stand_center=dict(x_m=args.stand_x, y_m=args.stand_y),
        candidate_snapshot_sha256=candidate_snapshot_sha256(snapshot),
        robot_profile_sha256=real_robot_profile_sha256(adapter.profile),
        calibration_profile_sha256=camera_calibration_sha256(adapter.calibration),
        stand_model_profile_sha256=adapter.stand_model_profile.sha256)


def record_target_support(adapter, *, frame, update, source_freshness):
    """Consume one exact-TF processing context after all stronger associations."""
    adapter._target_support_failure = None
    current = getattr(adapter, "_current_target_support", None)
    adapter._current_target_support = None
    window = getattr(adapter, "_target_support_failure_window", None)
    if window is None:
        window = adapter._target_support_failure_window = TargetSupportFailureWindow()
    if (current is None or not source_freshness.accepted
            or current["frame_stamp_sec"] != frame["frame_stamp_sec"]
            or current["scan_stamp_sec"] != frame["scan_stamp_sec"]
            or abs(frame["frame_stamp_sec"] - frame["scan_stamp_sec"]) > adapter.args.sync_tolerance_sec):
        window.reset("fresh_exact_target_support_tuple_unavailable")
        return
    try:
        binding = target_support_binding(adapter)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        # Missing/mismatched provenance cannot turn into a target-local failure.
        window.reset(f"target_support_binding_unavailable:{exc}")
        return
    proof = getattr(adapter, "_current_position_epoch_proof", None)
    reconciled = False
    if proof is not None:
        try:
            validate_reconciliation(proof, candidate_uid=binding["candidate_uid"],
                stand_center=(binding["stand_center"]["x_m"], binding["stand_center"]["y_m"]),
                image_stamp_sec=frame["frame_stamp_sec"], scan_stamp_sec=frame["scan_stamp_sec"])
            reconciled = (proof["target_key"] == binding["target_key"]
                and proof["epoch"] == update.snapshot.motion_epoch
                and proof["snapshot_sha256"] == binding["candidate_snapshot_sha256"])
        except (ValueError, TypeError, KeyError, OSError):
            pass
    adapter._target_support_failure = window.observe(
        association=current["association"],
        frame={**frame, "motion_epoch": update.snapshot.motion_epoch, "tf_validated": True},
        target_binding=binding, now_sec=source_freshness.checked_at_sec,
        reconciliation_validated=reconciled,
        associated_head=current.get("associated_head", False))


def commit_target_support_failure(adapter, *, state, stronger_pending=False):
    """Consume once, after successful identity/head/centering paths had priority."""
    failure = getattr(adapter, "_target_support_failure", None)
    adapter._target_support_failure = None
    if (failure is None or state not in RESULT_STATES or stronger_pending
            or getattr(adapter, "completed", False)):
        return None
    # Reload provenance at publication so a changed snapshot cannot authorize
    # a receipt from an earlier target binding.
    binding = target_support_binding(adapter)
    payload = validate_target_support_failure(failure, target_binding=binding)
    latest = payload["samples"][-1]["frame"]
    destination = Path(adapter.args.status_json).with_name("target_support_failure.json")
    if destination.exists():
        raise ValueError("target-support receipt already exists for this observer attempt")
    hashed = content_hashed_payload(payload, hash_field=HASH_FIELD)
    if not adapter._commit_sensor_artifact(destination, hashed,
            image_stamp_sec=latest["frame_stamp_sec"], scan_stamp_sec=latest["scan_stamp_sec"],
            artifact_kind="target_support_failure"):
        return None
    adapter.completed = True
    return STATE, dict(reason=payload["reason"], target_support_failure={
        "path": str(destination), "sha256": hashed[HASH_FIELD]},
        motion_authorized=False, completion_authorized=False)
