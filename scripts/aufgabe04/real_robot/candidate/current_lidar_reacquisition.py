"""Bounded passive reacquisition of an otherwise supported ambiguous target.

Each cohort is assessed independently with the original admission rules. No
scan is discarded, clusters are not merged, and estimates are never averaged
across attempts. The original stopped planning frame remains the binding;
the capture owner verifies fresh scans and the stopped base pose every time.
"""

from __future__ import annotations

from pathlib import Path

from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)


MAX_COHORT_COUNT = 3
REACQUISITION_PAUSE_SEC = 2.0
_AMBIGUITY = "ambiguous_current_target_correspondence"


def require_current_lidar_target(*, config, effects, planning_frame, candidate_uid,
                                 output_dir, attempt_index):
    # Keep capture/validation at its existing replaceable sensor boundary.
    from .current_lidar_targets import capture_current_lidar_targets

    root = Path(output_dir)
    attempts = []
    previous_last_stamp = None
    for cohort_index in range(MAX_COHORT_COUNT):
        capture_root = root if cohort_index == 0 else root / f"reacquire_{cohort_index:03d}"
        estimates, evidence = capture_current_lidar_targets(
            config, effects, planning_frame, {candidate_uid}, capture_root)
        if previous_last_stamp is not None:
            # A new output directory alone does not establish a new capture.
            # The capture owner's validation also checks the full sensor and
            # frame bindings; this check rules out overlap with a prior cohort.
            if evidence["observation_not_before_sec"] <= previous_last_stamp:
                raise ValueError("LiDAR reacquisition reused the prior scan epoch")
        decision = evidence["candidate_decisions"][candidate_uid]
        accepted = candidate_uid in estimates
        scan_reasons = {scan["reason"] for scan in decision.get("scans", ())}
        retry = (not accepted and decision["reasons"] == [_AMBIGUITY]
                 and "ambiguous_clusters" in scan_reasons
                 and "competing_candidate" not in scan_reasons
                 and cohort_index + 1 < MAX_COHORT_COUNT)
        attempts.append({
            "cohort_index": cohort_index,
            "accepted": accepted,
            "reasons": list(decision["reasons"]),
            "evidence_path": evidence["evidence_path"],
            "evidence_sha256": evidence["evidence_sha256"],
            "retry_scheduled": retry,
        })
        # Successful first captures retain their established evidence shape.
        if retry or cohort_index:
            effects.event_sink(root / "reacquisition_events.jsonl", {
                "schema_version": 1,
                "event": "candidate_current_lidar_reacquisition",
                "candidate_uid": candidate_uid,
                "observation_attempt_index": attempt_index,
                "maximum_cohort_count": MAX_COHORT_COUNT,
                **attempts[-1],
                "timestamp_unix_sec": effects.clock(),
                "motion_authorized": False,
                "stand_axis_authorized": False,
                "keepouts_changed": False,
            })
        if not retry:
            if cohort_index:
                evidence = {**evidence, "reacquisition": {
                    "maximum_cohort_count": MAX_COHORT_COUNT,
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "motion_authorized": False,
                }}
            if accepted:
                return estimates[candidate_uid], evidence
            raise CandidateObservationUnavailableError(
                candidate_uid=candidate_uid, observation_attempt_index=attempt_index,
                reason="candidate_target_ineligible",
                process_evidence={"observer_started": False, "motion_authorized": False},
                status_evidence={"reason": "current_lidar_target_unavailable",
                                 "current_lidar_support": evidence},
            )
        previous_last_stamp = max(scan["scan_stamp_sec"] for scan in decision["scans"])
        effects.wait_for_lidar_reacquisition(REACQUISITION_PAUSE_SEC)
    raise AssertionError("bounded LiDAR reacquisition did not terminate")
