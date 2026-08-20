"""Pure terminal reporting for autonomous camera exploration.

The live parent runner supplies already-validated artifact identities and the
completed candidate-phase fields.  This module keeps phase-specific status
semantics out of the motion/ROS wiring and never grants residual motion
authority.
"""

from __future__ import annotations

from pathlib import Path

from scripts.aufgabe04.real_robot.autonomous_modes import AutonomousRunMode


def build_completed_camera_mission_summary(
    *,
    run_mode: str,
    session_id: str,
    snapshot_path: Path,
    snapshot_sha256: str,
    survey_root: Path,
    candidate_population_admission_path: Path,
    candidate_population_admission_sha256: str,
    candidate_phase_fields: dict[str, object],
    exact_two_coverage_summary: dict[str, object] | None = None,
    exact_two_camera_handoff_path: Path | None = None,
    exact_two_camera_handoff_sha256: str | None = None,
) -> dict[str, object]:
    """Build one unambiguous, motion-neutral terminal camera summary."""

    exact_mode = run_mode == AutonomousRunMode.EXECUTE_EXACT_TWO_CAMERA.value
    exact_evidence = (
        exact_two_coverage_summary,
        exact_two_camera_handoff_path,
        exact_two_camera_handoff_sha256,
    )
    if exact_mode != all(value is not None for value in exact_evidence):
        raise ValueError(
            "exact-two final reporting requires its coverage summary and "
            "camera handoff path/hash together"
        )
    if candidate_phase_fields.get("motion_authorized") is not False:
        raise ValueError(
            "completed candidate phase must revoke residual motion authority"
        )
    if type(candidate_phase_fields.get("stand_count")) is not int:
        raise ValueError("completed candidate phase has no stand count")

    result: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "run_mode": run_mode,
        "motion_published": True,
        "prior_leg_motion_published": True,
        "motion_authorized": False,
        "session_id": session_id,
        "candidate_snapshot": str(snapshot_path),
        "candidate_snapshot_sha256": snapshot_sha256,
        "candidate_snapshot_ready": True,
        "survey_root": str(survey_root),
        "candidate_population_admission": str(
            candidate_population_admission_path
        ),
        "candidate_population_admission_sha256": (
            candidate_population_admission_sha256
        ),
        # Preserve schema-v1 consumers while the scoped name above removes
        # ambiguity for exact-two camera admission.
        "coverage_candidate_admission": str(
            candidate_population_admission_path
        ),
        "coverage_candidate_admission_sha256": (
            candidate_population_admission_sha256
        ),
        "lidar_coverage_complete": True,
        "lidar_checkpoint_complete": True,
        "camera_validation_population_ready": True,
        "camera_approach_authorized": False,
        "camera_approach_executed": True,
        "camera_validation_complete": True,
        "camera_exploration_complete": True,
        "exploration_complete": True,
    }
    result.update(candidate_phase_fields)
    result["motion_authorized"] = False

    if exact_mode:
        assert exact_two_coverage_summary is not None
        assert exact_two_camera_handoff_path is not None
        assert exact_two_camera_handoff_sha256 is not None
        required = (
            "lidar_checkpoint_admission",
            "lidar_checkpoint_admission_sha256",
            "camera_validation_admission",
            "camera_validation_admission_sha256",
            "camera_validation_candidate_uids",
            "multi_view_candidate_uids",
            "single_view_requires_camera_validation_candidate_uids",
        )
        missing = tuple(
            name for name in required if name not in exact_two_coverage_summary
        )
        if missing:
            raise ValueError(
                "exact-two coverage summary is missing: " + ", ".join(missing)
            )
        camera_uids = exact_two_coverage_summary[
            "camera_validation_candidate_uids"
        ]
        if (
            not isinstance(camera_uids, list)
            or len(camera_uids) != candidate_phase_fields["stand_count"]
        ):
            raise ValueError(
                "completed camera stand count differs from exact-two handoff"
            )
        result.update(
            {
                "lidar_checkpoint_admission": exact_two_coverage_summary[
                    "lidar_checkpoint_admission"
                ],
                "lidar_checkpoint_admission_sha256": (
                    exact_two_coverage_summary[
                        "lidar_checkpoint_admission_sha256"
                    ]
                ),
                "camera_validation_admission": exact_two_coverage_summary[
                    "camera_validation_admission"
                ],
                "camera_validation_admission_sha256": (
                    exact_two_coverage_summary[
                        "camera_validation_admission_sha256"
                    ]
                ),
                "exact_two_camera_handoff": str(
                    exact_two_camera_handoff_path
                ),
                "exact_two_camera_handoff_sha256": (
                    exact_two_camera_handoff_sha256
                ),
                "camera_validation_candidate_uids": camera_uids,
                "multi_view_candidate_uids": exact_two_coverage_summary[
                    "multi_view_candidate_uids"
                ],
                "single_view_requires_camera_validation_candidate_uids": (
                    exact_two_coverage_summary[
                        "single_view_requires_camera_validation_candidate_uids"
                    ]
                ),
            }
        )
    return result


__all__ = ["build_completed_camera_mission_summary"]
