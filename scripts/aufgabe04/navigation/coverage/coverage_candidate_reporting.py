"""Pure candidate-count reporting contracts for coverage artifacts.

The stopped LiDAR epoch and the fused survey registry have different scopes:
an epoch count describes only the latest static-map admission batch, while a
registry count describes every candidate fused so far.  Keep those scopes
explicit and retain the schema-v1 field names only as documented aliases.
"""

from __future__ import annotations

from collections.abc import Mapping


REJECTED_CANDIDATE_STATUS = "rejected"


def coverage_phase_completion_fields(
    *,
    lidar_coverage_complete: bool,
    camera_candidate_resolution_complete: bool,
    camera_expected_stand_count_met: bool,
) -> dict[str, object]:
    """Expose phase-scoped completion and label the old ambiguous alias."""

    values = (
        lidar_coverage_complete,
        camera_candidate_resolution_complete,
        camera_expected_stand_count_met,
    )
    if any(type(value) is not bool for value in values):
        raise ValueError("coverage completion fields must be boolean")
    camera_complete = (
        lidar_coverage_complete
        and camera_candidate_resolution_complete
        and camera_expected_stand_count_met
    )
    if not lidar_coverage_complete:
        phase = "lidar_coverage_in_progress"
    elif not camera_candidate_resolution_complete:
        phase = "lidar_complete_camera_validation_pending"
    elif not camera_expected_stand_count_met:
        phase = "camera_resolved_expected_stand_count_not_met"
    else:
        phase = "camera_exploration_complete"
    return {
        "completion_phase": phase,
        "lidar_coverage_complete": lidar_coverage_complete,
        "camera_candidate_resolution_complete": (
            camera_candidate_resolution_complete
        ),
        "camera_expected_stand_count_met": camera_expected_stand_count_met,
        "camera_exploration_complete": camera_complete,
        # Backward-compatible schema-v1 alias. It means camera completion,
        # never LiDAR checkpoint readiness.
        "exploration_complete": camera_complete,
        "legacy_completion_aliases": {
            "exploration_complete": "camera_exploration_complete",
        },
    }


def fused_registry_candidate_count_fields(
    candidate_counts: Mapping[str, int],
) -> dict[str, object]:
    """Return explicit cumulative registry counts plus schema-v1 aliases."""

    counts = _validated_candidate_counts(candidate_counts)
    total_count = sum(counts.values())
    active_count = total_count - counts.get(REJECTED_CANDIDATE_STATUS, 0)
    return {
        "fused_registry_candidate_counts": counts,
        "fused_registry_total_candidate_count": total_count,
        "fused_registry_active_candidate_count": active_count,
        # Backward-compatible schema-v1 aliases.  New readers should consume
        # the fused_registry_* fields above instead.
        "candidate_counts": counts,
        "candidate_count": total_count,
        "legacy_fused_registry_candidate_count_aliases": {
            "candidate_counts": "fused_registry_candidate_counts",
            "candidate_count": "fused_registry_total_candidate_count",
        },
    }


def coverage_epoch_candidate_count_fields(
    *,
    confirmed_epoch_candidate_count: int,
    morphology_admitted_candidate_count: int | None = None,
    morphology_rejected_candidate_count: int | None = None,
    static_map_admitted_candidate_count: int,
    static_map_boundary_provisional_candidate_count: int = 0,
    static_map_rejected_candidate_count: int,
    fused_registry_candidate_counts: Mapping[str, int],
) -> dict[str, object]:
    """Report one epoch separately from the cumulative post-fusion registry."""

    confirmed_count = _nonnegative_int(
        confirmed_epoch_candidate_count,
        "confirmed_epoch_candidate_count",
    )
    morphology_admitted_count = _nonnegative_int(
        (
            confirmed_count
            if morphology_admitted_candidate_count is None
            else morphology_admitted_candidate_count
        ),
        "morphology_admitted_candidate_count",
    )
    morphology_rejected_count = _nonnegative_int(
        (
            0
            if morphology_rejected_candidate_count is None
            else morphology_rejected_candidate_count
        ),
        "morphology_rejected_candidate_count",
    )
    admitted_count = _nonnegative_int(
        static_map_admitted_candidate_count,
        "static_map_admitted_candidate_count",
    )
    boundary_provisional_count = _nonnegative_int(
        static_map_boundary_provisional_candidate_count,
        "static_map_boundary_provisional_candidate_count",
    )
    rejected_count = _nonnegative_int(
        static_map_rejected_candidate_count,
        "static_map_rejected_candidate_count",
    )
    if confirmed_count != morphology_admitted_count + morphology_rejected_count:
        raise ValueError(
            "confirmed epoch candidate count must equal morphology admitted "
            "plus morphology rejected"
        )
    if (
        morphology_admitted_count
        != admitted_count + boundary_provisional_count + rejected_count
    ):
        raise ValueError(
            "morphology-admitted candidate count must equal admitted plus "
            "boundary-provisional plus rejected at the static-map gate"
        )

    return {
        "epoch_confirmed_lidar_candidate_count": confirmed_count,
        "epoch_morphology_admitted_candidate_count": (
            morphology_admitted_count
        ),
        "epoch_morphology_rejected_candidate_count": (
            morphology_rejected_count
        ),
        "epoch_static_map_admitted_candidate_count": admitted_count,
        "epoch_static_map_boundary_provisional_candidate_count": (
            boundary_provisional_count
        ),
        "epoch_static_map_population_retained_candidate_count": (
            admitted_count + boundary_provisional_count
        ),
        "epoch_static_map_rejected_candidate_count": rejected_count,
        **fused_registry_candidate_count_fields(fused_registry_candidate_counts),
        # Backward-compatible schema-v1 aliases.  Their scope is one epoch,
        # despite the missing epoch prefix.
        "confirmed_epoch_candidate_count": confirmed_count,
        "static_map_candidate_admitted_count": admitted_count,
        "static_map_candidate_boundary_provisional_count": (
            boundary_provisional_count
        ),
        "static_map_candidate_rejected_count": rejected_count,
        "legacy_epoch_candidate_count_aliases": {
            "confirmed_epoch_candidate_count": (
                "epoch_confirmed_lidar_candidate_count"
            ),
            "static_map_candidate_admitted_count": (
                "epoch_static_map_admitted_candidate_count"
            ),
            "static_map_candidate_boundary_provisional_count": (
                "epoch_static_map_boundary_provisional_candidate_count"
            ),
            "static_map_candidate_rejected_count": (
                "epoch_static_map_rejected_candidate_count"
            ),
        },
    }


def active_lidar_registry_count_fields(
    active_lidar_candidate_count: int,
    *,
    static_map_admitted_candidate_count: int | None = None,
    boundary_provisional_candidate_count: int = 0,
) -> dict[str, object]:
    """Return the exact-two gate count without calling it epoch admission."""

    active_count = _nonnegative_int(
        active_lidar_candidate_count,
        "active_lidar_candidate_count",
    )
    strict_count = _nonnegative_int(
        (
            active_count
            if static_map_admitted_candidate_count is None
            else static_map_admitted_candidate_count
        ),
        "static_map_admitted_candidate_count",
    )
    boundary_count = _nonnegative_int(
        boundary_provisional_candidate_count,
        "boundary_provisional_candidate_count",
    )
    if strict_count + boundary_count != active_count:
        raise ValueError(
            "active LiDAR count must equal static-map admitted plus "
            "boundary-provisional candidates"
        )
    return {
        "active_lidar_registry_candidate_count": active_count,
        "fused_registry_active_candidate_count": active_count,
        "lidar_static_map_admitted_candidate_count": strict_count,
        "lidar_boundary_provisional_candidate_count": boundary_count,
        "lidar_population_retained_candidate_count": active_count,
        "legacy_lidar_checkpoint_candidate_count_aliases": {
            "fused_registry_active_candidate_count": (
                "active_lidar_registry_candidate_count"
            ),
        },
    }


def _validated_candidate_counts(
    candidate_counts: Mapping[str, int],
) -> dict[str, int]:
    if not isinstance(candidate_counts, Mapping):
        raise ValueError("candidate counts must be a mapping")
    counts = dict(sorted(candidate_counts.items()))
    if any(
        not isinstance(status, str)
        or not status.strip()
        or type(count) is not int
        or count < 0
        for status, count in counts.items()
    ):
        raise ValueError(
            "candidate counts must map non-empty names to non-negative integers"
        )
    return counts


def _nonnegative_int(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "active_lidar_registry_count_fields",
    "coverage_phase_completion_fields",
    "coverage_epoch_candidate_count_fields",
    "fused_registry_candidate_count_fields",
]
