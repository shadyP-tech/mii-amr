"""Pure mission-leg identity resolution for runner argument objects.

These helpers normalize the generic routine-leg identity while retaining the
coverage-only aliases used by older autonomous children and evidence readers.
They describe semantic evidence only and never grant motion authority.
"""

from __future__ import annotations

from typing import Any

from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind


MissionLegIdentity = tuple[MissionLegKind, int, str]


def resolve_explicit_mission_leg_evidence_identity(
    args: Any,
) -> MissionLegIdentity | None:
    """Return a complete non-authorizing evidence identity, when supplied."""

    fields = (
        args.mission_leg_evidence_kind,
        args.mission_leg_evidence_index,
        str(args.mission_leg_evidence_target_id).strip() or None,
    )
    if all(value is None for value in fields):
        return None
    if any(value is None for value in fields):
        raise ValueError(
            "non-authorizing mission-leg evidence arguments must be supplied "
            "together"
        )
    index = args.mission_leg_evidence_index
    if type(index) is not int or index < 0:
        raise ValueError("mission-leg evidence index must be non-negative")
    return (
        MissionLegKind(args.mission_leg_evidence_kind),
        index,
        str(args.mission_leg_evidence_target_id).strip(),
    )


def resolve_coverage_mission_leg_identity(
    args: Any,
) -> MissionLegIdentity | None:
    """Return the legacy coverage-replan identity, when that mode is active."""

    if not args.coverage_transient_replan_enabled:
        return None
    return (
        MissionLegKind.COVERAGE,
        args.coverage_transient_replan_leg_index,
        str(args.coverage_transient_replan_target_viewpoint_id).strip(),
    )


def resolve_mission_leg_event_identity(
    args: Any,
) -> MissionLegIdentity | None:
    """Resolve one consistent semantic identity without granting authority."""

    identities: list[MissionLegIdentity] = []
    evidence = resolve_explicit_mission_leg_evidence_identity(args)
    if evidence is not None:
        identities.append(evidence)
    if (
        args.mission_leg_kind is not None
        and args.mission_leg_index is not None
        and str(args.mission_leg_target_id).strip()
    ):
        identities.append(
            (
                MissionLegKind(args.mission_leg_kind),
                args.mission_leg_index,
                str(args.mission_leg_target_id).strip(),
            )
        )
    if (
        args.startup_reseal_mission_leg_kind is not None
        and args.startup_reseal_mission_leg_index is not None
        and str(args.startup_reseal_target_id).strip()
    ):
        identities.append(
            (
                MissionLegKind(args.startup_reseal_mission_leg_kind),
                args.startup_reseal_mission_leg_index,
                str(args.startup_reseal_target_id).strip(),
            )
        )
    runtime_kind = getattr(
        args, "runtime_localization_mission_leg_kind", None
    )
    runtime_index = getattr(
        args, "runtime_localization_mission_leg_index", None
    )
    runtime_target = str(
        getattr(args, "runtime_localization_target_id", "")
    ).strip()
    if (
        runtime_kind is not None
        and runtime_index is not None
        and runtime_target
    ):
        identities.append(
            (
                MissionLegKind(runtime_kind),
                runtime_index,
                runtime_target,
            )
        )
    coverage = resolve_coverage_mission_leg_identity(args)
    if coverage is not None:
        identities.append(coverage)
    if not identities:
        return None
    first = identities[0]
    if any(identity != first for identity in identities[1:]):
        raise ValueError("conflicting mission-leg evidence identities")
    return first


def build_mission_leg_event_fields(args: Any) -> dict[str, object]:
    """Build generic event fields plus the legacy coverage-only aliases."""

    identity = resolve_mission_leg_event_identity(args)
    if identity is None:
        return {
            "coverage_leg_index": None,
            "target_viewpoint_id": "",
        }
    kind, index, target = identity
    return {
        "mission_leg_kind": kind.value,
        "mission_leg_index": index,
        "target_id": target,
        "coverage_leg_index": (
            index if kind is MissionLegKind.COVERAGE else None
        ),
        "target_viewpoint_id": (
            target if kind is MissionLegKind.COVERAGE else ""
        ),
    }


def resolve_startup_reseal_permit_identity(
    permit: Any,
) -> MissionLegIdentity:
    """Read generic permit identity with the coverage-only API fallback."""

    generic_values = (
        getattr(permit, "mission_leg_kind", None),
        getattr(permit, "mission_leg_index", None),
        str(getattr(permit, "target_id", "")).strip() or None,
    )
    if all(value is None for value in generic_values):
        return (
            MissionLegKind.COVERAGE,
            permit.leg_index,
            permit.target_viewpoint_id,
        )
    if any(value is None for value in generic_values):
        raise ValueError("startup-reseal permit has partial mission-leg identity")
    kind = MissionLegKind(generic_values[0])
    index = generic_values[1]
    target = generic_values[2]
    if index != permit.leg_index or target != permit.target_viewpoint_id:
        raise ValueError("startup-reseal permit identity aliases mismatch")
    return kind, index, target


def resolve_runtime_localization_permit_identity(
    permit: Any,
) -> MissionLegIdentity:
    """Read generic runtime permit identity with the coverage v1 fallback."""

    generic_values = (
        getattr(permit, "mission_leg_kind", None),
        getattr(permit, "mission_leg_index", None),
        str(getattr(permit, "target_id", "")).strip() or None,
    )
    if all(value is None for value in generic_values):
        return (
            MissionLegKind.COVERAGE,
            permit.leg_index,
            permit.target_viewpoint_id,
        )
    if any(value is None for value in generic_values):
        raise ValueError(
            "runtime-localization permit has partial mission-leg identity"
        )
    kind = MissionLegKind(generic_values[0])
    index = generic_values[1]
    target = generic_values[2]
    if index != permit.leg_index or target != permit.target_viewpoint_id:
        raise ValueError(
            "runtime-localization permit identity aliases mismatch"
        )
    return kind, index, target
