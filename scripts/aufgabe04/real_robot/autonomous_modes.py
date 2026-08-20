"""Pure run-mode contract for autonomous real-robot stand exploration.

Selecting an execution mode expresses operator intent; it does not authorize
motion.  The caller must still enforce every physical-run gate, including the
typed ``RUN`` confirmation and exact one-use child-leg permits.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class AutonomousRunMode(str, Enum):
    """Mutually exclusive autonomous exploration workflows."""

    DRY_FIRST_LEG = "dry-first-leg"
    EXECUTE_COVERAGE_CHECKPOINT = "execute-coverage-checkpoint"
    EXECUTE_COVERAGE_ONLY = "execute-coverage-only"
    EXECUTE_EXACT_TWO_CAMERA = "execute-exact-two-camera"
    EXECUTE_FULL = "execute-full"
    RESUME_NEXT_COVERAGE_LEG = "resume-next-coverage-leg"


class AutonomousAuthorizationScope(str, Enum):
    """Machine-readable scope to bind to a later motion authorization."""

    NONE = "none"
    BOUNDED_COVERAGE = "bounded-coverage"
    COVERAGE_ONLY = "coverage-only"
    EXACT_TWO_CAMERA = "exact-two-camera"
    FULL_MISSION = "full-mission"
    RESUMED_COVERAGE_LEG = "resumed-coverage-leg"


@dataclass(frozen=True)
class ResolvedAutonomousRunMode:
    """Canonical mode and its legacy runner settings.

    ``execute`` only tells the caller whether the selected workflow can reach
    physical execution.  It is never proof that motion was authorized.
    """

    mode: AutonomousRunMode
    execute: bool
    coverage_leg_limit: int
    stop_after_coverage: bool
    authorization_scope: AutonomousAuthorizationScope

    @property
    def authorization_scope_text(self) -> str:
        """Human-readable scope for the typed-``RUN`` prompt and evidence."""

        if self.authorization_scope is AutonomousAuthorizationScope.NONE:
            return "no physical motion; first coverage leg dry-run only"
        if (
            self.authorization_scope
            is AutonomousAuthorizationScope.BOUNDED_COVERAGE
        ):
            return (
                f"at most {self.coverage_leg_limit} center-corridor "
                "coverage leg(s)"
            )
        if (
            self.authorization_scope
            is AutonomousAuthorizationScope.COVERAGE_ONLY
        ):
            return (
                "the complete center-corridor coverage pass, with no "
                "candidate-approach legs"
            )
        if (
            self.authorization_scope
            is AutonomousAuthorizationScope.EXACT_TWO_CAMERA
        ):
            return (
                "exactly two center-corridor coverage legs followed by the "
                "camera-approach phase for all LiDAR-admitted stands"
            )
        if (
            self.authorization_scope
            is AutonomousAuthorizationScope.RESUMED_COVERAGE_LEG
        ):
            return (
                "exactly one next center-corridor coverage leg from an "
                "admitted immutable checkpoint"
            )
        return "the complete multi-leg stand exploration mission"

    @property
    def camera_phase_enabled(self) -> bool:
        """Whether this mode may continue from coverage into camera motion."""

        return self.mode in {
            AutonomousRunMode.EXECUTE_EXACT_TWO_CAMERA,
            AutonomousRunMode.EXECUTE_FULL,
        }


_RESOLVED_MODES = {
    AutonomousRunMode.DRY_FIRST_LEG: ResolvedAutonomousRunMode(
        mode=AutonomousRunMode.DRY_FIRST_LEG,
        execute=False,
        coverage_leg_limit=0,
        stop_after_coverage=False,
        authorization_scope=AutonomousAuthorizationScope.NONE,
    ),
    AutonomousRunMode.EXECUTE_COVERAGE_ONLY: ResolvedAutonomousRunMode(
        mode=AutonomousRunMode.EXECUTE_COVERAGE_ONLY,
        execute=True,
        coverage_leg_limit=0,
        stop_after_coverage=True,
        authorization_scope=AutonomousAuthorizationScope.COVERAGE_ONLY,
    ),
    AutonomousRunMode.EXECUTE_EXACT_TWO_CAMERA: ResolvedAutonomousRunMode(
        mode=AutonomousRunMode.EXECUTE_EXACT_TWO_CAMERA,
        execute=True,
        coverage_leg_limit=2,
        stop_after_coverage=False,
        authorization_scope=AutonomousAuthorizationScope.EXACT_TWO_CAMERA,
    ),
    AutonomousRunMode.EXECUTE_FULL: ResolvedAutonomousRunMode(
        mode=AutonomousRunMode.EXECUTE_FULL,
        execute=True,
        coverage_leg_limit=0,
        stop_after_coverage=False,
        authorization_scope=AutonomousAuthorizationScope.FULL_MISSION,
    ),
    AutonomousRunMode.RESUME_NEXT_COVERAGE_LEG: ResolvedAutonomousRunMode(
        mode=AutonomousRunMode.RESUME_NEXT_COVERAGE_LEG,
        execute=True,
        coverage_leg_limit=1,
        stop_after_coverage=False,
        authorization_scope=(
            AutonomousAuthorizationScope.RESUMED_COVERAGE_LEG
        ),
    ),
}


def resolve_autonomous_run_mode(
    *,
    run_mode: AutonomousRunMode | str | None = None,
    execute: bool = False,
    coverage_leg_limit: int = 0,
    stop_after_coverage: bool = False,
) -> ResolvedAutonomousRunMode:
    """Resolve explicit and legacy CLI settings to one fail-closed contract.

    With no explicit ``run_mode``, the three legacy settings select the same
    workflow they historically represented.  With an explicit mode, asserted
    legacy flags may only agree with it.  A false ``execute`` or
    ``stop_after_coverage`` value is treated as an omitted ``store_true`` flag,
    so the explicit mode can replace the legacy spelling without redundant
    CLI arguments.
    """

    _validate_legacy_values(
        execute=execute,
        coverage_leg_limit=coverage_leg_limit,
        stop_after_coverage=stop_after_coverage,
    )
    explicit_mode = _parse_explicit_mode(run_mode)

    if explicit_mode is None:
        return _resolve_legacy_mode(
            execute=execute,
            coverage_leg_limit=coverage_leg_limit,
            stop_after_coverage=stop_after_coverage,
        )

    _validate_explicit_mode_compatibility(
        explicit_mode,
        execute=execute,
        coverage_leg_limit=coverage_leg_limit,
        stop_after_coverage=stop_after_coverage,
    )
    if explicit_mode is AutonomousRunMode.EXECUTE_COVERAGE_CHECKPOINT:
        return ResolvedAutonomousRunMode(
            mode=explicit_mode,
            execute=True,
            coverage_leg_limit=coverage_leg_limit,
            stop_after_coverage=False,
            authorization_scope=AutonomousAuthorizationScope.BOUNDED_COVERAGE,
        )
    return _RESOLVED_MODES[explicit_mode]


def validate_session_id_mode_label(
    session_id: str,
    resolved: ResolvedAutonomousRunMode,
) -> None:
    """Reject operator labels that contradict the canonical motion mode."""

    if not isinstance(session_id, str) or not _SAFE_SESSION_ID.fullmatch(
        session_id
    ):
        raise ValueError(
            "session_id must be a safe 1-128 character identifier using "
            "only letters, digits, dot, underscore, or hyphen"
        )
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", session_id.lower())
        if token
    }
    if resolved.execute and "dry" in tokens:
        raise ValueError(
            "physical execution session_id must not be labelled as dry"
        )
    if not resolved.execute and "execute" in tokens:
        raise ValueError(
            "dry-run session_id must not be labelled as execute"
        )


def validate_autonomous_viewpoint_scope(
    resolved: ResolvedAutonomousRunMode,
    *,
    exact_inspection_point_count: int | None,
) -> None:
    """Bind exact-two planning to an explicit compatible workflow.

    Exact-two planning is useful for focused geometry diagnostics, but it does
    not provide the redundant observations expected by a complete five-stand
    discovery pass.  The dedicated camera-continuation mode deliberately owns
    that reduced evidence contract and therefore requires exactly two points.
    Other complete-coverage modes still reject it so a stale shell option
    cannot silently narrow a full mission.
    """

    if not isinstance(resolved, ResolvedAutonomousRunMode):
        raise ValueError("resolved run mode is required")
    if resolved.mode is AutonomousRunMode.EXECUTE_EXACT_TWO_CAMERA:
        if (
            isinstance(exact_inspection_point_count, bool)
            or not isinstance(exact_inspection_point_count, int)
            or exact_inspection_point_count != 2
        ):
            raise ValueError(
                "--run-mode execute-exact-two-camera requires "
                "--exact-inspection-point-count 2"
            )
        return
    if exact_inspection_point_count is None:
        return
    if (
        isinstance(exact_inspection_point_count, bool)
        or not isinstance(exact_inspection_point_count, int)
        or exact_inspection_point_count != 2
    ):
        raise ValueError(
            "exact_inspection_point_count must be exactly 2 when supplied"
        )
    if resolved.mode in {
        AutonomousRunMode.EXECUTE_COVERAGE_ONLY,
        AutonomousRunMode.EXECUTE_FULL,
    }:
        raise ValueError(
            "--exact-inspection-point-count 2 is diagnostic-only and cannot "
            f"be used with --run-mode {resolved.mode.value}; omit it so stop "
            "spacing determines the complete viewpoint set"
        )


def _parse_explicit_mode(
    run_mode: AutonomousRunMode | str | None,
) -> AutonomousRunMode | None:
    if run_mode is None or run_mode == "":
        return None
    if isinstance(run_mode, AutonomousRunMode):
        return run_mode
    if not isinstance(run_mode, str):
        raise ValueError("run_mode must be a string, AutonomousRunMode, or None")
    try:
        return AutonomousRunMode(run_mode)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in AutonomousRunMode)
        raise ValueError(
            f"unknown autonomous run mode {run_mode!r}; choose one of: {choices}"
        ) from exc


def _validate_legacy_values(
    *,
    execute: bool,
    coverage_leg_limit: int,
    stop_after_coverage: bool,
) -> None:
    if not isinstance(execute, bool):
        raise ValueError("execute must be a boolean")
    if not isinstance(stop_after_coverage, bool):
        raise ValueError("stop_after_coverage must be a boolean")
    if isinstance(coverage_leg_limit, bool) or not isinstance(
        coverage_leg_limit, int
    ):
        raise ValueError("coverage_leg_limit must be an integer")
    if coverage_leg_limit < 0:
        raise ValueError("coverage_leg_limit must be non-negative")


def _resolve_legacy_mode(
    *,
    execute: bool,
    coverage_leg_limit: int,
    stop_after_coverage: bool,
) -> ResolvedAutonomousRunMode:
    if not execute:
        if coverage_leg_limit:
            raise ValueError("coverage_leg_limit requires an execution mode")
        if stop_after_coverage:
            raise ValueError("stop_after_coverage requires an execution mode")
        return _RESOLVED_MODES[AutonomousRunMode.DRY_FIRST_LEG]

    if coverage_leg_limit and stop_after_coverage:
        raise ValueError(
            "coverage_leg_limit and stop_after_coverage select different "
            "execution checkpoints"
        )
    if coverage_leg_limit:
        return ResolvedAutonomousRunMode(
            mode=AutonomousRunMode.EXECUTE_COVERAGE_CHECKPOINT,
            execute=True,
            coverage_leg_limit=coverage_leg_limit,
            stop_after_coverage=False,
            authorization_scope=AutonomousAuthorizationScope.BOUNDED_COVERAGE,
        )
    if stop_after_coverage:
        return _RESOLVED_MODES[AutonomousRunMode.EXECUTE_COVERAGE_ONLY]
    return _RESOLVED_MODES[AutonomousRunMode.EXECUTE_FULL]


def _validate_explicit_mode_compatibility(
    mode: AutonomousRunMode,
    *,
    execute: bool,
    coverage_leg_limit: int,
    stop_after_coverage: bool,
) -> None:
    if mode is AutonomousRunMode.DRY_FIRST_LEG:
        if execute or coverage_leg_limit or stop_after_coverage:
            raise ValueError(
                "dry-first-leg contradicts legacy physical-execution options"
            )
        return

    if mode is AutonomousRunMode.RESUME_NEXT_COVERAGE_LEG:
        if coverage_leg_limit or stop_after_coverage:
            raise ValueError(
                "resume-next-coverage-leg owns its one-leg checkpoint scope "
                "and contradicts legacy checkpoint options"
            )
        return

    if mode is AutonomousRunMode.EXECUTE_COVERAGE_CHECKPOINT:
        if coverage_leg_limit <= 0:
            raise ValueError(
                "execute-coverage-checkpoint requires a positive "
                "coverage_leg_limit"
            )
        if stop_after_coverage:
            raise ValueError(
                "execute-coverage-checkpoint contradicts stop_after_coverage"
            )
        return

    if mode is AutonomousRunMode.EXECUTE_EXACT_TWO_CAMERA:
        if coverage_leg_limit not in {0, 2}:
            raise ValueError(
                "execute-exact-two-camera owns its fixed two-leg coverage "
                "scope; coverage_leg_limit must be omitted or exactly 2"
            )
        if stop_after_coverage:
            raise ValueError(
                "execute-exact-two-camera requires the camera phase and "
                "contradicts stop_after_coverage"
            )
        return

    if coverage_leg_limit:
        raise ValueError(
            f"{mode.value} contradicts a positive coverage_leg_limit"
        )
    if mode is AutonomousRunMode.EXECUTE_FULL and stop_after_coverage:
        raise ValueError("execute-full contradicts stop_after_coverage")
