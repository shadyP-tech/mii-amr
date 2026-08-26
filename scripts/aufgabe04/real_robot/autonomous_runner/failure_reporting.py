"""Pure fail-closed mission reporting for autonomous runner exceptions."""

from __future__ import annotations

from collections.abc import Mapping


def build_failed_closed_mission_summary(
    *,
    run_mode: str | None,
    error: BaseException,
) -> dict[str, object]:
    """Build one invariant-preserving terminal mission failure payload.

    Typed phase errors may contribute diagnostic fields, but they cannot
    overwrite the runner's fail-closed status, run mode, reason, or motion
    authorization result.  Diagnostic extraction is deliberately best-effort
    so a malformed error reporter cannot hide the original terminal failure.
    """

    failure: dict[str, object] = {}
    structured_fields = getattr(error, "to_failure_fields", None)
    if callable(structured_fields):
        try:
            candidate_fields = structured_fields()
        except Exception:
            candidate_fields = None
        if isinstance(candidate_fields, Mapping):
            failure.update(dict(candidate_fields))

    failure.update(
        {
            "schema_version": 1,
            "status": "failed_closed",
            "run_mode": run_mode,
            "reason": str(error),
            "motion_continues_authorized": False,
        }
    )
    return failure


__all__ = ["build_failed_closed_mission_summary"]
