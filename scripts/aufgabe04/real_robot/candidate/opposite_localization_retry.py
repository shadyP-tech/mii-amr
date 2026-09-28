"""One stopped localization opportunity after opposite-route exhaustion.

The attempt callback must admit a fresh stationary frame and replan from the
original observer receipt for every epoch. Only exhausted, certified no-motion
uncertainty rejections qualify; malformed evidence and motion failures escape.
This policy does not lower clearance limits or issue a motion permit.
"""

from collections.abc import Callable, Mapping
from typing import TypeVar

from scripts.aufgabe04.real_robot.candidate.inspection_route_search import (
    CandidateInspectionRouteUnavailableError,
)


Result = TypeVar("Result")
OPPOSITE_UNCERTAINTY_EXHAUSTED = "opposite_route_uncertainty_exhausted"


def with_opposite_localization_retry(
    *, attempt: Callable[[int], Result], enabled: bool,
    event_sink: Callable[[Mapping[str, object]], None],
) -> Result:
    """Try at most two fresh planning epochs, never retrying after motion."""
    failures = []
    for epoch in range(2 if enabled else 1):
        try:
            return attempt(epoch)
        except CandidateInspectionRouteUnavailableError as exc:
            failures.append({"epoch": epoch, "reason": str(exc),
                             "reason_code": exc.reason_code, "evidence": exc.evidence})
            eligible = (
                exc.reason_code == OPPOSITE_UNCERTAINTY_EXHAUSTED
                and exc.evidence.get("motion_published") is False
                and exc.evidence.get("motion_permit_issued") is False
                and exc.evidence.get("no_motion_uncertainty_rejections_verified") is True
            )
            retry = enabled and epoch == 0 and eligible
            evidence = {
                "event": "opposite_localization_refresh_requested" if retry else "opposite_localization_recovery_exhausted",
                "planning_epoch": epoch, "maximum_refresh_count": 1,
                "fresh_stationary_frame_required": retry,
                "retained_source_orientation_unchanged": True,
                "route_limits_unchanged": True,
                "motion_authorized": False,
                "planning_epoch_failures": list(failures),
            }
            event_sink(evidence)
            if not retry:
                raise CandidateInspectionRouteUnavailableError(
                    str(exc), reason_code=exc.reason_code,
                    evidence={**exc.evidence, "planning_epoch_failures": failures},
                ) from exc
    raise AssertionError("bounded opposite localization loop did not terminate")
