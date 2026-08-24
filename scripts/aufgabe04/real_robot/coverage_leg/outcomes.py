"""Pure classification and completion checks for child outcomes."""

from __future__ import annotations

from collections.abc import Mapping

from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome

def _require_completed_motion(outcome: MotionLegOutcome) -> None:
    if outcome.status != "completed":
        raise RuntimeError(
            f"physical route failed for {outcome.run_id}: {outcome.stop_reason}"
        )

def _claims_prestart_localization_phase(stop_details: object) -> bool:
    """Keep malformed before-motion evidence out of runtime recovery."""

    if not isinstance(stop_details, Mapping):
        return False
    return (
        stop_details.get("execution_phase") == "before_motion"
        or stop_details.get("phase") == "initial_runtime_input_wait"
    )

