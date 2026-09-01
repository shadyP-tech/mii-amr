"""Durable publication boundary for route-uncertainty admission evidence.

This module performs no planning and publishes no commands.  It atomically
stores the complete budget payload before exposing a rejected decision to the
station-segment runtime.  Accepted decisions return their artifact identity;
rejected decisions raise a typed error carrying the same durable identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.navigation.foundation.content_hashed_evidence import (
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionResult,
)


ROUTE_UNCERTAINTY_ARTIFACT_HASH_FIELD = "route_uncertainty_artifact_sha256"


class RouteUncertaintyAdmissionRejected(ValueError):
    """Typed no-motion rejection with durable uncertainty evidence identity."""

    def __init__(
        self,
        *,
        limiting_segment_id: str,
        remaining_margin_m: float | None,
        uncertainty_budget_json: Path,
        uncertainty_budget_sha256: str,
    ) -> None:
        margin_text = (
            "unknown"
            if remaining_margin_m is None
            else f"{remaining_margin_m:.6f} m"
        )
        super().__init__(
            "route uncertainty budget exhausted: "
            f"limiting_segment={limiting_segment_id} "
            f"remaining_margin={margin_text}"
        )
        self.limiting_segment_id = limiting_segment_id
        self.remaining_margin_m = remaining_margin_m
        self.uncertainty_budget_json = Path(uncertainty_budget_json)
        self.uncertainty_budget_sha256 = uncertainty_budget_sha256

    def to_stop_details(self) -> dict[str, object]:
        return {
            "uncertainty_budget_accepted": False,
            "uncertainty_budget_json": str(self.uncertainty_budget_json),
            "uncertainty_budget_sha256": self.uncertainty_budget_sha256,
            "route_uncertainty_limiting_segment_id": self.limiting_segment_id,
            "route_uncertainty_remaining_margin_m": self.remaining_margin_m,
        }


def publish_route_uncertainty_budget(
    path: Path,
    *,
    payload: Mapping[str, object],
    admission: RouteUncertaintyAdmissionResult,
) -> str:
    """Persist one decision and return its hash or raise a typed rejection.

    The payload must embed the exact admission evidence supplied separately.
    This prevents callers from publishing one decision while classifying a
    different in-memory result.
    """

    expected_admission = admission.to_evidence_dict()
    if payload.get("admission") != expected_admission:
        raise ValueError(
            "route uncertainty payload differs from the evaluated admission"
        )
    destination = Path(path)
    digest = write_content_hashed_json(
        destination,
        payload,
        hash_field=ROUTE_UNCERTAINTY_ARTIFACT_HASH_FIELD,
    )
    if admission.decision.accepted:
        return digest
    raise RouteUncertaintyAdmissionRejected(
        limiting_segment_id=(
            admission.decision.limiting_segment_id or "unknown"
        ),
        remaining_margin_m=admission.decision.remaining_margin_m,
        uncertainty_budget_json=destination,
        uncertainty_budget_sha256=digest,
    )


__all__ = [
    "ROUTE_UNCERTAINTY_ARTIFACT_HASH_FIELD",
    "RouteUncertaintyAdmissionRejected",
    "publish_route_uncertainty_budget",
]
