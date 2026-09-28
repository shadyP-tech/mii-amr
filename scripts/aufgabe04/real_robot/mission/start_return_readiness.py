"""Stopped-localization inputs for the faster, unloaded Start return."""

from dataclasses import replace

from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import CandidateRouteUncertaintyContext
from scripts.aufgabe04.navigation.control.return_to_start_speed_policy import (
    RETURN_TO_START_BRAKING_LATENCY_DISTANCE_M, RETURN_TO_START_SPEED_POLICY,
)
from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import (
    CandidateRouteUncertaintyReadinessRequest, load_candidate_route_uncertainty_readiness,
)


def load_start_return_readiness(
    request: CandidateRouteUncertaintyReadinessRequest,
) -> CandidateRouteUncertaintyContext:
    """Keep the shared child defaults, with the return policy's larger reserve."""
    context = load_candidate_route_uncertainty_readiness(request)
    return replace(
        context,
        admission_config=replace(context.admission_config,
            braking_latency_distance_m=RETURN_TO_START_BRAKING_LATENCY_DISTANCE_M),
        source_evidence={
            **context.source_evidence, "motion_speed_policy": RETURN_TO_START_SPEED_POLICY,
            "braking_latency_distance_m": RETURN_TO_START_BRAKING_LATENCY_DISTANCE_M,
        },
    )
