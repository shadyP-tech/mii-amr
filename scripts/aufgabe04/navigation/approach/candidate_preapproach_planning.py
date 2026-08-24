"""Compatibility facade for camera candidate pre-approach planning.

The implementation is split by responsibility:

* immutable route-preview contracts live in ``candidate_preapproach_models``;
* ROS-free, no-write planning lives in ``candidate_preapproach_compute``;
* filesystem writes and sealing live in ``candidate_preapproach_materialization``.

Existing callers may continue importing the original public API from this
module while the smaller modules provide explicit dependency boundaries.
"""

from scripts.aufgabe04.navigation.approach.candidate_preapproach_compute import (
    compute_candidate_preapproach_plan,
    load_candidate_planning_context,
    route_turn_metrics,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_materialization import (
    materialize_candidate_preapproach_plan,
    plan_candidate_preapproach,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_models import (
    CandidatePlanningContext,
    CandidatePreapproachPlan,
    CandidatePreapproachUnreachableError,
)


__all__ = [
    "CandidatePlanningContext",
    "CandidatePreapproachPlan",
    "CandidatePreapproachUnreachableError",
    "compute_candidate_preapproach_plan",
    "load_candidate_planning_context",
    "materialize_candidate_preapproach_plan",
    "plan_candidate_preapproach",
    "route_turn_metrics",
]
