"""Shared physical defaults for route-uncertainty admission.

These constants are one argv contract shared by startup route selection and
the dry/live child admission.  They remain conservative certified allocations;
centralizing them prevents a planning-only optimization from silently using a
different budget than execution.
"""

DEFAULT_TRACKING_TUBE_RADIUS_M = 0.03
DEFAULT_COLLISION_MARGIN_M = 0.02
DEFAULT_UNCERTAINTY_ODOM_DRIFT_BOUND_M = 0.02
DEFAULT_UNCERTAINTY_BRAKING_LATENCY_DISTANCE_M = 0.015
DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M = 0.005
DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER = 2.0


__all__ = [
    "DEFAULT_COLLISION_MARGIN_M",
    "DEFAULT_TRACKING_TUBE_RADIUS_M",
    "DEFAULT_UNCERTAINTY_BRAKING_LATENCY_DISTANCE_M",
    "DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M",
    "DEFAULT_UNCERTAINTY_ODOM_DRIFT_BOUND_M",
    "DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER",
]
