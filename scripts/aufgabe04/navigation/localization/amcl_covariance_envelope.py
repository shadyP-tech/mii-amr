"""Conservative covariance envelopes from stationary AMCL samples.

This module is ROS-free.  It converts the covariance matrices already
captured by a stopped localization preflight into the same isotropic position
envelope and maximum yaw standard deviation used by route admission.  The
envelope deliberately dominates every accepted sample in every route-normal
direction; it is evidence for conservative planning, not an accuracy claim.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
)


def conservative_amcl_covariance_envelope(
    stationary_amcl_samples: Sequence[Mapping[str, object]],
) -> tuple[PlanarCovariance, float, dict[str, object]]:
    """Return a position envelope, yaw sigma, and deterministic evidence."""

    if not stationary_amcl_samples:
        raise ValueError("preflight has no accepted stationary AMCL samples")

    maximum_position_variance_m2 = 0.0
    maximum_yaw_variance_rad2 = 0.0
    sample_evidence: list[dict[str, object]] = []
    for index, sample in enumerate(stationary_amcl_samples):
        if not isinstance(sample, Mapping):
            raise ValueError(
                f"preflight AMCL sample {index} is not a mapping"
            )
        raw_covariance = sample.get("covariance")
        if not isinstance(raw_covariance, list) or len(raw_covariance) != 36:
            raise ValueError(
                f"preflight AMCL sample {index} covariance is incomplete"
            )
        try:
            values = [float(value) for value in raw_covariance]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"preflight AMCL sample {index} covariance is malformed"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"preflight AMCL sample {index} covariance is non-finite"
            )

        xx_m2 = values[0]
        xy_m2 = values[1]
        yx_m2 = values[6]
        yy_m2 = values[7]
        yaw_variance_rad2 = values[35]
        symmetry_tolerance = max(
            1.0e-12,
            1.0e-6 * max(abs(xy_m2), abs(yx_m2)),
        )
        if abs(xy_m2 - yx_m2) > symmetry_tolerance:
            raise ValueError(
                f"preflight AMCL sample {index} covariance is asymmetric"
            )
        covariance = PlanarCovariance(
            xx_m2,
            0.5 * (xy_m2 + yx_m2),
            yy_m2,
        )
        largest_position_variance_m2 = 0.5 * (
            covariance.xx_m2
            + covariance.yy_m2
            + math.hypot(
                covariance.xx_m2 - covariance.yy_m2,
                2.0 * covariance.xy_m2,
            )
        )
        if yaw_variance_rad2 < 0.0:
            raise ValueError(
                f"preflight AMCL sample {index} yaw covariance is negative"
            )

        maximum_position_variance_m2 = max(
            maximum_position_variance_m2,
            largest_position_variance_m2,
        )
        maximum_yaw_variance_rad2 = max(
            maximum_yaw_variance_rad2,
            yaw_variance_rad2,
        )
        sample_evidence.append(
            {
                "sample_index": index,
                "xx_m2": covariance.xx_m2,
                "xy_m2": covariance.xy_m2,
                "yy_m2": covariance.yy_m2,
                "yaw_variance_rad2": yaw_variance_rad2,
                "largest_position_variance_m2": (
                    largest_position_variance_m2
                ),
            }
        )

    covariance_envelope = PlanarCovariance(
        maximum_position_variance_m2,
        0.0,
        maximum_position_variance_m2,
    )
    return (
        covariance_envelope,
        math.sqrt(maximum_yaw_variance_rad2),
        {
            "envelope_kind": "isotropic_maximum_eigenvalue",
            "sample_count": len(sample_evidence),
            "samples": sample_evidence,
        },
    )


__all__ = ["conservative_amcl_covariance_envelope"]
