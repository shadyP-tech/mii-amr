"""Pure localization ownership decisions for Aufgabe 04 ROS preflight."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence


LOCALIZATION_SOURCE_AMCL = "amcl"
LOCALIZATION_SOURCE_TF = "tf"

FAIL_UNSUPPORTED_SOURCE = "localization ownership: unsupported localization source"
FAIL_AMCL_STALE = "localization ownership: amcl data missing or stale"
FAIL_MAP_TO_ODOM = "localization ownership: dynamic map->odom unavailable"
FAIL_ROUTE_TRANSFORM = "localization ownership: route transform unavailable"
FAIL_AMCL_WITH_EXTERNAL_TF = "localization ownership: amcl conflicts with external tf owner"
FAIL_TF_WITH_AMCL = "localization ownership: tf conflicts with fresh amcl"
FAIL_AMBIGUOUS = "localization ownership: ambiguous owner evidence"


@dataclass(frozen=True)
class LocalizationOwnershipEvidence:
    localization_source: str
    amcl_fresh: bool
    map_to_odom_dynamic_fresh: bool
    route_transform_fresh: bool
    odom_to_base_fresh: bool = False
    map_odom_identity: bool = False
    external_tf_owner_candidates: Sequence[str] = field(default_factory=tuple)
    ambiguous_owner_evidence: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class LocalizationOwnershipDecision:
    ok: bool
    failure: str
    data: Dict[str, object]


def evaluate_localization_ownership(
    evidence: LocalizationOwnershipEvidence,
) -> LocalizationOwnershipDecision:
    """Return a stable pass/fail decision from ROS-free primitive evidence."""
    source = evidence.localization_source
    external_candidates = sorted(set(evidence.external_tf_owner_candidates))
    ambiguous_evidence = sorted(set(evidence.ambiguous_owner_evidence))

    data: Dict[str, object] = {
        "localization_source": source,
        "amcl_fresh": evidence.amcl_fresh,
        "map_to_odom_dynamic_fresh": evidence.map_to_odom_dynamic_fresh,
        "route_transform_fresh": evidence.route_transform_fresh,
        "odom_to_base_fresh": evidence.odom_to_base_fresh,
        "map_odom_identity": evidence.map_odom_identity,
        "external_tf_owner_candidates": external_candidates,
        "ambiguous_owner_evidence": ambiguous_evidence,
    }

    failure = _localization_ownership_failure(
        source=source,
        amcl_fresh=evidence.amcl_fresh,
        map_to_odom_dynamic_fresh=evidence.map_to_odom_dynamic_fresh,
        route_transform_fresh=evidence.route_transform_fresh,
        map_odom_identity=evidence.map_odom_identity,
        external_tf_owner_candidates=external_candidates,
        ambiguous_owner_evidence=ambiguous_evidence,
    )
    return LocalizationOwnershipDecision(ok=not failure, failure=failure, data=data)


def _localization_ownership_failure(
    *,
    source: str,
    amcl_fresh: bool,
    map_to_odom_dynamic_fresh: bool,
    route_transform_fresh: bool,
    map_odom_identity: bool,
    external_tf_owner_candidates: Sequence[str],
    ambiguous_owner_evidence: Sequence[str],
) -> str:
    if source not in (LOCALIZATION_SOURCE_AMCL, LOCALIZATION_SOURCE_TF):
        return FAIL_UNSUPPORTED_SOURCE
    if ambiguous_owner_evidence:
        return FAIL_AMBIGUOUS
    if not route_transform_fresh:
        return FAIL_ROUTE_TRANSFORM
    # An odom-frame route needs no localization owner between map and odom:
    # they are the same frame, so the relationship is the identity.  AMCL or
    # SLAM freshness must not be required merely to prove that identity.
    if map_odom_identity and source == LOCALIZATION_SOURCE_TF:
        return ""
    if not map_to_odom_dynamic_fresh:
        return FAIL_MAP_TO_ODOM
    if source == LOCALIZATION_SOURCE_AMCL:
        if not amcl_fresh:
            return FAIL_AMCL_STALE
        if external_tf_owner_candidates:
            return FAIL_AMCL_WITH_EXTERNAL_TF
        return ""
    if amcl_fresh:
        return FAIL_TF_WITH_AMCL
    return ""
