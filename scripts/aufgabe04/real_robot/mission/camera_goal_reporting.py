"""Validate terminal QR-goal fields without conflating them with the pool.

The candidate executor authenticates the saved observations and completion
artifacts. This projection checks its goal against the admitted pool before a
parent summary may announce success. It does not authorize motion.
"""

from __future__ import annotations

from collections.abc import Mapping

from scripts.aufgabe04.navigation.coverage.candidate_inspection_pool import (
    candidate_inspection_pool_count_reasons,
    candidate_inspection_pool_policy_evidence,
)


def validate_completed_qr_goal(
    fields: Mapping[str, object],
    *,
    snapshot_sha256: str,
    coverage: Mapping[str, object] | None,
) -> None:
    expected = fields.get("expected_stand_count")
    pool_count = fields.get("candidate_pool_count")
    reasons = candidate_inspection_pool_count_reasons(expected, pool_count)
    if reasons:
        raise ValueError("invalid completed QR inspection pool: " + ", ".join(reasons))
    if fields.get("goal_completed") is not True or (
        type(fields.get("stand_count")) is not int
        or fields["stand_count"] != expected
    ):
        raise ValueError("completed camera mission has not met its distinct QR goal")
    confirmed = _uids(fields.get("confirmed_candidate_uids"), "confirmed candidates")
    if len(confirmed) != expected:
        raise ValueError("confirmed candidate count differs from the QR goal")
    for prefix in ("candidate_goal_progress", "confirmed_candidate_snapshot"):
        if not isinstance(fields.get(prefix), str) or not fields[prefix]:
            raise ValueError(f"completed camera mission is missing {prefix}")
        digest = fields.get(prefix + "_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise ValueError(f"completed camera mission has invalid {prefix} hash")
    if fields.get("candidate_snapshot_sha256", snapshot_sha256) != snapshot_sha256:
        raise ValueError("completed QR goal changed the full candidate pool snapshot")
    if coverage is None:
        return
    if coverage.get("expected_stand_count") != expected or (
        coverage.get("inspection_pool_policy")
        != candidate_inspection_pool_policy_evidence(expected)
    ):
        raise ValueError("completed QR goal differs from the admitted pool policy")
    pool = _uids(coverage.get("camera_seed_candidate_uids"), "camera seed candidates")
    if len(pool) != pool_count or not confirmed.issubset(pool):
        raise ValueError("confirmed candidates differ from the admitted inspection pool")
    remaining = _uids(fields.get("remaining_candidate_uids"), "remaining candidates")
    if remaining != pool - confirmed:
        raise ValueError("completed QR goal omits unresolved pool candidates")


def _uids(value: object, name: str) -> set[str]:
    if not isinstance(value, list) or any(
        not isinstance(uid, str) or not uid for uid in value
    ) or len(set(value)) != len(value):
        raise ValueError(f"{name} must be unique candidate UIDs")
    return set(value)
