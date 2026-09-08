"""Motion-neutral count policy for inspecting LiDAR hypotheses for QR stands.

A LiDAR hypothesis is not a decoded stand identity. Every retained hypothesis
must remain eligible for camera inspection, including boundary proposals. The
finite pool cap bounds work without selecting an arbitrary best-looking subset.
"""

from __future__ import annotations

CANDIDATE_INSPECTION_POOL_POLICY_ID = "all_usable_candidates_up_to_twice_qr_goal_v1"
CANDIDATE_INSPECTION_POOL_MULTIPLIER = 2


def candidate_inspection_pool_limit(expected_stand_count: int) -> int:
    """Return the sealed default cap; the expected count is the QR goal."""

    if type(expected_stand_count) is not int or expected_stand_count <= 0:
        raise ValueError("expected_stand_count must be a positive integer")
    return CANDIDATE_INSPECTION_POOL_MULTIPLIER * expected_stand_count


def candidate_inspection_pool_count_reasons(
    expected_stand_count: object, candidate_count: object
) -> tuple[str, ...]:
    """Fail closed for malformed counts, a deficit, or an oversized pool."""

    if expected_stand_count is None:
        return ("expected_stand_count_missing",)
    if type(expected_stand_count) is not int:
        return ("expected_stand_count_not_integer",)
    if expected_stand_count <= 0:
        return ("expected_stand_count_not_positive",)
    if type(candidate_count) is not int or candidate_count < 0:
        return ("inspection_pool_candidate_count_invalid",)
    if candidate_count < expected_stand_count:
        return ("usable_candidate_count_below_expected",)
    if candidate_count > candidate_inspection_pool_limit(expected_stand_count):
        return ("inspection_pool_candidate_count_exceeds_limit",)
    return ()


def candidate_inspection_pool_policy_evidence(
    expected_stand_count: int | None,
) -> dict[str, object]:
    """Policy parameters recorded in, and hashed with, every admission."""

    return {
        "policy_id": CANDIDATE_INSPECTION_POOL_POLICY_ID,
        "expected_distinct_qr_count": expected_stand_count,
        "inspection_pool_limit": (
            candidate_inspection_pool_limit(expected_stand_count)
            if type(expected_stand_count) is int and expected_stand_count > 0
            else None
        ),
        "selection_basis": "all_usable_strict_and_boundary_candidates",
        "motion_authorized": False,
    }
