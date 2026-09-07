"""Bounded topology proposal diversity for target-associated backside heads.

These images only locate proposals. Neither morphology nor fallback pixels
constitute a head measurement; the acquisition module still proves the four
raw sides, paired neck, metric scale/centre, and unambiguous planar pose.
"""

from __future__ import annotations

from scripts.aufgabe04.perception.stand_axis.preprocessing import (
    _edge_topology_hypotheses,
)


def backside_topology_proposal_batches(
    cv2,
    filtered_seed,
    raw_edges,
    *,
    edge_preprocess: str,
):
    """Yield the preferred locator, then one bounded boundary-recovery batch.

    The caller must stop after the first batch with a raw-supported head. A
    texture-rich recovery proposal must not displace an existing filtered
    proposal or provide a second chance after a metric/pose rejection.

    Heavy texture suppression can erase the complete outer sides of a small,
    low-contrast head. In that case only, the original Canny edges may locate
    additional hypotheses inside the already projected target crop. Reusing
    the same three bounded morphology variants keeps the search finite and
    leaves the authoritative Canny image unchanged.
    """

    yield _edge_topology_hypotheses(
        cv2,
        filtered_seed,
        close_kernel=3,
        close_iterations=1,
        include_gap_recovery=True,
    )
    if edge_preprocess != "channel_union":
        return
    if cv2.countNonZero(cv2.bitwise_xor(filtered_seed, raw_edges)) == 0:
        return
    yield _edge_topology_hypotheses(
        cv2,
        raw_edges,
        close_kernel=3,
        close_iterations=1,
        include_gap_recovery=True,
    )
