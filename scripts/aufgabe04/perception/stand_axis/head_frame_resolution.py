"""Resolve completed current-border recoveries without hiding competing frames.

Enclosure and similar scale only locate a refinement search. They never prove
that two hints or two measured frames describe the same physical boundary.
Aliases require current supporting-pixel agreement. A completed raw recovery
can additionally replace its measured original; ambiguous rail bridges keep all
of their evidence for the ordinary head selection policy.
"""

from dataclasses import dataclass

from scripts.aufgabe04.perception.stand_axis.geometry import _polygon_area
from scripts.aufgabe04.perception.stand_axis.head_outer_border import (
    HeadOuterBorderEvidence, _encloses, validated_current_head_boundary,
)


@dataclass(frozen=True)
class CurrentFrameResolutionRecord:
    """One completed current-pixel refinement, local to one acquisition call."""

    hint_corners: tuple
    original_corners: tuple
    proposal: object
    recovery_evidence: object = None


def resolve_measured_current_frames(measured, resolutions, families):
    """Apply completed raw recoveries to their already measured original rails.

    Different location hints can measure the same initial frame, while only one
    hint's outward search finds its enclosing border. That completed search can
    resolve the other *measured* original. Its final proposal must already have
    passed the caller's candidate association; no new corners are constructed.

    This is a single substitution pass, not a transitive enclosure relation.
    Conflicting originals or final frames stay ambiguous. Untested hints still
    use the stricter neutral-hint policy in ``resolved_tested_head_hint``.
    """
    accepted_ids = {id(item) for item in measured}
    recoveries = []
    for entry in resolutions:
        evidence = entry.recovery_evidence
        if (id(entry.proposal) not in accepted_ids or not isinstance(evidence, HeadOuterBorderEvidence)
                or evidence.recovered is not True
                or evidence.original_corners != entry.original_corners
                or not validated_current_head_boundary(evidence,
                    corners=entry.proposal.corners, profile_sha256=evidence.profile_sha256)):
            continue
        original, recovered = entry.original_corners, entry.proposal.corners
        if (not 1.03 * _polygon_area(original) <= _polygon_area(recovered)
                <= 1.70 * _polygon_area(original) or not _encloses(recovered, original)):
            continue
        recoveries.append(entry)
    resolved = []
    for proposal in measured:
        matches = [entry for entry in recoveries
                   if families.same(proposal.corners, entry.original_corners)]
        compatible = bool(matches) and all(
            first.recovery_evidence.profile_sha256 == second.recovery_evidence.profile_sha256
            and families.same(first.original_corners, second.original_corners)
            and families.same(first.proposal.corners, second.proposal.corners)
            for index, first in enumerate(matches) for second in matches[index + 1:])
        resolved.append(max(matches, key=lambda entry: _polygon_area(entry.proposal.corners)).proposal
                        if compatible else proposal)
    return resolved


def resolved_tested_head_hint(corners, resolutions, families):
    """Reuse a tested origin's canonical frame only on the same observed rails.

    A strict current refinement can resolve an inset locator to its physical
    frame. Cover only aliases of both its tested hint and its initial measured
    rails: the outer search still depends on the neutral hint. Matching only
    the initial measurement would permit a different outward search. No
    untested enclosure creates a mapping, and conflicting current rails or
    final physical frames remain unresolved.
    """
    matches = [entry for entry in resolutions if all(
        _polygon_area(corners) <= 1.03 * _polygon_area(origin)
        and families.same(corners, origin)
        for origin in (entry.hint_corners, entry.original_corners))
        and families.same(entry.hint_corners, entry.original_corners)]
    if not matches:
        return None
    for index, first in enumerate(matches):
        for second in matches[index + 1:]:
            if (not families.same(first.original_corners, second.original_corners)
                    or not families.same(first.hint_corners, second.hint_corners)
                    or not families.same(first.proposal.corners, second.proposal.corners)):
                return None
    return max(matches, key=lambda entry: _polygon_area(entry.proposal.corners)).proposal


def resolved_current_frame(corners, measured, families):
    """Cover a hint only by the same observed rails and the existing area guard.

    An untested outward rectangle cannot inherit verification from an inset
    frame. Agreement among matches must also be pairwise: one intermediate
    hint cannot merge two otherwise different current border families.
    """
    matches = [item for item in measured
               if _polygon_area(corners) <= 1.03 * _polygon_area(item.corners)
               and families.same(corners, item.corners)]
    if not matches or any(not families.same(first.corners, second.corners)
                          for index, first in enumerate(matches)
                          for second in matches[index + 1:]):
        return None
    return max(matches, key=lambda item: _polygon_area(item.corners))


def distinct_current_frames(measured, families):
    """Collapse only isolated, fully pairwise-agreeing measured-frame aliases.

    Current rail agreement need not be transitive. If A agrees with B and C,
    but B and C disagree, discarding B could hide that ambiguity from the next
    selection stage. Preserve every proposal in such a component; connectivity
    is used solely to find these conflicts, never to authorize a merge.
    """
    ordered = sorted(measured, key=lambda item: -_polygon_area(item.corners))
    neighbours = [{index} for index in range(len(ordered))]
    for index, first in enumerate(ordered):
        for other, second in enumerate(ordered[index + 1:], index + 1):
            if families.same(first.corners, second.corners):
                neighbours[index].add(other)
                neighbours[other].add(index)
    remaining = set(range(len(ordered)))
    selected = []
    while remaining:
        component, pending = set(), {min(remaining)}
        while pending:
            current = pending.pop()
            component.add(current)
            pending.update(neighbours[current] - component)
        remaining.difference_update(component)
        if all(component <= neighbours[index] for index in component):
            selected.append(ordered[min(component)])
        else:
            selected.extend(ordered[index] for index in sorted(component))
    return selected
