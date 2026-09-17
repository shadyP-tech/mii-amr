# Full-image candidate screening — 2026-09-17

This report records the implementation committed as `ebd73d8`. The subsequent
[guided-hint implementation](aufgabe04_guided_head_hints_20260917.md) adds current
scan filtering and a final bounded observed-pair fallback after both searches
described here complete without a head or unresolved alternative. Results below
retain their original baseline and experiment scope.

Following the user's request to test early candidate association, camera
exploration now supplies an optional conservative candidate screen to the
shared full-image geometry pipeline. The debug viewer leaves this screen unset.
The original rectified image, intrinsics, Canny edges, raw border checks,
physical refinement and 3D quality gates remain unchanged. Initial acquisition
retains the existing rail distribution across scales and image regions. Only
after a completed candidate miss with no accepted or unresolved head may one
retry prioritize detected segment lengths near the projected head size. That
retry uses the same full image and original deadline, and only the strict
verification slots left from the original twelve. It never retries a success,
ambiguity, budget exhaustion or deadline failure. Size ordering is a search
priority, not proof that a segment belongs to the stand; no positional crop is
added. Unscreened viewer acquisition does not use this retry.

The screen uses the current candidate projection to exclude locations that
cannot satisfy the observer's existing post-fit association bounds. Those
bounds are an average measured side height between **0.60 and 1.35** times the
projected height, side balance of at least **0.65**, and the configured radial
center offset (normally 1.5 head heights). There is no added vertical cap.

Rough rectangles receive a wider bound: raw refinement can change their height
by 0.85–1.15, the enclosing-border search can grow the same hint by up to 1.25,
and the measured center can move by up to ten pixels. A hint survives whenever
this possible outcome overlaps the final size and center limits. Completed
measurements must satisfy the original exact association bounds.

Supported rejected hints remain in the existing texture comparison pool;
small printed rectangles can still explain interior edge combinations without
consuming a strict head-verification slot. Two surviving independent heads
remain ambiguous. Projection supplies no accepted corners or angle. The final
current scan association, freshness, stationary checks and identity binding are
still required before a candidate is admitted.

Invalid, missing or behind-camera projections leave the ordinary full-image
path available and remain rejected by the later association stage. The screen
is included in the exact-image geometry cache context so a changed candidate
cannot reuse a result selected for another candidate.

Regression coverage checks refinement-envelope endpoints, displaced heads,
the exact current post-fit scale-gate decision matrix, full-image/intrinsics
preservation, texture retention, measured rechecks, competing-head ambiguity,
invalid projections and cache separation. Saved-image replay and live timing
results must be reported separately; this screen alone does not establish a
successful hardware admission.

## Recording audit and workload reduction

The workstation audit replays 150 original calibrated images against deployed
commit `35aa6aab16dac065df0d08ff9731e1e8643c9939`, then loads the local changes
through an in-memory overlay. The repository, recordings and robot processes
on the workstation are unchanged. The five recordings include windows, blinds,
radiators, the opposite-side inspection arrival and an off-center foreground
head. Candidate projections come from recorded mission metadata; no hand-picked
center, diagnostic magenta corners, QR coordinates or previous pose is supplied.

The baseline spends all twelve strict checks on background rectangles in
debug recording `154824`, frame 0. In frame 25 it reaches the stand only on
check eleven, then refuses selection because untested alternatives remain.
Opposite-arrival frame 6 stops at 1,036 raw contours before examining the head.
These are proposal-budget failures before a unique 3D head can be selected.

In addition to the candidate screen:

- Contours shorter than the existing minimum four-side perimeter are rejected
  **before** charging the 1,024-contour quota. Their pixels remain available for
  all raw-border checks and texture evidence.
- Duplicate line comparisons are vectorized while retaining the exact greedy
  longest-first representatives. Exact repeated quadrilateral seeds are removed.
- Each finite raw side fit reads only an enclosing box of its existing corridor.
  Global pixel coordinates, point order and the original exact predicates are
  preserved. Paths that extend endpoints retain their full-image input.
- Distance-transform support uses the quadrilateral bounds plus a conservative
  tolerance margin. This preserves the original supported/unsupported decision;
  it does not erase background pixels or manufacture a border.

The optimization-only replay preserves all 54 previously usable geometries,
including corners, angle, quality, outer-border evidence and strict checks.
Three repeated calls on the same first/middle images reduce median unconditioned
geometry time from 203 to 140 ms and 197 to 130 ms in the two debug recordings,
and from 149 to 89 ms and 197 to 119 ms in the first-inspection recordings.

Unconditional size priority was tested and rejected: it starved short rails on
a foreshortened/neck-split head, and exposed extra mixed inner/outer panel hints
that reduced usable frames in recording `133025` from 53 to 43. The final bounded
retry retains the original first search and cannot replace a success or resolve
an ambiguity by trying a different ordering.

## Final replay result

| Recording | Usable geometry before → after | Median geometry time before → after |
| --- | ---: | ---: |
| Opposite-side arrival `093041` | 0/15 → **5/15** | 2.7 ms premature abort → **147 ms completed search** |
| First inspection `133025` | 53/55 → **54/55** | 146 → **53 ms** |
| Off-center inspection `150723` | 1/10 → **1/10** | 193 → **91 ms** |

These are full-set unprofiled cold calls, one per image, with preparation timing
separate. The very short opposite-side baseline usually aborted at the contour
limit and is not a successful-search latency. All 54 previously usable fits
retain their geometry, quality and enclosing-border evidence within absolute
tolerance `1e-8` / relative tolerance `1e-10`. Relative to the optimization-only
variant, all 70 debug images retain their ordered strict checks and receive no
invented candidate context.

The five newly acquired opposite-side frames are 8, 9, 33, 34 and 44. They use
10, 8, 7, 5 and 6 strict checks respectively **including both attempts**. The
returned corners cover the foreground frame at approximately x=391–497,
y=228–339, with measured yaw approximately 18.3 degrees. No replay exceeds the
twelve-check limit. The other ten opposite-side captures remain ambiguous;
the implementation does not choose a border simply because it is nearer the
projected center.

This validates current-image geometry and bounded acquisition, not completed
robot admission. The replay does not exercise live scheduling/source ages,
QR identity, current-scan association or stationary consensus. Those existing
gates remain required. No robot deployment or motion was performed.

Validation: **187 targeted tests and 353 subtests pass**. This includes unchanged
pixel-support decisions, displaced/perspective and neck-split heads, candidate
cache separation, admission wiring, ambiguity rejection and real-pixel retry
tests with eleven checks already consumed before retry. Deadline expiry never
publishes a partial result. `git diff --check` passes. All eight production
module hashes in the final workstation replay match the working tree.

Detailed inputs, source hashes, intermediate experiments and timing caveats are
in [the replay audit](../../results/aufgabe04/implementation_checks/background_edges_20260917/README.md).
