# Candidate camera acquisition optimizations — 2026-09-10

Implemented in the local working tree following the audit of
`stand_explore_exact2_camera_all5_20260910T145318Z` on baseline `cfd1439`.
This is offline implementation evidence, not a successful robot experiment.

## Processing and module boundaries

1. Try the existing projected ROI. If acquisition or head refinement fails,
   search the existing bounded wider ROI for current 2D head geometry.
2. Proposals need physical scale, four raw borders, supported corners and a
   paired neck. They have no QR identity, pose, angle or motion authority.
   Ambiguous proposals stop that image's search.
3. Associate the proposal's camera bearing with the original candidate using
   the unchanged displacement, bearing, range, fresh-scan and unique-cluster
   gates. A positive proposal rejected by association stops further metric
   fitting on that image; it cannot become a camera measurement.
4. Recenter and shrink the crop around the complete observed head, neck and
   margin. Transform corners and principal point into that crop exactly once.
5. A verified current QR marker plus the independent head proposal may seed
   strict raw-border refinement before a successful QR pose exists. The
   ordinary joint QR/head fit, reprojection limit and ambiguity tests remain
   authoritative. QR-free observations retain the existing undirected
   backside path and its head/neck checks.

| Module | Responsibility |
| --- | --- |
| `perception/stand_axis/head_proposal.py` | Bounded grayscale/raw line proposals, four-border/corner and neck validation, complete crop bounds |
| `perception/stand_axis/head_border_seed.py` | Validated crop-local 2D seed selection for current raw refinement |
| `perception/stand_axis/qr_marker_validation.py` | Current-pixel finder-pattern verification, independent of decoding and PnP |
| `real_robot/observer/head_proposal_registration.py` | Candidate association before strict fitting, crop coordinates and qualified framing hints |
| `real_robot/observer/roi_qr_evidence.py` | Preserve marker vetoes and identity conflicts across crops of the same image |
| `real_robot/observer/camera_framing.py` | Validate an advisory framing envelope from observed head extent and current camera geometry |
| `real_robot/candidate/camera_distance_recovery.py` | Select one bounded outward search while preserving the viewing bearing |

Paths in this table are relative to `scripts/aufgabe04/`. ROS orchestration,
status output and the existing candidate controller call these modules.

## Front evidence and distance recovery

A tentative OpenCV quadrilateral still withholds a backside axis for its
current frame. Only verified finder patterns or decoded text set the persistent
front-marker latch. The original backside frame 000022's false quadrilateral
has zero verified finder patterns and no longer poisons later QR-free frames.

Recentring cannot erase a marker or conflict seen in another evaluated crop.
The observer merges conservative current-image veto evidence and checks
conflicting identities, same-crop multiplicity and disjoint symbols. Target
identity still requires the selected symbol's own QR geometry and LiDAR binding.

An outward recovery is offered only after a fresh, accepted, unpoisoned frame
has a candidate-associated head, a verified front marker and a rejected strict
joint QR/head fit. It is advisory; no rejected angle is used. The hint is cleared
on motion, sensor-contract resets, identity poison or usable geometry. It is
carried through status and hashed advisory inspection receipts.

The controller offers at most three outward standoffs once per candidate,
within the existing shared route budget and camera view budget. There must be
at least 0.10 m of outward room toward the configured preferred range. A
materialized goal must increase range by at least 0.08 m and preserve the
viewing bearing within 5 degrees. Existing keepouts, route clearance, fresh
planning, child motion admission and arrival checks still apply. Certified
backside opposite-side inspection retains priority. Blocked outward goals
return to the normal bounded view search; they never authorize a closer move.

## Recorded replay results and remaining blockers

Original compressed frame 000045 and its saved CameraInfo, scan and TF were
used without modifying source bytes. The automatic proposal recovers the head
through approximately x=581, beyond the nominal crop's right edge near x=525.
The recentered crop is `[431, 191, 597, 390]` in the 800×600 rectified image.

The **actual candidate association still fails** on this frame. Its camera
bearing correction is 9.347 degrees, inside the 12-degree limit, but the nearest
eligible scan return is 0.473 m against an upper candidate range of 0.454876 m:
an 18.124 mm discrepancy. No strict registered measurement or distance hint is
authorized from this frame.

In three local native-only replay trials, the old selection took a median
755.95 ms and three metric evaluations. The new selection took 389.17 ms and
one metric evaluation, stopping after the rejected early association. These
are macOS/OpenCV 4.13 timings; WeChat was unavailable. They are not robot latency
measurements and cannot be compared directly with the recorded live timings.

A separate, explicitly unbound diagnostic of the recentered crop decodes
`Start` and passes raw-border refinement. Its joint QR/head RMSE remains
3.737 px, above the unchanged 2 px limit. Head-only and QR-only diagnostic fits
are individually lower, but neither replaces the failed joint fit. Physical
head versus paper/QR association, corner bias and calibration still need
independent measurement. The code does not adjust the physical model or widen
LiDAR tolerances to make this recording pass.

Replay scripts and machine-readable evidence are in the local audit's
`derived/neutral_head_proposal_replay.{py,json}` and
`derived/head_proposal_handoff_replay.{py,json}`. Original compressed-image and
metadata hashes remained unchanged. Portable source-provenance fixtures are
included with the regression tests.

## Validation and next robot evidence

The final combined run passed **286 tests across 24 focused suites** using
Python 3.12.14, OpenCV 4.13.0 and NumPy 2.5.3. `git diff --check` passed.

Focused tests cover automatic recorded acquisition, malformed/ambiguous
proposals, missing raw support, false-marker latch recovery, cross-crop QR
conflicts, crop intrinsics, unavailable QR pose, strict joint-fit rejection,
freshness and poison handling, association before fitting, admission of
unresolved registered frames, route budgets, occupied and ineffective outward
goals, ordinary backside/opposite behavior and status/receipt serialization.

No additional CLI options are needed. The standard camera experiment enables
this path through the existing bounded reacquisition configuration. The robot
checkout must contain these working-tree changes before execution.

The next useful hardware evidence remains one associated candidate approach,
stopped camera observation and validated QR/facing pose. Record acquisition
timing, final crop, marker-verification reason, early camera/LiDAR association
and the joint fit diagnostics. If association or geometry still disagrees,
measure that discrepancy before attempting five-station completion.
