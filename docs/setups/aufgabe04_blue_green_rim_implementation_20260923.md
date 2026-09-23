# Blue/green head acquisition correction — 2026-09-23

Implemented the correction investigated in
[the blue/green rim report](aufgabe04_blue_green_rim_solution_20260923.md).
No robot motion, ROS nodes, or workstation-checkout deployment was performed.

## Acquisition and measurement

`perception/stand_axis/material_rim_locator.py` finds enclosed, head-scale holes
in individual chromatic material masks, inside the current candidate bounds.
The automatic search considers blue, green, both red hue ranges, and purple;
it does not receive an expected station color or assign QR identity. Achromatic
heads and missing/weak rims retain the ordinary acquisition fallback.

`material_rim_acquisition.py` uses existing Canny pixels near those material
exteriors only to locate a head. It then independently refines the complete
border against the untouched raw edges, verifies the same border family, and
captures a proof bound to that exact image and raw-edge array. The existing
physical fit, selected-border binding, pose-quality and candidate/source filters
remain mandatory. Separate material proposals and unowned raw rectangles/rails
within the search region prevent a nearby colored or gray head from disappearing
from ambiguity checks.

`physical_head_pipeline.py` integrates this before ordinary cold acquisition.
Both stages share the original deadline and twelve-verification allowance;
failed rim-search work is deducted before fallback. Image size, contour and
component limits bound the additional work. Expiry and unresolved competitors
produce unavailable geometry, not a partially checked result.

The chromatic mask supplies neither physical corners nor stand identity.
Recorded LiDAR association, reconciliation, current QR association, and TF
freshness gates remain separate requirements. These tests do not establish
robustness under every lighting condition or overlapping physical-head layout.

## Immediate QR and bounded green geometry

`artifacts/bounded_front_geometry.py` writes `qr_bounded_front_geometry.json`
after a successful QR receipt commit, when the already prepared, associated
current front sample has valid orientation bounds and matching QR/image/scan
identity. This is a hashed **one-frame inspection-only** sidecar referencing
the QR receipt, preserving all pose hypotheses and their engineering uncertainty.
It cannot claim seven-frame consensus, facing readiness, or motion permission.
A missing/stale/different sample is not attached; an optional write failure does
not revoke successful QR admission.

Green's nearly frontal planar fit still fails the strict scalar-angle criteria.
Its explicit interval is retained rather than narrowed or discarded. Existing
bounded facing/backside producers still require seven fresh samples. This does
not add a one-frame green-facing policy or refit a retained backside angle.

## Retained-angle catalog support

`artifacts/projected_retained_facing.py` adds schema-5 recommendations derived
from an unchanged schema-4 retained-angle receipt. The derivation verifies the
original QR chain, its certified arrival projection, the two frame certificates,
shared candidate ancestry and projected snapshots. It transforms the measured
center, full angle interval, observing pose, face normals and endpoints through
odom into the common map frame. Source changes, altered geometry, recursive
sources and changed candidate anchors fail validation.

Arrival catalogs containing this policy use schema 2. The record persists both
the bounded orientation and projection evidence. Loading re-derives and compares
the geometry; legacy schema-1 record serialization stays unchanged. Referenced
source receipts/projections must remain available and unchanged when the catalog
is read.

`stations/retained_catalog_geometry.py` handles record validation, immutable
candidate binding and the original/measured-center clearance envelopes. Offline
promotion rechecks the exact endpoint and terminal route against both envelopes
and the complete candidate pool. The original collision clearance must be
covered by the frozen envelope. Later route construction conservatively covers
the frozen envelope from the measured center, including tracking margin; it may
reject an endpoint whose enlarged exclusion no longer permits the route.
No QR-only observation or one-frame green sidecar is promoted as facing geometry.

## Validation

Production replay retained real source-domain filtering and stored candidate
search inputs. All 12 captured failure frames with complete metadata acquired a
head and passed the full-raw selected-border check, using **2–4 strict checks**:

- Two blue frames: strict measured-head angle accepted.
- Ten green frames: explicit orientation bounds accepted; nine still report
  scalar yaw uncertainty and one reports scalar planar-axis ambiguity.

Local timings were approximately 80–148 ms per geometry call in this replay.
They are offline measurements, not ROS latency or absolute angle-accuracy claims.
Frames lacking stored search metadata were not reconstructed with guessed inputs.

The actual ROS Apptainer/OpenCV **4.5.4** environment on `mii001` also passed the
representative blue and green production-pipeline replay, with two strict checks
each, including the final contour/image-budget and invalid-source competitor guards. This used the authorized temporary directory
`/tmp/a04-border-investigation.EBMLG2`; the robot checkout was not changed.

Focused final run: **96 passed plus 47 subtests**. Coverage includes actual recorded source
filtering, red/blue/green/purple material selection, gray and colored competitors,
broken/weak rims, deadlines, shared verification budgets, current raw proof
binding, immediate QR retention and write failure, seven-frame behavior,
projection/candidate tampering, catalog round-trip and downstream envelope coverage.

A broader targeted run had 582 passes and 652 passing subtests, with 21 failing
stand-axis tests and two failing scheduling subtests. The 21 test node IDs match
the previously reproduced untouched-HEAD failures. The scheduling subtests were
also rerun on untouched HEAD and reproduce its missing synthetic
`head_top_height_m` fixture field. Thus the broader suite is not green; those
failures were not repaired by this change.

Evidence directory:
`results/implementation_checks/blue_green_solution_20260923/` contains
`production_sequence_replay.py`, `production_sequence_replay.json`, and
`production_ros_replay.json`. Tests use the small committed blue/green fixtures;
the complete capture archive remains local evidence.

No end-to-end live exploration or logistics execution was performed. Catalog
checks cover recorded retained-source projection/persistence, clearance helpers,
route-node envelope construction and existing promotion regressions; they do
not constitute a new real-run retained-catalog execution result.
