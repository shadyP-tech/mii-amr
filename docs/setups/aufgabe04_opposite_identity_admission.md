# Camera exploration: decoded identity without QR outlines

Physical camera exploration uses QR **payloads only**. Head borders and the
synchronized LiDAR target supply image ownership, position and orientation.
No standalone QR-outline detector or QR-corner recovery runs in either ordinary
exploration or the retained-orientation opposite-side branch. The physical
observer requires its candidate snapshot so neighboring stands can be excluded.

## Current identity admission

Ordinary acquisition first measures and associates the current head. The
opposite-side branch uses a bounded current-border detector to locate a head
without another angle fit. Both paths mask pixels outside that measured head,
check candidate/scan association and neighboring projections, and run the
payload decoder with `identity_only=True`. A deeper projected neighbor may be
excluded only with separated depth bounds and foreground head support. Missing
head support cannot authorize decoding an overlapping background crop.

One nonempty decoded ID is sufficient. QR corners remain absent; multiple
symbols, including duplicate payloads, are rejected. The decoder returns after
its first successful payload batch instead of searching for corners. Exact-time
transforms, calibration, stopped-motion epochs and source freshness still apply
at observation and durable publication. Historical QR-geometry receipts remain
readable; they are no longer produced by physical exploration.

A complete current head plus ID can still yield the strict immediate-front
recommendation. Ordinary discovery uses schema-3 QR receipts. An opposite-side
ID uses schema 2 and the original retained angle, uncertainty and sample count.
The scan-seam branch confirms its endpoint fragments using a current head region
and binds decoding to that same masked region. Neither branch invents a QR pose.

## Unidentified sides and retained angles

A failed payload decode does not prove marker absence or a backside. Seven fresh,
associated bounded-head samples with completed exclusive ID attempts can instead
publish a schema-5 **unidentified-head** orientation. Skipped decoding, failed
backend calls, duplicate images, motion changes and conflicting IDs cannot fill
that window. Its receipt records unknown marker state and geometric confidence,
with no claim of physical side or motion permission. The existing route planner
can inspect the other side of that measured axis after its normal route checks.

Retained evidence is reprojected into each admitted arrival frame. Current head
support may also supply the existing bounded centering advisory. Identity success
finishes immediately, retaining the angle without another consensus window.
Once a current ID establishes the QR-facing side, the whole-interval view check
uses the user's **30-degree** allowance, including position and angle uncertainty.
Unidentified opposite-route planning retains its existing limits.

## Regression evidence (October 1, 2026)

The fixture `tests/aufgabe04/fixtures/opposite_head_identity_20261001/` contains
original frames 5, 16 and 31 from run
`stand_explore_exact2_camera_all5_20261001T141116Z`, original sensor/TF inputs,
and the authentic seven-sample orientation-only source. Production scan
reconciliation and head association now produce an exclusive foreground crop
and admit cornerless `Start`, preserving the angle through receipt reload and
retained-facing construction at 0.4 m. The tests also reject facing geometry
above 30 degrees, stale sources, missing support, neighbor ambiguity and
rehashed evidence substitutions.

An additional local OpenCV **4.14.0 / WeChat** replay performed real border
acquisition, real ID decoding and durable admission with the recorded entry age
plus advancing wall time. All six runs (each image twice) admitted `Start`
without corners and passed the unchanged 0.5-second freshness checks. See
`results/implementation_checks/qr_identity_only_20261001/local_wechat_replay.json`
and its `offline_replay.py` reproducer. This is recorded-data validation, not a
new live robot run or a workstation OpenCV-4.5.4 timing claim.

The focused regression suite passed 415 tests (one skipped), with 263 subtests.
The final startup, ordinary-admission and optional-odometry checks passed 49
tests with 16 subtests. Broader repository checks still encounter pre-existing
map/timeout fixtures, a stale source-string assertion, and sandbox socket
restrictions; this is not a claim that the full repository suite passes.

The user requested that this correction remain local. No correction or test
bundle was transferred, and the workstation checkout was not changed. A new
workstation/live validation has not been performed.
