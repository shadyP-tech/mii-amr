# Run audit: 20260929T135018Z

The last visited candidate reached its first inspection point, but camera/LiDAR
association rejected every processed observation. Its expected scan range
excluded the visible stand. The recovery code widened the bearing search while
retaining that incorrect range interval, so it could not recover the target.

The user's off-centering observation is also confirmed. All five initial
approaches reached their commanded heading within 3 degrees, but those headings
aim at estimated candidate geometry. Live camera centering is optional and was
either deferred, vetoed, bypassed by observation completion, or blocked by failed
association.

## Run state and failure sequence

- Run: `stand_explore_exact2_camera_all5_20260929T135018Z`.
- Recorded clean revision: `552749875e2ce510232fdae2f77bb900d5add248`.
- Four of five QR identities recorded: `Start`, `QR_002`, `QR_003`, `QR_004`.
  Only `Start` and `QR_003` were facing-ready.
- The last visited candidate was `survey_candidate_0005`, visit 004. Its camera
  repeatedly decoded the missing text `QR_001`, but never admitted that identity.
  `survey_candidate_0004` remained unvisited; the survey pool contained six UIDs.
- Candidate 0005's 0.417204 m route completed in 19.862 seconds at
  **14:03:15.338856 UTC / 16:03:15.338856 CEST**. Arrival passed the strict
  estimated-target gate with a 0.909-degree base bearing error.
- During the following 90-second observer window, **372** frames were processed
  with TF available, **329** frames decoded `QR_001`, and **zero** frames passed
  target association. All 372 observations recorded LiDAR rejection.
- The observer's exit 130 was caused by its parent's deadline-triggered SIGINT.
  `observer_process.json` explicitly records `completion_kind: deadline`.
- A subsequent diverse inspection attempt passed its route dry run, then logged
  a separate `KeyboardInterrupt` at **14:06:27.213171 UTC**. No completed live
  result was persisted for that retry, and no `mission_failure.json` exists.

Thus the observed admission failure is established, but the mission was
interrupted during its next attempt. The evidence does not show that all
recovery views were exhausted or establish whether the interrupted dispatch
moved the robot.

## Why candidate 0005 could not be admitted

The arrival projection placed the candidate at `(1.248697, 0.598118)` in map
coordinates; the frozen survey hypothesis was `(1.138268, 0.798422)`. These
hypotheses differ by **0.228727 m**. The logs establish a localization-frame
reprojection difference, not whether the physical stand moved or which
localization estimate was correct.

The current target geometry produced a LiDAR surface-range interval of
approximately **0.3525–0.5725 m**. The observed compact stand cluster was near
**0.665 m**, outside that interval. Replaying the production association helper
on all **248 saved processed frames** returns zero eligible clusters, including
with the recovery bearing widened to 35 degrees.

In `scripts/aufgabe04/real_robot/observer/candidate_position_epoch.py:70`, the
ordinary and recovery searches share `accepted_range_m`. Line 76 widens only
the bearing. Later checks compare the cluster with both position hypotheses,
but the radial filter prevents the relevant cluster from reaching those checks.

This fails even while the position epoch is fresh: the first 125 saved processed
frames report `epoch recovery requires one unique three-beam cluster`. The next
123 report expiration of the 30-second epoch. Expiration compounds the problem;
extending the timer would not fix the initial range exclusion.

The QR diagnostic `finite_qr_ray_target_not_unique` is not evidence of competing
QR identities here. The range gate supplied **zero** eligible scan clusters.
Repeated QR text alone cannot establish that a particular candidate owns it.

There is also a consistent camera discrepancy. In saved frame 256 the measured
head center is **u=245.93 px**, while calibrated candidate projection predicts
**u=380.50 px**. The 800-pixel image midpoint is 400; the recorded optical
principal point is approximately 405.87. The head was about 154 pixels left of
the image midpoint. Its camera/map bearing disagreement was approximately
12.2–12.5 degrees in the saved detections. These projected pixels already
include camera extrinsics, so the roughly 1.05-degree optical/base yaw offset
does not explain the approximately 135-pixel model mismatch.

Intermittent TF waiting, QUIRC warnings, and ambiguous head-axis fits appear in
the logs. They do not explain the zero accepted observations: 372 frames had
usable TF and QR text decoded repeatedly. The run already accepted two other
candidates through QR-only observation, without facing-ready geometry.

## Offline check of a bounded recovery hypothesis

The diagnostic replay derives a second radial interval from the frozen position
in each recorded scan frame, using the existing radius and tolerance formula.
The resulting interval is approximately **0.5259–0.7459 m**. It retains the
35-degree angular cap and the recorded exact scan transform.

This produces one compact cluster satisfying the checked downstream geometry
conditions in **244 of 248** frames. Four frames still fail uniqueness and are
retained as failures. For frame 2:

| Quantity | Recorded/replayed value |
| --- | ---: |
| Selected scan beams | 4 |
| Median range | 0.6645 m |
| Maximum cluster width | 0.0577 m |
| Surface centroid distance to current hypothesis | 0.1738 m |
| Surface centroid distance to frozen hypothesis | 0.0553 m |
| Nearest competing candidate hypothesis | 0.5982 m |
| Existing competitor exclusion requirement | 0.5100 m |

The first three distinct scans span 0.567 seconds and their surface centroids
differ by at most 0.0185 m. Those scans were captured within the fresh epoch.
This supports bounded radial recovery as a solution to the observed search
failure. It is **not** an authoritative reconciliation replay, independent QR
identity proof, facing-ready result, or prediction of full mission success.
The 244-frame count includes later observations whose epoch had expired.

## Why the first inspection views remain off-center

The following measurements are from matched measured/projected metadata within
each first physical inspection view. They are not all the first camera frame;
the exact capture index is included. Raw first-arrival images independently
confirm the visible left offsets.

| Visit / candidate | Capture | Measured head u | Projected u | Commanded minus actual yaw | Centering outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| 000 / 0003 (`QR_003`) | 2 | 330.11 px | 353.67 px | +2.423° | Held to preserve productive geometry |
| 001 / 0001 (`Start`) | 12 | 333.25 px | 398.65 px | −0.211° | Held to preserve productive geometry |
| 002 / 0002 (`QR_002`) | 3 | 349.15 px | 409.90 px | +1.569° | Proposed destination crosses scan boundary |
| 003 / 0006 (`QR_004`) | 93 | 199.78 px | 432.17 px | −0.197° | QR observation completed before centering |
| 004 / 0005 (decoded `QR_001`) | 256 | 245.93 px | 380.50 px | +1.624° | No accepted target association |

Image midpoint is u=400 px. The yaw comparison converts the planned map yaw
using the executed certificate's map/odom transform, then compares it with the
first available exact-time observer odom/base transform. It does not compare
headings across changing map frames. All five are within the 3-degree terminal
tolerance. The blue candidate's separate arrival check was acquisition-only
because its updated map-target bearing missed the strict arrival gate.

The planner already recomputes terminal yaw from the quantized route endpoint
to the estimated candidate (`candidate_preapproach_compute.py:308`). The
evidence does not support a stale yaw after endpoint snapping or a controller
failure to achieve the commanded heading.

Arrival admission explicitly records `camera_centered: false` and requires live
association. Centering depends on that accepted association. Productive-view
holds and the scan-boundary veto can suppress an advisory. Additionally,
`real_robot/candidate/centering_execution.py:86` gives a completed recommendation
or QR observation precedence over optional centering.

For blue candidate 0006, frame 93 records a calibrated required correction of
15.95 degrees and a limited proposed step of 6.95 degrees. No centering turn
was dispatched; QR observation completion took precedence. For green candidate
0005, association never passed, so no corrective turn could be proposed.

## Recommended correction and validation

1. Make epoch recovery search the bounded radial hypotheses as well as bearing.
   Derive intervals from recorded frozen/current positions through the current
   exact scan transform; do not substitute a global wider range threshold.
   Require uniqueness across the recovery search, compact support from at least
   three beams, the existing 0.35 m displacement bound, and exclusion of other
   candidates at both hypotheses. Preserve three fresh stopped scan witnesses,
   epoch/frame bindings, and independent camera/QR association.
2. Once a target is associated, use the calibrated current camera observation
   for centering and verify the result with a new stopped frame. Do not equate
   planned base yaw or decoded text with camera centering. Preserve the existing
   scan-boundary veto and report a blocked centering outcome explicitly.
3. If centered first inspection is a required behavior, make it an explicit
   completion condition. The present optional policy allows productive geometry
   or QR-only completion to bypass it, as candidate 0006 demonstrates.
4. Distinguish zero in-range clusters from competing clusters in diagnostics.
   An expired recovery epoch must be replaced through fresh stopped admission,
   not by extending its timestamp. Increasing the 90-second observer timeout
   cannot correct this range-model mismatch.

The range correction requires consistent proof handling, not just a change to
`epoch_cluster()`. `current_head_association.py:160`,
`qr_target_binding.py:114`, and `qr_candidate_search.py:72` currently require
the recovered interval to equal the original interval. A correction must retain
the original source binding and explicitly validate the additional recovery
interval in each consumer. Keep the original candidate bearing for the broad
search and the measured cluster bearing as a separate reference for narrow
camera-ray association. The observed surface centroid must not directly replace
the stand center, collision geometry, or facing pose.

The finite-distance camera geometry is compatible with that approach: in frame
2, the calibrated head ray at the recovered 0.6645 m range is 0.19368 rad,
against a cluster bearing of 0.20446 rad. Error plus range uncertainty remains
within the existing 3-degree narrow gate, including when considering the
combined current/frozen radial interval. Widening the camera bearing gate is
not supported as the necessary correction. This still does not establish a
completed QR identity receipt or guarantee a centered first arrival.

Regression validation should reproduce this radial displacement and also cover
multiple clusters, nearby competing candidates, stale epochs, changed scan
bindings, and insufficient stopped support. Centering checks should cover
simultaneous QR completion/advice and scan-boundary vetoes. A recorded replay
can validate the geometry correction; a later robot run is needed to establish
end-to-end admission and centering behavior.

## Evidence and scope

Local evidence root:
`results/implementation_checks/run_audit_20260929T135018Z/`.

- `source_metadata.tar.gz` and extracted mission/child bundles preserve copied
  metadata. Ten principal source files were checked against workstation SHA-256
  hashes in `verified_source_hashes.json`.
- `arrival_images/` contains binary copies of the first compressed image for
  each visited candidate, verified against its capture metadata hash.
- `analyze_last_candidate_association.py` reproduces the scan diagnostic and
  writes `last_candidate_association_audit.json`.
- `centering_audit.json` records matched capture indices, source paths,
  execution-frame yaw comparisons, and centering gates.

The saved capture limit is 256 frames, of which 248 have processed detector
metadata. Full-window observer counts cover all 372 processed frames. Image
detectors were not rerun; their saved results and raw scans were inspected.

The previous stationary-selection correction did not change the camera observer,
perception code, or preapproach yaw calculation between `b9bf375` and `5527498`.
This audit identifies an existing recovery limitation exposed by the later
candidate, not a demonstrated regression introduced by that correction.

## Implemented correction

The subsequent implementation adds a recovery range derived from the frozen
position through the exact scan transform, preserving the original surface
offsets. It checks uniqueness across the enclosing current/frozen range first,
including single-beam competitors and the gap between hypotheses. Only then
does it select the original or frozen interval that contains every selected
cluster beam. This preserves the tighter depth uncertainty needed by the
calibrated camera ray without trimming away inconvenient evidence.

A shared proof-consumer helper replays the reconciliation and binds the caller
to its original candidate range and bearing. Head association, QR search, QR
binding and opposite-view support use the authenticated recovered interval for
their finite-distance camera geometry. Epoch freshness, three stopped scans,
raw scan/receipt bindings, compactness, displacement and candidate exclusions
remain required. No landmark or planning-center geometry is replaced.

A ready, framing-safe centering advisory now precedes QR-only completion in the
observer, process monitor and parent capture loop. Full geometry completion and
the bounded productive-view opportunity retain their priority. The parent keeps
the existing motion budgets and requires a new stopped capture after a turn.
Fresh measured framing is evaluated even after the turn budget disables further
advice. Status distinguishes centered, blocked and deferred observations; a
scan-boundary veto still permits valid QR discovery without claiming centering.

`replay_corrected_admission.py` exercises the production reconciliation on the
original saved tuples. The first proof appears at frame 3. On the matching,
hash-checked image, the production decoder reads `QR_001` with its own corners.
The original association still rejects it; the corrected proof accepts it and
the QR receipt passes serialization and validator replay. Across 248 saved
processed frames, 104 reconcile, 12 collect support, four retain cluster
ambiguity, eight fail source freshness/skew, and 120 have expired epochs. The
last categories remain rejected; the replay does not renew historical evidence.

The replay assembles recorded source-gate inputs to check artifact integration;
it is not a live observer publication. It does not claim successful physical
centering, facing-ready completion, or a completed mission. No deployment or
robot motion was performed during this correction.

Validation runs (overlapping suites, not additive):

- Range/head/QR/centering integration: **119 tests and 81 subtests passed**.
- Final radial regression suite, including two additional compactness/support
  rejection cases: **15 tests passed**.
- Centering/observer/orchestration: **168 tests and 125 subtests passed**,
  with one deployed-WeChat-dependent test skipped.
- Broader observer, QR transport, opposite-view and discovery integration:
  **160 tests and 91 subtests passed**, with the same WeChat-dependent test
  skipped. JUnit evidence is `correction_integration_tests.xml` in the audit
  evidence directory.
- `git diff --check` passed. The new radial fixture is explicitly included in
  the repository allowlist; recorded source evidence remains under results.
