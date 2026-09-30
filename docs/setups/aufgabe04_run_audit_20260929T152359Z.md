# Opposite-side branch and camera offset: 20260929T152359Z

The opposite-side branch lacked an admitted backside axis. The underlying
failure was earlier: the normal LiDAR envelope clipped the visible stand's
cluster to one or two beams, and those partial returns prevented the broader
position-epoch recovery from running. Without an associated current head,
both camera centering and backside certification stayed blocked.

The implemented correction distinguishes a clipped portion of the same cluster
from a separate competing object. On the recorded scans, production code now
produces a valid three-scan reconciliation by frame 5. The original audit and
its in-memory proposal are preserved alongside the implementation replay.

## What the run did

Run `stand_explore_exact2_camera_all5_20260929T152359Z` used clean revision
`3e6dfe14c829ad89aade52b3dbb60d1a57fe764d`. The preceding local TF startup patch
was not deployed and is unrelated to this run's camera failure.

The first visited candidate, `survey_candidate_0003`, produced `QR_003`.
The second, `survey_candidate_0001`, completed its initial approach at
**15:29:51.795 UTC**. Its passive observation then ran from **15:29:57.688**
to **15:31:27.907 UTC**:

- 412 processed, synchronized, TF-ready camera frames;
- 412 LiDAR-association rejections;
- zero accepted candidate frames;
- axis consensus **0/7**, including a peak of zero;
- centering blocked with `fresh_admitted_current_head_required`;
- geometry unavailable with `model_current_head_border_unavailable`.

The parent deliberately sent SIGINT to stop the observer after its 90-second
deadline. That explains the observer's return code 130; it is not evidence of
an unrelated observer crash.

With no certified backside orientation, the controller selected a generic
`diverse_inspection` view. Its 0.710 m route passed dry admission at
**15:31:42.353 UTC**. The child began live preflight at **15:31:49.107 UTC**,
and the parent recorded `KeyboardInterrupt` at **15:31:52.841 UTC**. There is
no completed execute-preflight artifact, motion-start event or controller
trace for that child. The source of that later interruption is not established.

The shared `opposite_face_planning_localization.json` filename does not prove
an opposite-side branch was selected: the generic inspection planner also
uses that helper. Its saved view explicitly says `purpose: diverse_inspection`,
`source_observation: null`, and `stand_axis_authorized: false`.

This run therefore does not show another opposite-route admission failure.
The previous mixed uncertainty/orientation failure is a different case.

## Why the first inspection image was offset

Initial approach planning aims the robot's base at the projected survey
candidate, using `atan2(candidate - route_endpoint)`. Arrival admission checks
that map-based geometry, not the measured camera head center. Passive
centering acquisition can accept up to 6 degrees of map-bearing residual;
its record explicitly does not certify camera centering.

The two arrivals show different outcomes:

| Candidate | Map-bearing residual | Predicted head u | Observed head u | Image center u | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `0003`, first visited | +1.761° | 367.54 px | 312.98 px | 400 px | Accepted geometry and QR; centering deferred |
| `0001`, second visited | −3.580° | 431.89 px | approximately 201.5 px | 400 px | Head association blocked |

The first candidate's measured head is 87 pixels left of center. Its centering
record explicitly says `preserve_productive_geometry_view`: usable geometry
completed before optional centering. This is not evidence of a failed turn.

For candidate `0001`, the approximate 201.5-pixel head center is a manual
measurement of the first saved image, not a production admission. The same
frame's raw LiDAR cluster has five contiguous returns at **11.439–18.106°**,
with ranges **0.535–0.555 m**. Their Cartesian centroid is at **14.802°**,
whereas the projected survey target is at **−3.419°** in that scan frame.
The cluster is approximately **17.5 cm** from the predicted point, predominantly
to its left. This target-position mismatch dominates the image offset.

The fused frozen survey point was `(-1.121370, -0.451117)` m; arrival reprojection
placed it at `(-1.144855, -0.334741)` m. The recorded position-epoch displacement
is 0.118722 m. The proposed first valid reconciliation, using frames 3–5,
locates the current cluster at `(-1.109150, -0.495230)` m. These are separate
position hypotheses and measurements; recovery must prove their association
rather than silently replace the survey point.

Follow-up tracing on September 30 distinguishes the fused candidate's provenance
from its earlier first-view advisory. The latter contains
`(-1.136116, -0.434817)` m and is not the frozen point used by arrival reprojection.
The corrected framing diagnostic also compares the scan centroid with the exact
transformed candidate point, rather than the midpoint of its surface-range
interval. Under the recorded arrival transform, the fused frozen hypothesis is
5.6 cm from the current cluster, while the arrival-reprojected hypothesis is
17.5 cm away. Reprojection moved the candidate 11.9 cm. This identifies the
survey-to-arrival position hypothesis as an important source of the mismatch;
it does not establish that keeping frozen map coordinates is always correct or
separate odometry drift from localization error without external ground truth.

The source candidate itself has 106 observations across two survey viewpoints.
Recomputing those observations in their recorded canonical odometry frames gives
view means separated by 17.25 mm (56 observations from viewpoint 1 and 50 from
viewpoint 2). Within-view RMS position scatter is 8.98 mm and 14.70 mm,
respectively. The survey cluster was repeatable; the recorded data do not show
a 17 cm disagreement between its original views. Its location is nevertheless
an average of visible surface returns, not a fitted physical stand center.
Arrival reprojection retains the configured 2 cm uncertainty. A coordinate
frame displacement is not itself measurement uncertainty, but this fixed value
also does not model accumulated odometry error in an older candidate landmark.

The calibrated camera has a −1.047° optical yaw offset and a translation of
approximately `(0.04554, -0.00497, 0.12575)` m relative to the base. At a
0.55 m target range, perfectly aiming the base would place the projected
target near u=387.8, about 12 pixels left. This is a smaller geometric effect
and cannot explain the observed 199-pixel error. A fixed global yaw bias is
therefore not the appropriate correction.

Applying the existing calibrated centering solver diagnostically to the
visible head and recorded range gives approximately **15.8° counterclockwise**.
This establishes the correction direction and scale, not motion authority.
Current head admission, scan-boundary checks and a separately certified turn
remain necessary.

## The blocking condition

At the recorded revision, `candidate_position_epoch.py:epoch_cluster` requires the ordinary
15-degree candidate search to contain no eligible cluster before using the
bounded 35-degree recovery envelope.

In frame 1, the predicted bearing is −3.419°, so the ordinary search ends at
**+11.581°**. It catches only beam 6 at **+11.439°**. The adjacent four stand
returns lie just outside that boundary. This single beam is insufficient for
the normal three-beam reconciliation and measured-head search, yet it makes
the recovery envelope ineligible with:

`ordinary candidate envelope must be empty for epoch recovery`

This creates a gap: a partially visible target is less recoverable than an
entirely displaced target. The expanded envelope would find the complete
five-beam cluster, but the early empty-envelope check prevents that evaluation.
The same pattern recurs throughout the fresh position-epoch window. After
the existing 30-second epoch lifetime expires, recovery correctly stays closed.

## Implemented correction

The ordinary-envelope decision now follows the expanded envelope's raw cluster
uniqueness checks. The existing empty-envelope case is preserved. A nonempty
ordinary envelope is permitted only when all of the following hold:

1. It contains exactly one associated cluster of one or two beams.
2. Its exact raw beam indices are a strict subset of the expanded cluster.
3. The expanded envelope contains exactly one contiguous cluster of at least
   three beams, counting even one-beam competitors.
4. All existing compactness, frozen/current position hypotheses, competitor
   exclusions, three fresh stationary scans, hash/frame binding and epoch
   lifetime checks pass.

This must not become a general “ignore sparse clusters” rule. Separate
one-beam objects remain competitors. An ordinary cluster already providing
three beams is not overridden, and no returned cluster is trimmed to fit a
preferred hypothesis.

After that reconciliation, the existing `recovered_search` projection,
current image head association, and calibrated centering path apply. The
survey map geometry and the initial route's admission contract remain intact. The
reconciliation itself grants neither a turn nor backside orientation.
Backside consensus must still reach seven accepted samples before the
opposite-side branch becomes eligible.

For the successful `0003` view, mandatory centering would be a separate
behavior change: its productive-view deferral is intentional and did not
prevent completion. The defect demonstrated here is the blocked recovery of
the displaced `0001` target.

## Offline validation

`results/implementation_checks/run_audit_20260929T152359Z/replay_candidate_recovery.py`
applies only the proposed gate in memory and replays the original 256 archived
sensor tuples. Timestamps remain historical; no robot command is published.

| Outcome | Current code | Proposed gate |
| --- | ---: | ---: |
| Valid current-target reconciliation | 0 | 67 |
| Ordinary-envelope early veto | 118 | 0 |
| Collecting three stopped scans | 9 | 27 |
| Separate competing clusters | hidden by early veto | 33 rejected |
| Stale/mismatched sources | 7 rejected | 7 rejected |
| Expired position epoch | 122 rejected | 122 rejected |

The first valid proof uses frames **3, 4 and 5**, with five raw beams in each
selected cluster. It retains `candidate_geometry_updated: false` and
`motion_authorized: false`. Added negative controls reject a separate one-beam
competitor, only two supporting beams, stale sources and missing epoch context.

Under the proposed in-memory policy, **74 existing tests and 67 subtests pass**
across position-epoch recovery, reconciliation, opposite-view reconciliation,
current-head association and centering. These checks support the proposal;
they do not establish a successful physical turn or completed opposite route.

The matching, hash-verified **frame 5 image** also passes a production geometry
replay (`replay_recovered_head.py`). The recovered projection is **u=203.823**,
instead of **u=432.092**. The detector measures the complete head at
**u=204.899**, produces `axis_estimated_current_measured_head_backside`, and
passes current-head geometry, unique LiDAR association and complete-head
marker-absence review. No corners, axis or QR result are supplied by hand.

The calibrated centering solver then returns a validated arrival-recovery
advisory: required turn **+15.52694°**, initial requested turn **+8.52694°**.
The existing scan-boundary check deliberately retains seven degrees of
residual offset; a fresh stopped observation must determine the next action.
The recovered projection is thus useful to the actual detector and downstream
centering path, not just to a scan-only eligibility test.

This image replay excludes the detector's wall-clock deadline while preserving
historical sensor-age checks. It proves one frame's geometric usability and
advisory validity, not seven-frame consensus, productive-view scheduling,
runtime performance, issuance of a motion permit or a completed robot turn.

Source artifacts include `opposite_branch_trace.json`,
`arrival_framing_analysis.json`, `candidate_recovery_replay.json`,
`proposed_frame_000005_reconciliation.json`, `recovered_head_replay.json` and
`proposed_recovery_tests.xml`.
Sixteen copied camera images match their recorded capture SHA-256 hashes.
The subsequent workstation hash-verification connection timed out; this is
recorded in `source_integrity.json`. The original metadata archive is retained.

The production change is confined to the eligibility guard in
`scripts/aufgabe04/real_robot/observer/candidate_position_epoch.py`.
`verify_clipped_cluster_correction.py` compares the recorded revision with
current production code, without substituting the proposal. It confirms zero
versus 67 valid reconciliations and the first proof at frame 5; results are in
`implemented_recovery_replay.json`.

Implementation validation: **169 tests and 83 subtests pass**. This includes
16 new scan/proof regressions and one same-frame camera regression, backed by
the self-contained fixture in `tests/aufgabe04/fixtures/candidate_position_epoch_clipped/`.
Three positive cases fail against the original committed policy loaded only
in memory, confirming that they exercise the corrected eligibility gate.
The new negative cases cover separate one-beam competitors, insufficient
support, complete ordinary clusters, multiple ordinary fragments, mismatched
raw indices and source/epoch/stationarity violations. The camera regression
derives the head from the original image and checks backside association,
the validated bounded left-turn advisory, and the retained scan-boundary veto.

Results are preserved in `implemented_recovery_integration_tests.xml`,
`clipped_scan_regression_tests.xml`, `clipped_head_regression_tests.xml` and
the baseline comparison `clipped_scan_baseline.json`. `git diff --check`
passes. These tests exclude live motion and seven-frame backside consensus.

No code was deployed and no robot motion was started. Earlier uncommitted
TF-startup edits remain separate.

## Arrival acquisition before inspection — September 30 correction

The user requested the complete survey-to-camera handoff: acquire the current
target, perform a bounded calibrated centering correction, then inspect from a
fresh stopped view. Existing reconciliation already acquires the displaced
candidate from current scans and pixels. The remaining sequencing issue was
that the observer could accumulate geometry, delay centering for five seconds,
or publish a full recommendation before its actionable correction.

Centering review now runs inside the evidence accumulator's post-admission
review, after the common association, source freshness, synchronization,
duplicate-frame and QR-conflict checks, and before an axis sample is recorded.
When safe current-target advice is ready, it withholds the current angle,
discards older angle buckets and clears staged geometry/QR completion results.
The common QR identity/conflict record remains intact. Ready centering takes
priority over geometry completion and QR/backside grace windows. Expired
publication consumes the current opportunity without falling through to a
precision result from the same frame.

The parent capture loop also gives enabled centering advice priority over a
simultaneously supplied recommendation or QR-only completion. Its existing
sealed turn, cumulative travel budget, stopped arrival refresh and exclusive
post-turn sensor timestamp floor remain in force. Inspection resumes through
the fresh capture. The obsolete productive-view delay was removed.

This does not change candidate map geometry, radius, uncertainty, association
thresholds, camera calibration, turn limits or motion authorization. An already
centered target needs no turn. Existing scan-boundary vetoes remain: exact
optical centering may be unsafe for a particular scan seam, and a usable view
can still complete without claiming that it is centered. Retained opposite-side
identity completion and disabled motion budgets retain their previous behavior.

The recorded frame-5 regression now continues through the actual node's frame
admission and status publication. Its recovered current head yields the bounded
scan-safe advisory with zero admitted axis samples and no precise recommendation.
The candidate snapshot and projection fixture remain byte-for-byte unchanged.
This extends the offline handoff evidence; it does not measure live throughput,
execute the turn or establish a completed opposite-side visit.

Consolidated validation: **273 tests and 177 subtests passed** across 21 modules.
One recorded opposite-side decode test was skipped because this local OpenCV
build lacks the deployed WeChat backend. The suite covers current-target
recovery, observer ordering, geometry/QR completion, post-turn freshness,
expired or duplicate evidence, ambiguity, scan-boundary vetoes, cumulative
budgets and simulated controller stops. Results are in
`results/implementation_checks/run_audit_20260929T152359Z/arrival_handoff_tests.xml`.
`git diff --check` passes. No deployment or robot motion was performed.

## LiDAR head geometry and calibrated arrival — September 30 implementation

The camera phase now has a bounded LiDAR acquisition stage before its first
camera observation. It preserves the immutable survey registry and candidate
snapshot. Local scans have separate candidate-bound, content-hashed evidence;
they do not impersonate survey viewpoints or grant motion permission.

The surface fitter retains original beam adjacency and competing-candidate
exclusion, but derives adjacency distances from range and angular sampling.
It fits the measured 7.8 cm by 0.6 cm head section, retaining a feasible center
interval when endpoints are unseen. A hint requires four spatial returns per
usable scan, at least three scans and 75% support per stopped viewpoint, and
two viewpoints separated by at least 20 degrees modulo pi. Repeated samples
do not divide the uncertainty by the square root of their count. The center
and normal carry explicit engineering uncertainty bounds.

Original survey support is still insufficient. Replaying the same 160 source
scans produces zero usable multiview hints for the six candidates. The saved
replay and source hashes are in
`results/implementation_checks/lidar_head_geometry_20260930/`.
The closer successful scan fits have approximately 1.3–2.3 cm center bounds
and 7.5–14.3 degree angular bounds; these are conditional engineering bounds,
not measurements of physical accuracy. A 3 mm point-noise allowance is stated
in the evidence and has not been empirically calibrated on this robot.

When needed, local acquisition proposes at most three closer viewing moves,
trying 0.55, 0.60 and 0.65 m standoffs only through the existing route and
clearance checks. Unknown orientation uses the initial bearing and +/-60
degrees; a supported single-view tangent guides separated views nearer its
normal. A sampling regression showed that a 35-degree change can still leave
an edge-on head with only three returns. Each capture requires three fresh,
separated, stopped scans with exact scan-time base and scanner transforms.
Timeout or ordinary freshness expiry leaves alignment unverified; corrupt
identity, hash or transform evidence remains a hard failure.

Once independent support exists, both opposite normal directions are previewed
with fitted center, calibrated camera translation and optical yaw. The route
retains the original keepouts and also checks clearance around the fitted
center and its uncertainty. Map quantization, localization uncertainty,
tracking allowance and terminal heading error enter a 20-degree planning
budget. Startup/runtime reseals reproject the geometry and reload uncertainty
from the exact fresh localization artifact rather than reuse old bounds.

Arrival is measured again from a new stopped cohort. It uses actual camera
position and yaw, checks current-fit agreement with independent views, source
freshness, range, and a 20-degree normal-alignment budget. Centering within
3 degrees is reported separately. At most two alignment moves are attempted
(initial placement and one correction); missing support or unavailable routes
remain explicitly unverified. A verified normal arrival no longer passes
through the old base-to-survey-centroid yaw correction that could undo it.
The existing camera association, visual centering, QR and front/back identity
checks still run; an unsigned LiDAR normal cannot certify backside identity.

Verification also requires the configured measured model to match the fit and
the exact-time scanner height/tilt relative to `base_footprint` to put its beam
plane inside the measured head height at the observed range. Missing mounting
evidence, incompatible dimensions or excessive tilt disables this acquisition
branch. This validates recorded transforms, not physical calibration accuracy.

Regression evidence includes real geometry fits and calibrated arrival checks,
both motion budgets, blocked-side alternatives, stale/ambiguous evidence,
mounting/model rejection, artifact sealing and recovery, and the regression
that prevented a verified arrival from being redirected toward the old
centroid. The consolidated suite passed **492 tests and 265 subtests**, with
one recorded decode test skipped because local OpenCV lacks the robot's WeChat
QR backend. The result is saved in `integrated_tests.xml` in the replay
directory above. Physical arrival angles and success rates still need
measurement on the robot; no deployment or motion was performed here.
