# Candidate aiming and admission audit

Run: `stand_explore_exact2_camera_all5_20260930T125903Z`  
Workstation access: `ssh mii002`  
Executed revision: `e018878a583131f5dc051c0ea7ce777df91429cf`, clean checkout  
Unresolved candidate: `survey_candidate_0005`, third visited  
Audit date: September 30, 2026

The robot corrected its heading toward a survey position that disagreed with
the visible stand. That correction passed arrival admission, but left the
physical head roughly 199 pixels left of image center. The ordinary camera
and LiDAR association gates consequently rejected every processed observation.
A restrictive recovery condition prevented the system from using a complete
current cluster when the narrower search already contained three of its beams.

The run confirmed `survey_candidate_0003` as `QR_003` and
`survey_candidate_0002` as `QR_002`. Candidate `0005` remained
`inspection_started`, with no final candidate decision. The observer decoded
`QR_004`, but that text never became a candidate-bound identity. The saved
mission progress is two confirmed stands out of five.

## Recorded arrival and observation

Times below are Berlin local time, UTC+2.

| Time | Event | Evidence |
| --- | --- | --- |
| 14:59:03 | Parent run began | Parent bundle manifest |
| 15:08:42.089 | Third candidate approach completed | Child reported completed motion |
| 15:08:57.164 | Initial arrival rejected | Bearing error −6.444°, range 0.54645 m |
| 15:09:23.638 | Arrival correction completed | Approximately clockwise 6.4°; accumulated translation estimate 0.00106 m |
| 15:09:29.879 | Corrected arrival admitted and camera started | Bearing error −0.132°, range 0.54527 m |
| 15:11:00.147 | Camera observation deadline expired | No associated candidate frame or usable axis |

Both range checks passed. The first bearing error exceeded the 3° strict and
6° passive-acquisition limits. The correction reduced the residual against the
reprojected survey point; it did not verify the measured head center or normal.
Its saved view is `purpose: arrival_alignment`, `source_observation: null`,
and `stand_axis_authorized: false`.

The root `candidate_arrival_admission.json` preserves the initial rejection.
The later successful admission is under
`alignment_01/arrival/candidate_arrival_admission.json`. Reading only the root
file would miss the correction and incorrectly suggest that the camera never
started.

The preliminary LiDAR acquisition failed immediately with
`fresh_scan_cohort_unavailable` and `scan is stale or future-dated at receipt`.
It executed no support or normal-alignment moves. This run therefore differs
from the earlier run where preliminary LiDAR orbiting blocked the camera.

## Why the correction left the stand off center

Candidate reprojection retained its canonical odometry point and transformed
it through the newer map-to-odometry relationship. The resulting target moved
17.02 cm from its frozen survey position, predominantly along map y.

| Position evidence | Map x in m | Map y in m |
| --- | ---: | ---: |
| Frozen survey hypothesis | 1.457270 | −0.315612 |
| Corrected arrival hypothesis | 1.447633 | −0.485547 |
| Diagnostic current cluster in saved frame 1 | 1.485885 | −0.344166 |

That current surface cluster is 14.65 cm from the arrival hypothesis and
4.04 cm from the frozen hypothesis. These are surface measurements and
position hypotheses, not external ground truth of the stand center. They
establish a disagreement that must be reconciled; they do not establish that
frozen coordinates should always replace reprojection, or distinguish wheel
odometry error from localization error.

The pre-correction diagnostic scan contains five forward returns at
4.79–11.41°, centered at 8.09° left of the scanner's forward direction.
The first archived post-correction scan contains four contiguous returns at
11.95–16.88°, centered at 14.41°. The approximately 6.3° increase matches the
clockwise correction toward the reprojected point. The pre-correction ROS
echo truncates the later range array; these intact forward samples are used
only as a diagnostic comparison, not as a new identity proof.

For saved frame 31, a same-image geometric replay measures the head center at
`(201.533, 296.824)` px. The original target projection is u=389.702 px;
image center is u=400 px. The original picture and the recovered current
cluster agree that the visible head is far to the left. The calibrated
centering solver gives approximately **15.86° counterclockwise** for that
measured center and range. This is a diagnostic direction and magnitude,
not a scan-safe turn or motion authorization.

The measured camera mounting offset is small relative to this discrepancy.
The camera has approximately −1.047° optical yaw relative to the base and
translation `(0.04554, −0.00497, 0.12575)` m. Using optical calibration in
arrival admission addresses the optical/base difference, but cannot remove
this much larger disagreement about the target position.

## Why identity and centering stayed blocked

The 90-second observer processed **217 synchronized, transform-ready images**.
It recorded 217 LiDAR-association rejections, zero accepted candidate frames,
zero axis samples out of seven required, and a peak consensus of zero.
Centering remained blocked with `fresh_admitted_current_head_required`.

The narrow ±3° candidate cone looked near scanner bearing zero, where the
closest returns were approximately 0.96 m away and outside the candidate's
approximately 0.397–0.617 m surface-range interval. The actual stand returns
were around +14° and 0.58 m. Independent identity search could still decode
the visible QR in a wider crop: **77 decoded frames**, with `QR_004` recorded.
Every usable decode still required association to the selected candidate.

In saved frame 29, the calibrated QR ray is +13.985° with approximately
1.224° finite-range uncertainty. Its map reference is −0.136°. Their
approximately 15.35° combined difference exceeds the existing 12° bound,
giving `camera_map_bearing_interval_exceeds_limit`. The QR-only fallback
therefore remains `current_sensor_frame_not_admitted`. A visible or decoded
QR alone cannot confirm the selected candidate.

The wrong target projection also constrains head search around the wrong
image region. In the original frame 29 metadata, the projected candidate
edge region begins around u=205 while the physical head's left border is
around u=148. Recovery can supply a current search position without changing
the underlying survey geometry, but no such proof was obtained in this run.

## The recovery condition reproduced offline

The stopped position-epoch recovery requires three fresh scans and a unique
compact cluster within authenticated frozen/current position hypotheses.
It preserves candidate exclusion, topology, source freshness, the 30-second
epoch lifetime, and the original candidate snapshot.

Its additional ordinary-envelope condition permits a nonempty ±15° search
only when it contains one or two raw beams that are a strict subset of the
expanded ±35° cluster. This restriction is at
`observer/candidate_position_epoch.py`, inside `epoch_cluster`.

Saved frame 4 contains ordinary indices `[5, 6, 7]` and expanded indices
`[5, 6, 7, 8, 9]`. The third ordinary beam causes rejection even though the
ordinary cluster is still a clipped portion of the same unique full cluster.
Other scans with only two ordinary beams can begin collection, but the
repeated three-beam rejection clears the history before three qualifying
stopped observations accumulate.

The replay imports the exact clean executed revision archived locally and
reproduces every recorded reconciliation outcome in the 133 archived
transform-ready detector frames. The 123 other archived frames lack a
transform-ready detector result. All 256 corresponding image hashes pass.

| Outcome in the archived detector frames | Executed code | Diagnostic strict-subset change |
| --- | ---: | ---: |
| Collecting stopped support | 24 | 33 |
| Ordinary cluster eligibility rejected | 51 | 0 |
| Separate competing clusters rejected | 3 | 3 |
| Expired position epoch rejected | 55 | 55 |
| Valid current-target reconciliation | 0 | 42 |

The diagnostic change removes only the ordinary cluster's fewer-than-three
beam requirement. It retains the strict raw-index subset requirement and
all downstream checks. The first reconciliation then uses archived frames
1, 2 and 3, with a current cluster near `(1.484708, −0.343637)` m. Negative
controls continue to reject a separate one-beam competitor, only two beams,
stale sources, and missing epoch evidence. Permitting equality of the raw
index sets produced no additional successes in this recording; this audit
does not support broadening that separate condition.

A further frame 31 image replay using this diagnostic proof changes the
search projection from u=389.702 to u=191.970 and accepts the measured head
against the current LiDAR cluster. Its stand angle remains
`head_model_planar_axis_ambiguous`. The replay disables wall-clock detector
work budgets and uses the local OpenCV backend. It establishes diagnostic
geometry and association, not live performance, a reliable stand angle,
QR admission, a centering permit, or completed mission behavior.

## Additional failures and the recorded ending

The final observer status is `tf_pending_exact_time`, including a final
8.64 ms future-extrapolation gap. TF delivery reduced throughput and expired
170 retry tuples, but the 217 processed frames establish a separate persistent
association failure. The parent intentionally sent SIGINT at the observer's
deadline; return code 130 here is its cleanup result.

The terminal also contains 545 native QUIRC warnings. They are not evidence
that all decoding failed: the recorded pipeline decoded 77 frames. Increasing
the camera timeout or replacing the native decoder would not repair the
candidate-binding disagreement demonstrated above.

After the timeout, two generic inspection directions exhausted four standoff
proposals each because their target cells were blocked. A third direction
has planning artifacts through `standoff_001`, but no saved motion execution.
There is no candidate rejection, mission failure, final mission summary, or
parent end-time/exit marker. No autonomous mission process remained when the
workstation was inspected. The artifacts establish an incomplete ending;
they do not identify the exact cause or signal that ended the parent.

## Recommended correction

Generalize the clipped-cluster recovery eligibility to a **strict subset of
the same unique complete cluster**, including a three-beam clipped subset.
Preserve ambiguity, compactness, competitor exclusion, candidate and frame
binding, three stopped scans, freshness, and epoch limits. This is the
smallest correction supported by the replay; broadening the normal 12°
identity gate or relaxing stand-angle reliability is not required by it.

Use the validated current target association to obtain camera centering
advice. Preserve scan-boundary checks and the separately certified coarse
recovery/fine-turn budgets. The diagnosed 15.86° correction must not be sent
as an unchecked turn. When head angle remains ambiguous, attempt the existing
fresh candidate-bound QR observation-pose fallback without claiming a facing
pose or perpendicular arrival.

Add a regression containing the recorded alternating two/three-beam ordinary
subsets and complete four/five-beam clusters. It should require current-target
reconciliation to reach camera association and preserve rejection of separate
competitors and an ordinary cluster equal to the expanded cluster. Retain
the phase distinction between map arrival admission, measured camera
centering, QR identity admission, and stand-angle validation.

The workstation used the revision before the newer local camera-first and
eight-scan capture changes. However, the two relevant recovery files,
`candidate_position_epoch.py` and `target_reconciliation.py`, are byte-identical
between the executed revision and local `7221553`. Deploying those existing
local changes alone does not remove the demonstrated recovery restriction.

## Evidence and reproduction

The copied mission and parent/child bundles are under
`results/implementation_checks/run_audit_20260930T125903Z/source/`.
The workstation's final mission progress, inspection progress, observer
status, terminal log, and recovery source hashes were independently checked
against the local copies. The same run was still the latest workstation run
at the final verification.

The audit directory contains:

- `replay_admission.py`, `admission_replay.json`, and `admission_replay.log`;
- `replay_head.py` and `head_replay.json`;
- `strict_subset_any_count_first_proof.json`, an explicitly diagnostic proof;
- `frame_000029_original.jpg`, an unchanged, hash-verified captured image;
- `source_sha256.json` and the relevant `executed_source/` archive.

From the repository root, reproduce using:

```bash
/Users/stephpark/miniconda3/envs/mii-project/bin/python \
  results/implementation_checks/run_audit_20260930T125903Z/replay_admission.py
/Users/stephpark/miniconda3/envs/mii-project/bin/python \
  results/implementation_checks/run_audit_20260930T125903Z/replay_head.py
```

This investigation changed no production behavior, deployed no code, and
started no robot motion. The proposed eligibility correction remains an
offline diagnostic; physical candidate admission and successful aiming
require implementation and verification.

## Implemented correction and regression evidence

The follow-up implements the strict-subset correction in
`real_robot/observer/candidate_position_epoch.py`. A clipped ordinary cluster
can now contain three or more beams. It still must be the only ordinary
cluster and a strict raw-index subset of the only expanded cluster. Equal
clusters, separate one-beam competitors, insufficient complete support,
incompatible position hypotheses, stale sources and moved robots remain
rejected. The change preserves the original snapshot, the 12-degree ordinary
registration bound, and the 30-second position epoch. Reconciliation grants
neither motion nor stand-angle authority.

`verify_implemented_recovery.py` replays the 133 saved detector tuples using
current production code, with the archived epoch guard substituted only for
the baseline comparison. The result matches the earlier counterfactual:
**zero versus 42 valid reconciliations**, with the first current proof at
frame 3. All three competing-cluster cases and all 55 expired epochs still
reject. Results are in `implemented_recovery_replay.json`.

This verification also corrects a derived timestamp in the earlier audit:
`scan_age_sec` is measured from scan receipt, so the replay clock is receipt
time plus that age, rather than scan source time plus age. Across the archived
detector tuples this changes the replay clock by −24.1 to +63.4 milliseconds;
the baseline and corrected reconciliation counts above are unchanged. The
committed regression fixture uses the corrected clock and preserves both
original source and receipt timestamps.

The self-contained regression fixture
`tests/aufgabe04/fixtures/candidate_position_epoch_partial_20260930/` includes
the original arrival snapshot/projection, six stopped camera/scan tuples,
calibration and exact-time transforms, and the unchanged frame-31 image.
One test checks the original two/three-beam variation in frames 1–3. Another
derives the frame-31 head from its pixels and carries it through production
association, crop review, centering-advisory validation, observer frame
admission and status publication. The original snapshot and projection are
checked unchanged.

That image still reports `head_model_planar_axis_ambiguous`. Its independently
validated orientation bounds allow current-head detection and centering, but
do not become an accepted axis: the test retains **zero angle samples and no
precise pose recommendation**. With the recovered target, the calibrated
solver requires **+15.85698 degrees**, while the existing scan-boundary review
permits an initial **+6.85698-degree** advisory. A full correction is rejected
because it would put the head across the malformed scan seam. This test
confirms a bounded advisory, not an executed turn or complete centering. The
existing parent must obtain its normal movement authorization, turn, stop,
and inspect a fresh sensor tuple. No new angle threshold or motion path was
introduced.

The final-TF diagnostic issue is also corrected. The parent status loader,
serialized failure evidence and failure text now preserve validated lifetime
camera counters. New observer snapshots retain the last processed-frame
outcome and bounded counts of estimator/association reasons separately from
the latest transient status. A final `tf_pending_exact_time` can therefore
coexist visibly with `tf_ready_tuples=217` and `processed_images=217` and the
preceding image failures. TF retries do not increment image-outcome counts.
These diagnostics do not change timeout classification, identity admission,
or source-freshness checks.

`implemented_diagnostics_replay.json` verifies the parent failure report
directly from the saved final status. It separately reconstructs processed
outcomes from the event log, explicitly labeling that reconstruction rather
than attributing new fields to the historical status file. The 217 distinct
processed images comprise 187 unavailable-head-border results, 11 detector
deadlines, 10 ambiguous planar angles, eight excessive yaw uncertainties, and
one ambiguous proposal. Repeated TF publications do not inflate these counts.

All **1,150 copied source files** were independently hash-checked against
`mii002` during this follow-up, with no mismatch. The manifest is
`source_integrity_verified.json` in the audit evidence directory. The clean
workstation checkout remains `e018878`; this correction is local. No code was
deployed and no robot motion was started.

The offline camera regression excludes wall-clock detector deadlines and uses
OpenCV 4.13. It does not establish live throughput, physical centering,
perpendicular arrival, or reliable planar angle. A new logged physical run is
needed to measure those outcomes. Fresh QR-only admission remains available
when identity is valid but head angle cannot be certified; it must continue to
report the facing pose as unavailable.

Final focused validation after the replay-clock correction: **172 tests and
252 subtests passed** across 16 modules, with no failures or skips. The suite
covers recorded recovery, source/epoch and competitor rejection, ambiguous
head bounds, centering, observer publication, parent capture and diagnostics.
Its receipt is `integration_tests.xml` in the audit evidence directory.
`git diff --check` passes. The recorded head handoff outputs are retained in
`production_head_handoff/{centering,status,summary}.json`.
