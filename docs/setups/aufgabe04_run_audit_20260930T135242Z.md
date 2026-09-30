# Last-candidate alignment audit: 20260930T135242Z

Audited September 30, 2026 through SSH alias `mii002` (hostname `mii001`).
This is the stopped run, not the newer `20260930T141147Z` run. Workstation
access was read-only; no production code, process, or robot command changed.

## Finding

The final route completed its terminal turn toward the reprojected survey
candidate. That target was about **22 cm from the contemporaneous stand
cluster**, so the robot faced the wrong point. The visible `Start` head was at
the right image edge, with its outer border partly clipped. Candidate/QR
association succeeded, but ordinary centering required an admitted complete
head. With that missing, QR-only completion ended inspection without a turn
or a reliable stand angle.

Thus the recorded outcome was five discovered identities, but only two
validated facing poses. The mission summary says `goal_completed=true`,
`camera_geometry_complete=false`, `facing_ready_stand_count=2`, and
`qr_only_stand_count=3`. It had entered `returning_to_start` with return pending
and `start_pose_reached=false`. The return directory has no saved execution
artifacts. The user's stop explains the interruption; the saved evidence does
not establish a completed return.

## Run provenance and the fourth inspection label

The parent bundle started on clean `7221553`. The last candidate's initial
approach and both executed inspection child bundles record clean `06b8bda`.
Its observers also publish the new processing diagnostics. This is a mixed
revision session: the latest strict-subset association correction was already
available to the final candidate's child processes. Its failure must not be
explained as simply missing that correction.

Last candidate: `004_survey_candidate_0001`, eventually identified as `Start`.
`inspection_004` is a route-attempt serial, not the fourth observed viewpoint.
Only three physical camera views were recorded, indexed 0, 1 and 2.

| Child | Requested standoff | Outcome |
| --- | ---: | --- |
| `inspection_001` | 0.50 m | Executed; camera view 1 |
| `inspection_002` | 0.50 m | No motion: route uncertainty margin −0.080278 m |
| `inspection_003` | 0.45 m | No motion: route uncertainty margin −0.000892 m |
| `inspection_004` | 0.40 m | Executed; camera view 2 |

The final child uses `inspection_view_02_00/standoff_002/`. Its saved view
is `purpose: diverse_inspection`, `source_observation: null`, and
`stand_axis_authorized: false`. The requested radial viewpoint is not a
measured head normal or evidence of perpendicular alignment.

## What the robot actually executed

Times below are Berlin local time (UTC+2).

- 16:10:36.388: final child motion began.
- 16:11:10.289: child reported completed motion after 33.72 seconds and an
  estimated 0.627 m travel, on a 0.636 m route.
- Its trace contains 115 terminal-heading cycles over 11.40 seconds. During
  those logged cycles, odometry yaw changed from −93.79° to +15.57°.
- The final waypoint's map yaw was +7.62°. Subsequent arrival admission
  measured +2.30° optical bearing error against the reprojected target,
  inside the strict 3° limit. It explicitly retained `camera_centered=false`.
- 16:11:25.250: the observer published QR-only completion for `Start`.

These records establish an executed final turn and successful arrival at the
planned target geometry. They do not establish physical stand alignment.
The trace's odometry yaw and route's map yaw use different frames and must not
be directly subtracted.

Generic planning computes terminal yaw as `atan2(candidate - endpoint)` in
`navigation/approach/candidate_preapproach_compute.py:334`. A calibrated
fitted-head alignment is used only when the separate `camera_alignment`
evidence exists. That evidence was absent for this route.

## The target-position discrepancy

The final admitted QR artifact carries a three-scan reconciliation proof.
Replaying that proof with production validation, after resolving only its
file references to the copied evidence, gives:

| Position hypothesis / measurement | Map x (m) | Map y (m) |
| --- | ---: | ---: |
| Frozen candidate in the arrival projection's provenance | −1.135632 | −0.478342 |
| Arrival candidate | −1.199838 | −0.291274 |
| Current cluster at the successful QR image | −1.121438 | −0.499600 |

Reprojection displaced the candidate **19.78 cm**. The current surface cluster
is **22.26 cm** from the arrival target and **2.56 cm** from the frozen
hypothesis. The corresponding earlier frame 12 gives a 22.56 cm current-target
residual. These measurements diagnose disagreement, not external ground truth
or a unique attribution to odometry versus localization. Keeping frozen map
coordinates globally is not justified by this one recording.

At successful frame 27, the predicted head center is **u=380.21 px**, whereas
the independently bound QR center is **u=726.16 px** in the 800-pixel image.
The QR corners span u=668.5–785.88. Its calibrated scan-frame bearing is
**−24.315°**; the independently reconciled cluster bearing is **−24.208°**.
The source frame 24 visibly shows the stand at the right edge with part of
the outer head clipped, while its QR remains readable.

The latest association correction therefore did useful work: it found the
current stand and bound its QR. It did not change the frozen candidate
geometry, retroactively change the executed waypoint yaw, or provide a
complete current head for the ordinary centering path.

## Why no final centering turn followed

The final observer processed 26 transform-ready, fresh images:

| Detector outcome | Frames |
| --- | ---: |
| Head acquisition deadline exceeded | 21 |
| Current head border unavailable | 3 |
| Geometry estimated but rejected by current target association | 2 |

The two geometry fits selected a narrow quadrilateral around u=610.6, rather
than the real QR/head near u=726. They correctly failed with
`head ray interval misses reconciled cluster`. No angle sample was admitted.
Four QR-associated frames were accepted, and the observer committed `Start`
with `completion_scope=discovery_only`, `facing_ready=false`, and
`stand_axis_rad=null`. Its centering ledger records zero turns for this view.

`observer/candidate_centering_receipt.py:38` stages ordinary centering only
when the current complete-head crop and head association both pass. The final
status consequently remains `fresh_admitted_current_head_required`. The
QR-only fallback in `observer/qr_observation_pose.py:126` uses zero additional
geometry grace when a valid reconciliation proof exists. It may complete
identity discovery without centering or orientation; that is the behavior
recorded here, not proof the stand was centered.

Earlier, the initial view executed three centering turns totaling 17.01°.
Its measured head shifted approximately u=91.9 → 300.5 → 317.1 → 320, still
left of center. The last capture reached peak axis consensus 7/7, but one
publication failed `metric head center exceeds candidate position bounds`,
then three failed `head_border_choice_unstable`. A peak of seven samples was
not a completed facing result. View 1 subsequently processed 365 frames with
no accepted head or centering advice.

## Why LiDAR normal alignment did not help

After the first unsuccessful camera view, optional LiDAR recovery failed to
collect a fresh cohort within its three-second capture deadline. Its saved
last error was a 3.77 ms future-TF extrapolation. The capture already retries
within that deadline, so this message does not prove that a single 3.77 ms
delay alone exhausted the capture or identify all preceding rejections.

The definite control-flow limitation is later: `candidate/lidar_acquisition.py`
returns `acquisition_unavailable` on the first capture timeout (lines 268–271),
then persists it as `terminal_reason` (lines 99–106). The second recovery call
returns the cached outcome without collecting another cohort. No local
support probe, sampling turn, or normal-alignment move was executed.

## Recommended correction

1. Add an ordinary, bounded **QR-supported framing recovery** for this case.
   Require a complete current QR quadrilateral, independent unique candidate
   and scan binding, the validated reconciliation proof, finite-range camera
   geometry, fresh synchronized timestamps and a stopped exact-odometry anchor.
   Use it solely to bring the head into view. The opposite-side code already
   supports QR-outline centering without a head angle; generalize that support
   explicitly rather than pretend that retained backside evidence exists.
   The existing status writer already gives valid centering advice priority
   over QR-only completion, so the missing step is producing justified advice.

2. Preserve the existing scan-boundary, cumulative-turn, route/permit and
   post-turn freshness checks. The calibrated solver applied diagnostically
   to this QR center requires **24.78° right** for full optical centering.
   That number is not an authorized turn. The existing boundary policy must
   bound any initial step, then require a new stopped observation. Do not
   infer a head angle or perpendicular arrival from the QR framing correction.

3. Treat a timeout or stale cohort as a bounded retryable passive acquisition
   failure, instead of permanently disabling LiDAR recovery. Keep explicit
   limits on cohorts, elapsed time and moves; require a usable fresh cohort
   before any probe. Invalid model/mount/identity evidence should remain
   terminal. Persist rejection counts so TF, timestamp and stationarity
   failures can be distinguished rather than retaining only the last error.

4. Keep identity discovery and facing completion distinct. If the required
   outcome is all five reliable stand angles, the current
   `geometry_or_qr_verified_observation_pose` completion policy does not meet
   that requirement. Better framing should precede another geometry attempt,
   and unresolved orientation must remain reported as unresolved.

At the time of the initial audit these were recommendations only. The local
implementation follow-up below records the subsequently requested changes.

## Evidence and verification

Evidence root: `results/implementation_checks/run_audit_20260930T135242Z/`.

- `source/`: copied mission, parent/last-candidate child bundles and three
  unchanged compressed camera images.
- `source_integrity.json`: all 1,242 copied files match their workstation
  SHA-256 hashes; the three images also match their capture metadata digests.
- `audit.py` and `audit_summary.json`: reproducible production reconciliation
  replay, calibrated QR-ray calculation, controller summary, revision mapping
  and final mission state. No motion or fresh perception is generated.
- `camera_summary.json`: all camera attempts, centering stages and blockers.
- `final_view_frame_000024.jpg`: original frame bytes under a previewable
  extension, without annotations or alteration.

The initial offline QR-center calculation alone did not constitute a validated
centering receipt. The follow-up regression below now exercises that boundary.

## Follow-up: why inspection proposal 004 aimed at the wrong point

The large error was present in the planned goal before the robot moved.
Generic inspection requests a fresh planning frame from the original
`source_config` and `source_registry` in `candidate/inspection_adapters.py`.
`navigation/approach/candidate_frame_reprojection.py` transforms the old
canonical odom point using the current `map <- odom` transform. It does not
replace that old point with the newly observed camera/scan position.

| Quantity | Value |
| --- | --- |
| Age of candidate's last survey observation at final planning | 826.824 s (13 min 46.8 s) |
| Canonical survey point in odom | (0.406677, −0.015985) m |
| Frozen map hypothesis | (−1.135632, −0.478342) m |
| Reprojected planning target | (−1.160826, −0.263621) m |
| Requested inspection goal | (−1.545000, −0.315000) m, yaw +7.6175° |
| Bearing from requested goal to frozen hypothesis | −21.7525° |
| Planned heading difference between those target hypotheses | 29.37° |

Since the survey reference, `map <- odom` changed by 28.13 cm translation and
−9.429° yaw. Reprojecting the aged odom point moved this candidate by 21.62 cm
in map. The execution snapshot retained the old 2 cm uncertainty; projection
displacement diagnostics did not gate precision aiming or require a new
position observation. `candidate_preapproach_compute.py` then pointed the
robot exactly at that reprojected point.

The arrival refresh moved the target another 4.78 cm to
(−1.199838, −0.291274) m. Arrival checked the reprojected hypothesis and passed
at 2.30° optical error. The contemporaneous scan cluster instead lay at
(−1.121438, −0.499600) m: 22.26 cm from the arrival target and only 2.56 cm
from the frozen hypothesis. The final measured optical map yaw was +3.581°,
about 26.66° left of the bearing to that cluster. The calibrated QR-ray solver
requires 24.78° right; it includes the camera lever arm and the QR location,
so it need not equal the planar surface-cluster bearing.

Generic goal yaw also omits the calibrated camera offset: aiming at the same
planning target would require +9.2762°, versus the requested +7.6175°.
That 1.66° contribution cannot explain the large miss. The dominant proven
mechanism is treating an aged odom landmark as current precision aiming
geometry. This recording cannot separate underlying odometry drift, AMCL
correction error, and original survey bias. It therefore does not justify
globally replacing reprojected coordinates with frozen map coordinates.

Upstream route correction remains a separate change: accept fresh uniquely
candidate-bound **position** evidence without requiring a stand normal, carry
its source/frame/age and surface-to-center uncertainty, and use it for both
inspection planning and calibrated optical yaw. If position is unresolved,
obtain bounded passive evidence before another close precision view. Preserve
survey identity, candidate exclusions, route clearance and source hashes.
The existing `validated_target_center` override is restricted to certified
opposite-face routes and a 16 cm displacement; simply enabling it or widening
that limit would not establish this contract. The local framing recovery
below addresses the failed recovery after arrival; it does not change this
upstream route-target policy.

`inspection_four_aiming.py` and `inspection_four_aiming.json` in the audit
evidence directory reproduce these quantities, with source paths, consumed
keys, artifact hashes and the limits of the retrospective comparisons.

## Local implementation follow-up

- Added `observer/qr_target_support.py` with an explicit ordinary decoded-QR
  framing policy. It requires a complete current quad and independent QR
  binding, replays the three-scan reconciliation and finite-range ray, and
  reproduces the unique raw-scan association. It does not supply an angle or
  identity authority, rewrite the candidate, or authorize motion.
- Ordinary centering can use this support when the complete-head crop or
  association fails. Existing identity/conflict gates and exact odometry still
  apply. The serialized advisory checks the support against its candidate,
  epoch, image/scan stamps, calibration and reconciliation; status publication
  retains centering priority over same-frame QR-only completion.
- Passive LiDAR timeout/stale failures now consume the existing maximum three
  cohort attempts. Exhaustion returns control without motion and without a
  permanent acquisition latch. The cumulative 120-second recovery budget and
  movement limits remain. Missing dependencies and invalid model/mount remain
  terminal; identity/hash mismatches still reject. Capture history retains
  rejection-attempt categories, accepted scan counts and the last error.
- A self-contained recorded capture-27 fixture preserves the original QR
  receipt, candidate snapshot and position epoch, with hashes. It verifies
  a required correction of 24.7829° right and an initial scan-boundary-limited
  step of 17.7829° right, followed by mandatory newer sensor timestamps.
  The observer regression must publish centering before QR-only completion
  while admitting zero angle samples. The full turn remains scan-vetoed.

Verification: 240 tests and 136 subtests passed across the recorded regression,
QR proof/transport, observer and parent/child centering, opposite recovery,
partial/clipped head recovery, and passive LiDAR capture/retry suites. The
test log is `implementation_tests.txt` in the audit evidence directory.
`git diff --check` passed. Replay accepts only a 1e-9 absolute tolerance for
platform differences in floating-point association results; discrete fields,
source identities and source hashes remain exact.

This is an offline implementation. No deployment, physical centering success,
or reliable stand angle is claimed; the newer workstation run was untouched.
