# Wall-facing inspection audit: 20260930T150526Z

Audited September 30, 2026 through workstation SSH alias `mii002`
(hostname `mii001`). Mission:
`stand_explore_exact2_camera_all5_20260930T150526Z`, started **17:05:26 Berlin time**.
All eight parent/child evidence bundles record clean revision
`99ac30090c812c866aec818ff87c125e42563dae`, including the previous correction.
All **706 copied files** match workstation SHA-256 hashes.

## Finding

The previous fix is present and rejected the old wall hypothesis. The remaining
failure is **using a historical candidate position without current physical
support**, followed by an observer timeout gap that keeps inspecting that
unsupported location.

The stopped fourth visit was `survey_candidate_0005`. Its original location
closely matches a previously confirmed stand location. Its projected target
was inside map-free space, so the new static-map/morphology gate accepted it.
At arrival, however, current LiDAR had no target at the predicted distance,
and the camera was facing the radiator. No head, QR identity, axis, or camera
centering command was accepted at this visit.

The evidence supports an outdated target position, consistent with accumulated
odometry error. It does **not** establish that the original survey hypothesis
was a fake stand, nor does it show a mathematical frame-transform error.

## What happened

| Berlin time | Recorded event |
|---|---|
| 17:09:27 | `0004` rejected with `target_static_map_incompatible` and `unresolved_morphology_conflict`; its keepout retained. |
| 17:10:17–17:15:01 | Candidates `0003`, `0001`, `0002` resolved as `QR_003`, `Start`, `QR_002`. |
| 17:15:19 | `0005` selected; the other remaining candidate, `0006`, failed route uncertainty admission. |
| 17:15:42–17:16:26 | The certified approach to `0005` completed, reporting 1.718 m travelled in 43.975 s. |
| 17:16:32 | Fresh arrival geometry and target-map checks passed; first passive camera capture began. |
| 17:16:57 | `KeyboardInterrupt` stopped capture after 25.012 s. |

There was **no centering turn, LiDAR recovery, or additional inspection route
at the wall**. The mission remained incomplete with three QR identities and
two facing-ready stands. The robot reached the wrong inspection aim and was
still in its first camera attempt when stopped.

## Why the new gate allowed this target

`0004`, the old wall hypothesis, was never visited. It was deferred before
the first route selection and remains `target_reconciliation_required`.

`0005` had different evidence:

- Original position `(1.16218, 0.77162)` m: 5.68 cm from the candidate confirmed
  as `QR_001` in the 13:52 run, and 1.17 cm from the corresponding 14:11 candidate.
- 47 distinct scans from the second survey viewpoint; 47/47 morphology inliers,
  median width 4.81 cm and median absolute deviation 0.29 cm.
- Each proposal had two LiDAR points. Repeated consistency is evidence for a
  hypothesis, not independent proof of object identity.
- No unresolved morphology conflict. Frozen static-map clearance was 18.84 cm.

The gate tests static-map compatibility and recorded morphology conflicts;
it does not require a new physical observation. Its production replay exactly
reproduces acceptance at selection and arrival.

| Target epoch | Map x / y (m) | Displacement from frozen | Static clearance |
|---|---:|---:|---:|
| Frozen survey | 1.16218 / 0.77162 | 0 | 18.84 cm |
| Fourth selection | 1.29940 / 0.43312 | 36.53 cm | 52.69 cm |
| Stopped arrival | 1.31082 / 0.31381 | 48.13 cm | 51.92 cm |

The required nominal-plus-uncertainty clearance was 8 cm. Consequently, this
target passed all new checks even though current sensor evidence contradicted
its assumed location. Arrival also passed range/bearing checks: predicted
base-to-target distance 51.63 cm, optical bearing error only 0.141°.
Those checks aim at the supplied coordinate; they do not establish a stand
at that coordinate.

Relevant code: `navigation/approach/candidate_target_admission.py:63`,
`navigation/approach/candidate_preapproach_selection.py:148`, and
`real_robot/candidate/approach.py:_admit_camera_arrival_geometry`, beneath
`scripts/aufgabe04/`.

## Position evidence and its limits

The stored canonical odometry point is `(2.615327636, 1.330682205)` m.
The frozen map-from-odometry transform was
`(-1.508448323, -0.444277968, -0.043423437 rad)`; arrival used
`(-1.572247943, -0.232618216, -0.283354626 rad)`.
Applying the recorded transform reproduces the arrival target exactly.
The map/odometry yaw change is 13.75°. The candidate was last observed
**444 seconds earlier**, yet still carried **2 cm position uncertainty**.

Crucially, applying the same rigid transform to the robot and target cannot
change their relative bearing. The problem is continued reliance on the
historical odometry point, not the arithmetic of changing coordinate frames.
The projection code changes x/y while retaining the candidate uncertainty;
it does not independently observe the object again
(`navigation/approach/candidate_frame_projection.py:226`).

Across all 66 captured scans with valid scan transforms, the nearest returns
to the original frozen map point are 2.76–4.22 cm away, at approximately
44.6° left of the scan axis. The projected target is approximately 1.35°
right of that axis, and its nearest return anywhere in the scan is
30.80–44.73 cm away. The captured robot pose remained effectively stationary.
This is consistent with the robot aiming beside the stand after target-position
drift. These returns alone do not identify `QR_001`, and saved telemetry cannot
separate all odometry, localization, and initial-estimation errors into an
independent physical ground truth. Keeping the old map point unconditionally
would therefore be an unjustified general fix; fresh association is needed.

## What the camera and LiDAR actually accepted

The saved image shows the radiator and wall:

![Last saved camera image, original bytes](../../results/implementation_checks/run_audit_20260930T150526Z/camera_last.jpg)

Across 77 saved frames, 66 reached detector processing and 11 exhausted TF
retries. All 66 detector frames had zero accepted target associations and no
final head, QR, or axis evidence. The narrow LiDAR cone had **zero in-range
samples**: expected range was approximately **0.368–0.588 m**, while available
nearest cone returns were **1.034–1.308 m** away. Intermediate border-refinement
successes did not become accepted stand detections.

The metadata also reports `candidate epoch displacement outside recovery bound`.
The observer thus recognized that it could not support/reconcile the target,
but this did not promptly end the inspection.

## Why inspection continued

The configured process timeout was **90 seconds**; the user stopped it at
25.012 seconds. `head_acquisition_deadline_exceeded` on 20 frames refers to
each image's processing/freshness budget, not expiration of the whole observer.

The short five-second acquisition window starts only after an accepted frame.
Every processed frame failed LiDAR association, so the frame acceptance path
returned false and the acquisition timer never started. Status records
`deadline_monotonic_sec=null`. The epoch-recovery timer also requires a valid
recovery proof and does not cover rejected reconciliation.

Relevant code beneath `scripts/aufgabe04/real_robot/observer/`:

- `evidence.py:522`: failed association prevents accepted-frame evidence.
- `inspection_progress.py:172`: rejected frames return before starting the
  bounded acquisition window.
- `position_epoch_opportunity.py:19`: recovery opportunity depends on a valid
  recovery proof.
- `node.py:1639` and `head_acquisition_schedule.py:27`: per-image work deadline.

If the full process timeout had elapsed, the LiDAR rejection history could
qualify for optional recovery and subsequent viewpoint search. That is a
possible continuation in `timeout_policy.py:42` and
`real_robot/candidate/inspection_execution.py:193`, **not an action observed
in this stopped run**.

## Correction implemented locally

The observer already attempts bounded current-scan reconciliation. The new
`target_support_failure.py` policy consumes explicit missing-range evidence
when that reconciliation has not established a current target. It requires
at least seven distinct image/scan tuples over at least five seconds, fresh
within 0.5 seconds and synchronized within 0.1 seconds, with at least two
seconds of image-time span and a maximum 15-second history. Robot position
must remain within 2 cm and 2 degrees in one motion epoch.

The absence check uses raw finite returns over the entire allowed registration
envelope (15 degrees), with zero samples inside the target's range interval.
Empty/invalid rays, ambiguous clusters, stale data and missing TF cannot prove
absence. Positive raw support, accepted observations, current head association
or validated reconciliation clear the window. Motion, changed target bindings
and sensor-contract resets also discard accumulated evidence. Existing
reconciliation displacement/bearing bounds and accepted-frame rules remain
unchanged.

After stronger QR/head/centering outcomes have had priority, the observer
publishes a fresh content-hashed `target_support_failure.json`, with state
`target_reconciliation_required`, and exits cleanly. The receipt binds the
candidate snapshot, target position/key, stream, profiles and motion epoch.
It grants no motion, geometry update or completion authority.

`target_support_handoff.py` authenticates that receipt and the final status
after the child has been reaped. Failed/forced/deadline exits, conflicting
perception artifacts and mismatched provenance remain terminal errors.
An authenticated failure uses the existing `candidate_target_ineligible`
path: persist `target_reconciliation_required`, skip LiDAR/centering/viewpoint
recovery and further attempts for this target, preserve all obstacle keepouts,
then continue other eligible candidates. Insufficient resolved identities
still leave the mission incomplete. The event now correctly reports
`retry_eligible=false` for this disposition.

This correction bounds the initial passive inspection at an unsupported aim.
It does not relocate that candidate or establish a replacement stand identity.

## Evidence and verification

Evidence root: `results/implementation_checks/run_audit_20260930T150526Z/`.

Offline replay of **all 77 original captures**, including every intervening
failure, finds 66 detector results with accepted source freshness and 11
TF-only captures that supply no negative evidence. With the production raw
registration envelope, the first terminal receipt occurs at capture
`000020`: **19 distinct negative tuples, 5.253284 seconds elapsed and
5.265656 seconds of image-time span**, or **7.535449 seconds after camera
capture invocation**. Capture `000016` has no valid returns
in the narrow cone, but the full allowed registration envelope has finite
background returns outside the target range. The policy therefore does not
mistake invalid rays for absence.

The regression fixture under
`tests/aufgabe04/fixtures/target_support_20260930/` preserves original scan/TF
data and source hashes. It covers the contiguous sequence through that stop,
TF exclusion and an independently recorded valid head/current-scan control.
This is a recorded-sensor replay, not a physical run of the new code.

Focused validation covers the pure policy, actual observer record/status
methods, parent process handoff, no-recovery mission disposition, previous
wall gate, valid position-epoch/opposite reconciliation, centering and measured
head processing. The opposite-arrival test fixture now supplies the genuine
map and snapshot-bound plan required by the existing map gate and verifies
that the retained corrected target passes that gate.

Final focused validation: **312 tests and 364 subtests passed** across 23
distinct test files. This combines the 18-file parent/mission regression set
(237 tests / 258 subtests, including targeted reruns after fixture repairs),
the pure policy and actual observer integration (26 / 21), existing centering
and measured-head processing (45 / 43), and recorded sequence replay (4 / 42).
Python compilation of all changed production modules and `git diff --check`
also passed. No live robot validation was performed.

- `source.tar.gz`, `source/`, `source_integrity.json`: complete stopped mission
  and eight bundles, 706 verified files.
- `audit.py` / `audit_summary.json`: clean revisions, gate receipts, timeline,
  interruption, and incomplete goal.
- `population_compare.py` / `population_comparison.json`: cross-run spatial
  comparison, source morphology, and exact production gate replay.
- `frame_audit.py` / `frame_audit.json`: frame/scan reproduction, historical
  QR-associated surface comparison, and coordinate consistency.
- `camera_evidence.py` / `camera_evidence.json`: all-frame detector/association
  counts and original image hashes.
- `camera_first.jpg`, `camera_last.jpg`: original saved image bytes.
- `target_support_replay.json`: complete 77-capture replay, including raw
  broad-envelope associations, source-freshness checks and first deferral.
- `population_parent_regression_summary.json`: focused parent/mission and
  positive-reconciliation test results, including targeted fixture repairs.

Workstation access was read-only. The correction is in the local workspace;
it has not been deployed and no robot commands were issued.
The final check at 17:28 Berlin time still identified this mission as latest,
with the same clean revision and no matching mission processes running;
see `final_workstation_state.json`.
