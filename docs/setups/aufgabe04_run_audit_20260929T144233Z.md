# Opposite-side continuation audit: 20260929T144233Z

The opposite-side route was rejected before dispatch. The robot had successfully
certified the stand's backside, but two opposite routes failed uncertainty
admission. A subsequent endpoint-angle rejection aborted the standoff search
and discarded those uncertainty failures, bypassing the existing bounded
stationary-localization retry.

The robot then executed a separate diverse-inspection fallback. That movement
was progressing when the run was interrupted; its ROS shutdown error is a
separate event from the opposite-route admission failure.

## Recorded sequence

- Run: `stand_explore_exact2_camera_all5_20260929T144233Z`.
- Workstation and recorded revision: clean
  `6423304143b5e230bb2f65731e867d22b543ac2d`.
- Candidate: visit 001, `survey_candidate_0001`. Its first approach completed
  at **14:48:29.743 UTC / 16:48:29.743 CEST**.
- Backside certification succeeded at approximately **14:48:39.17 UTC**, with
  seven accepted axis samples and eight accepted observation frames. Its bounded
  orientation had a half-width of **7.815 degrees**. The stand's QR identity
  had not yet been admitted in this run.
- The 0.50 m opposite approach failed dry admission at **14:48:57.436 UTC**.
- The 0.45 m alternative failed dry admission at **14:49:10.046 UTC**.
- The next, closer alternative raised
  `BoundedOrientationViewUnavailableError`: `bounded orientation endpoint
  exceeds QR viewing obliquity for plausible angles`.
- Inspection progress recorded only that final reason, with
  `reason_code: route_unavailable`, empty underlying evidence, and planning
  epoch 0. No opposite localization-refresh request or checkpoint dispatch
  followed.

The two opposite children explicitly record `motion_published: false`.
Neither was a live driving failure. Camera centering was blocked by its travel
budget at this view, but that did not prevent successful backside certification.

## The uncertainty rejections were valid

The production budget evaluator exactly reproduces both failures over all 284
sampled route intervals:

| Approach standoff | Available clearance at limiting interval | Required clearance | Remaining margin | Rejected intervals |
| --- | ---: | ---: | ---: | ---: |
| 0.50 m | 0.372506 m | 0.469263 m | −0.096757 m | 47 |
| 0.45 m | 0.422726 m | 0.437278 m | −0.014552 m | 23 |

The first limiting interval is `segment:0004:0053`, near the route endpoint;
the second is `segment:0005:0010`, also at the endpoint. The fixed allocation
is 0.190 m: robot radius 0.105 m, collision reserve 0.020 m, tracking reserve
0.030 m, drift reserve 0.020 m, and braking/latency reserve 0.015 m.

The additional localization/heading allocations are 0.140675/0.138588 m for
the first route and 0.116311/0.130968 m for the second. Heading uncertainty
contributes more near the far end of the route because of its distance from
the stationary reference. These failures must not be resolved by weakening
the clearance thresholds.

Offline planning reproduces the two recorded endpoints exactly. The 0.40 m
alternative quantizes to `(-1.495, -0.515)`, with actual target range
0.392111 m. Its worst-case viewing angle is **20.595287 degrees**: radial
error 4.838133 degrees, orientation half-width 7.815210 degrees, and position
reserve 7.941944 degrees. That exceeds the existing **20-degree** limit. The
final bounded standoff, approximately 0.399535 m, would quantize to the same
endpoint and fail the same check. The angle rejection itself is therefore
correct; its treatment as a terminal search error is the defect.

## Control-flow defect

At the recorded revision, the per-standoff planning handler in
`scripts/aufgabe04/real_robot/candidate/approach.py:1922` immediately converts `BoundedOrientationViewUnavailableError`
into a generic `CandidateInspectionRouteUnavailableError` and raises it.

At that point the loop already holds two verified no-motion uncertainty
rejections. The immediate raise skips the aggregate exhaustion block around
line 2045, which would preserve those failures and emit the typed
`opposite_route_uncertainty_exhausted` result at line 2072.

`opposite_localization_retry.py:34` permits one fresh stationary planning epoch
only for that typed failure with verified no-motion/no-permit evidence. The
generic error has neither, so it correctly declines to retry. The enclosing
inspection controller then chooses a diverse local view. The defect is the
premature loss of recovery evidence, not the retry wrapper accepting too few
failure types.

Checkpoint recovery is also bypassed by the early exit. It is conditional on
the allowed recovery epoch, a certified current target center, verified rejected
routes, and fresh prefix/suffix admission; the log does not establish that a
checkpoint would have succeeded.

Git history dates the immediate exception conversion to `a9aac95e` on September
16. The latest range/centering correction did not modify the opposite planner,
this exception handler, or route uncertainty admission. This is an existing
mixed-failure handling defect exposed by the current geometry and uncertainty.

## Why a fresh stationary retry is worth preserving

The first complete stationary window collected for the subsequent fallback,
**14:49:16.596778–14:49:18.952877 UTC**, still precedes its motion. All five
samples give a maximum planar variance of 0.0021849304 m² and maximum yaw
variance of 0.0026580804 rad².

As a sensitivity calculation, applying that complete covariance envelope to
the already-recorded route geometry gives:

- 0.50 m route: still rejected, margin **−0.024874 m**.
- 0.45 m route: all 284 intervals pass, margin **+0.030451 m**.

This uses the entire later window, not a selected low-variance sample. It is
still only a diagnostic: the route and heading reference are frozen. Actual
recovery must admit the new pose/transform, reproject the original backside
evidence, replan, and pass fresh dry and live checks. The calculation supports
restoring that bounded opportunity; it does not authorize the recorded route.

## What the later motion actually did

The subsequent child was `...candidate_001_inspection_002`, purpose
`diverse_inspection`. Its 0.200939 m route passed dry admission at
**14:49:40.324 UTC** and was dispatched at approximately **14:49:52 UTC**.

Its 171 controller trace rows span 16.906 seconds and show approximately
**0.170546 m** cumulative odometry translation. At the last sample it was
**0.023422 m** from the endpoint, inside the 0.03 m position tolerance. It
was in `terminal_heading`, with **5.292 degrees** remaining against a
3-degree heading tolerance, and was still commanding a turn.

At **14:50:10.610 UTC** the child recorded `RCLError: Failed to publish:
publisher's context is invalid`. At **14:50:10.782 UTC** the parent recorded
`KeyboardInterrupt`. Their timing supports a shutdown-related publisher error;
the saved evidence does not identify the source of the interrupt. No controller
timeout or obstacle stop preceded it.

Final saved progress is one of five QR identities (`QR_003`), one facing-ready,
candidate 0001 still inspecting, and four unvisited candidates. There is no
`mission_failure.json` or normal mission-completion record.

## Recommended correction

Treat a typed bounded-orientation endpoint rejection **inside the standoff
loop** as a failure of that proposed view: record it and continue the bounded
search. Preserve earlier verified uncertainty rejections in the eventual
aggregate result so the existing one-refresh policy can run. If its conditions
are met after that refresh, allow the existing checkpoint logic to evaluate its
bounded prefix/suffix alternative.

Do not change the global QR-angle limit, collision/uncertainty margins, or
maximum retry count. A source-orientation rejection before a valid direction is
established must still terminate that opposite attempt. Malformed evidence and
unexpected errors must still propagate, and no failure after motion may be
reclassified as a no-motion retry.

Regression coverage should exercise the mixed sequence seen here: two verified
uncertainty rejections followed by a typed endpoint-angle rejection. It should
check preserved evidence, exactly one fresh localization epoch, correct
checkpoint eligibility, and distinct outcomes for all-geometric rejection,
malformed input, and any failure after motion.

An offline coordinator replay reproduces the pre-fix loss of two verified
uncertainty failures: zero localization refreshes and zero checkpoint-selection
opportunities. A mocked counterfactual that classifies only the typed endpoint
exception as local feasibility reaches one refresh and one checkpoint-selection
opportunity, while an unrelated malformed-input `ValueError` still propagates.
This tests control flow, not whether a motion would be admitted.

The existing retry test supplies `CandidatePreapproachUnreachableError` for the
inner standoffs, which takes the normal continue path. The bounded-orientation
tests check angle rejection separately. Their missing combined failure chain
explains why the premature exit was not covered.

## Evidence and scope

Evidence is copied under
`results/implementation_checks/run_audit_20260929T144233Z/`. Eleven principal
source files match workstation SHA-256 hashes.

- `replay_uncertainty_admission.py` and `uncertainty_admission_replay.json`
  reproduce the admission failures and clearly separate the later-window
  sensitivity calculation.
- `analyze_opposite_trace.py` and `opposite_motion_trace_audit.json` summarize
  child/parent timestamps, motion, observer status and final progress.
- `replay_opposite_control_flow.py` and `opposite_control_flow_replay.json`
  reproduce endpoint geometry and the mixed-failure recovery branch.
- Principal raw sources are `station_segment_runs.csv`, both opposite
  uncertainty-budget files, `inspection_progress.json`,
  `inspection_handoff_events.jsonl`, and the fallback controller trace.

## Implemented correction

The per-standoff planning handler now records
`BoundedOrientationViewUnavailableError` through the existing local-feasibility
path and continues the bounded search. Aggregate exhaustion retains the prior
verified uncertainty failures, enabling the existing one-refresh policy and
conditional checkpoint opportunity. The source-orientation handler before the
loop is unchanged. Unrelated malformed `ValueError`s still propagate.

No viewing-angle or clearance threshold, motion budget, retry count, source
receipt, or motion-permit rule is changed. The production edit is confined to
this exception-handling branch.

`verify_opposite_correction.py` preserves the pre-fix replay output and checks
the patched production coordinator. It confirms one refresh instead of zero,
two planning epochs with four distinct rejected child IDs, and two preserved
uncertainty failures per epoch. Checkpoint selection is reached only after the
refreshed epoch; malformed evidence remains terminal. The saved-geometry replay
still accepts the two outer endpoints and rejects both inner endpoints.
Results are in `opposite_correction_verification.json` beside the original
evidence. This coordinator replay mocks child outcomes and establishes recovery
control flow, not successful robot motion.

Validation: **14 opposite-localization tests passed**, including five new cases
for mixed-failure exhaustion, successful refreshed planning, geometric-only
exhaustion, malformed evidence after uncertainty rejection, and source-orientation
rejection. Three new regression cases fail when the original epoch handler is
substituted in memory, without modifying the working tree. Existing motion and
permit guards also pass. The broader eight-file integration suite passed
**110 tests and 70 subtests**, covering opposite fallback/checkpoints, bounded
orientation, inspection execution, frame projection, and candidate recovery.
JUnit results are saved as `correction_integration_tests.xml` in the evidence
directory. `git diff --check` and an independent read-only review passed.

No deployment or robot operation was performed during the correction.
