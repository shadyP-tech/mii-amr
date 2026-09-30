# First-arrival LiDAR acquisition and missing camera handoff

Run: `stand_explore_exact2_camera_all5_20260930T112749Z`  
Executed revision: `e018878a583131f5dc051c0ea7ce777df91429cf`  
First candidate: `survey_candidate_0003`  
Audit date: September 30, 2026

## Finding

The new LiDAR acquisition controller prevented the first camera observation
from starting while it searched for a usable head normal. It completed two
support moves, acquired three local scan cohorts, and reserved a third support
move. It never entered the normal-alignment branch. The user confirmed manually
stopping the run or closing its terminal at this point.

This is a sequencing regression in the newly added acquisition stage. A finite
number of LiDAR moves still allowed more than two minutes to pass after arrival
without using the existing camera head-angle detector. The observed orbit did
not use a measured head frame and did not establish perpendicular alignment.

## Recorded sequence

Times below are UTC; local time was UTC+2.

| Time | Event | Result |
| --- | --- | --- |
| 11:32:31.386 | Initial candidate approach completed | Camera had not started |
| 11:32:38.208 | Local `view_00` captured | 2 of 3 scans passed the surface fit; no usable view |
| 11:33:33.808 | Support probe 1 completed | Planned route 0.032 m; stopped scan poses differed by 0.0075 m |
| 11:33:41.207 | Local `view_02` captured | 0 of 3 scans passed; no usable view |
| 11:34:39.436 | Support probe 2 completed | Planned route 0.529 m; stopped scan poses differed by 0.498 m |
| 11:34:46.508 | Local `view_04` captured | 0 of 3 scans passed; no usable view |
| After the final capture | Third support proposal persisted | No third-probe planning or motion artifacts |

There were about 135 seconds between the initial motion completion and the last
capture. The scan-time pose displacements in the table are net displacements,
not the motion executor's accumulated distance estimates.

No camera attempt, `observer_capture` event, camera axis observation,
`candidate_arrival_admission.json`, or final LiDAR `arrival_review.json` was
saved. There is no evidence that camera head detection ran and rejected the
head. Its configured 90-second timeout had not started.

The parent bundle lacks an end time and exit status, and no mission failure or
summary was written. The user's confirmation explains the interrupted ending;
the artifacts alone do not identify the exact interruption instruction or signal.
The last persisted proposal precedes the next fresh planning-frame preflight.

## Why the motion was an orbit rather than perpendicular alignment

`scripts/aufgabe04/real_robot/candidate/lidar_acquisition.py:55` chooses
`normal_alignment` only when a multiview hint exists. Without it, line 72
proposes support positions at the original stand-to-robot bearing, then +60 and
-60 degrees. A valid current-view fit could guide these proposals, but none of
the three local cohorts qualified even for that fit.

Every saved motion proposal is `support_view`; the alignment-move count stays
zero. Probe 1 requested the original direction and added almost no independent
viewing information. Probe 2 requested +60 degrees, and its measured scanner
bearing changed by about 56.8 degrees from the preceding scan position. The
planned third probe would request -60 degrees relative to the original view.

These directions are relative to the initial approach, not the head tangent or
normal. The existing camera policy also has 90-degree orbit options, but that
policy was never reached in this run. There is no evidence here of an executed
head-normal calculation being rotated incorrectly by 90 degrees.

Both executed probe artifacts use the label `purpose: lidar_axis_hint` and a
field named `view_normal_rad`, but have no source observation or calibrated
`camera_alignment`. In this branch that field is the requested radial viewing
direction, not a fitted head normal. The generic planner points the robot base
toward the candidate centroid; the calibrated fitted-center/head-normal branch
was never selected. These artifact names should not be read as proof of head
alignment.

## Why no usable normal was obtained

The replay calls the production scan fitter, current-view fitter and multiview
hint generator on the saved receipts without changing their thresholds.

| Source | Passing scans | Main limitation |
| --- | --- | --- |
| Survey viewpoint 1 | 0/78 | Only 1–2 returns in the candidate envelope |
| Survey viewpoint 2 | 21/77 | Mostly 3 returns; insufficient passing fraction |
| Local `view_00` | 2/3 | Last scan split into two fragments at missing beam 2 |
| Local `view_02` | 0/3 | Candidate returns split across the first/last scan indices; one scan also has an internal gap |
| Local `view_04` | 0/3 | First two scans have only 3 returns; last scan is split at the scan boundary |

The fitter requires four spatial returns per accepted scan and at least three
accepted scans covering 75% of a stopped viewpoint. The local capture records
exactly three scans over about 0.19 seconds, making one rejected scan enough to
reject that entire viewpoint. Two separated usable viewpoints are required for
the normal-alignment hint. No viewpoint qualified here.

The first two local scan fits themselves are plausible under the existing
checks, with observed spans of approximately 6.35 cm and 4.81 cm. Increasing the
clustering distance would not repair the missing raw beam in the third scan.
The model and scanner-height observability gates passed; they were not the
reason alignment remained unavailable.

The boundary issue also cannot be repaired by blindly joining indices 219 and
0. The saved local receipt/cohort lacks the original `angle_max`, and several
inferred boundary gaps exceed one nominal beam interval. A diagnostic forced
join still produces at most 2/3 passing scans for `view_02` and 1/3 for
`view_04`; it would not provide the missing multiview hint. Such a forced join
is not accepted production evidence.

## Why camera detection did not take over

`scripts/aufgabe04/real_robot/candidate/inspection_adapters.py:471` synchronously
calls `prepare_lidar_camera_arrival` before line 498 starts
`execute_candidate_inspection`. The LiDAR controller can consume three support
moves and two alignment moves before returning. No camera observation is
interleaved, and the camera timeout does not bound this preliminary stage.

Consequently, insufficient LiDAR support caused further orbiting instead of
allowing the existing camera centering and head-angle detector to contribute.
The previous offline tests covered bounded move counts and geometric rejection,
but did not require a camera observation at the first arrived inspection point.

## Recommended correction

1. Preserve normal-based initial planning when usable LiDAR evidence already
   exists. At the first stopped arrival, start the camera observer promptly
   through the existing arrival, candidate-association and centering gates.
   Missing LiDAR orientation must not impose a sequence of preliminary orbits.
2. Preserve camera-certified backside/opposite-side routing priority. If camera
   evidence remains unresolved, allow a bounded LiDAR recovery step and return
   to camera observation after that step. Share the inspection and motion
   budgets, and bound the acquisition stage by elapsed time as well as moves.
3. Improve acquisition using a brief bounded stopped cohort with an explicit
   support test, preserving distinct-scan, beam-count, fit-fraction and ambiguity
   requirements. Prefer actions that measurably improve angular sampling or
   independent viewpoint support; reject redundant moves such as probe 1.
   Preserve original topology evidence before considering boundary-aware fits.
4. Continue to use fitted center and the two signed choices of the unsigned
   normal for calibrated camera-pose planning. Count an arrival as perpendicular
   only after the existing fresh alignment verification succeeds. An arbitrary
   support orbit must remain explicitly unverified.
5. Add integration regressions for this exact 2/3, 0/3, 0/3 sequence: camera must
   be attempted at the first arrival; an optional recovery must return to camera;
   no usable normal must never be reported as normal alignment. Record durable
   phase boundaries around observation, planning preflight and motion, including
   interrupted execution.

Simply lowering fit or identity thresholds is not the correction. Restoring the
camera handoff and improving measurement support address different problems,
and both matter for eventual head-facing arrival. This run contains no saved
camera image or physical head-angle ground truth with which to validate final
alignment accuracy.

## Reproduction and evidence

Read-only copies of the latest mission and parent run bundle are under:

`results/implementation_checks/run_audit_20260930T112749Z/source/`

The reproducible scan audit and its output are:

- `results/implementation_checks/run_audit_20260930T112749Z/lidar_replay.py`
- `results/implementation_checks/run_audit_20260930T112749Z/lidar_replay.json`

Run the replay from the repository root using an environment with the project's
Python dependencies. It records source hashes, individual raw beam indices,
cluster metrics, production fit decisions, view-level diagnostics and measured
probe displacements. Source evidence and generated results retain the
repository's existing results-ignore policy.

The investigation above changed no production behavior. The subsequent
correction is described below; no robot deployment or motion was performed.

## Implemented correction

The first admitted arrival now starts camera acquisition and visual centering
without running a LiDAR acquisition loop first. Initial LiDAR hints still feed
the existing calibrated normal-based approach planner. Their absence no longer
blocks the first camera head-angle observation.

Arrival admission measures optical bearing using the camera's calibrated
translation and yaw, while preserving the physical range check at the robot
base. The strict 3-degree and passive-centering 6-degree limits are unchanged.
The runtime binds the calibration's base frame to the robot profile. A
calibrated arrival that fails admission is not redirected by a map-only
base-to-centroid correction. Its evidence remains unverified for head alignment
and still requires live target association.

Optional LiDAR recovery is available only after an unobservable camera attempt.
Camera angle guidance, certified opposite-side motion, retained backside
orientation and camera distance recovery retain priority. Each LiDAR recovery
call executes at most one displacement and returns to the next camera attempt.
Both use the existing camera-view budget and shared candidate route-proposal
ledger; recovery does not create a separate set of camera attempts.

Recovery keeps its local receipts and cumulative limits across calls: at most
three support proposals and two alignment proposals, plus a 120-second budget
for time spent inside recovery. Time spent in camera observation is excluded.
The elapsed budget is checked after planning and immediately before dispatch.
It prevents starting more optional work, rather than interrupting an active
certified motion leg, which retains its existing execution limits.

At a stopped view, recovery can acquire up to three three-scan cohorts before
moving. All scans, including rejected fits, count toward the same viewpoint's
support fraction. They do not become separate independent viewpoints. Fresh
arrival verification uses only the newest cohort; older planning support cannot
stand in for a fresh arrival measurement. Beam-count, fit-fraction, ambiguity,
mounting, freshness and topology checks remain unchanged.

Support goals are checked after endpoint materialization. They must provide at
least 20 degrees of axial viewpoint separation or at least 5 cm of inward range
improvement. The recorded tiny first probe fails this check. Fitted normal-pose
corrections are separate and are not subject to support-view novelty criteria.
An ordinary support move never becomes proof of perpendicular alignment.

After a recovery move, fresh range and optical admission precede the camera.
When fresh LiDAR evidence verifies both normal alignment and camera centering,
the measured fitted pose is retained without the old centroid-yaw correction.
Otherwise alignment remains unverified. Admission failure or an unknown motion
outcome propagates instead of authorizing another displacement within recovery.

Durable history now records camera/recovery ordering, preflight boundaries,
capture stages, proposal reservations, completed motion, budget exhaustion and
interruption. Per-step arrival reports are immutable so later recovery calls
cannot overwrite earlier evidence.

### Validation

- Focused integration/planning/perception suite: **574 tests and 312 subtests
  passed**, with one recorded QR decode test skipped because local OpenCV lacks
  the deployed WeChat backend. This includes camera-first ordering, the recorded
  insufficient-fit sequence, newest-cohort verification, motion/time budgets,
  calibrated arrival, candidate association and opposite-side regressions.
- Runtime suite: **64 tests and 18 subtests passed** with local Unix-socket access.
  The wrapper fixtures now include their scan topic/frame and a valid camera
  calibration so they exercise the current configuration contract.
- One unrelated runtime test was excluded from that passing runtime run after
  independently reproducing its failure on a clean archive of `e018878`:
  `test_runtime_localization_recovery_uses_scoped_permit_without_parent_input`.
  Its synthetic artifact lacks `odom_execution_certificate_sha256`. The motion
  certificate validation was not weakened to accommodate the fixture.
- `git diff --check` passes. No deployment or robot motion was performed.

Evidence is saved under the audit directory in `correction_focused_tests.xml`,
`correction_runtime_validated.xml`, `correction_validation.json`, and
`baseline_runtime_fixture.{py,json,log,xml}`. An earlier broader run also records
the sandbox's local-socket restrictions; the runtime rerun above resolves those
environment failures. The focused and runtime counts overlap on the new
calibration/base-frame binding test and should not be added as distinct tests.

The existing robot command needs no new flags. Physical success and final head
alignment still need verification using the updated robot-side checkout.
