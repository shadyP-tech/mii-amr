# Run audit: 2026-09-18 13:29:46 UTC

The mission terminated because a camera-centering execution module is missing.
The light-grey head was detected and its pose passed the current quality gates
before the crash. The captured grey view is its back side; this run does not
resolve the earlier frontside-recording failure.

## Run and evidence

Run: `stand_explore_exact2_camera_all5_20260918T132946Z`.
Recorded revision: `e0cbf5e`, with clean recorded Git status. The workstation
checkout matched this revision when inspected. Original mission artifacts,
camera captures, and parent/child diagnostic bundles were copied locally from
`mii001` for this read-only audit. No ROS commands, motion, remote edits, or
production-code changes were made.

Local mission directory:
`results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260918T132946Z/`.
Parent bundle:
`results/real_runs/stand_explore_exact2_camera_all5_20260918T132946Z/`.
Derived summaries, reproduction script, source hashes, and targeted test output:
`results/aufgabe04/implementation_checks/run_audit_20260918T132946Z/`.

## Terminal cause — missing execution adapter

The parent `terminal_run.log`, lines 378–401, records an uncaught
`ModuleNotFoundError` for
`scripts.aufgabe04.real_robot.execution.candidate_centering`.

`autonomous_runner/runtime.py:1954` imports `CandidateCenteringChildRequest`
and `run_candidate_centering_child` from that module inside
`_run_camera_centering_turn`. Neither symbol has an implementation anywhere
in the current repository. The import/call was introduced in `f8cde1d`; the
referenced module has no history in the available Git history. This is an
incomplete integration present in the recorded code, not evidence that the
workstation failed to receive the latest depth-based change.

The second candidate emitted a valid centering advisory. Its measured head
center was `(334.24, 290.10)` px versus a target horizontal center of 400 px.
The requested turn was `0.0939398932 rad`, or **5.38236 degrees**. The
inspection coordinator persisted `phase: turn_reserved`, then called the
missing adapter. Recorded centering travel remains zero; no centering child
was launched, since the exception precedes even reading the advisory in the
execution adapter. Both observer subprocesses exited normally with return
code zero. The parent failed while consuming the second observer's result.

Directly invoking `_run_camera_centering_turn` with placeholder arguments
reproduces the same missing-module exception before any ROS or motion work.
Argument parsing and normal startup do not exercise this deferred import.
The first candidate's completed recommendation took precedence over optional
centering, so that successful observation never entered the broken branch.

## What completed

- Both LiDAR survey viewpoints completed. Recorded visited coverage was
  95.31%, exceeding the configured 95% threshold.
- Both attempted candidate approach legs completed.
- First inspected candidate, `survey_candidate_0003`: visually red, QR-facing.
  Camera committed **QR_003** and an accepted current measured-head pose at
  13:34:35 UTC. Its candidate was confirmed and marked facing-ready.
- Second inspected candidate, `survey_candidate_0001`: visually light grey,
  plain back side with a small handwritten label. At 13:35:49 UTC its current
  head and pose passed; the observer emitted the centering advisory. It did
  not yet establish QR identity or complete inspection.
- Mission progress remained **1/5 confirmed**, with four candidate-pool entries
  still unvisited. Six LiDAR hypotheses were retained for the five-stand goal;
  that pool count is not six confirmed physical stands.

## Light-grey head evidence

The second view processed two detector images. The first was rejected for
`head_proposal_ambiguous`; the next returned
`axis_estimated_current_measured_head`, with both `head_frame_detected` and
`yaw_reliable` true in the new diagnostics.

On that accepted image:

| Measurement | Recorded value |
| --- | ---: |
| Candidate camera depth | 0.4970 m |
| Expected projected side height | 100.21 px |
| Minimum measured edge length | 95.90 px |
| Raw border support mean | 1.00 |
| Pose reprojection RMSE | 0.531 px |
| Pose ambiguity gap | 1.828 px |
| Estimated yaw standard deviation | 1.518 degrees |
| Final observation age | 0.279 s; accepted as fresh |
| Geometry processing time | 65.5 ms |
| Geometry plus identity work | 166.1 ms |

The LiDAR search region was `[187,171,620,401]`. It retained 3,837 of 13,762
edge pixels for contour discovery and rejected 187 of 309 line segments before
line quotas. Metric size filtering rejected 13 contours. Final fitting retained
original raw evidence. This verifies the filtering was active and a grey head
was accepted; it is not an ablation proving which filter caused that success,
and two processed frames do not establish a reliability rate.

## Other observed limitations

The first candidate needed three processed images: one acquisition deadline
at `cold_texture_topology`, one raw-verification-budget failure, then one
accepted head/QR result. The second needed one ambiguous proposal followed by
its accepted result. These acquisition failures remain optimization targets,
but neither caused the terminal mission failure.

There were four `tf_retry_exhausted` events across the two observations. Later
frames recovered and all five processed detector results were marked fresh.
These transient transform failures therefore do not explain the fatal stop.

The grey size interval is conservative: approximately **42–233 px** around
100 px nominal, with 0.08 m configured depth uncertainty and the tilted-camera
projection bound. The region gives useful spatial rejection, but the size
prior still admits a broad scale range. Tightening that bound from calibrated
uncertainty could be investigated after repairing the execution path; this run
does not justify loosening pose or freshness gates.

The QUIRC warning in the terminal was not the terminal cause: QR_003 was
successfully decoded and committed by the available decoder path.

All captures were saved: seven frames for the red candidate and five for the
grey candidate, with no drops or write failures. Capture counts include sensor
or transform failures; they are not the number of completed detector passes.
The enlarged 256-frame/128-MiB limits were configured and were not exhausted.

## Verification gap and next correction

The targeted existing centering tests produced **19 passes, two failures, and
62 passing subtests** on the local OpenCV environment. The failing tests are:

- `test_missing_optional_odom_does_not_suppress_current_qr_completion`
- `test_real_processing_stages_center_after_current_crop_and_stopped_frame`

Their observed integration result was `head_proposal_unavailable`. They need
separate fixture/integration investigation; they do not reproduce the missing
child module and should not be presented as the cause of this physical run.
No test currently references the two missing child-adapter symbols. The prior
camera-focused regression checks and CLI validation therefore left this
execution branch uncovered.

Before another physical experiment:

1. Implement and connect the missing centering child adapter using the existing
   permit, fresh-target, bounded-turn, stop, and result contracts. The existing
   waypoint-follower centering mixin is a lower-level component, not a drop-in
   replacement for the missing child runner.
2. Exercise the real parent-to-child centering boundary in a no-motion test,
   including turn completion and failure outcomes. Add startup dependency
   validation so this missing integration cannot remain hidden until a mission
   has already driven to a stand.
3. Resolve the two observer integration-test failures and preserve a structured
   failure artifact if an execution adapter fails, instead of leaving only a
   traceback and a reserved turn.
4. Then continue with synchronized light-grey **frontside** evidence. The new
   run supplies a successful backside observation, not that missing frontside
   validation.

## Implemented correction (local checkout, 2026-09-18)

The missing centering adapter and internal child entrypoint are now implemented.
The parent validates these dependencies before a camera mission starts. The
child binds each turn to the existing mission authorization, current advisory,
robot configuration, and previous turn result. A persistent exclusive claim
prevents replaying the same turn, including through a different output path.
Execution uses the existing follower's bounded rotation, fresh scan/odometry,
target association, translation limit, and confirmed stationary-stop checks.
Exceptions trigger repeated zero commands before the ROS node is destroyed.

A second integration defect was corrected: centering progress previously tried
to overwrite an immutable content-hashed artifact after a turn. Progress now
uses immutable numbered revisions plus an atomically replaced latest snapshot.
Failed turns leave a structured `turn_failed` record and cannot reset the budget.

The two observer integration fixtures now exercise the physical measured-head
path used by the live observer. Their legacy synthetic scenario prefix had
selected a different acquisition path; both tests now pass.

Offline verification used the exact hashed advisory from this recording
(5.38236 degrees), with controlled time and simulated motion. Tests cover the
actual parent-to-child adapter call, permit/result binding, replay rejection,
stale evidence, failed stops, cumulative two-turn travel, fresh re-observation,
progress persistence, controller convergence, translation drift, and exception
cleanup. The child-adapter suite passed all 21 tests; the controller suite passed
all eight tests. These are software tests, not evidence of a physical turn.

The broader mission, observer, centering, and perception selection completed
with **555 passed, 480 passing subtests, and one failure**. The failing test,
`test_runtime_localization_recovery_uses_scoped_permit_without_parent_input`,
also fails in a separate export of unchanged `HEAD`: its fixture lacks
`odom_execution_certificate_sha256`. This pre-existing localization-fixture
failure is outside the centering correction. Compilation, entrypoint help, and
whitespace validation passed.

No workstation deployment or robot motion was performed for this correction.
The existing operator command needs no additional centering flags, but the
workstation checkout must include the new modules before another run. Physical
centering and light-grey frontside detection remain to be validated on the robot.
