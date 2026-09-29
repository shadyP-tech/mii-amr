# Run audit: 20260929T124849Z

The latest run stopped because a fixed **45-second waypoint deadline expired
during steady progress** on the third candidate approach. The robot spent about
12.6 seconds aligning with a 1.725 m segment, then about 32.3 seconds driving.
It stopped 12.54 cm from the target, outside the 3 cm arrival tolerance.

This is an existing mismatch between planned route duration and the execution
deadline. The previous backside-validator fix was deployed, but its affected
branch was not exercised in this run. The correction implemented after this
audit is described below; it has been tested locally and is not deployed.

## Run identity and mission progress

- Session: `stand_explore_exact2_camera_all5_20260929T124849Z`.
- Recorded clean revision: `de083b8d6bd2a9d040d96341ceeb09ca1af62d4d` on `main`.
- Workstation read through alias `mii002`, reporting hostname `mii001`.
- Run began September 29 at 14:48:49 CEST / 12:48:49 UTC. Final child stop was
  recorded at 14:56:05.943 CEST / 12:56:05.943 UTC.
- Both coverage motions completed, followed by two successful candidate
  preapproaches and QR observations.

| Visit | Candidate UID | Outcome |
| --- | --- | --- |
| 1 | `survey_candidate_0006` | `QR_004`, discovery-only observation |
| 2 | `survey_candidate_0005` | `QR_001`, discovery-only observation |
| 3 | `survey_candidate_0001` | Preapproach timeout; camera observer never started |

Final progress was **2/5 identities and zero facing-ready stands**. Both saved
QR receipts explicitly record `completion_scope=discovery_only` and
`facing_ready=false`. The current `candidate_goal_progress.json` is the source
for these counts; `coverage/survey_summary.json` retains an earlier LiDAR
handoff snapshot and is not the final camera-progress record.

## What actually stopped the robot

Failed child: `stand_explore_exact2_camera_all5_20260929T124849Z_candidate_002`.
Both dry and execution preflight passed. The route was admitted, its one-use
motion permit was consumed, and motion was published.

The controller trace contains 449 ordinary control cycles:

| Phase | Evidence | Result |
| --- | --- | --- |
| Initial alignment | 126 cycles; approximately 12.6 s | Zero translation; one turn of about 119.7 degrees |
| Translation | 323 cycles; approximately 32.3 s through the stop tick | Remaining distance fell from 1.716 m to 0.125 m |
| Termination | Waypoint elapsed 45.002241 s | `waypoint timeout`, still in `path_tracking` |

The last recorded control sample has 0.130329 m remaining. The subsequent
stop-state sample has 0.125394 m remaining. These are successive measurements,
not conflicting outcomes. The child duration of 46.368218 s includes startup;
the 45-second waypoint timer is separate.

Evidence against a stall, obstacle veto, or route-tracking failure:

- Distance never increases during the recorded translation phase: 321 of 322
  successive comparisons decrease, and one is equal.
- Mean commanded linear speed was 0.054420 m/s against a 0.055 m/s maximum.
  Progress over the final approximately ten seconds was 0.049925 m/s.
- Front clearance stayed at or above **0.685 m**, well above the 0.20 m stop
  threshold and 0.38 m slowdown threshold. No front-obstacle throttling occurs.
- Maximum lateral deviation was **0.015714 m** inside the **0.03 m** route tube.
- Every control cycle has an empty failure reason and `fail_closed=false`.
  The final lifecycle timeout then sets the terminal failure.
- Cycle spacing remained approximately 0.1 s; there is no trace evidence of
  scheduling starvation or a localization/sensor interruption causing the stop.

The saved selected control poses are expressed in the admitted odometry
execution frame, despite the trace's generic `map_pose` field name. Route CSV
endpoints are in map coordinates. They must be transformed through the saved
execution certificate before comparing coordinates directly.

## Why the deadline did not fit the leg

The planner reduced a 35-point, 1.875 m path to two executable points and a
1.725130 m segment. The follower pursued target index 1 for the entire leg.
There was no intermediate target change to begin a new waypoint budget.

The selected candidate already had a nominal duration estimate of
**43.591601 seconds**. The ranking calculation is simply:

`route length / linear speed + turn burden / angular speed`

Here, `1.725129519 / 0.055 + 2.200609692 / 0.18 = 43.591601` seconds.
Only **1.408399 seconds** remained below the execution deadline. That estimate
is a ranking score; it is not a deadline-feasibility gate and does not allocate
an execution margin for actual acceleration, convergence, or delivered speed.

`survey_candidate_0001` ranked ahead of a nearer candidate whose estimated
duration was 22.305 seconds because its multi-view LiDAR support had higher
priority. Choosing a geometrically feasible route did not establish that its
resolved controls could finish within the fixed deadline.

Relevant code paths at the recorded revision `de083b8`:

- `navigation/station_segment/cli.py:416` supplies the default 45-second timeout.
  `real_robot/execution/child_runner.py:211` supplies the 0.055 m/s and 0.18 rad/s
  speed limits, but does not override that waypoint timeout.
- `navigation/control/waypoint_controller.py:679` requires rotation without
  translation until the initial heading fits the three-degree alignment limit.
- `waypoint_follower/runtime_components/step_cycle_guard.py:277` starts/resets
  the waypoint timer on target changes; line 326 checks total elapsed time.
  Alignment and travel consume the same budget. Physical progress does not
  renew that absolute deadline.
- `navigation/approach/camera_candidate_selection.py:401` computes nominal
  duration; its ranking policy does not check the follower's time allowance.
- The separate 24-second terminal-heading budget applies only after reaching
  the final position before the waypoint deadline expires. The robot had not
  reached that phase and could not use that allowance.

At the stop, another 0.095394 m of translation was needed to enter the 3 cm
position tolerance. Even at maximum commanded speed this requires at least
**1.734 seconds** more. Actual completion would also depend on subsequent
motion, approach taper, heading, and live safety checks; the recording cannot
prove that a longer timeout would complete the mission.

## Why the final error mentions startup recovery

The primary failure is `waypoint timeout`. The outer candidate coordinator
then rejects startup recovery because the child had already published motion.
`candidate/startup_recovery.py:544` correctly refuses to treat this as a
motion-free startup failure. `candidate/recovery_failure.py:194` labels the
reported exception with the coordinator's name, producing
`failure_phase=candidate_startup_recovery`.

That label identifies the recovery rejection, not the execution stage where
the robot first failed. This run did not fail during startup. It also did not
produce the specific localization-stop evidence required for runtime resealing.
No opposite-face or localization-checkpoint motion was attempted.

## Relationship to the preceding fix

The parent and child bundles both record the deployed validator-fix revision.
Compared with the previous `20260929T121800Z` run, the parent command is
unchanged apart from session ID. The intervening commit changes the backside
validator, its regression test, and the earlier audit document; it changes no
navigation or motion configuration.

Neither observer in this run attempted backside receipt publication; both
committed QR-only poses. Therefore this run confirms deployment, not live
coverage of the corrected subset-proof branch.

The controller, waypoint deadline logic, candidate time scoring, and relevant
route smoothing are unchanged relative to `3ad9b00`. The new route and initial
heading exposed an existing duration-budget problem; there is no evidence that
the validator fix introduced this motion failure.

## Correction implemented locally

Real `detected_stand_preapproach` routes now share one finite segment-time
policy across candidate previews, child preflight, and the follower. It budgets
translation, initial alignment, and certified outgoing-corner alignment using
the resolved speed limits. Enabled command smoothing adds its acceleration
allowance. The resulting deadline is:

`max(45 s, 1.25 * (nominal motion time + acceleration allowance) + 5 s)`

The 25 percent and five-second margins are bounded engineering allowances,
not a claim that every admitted route will finish. Requirements above 120
seconds are rejected before motion. An explicit `--waypoint-timeout-sec` must
cover the computed requirement and remain within the cap. Omitted timeouts
select this automatic policy for real detected-stand approaches; other routes
retain their existing 45-second default.

The follower freezes the budget once for each admitted route revision and
target, using the actual execution pose. Progress does not renew the timer.
The eight-second no-progress watchdog, separate 24-second terminal-heading
deadline, speed limits, arrival tolerances, and live safety checks are unchanged.
Preflight and controller traces record the computed budget components.

Replaying all 449 recorded controller samples through the corrected lifecycle
guard produces `proceed` throughout. The actual first execution pose gives a
**59.963852-second** budget; the saved 45.002241-second stop now produces
`proceed`. A synthetic check just after the derived deadline still produces
`waypoint timeout`. This validates the decision boundary only, not subsequent
vehicle motion or mission completion.

Local validation: **304 tests and 139 subtests passed** across the time-budget,
candidate planning, station preflight, follower lifecycle, safety, route handoff,
odometry execution, event reporting, and module-boundary suites. New cases cover
the recorded failure, frozen deadlines, stalls, separate terminal-heading time,
over-cap routes, insufficient explicit limits, and pre-motion rejection.
The outer startup-recovery failure label remains a separate reporting issue.

## Evidence and reproduction

Local root: `results/implementation_checks/run_audit_20260929T124849Z/`.

- `audit_summary.json` derives the timing, progress, clearance, and ranking
  values from the saved artifacts.
- `analyze_controller_timeout.py` also passes the recorded stop state through
  the legacy fixed-timeout configuration. Its output exactly reproduces the
  stored `waypoint timeout` and stop details. This is a decision-boundary replay,
  not a replay of vehicle dynamics or unobserved future motion.
- `corrected_deadline_replay.json` records the corrected lifecycle replay,
  the single frozen budget, and the finite-deadline stop check.
- `verified_source_hashes.json` independently verifies 14 copied files against
  workstation SHA-256 hashes, including the full controller trace, run
  provenance, mission failure, ranking log, route, diagnostics, and certificate.
- Under the extracted mission directory: `mission_failure.json`,
  `candidate_goal_progress.json`, `station_segment_runs.csv`,
  `candidate_selection.jsonl:7`, and the final child `run_events` file lines
  13–16 establish resolved settings and termination.
- Under the final child's extracted real-run bundle:
  `controller_trace.jsonl:2` begins alignment, line 128 begins translation,
  and line 450 is the last ordinary control cycle.

Source artifacts and workstation code were read only. The correction changes
the local working tree. No ROS nodes, robot motion, or deployment were performed.
