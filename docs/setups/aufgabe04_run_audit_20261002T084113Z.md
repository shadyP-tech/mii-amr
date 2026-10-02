# Latest-run audit: October 2, 2026, 10:41 Berlin

Run: `stand_explore_exact2_camera_all5_20261002T084113Z`.
Workstation: SSH alias `mii002`, hostname `mii001`.
All six recorded run bundles used clean revision
`8f566e14e2ec088508035fa33662c5c306f23faa`.

## Finding

The run ended with **one of five identities confirmed and facing-ready**.
It failed during navigation to the opposite side of `survey_candidate_0001`.
The opposite-side camera never started.

The immediate motion stop was an anchored map-to-odometry yaw-drift limit.
Recovery obtained fresh localization and planned a replacement, but its
uncertainty-adjusted clearance was negative. That structured rejection then
ended the mission. The camera ID-only change was active and worked in the
branches this run reached.

## Camera behavior before the failure

- Candidate `survey_candidate_0003`: two processed camera frames; the second
  committed a recommendation. Its measured-head crop was accepted, decoding
  was attempted, and one ID was obtained. QR marker geometry remained unknown
  with reason `identity_disabled_for_geometry_only`.
- Candidate `survey_candidate_0001`: 12 processed camera frames. Seven valid
  bounded observations produced a schema-5 orientation with
  `visible_face="unidentified"`, sample source
  `current_measured_head_unidentified`, and half-width **5.521°**. The accepted
  head crop was decoded without obtaining an ID.
- That orientation correctly enabled planning an opposite-side inspection.
  Only the initial camera observer exists for this candidate; navigation
  stopped before a post-opposite capture could be started.

This is the expected distinction between an unidentified side and a confirmed
backside. It is not a rejection caused by missing QR corners, nor the previous
run's tiny camera-centering turn timeout.

## Navigation and recovery sequence

Four motion legs completed: coverage 000/001 and candidate approaches 000/001.
The opposite-side leg then stopped after **40.0 seconds** and approximately
**1.113 m** of estimated travel.

Its stop reason was `map_from_odom_yaw_drift`:

| Continuity quantity | Observed | Permitted |
|---|---:|---:|
| Anchored yaw change | 3.232° | 3.120° |
| Anchored translation change | 25.70 mm | 112.40 mm |

The yaw exceeded its limit by approximately **0.112°**. The translation bound
was not exceeded. This monitors changes to the localization transform while
executing a frozen route; it is independent of the 30° camera-facing allowance.

Recovery events, Europe/Berlin:

- 10:50:02.660: stopped-leg handoff to runtime localization recovery.
- 10:50:07.726: fresh localization admitted.
- 10:50:09.847: replacement route replanned.
- 10:50:15.726: replacement preflight rejected before motion.

The replacement run ends in
`candidate_001_inspection_001_localization_001_opposite_standoff_001_runtime_localization_reseal_001`.
The parent recorded `failed_closed`; no replacement motion was authorized.

## Why the replacement failed its budget

Route admission requires a strictly positive clearance remainder:

`raw clearance − radius − collision margin − tracking bound − drift bound − braking allowance − position uncertainty − heading contribution`.

For the rejected replacement's limiting sampled subsegment `segment:0000:0064`:

| Component | Millimetres |
|---|---:|
| Conservative available centerline clearance | 372.508 |
| Robot radius | 105.000 |
| Collision margin | 20.000 |
| Tracking bound | 30.000 |
| Odometry drift bound | 20.000 |
| Braking allowance | 15.000 |
| Position uncertainty | 129.376 |
| Heading contribution | 56.892 |
| Total required clearance | 376.268 |
| Remaining margin | **−3.759** |

Two of the replacement's 65 sampled subsegments were rejected. The limiting
identifier is a sampled route subsegment, not the 65th navigation waypoint.
These are computed admission margins, not evidence of physical contact.

The earlier route had passed execution preflight with a **+61.439 mm** minimum
margin. Its limiting segment had 422.618 mm raw clearance, 112.404 mm position
uncertainty, and 58.775 mm heading contribution. The replacement is a different
route from the stopped pose, so these are different limiting locations, not a
same-point clearance measurement. Its geometry and uncertainty both differ.

An earlier startup localization refresh had already recovered from an initial
opposite-route margin of −10.055 mm: the refreshed dry route passed at
+49.969 mm. Thus localization recovery did run; the terminal problem is what
happens after its post-motion replacement fails admission.

## Recovery-policy limitation

The opposite-route fallback in
`scripts/aufgabe04/real_robot/candidate/approach.py:2154` catches
`CandidateStartupRecoveryError` and can try another standoff before initial
motion. The runtime replacement rejection instead raises
`CandidateRuntimeRecoveryError` from `runtime_recovery.py:479`.
It escapes that fallback and ends the mission. The checkpoint fallback is also
reached only after exhausting initial route attempts, not from this runtime
replacement rejection.

The safety rejection is consistent with the recorded numbers. The useful next
correction is bounded recovery after a validated motion stop: use the fresh
localization frame to evaluate another standoff or route/checkpoint, with all
normal candidate, clearance, uncertainty and motion-authority checks. Reusing
the rejected route or simply ignoring the small negative margin is not supported
by this evidence. No such correction was implemented during this audit.

Relevant source:

- `navigation/execution/route_uncertainty_budget.py:338`: clearance deductions.
- `navigation/execution/route_uncertainty_admission.py:471`: heading lever arm.
- `navigation/localization/amcl_covariance_envelope.py:70`: conservative covariance envelope.
- `real_robot/candidate/runtime_recovery.py:479`: terminal replacement-preflight handling.
- `real_robot/candidate/approach.py:2154`: startup-only alternate-standoff fallback.

Read-only source verification ran 41 tests and 63 subtests successfully, including
the test that deliberately expects terminal failure for a negative-budget runtime
replacement. Two historical map-dependent tests were excluded. This confirms
existing policy behavior; it does not validate a new recovery implementation.

## Evidence handling

At the user's request, raw logs and recordings remained on the workstation.
Analysis used remote aggregate counts, rounded numeric budget/continuity
summaries, saved revision checks and local source inspection. No raw run bundle
was copied, no production code changed, and no ROS nodes or robot motion were
started by this audit.

The workstation run root is under
`results/aufgabe04/real/autonomous_exploration/<run>`.
Sources include `mission_failure.json`, `candidate_goal_progress.json`,
`station_segment_runs.csv`, `adaptive_replans.jsonl`, camera observer events and
status files, the schema-5 axis receipt, and
`odom_execution/*opposite_standoff_001*uncertainty_budget.json`.
