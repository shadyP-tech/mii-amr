# Camera exploration candidate-admission review — October 2, 2026

Historical snapshot before `ef3ef71`. The arrival correction is now committed
and deployed. See [the follow-up simplification audit](aufgabe04_camera_exploration_simplification_audit_20261002.md)
for the updated baseline, current-run decoder blockage, and already implemented
opposite runtime recovery.

## Assessment

Several gates are too early or have too final a consequence. The clearest
demonstrated problem is requiring a fresh, unambiguous LiDAR target again
after a successful approach **before allowing passive camera observation**.
The recent runs support changing that ownership. They do not establish that
all rejected candidates are real stands, or justify removing association,
collision, localization, or motion-authority checks.

The useful simplification is four explicit decisions: eligible hypothesis,
permission to observe, permission to commit an identity/orientation, and
permission to move. Each should have one policy owner and an explicit
recoverable outcome. Revalidating evidence when loading it or executing a
later action is not by itself a redundant policy gate.

## Scope and versions

- Read-only SSH inspection of `mii002` (hostname `mii001`), repository
  `/home/amr_gruppe01/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr`.
- Inspected October 1–2 exploration outcome records, with detailed admission
  evidence for the two newest runs. Existing audit reports provide additional
  historical explanations; their older replays were not repeated here.
- Workstation checkout: clean `34bd90648e4b172ea9db9941e04e8d581596105d`.
  The latest run's parent bundle records that revision.
- Local HEAD matches. Local uncommitted changes already retain the completed
  route's target for arrival and passive frame refresh. Those changes were
  present before this review and are not in the workstation checkout.
- No robot processes were started or changed, no deployment occurred, and
  no raw run bundles or recordings were copied. This report is the only file
  added by the review.

## What the runs demonstrate

Times are Europe/Berlin. Candidate IDs are local to each run and must not be
equated across runs. These are different runs and revisions, not a controlled
before/after comparison.

| Run start | Revision | Recorded discovery | Relevant outcome |
| --- | --- | --- | --- |
| Oct 1, 12:46:18 (`20261001T104618Z`) | `fbe6926` | 5/5 QR IDs; 3 facing-ready, 2 QR-only | Discovery completed; subsequent return failed because the exact stored target was blocked. |
| Oct 1, 17:11:13 (`20261001T151113Z`) | `144f954` | 3/5, all facing-ready | A stopped camera-centering timeout became a mission failure. |
| Oct 2, 10:41:13 (`20261002T084113Z`) | `8f566e1` | 1/5, facing-ready | Opposite-side motion stopped on localization drift; replacement route failed its clearance budget. |
| Oct 2, 12:42:47 (`20261002T104247Z`) | `4bed40c` | 0/5; no camera observers | Both reached candidates failed fresh arrival LiDAR admission. A later startup-replan target rejection ended the mission. |
| Oct 2, 13:30:12 (`20261002T113012Z`) | `34bd906` | 0/5 recorded; no camera observers | First approach completed, but all three arrival cohorts failed. The recording has no parent exit/end marker or terminal mission result. |

For the newest run, the next candidate's child event stream ends at
`route_validated` at 13:35:26. No matching exploration/segment process was
found during inspection. This is an incomplete recording, not evidence of a
particular terminal gate failure or a currently running mission.

### Arrival admission prevents the camera from starting

In the 12:42 run, candidate 0001 had **7/8 supported scans**, 10.13 mm center
scatter and 79.45 mm displacement against a 160 mm limit. Its only rejection
was `ambiguous_current_target_correspondence`: one ambiguous scan vetoed the
whole batch. Candidate 0003 had only 1/8 supported scans and seven ambiguous
scans; that case also failed the support threshold.

The existing detailed audit documents the former as a missing-beam split and
shows a later stationary cohort passing unchanged thresholds. This review
rechecked the saved original support and rejection records, not that earlier
counterfactual replay.

In the newest 13:30 run, candidate 0003 passed selection with **8/8** supported
scans and completed its approach at 13:34:44. Its stopped arrival then produced:

| Cohort | Supported | Ambiguous | Center scatter | Displacement | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Initial | 6/8 | 2/8 | 17.17 mm | 15.05 mm | Ambiguity veto |
| Retry 1 | 6/8 | 2/8 | 10.11 mm | 17.85 mm | Ambiguity veto |
| Retry 2 | 5/8 | 3/8 | 14.08 mm | 10.93 mm | Ambiguity and insufficient support |

Arrival admission ran from 13:34:45.729 to 13:34:58.671, approximately
12.94 seconds. It ended with `retry_eligible=false` and no camera start.
Thus the latest deployed retry correction is active, but retries alone do
not resolve the policy problem. The evidence does not independently prove
that the ambiguous clusters are one physical stand.

### Selection is also sensitive to a single batch

In that same run, candidate 0005 had 8/8 supported scans but center scatter
31.17 mm against a 30 mm threshold, so it was excluded from that selection.
Candidate 0001 failed an earlier batch at 38.96 mm scatter, then passed the
next selection with 8/8 support and 23.42 mm scatter. This is direct evidence
that selection eligibility can change between batches. It does not prove
that the 30 mm bound should simply be increased.

The multi-candidate selection path currently performs one capture. If every
candidate fails, it raises an incomplete-goal error immediately. It does
not use the three-cohort single-target reacquisition policy.

## Prioritized code findings

Line references describe the inspected local working tree.

| Priority | Finding | Recommendation / current status |
| --- | --- | --- |
| 1 | Deployed arrival and passive observation paths require repeated current LiDAR admission before camera acquisition. | The existing local edits already retain and reproject the executed route's target, and request new support for translation separately. Validate and finish this direction before adding more retries. |
| 2 | One initial selection cohort can terminate the entire remaining camera goal. | Add bounded passive reacquisition and distinguish temporarily unobservable from exhausted. A later viewpoint requires an independently safe route; do not drive toward an unsupported precise target by default. |
| 3 | An arrival rejection can consume an inspection episode with zero camera views. | Track `not_yet_observed`, recoverable acquisition failure, and exhausted inspection separately. The eight-view budget is inside an episode, while the outer ledger allows one episode. |
| 4 | Any historical morphology conflict permanently vetoes target selection and passive handoff. | Provide explicit reconciliation by sufficiently strong current evidence while preserving history and keepouts. Keep the recorded wall-rejection controls. This was not the newest runs' binding gate. |
| 5 | QR-only completion skips angle consensus but the physical acquisition path still requires an associated measured-head crop before decoding. | Assess a second, independently candidate-bound identity acquisition path when head fitting fails. Decoding arbitrary search regions without exclusive association would lose the ID-to-stand guarantee. |
| 6 | Optional centering can become a session-wide failure. | After a verified stop and closed motion authority, permit a bounded passive observation or candidate deferral. Opposite-route replacement recovery is already implemented in `7aef892`; reuse its outcome distinctions rather than adding another retry layer. Keep systemic frame/integrity and unconfirmed-stop failures terminal. |

Specific code anchors:

- `real_robot/candidate/approach.py:1428`: local retained-target arrival policy;
  `:2591`: single initial capture; `:2805`: all-unavailable termination;
  `:2732`: one outer inspection episode; `:3097`: observation deferral handling.
- `real_robot/candidate/inspection_adapters.py:120`: passive frame refresh;
  `:517`: target gate on frame handoff; `:653`: arrival before inspection loop.
- `real_robot/candidate/current_lidar_targets.py:37`: 1 s latest-age,
  30 mm scatter and 75% support policy; `:59`: complete eight-scan cohort;
  `:181`: any-ambiguity veto, independent of the support fraction.
- `real_robot/candidate/current_lidar_reacquisition.py:44`: retry only when
  the sole reason is ambiguity from split clusters with no competing candidate.
  Sparse support, scatter failure, or ambiguity plus low support does not retry.
- `navigation/approach/candidate_target_admission.py:95`: historical
  morphology veto; `candidate_preapproach_selection.py:159`: fresh fitted-target
  rescue explicitly excludes that reason.
- `real_robot/observer/node.py:2029`: head geometry acquisition without QR
  identity; `:2117`: physical-profile head failure disables fallback decoding.
  `observer/current_head_identity.py:140`: decode follows association and
  complete crop admission.

The morphology veto has useful true-negative evidence in
`tests/aufgabe04/test_recorded_wall_candidate_admission.py`. It should be
reconciled, not blindly deleted. The README's description of these records
as advisories is incomplete relative to their current admission effect.

## What should remain separate

| Decision | Necessary evidence | Evidence that should not automatically veto this decision |
| --- | --- | --- |
| Retain a hypothesis for inspection | Bound source, plausible location, explicit uncertainty/conflict state | Missing QR, missing precise angle, one temporary visibility miss |
| Acquire a passive camera observation | Stopped pose, valid synchronized frame/target projection; subsequent image association determines usefulness | A newly fragmented LiDAR batch alone, when a valid retained observation target exists |
| Commit QR identity or orientation | Unique candidate association and decoded identity; orientation requires its own geometry evidence | Angle consensus for an identity-only result |
| Authorize movement | Fresh pose/obstacle evidence, target support appropriate to the maneuver, route clearance/uncertainty, exact motion authority | Historical permission from a previous route or observation |

The exact-two pool already accepts strict and boundary hypotheses, **5–10**
for this five-stand mission. Nominal-fit boundary candidates remain eligible.
Legacy strict exact-count coverage checks are not the active explanation for
these runs. The 5–10 population prerequisite is a workflow policy that could
eventually allow partial inspection/batching, but it was not the recent blocker.

Planning, child dry-run and live admission evaluate different timestamps and
actions. Their evidence binding, stopped-state verification, obstacle stop,
candidate/QR uniqueness and keepout preservation should remain.

The navigation budget is conservative, but this review establishes no double
counting that warrants deleting a term. Current fixed reserves total 75 mm
beyond the 105 mm robot radius, before position and heading uncertainty.
The 10:41 run used the older 20 mm collision margin and failed its replacement
by 3.759 mm. Reducing that margin to the current 10 mm is already committed;
it does not establish that the earlier runtime drift stop or alternative
routes would disappear. Direction-aware uncertainty and shorter route legs
are calibration/design questions, separate from passive camera admission.

There is also a hidden strict-backside threshold: a nominal 8° maximum
deviation combined with confidence >=0.60 yields an effective 3.2° limit.
The modern bounded-orientation path bypasses that confidence gate, as do
immediate-front and QR-only paths. It is a cleanup target, not an explanation
for the newest runs, which never started the observer. Likewise, the
seven-frame bounded-orientation requirement is not a universal QR gate.

## Verification and evidence locations

Focused tests against the local working tree:

- Current LiDAR, reacquisition, approach integration, retained target and
  passive recovery frames: **87 tests and 17 subtests passed**.
- Eight camera admission/geometry/identity modules: **103 tests passed**.
- Used the existing `/private/tmp/a04-id-only-venv` environment. No package
  installation or live sensor execution was required. Passing tests establish
  implementation behavior, not new robot-run success.

Workstation evidence is below the remote repository root:

- `results/aufgabe04/real/autonomous_exploration/<run>/candidate_goal_progress.json`
- `candidate_selection.jsonl`, `station_segment_runs.csv`, `mission_failure.json`
  when present, and `run_events/*.jsonl` within each session.
- `current_lidar_targets/selection_*/current_lidar_targets.json`.
- `candidates/*/current_lidar_target/`, including `reacquire_001`,
  `reacquire_002`, and `reacquisition_events.jsonl` for the newest run.
- `results/real_runs/<run>/{git_rev.txt,git_status.txt,manifest.txt,terminal_run.log}`.

Historical supporting reports: `aufgabe04_run_audit_20261001T104618Z.md`,
`aufgabe04_run_audit_20261001T141116Z.md`,
`aufgabe04_run_audit_20261001T151113Z.md`,
`aufgabe04_run_audit_20261002T084113Z.md`, and
`aufgabe04_run_audit_20261002T104247Z.md` in this directory.
