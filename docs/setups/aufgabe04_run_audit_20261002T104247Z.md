# Latest-run audit: candidate 0001 opposite-side branch

Run: `stand_explore_exact2_camera_all5_20261002T104247Z`.
Workstation: SSH alias `mii002`, hostname `mii001`.
Recorded interval: **2026-10-02 12:42:47–12:51:20 Europe/Berlin**
(10:42:47–10:51:20 UTC); parent exit code **2**.
All seven parent/child bundles record clean revision
`4bed40c895dfbaa4a53238014e163f78e7d51665`, matching the inspected source.

## Finding

**Candidate `survey_candidate_0001` never entered the opposite-side branch.**
Its approach completed, but the fresh LiDAR target check at arrival rejected
one ambiguous scan before starting the camera observer. The candidate was
marked `target_reconciliation_required`, with `retry_eligible=false`.
Consequently, there was no camera orientation receipt from which to request
an opposite-side route.

The immediate cause was a single missing LiDAR return splitting nearby
returns into two independently plausible clusters, combined with an
any-ambiguity veto across the complete eight-scan cohort. The application then
deferred the candidate without a fresh stationary retry. This differs from the
08:41 UTC run, where an opposite route actually moved and later failed
localization/clearance admission.

The latest run ended later on **candidate `survey_candidate_0002`**, during
startup recovery. That terminal error is separate from candidate 0001's
earlier rejection.

## Candidate 0001 timeline

Times below are Europe/Berlin on October 2, 2026.

| Time | Recorded event |
| --- | --- |
| 12:48:56.916 | Selection-stage LiDAR accepts candidate 0001: 7/8 supported scans; one non-stand cluster |
| 12:49:05.453 | Candidate 0001 selected for the next approach |
| 12:49:13.163 | Approach dry-run passes |
| 12:49:55.104 | Approach completes after 26.418 s, estimated travel 0.897596 m |
| 12:49:55.977 | Arrival admission starts |
| 12:50:01.457–12:50:02.126 | Fresh eight-scan arrival cohort captured |
| 12:50:02.252 | Current target rejected: `ambiguous_current_target_correspondence` |
| 12:50:02.266 | Arrival admission fails; observer not started; candidate deferred without retry |
| 12:50:08.687 | Next selection assesses the remaining candidates; candidate 0001 is absent from its assessed UID list |

The full run contains zero opposite-side, camera-observer, axis-observation,
inspection-observation, or recommendation artifacts. Both approached
candidates, 0003 and 0001, failed arrival target admission. The final progress
records **zero of five required QR identities**, with seven candidates retained
as keepouts.

## Why 7/8 support still failed

The arrival evidence for candidate 0001 records:

| Quantity | Recorded value | Relevant condition |
| --- | --- | --- |
| Supported scans | 7/8 = 87.5% | At least 75% required |
| Ambiguous scans | 1/8 | Any ambiguity vetoes the estimate |
| Supported-center scatter | 10.131 mm | Maximum 30 mm |
| Displacement from retained candidate anchor | 79.447 mm | Maximum 160 mm |
| Maximum plausible cluster width | 86.399 mm | Derived from measured head dimensions and tolerance |
| LiDAR head-plane check | Accepted | Exact-TF laser plane at approximately 182 mm, within measured head interior |

The recorded decision has exactly one rejection reason:
`ambiguous_current_target_correspondence`. Support fraction, center scatter,
displacement and head-plane observability were not the rejecting quantities.
The mount evidence reports `physical_mount_calibration_verified=false`; the
accepted head-plane check is the implementation's exact-TF check, not an
independent physical calibration claim.

Read-only reconstruction of scan `1790938201.9345844` found:

| Cluster | Original beam indices | Points | Diameter | Distance to the other seven scans' mean center |
| --- | --- | --- | --- | --- |
| A | 2, 3 | 2 | 16.48 mm | 20.36 mm |
| B | 5, 6 | 2 | 16.21 mm | 27.83 mm |

Beam **4 is null**. The clustering function requires consecutive source
indices as well as spatial proximity, so the missing beam separates A and B.
Their nearest endpoints are only **31.91 mm** apart, below the configured
80 mm spatial gap. The combined four-point diameter is **64.31 mm**, below
the 86.399 mm plausible-head limit. Nevertheless, each two-point fragment
individually passes the 10 mm minimum width and two-point minimum, producing
two plausible clusters and the ambiguity veto.

This is an interior missing-beam split, not a scan-seam wrap error. The numbers
support a transient fragmented-surface explanation; they do not independently
prove that both fragments are one physical stand. Accepting either fragment
or merging them requires an explicit, tested correspondence policy.

## A later recorded stationary cohort resolves the ambiguity

The next selection captured another complete cohort at
12:50:07.941–12:50:08.611, approximately **6.48 seconds later**. No motion
occurred between the two captures. Runtime selection 002 assessed candidates
0002 and 0004–0007; candidate 0001 had already been removed from the
unresolved set.

An offline counterfactual reassessed candidate 0001 using this later cohort's
own projected snapshot, planning frame and mount evidence, with unchanged
thresholds. The current-target gate **accepted** it:

| Quantity | Later-cohort result |
| --- | --- |
| Supported scans | 8/8 |
| Ambiguous scans | 0 |
| Center scatter | 17.498 mm |
| Displacement from candidate anchor | 81.058 mm |
| Target uncertainty | 97.498 mm |
| Rejection reasons | None |

This strengthens the case for bounded stationary reacquisition: a subsequent
recorded batch would have cleared this particular target gate. It is not a
runtime acceptance, a camera result, or evidence that opposite routing would
have passed its independent checks.

## Exact code path

Source locations refer to the recorded revision above.

1. `real_robot/candidate/inspection_adapters.py:607` performs initial arrival
   admission before entering the camera inspection loop.
2. `real_robot/candidate/approach.py:1422` requires a fresh current LiDAR
   target at that admission.
3. `perception/lidar_stand_detector.py:69` splits nonconsecutive beam indices.
   `real_robot/candidate/current_lidar_targets.py:159` labels multiple
   plausible clusters ambiguous; line 181 vetoes the full cohort if any scan
   is ambiguous or competes with another candidate. The 75% support check is
   separate at line 183.
4. `real_robot/candidate/approach.py:1536` raises
   `CandidateObservationUnavailableError(candidate_target_ineligible)` with
   `observer_started=false` when no current target estimate is available.
5. `real_robot/candidate/approach.py:2984` handles that error as a candidate
   deferral. Lines 2993–3009 disable retry for this reason, record
   `target_reconciliation_required`, and remove the candidate from the current
   unresolved set while retaining its keepout and incomplete-goal status.
6. `real_robot/candidate/inspection_execution.py:152` only calls
   `move_opposite` after an observation supplies `axis_observation_path`.
   Arrival admission failed before that prerequisite could be produced.

Opposite-route standoff search, checkpoint recovery, clearance budgets and
opposite runtime recovery were never reached for candidate 0001. The newly
clarified 3.70 m wall length is not the recorded rejecting condition here.

## Separate whole-run termination on candidate 0002

Candidate 0002's first approach attempt and its first startup replacement both
passed preflight, then stopped **before publishing motion** because the
follower could not acquire `map <- odom`. The stops occurred at 12:50:38.599
and 12:51:12.277. Each capture reports 25 unsuccessful global-transform
lookups across about five seconds while `odom <- base_footprint` was available.
The evidence establishes a missing transform in those follower buffers; it
does not by itself establish a globally absent AMCL publisher or a DDS cause.

During the second startup recovery, fresh stationary localization was admitted
at 12:51:18.153. Its current-target check then had only **3/8 supported scans**
and five non-stand clusters. That failed the 75% support requirement. At
12:51:19.541, the recovery recorded `same_routine_replan` failure and the
mission terminated.

The same type of target rejection has different scope in these two contexts:
the arrival path defers candidate 0001, whereas
`real_robot/candidate/startup_recovery.py:719` wraps the replan exception in
`CandidateStartupRecoveryError`. The candidate route deferral handler at
`real_robot/candidate/route_admission_deferral.py:365` only admits
`outcome_rejection`, so this `same_routine_replan` failure escapes as terminal.
This is a separate recovery-classification issue, not evidence that candidate
0001 attempted an opposite maneuver.

## Correction targets

- Add bounded stationary reacquisition for transient current-target ambiguity
  before making the candidate unavailable for the remainder of the pass.
  Continue to require fresh, unique correspondence; do not override ambiguity
  solely because 7/8 scans have support.
- Evaluate a narrowly bounded missing-beam fragmentation policy using the
  saved scan. Preserve whole-cluster wall rejection, competing-candidate
  rejection, measured-head limits and scan topology. A blanket increase in
  association radius or acceptance of multiple clusters is not justified.
- Preserve typed candidate-local target rejection through a verified
  before-motion startup reseal, so that one unavailable candidate need not
  become a session-wide failure. Any continuation still needs the existing
  stopped-state, evidence, route and motion-permit checks.

These were the audit recommendations; the implementation follow-up below
records the subsequent local correction.

## Implementation follow-up

The single-target admission path now allows **three independent stationary
cohorts in total**, with a **two-second passive pause** between attempts, when
the only rejection reason is `ambiguous_current_target_correspondence` and
the scan decisions contain `ambiguous_clusters` without a competing
candidate. This applies wherever the existing single-target admission helper
is used, including arrival and startup replanning. Multi-candidate selection
still uses its existing single capture.

Every attempt retains the original planning-frame binding and the existing
capture, stationary-pose, freshness, head-plane, support, scatter and
displacement checks. A later cohort must start after the preceding cohort's
last scan. Attempts are neither combined nor averaged; a later independently
accepted cohort is required. Persistent ambiguity or another rejection reason
still makes the candidate unavailable. No cluster merging or admission
threshold change was implemented.

The initial assessment keeps its original artifact path. Additional captures
are stored under `reacquire_001/` and `reacquire_002/`, with attempt paths and
hashes recorded in `reacquisition_events.jsonl`. The admitted evidence points
to the accepted cohort so that downstream replay validates that capture.
Reacquisition itself does not start the camera or authorize motion.

For a fresh `candidate_preapproach` routine whose children all stopped before
motion, startup recovery now preserves a typed current-target rejection.
Before candidate-local continuation, it verifies the exact routine identity
and closure of every issued permit. The parent records
`camera_candidate_startup_target_deferred`, retains the candidate's keepout,
marks its QR goal `target_reconciliation_required`, and selects the next
candidate through the existing fresh gates. Pilot mode still stops with an
incomplete goal. Runtime/opposite-side recovery, resumed routines, integrity
failures and other recovery exceptions remain terminal.

Local tests cover an ambiguous arrival followed by an accepted fresh cohort
reaching the opposite-side dispatch, persistent rejection before camera start,
artifact replay, candidate continuation and its excluded recovery contexts.
The combined LiDAR, candidate recovery, permit and opposite-side regression
suite passes **280 tests and 221 subtests**. All five changed production
modules also pass Python 3.10 syntax checks.
The correction has not been deployed or validated in a new robot run. The
recorded later cohort supports the retry design; it does not prove that the
next run's camera or opposite-side maneuver will succeed.

## Audit validation and evidence

The focused current-target, approach-integration and preflight suites pass:
**35 tests and 15 subtests**. An additional in-memory synthetic probe using
the existing test fixture reproduces the policy distinction: seven supported
scans plus one absent scan are accepted; seven supported scans plus one
ambiguous scan are rejected. This verifies current behavior, not a fix.

The original arrival assessment was replayed in place with its exact
`arrival_frame_projection/candidate_snapshot.json`, hash
`555c4ae0f8ce9aecbf07d2c9b8d82d07fd995c353b795e53f02b8785b4649e69`.
Assessment, view and capture content hashes, source paths, receipt and mount
bindings, and the reconstructed capture contract all validated. The replay
produced **zero differing assessment fields**. The mutable run-root snapshot
is a different projection and was not used as a substitute.

The saved selection 002 assessment also replayed with zero differing fields
using `candidate_frame_projections/selection_002/candidate_snapshot.json`,
hash `43cf60d24ba98dcc3a74f70135eb41de96f5b98dbe3bbf7c278b578386352e36`.
Only the counterfactual reassessment added candidate 0001 to the assessed UID
list; all sensor inputs, physical limits and algorithm thresholds were retained.

Raw run files, scans and recordings remain on the workstation. No ROS nodes
or robot motion were started during this audit. At the audit stage, only this
report was added; production code and recorded evidence were not edited.

The remote repository root is
`/home/amr_gruppe01/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr`.
Primary sources beneath that root are:

- `results/real_runs/stand_explore_exact2_camera_all5_20261002T104247Z*/`
  manifests, revision and status records;
- `results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20261002T104247Z/`
  containing `station_segment_runs.csv`, `candidate_selection.jsonl`,
  `candidate_goal_progress.json`, `adaptive_replans.jsonl`,
  `mission_failure.json`, and child `run_events`;
- `candidates/001_survey_candidate_0001/inspection_handoff_events.jsonl`
  and `current_lidar_target/{current_lidar_targets.json,capture/scan_cohort.json,capture/candidate_lidar_view.json}`
  within that session;
- candidate 0002's `candidate_preapproach_startup_reseals/startup_reseal_002/route_source/current_lidar_target/`
  within the same session.
