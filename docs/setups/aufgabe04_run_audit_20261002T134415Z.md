# Audit: mission stopped after the second camera admission

## Outcome

Run `stand_explore_exact2_camera_all5_20261002T134415Z` on workstation alias
`mii002` used clean commit `f7a6f15`. Both visited candidates were admitted:

- `survey_candidate_0003`: `QR_003`, facing-ready.
- `survey_candidate_0001`: `Start`, facing-ready after the opposite-side
  observation was combined with retained head geometry.

The mission then failed during selection of the next candidate. It finished
at **15:52:22 Europe/Berlin** with exit code **2** and persisted
`mission_failure.json`, phase `candidate_qr_goal_incomplete`. The recorded
reason was: `candidate approach incomplete: bounded candidate pool yielded
2 unambiguous validated QR identities; expected 5`.

This is distinct from the earlier 15:24 run that the operator stopped. The
latest run has an explicit terminal failure and a completed bundle manifest.

## Sequence

All times are Europe/Berlin on 2026-10-02.

| Time | Event |
| --- | --- |
| 15:49:09.941 | First candidate persisted as resolved, `QR_003`. |
| 15:50:19.925 | Second candidate's first camera observation returned an axis receipt. |
| 15:50:39.501 | Initial 0.50 m opposite-side route failed uncertainty admission, with -85.7 mm remaining margin. No motion on that rejected route. |
| 15:51:04.202–15:52:04.042 | Existing fallback executed the admitted 0.45 m opposite-side route successfully. |
| 15:52:10.581–15:52:13.219 | Opposite camera observation decoded `Start`. |
| 15:52:15.041 | Retained facing recommendation persisted; two candidates facing-ready. |
| 15:52:21.589 | One new eight-scan cohort rejected every remaining candidate as a current LiDAR target. |
| 15:52:21.605 | `camera_candidates_current_lidar_unavailable` emitted; mission declared incomplete. |

The earlier route-uncertainty rejection recovered successfully. It was not
the terminal failure, and neither admitted candidate was lost afterward.

## What rejected the remaining candidates

Final target assessment `current_lidar_targets/selection_002` produced zero
accepted targets. All five candidates remained unvisited and were marked
`target_reconciliation_required`. Mount/head-plane observability passed for
all of them; this was a scan-cluster support failure.

| Candidate suffix | Range from scanner, final scan | Final scan reasons across eight scans | Earlier evidence in this run |
| --- | ---: | --- | --- |
| 0002 | 1.44 m | `unsupported` 8/8 | No accepted current support in earlier selections. |
| 0004 | 3.47 m | `non_stand_cluster` 8/8 | Earlier competing/unsupported/non-stand results. |
| 0005 | 3.00 m | `unsupported` 8/8 | Accepted 8/8 in selection 000; 5/8 support in selection 001. |
| 0006 | 3.05 m | `unsupported` 8/8 | Accepted 8/8 in both earlier selections. |
| 0007 | 3.42 m | `non_stand_cluster` 8/8 | Earlier ambiguous/unstable correspondence. |

The saved assessment was replayed from its original hashed snapshot, capture
and scan receipts on the workstation. The unmodified production loader
reproduced the empty accepted-target set.

## Why this is an approach-selection problem

The current target detector requires at least two points in a compact whole
cluster, with an adjacent-point gap at most **80 mm**, and a maximum total
width of **86.399 mm**. See `current_lidar_targets.py:96–113`.

The final cohort's angular increments were **1.690–1.704 degrees**. At the
four distant candidates, same-range adjacent beam spacing was **88.6–103.3
mm**, already exceeding the cluster gap limit. A 78 mm head's broad face at
those ranges subtends only **1.29–1.49 degrees**, less than one beam interval.
It can therefore produce only a singleton, or be missed between beams while
background surfaces still return valid ranges. These are inadequate samples
for validating the small head; they do not by themselves prove that the
candidate is invalid.

The final raw scan confirms the classifier's mechanics:

- 0004 and 0007 had nearby singleton returns, recorded as zero-width
  non-stand clusters. Their nearest raw points were about 62 mm and 40 mm
  from the respective expected centers.
- 0005 and 0006 had no nearby target returns; rays through their candidate
  cones hit farther surfaces. They remained `unsupported` rather than
  `insufficient_visible_returns`, because the cone contained background
  measurements.
- 0002 is a separate unresolved case: at 1.44 m, its broad face would span
  about 3.11 degrees, but the nearest raw return was about 323 mm from its
  expected center. The far-distance explanation alone does not settle this
  candidate's position, existence, or possible edge-on view.

The survey observation fallback in `permits_survey_observation`
(`current_lidar_targets.py:312–325`) permits only `supported`, `occluded`, and
`insufficient_visible_returns`. It refuses `unsupported` and
`non_stand_cluster`, including the under-resolved cases above. Thus it never
made any final candidate eligible for an observation approach.

`approach.py:2706–2712` captures one cohort and immediately raises
`NoCurrentLidarTargetsError` when the combined current/fallback pool is empty.
The handler at `approach.py:2924–2934` turns that into mission failure. There
is no additional cohort, alternative observation position, or third-candidate
route search on this branch. Existing route and camera retries do not handle
this failure.

The central defect is that temporary inability to resolve a small head from
the current distant pose is treated as exhaustion of the remaining mission.
The earlier 8/8 support for 0005 and 0006 makes that distinction concrete.

### Nearest remaining candidate

After `Start` was admitted, 0002 was the nearest remaining candidate: **1.418 m
from the robot base** (1.438 m from the scanner). It was already in the survey
snapshot with 70 historical observations and source kind
`lidar/exact_two_single_view_requires_camera_validation`. It was not skipped
in favor of a farther candidate: the fresh-support filter removed it before
route ranking could run. The final selection never called the route selector
because every candidate failed that preceding filter.

For 0002, all eight scans were `unsupported`; the last scan's nearest return
was 323 mm from its stored center, beyond the 160 mm association bound.
Background returns were present, so the current classifier supplied neither
`occluded` nor `insufficient_visible_returns`, the labels that would enable
the survey observation fallback. The range-resolution explanation for the
other four candidates does not establish why this nearest candidate lacked
returns. Its historical position or current head visibility needs separate
reconciliation. The current policy prevents that camera observation approach
despite retaining the candidate in the unvisited survey pool.

## Correction direction

1. Distinguish insufficient angular sampling from positive contradictory
   geometry. A far singleton or background-only scan must not automatically
   classify the historical target as a non-stand.
2. Preserve source-bound, previously supported candidates as hypotheses for
   an observation approach or reacquisition from a useful viewpoint. Retained
   evidence must not acquire fresh precision-alignment or identity authority.
3. Add bounded recovery for an empty current target pool before declaring the
   whole mission exhausted. Fresh scan retries help transient misses; repeating
   the same distant pose alone cannot solve the demonstrated resolution limit.
4. Preserve wall/large-cluster and competing-candidate protections, keepouts,
   collision checks, localization checks, and route uncertainty admission.
   Do not merely allow every `unsupported` candidate or increase the 160 mm
   precision-association limit.

Regression coverage should include previously supported targets becoming
under-resolved after an opposite-side move, real angular increments around
1.7 degrees at 3–3.5 m, singleton versus broad-wall distinctions, and an empty
selection cohort with unvisited candidates. Completion of all five identities
remains unverified.

## Evidence and scope

All raw recordings and logs remain on the workstation. Read-only replay used
the existing container with a read-only repository mount; no ROS node or robot
motion was started. No production code changed.

Primary evidence, relative to the workstation repository:

- Run root: `results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20261002T134415Z/`.
- `mission_failure.json`, `candidate_goal_progress.json`, `candidate_selection.jsonl`.
- `current_lidar_targets/selection_000`, `selection_001`, and `selection_002`, including source capture/receipts and projected snapshots.
- Both visited candidates' `inspection_progress.json`, `inspection_handoff_events.jsonl`, and the second candidate's `retained_facing_recommendation.json`.
- Opposite-side child run events and fallback event in `candidate_selection.jsonl`.
- `results/real_runs/stand_explore_exact2_camera_all5_20261002T134415Z/manifest.txt` and `terminal_run.log`.
