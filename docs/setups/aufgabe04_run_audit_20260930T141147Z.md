# Wall-candidate camera audit: 20260930T141147Z

Audited September 30, 2026 through SSH alias `mii002` (hostname `mii001`).
Latest recorded mission: `stand_explore_exact2_camera_all5_20260930T141147Z`,
started **16:11:47 Berlin time**. Parent and all copied child bundles record
`06b8bdad6e40b2905e4a4751277baeacb281b662`. Workstation access was read-only.

## Finding

The robot selected an unresolved wall-adjacent LiDAR hypothesis as its fourth
camera target. Its saved camera image shows the radiator and wall, with no
stand visible. Contradictory morphology from the second survey viewpoint was
already recorded, but remained advisory and did not prevent route selection.
Arrival then checked distance and bearing to the hypothesis, permitting a
passive camera search without a verified physical head.

**The latest head detector did not accept the wall as a head.** All 18 processed
frames failed `model_current_head_border_unavailable`; no QR, head axis or
centering turn was accepted. The regression in observed behavior is the visit
to a retained false hypothesis. The decisive difference from the preceding
run is that this time its approach route passed the uncertainty/clearance
check. Previous success had masked the unresolved candidate-admission problem.

There is also a geometry-validation gap: reprojection moved the target by
53.80 cm between its frozen survey location and arrival. The arrival target
lies in a blocked static-map cell, but retained its original boundary-candidate
eligibility. Route clearance concerns the robot's path; it does not establish
that a separate inspection target is a plausible stand.

## Why this hypothesis survived the survey

The target is `survey_candidate_0004`, visited as
`candidates/003_survey_candidate_0004`.

| Evidence | Recorded result |
| --- | --- |
| Frozen position | `(1.761380, 0.643481)` m |
| Accepted survey support | Only `survey_vp_001` |
| Hit count / confidence | 45 / 0.666098 |
| Original morphology | Median width 7.93 cm; accepted |
| Static-map clearance | 6.862 cm |
| Nominal / uncertainty-expanded required radius | 6.0 / 8.0 cm |
| Static-map disposition | `boundary_provisional` |
| Camera population | Six hypotheses for a five-distinct-QR goal; all retained |

The 45 observations came from 40 scan timestamps at the same viewpoint: 43
two-return clusters and two three-return clusters, at approximately
2.68–2.73 m. Adjacent-ray spacing was approximately 7.86 cm, close to the
8 cm clustering gap and the reported cluster width. Repeated sparse fragments
can therefore resemble a stable stand-width object. This supports a boundary
segmentation false-positive explanation; it is not an independently measured
physical identity for the original distant returns.

The second viewpoint supplied contrary evidence. Two nearby tracks were
rejected by morphology and associated spatially with candidate `0004`:

- `detected_stand_02`: median width 25.46 cm, only 3/84 width inliers.
- `detected_stand_10`: only 4/7 inliers and upper-quartile width 22.71 cm.

Both became unresolved `cross_view_morphology_conflict` advisories. The policy
explicitly sets `candidate_rejection_authorized=false`; those advisories do
not affect camera-route eligibility or ranking. Visibility evidence could not
reject the target either: no other eligible planned viewpoint, zero selected
visibility rays, and a 1.35 m visibility radius versus a 3.5 m proposal range.

Relevant code: `navigation/coverage/stand_candidate_static_map_admission.py:90`
retains boundary candidates; `coverage_morphology_conflict.py:132` attaches
conflicts and line 162 labels them advisory only. Ordinary boundary retention
is intentional so real near-wall stands can still receive an interior approach.
The missing distinction is an unresolved contradictory hypothesis versus an
ordinary boundary candidate.

## Exact comparison with the preceding run

A spatially matching wall-location hypothesis was present on September 29
at 13:50 and 15:23 UTC, and in the preceding September 30 13:52 UTC run. The
September 30 11:27 and 12:59 UTC snapshots had five candidates without this
wall-location hypothesis. Candidate numbers are run-local; this comparison
uses saved positions, not a cross-run identity claim.

At the fourth selection, **both the preceding and latest run initially ranked
the wall hypothesis first**. Production ranking replay reproduces both the
geometric ordering and the ordering after route-uncertainty filtering.

| Fourth-selection evidence | Previous `135242Z` | Latest `141147Z` |
| --- | ---: | ---: |
| Wall route minimum raw clearance | 37.25 cm | 42.91 cm |
| Required clearance | 39.97 cm | 40.29 cm |
| Remaining margin | **−2.72 cm** | **+2.61 cm** |
| Wall route admitted | No | Yes |
| Wall route estimated duration | 15.33 s | 13.91 s |
| Real candidate `0005` estimated duration | 15.41 s | 15.06 s |
| Selected candidate | `0005` | Wall hypothesis `0004` |

The wall route endpoint changed from `(1.455, 0.035)` to `(1.405, -0.065)` m.
The required clearance slightly increased; this was not a loosened uncertainty
threshold. Changed route geometry made the wall route admissible. Within the
same turn-risk and single-view support classes, its lower estimated duration
then won. In the earlier run the other five candidates yielded five QR
identities first, leaving the wall candidate `not_visited_goal_reached`.

`navigation/approach/camera_candidate_selection.py` is byte-identical to the
previous parent's revision `7221553`. The decisive saved decisions are line
10 of each run's `candidate_selection.jsonl`. The reproduction reuses recorded
route metrics and uncertainty decisions; it does not rerun route generation
or claim a new live motion certificate.

## Reprojection and arrival did not establish a physical target

The candidate's frozen map point became `(1.867464, 0.175820)` m at route
selection, a **47.95 cm** displacement, then `(1.906769, 0.125508)` m at
arrival, a **53.80 cm** displacement. These are changes in a position
hypothesis under map/odometry reprojection, not measured movement of a stand.
The evidence does not uniquely attribute them to odometry versus localization.

Replaying the production static-map admission at both the selection and
arrival targets yields blocked cells with zero clearance: both would be
rejected by the existing policy. The replay verifies the map YAML and image
hashes and reconstructs the recorded arena overlay. The original query was
only boundary-provisional at 6.862 cm clearance.
`navigation/approach/candidate_frame_projection.py:226`
reprojects coordinates and preserves source eligibility; line 246 validates
the snapshot schema without recomputing static-map admissibility. This allows
old eligibility to accompany materially different target coordinates.

The robot stopped at approximately `(1.411210, -0.091908)` m. Arrival admitted
the hypothesis at 0.54115 m base range and **−4.620° optical bearing error**.
That fails the strict 3° check but passes the explicit 6° passive-acquisition
allowance. The receipt correctly records `acquisition_only=true`,
`camera_centered=false`, `head_alignment_verified=false`,
`validated_target_center=null`, and `requires_live_target_association=true`.
No physical-head check is part of that arrival decision.

## What camera processing actually did

Times below are Berlin local time, UTC+2.

- **16:23:07.035:** passive camera capture started after arrival admission.
- **16:23:14.727:** capture returned `unobservable` after 18 fresh processed
  frames; all failed current-head detection. The capture archive contains 24
  synchronized tuples, including tuples not admitted for detector processing.
- **16:23:14.735:** optional LiDAR recovery started.
- **16:23:19.218:** its planning-frame admission was interrupted with
  `KeyboardInterrupt`; the saved inspection termination is
  `lidar_recovery_terminal_failure`. No recovery motion is recorded.

The observer exited successfully with an unobservable progress artifact; it
did not consume the configured 90-second timeout. Camera-centering history
contains zero turns. Mission progress remained three confirmed identities
(`QR_002`, `QR_003`, `QR_004`), one facing-ready stand, and an incomplete goal.
There is no completed mission summary for this interrupted run.

`all_samples_lidar_associated=true` does not mean that a stand shape was
verified. Production replay of frame 24 reproduces its coarse association:
beams `[212, 213, 214]`, median 0.492 m, inside a ±3° cone and
`[0.392999, 0.612999]` m range window. Those returns are a cropped subset of an
extended nearby surface. The association code filters by cone/range before
clustering and does not test full-surface morphology. Its minimum is one beam.
The wider identity search was ambiguous, and displaced-target reconciliation
reported `candidate epoch displacement outside recovery bound`.

Relevant executed code: `perception/candidate_lidar_association.py:209`,
`real_robot/observer/node.py:2360` and `node.py:2552`. The preliminary match can
support an unobservable progress record while head/identity admission remains
blocked. It must not be interpreted as physical-head validation.

## Recent changes and the correction boundary

Commit `7221553` intentionally moved optional LiDAR alignment after the first
camera attempt (`real_robot/candidate/inspection_adapters.py:578`,
`inspection_execution.py:185`). This corrected earlier runs that spent their
initial inspection budget on LiDAR recovery before opening the camera.
The preceding `135242Z` run already had this camera-first behavior. The
`e018878` range-aware fitting and `06b8bda` association correction did not
change survey clustering, morphology admission, or the camera-ranking policy.
The evidence does not support blaming a newly introduced head-detector
false-positive or the recent range-aware fitting for the wall visit.

Corrections identified by the read-only audit (implementation follow-up below):

1. Consume bound unresolved morphology conflicts before camera-route selection.
   Defer contradictory hypotheses, retaining their obstacle keepouts, until
   fresh full-cluster evidence resolves the conflict. Do not blanket-reject
   all boundary or single-view candidates.
2. Revalidate target static-map admissibility after significant reprojection,
   before selecting a route and again at arrival. Require reconciliation or a
   new observation when the projected target becomes blocked; do not silently
   retain old boundary eligibility or simply force frozen coordinates.
3. Keep coarse cone/range association explicitly separate from stand-shape
   evidence. A small cropped wall fragment must not clear a morphology conflict.
4. Preserve the first passive camera opportunity for ordinary admitted
   candidates. Regression coverage should use this recorded wall case and
   verify that it remains a keepout but cannot win routine camera-route
   selection while conflicted/blocked. Also preserve real boundary candidates,
   normal single-view discovery and camera-first execution.

## Evidence and verification

Evidence root: `results/implementation_checks/run_audit_20260930T141147Z/`.

- `source/`: complete copied latest mission and nine parent/child run bundles.
- `source_integrity.json`: **1,246 files**, all matching workstation SHA-256.
- `audit.py` / `audit_summary.json`: integrity, timing, provenance, camera
  outcome, displacement, and production coarse-association replay. All fields
  match, allowing 1e-12 for platform floating-point trigonometry differences.
- `historical_compare.py` / `historical_comparison.json`: six-run population
  comparison and production replay of both previous/latest fourth rankings.
- `wall_evidence.py` / `wall_evidence.json`: morphology, visibility, production
  static-map admission replay and exact detector replay of all 45 original
  sparse-cluster observations.
- `wall_frame_000024.jpg`: original captured JPEG bytes under a previewable
  extension, verified against the saved image digest.

The existing camera-first handoff regression test
`CandidateLidarHandoffTest.test_first_arrival_uses_camera_without_any_lidar_acquisition`
passed. These are offline
reproductions and recorded-run evidence, not a physically validated fix.
No production behavior or workstation state was changed by this audit.
Concurrent local development changes were left untouched.
The final workstation check still identified this as the latest run, with
the same clean revision and no matching mission/child processes running.

## Correction implementation

The local implementation now checks the actual camera target against the
bound, uninflated static map and arena overlay before route selection,
route materialization, arrival, and subsequent camera capture. It uses the
existing stand-envelope policy, so nominal-fit boundary candidates remain
eligible. Unresolved morphology conflicts defer camera inspection;
visibility-gap evidence alone does not.

Rejected targets receive `target_reconciliation_required`. They remain in
the immutable candidate snapshot and every obstacle keepout. A six-hypothesis
pool can therefore complete the five-stand goal using its five valid targets.
If no eligible targets remain, the mission reports an incomplete goal without
starting an observer or issuing a candidate motion command. Target rejection
also terminates optional inspection recovery, including direct LiDAR sampling
turns, instead of authorizing more viewpoints around an invalid target.

The first passive camera opportunity is preserved for admitted candidates.
Coarse cone/range association cannot clear a morphology conflict. The recorded
fixture in `tests/aufgabe04/fixtures/wall_candidate_20260930/` retains the exact
six-candidate snapshot, source registry, and selection/arrival projections,
with original-file hashes and map bindings in its manifest.

A validated LiDAR center used during selection is carried into the fresh
arrival frame with its source snapshot and calibration bindings checked.
Subsequent centering updates reproject the point through odometry; they do not
copy old map coordinates. A newer verified LiDAR fit takes precedence. Each
resulting target is checked again, and live association remains mandatory.

Offline validation passed **287 tests and 133 subtests** across 25 focused
modules, including the recorded wall fixture, five-of-six goal completion,
all-invalid termination, direct sampling, post-centering recapture, corrected
target retention, and existing route/uncertainty/startup recovery tests.
`git diff --check` also passed. Commands, module list, and log hashes are saved
in `results/implementation_checks/run_audit_20260930T141147Z/correction_validation.json`.

The corrected production selector also replays the recorded fourth decision:
it excludes `survey_candidate_0004` and selects `survey_candidate_0005`.
For both remaining targets, every route/ranking metric and the route
uncertainty admission hash exactly match the original recorded computation.
All six candidates remain in the unchanged snapshot. The replay script and
structured result are `correction_replay.py` and `correction_replay.json` in
the same evidence directory. The original camera calibration file is absent
locally; this replay verifies the recorded empty-LiDAR-hint route branch,
which does not consume calibration, and makes no calibrated-arrival claim.

This follow-up changes local source and offline tests. It has not been
deployed to the workstation or validated through a physical robot run.
