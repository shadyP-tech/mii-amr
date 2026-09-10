# First-candidate QR audit — 10 September 2026

**The latest available run detected the first selected candidate's QR code and
completed its configured pilot successfully.** Session
`stand_explore_exact2_camera_pilot_20260910T134103Z` ran from
**13:41:03–13:45:49 UTC (15:41:03–15:45:49 CEST)** on clean commit `770ea6b`.
The parent process returned **0**, with `status: camera_checkpoint_complete`.
Its first selected candidate was `survey_candidate_0003`, and the confirmed
identity was **`QR_003`**. Candidate numbering is not inspection order.

The command explicitly requested `--stop-after-camera-candidates 1`. The runner
stopped after one validated camera receipt, leaving the five-stand goal
incomplete. There is no terminal QR-detection failure in this run.

## Evidence and timeline

| UTC | Recorded outcome |
| --- | --- |
| 13:43:37.822 | First coverage leg completed; estimated travel 0.910806 m. |
| 13:44:40.326 | Second coverage leg completed; estimated travel 1.158217 m. |
| 13:45:36.389 | First candidate approach completed; estimated travel 0.042248 m. |
| 13:45:44.027 | Camera observer reported `waiting_for_sensors`. |
| 13:45:45.933 | First accepted `QR_003` sample, from capture 7. |
| 13:45:46.932 | Second accepted `QR_003` sample, from capture 12; identity latched. |
| 13:45:47.101 | Seventh axis sample available; recommendation committed. |
| 13:45:49 | Parent bundle finished with exit code 0. |

The observer completed in **3.074 s** from its first recorded state, within the
90-second limit. `observer_process.json` records `deadline_expired: false`,
`completion_kind: artifact`, return code 0 and no termination signals.
`inspection_progress.json` records one resolved view and
`termination_reason: joint_observation_ready`; neither the eight-view budget
nor the route-proposal budget was exhausted.

The final observer evidence contains seven axis samples with confidence
**0.922874**, two QR samples, `latched_qr_id: QR_003`, no identity conflict and no
motion-epoch reset. At recommendation publication the source image was
**126.5 ms** old and the scan **184.2 ms** old; both passed the existing gates.
The candidate decision is `confirmed`, and the checkpoint binds `QR_003` to
`survey_candidate_0003` and its validated facing-pose record.

The facing pose at 0.35 m standoff was planned and checked. The checkpoint
explicitly records `motion_to_facing_pose_authorized: false`; this experiment
did not execute final docking or logistics.

## Why intermediate output could look like failure

The observer recorded transient rejections before reaching consensus:

- **Four initial synchronized tuples exhausted their bounded exact-time TF
  retries.** The new buffer initially had no history covering the requested
  sensor times. Later tuples had usable transforms and processing continued.
- **One tuple failed LiDAR target association.** Its nearest return in the
  bearing cone was 1.010 m, outside the accepted 0.556–0.776 m range. Later
  frames associated the target at approximately 0.737 m.
- **One model measurement failed the reprojection gate.** The joint head/QR
  fit reported 12.013 px RMSE and a malformed QR corner. The rejected frame did
  not prevent later valid samples from completing the consensus.

These are frame-level outcomes, not terminal experiment failures. Of 13
synchronized tuples, nine reached image processing, eight produced verified
geometry and seven contributed axis samples. All 13 diagnostic captures were
written, with zero capture drops or write failures.

There is also an important distinction in the final debug output:
`decoded_qr_target_binding.reason: no_decoded_qr_geometry` describes the **last
frame**. The final status's `qr_texts: ["QR_003"]` describes the **retained
consensus**. The observer's [evidence policy](../../scripts/aufgabe04/real_robot/observer/evidence.py)
retains matching QR samples for five seconds in the same stationary epoch;
empty later frames do not erase them. Motion, expiry and conflicting identities
invalidate that evidence. Both accepted QR samples were still within the TTL
when the final axis sample arrived. This behavior is intentional.

## Independent image verification

The [offline replay](../../results/audits/stand_explore_exact2_camera_pilot_20260910T134103Z/derived/qr_replay.py)
decoded **all 13 original 800×600 compressed captures as `QR_003`** using the
repository's bounded preprocessing decoder with OpenCV 4.13.0. Successful
full-image detections used its 2× scaling path. Plain OpenCV decoding of the
unprocessed full images decoded 0/13, showing that image preprocessing matters
for this recording.

This verifies the saved image payloads, independently of the terminal status.
It does not reproduce live timing, rectification, ROI selection, geometry or
target association, and does not establish a hardware success rate. The
[replay results](../../results/audits/stand_explore_exact2_camera_pilot_20260910T134103Z/derived/qr_replay.json)
record methods, per-frame results, library versions and source hashes. The
[final camera image](../../results/audits/stand_explore_exact2_camera_pilot_20260910T134103Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_pilot_20260910T134103Z/candidates/000_survey_candidate_0003/camera_lidar_attempt_00/perception_debug/latest_frame.png)
also visibly contains the stand's QR face.

## Actual defect found: contradictory final reporting

The final `mission_summary.json` mixes current QR progress with an earlier
LiDAR-stage summary:

| Field | Final mission summary | Current evidence |
| --- | --- | --- |
| `candidate_counts.confirmed` | 0 | Current `coverage/survey_summary.json`: 1 |
| `candidate_counts.pending_camera` | 2 | Current survey summary: 1 |
| `confirmed_stand_count` | 1 | Correct: `survey_candidate_0003` confirmed |
| `confirmed_qr_ids` | `["QR_003"]` | Correct: matches observer and checkpoint |
| `next_required_action` | `begin_exact_two_camera_validation` | Stale: the first camera validation already completed |
| `goal_completed` | false | Correct: four identities remain unvisited |

The [autonomous runtime](../../scripts/aufgabe04/real_robot/autonomous_runner/runtime.py)
captures `coverage_summary` before camera execution at line 2749, then expands
that dictionary into the terminal checkpoint summary at line 2924. New goal
fields overwrite some values, but the old counts and next-action string
survive. The current registry and survey summary were correctly updated by the
candidate decision; the defect is in terminal summary assembly.

The next code correction should refresh or explicitly qualify those stage
counts and replace the terminal next action. Add a regression with populated
pre-camera counts and next-action fields; the existing successful-pilot fixture
omits them. Distinguishing latest-frame QR state from retained identity in
operator output would also make the successful result easier to interpret.
No QR-detector threshold change is justified as a remedy for a terminal failure
here, because the pilot succeeded.

## Handoff validation and audit scope

Both stopped LiDAR epochs now imported successfully: **77 and 78 receipts**,
with matching runtime configuration, committed epochs and coverage checkpoints.
Independent replay through the strict visibility importer also accepted both.
This run therefore exercises the latest runtime-configuration correction on
hardware. The two survey legs covered 95.31% of the modelled coverage cells;
that is not a measurement of complete physical visibility.

The candidate follower acquired its initial transforms on the first attempt
in 0.506 s. Coverage leg 001 recovered from four missing global-transform
lookups in 1.231 s. No stale-first transform or extended startup wait occurred,
so this run does not validate that separate recovery branch. Backside inspection,
the opposite-face route and all-five discovery were also not exercised.

All **260 copied source artifacts** match the read-only transfer's
[manifest](../../results/audits/stand_explore_exact2_camera_pilot_20260910T134103Z/source_manifest.json).
Derived timing and outcome data are in
[audit_metrics.json](../../results/audits/stand_explore_exact2_camera_pilot_20260910T134103Z/derived/audit_metrics.json).
Twelve focused evidence-policy and pilot tests passed. This audit made no
production changes, deployed nothing and issued no robot commands.
