# Run and camera-recording audit: October 1, 2026, 17:11 Berlin

Run: `stand_explore_exact2_camera_all5_20261001T151113Z`.
Recording: `recording_20261001_172410_258156366`, 17:24:10–17:24:17 Berlin.
Workstation: SSH alias `mii002`, hostname `mii001`.

## Finding

The mission stopped at candidate `survey_candidate_0006` because its second
camera-centering turn timed out. It did not stop in the opposite-side branch.
Three of five identities were confirmed and facing-ready, and all seven
executed waypoint legs completed. The initial opposite route failed its
clearance preflight (margin −0.060384 m); the alternate standoff route completed.
The opposite-side camera subsequently committed a QR observation after seven
crop-conflict outcomes.

All eight run bundles record revision
`144f95425e36256b2db82b84cc555e6864c726f1`. The workstation has separate station-tour
edits, but this run predates the local QR-outline-removal revision `8f566e1`.
It therefore does not test that correction. The centering controller and
scan-boundary selection code discussed here are unchanged between those revisions.

## Why centering stopped the mission

The failure is recorded under
`candidates/003_survey_candidate_0006/camera_lidar_attempt_00/recenter_02`.

| Quantity | First turn | Second turn |
|---|---:|---:|
| Full centering correction required | 24.828° | 8.5885° |
| Permitted turn | 16.828° | 0.5885° |
| Achieved signed rotation | 16.569° | 0.281° |
| Final remaining rotation | 0.259° | 0.308° |
| Control cycles | 73 | 215 |
| Runtime, including preflight and stopping | 5.1 s | 12.8 s |
| Outcome | Completed | Turn timeout |
| Stationary stop confirmed | Yes | Yes |

Both turns used candidate-position recovery. That policy reduces the requested
turn in 1° steps until its destination clears the scan boundary. Here it removed
8° from each full correction. Thus the second request was a very small allowed
step, not evidence that the head was already within 0.5885° of image center.

The controller requires a residual no larger than **0.300°**. It commands
`min(0.12, 1.5 * abs(error))` rad/s, without a minimum effective speed. The second
turn's actual recorded commands ranged from 0.015410 to 0.008081 rad/s and ended
at 0.008085 rad/s. Over the last 50 cycles (2.53 s), angular travel was only
0.017°; error remained between 0.3087° and 0.3177°. This is consistent with a
low-speed stall or hardware deadband, although the logs do not independently
measure motor deadband. Translation remained below 1 cm in both turns.

The 15-second child budget reserves 2.5 seconds for stopping and counts from
function entry, including preflight. The runtime reached its timeout at about
12.5 seconds, published zeros, and confirmed a stationary stop at 17:23:13.
The reported reason is `centering turn timeout`, not a collision, translation
limit, or unconfirmed stop. The parent treats the stopped child as a fatal
error and records `failed_closed`. There is no second post-turn camera capture.

Ready centering advice takes priority over head-orientation accumulation and
camera completion. Consequently this small framing correction prevented the
observer from continuing to candidate admission. The 30° front-facing allowance
does not govern this controller: face obliqueness, the 1° image-centering
deadband, and the 0.3° turn tolerance are separate quantities.

Code anchors:

- `real_robot/observer/candidate_position_epoch.py:196`: scan-boundary-limited step.
- `real_robot/observer/inspection_framing.py:30`: destination boundary veto.
- `navigation/waypoint_follower/runtime_components/candidate_centering.py:58`:
  proportional command; line 182 checks the timeout before the completion test.
- `real_robot/candidate/centering_execution.py:96`: centering precedence;
  line 120 propagates a failed turn.
- `real_robot/execution/candidate_centering.py:123`: stopped child becomes an error.

## What the later viewer recording adds

The recording contains 54 frames. All passed freshness checks; none was
strict-pose usable. The reasons were:

- 37 `head_model_planar_axis_ambiguous` frames, with complete supported borders
  and an independently accepted orientation interval.
- 17 `head_proposal_ambiguous` frames, rejected earlier because proposal selection
  did not resolve competing borders.

Across the 37 bounded frames, interval centers range from 9.424° to 11.920° and
half-widths from 12.733° to 17.570°. Every complete interval is inside ±30°;
the largest absolute endpoint is 29.490°. Only 16 have half-width at most 15°,
the stricter unidentified/backside orientation-window limit. These are individual
frame bounds, not a demonstrated seven-frame mission admission.

The viewer's strict single-pose rejection does not mean the border failed to
provide approximate orientation. Its displayed usability follows the strict
estimate, while bounded orientation is a separate proof. The viewer, strict
quality evaluator, and bounded evaluator are unchanged by the local ID-only fix.

The recording was made with `no_qr_decode=True`. It therefore cannot establish
which ID is readable, prove front/back, or supply an attempted-but-empty identity
sample. It also uses nearest-head selection, rather than the mission's full
candidate receipt. Its timing and geometry are useful follow-up evidence, but
do not alone establish an identical candidate and sensor tuple to the failed run.

## Correction indicated by this audit

The immediate issue is how optional camera framing interacts with a tiny
scan-boundary-limited turn and mission failure. A correction should avoid
repeated ineffective small steps while retaining the scan-boundary veto,
motion budgets, and verified stationary stop. It should allow a fresh stopped
observation when further useful centering is unavailable, with explicit handling
for that outcome rather than treating every incomplete framing turn as a fatal
mission error. Controller progress and realizable small-turn accuracy also need
coverage using this 0.5885° request. Simply extending the timeout does not address
the observed lack of progress.

The viewer should additionally expose accepted approximate angle bounds beside
its strict-pose rejection, so its display reflects the mission's two angle paths.
The existing local ID-only correction does not fix the centering failure.

## Evidence handling and limits

At the user's request, all raw logs, images and videos stayed on the workstation.
This audit used remotely computed state counts, rounded control statistics and
frame-metadata summaries, plus local source comparison. No raw bundle or recording
was copied, and no local visual frame inspection or new decoding replay was
performed. Source files live beneath the workstation repository's
`results/aufgabe04/real/autonomous_exploration/<run>` and
`results/aufgabe04/stand_axis_debug_recordings/<recording>` directories.

Relevant sources are `mission_failure.json`, `candidate_goal_progress.json`,
`station_segment_runs.csv`, the observer-event streams, both centering permits,
results and controller traces, and the recording's `metadata.jsonl`. A final
remote check confirmed this was still the latest exploration run. No production
code, workstation checkout, ROS process or robot state was changed by this audit.
