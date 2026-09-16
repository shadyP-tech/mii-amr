# First-candidate admission audit — 2026-09-16 11:41 UTC run

Run `stand_explore_exact2_camera_all5_20260916T114135Z`, commit
`a9aac95e743e17c3ceb671280b4e2be8de0b81fd`, includes the bounded-head admission
implementation. The first candidate was `survey_candidate_0003`, whose QR is
`QR_003`. Neither of its first two camera observations admitted it.

The failures are principally **acquisition/association and evidence coupling**.
They are not explained by a generally invisible stand, stale camera results,
or the debug viewer being unable to compute its angle. The new bounded front
path unnecessarily requires a current QR decode on every geometric sample.
Separately, the mission's head acquisition sometimes selects internal borders
and clips the actual QR, and at the second pose repeatedly fails scan
association or its acquisition time budget.

Audit was read-only. Original recordings, scans, images and run files remained
on mii001; only derived diagnostic summaries returned. Saved-image replays used
the existing read-only repository/container and unchanged production code. No
ROS commands, robot motion, deployment or production edits were performed.

## Mission timeline and limits

The bundle reports host `mii0002`; files were accessed through mii001. Times
below are UTC; add two hours for CEST.

| Time | Event |
| --- | --- |
| 11:41:35 | Parent run started. |
| 11:44:01 / 11:45:19 | Both LiDAR survey legs completed. |
| 11:46:26 | First candidate approach completed. |
| 11:46:34.500–11:47:04.858 | First camera observation processed 123 images, then emitted an advisory. |
| 11:48:00 | Drive to second local inspection pose completed. |
| About 11:48:44 | Second camera observation emitted an advisory after its stopped recovery budget expired. |
| 11:51:01 | Another inspection route passed dry-run preflight. |
| 11:51:08 | Last inspected parent-terminal line: running the wrapped command for that next leg. |
| 11:51:42–11:51:47 | User's debug recording at the stated second inspection pose. |

Both observation processes returned zero with `completion_kind=artifact` and
`deadline_expired=false`. They produced `inspection_observation.json`, not a QR
recommendation. Their 30-second stopped front-recovery periods expired normally;
the earlier premature-recovery-exit bug did not recur.

At the audit snapshot the goal ledger still reported **0/5 confirmed stations**,
only this candidate inspected, with no parent completion/exit marker or
`mission_failure.json`. This establishes two unsuccessful observations, not an
explanation for the overall parent termination. The last process query did not
find a matching exploration parent; operator interruption is not established.

Initial arrival passed at **0.5369 m** range and **−2.452°** target-bearing error.
The second arrival passed at **0.5097 m** and **−1.873°**. Candidate position
uncertainty was 0.02 m. Arrival admission was not the blocker at either pose.
The first fallback changed the achieved canonical viewing direction by about
40.3°; the bounded-head 20° hint did not survive into the first advisory.

## First pose: wrong head crop plus seven paired-success requirement

All **123/123 processed images were fresh**. Median receipt age was 29.87 ms;
median completion age 181.38 ms; maximum completion age 454.66 ms.

There were 56 individually usable geometry results. Across current bounded
proofs, 82 images had accepted orientation bounds and 65 also had accepted
current-head candidate association. However, only **two** combined accepted
head bounds with a currently bound QR decode and entered the bounded front
window:

| Image stamp | Head angle | Noise-expanded half-width | Window count |
| --- | ---: | ---: | ---: |
| 11:46:46.282522 | 23.8218° | 3.8788° | 1 |
| 11:46:48.649413 | 24.0162° | 3.8498° | 2 |

These two did not fail the interval-width or temporal-border tests. They simply
did not reach the seven paired samples required by the new path. Nineteen
bounds-plus-associated images already had a live QR identity latch, but current
QR decode/binding was still required on every geometric sample. This couples
the otherwise independent head and identity channels too tightly.

`QR_003` was decoded in 19 distinct images when combining status and ROI
evidence. Seven QR bindings were accepted, and five survived the common sensor
gates into QR evidence. The five-second QR latch eventually expired. The final
advisory was `front_unreadable`, with no live retained identity.

Saved-image replay establishes a second concrete defect. Frames 000011 and
000015 selected a crop **x=385–463** around a purported head only
**x≈396.9–450.4** wide. In the original full image, QR corners extended over
**x≈368.71–449.07**, y≈237.39–329. The selected crop therefore removed the left
part of the actual QR before decoding. Full-image decoding and a diagnostic
complete crop `[340,200,480,380]` recovered `QR_003` with valid corners in both
images; the selected crop decoded nothing.

Proper full-head fits on frames 000054 and 000063 enclosed approximately
**x=360.5–455.3, y=229.4–337.3**, and decoded the QR. Across five replayed images,
full-image decoding succeeded 5/5, selected-crop decoding 3/5, complete-crop
decoding 5/5. This is evidence of recoverable pixels, not live motion authority.

The narrow internal rectangle was repeatedly classified as backside: 41 frames
reported `axis_estimated_current_measured_head_backside`, with angles
**53.07–62.09°**, rather than the approximately 24° complete-head orientation.
The persistent front-marker veto correctly prevented an opposite-side
commitment and reported `front_seen_axis_unresolved`. Removing this veto would
turn a real crop defect into a false backside decision.

Other final geometry/acquisition outcomes included 34 proposal-association
rejections, 17 planar ambiguities, six insufficient-corner results and one
unresolved physical boundary. The proposal failures reported competing
registered camera clusters, not source staleness.

## Second pose: acquisition rarely reaches the geometric estimator

All **52/52 processed results were fresh**, but there were **zero usable
single-angle fits, zero admitted axis samples and zero admitted QR samples**.

| Per-image result | Count |
| --- | ---: |
| Head-acquisition deadline exceeded | 33 |
| Head proposal rejected by candidate association | 15 |
| Head pose rejected | 2 |
| Planar ambiguity | 1 |
| Insufficient corners | 1 |

Thus **48/52 images never reached the full head-fit/QR path**. The wide cold
search ROI was `[168,59,652,542]` (484×483 pixels), with roughly 3,000–7,000
proposal hypotheses and approximately 333 ms of acquisition work. Median
overall detector time was about 349 ms. These were cooperative acquisition
deadline aborts, not obsolete detector results or the 90-second camera timeout.

The proposal diagnostics contain 162 candidate-filter rejections, all
`ambiguous_registered_camera_clusters`. Scan metadata was accepted as a
validated full rotation for only **2/52** processed scans; 42 had inconsistent
endpoint metadata and eight had a seam that was not one sampling step.

Frame 000025 illustrates the consequence. It decoded `QR_003` with a quad, but
the QR ray encountered tail beams 222/223 at 0.531/0.539 m and beam 0 at 0.541 m
as two clusters. Their spatial endpoint gap was 2.749 cm, but the scan's indexed
seam was 1.81692 sampling steps versus 1.01699 implied by its reported endpoint.
The endpoint disagreement of 0.79994 steps exceeded the 0.25-step consistency
limit, so circular merging was not authorized. The QR binding was rejected as
`ambiguous_qr_target_clusters`. Temporal scan witnesses were also often lost
before exact-time TF became available; 138 such witness expirations were logged.

Only frame 000024 retained a bounded fit: center about **3.67° ±23.30°**, with
competing angles −13.065° and +16.480°. Its selected head ended roughly 4–7 px
inside the right/bottom borders followed by the later viewer. Native finder
evidence was present but full QR acquisition had exhausted its budget. The
following tracked frame was fast (22.98 ms) and decoded the QR, but its head
corner evidence failed and its QR association was ambiguous. Tracking was lost
and cold acquisition resumed.

Read-only replay of six original mission images recovered `QR_003` plus valid
corners in all six full images. Full-image **cold** geometry yielded one strict
fit, four ambiguities and one unavailable head. Replaying selected ROIs without
the timer yielded three strict fits, but their angles varied substantially
(−20.1°, −9.68°, +13.38°). Removing only the timer or only the admission threshold
would not establish stable correct physical-border selection.

## What the debug recording proves

Recording:
`results/aufgabe04/stand_axis_debug_recordings/recording_20260916_135142_300910942/metadata.jsonl`.

The recording supports the user's observation: **48/48 distinct fresh frames
had strict measured-head geometry and the purple overlay accepted**.

- Angle: **−11.042° to −10.566°**, median **−10.872°**.
- Yaw standard-deviation estimate: 1.924°–1.978°.
- Reprojection RMSE: 0.655–0.679 px; raw border support 1.0 throughout.
- Bounded proofs accepted throughout, around ±5.82° per image.
- All 42 rolling seven-frame windows had half-width 5.906°–6.059° and spanned
  0.647–0.693 seconds. Maximum corner movement was 0.211 px, about 0.00197 head
  heights, well below the temporal 0.04 limit.
- Median processing time: **30.58 ms**, maximum 52.44 ms.

The viewer followed the full head at approximately
**x=338.6–443.2, y=231.3–340.5**. Every recorded frame used `tracked_head_search`;
the recording begins after acquisition and does not show how the track was
bootstrapped. It used the full 800×600 image with `target_roi=None`, no hold and
no median smoothing. Channel-union/Canny 20/60 was configured.

Crucially, **`no_qr_decode=True`**. Native QR geometry was detected, but this
recording does not establish decoded identity or association to the mission's
named LiDAR candidate. The geometry result is nevertheless strong evidence
that the pipeline can measure this head once it tracks the correct borders.

## Admission simplification supported by this evidence

Candidate observation completion can use a decoded, candidate-bound QR identity
and a fresh head angle accepted by the **same current-pixel 3D method as the
viewer**. They can be obtained within one short, unchanged stopped observation
epoch; requiring seven simultaneous QR-decode-plus-angle successes is not
necessary. Keep conflicting identities, stale pixels, a different candidate
and incomplete physical heads out of that combined observation.

This change would directly address first-pose starvation. It cannot by itself
fix the second pose, where acquisition and scan association prevent the good
head/QR pair from reaching admission. The implementation priorities are:

1. Separate current geometry accumulation from the existing fresh QR identity
   latch. A decode dropout must not erase an independently valid head angle.
2. Make mission acquisition track the complete physical outer head, as the
   viewer does; reject internal rectangles using current outer-border evidence
   and measured camera/LiDAR scale, without restoring QR-size agreement as an
   angle gate. Retain bounded search hints through transient failures without
   reusing historical measurements.
3. Correct scan angular provenance or establish a separately validated seam
   association. Do not blindly merge ends whose metadata are inconsistent.
4. Reduce repeated cold-search cost so complete-head fit, QR decoding and target
   association can all finish inside the fresh-image budget.

Route clearance, localization and any uncertain-orientation viewing check are
separate movement gates. A purple overlay or QR decode alone does not certify
the route. No implementation changes were made during this audit.

## Code and artifact references

- `scripts/aufgabe04/real_robot/observer/bounded_head_observation.py:54`: current
  bound QR required when preparing every front sample.
- `scripts/aufgabe04/real_robot/observer/bounded_head_window.py:88`: repeated QR
  sample requirement, in addition to the seven geometric observations.
- `scripts/aufgabe04/real_robot/observer/node.py:1954`: persistent front evidence
  vetoes a contradictory per-frame backside angle.
- `scripts/aufgabe04/real_robot/observer/head_acquisition_schedule.py`: bounded
  cold search and cooperative processing deadline.
- `scripts/aufgabe04/real_robot/observer/candidate_head_tracking.py`: mission
  current-head locator lifetime.

Remote run root is under
`/home/amr_gruppe01/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260916T114135Z`.
Inspect `candidates/000_survey_candidate_0003/camera_lidar_attempt_00` and
`camera_lidar_attempt_01`, especially observer events/status, capture history,
inspection observation and process evidence. Arrival proofs, inspection-history
revisions, `station_segment_runs.csv`, goal ledger and the parent bundle
`terminal_run.log` establish the mission timeline independently of image fits.
