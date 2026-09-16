# Opposite-side observer audit — 2026-09-16 09:30 UTC run

The first candidate and the second candidate's backside/opposite-side branch
worked. The failure occurred after the opposite-side drive: the new observer
repeatedly searched a crop that clipped the head, spent its remaining freshness
budget on wider acquisition, and rejected every result. TF delivery also lagged.
The 90-second timeout therefore aborted the mission with one of five identities
confirmed, instead of deferring this candidate.

## Evidence and successful behavior

Run `stand_explore_exact2_camera_all5_20260916T093041Z`, clean commit
`6449a19559f6bf89b45ef1d0d1472085813035c9`, executed on mii001 from
09:30:41–09:39:57 UTC (11:30:41–11:39:57 CEST), parent exit code 2.
Original logs, scans and recordings remained on mii001. Only diagnostic summaries
returned. Saved-image replays used the existing read-only Apptainer environment,
Python 3.10/OpenCV 4.5.4, original calibration and unchanged production geometry.
No production edits, ROS commands or robot motion were performed for this audit.

| UTC | Evidence |
| --- | --- |
| 09:33:05 / 09:34:09 | Both LiDAR legs completed, 0.856615 m and 1.184844 m. |
| 09:35:16.732 | First candidate `survey_candidate_0003` reached. |
| 09:35:32.771 | First-view QR_003 recommendation committed with seven measured-head samples; fit RMSE 0.4416 px, angular uncertainty 1.459°. |
| 09:36:35.550 | Second candidate `survey_candidate_0001` reached. |
| 09:36:45.752 | Backside receipt committed with seven angles and seven QR-absence samples; fit RMSE 0.3509 px. |
| 09:37:01.827 | Initial opposite route at 0.50 m standoff rejected before motion, uncertainty margin −0.064447 m. |
| 09:37:25.864–09:38:20.220 | Bounded 0.45 m standoff alternative drove successfully: 1.229549 m in 54.220 s. |
| 09:39:56 / 09:39:57 | Opposite-side observer deadline, then parent failure/bundle closure. |

Opposite-side arrival passed at 0.456811 m target range and 0.271° bearing error.
The standoff fallback worked and was not the terminal failure. The first-view
observer used candidate tracking six times; the backside observer used it twelve
times after its first registered acquisition. Neither result depended on neck
validation. These are real-run successes of the previous correction.

## Failure at the opposite-side camera pose

`candidates/001_survey_candidate_0001/camera_lidar_attempt_01` processed 54
images. **All 54 were obsolete**, with zero fresh results, zero retained angles,
zero tracking seeds and no observation-evidence snapshot.

| Timing | First candidate, median | Second backside, median | Opposite-side view, median |
| --- | ---: | ---: | ---: |
| Image source age at callback receipt | 34.3 ms | 33.9 ms | **270.9 ms** |
| Image source age at processing start | 68.7 ms | 81.0 ms | **298.8 ms** |
| Detector processing | 258.4 ms | 35.8 ms | **328.2 ms** |
| Image source age at completion | 322.3 ms | 121.7 ms | **614.5 ms** |

Opposite-side completion ages were 547.8–804.6 ms, all beyond the unchanged
500 ms limit. Medians describe separate distributions and should not be added
as if they were one frame. Source-to-callback age includes executor queueing;
these measurements do not isolate camera/network transport latency.

The geometry path also failed independently of freshness:

- Nominal acquisition: 46 unavailable proposals, one ambiguous proposal, seven
  proposals that still did not produce an admissible selected result.
- Selected estimator reasons: 46 `model_current_head_border_unavailable`, three
  `head_model_pose_rejected`, two `model_corner_evidence_insufficient`, one
  `model_corridor_refinement_unavailable`, one `head_proposal_ambiguous`, one
  `backside_complete_head_crop_unverified`. Zero final usable estimates.
- Wider acquisition: 33 proposals rejected at candidate association, 20
  ambiguous searches and one unavailable search.
- All 33 wider association rejections were
  `scan_persistence_current_input_invalid`; the recorded persistence reason was
  `scan persistence sources are not fresh`. This does not establish 33 competing
  LiDAR targets. The separate nested scan age can reflect the earlier pre-search
  clock; the persistence resolver rechecks the actual later clock and the image
  source too.

Nominal acquisition cost a median 133.1 ms and wider acquisition another
148.2 ms. The latter is stored separately in
`metric_model.head_acquisition.elapsed_ms`; a failed wider search is absent from
`processing_timing.attempts` because no strict ROI evaluation follows it.

## Saved-image replay isolates the crop defect

Nine original frames were replayed: 6, 7, 10, 18, 32, 34, 43, 54 and 55. The
complete current head occupies approximately **x=391–497, y=228–339**. Relevant
nominal crops end at **x=492–493**, removing 4–5 rightmost pixels and the additional
corner context required by refinement. Across all operational results, nominal
right bounds were 492–495 and wider fitted proposals extended to 496.44–497.04.

| Unchanged offline image path | Result |
| --- | --- |
| Nominal crop | 0/9 usable head fits |
| Wider acquisition | 4/9 unique proposals, 5/9 ambiguous |
| Recentered strict fit of the four unique current proposals | **4/4 usable** |
| Unprojected full-image cold viewer pipeline | 0/9 usable |
| Full-image, nominal and wider QR decoding | `Start` in 9/9 each |
| Recentered crop decoding | `Start` in 4/4 |

Frames 10, 32, 43 and 54 fit successfully in a recentered crop approximately
`[378,214,511,389]`: yaw **18.357–18.509°** (0.152° span), reprojection error
**0.305–0.349 px**, raw border support **1.0**, verified corners/outer border,
neck validation false. Their strict geometry takes **7.5–17.3 ms**. The original
observer never reached that strict fit because source freshness had expired
before the wider proposal could pass association.

The wider ambiguous frames also contain border-supported background rectangles
above the actual head. Proposal ambiguity is resolved before candidate
association and strict 3D verification; enlarging the image alone is therefore
insufficient. The replay does not select an arbitrary competing rectangle.

The decoded `Start` observations carry **no QR quadrilateral**. Payload decoding
alone cannot bind that identity to the candidate. This replay demonstrates
available image geometry and text, not fresh LiDAR association, seven-sample
consensus or an authoritative QR admission receipt.

## Additional debug recording at the termination pose

The user supplied `recording_20260916_114904_128857980`, stating that the robot
remained at the termination pose. Its 21 source frames cover
09:49:03.823–09:49:09.455 UTC (11:49:03.823–11:49:09.455 CEST). This is a later
recording, not a recording of the original observer deadline. The viewer metadata
does not independently prove the unchanged map pose.

All 21 recorded results lack an accepted head fit or tracking seed:

- 13 cold acquisitions exceed the 1,024-contour limit.
- Five have distinct competing head proposals.
- Three exhaust the verification budget with unresolved independent hypotheses.

The debug viewer searches the full image without the mission's candidate ROI.
Consequently, it sees background contours and competing rectangles; the mission
instead starts with the clipped projected crop. Both paths fail before obtaining
a usable current head. The profile field `model.committable=true` describes the
calibrated physical model, not a successful fit in the displayed frame.

All 21 results also display `obsolete_detector_result`. Median image age at
receipt is **243.7 ms** (233.4–249.7 ms), almost consuming the viewer's **250 ms**
source-age allowance before detection. Median detector duration is 40.2 ms;
actual source age at completion is **360.8 ms** (283.1–513.1 ms). The separate
180 ms result-age check rejects some frames first. Thus obsolete-result handling
and failed head acquisition are independent blockers. This viewer has a stricter
source-age limit than the mission observer's 500 ms limit; their stale-result
counts should not be treated as the same configuration.

All 21 original source images were rectified and replayed in place using the
recorded calibration, measured model and unchanged production fitting gates.
The last mission projection was reused only as a diagnostic search location,
consistent with the user's same-pose description; it is not fresh localization
or LiDAR association evidence.

| Offline path through the additional recording | Result |
| --- | --- |
| Full-image cold acquisition | 0/21 usable fits |
| Full-image acquisition with tracking enabled after a successful seed | 0/21; no initial seed |
| Last mission's nominal projected ROI | 0/21 usable fits |
| Wider acquisition | Five unique proposals, 16 ambiguous |
| Recentered strict fit of those five unique current proposals | **5/5 usable** |
| Full-image or nominal-ROI QR decode | `Start` in 21/21 each; no valid QR quadrilaterals |
| Recentered-ROI QR decode | `Start` in 5/5; one valid QR quadrilateral |

Successful fits occur in frames **2, 8, 11, 19 and 20**, with four recentered
crops approximately `[378,214,511,389]` and frame 19 using
`[370,206,512,392]`. Their angles are **17.951–18.551°**, with a median strict-fit
time of **8.0 ms**. These agree closely with the earlier run-image replay. Frame
19's top/left fitted boundaries differ by approximately 7–8 pixels from the
other four; similar yaw alone does not establish identical physical border
selection. Current-pixel border consistency still needs validation across a
successful observation window. The replay supports usable 3D geometry at this
view, but does not establish seven fresh, associated samples in live use.

The recording used `no_qr_decode=true`; the decoded text above comes from the
separate offline replay. WeChat commonly returns the entire input rectangle as
its symbol boundary, which the existing validator correctly rejects as
`full_input_extent`; scaled/padded fallback can similarly return out-of-bounds
corners. A readable payload is therefore not equivalent to a spatially bound
candidate identity. QR geometry should resolve identity binding without becoming
a size-agreement gate on the separately valid head angle.

Frame 19 does recover a valid quadrilateral through
`opencv_quad_wechat_rectified` at scale 4. Its full-image QR bounds are
`[395.75,234.75,493.27,332.25]`; all four corners lie inside that frame's accepted
head polygon, with minimum edge margin 3.38 pixels. QR decoding takes 33.3 ms,
strict head fitting 7.5 ms, and the preceding wider locator 175.7 ms. This proves
that payload and co-located head/QR geometry are recoverable from the recording.
It still lacks the fresh unique scan association, exact-time TF and matching
cluster evidence required by the live identity-binding path.

## TF and scheduling evidence

The opposite observer attempted exact-TF retry for 345 tuples and exhausted 318.
Of the exhausted attempts, **315** were future extrapolation on
`map <- base_footprint`, one on `base_scan <- map`, and two were startup
missing-frame/past-extrapolation cases. Requested time exceeded latest available
TF by a median **1.221 s**, maximum **4.568 s**, in the 316 future failures.

Deduplicated bounded receipt traces show `odom -> base_footprint`
source-to-ingestion-entry delay rising from approximately 18–19 ms median in
successful views to **169 ms median**, with a **4.973 s maximum** in the failed
view. The actual TF-buffer insertion remained below 0.15 ms. Thus there is delay
before insertion, not evidence that TF insertion itself is computationally slow.
Map/odom transforms mostly remain future-dated, consistent with the existing
broadcaster behavior. These logs cannot separate publisher, DDS/network and
executor queue delays.

The code runs camera processing, sensor callbacks, TF listener and scan-witness
retry callbacks on the same `rclpy.spin_once` executor. Scan collection is
independent of camera *success*, but not concurrent with camera computation.
The failed observer recorded 697 scans expired before witness TF out of 888
received scans. Blocking acquisition is a demonstrated architectural source of
queueing; these counts alone cannot prove a separate LiDAR fault.

During processed stationary observations the projected candidate moved only
2.038 horizontal pixels, 0.027 vertical pixels and 0.000054 m depth. No evidence
here establishes an AMCL pose jump during this observation. The projection is
nevertheless offset from the actual fitted head; without independent ground
truth, its spatial bias cannot be uniquely assigned to localization, candidate
position or calibration.

## Why the mission did not continue

Each inspection pose creates a new observer. The search hint correctly requires
a fresh, uniquely associated current head and does not carry a previous
camera-space pose across the opposite-side motion. All obsolete results return
before creating that hint, so this observer continually pays the cold search
cost.

`observer/timeout_policy.py:47` treats an obsolete-result timeout as a local
candidate failure only when earlier fresh accepted/LiDAR-rejected candidate
frames and explicit unpoisoned evidence establish a working observation path.
This attempt has none. `autonomous_runner/runtime.py:1550–1585` raises a terminal
error, and `candidate/inspection_execution.py:98–121` records
`observer_terminal_failure` and propagates it. The other three candidates remain
pending. Return code 130 is the recorded SIGINT deadline cleanup, not evidence
of a spontaneous observer crash.

## Corrections indicated by this audit

1. Make all physical acquisition and refinement respect the remaining image
   deadline, not only expensive QR decoding. Do not append a wide search after
   the current frame has insufficient time left for association, fit and output.
2. Perform bounded wider reacquisition on fresh images, retain complete head
   margins, and then refit current pixels in the recentered crop. A current 2D
   proposal may locate a subsequent search, but cannot supply a cached angle,
   face classification or admission evidence.
3. Review competing proposals against candidate association and measured 3D
   geometry before declaring unresolved ambiguity, preserving rejection of
   genuinely competing candidate targets.
4. Separate sensor/TF ingestion from expensive detection with bounded latest
   image work; verify that this removes pre-insertion delay. Trace publication,
   receipt, selection, each acquisition stage and final source freshness.
5. Resolve QR quadrilateral recovery/current-head binding explicitly. A decoded
   string without geometry must not silently become the station identity.

Acceptance: reproduce the successful first/front and backside receipts, then
obtain seven fresh consistent angles plus a spatially bound QR at the opposite
pose, allowing the parent to continue. Do not address this by relaxing freshness,
neck/QR-size angle gates, scan uniqueness or route uncertainty checks.

## Evidence locations on mii001

Run root: `results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260916T093041Z`.

- `mission_failure.json`, `candidate_goal_progress.json`, `candidate_selection.jsonl`
- `station_segment_runs.csv`, `run_events/*`
- First candidate `camera_lidar_attempt_00/recommendation.json`
- Second candidate `camera_lidar_attempt_00/axis_observation.json`
- Second candidate `inspection_opposite_01/camera_attempt_01_arrival/admission.json`
- Second candidate `camera_lidar_attempt_01/observer_events.jsonl`,
  `observer_process.json`, `observer_status.json`, `capture_history/*`
- Bundle `results/real_runs/<run>/manifest.txt`, `git_rev.txt`, `git_status.txt`
- Additional recording
  `results/aufgabe04/stand_axis_debug_recordings/recording_20260916_114904_128857980/metadata.jsonl`
  and its original `source_*.png` frames (replayed on mii001, not copied).

The capture limit saved 64 tuples but only 15 completed detector records. The
replay samples those saved records; it does not invent missing images or scans.

## Implemented correction and offline validation

The subsequent code correction changes cold acquisition for a committable
physical model to one bounded candidate search followed by one uniquely
associated, recentered strict fit. It no longer spends a nominal-crop acquisition
before that wider search. A vertical search band of at most 0.75 expected head
heights excludes unrelated upper/lower rectangles while preserving the existing
horizontal/candidate offset limit. It is a search bound, not angle evidence.
All borders and angles still come from current pixels.

`head_acquisition_schedule.py` owns cold-search scheduling and the source-age
budget. `head_acquisition_budget.py` provides cooperative checkpoints for
proposal loops; partial comparisons never return a chosen head. The QR budget
also respects the older synchronized scan's deadline. Skipped decoding is
explicitly unknown and cannot trigger an unbudgeted legacy QR fallback.

Competing proposals use `StoppedScanTargetPersistence.preview()` so an unrelated
rectangle cannot consume or reset the actual target's scan witnesses. Only the
selected head commits association state. The existing crop, uniqueness, current
image, geometry, consensus and publication checks remain enforced.

`ingestion_runtime.py` separates ROS receipt/TF work from detection. Bounded
mailboxes transfer sensor references and original receipt times to the main
owner, which alone mutates evidence, tracking, captures and scan persistence.
Missed processing ticks coalesce instead of queuing detector jobs. Scan overflow
resets consecutive-witness history; shutdown joins ingestion before node teardown.

The QR decoder's opt-in `prefer_native_geometry` mode is enabled only on
registered/tracked head crops. It prioritizes actual native symbol corners and
isolated payload confirmation. It never substitutes the head rectangle for QR
corners and does not impose QR-size agreement on head-angle fitting.

Read-only, in-memory patched replays on mii001 used OpenCV 4.5.4 and left all
original recordings/run data and the remote checkout unchanged:

| Evidence | Patched geometry/decoder result |
| --- | --- |
| Additional termination-pose recording, 21 frames | **21/21** unique proposals and usable strict head fits; `Start` decoded 21/21; valid QR quadrilaterals **17/21**. |
| First candidate, first eight saved detector frames | **8/8** usable fits and `QR_003` decodes. |
| Second candidate backside, first eight saved detector frames | **8/8** usable backside-candidate fits, no decoded QR. |
| Second candidate opposite view, first eight saved detector frames | **7/8** usable fits and `Start` decodes; one genuine unresolved proposal comparison remained rejected. |

For the additional recording, cold proposal + native/head fit + bounded full QR
work took median **212.2 ms**, range **187.5–313.1 ms**. Strict fitted angles were
**17.951–18.551°**. These measurements exclude live receipt/TF and do not prove
seven-sample sensor-bound admission. The eight-frame replay reports head fits
before refreshing face classification with the separately recovered payload;
its opposite-view intermediate face label is not a final backside receipt.
The unchanged full-image debug-viewer cold search is not claimed fixed by a
candidate-specific acquisition correction.

Local validation used Python 3.14/OpenCV 4.13; changed files also pass Python
3.10 syntax parsing. A 93-file perception/observer/QR regression selection
reported **1,068 passed, one skipped, six failures**. The same six failures in
legacy `test_stand_axis_image.py` reproduce with unmodified HEAD production
modules (`6449a19`): baseline 108 passed, six failed. They are not introduced by
this correction. After final deadline/preview integration, 110 focused tests
passed, and two added registration-preview tests plus the persistence/QR policy
selection passed (48 tests). `git diff --check` passes.

No live ROS or robot-motion validation was performed. The next hardware run
must verify receipt/TF latency, seven fresh associated samples, spatially bound
QR identity, and continuation after the opposite-side observation.
