# Camera admission regression and candidate-coordinate drift — 2026-09-14

The audit found three interacting problems: inconsistent physical outer-border selection, admission of already panel-rejected hypotheses into the temporal head window, and long-lived stand coordinates tied to drifting odometry. The debug viewer additionally draws purple model geometry independently of final freshness and admission, making its display an unreliable indicator of mission readiness.

The first candidate's QR decoder did work in both examined runs: **QR_003 was decoded and latched**. Neither run obtained the required joint QR/head-angle completion at that first view. Finding a QR, fitting one head, accumulating seven admissible angles, and committing a candidate are distinct recorded outcomes.

## Scope and evidence

- `stand_explore_exact2_camera_all5_20260914T143415Z` matches the report of three first-candidate views followed by offset second-candidate views. Its last stored mission artifact is dated **14:52:56 UTC / 16:52:56 CEST**. It records **0/5 confirmed candidates**. There is no mission failure/summary artifact or recorded terminal exception; the exact cause of the parent process ending cannot be established. The last navigation leg completed, but its subsequent third camera observation for the second candidate is absent.
- A newer run, `stand_explore_exact2_camera_all5_20260914T145529Z`, was active when the audit began. Its first camera observation had finished by the camera audit snapshot at **15:03 UTC**. That observation is analyzed separately; it is not presented as a completed full-run result.
- Recorded code for the offset run is clean **`6baeb220813ee9ab4e36e01543621f152e233d07`**. Local code inspection uses that commit.
- Camera statistics deduplicate detector results by `processing_timing.image_stamp_sec` in `observer_events.jsonl`. Capture-file count is not used as the processed-image count.
- Analysis read stored JSON and scans in place through `mii001`, under `/home/amr_gruppe01/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr/results/aufgabe04/real/autonomous_exploration/`. No raw bundle or image was copied, no file was written remotely, and no ROS or motion action was run. Automatic approval review rejected a proposed bundle download under the earlier no-transfer instruction; this audit continued through in-place derived summaries. New camera images were not visually inspected or replayed.

## Why the newest first view did not admit the visible head

The `145529Z` first view processed **56 images**: **41 fresh**, 15 obsolete. The selected results comprise five usable head fits, 30 QR-panel boundary rejections, nine pose rejections, nine unverified complete-backside-crop results, and three corner failures. QR_003 was latched with two QR samples. Axis evidence peaked at **2/7**, then fell to zero. The emitted artifact is an `unobservable` inspection-progress observation with 22 advisory samples, no angle and no completion authority.

The saved coordinates isolate a border-selection failure from target-projection motion:

1. Two accepted, outward-recovered head fits produced **29.720° and 28.330°**.
2. A subsequent panel-rejected fit remained temporally compatible.
3. The next fit lost outer-border recovery. Its left and bottom corners moved inward approximately **9 pixels**. The temporal displacement was **0.0843 head heights**, exceeding the unchanged 0.04 limit.
4. Across those four frames the nominal projected center moved only **0.087 pixels** and expected head height remained approximately **107.32 pixels**. The changing fitted borders cannot be explained by that tiny projection change.
5. Later selections used a different top border and produced **−16.121°**; a narrower right border produced **37.984°**. The retained temporal span reached **54.105°**. Each cited frame had one pose hypothesis below two pixels reprojection error. Different current corners, rather than two equally plausible poses of the same corners, caused the discontinuity.

There were 35 temporal reviews: three consistent, ten border-unstable and 22 both angle- and border-unstable. Thirty panel-rejected fits entered that window; 29 participated in unstable windows.

This exposes a concrete contract defect. [model_pipeline.py:192](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/model_pipeline.py:192) preserves pose hypotheses and raw-support flags after the QR-size boundary rejects a rectangle as a paper panel. [head_observation_window.py:32](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/head_observation_window.py:32) then admits its corners and plausible yaw as temporal head evidence despite that explicit boundary rejection. The temporal guard clears earlier accepted head angles.

Retaining ambiguous poses of a verified physical head is useful. Treating a rectangle rejected as a different physical boundary as another head measurement is the integration error. Fixing that distinction alone will not make the rejected rectangle usable: acquisition must still recover the actual current outer borders.

The marker-boundary decision itself also needs validation: almost identical recovered outer corners changed acceptance when measured QR spans changed from approximately `.843/.856` to `.864/.857`. Expected symbol spans are approximately `.795` for the physical head and `.873` for the paper panel, with a roughly `.04` discrimination margin. Recorded nominal acquisition does not retain all competing proposal diagnostics, so the evidence cannot always separate initial ranking from subsequent raw refinement as the first source of the inset choice.

## Why the preceding run also missed its first view

The `143415Z` first view has a different immediate bottleneck:

| Recorded quantity | Result |
| --- | ---: |
| Distinct processed images | 147 |
| Usable selected head fits before final admission | 58 |
| Obsolete detector results | 100, or 68.0% |
| Fresh detector results | 47 |
| Fresh frames rejected by LiDAR association | 23 |
| Admitted temporal reviews | 24, all consistent |
| Peak/final axis evidence | 6/7, then 2/7 |
| Median / 90th-percentile detector time | 490 / 607 ms |

Here the temporal gate was **not** the cause: maximum admitted corner displacement was 0.03083 and maximum angle span 3.944°. Too few admissible measurements coexisted in the five-second window. QR_003 was already latched when the 90-second observation deadline expired.

There were **142 same-image geometry refits after QR decoding**. All 81 nominal panel rejections triggered expanded acquisition, but every expanded registration failed because the persistence source data had become stale. The generic `scan_persistence_current_input_invalid` label masks the recorded internal reason, `scan persistence sources are not fresh`. Six images still passed the detector-result freshness check while scan/persistence source freshness failed. Increasing only the camera timeout would repeat this loss.

The later expanded proposals were approximately 37 pixels left and 15 pixels above nominal projection, with median bearing difference **3.79°**. In the newer first view those offsets were approximately 64 pixels left and 22 pixels above, or **6.40°**. Recentring succeeded on 38/56 newer frames; their outer/inner border switching remained. The viewer's successful image-based acquisition therefore does not establish that the autonomous projected crop and timing path are equivalent.

The preceding run's second first-candidate view produced `backside_unresolved`, with no certified axis; its third produced `front_readable`, QR_003 and two QR samples, but no angle. Neither artifact authorizes candidate completion. After **three actual views**, 23 generic route proposals and five exhausted directions, the search ended with `view_proposals_exhausted`. The **eight-view limit was not exhausted**. The candidate was marked `inspection_exhausted`, and the mission continued to the next candidate through its existing incomplete-candidate policy.

## Why the second candidate's observation region moved

All **34** saved candidate-frame projections use one source registry and one canonical odom point per UID. Independently recomputing each rigid transformation matches the saved point to within **2.5e−16 m**. There is no evidence of a camera result rewriting the next stand position, nor repeated addition of the preceding offset.

The code instead re-expresses an old odometry landmark through each new localization transform:

`stand_map_now = map_from_odom_now × stand_odom_saved_at_survey`

That is mathematically consistent, but assumes the old stand coordinate remains valid in current odometry. It cannot correct accumulated odometry error. AMCL can correct the robot relative to static map walls while this policy moves every old stand point through the correction. See [candidate_frame_projection.py:218](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/navigation/approach/candidate_frame_projection.py:218).

For the second inspected candidate, `survey_candidate_0001`, the survey map position was **(−1.1333, −0.4394) m**:

| Stage | Reprojected position, m | Displacement from survey point |
| --- | --- | ---: |
| First candidate's first arrival | (−1.1143, −0.4329) | 2.0 cm |
| First candidate's second view | (−1.1285, −0.3650) | 7.5 cm |
| First candidate's third view | (−1.1329, −0.3231) | 11.6 cm |
| Selection of second candidate | (−1.1370, −0.2907) | 14.9 cm |
| Second candidate's first arrival | (−1.1425, −0.2606) | 17.9 cm |
| Second candidate's second arrival | (−1.1627, −0.1943) | 24.7 cm |

The map←odom rotation changed **15.33°** from this candidate's survey basis by its second arrival. Source stand uncertainty remained 2 cm. Per-route and stopped-window consistency checks do not bound this accumulated landmark displacement across the camera phase.

A recorded safety stop at **14:43:54 UTC** occurred during the first candidate's third approach: anchor drift **0.02657 m** exceeded its **0.02479 m** allowance. The robot stopped and a fresh localization reseal completed the leg. That gate worked for the current route; the reseal did not independently remeasure the physical positions of all stored stands.

The second candidate's two recorded camera views processed **413 and 431 frames, all fresh**. Both had no QR, no head proposals, no raw head-verification candidates and no eligible LiDAR target clusters. Their expected target surface windows were **0.362–0.582 m** and **0.338–0.558 m**, but the nearest finite returns in the projected three-degree cone were approximately **1.412 m** and **1.275 m**. The robot was therefore inspecting an empty/wrong predicted region. Camera throughput and temporal angle thresholds cannot explain those two views.

These data support the reported offset, but projected coordinate displacement is not an independent measurement of the stand's physical displacement. A lightweight saved-scan diagnostic fitted long, low-residual line windows and transformed their directions into map. Median wall-axis deviations across the five recorded camera views were **−0.58°, −0.01°, +0.31°, +0.12°, −1.57°**, much smaller than the changing odometry/map basis. This supports AMCL compensating odometry error rather than arbitrary large global rotation, but is not surveyed ground truth or proof of a particular IMU/wheel fault. No odometry reset was established.

There is also a smaller observation-time frame gap: [runtime.py:1482](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/autonomous_runner/runtime.py:1482) passes fixed arrival-projected stand x/y, while [node.py:1088](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/node.py:1088) uses new exact-time camera/scan transforms. Localization changes during observation can shift the ROI relative to that fixed target. Camera captures omit direct map←odom and odom←base samples, limiting exact attribution within each image window. This is a separate contract issue; the newest initial nine-pixel border switch occurred with negligible projected-center motion.

## Why the purple debug overlay does not establish admission

The visible model should not be dismissed as an arbitrary guess. With the measured physical profile, [head_model_fit.py:111](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/head_model_fit.py:111) creates 3D projected landmarks after a successful current raw-head fit. However, later invalidation and drawing are disconnected:

- A QR/paper-boundary rejection clears the usable pose and quality acceptance but leaves `projected_landmarks` in debug artifacts.
- [stand_axis_viewer.py:3450](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/debug/stand_axis_viewer.py:3450) rejects an obsolete result without clearing those landmarks.
- [stand_axis_viewer.py:4204](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/debug/stand_axis_viewer.py:4204) draws the purple 3D overlay whenever the landmarks exist, independently of final admission. A separate dashed head outline uses proposal/seed corners and is even less restrictive.

Thus a correct-looking purple overlay may coexist with an explicitly rejected measurement, including with zero hold time and median window one.

The viewer also uses image/QR/tracker-seeded head acquisition and calibrated LiDAR bearing derived from the fitted head. Autonomous observation starts with a named map candidate's projected location, size, crop and accepted LiDAR range, then applies bounded current-image correction. A visually convincing head elsewhere in the image does not establish that it belongs to that candidate.

The earlier viewer command explicitly sets channel-union and Canny 20/60, matching the mission; different defaults should not be blamed for that command. `--no-qr-decode` disables the viewer's background identity decoder, but the metric estimator still performs QR quadrilateral/marker checks internally. It does not make the geometric estimator QR-free.

## Prioritized corrections and acceptance evidence

1. **Fix the measurement contract.** Separate verified physical-head alternatives from explicit panel/non-head rejection. Only physically eligible current-head hypotheses should enter temporal head comparison. Preserve ambiguity checks for real heads, freshness and seven-sample consensus.
2. **Make current outer-border choice reliable and observable.** Resolve the enclosing physical frame using current border support and independent marker geometry; retain nominal competing-hypothesis and refinement diagnostics. Test the actual nine-pixel inset switch and QR-span instability, rather than widening the temporal limit.
3. **Remove repeated same-image acquisition.** Reuse the current raw-edge fit when attaching newly decoded QR evidence; perform full reacquisition only when its geometry requires it. Measure image, scan and completion age at each stage. The first run's 81 stale expanded registrations are a concrete acceptance target.
4. **Repair landmark lifetime and frame admission.** Bound accumulated drift since survey, compare stored candidates to fresh stationary LiDAR/map evidence, and revalidate/relocalize or resurvey before planning further views when they disagree. A routine route reseal must not silently renew confidence in an old odometry landmark. Simply disabling reprojection or increasing association tolerance is not sufficient.
5. **Share an explicit current-measurement result between viewer and mission.** The viewer should distinguish accepted, rejected and obsolete geometry, and show the same admission reason, candidate binding and sample count. A rejection must not leave an indistinguishable accepted-looking overlay.

Acceptance should demonstrate one first-view QR_003 identification plus seven fresh, stable, candidate-associated physical-head samples, followed by a valid facing result. The second candidate must be independently present at the predicted bearing/range before its approach/inspection is accepted. Negative cases include panel borders, stale scans, unstable corners, competing stands and accumulated landmark/frame disagreement. No confidence threshold was changed during this audit.

## Selected source hashes

Computed in place for `143415Z`; hashes identify the inspected artifacts, not downloaded copies:

| Relative mission path | SHA-256 |
| --- | --- |
| `candidate_snapshot.json` | `12711f212d84fc7f4f6bbc9005177825ec1a27bea789f6c9c8c8b5cd69df7171` |
| `camera_source_stand_registry.json` | `f797a95281bd48a7cbb9bbbe56657261e87f28c872a5b51df7f85e1f32863d24` |
| `candidate_goal_progress.json` | `3733f351179a80adc444e7a65a76a6549f501047e4695186aadf4d9cec2da503` |
| `station_segment_runs.csv` | `2c0a4e3af9a19655f32e5cb924e4f6b3eb307dc20c36071fc2fd2bb9af775075` |
| `candidate_frame_projections/selection_001/candidate_frame_projection.json` | `416999852915700be8ab68a076aba12020ab4fbe020814d2a94e47e920de27a0` |
| `candidates/000_survey_candidate_0003/camera_lidar_attempt_00/observer_events.jsonl` | `d655fa9f83d5eef1d404a2e52a8027169d1ac6500e4a1e38c4bec33c4376b1ef` |
| `candidates/001_survey_candidate_0001/camera_lidar_attempt_00/observer_events.jsonl` | `f95b6ed5e3cc3eddd446a8017cf591fa119bbed29af101e2ee717fd21664a5dd` |

Audit only. Production code and run evidence are unchanged.
