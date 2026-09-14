# Backside acquisition, front-view association and deadline audit — 2026-09-14 12:37:17 UTC

Run `stand_explore_exact2_camera_all5_20260914T123717Z` ended with **1/5 confirmed QR identities**. The first camera view of its second candidate does show the backside, and the third view does show the QR frontside. The observer did not obtain a certified backside angle in the first view, and did not obtain a candidate-bound QR identity or accepted axis sample in the third. An additional timeout-classification defect then escalated that unsuccessful observation to a mission abort.

The recorded checkout was clean commit **`1cf07084d030e8d26f86f1c1481a5e5803c8af21`**. The run lasted **14:37:17–14:49:33 CEST / 12:37:17–12:49:33 UTC**, exiting 2. Bundle hostname is `mii0002`; the artifacts were retrieved through SSH alias `mii001`. All **841 original files** matched source SHA256 hashes and remained unchanged during retrieval. This audit downloaded recorded evidence only; it uploaded no files and performed no remote writes, ROS execution or robot motion.

## What completed

Both survey legs completed, recording **2.063935 m** of translation, **95.31% modelled coverage**, and five retained hypotheses. All **seven executed navigation legs completed**. The first candidate, `survey_candidate_0003`, confirmed **QR_003** with **7/7 measured-head axis samples** in approximately 3.71 seconds of observer history. Its small alignment movement preceded camera observation.

The second candidate, `survey_candidate_0001`, reached three actual camera poses. Neither a certified backside observation nor a validated final recommendation was written for it. The other three candidates remained unvisited.

| Camera observation | Time UTC | Distinct processed images | Fresh detector results | Usable model results | Accepted axis samples | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| First candidate / first view | 12:42:33–12:42:37 | 12 | 12 | 11 | 7/7 | QR_003 confirmed |
| Second candidate / backside, attempt 00 | 12:43:47–12:43:59 | 17 | 7 | 0 | 0/7 | Unobservable inspection-progress receipt |
| Second candidate / edge-on, attempt 01 | 12:45:06–12:46:35 | 273 | 232 | 0 | 0/7 | Candidate-local deadline; another view allowed |
| Second candidate / QR frontside, attempt 02 | 12:48:02–12:49:31 | 128 | 23 | 1 | 0/7 | Terminal deadline |

Counts deduplicate complete observer histories by processed image timestamp. A usable model result is not necessarily fresh or associated with the candidate. Capture histories are capped at 64 images per attempt; that cap limits saved diagnostics, not live detection.

## Why the first backside view did not trigger the opposite-side branch

![Second candidate at its first camera pose: the full backside is visible](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260914T123717Z/candidates/001_survey_candidate_0001/camera_lidar_attempt_00/perception_debug/latest_frame.png)

All **17 processed images** returned `model_backside_head_and_neck_unavailable`. Every wider-search attempt reported `head_proposal_unavailable`. No measured-head pose, head-quality result, backside-classifier proof or accepted axis sample existed. The presence of `model_backside_current_frame` in an unusable estimator's source field is not evidence of a classified backside.

The nominal crop spans **x=264..448**, clipping the visible head's left border near x=258. The wider crop **x=128..585** contains the complete head, so simply expanding the image is insufficient: acquisition still rejects its geometry before classification.

The remaining neck dependency is explicit. [head_proposal.py:235](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/head_proposal.py:235) rejects an otherwise raw-supported four-border proposal unless `_short_centered_neck_support` passes. Separately, [head_model_fit.py:73](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/head_model_fit.py:73) does not even invoke the planar pose solver without that short-neck evidence. The downstream physical head/neck-junction and pose-quality gates also remain. The previous rail-corridor correction is present in this run, but does not resolve this recording's acquisition failures.

A local diagnostic replay of all 17 processed saved images traced 34 ROI searches and 49 raw-border verifications. Nine of the 17 images yielded seventeen proposed rectangles that passed four-border refinement but then failed the short-neck gate: twelve paired-continuity failures and five start-gap failures. For example, saved `frame_000010` has strong border support but only seven paired neck rows against twelve required; `frame_000012` reaches twelve rows but begins 21 pixels below the head against eight allowed. Passing these 17 refined proposals into the unchanged downstream head fit also produced zero usable angles: every fit still failed `head_model_centered_neck_unavailable`. Removing only the early proposal filter would therefore not fix the handoff. These internal traces use local OpenCV 4.13.0 versus recorded 4.5.4; the runtime's own evidence establishes acquisition failure, while the local trace explains reproducible sensitivity to neck evidence. All 17 wider-search outcome/proposal-count/verification-count triples matched the recording, and the regenerated final nominal raw-edge mask was pixel-identical to the saved mask (zero differences across 33,672 pixels). These checks strengthen the replay without claiming complete backend equivalence.

The seven fresh, synchronized, stationary and LiDAR-associated frames produced an [inspection observation](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260914T123717Z/candidates/001_survey_candidate_0001/camera_lidar_attempt_00/inspection_observation.json) labelled **`unobservable`**, with `camera_relative_yaw_rad=null`. Its seven samples are advisory observations, not seven angle measurements.

[inspection_execution.py:103](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/candidate/inspection_execution.py:103) already prioritizes `move_opposite` when a validated `axis_observation_path` exists. That prerequisite never existed. The robot therefore selected generic diverse views: first nearly edge-on, then front-facing. No certified opposite-side branch was attempted. Direct opposite-side routing requires a candidate-associated, validated normal; visual backside appearance by itself cannot supply that route geometry.

## Why the third, front-facing view did not complete the candidate

![Second candidate at its third camera pose: the entire QR frontside is visible](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260914T123717Z/candidates/001_survey_candidate_0001/camera_lidar_attempt_02/perception_debug/latest_frame.png)

### The visible head was clipped out of the active decoder crop

The final nominal ROI is **x=293..516**, while the measured wider-search head bounds extend to approximately **x=429..537**. The nominal crop cuts off about 21 pixels of the right side, including part of the QR's right finder/border. Runtime `qr_detected` stayed false for all **128** distinct processed images. No QR identity was latched or committed for this candidate.

The wider 2D search found **107 complete-head proposals**, all rejected with `no_samples_in_accepted_range`; their bounds were clipped by **20.26–34.50 pixels** in the nominal crop. The final proposal had **0.995 raw-edge support**. However, candidate/LiDAR association rejected it, so [head_proposal_registration.py:151](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/head_proposal_registration.py:151) returned the failed nominal result before evaluating the recentered complete-head crop. This preserved a crop-and-association dependency: the pixels needed for QR recovery were found but did not reach the strict retry.

A local replay on the saved final image makes this dependency concrete. The recorded proposed recentered crop **[415,209]..[552,392]**, using its own raw head corners and adjusted intrinsics, decoded **`Start`** and produced a usable independent measured-head angle of **+20.245°**, with **1.382 px reprojection RMSE**, **1.830° yaw uncertainty**, and an accepted **2 px head/neck gap**. Current positive QR/marker evidence prevented backside classification. The nominal crop did not decode; the large search crop alone also did not decode. This uses local OpenCV 4.13.0 and demonstrates a viable complete-head path on this saved image, not a runtime decode or seven-sample hardware success. `Start` passes the current case-preserving station-ID syntax; authoritative logistics mapping is outside this experiment's completion evidence.

### The LiDAR range window uses the wrong origin

This is a concrete code defect, not a reason to widen the tolerance. [observer/node.py:1163](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/node.py:1163) transforms the candidate into `base_scan` for bearing, but lines 1182–1192 construct the accepted scan-range window using distance from **`base_footprint`**. The recorded scanner is **32 mm behind `base_footprint`**.

For this stopped front view:

| Quantity | Recorded/current calculation | Same-frame calculation |
| --- | ---: | ---: |
| Candidate centre distance | 0.448573 m from robot base | 0.480568 m from scanner |
| Accepted LiDAR range | 0.268573–0.488573 m | 0.300568–0.520568 m |
| Head-aligned scan returns | Approximately 0.503–0.509 m | Same recorded returns |

The current window rejects these returns as `no_samples_in_accepted_range`, typically only 14–20 mm beyond its upper limit. Using the candidate's existing exact-time `scan_from_map` point for both bearing and radial distance corrects the roughly 31.995 mm origin difference while preserving the **40 mm tolerance**, stand radius, uncertainty, clustering and uniqueness rules.

A [ROS-free range-origin replay](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/range_origin_replay.json) reproduced all **21** available recorded proposal-association rejections. Changing only the range origin made **20/21 uniquely associated**; one still failed `ambiguous_registered_camera_clusters`. Only one of those saved processed frames was fresh in the recorded run. This isolates the association defect; it does not establish seven fresh samples, a valid geometric fit, or completed exploration.

One runtime nominal-crop result was labelled backside at approximately **−55.54°** despite this being the front-view attempt. The current LiDAR gate blocked its admission. Its exact image was beyond the capture limit, so the logs cannot establish what rectangle generated it. This is another reason to preserve current complete-head QR/marker vetoes and outer-frame validation while fixing association; a source label or angle alone must not authorize the opposite-side route.

### Processing and TF also prevent accumulation

The newer QR scheduling substantially reduced repeated empty QR work. In this run's front view, median summed QR decode time was **12.2 ms**, while median detector duration was **380.5 ms**, input age at detector start **234.4 ms**, and result age **620.9 ms**. These are separate distributions, not a sum of medians. **105/128 results (82.0%) exceeded the 500 ms freshness limit**; all 23 fresh results were rejected by LiDAR association. All 128 tracker updates were also reported stale, so no tracked pose could rescue acquisition. Simply increasing the camera timeout repeats the same blocked pipeline.

Exact-time TF delivery was also materially worse in this view: **140 tuples entered retry and 118 exhausted** the 150 ms allowance. Initial future-extrapolation gaps had median **0.981 s** and maximum **2.504 s**. These are sensor timestamp minus latest available transform timestamp, not measurements of DDS latency. The recording cannot distinguish publication delay, transport delay or listener scheduling; per-transform receipt tracing is still needed. Existing successful-lookup snapshots do not provide that trace.

## Why the deadline aborted the mission instead of allowing bounded recovery

The parent stopped the last observer after its **90-second deadline**, sending **SIGINT**, which explains child return code **130**. This was deadline cleanup, not evidence of an independent crash or operator interruption.

The final-state classification is inconsistent with the accumulated-evidence policy:

| Actual observer event | Time UTC | State | Existing policy decision |
| --- | --- | --- | --- |
| Line 407 | 12:49:31.470201 | `obsolete_detector_result` | Candidate-local failure |
| Line 408 | 12:49:31.473380 | `stale_sensor_tuple` | Terminal failure |

Both snapshots have **23 prior LiDAR rejections**, valid accumulated evidence and `poisoned=false`. The final stale input has age 0.634152 s and `transient_tf_retry=false`. The actual loader and timeout-policy replay reproduce different decisions just **3.18 ms apart**.

[timeout_policy.py:47](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/timeout_policy.py:47) handles trailing TF states and obsolete detector results using prior processed evidence, but omits `stale_sensor_tuple`. Consequently [runtime.py:1579](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/autonomous_runner/runtime.py:1579) raises a generic `RuntimeError`. The inspection and candidate loops catch only the typed candidate-local error, so this escapes to mission failure. The eight-view budget was not exhausted. Making this deadline candidate-local would allow the existing bounded retry/defer policy; it would not mark this candidate complete or necessarily move immediately to the next candidate.

The final `inspection_progress.json` still reports two views because the terminal exception bypassed recording the third. Route `_inspection_002` was a **dry-run rejection** at the proposed 0.50 m standoff (−0.049976 m uncertainty margin); `_inspection_003` executed the admissible 0.45 m alternative to that same third camera view. The rejected dry route was recovered and did not terminate the mission.

## Recommended modular correction

1. **Unify scan geometry.** Extract one pure helper that computes candidate bearing and range bounds from the same exact-time scan-frame point. Replay these saved scans with unchanged tolerances, including the still-ambiguous negative case.
2. **Separate head acquisition, geometric quality and side classification.** A valid four-border proposal should be available for candidate association and crop placement even when the short neck is inconclusive. Expose four-border localization separately from body/neck verification, and correct the raw neck/junction measurement wherever final physical-head or backside admission still requires it. Preserve an explicit outer-frame verification strategy, pose ambiguity checks and a separate front/back evidence contract. Removing only one neck call leaves downstream gates blocking the same recording; deleting all neck/junction gates is not a validated correction.
3. **Evaluate the complete candidate head before side commitment.** Recenter once using the associated current proposal, preserve crop-adjusted intrinsics, and perform current QR/finder checks and strict model fitting on that complete crop. Budget the complete geometry/QR path, not only the decoder. A clipped nominal crop must not provide negative-QR evidence sufficient to commit a backside normal.
4. **Classify and persist failed observations consistently.** Handle a trailing stale tuple through the same explicit valid, unpoisoned accumulated-evidence policy; retain stale-frame rejection, crash/identity-conflict handling and all fresh motion gates. Persist every attempted view even when a terminal exception occurs. Add the two real terminal snapshots as regression evidence.
5. **Trace execution-listener timing.** Record transform source stamp, local receipt time, lookup request time and listener progress so the observed 0.98-second TF gap can be attributed. Do not substitute latest TF for exact-time geometry or increase freshness limits to admit these frames.

The next acceptance test should require the **first stopped backside view → seven fresh, uniquely associated validated backside samples → certified opposite-side branch → complete-head QR and facing-pose recommendation → next candidate**. The latest recordings must also remain negative for ambiguous LiDAR association, incomplete QR borders and stale observations. A correction to only the timeout or only the crop is insufficient to establish that milestone.

## Reproducible evidence

- [Original-source manifest and SHA256 hashes](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/remote_source_manifest.json).
- [Per-image run analysis script](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/audit_run.py) and [derived run statistics](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/audit_run.json).
- [Original mission failure](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260914T123717Z/mission_failure.json) and [candidate goal progress](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260914T123717Z/candidate_goal_progress.json).
- [Backside proposal/neck diagnostic script](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/derived/backside_first_view/trace_proposals.py) and [trace](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/derived/backside_first_view/trace_proposals.json).
- [Range-origin counterfactual script](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/range_origin_replay.py) and [results](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/range_origin_replay.json).

- [Front-view runtime analysis](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/derived/front_view_review/front_view_runtime_audit.json), [local replay script](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/derived/front_view_review/front_view_local_replay.py), and [local replay results](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/derived/front_view_review/front_view_local_replay.json).
- [Mission progression and view numbering](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/mission_progression_audit.json).
- [Backside raw-edge equivalence check](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/derived/backside_first_view/edge_equivalence.json).

- [Actual terminal-state policy replay script](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/timeout_policy_replay.py) and [asserted results](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/timeout_policy_replay.json).

Validation included replaying the production status parser and timeout classifier on both actual final snapshots, replaying saved scan association with only the coordinate origin changed, and the bounded image diagnostics above. The existing 29 observer diagnostics/TF retry tests passed; they do not cover the reproduced trailing-stale-tuple policy gap. All 841 original file hashes were rechecked after analysis with no mismatches.

Audit only. Production code and original run evidence remain unchanged. Local image replays are diagnostic and use OpenCV 4.13.0; they are not reported as hardware validation of the recorded OpenCV 4.5.4 pipeline.
