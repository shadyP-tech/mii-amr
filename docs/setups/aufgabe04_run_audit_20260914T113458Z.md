# Camera acquisition and deadline audit — 2026-09-14 11:34:58 UTC

Run `stand_explore_exact2_camera_all5_20260914T113458Z` ended with **1/5 distinct QR identities confirmed**. Navigation completed all six executed legs. The terminal failure occurred during the second camera observation of `survey_candidate_0001`: the acquisition path produced almost no fresh geometric evidence, then the observer deadline was reported through its last pending TF lookup.

The recorded run used clean commit `98fc4fa1cb78acb1ca7787467528fac37feaf7ba`, on bundle hostname `mii0002`, retrieved through `mii001`. It ran **13:34:58–13:46:07 CEST** / **11:34:58–11:46:07 UTC**, exiting 2. All **705 original files** were retrieved and verified against their source hashes, with no files changing during retrieval.

## What succeeded

Both LiDAR survey legs completed, with **2.0829 m** recorded translation and **95.31% modelled coverage**. The retained pool contains five hypotheses: one is now camera-confirmed, one remains pending camera, and three remain provisional/unvisited. Modelled coverage does not establish full camera visibility.

All six executed routes completed without a safety stop or localization reseal. Their odom execution certificates use schema 2 and `drift_reference.metric="certified_route_anchor_v1"`, with explicit map/odom anchors and the existing 0.03 m tracking tube. The route-anchor correction was deployed and exercised; the previous continuity-stop failure did not recur in this run.

The first inspected candidate, `survey_candidate_0003`, decoded **QR_003** and obtained **7/7 axis consensus samples in its first camera observation**, lasting approximately **3.50 seconds**. Its angle estimator was `model_current_measured_head`, with usable camera-relative angles of approximately 25.00–30.05°. This is the current-pixel measured 3D head model, not an angle inferred solely from a displayed projection.

Its extra arrival-alignment leg occurred **before** camera observation. Initial bearing error was 3.116°, just outside the 3° arrival gate; alignment reduced that to 0.572° at approximately 0.541 m. The robot did not leave an already successful QR observation to search another view. Its final 0.35 m facing route was validated for reachability, but was explicitly `facing_pose_validation_only` and was not executed.

## Camera evidence

Counts below use complete observer histories, deduplicated by processed image timestamp. The compressed capture histories are capped at 64 frames per attempt; that cap limits diagnostic recordings, not ongoing detection.

| Candidate / camera view | Processed images | Fresh results | Usable head fits | Axis consensus | Outcome |
| --- | ---: | ---: | ---: | --- | --- |
| 0003 / first view | 11 | 11 | 10 | 7/7 | QR_003 confirmed |
| 0001 / first view | 102 | 3 | 2 | 0/7, peak 0 | Local observation timeout |
| 0001 / second view | 105 | 0 | 0 | 0/7, peak 0 | Terminal observation timeout |

The second candidate's first saved view contains its complete, clearly visible backside. The nominal crop is approximately `x=320..499, y=216..395`; the visible head lies inside it. The expanded crop also contains the head. This recording does not support the earlier explanation that the right border was clipped out of the ROI.

![Second candidate, first view: complete backside visible](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T113458Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260914T113458Z/candidates/001_survey_candidate_0001/camera_lidar_attempt_00/perception_debug/latest_frame.png)

After this attempt failed, the planner selected a diverse local inspection view. The resulting image shows the stand nearly edge-on. No validated backside angle had been available to choose a directed opposite-face approach.

![Second candidate, fallback view: stand nearly edge-on](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T113458Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260914T113458Z/candidates/001_survey_candidate_0001/camera_lidar_attempt_01/perception_debug/latest_frame.png)

## Why fresh evidence was unavailable

### 1. Unsuccessful QR search and reacquisition exceed the timing budget

The execution observer uses a **500 ms source-age limit**. Across the second candidate's two views, **204/207 detector results were obsolete before admission**.

| Recorded median timing | Backside view | Edge-on view |
| --- | ---: | ---: |
| Image age at processing start | 69.8 ms | 86.9 ms |
| Detector elapsed | **742.6 ms** | **694.7 ms** |
| Image age at completion | **815.2 ms** | **785.8 ms** |
| QR decoding summed across ROI attempts | **420.4 ms** | **524.8 ms** |

Input age alone was not the dominant problem. Full QR decoding runs on the nominal crop and again on the expanded crop when no pose hint is available. These searches found no QR or verified marker on either view. Their total recorded time consumes 57.4% and 75.7%, respectively, of detector elapsed time. In the edge-on view, median QR decoding alone exceeds the entire 500 ms allowance.

The relevant order is in [observer/node.py:1263](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/node.py:1263): select full decoding when a pose hint is absent, decode before fitting the head, run the expanded acquisition path, then evaluate result freshness at line 1381. An obsolete result returns at line 1488 before candidate association, identity and axis evidence admission. This repeated unsuccessful path prevents useful evidence from accumulating.

Increasing the 90-second overall timeout repeats this same processing pattern. Increasing the freshness limit alone would also leave the geometry failures below unresolved.

### 2. The clear backside rarely reaches an accepted head fit

The first backside view records 99 `model_backside_head_and_neck_unavailable` results, one `head_model_centered_neck_unavailable`, and only two usable measured-head fits. The edge-on view records 97 `model_backside_head_and_neck_unavailable`, seven `model_backside_head_scale_mismatch`, and one `model_backside_neck_support_insufficient`.

The current implementation still has a neck-support dependency inside head acquisition. In [head_candidates.py:51](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/head_candidates.py:51), `_short_centered_neck_support` requires both post rails to occupy the same two fixed pixel columns for a consecutive run of at least 12% of head height, beginning close to the lower head edge. Although its docstring calls this validation-only, its rejection can discard an otherwise fitted outer head before the metric estimator receives it.

A ROS-free replay of the saved final backside image and original captures 000005/000010 reproduces outer-head candidates passing raw four-side refinement/support and then failing this neck gate at [head_candidates.py:140](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/head_candidates.py:140). For these roughly 100-pixel heads, the required run is 12–13 rows; the longest eligible fixed-column paired runs were 8, 5–6 and 7 rows respectively. Both nominal and wider crops exhibit this mechanism. In the final image alone, 17 nominal-crop and 25 wide-crop candidates reach this rejection. The independent head-proposal path also rejects proposals during refinement/support, so it does not rescue the acquisition.

A diagnostic straight-line corridor allowing ±1 pixel on the same unmodified raw edges finds 14, 28–29 and 21 paired rows. The probe retains the original gate result; this is evidence of sensitivity to rasterization and rail continuity, not an implemented acceptance-policy change.

**Replay limitation:** the local replay uses OpenCV 4.13.0, while the run records OpenCV 4.5.4. For the final nominal crop, regenerated Canny edges are pixel-identical to the runtime's saved edges: 179×179 pixels, 1,309 edge pixels, zero differences. The traces identify a reproducible structural bottleneck; they are not claimed to reproduce every internal proposal of the deployed OpenCV build. The real-run rejection counts and timings above come directly from its own saved histories.

### 3. Both otherwise usable fits fail LiDAR association

At 11:42:14.574 and 11:42:36.462 UTC, the observer obtained fresh measured-head fits with camera-relative yaws approximately −12.10° and −19.18°. Both passed head geometry quality and scale checks, but both failed `ambiguous_registered_camera_clusters`.

The fitted-head search cone contained three in-range scan samples split into two eligible clusters. Both scans report `inconsistent_scan_endpoint_metadata`, so circular adjacency is disabled even with `--scan-topology-profile full_rotation`. A seam-split stand is a credible explanation requiring raw-scan replay; these logs alone do not establish that the two clusters may safely be merged. The unique-target association check correctly withheld axis admission. The two rare fits are also about 22 seconds apart, exceeding the five-second axis evidence window.

## Why the terminal message names TF, and why the run stopped after two views

The last TF request was only **2.820 ms** newer than the latest available `map <- base_scan` transform, with one retry just begun. During that same second attempt, **105 tuples had already become TF-ready and reached detector processing**. Only three of 73 tuples that required TF retry exhausted their retry allowance. The last pending lookup does not establish a persistent TF outage.

Both unsuccessful observer processes reached their configured **90-second parent deadline**. Their process evidence records `deadline_expired=true`, `signals_sent=["SIGINT"]`, and return code 130. This exit code was generated by bounded parent cleanup; it is not evidence of operator Ctrl+C or a spontaneous crash.

The timeout policy distinguishes the attempts using fresh accumulated candidate evidence:

- First view: one accepted candidate frame, two LiDAR rejections and an explicit unpoisoned evidence snapshot. Its final `obsolete_detector_result` is classified as a candidate-local failure, allowing the existing inspection loop to select another admitted view.
- Second view: every detector result is stale, so no candidate evidence snapshot is created. The final `tf_pending_exact_time` therefore lacks the evidence required by [timeout_policy.py:35](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/timeout_policy.py:35), and the parent raises a generic terminal error at [runtime.py:1579](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/autonomous_runner/runtime.py:1579).

`--max-candidate-inspection-views 8` is an upper bound, not a requirement to continue through readiness failures. The second error escapes before the attempt is persisted in inspection progress, which explains `local_view_count=1` despite two actual camera attempts. The diagnostic should distinguish exhausted processing time from an ongoing TF outage without treating stale frames as usable observations.

## Corrections supported by this evidence

1. **Budget the entire no-QR acquisition path.** Obtain a bounded current-pixel head proposal/fit before repeatedly paying for full empty QR search pyramids. Avoid independent full decoding on nominal and expanded crops every frame; preserve prompt positive-marker vetoes and exact current-frame identity binding. Stop work that cannot finish inside the remaining source-age budget.
2. **Separate outer-head angle estimation from the neck cue.** Preserve valid raw-supported outer-head fits for the 3D estimator; use neck evidence for the specific association/classification checks that require it. If paired rails remain required, validate connected, perspective-following rails instead of identical fixed columns. Keep metric fit quality, target association and front/back ambiguity checks.
3. **Replay the actual scan-seam cases.** Resolve endpoint metadata and clustering against the recorded LDS-02 samples; do not simply admit multiple eligible clusters.
4. **Report attempt-wide failure causes.** Preserve typed TF readiness, detector budget, geometric acquisition and admitted-evidence counters separately, and persist every attempted view even on terminal exit. No retry or freshness limit needs to be increased to diagnose this run.

The next hardware milestone remains a fresh, candidate-associated backside angle at the first view, followed by an admitted opposite-face route and QR confirmation. This run establishes first-candidate QR/geometry success and successful execution with the route-anchor correction; it does not establish that backside pipeline milestone or five-station completion.

## Evidence and scope

- [Reproducible run analysis](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T113458Z/audit_run.py) and [derived JSON](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T113458Z/audit_run.json).
- [Verified retrieval manifest](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T113458Z/source_manifest.json).
- [Backside replay trace](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T113458Z/derived/backside_review/trace_geometry.json) and [replay script](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T113458Z/derived/backside_review/trace_geometry.py).
- [Runtime/replay raw-edge comparison](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T113458Z/derived/backside_review/raw_edges_comparison.json).
- [Terminal mission failure](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T113458Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260914T113458Z/mission_failure.json).

Audit only. No production changes, remote writes, ROS execution, new certificates or robot motion were performed. Original run evidence remains unchanged.
