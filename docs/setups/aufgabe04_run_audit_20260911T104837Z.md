**Audit: first-candidate camera geometry, 11 September 2026**

The first inspection point already shows the complete stand head and decodes `QR_003`. The new proposal acquisition and recentered crop work. Completion fails at two later stages: most decoded payloads have no usable, identity-bound QR corners; the one fresh frame with those corners fails the joint QR/head fit. The inspection policy then treats the unresolved front view as sufficient evidence to move to another angle. This explains the user's observation without assuming the first view was physically unsuitable.

Run `stand_explore_exact2_camera_all5_20260911T104837Z` ran on `mii001`, from **10:48:37–11:00:24 UTC / 12:48:37–13:00:24 CEST**, using clean revision `41d25b94d66b1a9485002e9663d504d306eb53cc`. Both LiDAR survey stops completed; five candidates were retained. Only `survey_candidate_0003`, directory `000_survey_candidate_0003`, received camera inspections. The mission recorded **zero confirmed QR identities and no completed facing recommendation**. Raw decoded text and confirmed candidate identity are different counters.

This was a read-only audit of production code and the robot. It used copied run artifacts, deterministic policy/model replays, and a replay of saved images inside the deployed container with the host home and checkout mounted read-only. No ROS node or motion was started by the audit.

**What happened at the first view**

The observer's recorded status interval was **10:53:20.095–10:53:36.533 UTC**, about **16.44 seconds**, at an admitted candidate range of **0.726 m**.

| Evidence | Recorded result | Consequence |
| --- | --- | --- |
| Processed detector results | 38 | Full selected-model metadata exists for all 38. |
| Fresh results | 12; all expose `QR_003` | Text detection works in this view. |
| Obsolete results | 26/38, or 68.4% | These cannot contribute current evidence. |
| Fresh text without usable QR corners | 11/12 | `decoded_qr_geometry_unavailable`; no candidate-bound identity sample. |
| Fresh text with bound QR corners | Frame 000024 only | One QR identity sample, followed by joint geometry rejection. |
| QR latch | Peak 1 sample; requires 2; five-second lifetime | No confirmed identity; the single sample later expires. |
| Axis consensus | Peak 0 of 7 required samples | No accepted facing angle. |
| Cross-crop identity conflicts / poisoned evidence | 0 | Conflict handling is not the cause in this view. |
| Selected estimator reason | 37 `model_qr_text_without_geometry`, 1 `reprojection_error_too_high` | Missing symbol geometry dominates; the sole joint fit also fails. |

Head proposals were found in all 38 processed frames. Twenty-six passed neutral candidate association and triggered `current_head_proposal_strict_retry`; twelve were rejected because the registered camera cone contained ambiguous LiDAR clusters. Those twelve are real association losses, but they do not explain the remaining associated frames' missing QR corners or failed joint fit.

Frame 000024's recentered crop is **x=292..376, y=263..382**. Its recorded head bounds are approximately **x=301.61..365.97, y=272.66..348.24**, so the complete detected head fits inside the crop. The proposal corrects the projected center by about **33.12 pixels / 0.446 head heights** and passes raw-border, neck and unique LiDAR association checks. The associated LiDAR distance is **0.730 m**. The image-specific framing range is **0.7195 m**; this and the admitted range above are separate recorded measurements.

The [annotated frame](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z/derived/first_view_geometry_frame24.png) overlays recorded corners and the rejected model on the original image after rectification using recorded camera intrinsics. It does not introduce a new head detection.

**Why decoded text usually cannot seed the model**

The deployed runtime is **OpenCV 4.5.4, NumPy 1.21.5, Python 3.10**. The parent run log contains **451 missing-QUIRC warnings**. Its native OpenCV decoder can locate quadrilaterals but cannot perform native payload decoding. The WeChat fallback decodes `QR_003`, but the sampled raw WeChat point arrays describe the entire input image rather than the QR symbol.

For example, frame 000011's nominal 135×135 crop returns corners `(0,0), (134,0), (134,134), (0,134)`. Enlarged, bordered variants restore to points outside the original crop. The adapter correctly rejects these as symbol corners and preserves text-only evidence. Treating a whole-crop rectangle as QR geometry would corrupt pose and candidate binding.

The deployed-image probe reproduces the production adapter outcomes:

| Saved input | Replayed decoder outcome |
| --- | --- |
| Frame 000011, nominal and recentered crops | `QR_003`, no usable corners |
| Frame 000024, nominal crop | `QR_003`, no usable corners |
| Frame 000024, recentered crop | `QR_003` with `opencv_quad_wechat_rectified` corners exactly matching the run |

The existing isolated native-quad/WeChat helper is already attempted through the multi-quad path. It fails on the sampled frame 000011 variants and succeeds on frame 000024's enlarged recentered crop. There is also a code coverage gap: native single-quad results skip this helper. However, explicitly trying the helper on those single quads still fails for frame 000011. **Adding the missing single-quad branch alone is not a demonstrated fix for this run.**

[QR decoding](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/qr_scanning/opencv_qr_detector.py:26) retains identity provenance. [QR pose seeding](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/qr_pose_seed.py:216) does not attach a decoded identity to an unrelated quadrilateral. Consequently, [the model pipeline](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/model_pipeline.py:185) cannot seed a joint front fit from a neutral head proposal plus text alone, and returns `model_qr_text_without_geometry` before raw head refinement. This is the main explanation for “visible head, model not applied” in the first view.

**Why the one fully bound frame still fails**

Replaying frame 000024's exact recorded head proposal and QR corners reproduces the recorded rejection:

| Fit | Reprojection RMSE | Camera-relative diagnostic yaw |
| --- | --- | --- |
| Head alone | 0.4568 px | 37.815° |
| QR alone | 0.3360 px | 34.104° |
| Joint QR + head | **3.2308 px**, above **2 px** | Rejected; no accepted axis |

The joint residual affects both groups: head RMSE **3.1280 px**, QR RMSE **3.3303 px**, maximum corner error **4.7991 px**. Separate diagnostic fits do not establish a correct physical angle or justify bypassing the joint check.

The recorded head-to-QR width/height ratios are **1.0794 / 1.1300**. The configured head/symbol ratio is **78/62 = 1.2581**. The [physical profile](/Users/stephpark/Documents/stephsWorld/mii-amr/configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json) specifies a measured 78 mm head and 71 mm paper panel, but its 62 mm QR symbol boundary is an image-derived estimate. For comparison, head/paper is 78/71 = 1.0986.

A diagnostic sweep keeps all recorded image points and the 78 mm head fixed, changing only synthetic QR object-point dimensions. At 62 mm it reproduces **3.2308 px**; at 71 mm it gives **1.4605 px**. The sampled minimum is near 70 mm. This supports investigating the symbol/head dimensional contract and corner semantics. **It is not metrology and does not establish that the real QR symbol is 71 mm.** Head-versus-paper border selection, symbol dimensions, QR corner bias and calibration remain possible contributors. The profile should not be changed merely to make this saved image pass.

Even the synthetic 70–71 mm fits have diagnostic yaw approximately **36.13–36.15°**, exceeding the current [35° QR-bound axis-admission cap](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/axis_sample_policy.py:19). Passing reprojection alone would therefore not demonstrate an admissible facing angle. This is a counterfactual downstream limit; the recorded joint fit failed before reaching that conditioning decision.

Local OpenCV 4.13 behaves differently: it can decode native QR corners on the saved images, and its automatic proposal for frame 000024 is slightly wider. Its resulting joint fits still fail. The exact recorded-corner replay establishes the recorded geometric decision; the deployed OpenCV 4.5.4 probe establishes the sampled runtime decoder behavior. These are distinct validations.

**Why the robot left before the 90-second timeout**

The first observer exited normally with an advisory artifact: `returncode=0`, `completion_kind=artifact`, `deadline_expired=false`. It was not timed out. The final advisory is `front_readable`, with **8 accepted stationary observation tuples spanning 9.298 seconds**, but `qr_id=null`, no angle and `completion_authorized=false`.

[Inspection progress](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/inspection_progress.py:27) can classify text-bearing evidence as `front_readable`. Its advisory threshold is at least seven fresh accepted tuples spanning two seconds, accumulated within fifteen seconds. This threshold establishes a description of an unresolved view, not a confirmed identity or pose. The one new bound QR sample initially restarts the advisory window; subsequent geometry failures accumulate until another advisory can be committed. `--camera-timeout-sec 90` is a maximum duration, not a minimum dwell.

A framing recovery hint exists from the failed joint fit. However, [distance recovery](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/candidate/camera_distance_recovery.py:51) requires at least **0.726 + 0.10 = 0.826 m**, while capping recovery at the preferred approach distance **0.70 m**. Therefore it offers no outward pose. [Inspection execution](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/candidate/inspection_execution.py:70) falls through to the generic angular policy, which selects **+45°** for this front-readable view without an angle. Production policy replay reproduces the recorded route choice.

There is no dedicated stopped recovery phase here for “candidate-associated front marker decoded, but own-corner binding or joint geometry unresolved.” That policy gap explains the unnecessary-looking relocation. The evidence supports preserving this promising view for a bounded recovery attempt; it does not prove that simply waiting longer with the same decoder/model would succeed.

**Latency further reduces the chance of collecting evidence**

For the first view, measured median image age at receipt is **36.4 ms**, in-process wait before detector start **318.4 ms**, detector duration **317.9 ms**, and completed image age **595.7 ms**. Median summed QR decoding work per frame is **282.9 ms**; neutral head acquisition is **33.9 ms**. These are separate medians, not additive components of one representative frame.

Twenty-six of 38 completed results exceed the **500 ms** result-age limit. The input often arrives fresh; substantial age accrues after receipt and during processing. The before-detector wait may include scheduling, synchronization and exact-time TF readiness. These measurements do not isolate DDS transport as the cause. Optimize obsolete-work avoidance and QR passes while retaining freshness checks.

**Later views and the terminal failure**

The +45° view runs for about **89.32 seconds**, decodes no QR and accepts no associated observation frames. Of 188 processed frames, 187 neutral proposals fail the LiDAR range gate; one proposal is ambiguous. In the final sample, the nearest return is approximately 0.749 m versus a 0.7309 m upper bound. This relocation does not improve candidate evidence.

After all eight +90° standoff proposals fail route admission, a −90° option completes, including a successfully recovered localization stop. Its camera observation lasts **26.49 seconds** and yields an oblique advisory of about **74.14°**, without QR identity or accepted axis consensus. Subsequent route search rejects eight angle-guided proposals, then two −45° standoffs.

The terminal proposal, index 20 at 0.60 m, records `RCLError: failed to shutdown: rcl_shutdown already called on the given context`. Motion was not authorized for that proposal. The parent command exits 120. The copied logs do not establish the initiating shutdown cause or a user interrupt; cleanup may mask an earlier exception. This terminal error is separate from the first-view decoder/model failure. There are 21 proposals in the append-only ledger, while the last progress snapshot records only 18; the ledger supplies the terminal chronology. Three camera views occurred, and the remaining four candidates were not inspected.

**Recommended corrections and validation**

1. Make QR identity-to-own-corner recovery reliable on the deployed backend. Record decoder capability, raw point provenance and corner rejection reasons. Improve bounded isolated-quad rectification/quiet-zone recovery and verify the same payload in the isolated symbol; cover both native single and multi results. Test frames 000011 and 000024 plus multi-symbol/conflict fixtures. Never substitute whole-input corners or blindly borrow another quad's identity.
2. Independently measure the actual printed symbol, paper and head boundaries. Validate which boundary each detector returns, then correct the model or association if warranted. Replay synchronized captures across distances and angles with independent angle ground truth. Preserve the joint-fit and ambiguity gates.
3. Add a bounded stopped recovery state for a promising candidate-associated front view, with separate reasons for missing QR corners and incompatible joint geometry. Exhaust that state before generic angular exploration. Give outward recovery an explicit validated camera envelope instead of treating the preferred 0.70 m approach distance as an absolute recovery ceiling; request motion only through the existing route, localization and arrival checks.
4. Reduce in-process waiting and repeated QR work, with measured fresh accepted samples per second on the robot. Do not extend freshness lifetimes to hide processing delay.
5. Preserve the initiating exception and make ROS-context cleanup tolerate an already-shut-down context. Investigate the terminal failure independently of camera perception.

The next validation should first reproduce repeated candidate-bound QR samples and accepted joint geometry on these stored images in the actual deployed runtime, then demonstrate one stopped first-view identity/facing completion on hardware. This audit does not implement or hardware-validate those corrections.

**Evidence and reproducibility**

The [audit evidence directory](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z) contains 863 copied source files, 58,846,093 bytes, and a [SHA-256 manifest](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z/source_manifest.json). The source archive SHA-256 is `519c5fd5c62e578c453fca304aab87785ba168439a9410ac9d98f75b3bbea1d4`. Replays verify that all copied source bytes remain unchanged; this is not a separately captured upstream whole-tree hash comparison. Audit outputs are local, under the ignored `results/audits` tree.

The `derived` directory contains executable Python analyses and corresponding JSON:

- [camera_metrics.py](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z/derived/camera_metrics.py): event counts, timing and source-integrity checks. Third-view detailed model metrics cover 40/45 frames because five statuses omit that payload; first-view coverage is complete.
- [inspection_decision_audit.py](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z/derived/inspection_decision_audit.py): production policy replay, observer process outcomes, route ledger and terminal failure.
- [qr_evidence_audit.py](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z/derived/qr_evidence_audit.py): identity/corner/latch audit and deployed-probe findings.
- [deployed_qr_probe.py](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z/derived/deployed_qr_probe.py) and [recorded output](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z/derived/deployed_qr_probe.json): saved-image OpenCV 4.5.4 replay, raw decoder returns and the single-quad counterfactual. These reproduce sampled outcomes; they are not a trace of every unlogged decoder call during the original run.
- [first_view_geometry_replay.py](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z/derived/first_view_geometry_replay.py): exact recorded-corner rejection, text-only early return, native-decoder comparison and diagnostic size sweep. The JSON records OpenCV 4.13.0 / NumPy 2.5.3 / Python 3.12.14 for the local replay.
- [first_view_geometry_plot.py](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z/derived/first_view_geometry_plot.py): annotated original pixels with source/output hashes and recorded-camera rectification provenance.

Production code, robot configuration and original artifacts were unchanged. Validation consisted of the evidence checks and offline/deployed saved-image replays above; no claim of a new production test-suite or successful robot experiment is made.
