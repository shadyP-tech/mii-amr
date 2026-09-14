# Stable current-head acquisition and independent confidence — 2026-09-14

This correction keeps the measured 3D head model and its existing pose-quality gates. It improves acquisition and scan association, then adds independent repeated backside evidence and a temporal veto over unstable head choices. Neck validation has no role. Implementation and validation are local; no workstation transfer, deployment or robot motion occurred.

## Assessment of the suggestions

The four main priorities are supported by the saved `stand_explore_exact2_camera_all5_20260914T123717Z` evidence. The scan diagnosis needs one correction: although two scans have questionable endpoint metadata, none of the three ambiguous target matches crosses a scan endpoint. All three split at an internal missing return.

| Finding | Evidence and implemented response |
| --- | --- |
| Six head-proposal misses | Their long-segment endpoints miss the true top/bottom by 12–22 pixels, beyond the unchanged 8-pixel refinement corridor. Independent four-line intersections and joint current-gradient/border-support ranking now locate complete hypotheses before strict refinement. |
| Three ambiguous scan matches | Frames 012/020 contain fragments `[3,4]` and `[6]` with beam 5 missing; frame 025 contains `[3]` and `[5,6]` with beam 4 missing. A narrow stopped-pose history can recognize witnessed fragmentation. It never joins scan ends or fills missing returns. |
| Backside versus angle confidence | Raw complete-head appearance is now recorded independently of successful planar-pose admission. Repeated appearance can be supported while orientation remains unresolved. |
| Stopped-pose angle instability | The earlier corrected replay spans about 9.15 degrees while corresponding corners move less than four pixels. Six stable angles plus one outlier can pass the old maximum-deviation-from-mean check. A full-window axial-span and border-displacement veto now prevents that combination from producing a receipt. |

Uncertain-angle route planning remains a later change. The implementation records the interval containing observed and independently plausible current axes; it is **not a calibrated confidence interval** and grants no planning authority. The opposite-side branch still requires the existing fresh, valid axis receipt and an admitted collision-free route. No pose-quality, freshness, raw-border, QR or route-clearance threshold was lowered.

## Module boundaries

- `perception/stand_axis/joint_head_borders.py` ranks closed four-rail hypotheses using current Canny support and image gradients. Projection and earlier detection can locate the bounded search; neither supplies fitted pixels or an angle.
- `perception/stand_axis/head_proposal.py` compares current candidate rectangles, retains competing-border diagnostics and performs at most 12 unchanged strict raw-border/corner refinements. The measured-head solver still owns every accepted angle.
- `observer/scan_target_persistence.py` requires three distinct, recent, uniquely observed connecting scans before recognizing two current fragments separated by one missing internal beam. It retains the raw two-cluster count and actual source indices. Candidate geometry, poses, clock freshness and the connecting returns are rechecked from persisted evidence.
- `perception/stand_axis/head_backside_appearance.py` evaluates raw head appearance and explicit current QR/finder evidence independently of angle confidence. `observer/head_observation_confidence.py` combines only fresh, associated, stationary updates with complete-head crop evidence. QR absence alone cannot qualify.
- `observer/head_temporal_consistency.py` compares all retained current choices modulo 180 degrees and compares corners in full-image coordinates normalized against the original candidate projection. `head_observation_window.py` adapts current raw-head/IPPE evidence to this additional veto.
- `observer/evidence.py` invokes a typed veto after shared synchronization, freshness, duplicate, pose and QR-identity checks. It can remove contradicted angle samples while retaining the independent QR channel. It cannot replace an angle, identity, pose or timestamp.

Complete-head crop validation is separate from marker absence. This matters for an ambiguous **front** pose: its current competing angles must also invalidate earlier head angles, even though it cannot supply backside evidence. Both cases are covered by observer-level regressions.

## Temporal behavior and receipts

The default temporal window uses the existing seven samples, five-second TTL and eight-degree configured limit as a maximum full-window axial span. It additionally limits projection-normalized corresponding-corner displacement to 0.04 head heights. An incompatible choice remains in the rolling window until time or capacity eviction. While it remains incompatible, new angles are withheld and earlier measured-head angle buckets are cleared. Recovery admits only the new current measurement; retained historical images are never replayed into the seven-sample receipt.

Independently plausible current IPPE solutions are retained when they have positive depth, at most two pixels reprojection error and lie within the existing 0.10-pixel residual gap. These alternatives can veto old angle evidence even if the current pose is rejected. They cannot promote that pose to an accepted axis.

Witnessed scan fragmentation uses a distinct schema-4 receipt path. Legacy schema-2/3 receipts retain their validation. The new path records three real historical scans, the current scan and explicit raw ambiguity; receipt loading recomputes the geometric proof. Contradictory current evidence clears the scan witnesses. Stale evidence is rejected at the resolver and again at ordinary observer admission/publication.

## Validation interpretation

The original 17 processed images reference 15 distinct scan stamps. Seven images passed the original observer's source-admission checks, but those seven span 7.798 seconds. At most six coexist within the unchanged five-second axis window. Therefore even perfect geometry on every originally admitted saved frame cannot demonstrate seven-sample live consensus for that recording.

Recorded replay and original timing are reported separately. New local pixel fits do not retroactively make an originally stale frame fresh, and replay never writes a new motion receipt. Synthetic observer tests separately exercise seven fresh samples, current scan fragmentation, receipt validation and opposite-side selection. They also reject six old good angles followed by an angle jump, changed border, ambiguous backside pose or ambiguous front pose.

The complete recorded front decodes `Start`, and its current finder pixels veto backside evidence even when payload decoding is removed. The clipped front also remains negative. Saved-image processing uses local Python 3.12/OpenCV 4.13; the recorded robot runtime used OpenCV 4.5.4. Target timing and physical angle accuracy still require hardware evidence.

## Final verified results

The source-pinned replay locates **17/17 current head proposals**, including all six previous proposal misses. It registers 16, obtains 15 usable measured-head angles, and passes complete-crop/backside checks for **14/17**, compared with 5/17 in the preceding replay. These are per-image geometry results before temporal and live freshness admission.

The remaining rejections are explicit: frame 017 lacks three recent independent scan witnesses, frame 027 fails strict raw corner support, and frame 028 contains an unverified QR-like detection that prevents marker-absence evidence. No gate was relaxed to admit these frames.

The all-pixel diagnostic detects the frame-012 angle/border outlier and withholds seven incompatible window updates. Later compatible windows permit only their new current measurements; old images are not backfilled into axis evidence. The historical-admission lane contains five consistent complete-backside measurements and cannot produce a seven-sample receipt. Observer-level synthetic tests cover successful seven-fresh-sample receipt creation and opposite-side branch selection, including a fragmented final scan, plus contradictory-angle and front-marker rejection.

Median local QR-plus-selection processing is **330.6 ms**, excluding image decompression/rectification. This is additional bounded acquisition work, not a demonstrated latency improvement; hardware throughput remains an outstanding validation requirement. The replay does not emulate the runtime QR scheduling budget.

The final regression run covers **124 modules: 1,326 passed, 6 failed, 1 skipped, and 1,106 passing subtests**. The six failures are the same `test_stand_axis_image.py` failures reproduced on clean commit `1cf07084d030e8d26f86f1c1481a5e5803c8af21` under the same local runtime; there are no new failures. All 59 changed/new Python files parse using Python 3.10 grammar, and `git diff --check` passes. SHA-256 checks confirm all 841 original run artifacts are unchanged and every consumed replay source matches the final implementation.

- [Recorded sequence report and per-frame results](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/correction_validation/stable_acquisition/STABLE_ACQUISITION_VALIDATION_20260914T142414Z.md)
- [Validation manifest, baseline failures and source hashes](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/correction_validation/stable_acquisition/validation_summary.json)
- [Final regression output](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/correction_validation/stable_acquisition/final_focused_tests.log)
