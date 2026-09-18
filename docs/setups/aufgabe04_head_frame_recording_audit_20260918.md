# Head-frame recording audit — 18 September 2026

The latest workstation recording contains **0 detected head frames in 257 frames**. Distance and measured head dimensions are already used, but most frames stop before visual acquisition because the nearest scan target is ambiguous. For the remaining frames, permissive geometric bounds and competing borders prevent automatic selection. Tightening the size filter or adding a projected volume mask alone did not solve this recording.

## Evidence and scope

- Recording: `recording_20260918_153656_468438163`, 257 source images, 13.131 seconds between first and last source timestamps. Confirmed still the newest recording at audit completion.
- Workstation: `mii001`; checkout `e0cbf5ebbbc1a99de3b5ced09c4d2c3f5d96a2d0`, clean at audit completion.
- Offline replay used the workstation Apptainer runtime: Python 3.10.12, OpenCV 4.5.4, NumPy 1.21.5. The repository was mounted read-only; source-image, metadata, model-profile and perception-code hashes were unchanged across replay.
- All 257 metadata records were counted. All 17 frames with an unambiguous recorded scan target were replayed. Frames 0, 128 and 256 additionally tested a **hypothetical** combined target. Five source images were inspected/sample-tested; this was not manual annotation of every image.
- Replay retained recorded calibration, rectification, source support and edge settings. It disabled live deadlines, QR geometry and tracking. Timings therefore describe offline work, not end-to-end live performance.
- No production code, deployment or robot commands were changed by this audit. Concurrent local navigation work is outside its scope.

Compact results: [audit JSON](/Users/stephpark/Documents/stephsWorld/mii-amr/docs/setups/aufgabe04_head_frame_recording_audit_20260918.json).
Full scripts, outputs, hash manifests, metadata and five copied images are in the ignored local [audit directory](/Users/stephpark/Documents/stephsWorld/mii-amr/results/aufgabe04/debug_audits/recording_20260918_153656_468438163).

## What failed in the recording

| Recorded outcome | Frames | Interpretation |
| --- | ---: | --- |
| `nearest_head_scan_candidates_ambiguous` | 240 | Stops before visual head acquisition |
| `head_proposal_ambiguous` | 8 | Multiple competing current-image proposals |
| `model_current_head_border_unavailable` | 9 | All nine report acquisition verification-budget exhaustion |
| Border diagnostic reports detected | 0 | No successful automatic head-frame display |

### 1. Candidate selection is the dominant blocker

All 240 ambiguous records contain a competing pair involving low scan indices beginning at zero and high indices at least 200. Frame 0 has indices `[218,219]` at 0.4055 m and `[0,1,2,3]` at 0.4091 m. Their projected centres are approximately `(428,282)` and `(347,284)` around the visible foreground head. Their distance difference is below the viewer's 3 cm ambiguity threshold.

This is consistent with a stand split across the scan-array boundary. It is **not proven**: the recording omits raw ranges, scan length and angular topology metadata needed to validate that merge. Nearby equal-distance objects must remain distinguishable.

In [nearest_scan_head.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/nearest_scan_head.py:48), the detector call does not forward `angle_max_rad` or a validated `scan_topology_profile`. The underlying detector supports conservative circular clustering, but defaults to linear topology. Preserve raw scans and topology in future recordings, propagate the actual metadata, and merge across the boundary only when angular continuity and physical endpoint separation justify it. Do this before stand-width filtering and nearest-candidate comparison.

### 2. Real dimensions are used, but the filter is broad

The profile specifies a 78 × 78 mm head and a 71 × 71 mm inset panel. Median optical depth for the 17 unique-target frames is 0.3636 m, with vertical focal length 641.16 px. The simple frontal scale estimate is:

`height_px ≈ fy × head_height_m / optical_depth_m ≈ 641.16 × 0.078 / 0.3636 ≈ 138 px`

The calibrated tilted-camera computation reports a median expected height of 136.56 px. This uses **camera optical depth**, not the approximately 0.405 m base-frame range shown for the candidate.

However, median accepted measured-height bounds are **75.15–244.77 px**. Rough proposals receive additional refinement allowance. The centre tolerance is approximately 95 px, and none of these recorded searches has a LiDAR edge region. The size-dependent search bounds are also broad; they do not represent four narrow predicted border bands.

[metric_head_search.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/metric_head_search.py:30) conservatively propagates dimension, depth, yaw and tilt uncertainty. For nonzero camera tilt, its independently bounded numerator and denominator lose correlation and widen the interval substantially. A diagnostic correlated bound narrowed frame 3 to roughly 106–177 px, but did not resolve its border ambiguity. This prototype needs mathematical and pose-range validation before production use.

Known size and distance cannot identify every pixel's depth. Window and radiator edges inside the projected uncertainty region can remain, and inset borders have nearly the same apparent size. Unknown yaw also means a physically square head need not appear square. A projected search region should restrict proposals; current image evidence must still select the measured borders.

### 3. Different edges of one head become competing heads

In frame 3, five proposals pass raw refinement around the same visible head, approximately `x=304–442, y=198–334`. Four also have accepted physical-frame evidence; one retains unresolved physical-boundary evidence. Several differ by only a few pixels, mixing outer, inset and neighbouring rim edges. Final selection is `distinct_current_heads_ambiguous`.

The present border-family rules do not sufficiently consolidate these alternatives. Distance alone is unlikely to distinguish them: 2 cm depth uncertainty at this distance changes expected scale by approximately 7–8 px, comparable to the rim separation. The 78 mm outer versus 71 mm inner dimensions imply roughly 6 px per side at frontal scale, making the **relationship between the boundaries** more informative than independent rectangle sizes.

Frame 4 also contains an accepted background rectangle around `x=195–312, y=215–360`. Consequently, suppressing all overlapping rectangles, choosing the largest rectangle, or relaxing ambiguity globally would be unjustified.

The next border-selection change should jointly compare the four outer sides, expected inset separation, enclosure, current gradient evidence and corner support. Group alternatives only when evidence shows they are different measurements of the same physical frame. Preserve separate-object and unresolved-boundary rejection.

### 4. Masking reduces pixels but can increase search work

On frame 3, an experimental projected head-volume mask reduced contour edge pixels from **43,879 to 14,396** (67.2%) and rejected 507 of 1,057 line segments. Yet considered proposals increased from **12 to 92**, with 37 unverified independent alternatives left after the verification quota.

Filtering occurs before the per-direction rail quotas. This changes which local rails fill the quotas, admitting many nearby QR/rim combinations. Fewer edge pixels therefore do not establish either lower runtime or better detection. Proposal generation needs to rank and consolidate physical border alternatives before spending the verification budget.

The recorded colour-mask option also does not mean the measured-head path receives HSV exclusions: its explicit scope is `preview_and_legacy_only`, avoiding removal of desaturated physical head borders.

## Offline experiments

Results below cover all **17 recorded unique-target frames**, with original acquisition quotas and no live deadline. All variants returned no automatic head proposal suitable for final border detection; usable yaw was also zero.

| Experiment | Ambiguous | Border unavailable | Median detector time |
| --- | ---: | ---: | ---: |
| Recorded prior | 8 | 9 | 139 ms |
| Tighter height interval | 10 | 7 | 145 ms |
| Projected volume mask | 0 | 17 | 242 ms |
| Tighter interval + mask | 0 | 17 | 226 ms |
| Guided rail search first + tighter interval | 0 | 17 | 457 ms |
| Guided rail search first + tighter interval + mask | 0 | 17 | 420 ms |

The recorded-prior replay exactly reproduced the 8/9 reason split. Unavailable outcomes above exhausted the verification budget. No variant demonstrated an improvement sufficient to deploy.

For the three ambiguous-target samples, weighted projected candidate centres were combined **conditionally**, without claiming a validated scan merge. Original acquisition then verified a head border in frame 256, but remained ambiguous in frames 0 and 128. Tighter size gave the same result. None admitted reliable yaw. This is diagnostic evidence that target selection matters, not a measured automatic success rate for a scan-merging fix.

### Manual-location diagnostic: borders versus orientation

For frames 0, 3, 4, 128 and 256, an approximate quadrilateral read from the image was supplied to the existing raw-border refinement. All five returned `head_frame_detected=true` through the actual `head_frame_detection` helper, with raw-border support approximately 98.4–99.5%. All five returned `yaw_reliable=false` due to pose rejection or planar-axis ambiguity.

This test establishes that usable head-border pixels exist and that the current refinement can verify them when location/selection is supplied. It is **not automatic detection**, does not validate an inferred target merge, and does not authorize pose or motion. Keep detected-border display and reliable orientation separate, as the current diagnostic API already does.

## Optimization order and acceptance evidence

1. **Resolve scan association from valid topology.** Record raw scans and actual angular metadata; propagate topology to clustering. Test true boundary fragments, two independent nearby stands, partial scans, invalid angular metadata and stale/synchronization failures.
2. **Use calibrated physical geometry earlier and more precisely.** Preserve projection correlations; derive uncertainty-aware centre and side bands from the candidate, floor-relative head height and full camera transform. Test across yaw, tilt, distance and calibration error rather than selecting fixed limits for this image. Retain original pixels for final verification.
3. **Select one physical outer frame before expensive fitting.** Use measured outer/inset relationships and current raw evidence to consolidate rim alternatives, while retaining distinct-background hypotheses. Change quota allocation alongside masking; do not merely raise the budget or choose the closest nominal size.
4. **Measure freshness and orientation separately.** Report detected borders, false border selections, admitted yaw, verification counts and detector latency independently. Replay this clip plus clutter-only, multiple-stand, oblique, blurred and partially occluded negatives before calling the change successful.

There is also a timing limitation: 61/257 recorded results were marked too old. Median header-to-receipt age was 226 ms, median detector-start age 233 ms, and unique-target detector work 150 ms. Clock offset was not measured, so these ages do not identify transport as the sole cause. Unlimited-time replay still fails acquisition, showing that freshness is an additional issue rather than the explanation for the border failures. Improving acquisition should precede increasing live-age tolerances.

## Reproduction artifacts

The local audit directory contains `audit_replay.py`/`replay.json`, `audit_guided_replay.py`/`guided_replay.json`, and `audit_border_replay.py`/`border_replay.json`. The last explicitly evaluates `head_frame_detection` separately from `estimate.usable`. `summary.json` contains the metadata census; the compact checked-in JSON adds the supplemental border/yaw outcomes.

Scripts were passed on standard input to `python3 -` inside the workstation's `ros2_humble_native.sif`, using `--cleanenv --containall`, the repository bind-mounted as `/audit:ro`, working directory `/audit`, `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH=/audit`. No perception files were edited to run the experimental variants. Each replay output includes its scope, runtime and unchanged-input check.

Metadata SHA-256: `7a66f237d5aea056f97fa9a1b9b0107bdebffeb1fcaed3c87771b23789b61631`.
