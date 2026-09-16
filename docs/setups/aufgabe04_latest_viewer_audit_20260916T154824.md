# Latest workstation camera recording audit — 16 September 2026, 15:48 CEST

The latest recording is `recording_20260916_154824_828837944`, with 50 distinct saved frames from **15:48:24.529 to 15:48:29.875 CEST**. The foreground stand is clearly visible and calibration/model inputs are ready in every frame. **No recorded frame reaches the 3D fitter. All 50 exhaust the source-age deadline first.**

Offline replay exposes a second, independent failure: automatic acquisition cannot select a unique physical head border even with no wall-clock deadline. Supplying a suitable manual search location allows the unchanged current-pixel fitter to accept 49/50 images, and the returned magenta border closely follows the visible head perimeter. This is strong evidence that the fitting/refinement stage works when given a suitable location; automatic acquisition is failing to supply that location reliably.

Interpretation corrected after reviewing the user's observation: the earlier description of the magenta result as an incorrect inset border was too strong. A small visual offset and the absence of a recovered larger enclosure do not establish that the printed panel was selected instead of the frame. The recording lacks independent physical-angle ground truth, but that does not negate the useful 49/50 fitting result.

The latest failure must therefore not be described simply as the earlier selected-panel/fitted-frame binding rejection. That rejection is not reached in this recording. The immediate live blocker is timing; the independently reproduced geometry blocker is incomplete/ambiguous border selection.

## Evidence and source state

- Workstation SSH alias `mii001`, reported hostname `mii0002`.
- Original recording directory: `/home/amr_gruppe01/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr/results/aufgabe04/stand_axis_debug_recordings/recording_20260916_154824_828837944`.
- Workstation checkout observed clean at `aac5510b3fa5aa63b1e089330491ac1bcec3fa6c` before and after replay. The same recording remained latest at the final check. The metadata does not itself contain an immutable runtime-code commit identifier; code explanations and replay use the observed workstation revision.
- The local in-progress camera-context, scan-preview and physical-border changes were not deployed into this clean workstation checkout by this audit.
- Model: `physical_stand_measured_20260826_v2`, measured head 0.078 × 0.078 m, printed panel 0.071 × 0.071 m. SHA256 `56fe19dcbfc8aa58682ea460e702a499c65cc719423940e6a892ca581e6d0b5f` matches the recording.
- Original 800×600 PNGs were rectified using the recording's calibration before replay. Viewer `channel-union` was normalized to the estimator's `channel_union`.
- Replay ran in the existing ROS container: Python 3.10.12, OpenCV 4.5.4, NumPy 1.21.5. The repository and recordings were mounted read-only; Python bytecode writes were disabled. No ROS node, robot motion, configuration change or deployment was performed.
- Metadata and three representative PNGs were copied locally and checked against workstation SHA256 values. Replay verifies unchanged hashes of all 50 source PNGs, metadata, model and the inspected perception source files.

Reproducible scripts, raw metadata and derived results are in [the audit evidence directory](../../results/aufgabe04/debug_audits/recording_20260916_154824_828837944/). Principal evidence is [timing summary](../../results/aufgabe04/debug_audits/recording_20260916_154824_828837944/latest_timing_summary.json), [replay results](../../results/aufgabe04/debug_audits/recording_20260916_154824_828837944/replay_results.json), [geometry isolation](../../results/aufgabe04/debug_audits/recording_20260916_154824_828837944/geometry_isolation.json) and [proposal summary](../../results/aufgabe04/debug_audits/recording_20260916_154824_828837944/geometry_summary.json).

## Why the live overlay never appears

The viewer computes its work deadline as the earlier of source timestamp + 250 ms and callback receipt + 180 ms. It passes that deadline into the physical-head pipeline. The source-age limit wins for every frame in this recording.

| Recorded timing | Median | Range |
| --- | ---: | ---: |
| Apparent header/source age at callback receipt | 229.8 ms | 197.9–249.5 ms |
| Remaining source-age budget at receipt | 20.2 ms | 0.5–52.1 ms |
| Callback receipt to detector start | 30.7 ms | 10.2–105.4 ms |
| Remaining work budget at detector start | −6.0 ms | −104.5 to +40.2 ms |
| Source timestamp to detector completion | 261.4 ms | 250.2–360.7 ms |
| Callback receipt to detector completion | 41.4 ms | 16.6–111.6 ms |

Medians are calculated independently and need not add exactly. All local receipt-to-completion durations satisfy the separate 180 ms limit. The short detector duration is early cancellation, not a fast successful fit.

Of 50 frames, 32 are already over deadline when detector work starts. Another six expire before the physical-head pipeline passes its entry gate after preprocessing. The remaining twelve expire during early cold acquisition: four at Hough detection, two at rail merging, two at rail processing, three at contour-hypothesis processing and one at line detection. **Zero reach strict border verification; zero have an `independent_head_fit` stage or a model pose.**

Every model result reports `head_acquisition_deadline_exceeded`. Every tracker update reports `pose_observation_stale`, and every tracker prediction reports `no_tracked_pose`. Consequently, the viewer cannot enter its faster tracking path.

Freshness takes priority in the overlay policy, so the screen reports `observation_too_old`/`obsolete_detector_result`, obscuring the more specific acquisition cancellation. Model and calibration remain enabled and ready in all 50 frames; QR work is disabled and negligible here.

Between the first and last diagnostic snapshots, 140 messages arrive, 116 pass the arrival gate, and 24 are rejected as already old. These are interval differences; the much larger cumulative counters belong to the process before recording began.

Code: [deadline](../../scripts/aufgabe04/perception/debug/viewer_frame_timing.py#L38), [deadline supplied to fitter](../../scripts/aufgabe04/perception/debug/stand_axis_viewer.py#L3365), [early physical-head gate](../../scripts/aufgabe04/perception/stand_axis/physical_head_pipeline.py#L105), [receipt gate and latest-image slot](../../scripts/aufgabe04/perception/debug/color_mask_viewer.py#L220).

### What the timing evidence cannot establish

Header age compares clocks on the source and receiver; `header_receipt_clock_offset_measured` is false throughout. Capture/encoding queues, transport, callback scheduling and clock offset are not separated by this recording. **A 230 ms Wi-Fi/network delay is not established.** The viewer already uses depth-one sensor QoS and an overwritten latest-image slot, so a viewer FIFO is not demonstrated either.

The earlier successful 15:36 recording has identical options but only 31.4 ms median apparent header-to-receipt age, 94.8 ms median source-to-completion age, and an established tracker for all 53 saved frames. This latest clip differs materially in delivered timestamp age as well as tracking state.

## What happens when timing is removed

Saved-image experiments omit the wall-clock deadline, live freshness and candidate/scan association. They retain production geometric quality gates and comparison limits unless explicitly marked otherwise. Marker/QR work is disabled.

| Diagnostic experiment | Outcome | Median processing |
| --- | --- | ---: |
| Original full-image cold acquisition | 0/50 selected/fitted heads; all comparison-budget failures | 171.6 ms |
| Manual 200×200 head crop, cold acquisition | 0/50; all comparison-budget failures | 218.6 ms |
| Initial approximate manual outer-region location | 1/50 numerically usable fits | 11.8 ms |
| Same manual location scaled to 0.93 about its center | 49/50 accepted current-pixel fits from a manually supplied search location | 25.1 ms |
| Crop comparison cap raised from 12 to 256, frames 0/25/49 | 0/3 selected heads; all remain ambiguous after 97–106 checks | Not a production timing test |

The manual crop is `(290,170)–(490,370)`. The initial neutral corner hint is `(324,200), (456,203), (460,338), (329,340)` in the rectified full image. These locations are diagnostic inputs, not automatically acquired observations. Supplied hints are intentionally not marked as already verified/selected borders.

### Background competes for the comparison budget

The unprojected viewer searches the whole image. In the 50 full-image replays, 373 of 600 strict-check slots concern upper-background/window rectangles, 93 concern head-sized rectangles in the visible head region and 134 concern other/composite rectangles. These categories use documented spatial bounds for audit interpretation; they are not a replacement detector.

The head region is checked and produces an accepted raw refinement in 44/50 frames, so the failure is not simply that the head is never found. The locator still withholds selection because alternatives remain unresolved. Median unresolved hypotheses: 11; allowed raw verifications: 12. Frame 0 spends every slot outside the foreground head region.

### Cropping and raising the cap do not resolve physical-border identity

Within the head crop, nearby physical edges, printed-panel edges and QR texture yield numerous paired-rail combinations. Median distinct locator hypotheses increase from 48.5 full-image to 160 cropped; border families increase from 27 to 95. Median unresolved hypotheses rise to 73. Thus the demonstrated crop does not solve selection even after background removal.

For frames 0, 25 and 49, raising the cap completes 104, 106 and 97 strict checks, with 36, 34 and 27 raw rectangles accepted respectively. All comparisons still end in `head_proposal_ambiguous` / `distinct_current_heads_ambiguous`. More time or a larger numerical cap exposes ambiguity rather than selecting the physical frame.

Code at the replayed revision: `head_proposal_selection.py:42–58` ranks larger supported rectangles and interleaves border families; `head_cold_acquisition.py:333–354` verifies the first twelve with a fixed four-pixel corridor; `head_cold_acquisition.py:396–406` rejects uncovered alternatives before final selection. Local ongoing edits may move these lines.

## What the successful manual-location replay establishes

Changing the manual location scale by only 7% changes accepted fits from 1/50 to 49/50. The latter return yaws 17.58–19.32° and reprojection errors 0.30–1.12 px. The manual input supplies only an approximate search location. Current image edges determine the returned corners, and the unchanged measured-head solver determines the pose. No corners or angle are manually installed as the final result, and no geometric acceptance threshold is relaxed.

In the image below, the magenta returned border closely follows the visible head frame. It is slightly inside the outermost visible perimeter, but this image alone does not establish that it is the wrong physical border. All 49 accepted fits have accepted model quality and `outer_border_verified=true`. Their `recovered=false` means that the initial refined border remained selected instead of being replaced by a larger enclosing alternative. `independently_resolved=false` means that no independent larger-enclosure recovery was established; it is not a rejection flag or proof of incorrect frame selection.

These results provide a concrete target for automatic acquisition: supply a suitable location and reproduce this same current-pixel refinement. They do not establish unassisted cold-start success, live freshness, candidate association or ground-truth angle accuracy. Those are separate validation questions, not evidence that the displayed fit is bad.

![Offline proposal competition and manual-location sensitivity](../../results/aufgabe04/debug_audits/recording_20260916_154824_828837944/diagnostic_panel.png)

Left: accepted raw 2D rectangles among frame 0's first twelve offline checks, in yellow. Right: blue initial hint, cyan 0.93-scaled hint, magenta fitted border. This figure is explicitly diagnostic and is not a live accepted model overlay.

## Corrections indicated by this audit

1. **Restore a usable freshness budget.** Measure camera/receiver clock offset, then isolate capture/encoding, publication, delivery and callback timing. The approximately 198 ms increase in median apparent arrival age relative to the successful clip consumes almost the entire allowed source age. Reducing the measured local preparation delay can help but does not explain or remove the upstream/timestamp change.
2. **Make automatic acquisition reach the demonstrated successful refinement.** Treat initial image rectangles as location hints, run the same current-pixel frame refinement that succeeds in the manual-location replay, then compare/group the resulting measured borders and recheck target association before binding a selection. Preserve genuine competing-head ambiguity. A valid existing border need not be rejected merely because refinement did not recover a larger enclosing rectangle.
3. **Reduce redundant border comparisons without hiding alternatives.** Candidate position/scale can bound mission search, but the cropped replay shows that head/panel/QR rail combinations still need correct grouping and physical-frame resolution. Increasing the cap or accepting the first low-error pose is not supported by these experiments.
4. **Validate cold start and tracking loss on this exact recording after correction.** Verify selected physical corners against the enclosing frame, geometric consistency, completed comparison and end-to-end timing. Then verify fresh live acquisition and tracking. Manual hints and deadline-free replay are isolation tests, not mission-admission evidence.

Production code and robot state were not changed during this audit. Existing local implementation work remains separate from these recorded findings.
