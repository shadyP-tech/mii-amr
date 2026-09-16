# Latest viewer head-acquisition audit — 2026-09-16 14:11 CEST

The latest workstation recording at audit start was
`recording_20260916_141146_741705309`, not the earlier successful 13:51 clip.
It contains 20 distinct saved frames from approximately 12:11:46–12:11:51 UTC.
The remote checkout was clean at `a9aac95e743e17c3ceb671280b4e2be8de0b81fd`.

**The model is enabled and calibrated, but automatic acquisition never supplies
it a selected head. All 20 frames stop before the 3D fitting stage. Display
freshness independently rejects every result.** The tested bounded crops and
higher work limits do not solve this recording. Sharing the current cold locator
with exploration does not by itself establish reliable acquisition.

The visible foreground QR decodes as `Start` in diagnostic crop replay. This
recording must not be described as another observation of `QR_003` from the
earlier mission audit. Its decoded identity does not establish association to a
mission LiDAR candidate.

## Evidence and method

Original recordings remained on mii001 under
`/home/amr_gruppe01/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr/results/aufgabe04/stand_axis_debug_recordings/`.
The audit read metadata and replayed original PNGs in the existing Python 3.10.12,
OpenCV 4.5.4, NumPy 1.21.5 container. The repository was mounted read-only;
Python bytecode writes were disabled. No ROS node, robot motion, deployment or
remote source edit was performed. Only derived JSON and a diagnostic image
panel returned. Both replay passes verified that all 20 source PNGs and metadata
retained their hashes.

Saved calibration rectified every source image before analysis. The model hash
was checked against the recording:
`56fe19dcbfc8aa58682ea460e702a499c65cc719423940e6a892ca581e6d0b5f`.
Viewer CLI `channel-union` was normalized to the estimator's `channel_union`.

Derived evidence and reproducible scripts are in
[`results/aufgabe04/debug_audits/recording_20260916_141146_741705309`](../../results/aufgabe04/debug_audits/recording_20260916_141146_741705309):

- `recorded_summary.json`: recorded counters, settings and timings.
- `replay_results.json`: original-pipeline, bounded-crop and raised-budget comparisons.
- `isolation_results.json`: neutral search-seed isolation, crop comparisons and QR decoding.
- `inspect_recording.py`, `replay_recording.py`, `isolate_head.py`: scripts executed through stdin in the read-only container.
- `diagnostic_panel.png`: rectified frames 0 and 19 with recorded accepted **2D proposal** borders in yellow. These are not accepted 3D overlays.

![Recorded 2D proposal competition](../../results/aufgabe04/debug_audits/recording_20260916_141146_741705309/diagnostic_panel.png)

## Why the viewer has no 3D overlay

| Recorded result | Frames |
| --- | ---: |
| Model enabled, inputs ready, measured profile committable | 20/20 |
| Cold search, no tracked pose | 20/20 |
| Twelve-verification limit exhausted with unresolved alternatives | 19/20 |
| 1,024-contour limit exceeded | 1/20 |
| Head model pose or quality result produced | 0/20 |
| Overlay rejected as obsolete | 20/20 |

The acquisition reason is retained below the generic
`model_current_head_border_unavailable` message. There is no
`independent_head_fit` timing entry because the 3D fitter was never called.
`model_overlay.reason=observation_too_old` takes display precedence and can hide
the more useful underlying acquisition reason.

The unprojected viewer searches the whole 800×600 image. Room/window frames,
radiator rectangles and QR texture compete in the same comparison. Some frames
already verify a complete head-sized 2D rectangle within their first twelve
checks, but the selector still withholds it while other independent hypotheses
remain unresolved. An accepted raw-border proposal does not imply accepted
metric geometry.

The cold locator first ranks current corner-supported hypotheses by area and
round-robins border families. It performs at most twelve raw refinements, then
rejects if uncovered alternatives remain. Increasing that cap completes more
comparisons but exposes another failure: the final selection policy demands
that **all** other verified rectangles form small, disjoint, similarly sized
inset texture. Cross-paired QR/outer-frame rails generate tall or overlapping
internal rectangles which fail that condition.

For example, the tight crop of frame 0 yields 201 proposals, 24 border families,
and 99 hypotheses requiring verification. With a diagnostic raised cap, 21 raw
rectangles pass. The largest verified head appears at trial 25. Internal
alternatives include approximately 81×118, 38×118 and 40×82 pixel composites,
alongside three approximately 38×39 pixel finder boxes. These composites remain
classified as competing heads. Across the tight-crop sequence the raised-budget
experiment performs 95–149 strict 2D verifications per frame.

Relevant source: `head_cold_acquisition.py:191,248–272`,
`head_proposal_selection.py:43–67,70–128`, and
`physical_head_pipeline.py:135–141`.

## Timing is an independent blocker

| Timing | Recorded median |
| --- | ---: |
| Header/source to callback receipt | 244.35 ms |
| Receipt to detector start, including preparation | 33.93 ms |
| Cold head acquisition | 159.65 ms |
| Native QR marker processing | 62.47 ms |
| Detector processing | 225.54 ms |
| Source to detector completion | 491.40 ms |

The configured source-age limit is 250 ms and the receipt-relative result limit
is 180 ms. Source-to-completion spans 425.48–567.34 ms. All result and tracker
freshness checks fail. Median components above need not sum exactly because
their medians can occur on different frames.

`--no-qr-decode` disables the background identity worker, but the model pipeline
still runs native marker detection after failed physical-head acquisition.
The native fallback can even call `detectAndDecodeMulti`. This optional work
cannot rescue a missing physical head and consumes substantial time.

The source-age figure does not prove network delay or a viewer FIFO. The
receiver uses depth-one sensor QoS and an overwritten latest-image slot. Its
callback silently drops images already older than 250 ms, so recorded receipt
ages are a censored sample. Header age also assumes synchronized sender/receiver
wall clocks. Publisher buffering, clock offset, transport and callback scheduling
cannot be separated using this recording alone. In contrast, local processing
already violates the 180 ms receipt-relative budget independently of clock skew.

The viewer pose tracker additionally uses a 250 ms source-time expiry. Merely
relaxing display thresholds does not bootstrap tracking from these old results.
Relevant source: `color_mask_viewer.py:204–240`, `viewer_frame_timing.py:37–54`,
`pose_tracking.py:82–94`, `model_pipeline.py:178–196`, and `qr_pose_seed.py:163–191`.

## What saved-image replay establishes

All fits below retain the production geometric quality/ambiguity gates. Crops
and neutral seeds are manual diagnostic inputs, not automatically acquired
candidate observations. Geometry-only experiments omit marker/identity work.
Offline times exclude live delivery and publication delays.

| Experiment | Accepted strict 3D fits | Median processing |
| --- | ---: | ---: |
| Original full-image model pipeline | 0/20 | 226.05 ms |
| Wide diagnostic crop `[295,83,659,447]`, cold geometry | 0/20 | 69.87 ms |
| Tight diagnostic crop `[385,175,560,355]`, cold geometry | 0/20 | 93.25 ms |
| Wide crop with tracking available after a success | 0/20; never bootstraps | 69.84 ms |
| Full image, raised search limits, four representative frames | 0/4; ambiguous | 282.14 ms |
| Tight crop, raised search limits | 0/20; ambiguous | 369.82 ms |
| Manually located outer-region seed, current-pixel fitting | 20/20 | 19.55 ms |
| Nearby manually located inner-region seed, current-pixel fitting | 20/20 | 19.65 ms |

The original-pipeline replay reproduces the recorded 19 verification-limit
failures and one contour-limit failure exactly. Raised-budget experiments change
only in-memory constants to 256 verifications and 4,096 contours; no production
source or acceptance threshold was changed.

The successful seeded fits prove that the image contains fit-able head-region
borders and the common 3D solver can use them quickly. They do **not** prove
which physical border is correct. The outer-region seed gives 9.89°–15.95°;
the nearby inner-region seed gives 15.23°–15.35°. On frame 4 they differ by
approximately 5.41° while both pass the existing strict checks. Several-pixel
changes in the selected left border explain much of that difference. Neither
sequence has external angle ground truth. A numerically stable pose or low
reprojection residual must not settle an unresolved physical-border choice.

Separately, production QR decoding on the tight crop recovers `Start` on 20/20
frames, with validated decoder-linked corners on 11/20. Nine are text-only
WeChat results and cannot independently bind identity to this head. The median
decoder time is 62.55 ms. The container's native OpenCV decoder reports missing
QUIRC; existing WeChat/isolated-quad recovery supplies the successful identities.
This is a reason to preserve tested decoder provenance and valid corners, not
to substitute crop bounds as QR geometry.

## Targeted optimization plan

1. **Fix physical-head selection before simplifying its consumer further.**
   Give the shared acquisition path candidate-centered search bounds and the
   uncertainty-bounded head scale from calibrated camera geometry and measured
   candidate range. Apply those physical constraints before spending the
   verification budget. Within a candidate region, compare enclosing frame
   families explicitly and account for nested printed/composite rectangles;
   the current flat all-rectangles comparison fails even in a tight crop.
   Reuse verification only when hypotheses demonstrably share the same supporting
   raw rails/corners. Grouping rectangles by approximate size alone can both
   waste checks on duplicate texture contours and hide distinct physical edges.
   Preserve real separate-head and unresolved outer-border ambiguity. Do not
   select the first rectangle, accept the largest room rectangle, assume every
   contained object is texture, or use QR size/yaw to validate the angle.

2. **Resolve border identity, then track current pixels.** The same shared
   acquisition, border comparison and fitter should serve viewer and mission.
   A previous verified head only locates current borders. Preserve that search
   hint across bounded transient misses without renewing its source stamp or
   reusing its angle. Keep acquisition ambiguity across physical border families
   visible; the solver's local yaw uncertainty does not capture wrong-border
   selection. The frame-4 seed discrepancy is a required regression case.

3. **Separate geometry scheduling from optional QR work.** A geometry-only
   viewer should avoid the hidden marker/decoder pass. A failed physical fit
   should not synchronously spend its remaining geometry budget on optional
   marker diagnostics. Skipped marker work means unknown side, never proof of
   backside. In the mission, run bounded identity decoding independently of
   successful head acquisition and retain same-symbol corners/provenance. This
   lets usable QR evidence survive while geometry finishes.

4. **Combine independent evidence once.** Admit when one fresh accepted
   physical-head fit and one recent decoded QR with valid spatial binding refer
   to the same candidate in the unchanged stopped epoch. A decode dropout need
   not demand another seven paired successes. Expire the latch on motion,
   candidate/context change, conflicting identity or timeout. Text-only decoder
   output and a drawn proposal cannot meet that contract.

5. **Restore a workable end-to-end timing budget.** Instrument all callback
   arrivals, including age-rejected ones; measure publisher stamp-to-publication
   delay and workstation clock offset before assigning the 244 ms receipt age
   to a specific cause. Count preparation, geometry, decoding and publication
   separately. Reuse rectification maps for unchanged calibration and budget
   cold acquisition independently of fast tracked updates. Preserve source-time
   freshness rather than relabeling an old result as current.

6. **Validate association independently.** The previous mission's scan-seam
   ambiguity remains a separate blocker; this viewer recording does not contain
   the evidence needed to resolve it. Correct angular provenance or use a
   separately validated association method. Shared geometry does not authorize
   merging inconsistent scan endpoints. Localization and collision-free route
   admission stay outside the visual completion rule.

The tested higher search cap does not resolve selection, and enlarging age limits
cannot repair missing geometry. Merely routing exploration through the current
viewer cold locator does not establish a fix. The tested bounded crops reduce
scene clutter but still require the border-family selection correction.

## Current workspace work and acceptance evidence

Another concurrent set of uncommitted local changes already introduces
immediate front admission, a one-second bound-QR latch, two tolerated tracking
misses, and a bounded trial of the viewer cold locator before mission projected
search. Those files were not edited by this audit. They were not deployed in
the recorded workstation revision and were not certified by these replays.
The current cold-locator trial alone cannot claim to fix this clip: both tested
diagnostic crop sizes still fail automatic acquisition.

Before claiming reliable admission, validate the shared implementation on the
original frames with automatic cold start and no manual corners, after forced
track loss, with QR decoding enabled, and with actual candidate/scan binding.
Require correct enclosing borders as well as accepted angles; include the
earlier `QR_003` observations and negative cases with multiple stands, nested
rectangles, clipped frames, conflicting identity, motion and stale inputs.
Finally validate live source-to-publication timing and successful observation
completion. This audit establishes causes and optimization targets, not a
completed reliability fix or hardware admission success.
