# Latest camera viewer audit — 2026-09-17, 16:21 CEST

The front head is visible and can be fitted by the measured 3D model when given
an approximate location. Automatic acquisition has two independent blockers:
the live source-age deadline expires before border verification; without that
deadline, the full-image search exhausts its twelve verification slots while
comparing background rectangles. Canny preprocessing itself is not the dominant
cost. Raising the time limit alone does not acquire this head.

## Evidence and scope

- Latest recording, checked before and after the audit:
  `mii001:/home/amr_gruppe01/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr/results/aufgabe04/stand_axis_debug_recordings/recording_20260917_162111_735876885`.
- 50 frames, source timestamps 16:21:11.465–16:21:17.697 CEST.
- Workstation checkout clean at `43f81f15836c56b648df4112ac21774c0b23d631`, matching
  local source. Recording metadata does not bind an immutable running-code hash;
  code explanations and replay use the observed checkout.
- Calibrated 800×600 full images, measured 78×78 mm head profile, channel-union
  Canny 20/60, no dilation/closing, no QR decoding, no legacy fallback. All frames
  have model inputs ready. No target crop or candidate screen is supplied.
- Saved-image replay used the workstation's Python 3.10.12 / OpenCV 4.5.4 /
  NumPy 1.21.5 container. Repository and recordings were mounted read-only;
  bytecode writes were disabled. No ROS node, robot command or deployment ran.
- Metadata and one representative source PNG were retrieved for inspection;
  their hashes match the workstation originals. All 50 source PNGs, metadata,
  model and inspected perception source hashes were unchanged after replay.
- [Recorded summary](../../results/aufgabe04/debug_audits/recording_20260917_162111_735876885/recording_summary.json),
  [replay summary](../../results/aufgabe04/debug_audits/recording_20260917_162111_735876885/replay_summary.json),
  [full replay](../../results/aufgabe04/debug_audits/recording_20260917_162111_735876885/replay.json),
  [replay script](../../results/aufgabe04/debug_audits/recording_20260917_162111_735876885/replay.py).

## 1. Live timing prevents all fifty fits

Every recorded model result is `head_acquisition_deadline_exceeded`, every
display result is `obsolete_detector_result`, and every tracker update rejects
`pose_observation_stale`. No strict head-border verification or final 3D fit is
reached. The tracker reports `no_tracked_pose` throughout.

| Recorded quantity | Median | Range |
| --- | ---: | ---: |
| Header-to-receipt age | 236.15 ms | 199.26–249.95 ms |
| Receipt to detector start | 25.19 ms | 9.14–90.46 ms |
| Source age at detector start | 255.70 ms | 214.26–329.13 ms |
| Edge preprocessing | 4.57 ms | 1.25–7.43 ms |
| Model pipeline time, mostly early aborts | 5.05 ms | 1.33–36.88 ms |
| Source age at render | 267.01 ms | 251.19–334.86 ms |

The work deadline is the earlier of **receipt + 180 ms** and **source observation
+ 250 ms**. The latter binds in this recording. It is already expired for 35/50
frames at detector start. Four more expire before independent acquisition starts;
39/50 have no acquisition object. The remaining eleven stop during contour or
line generation/merging: five at `cold_hough`, three at `cold_rail_merge`, two at
`cold_contour_hypotheses`, one at `cold_rails`. All report zero raw verifications.
Even the freshest frame has only 35.74 ms left when detection starts.

Between the first and last metadata snapshots, 51 of 156 incoming messages are
rejected by the receipt-age guard. These callback counters are not the count of
recorded processed frames. Header age compares camera and workstation clocks;
metadata explicitly says their offset is **not measured**. The 236 ms cannot be
assigned entirely to Wi-Fi or capture latency without a clock-offset check.

The code still computes Canny before its first physical-head deadline check.
Skipping this work on expired inputs would save a few milliseconds but cannot
make those observations fresh. See
[deadline construction](../../scripts/aufgabe04/perception/debug/viewer_frame_timing.py),
[preprocessing order](../../scripts/aufgabe04/perception/stand_axis/model_pipeline.py),
and [physical acquisition](../../scripts/aufgabe04/perception/stand_axis/physical_head_pipeline.py).

## 2. Removing the deadline exposes proposal-budget exhaustion

All 50 original full images were replayed independently cold with their recorded
calibration, profile and edge options. Only the live deadline was omitted; no
candidate position, scan, QR seed, crop or retained pose was supplied.

Result: **0/50 usable**, all
`head_cold_acquisition_verification_budget_exceeded`, with exactly **12 checks
per frame**. Median complete geometry time is **124.00 ms**; median independent
acquisition is **122.81 ms**, while edge preprocessing is **1.17 ms**. Offline
timings exclude live delivery, scheduling, display and recording and should not
be interpreted as live throughput.

Background window/blind rectangles survive the loose four-side locator and some
strict border checks. The ranking sorts supported candidates by decreasing area
before distributing checks across border families. Larger background families
therefore spend the scarce checks before or alongside the foreground head.

- Frame 0 spends all twelve checks on upper-window rectangles; fourteen
  independent hypotheses remain unverified.
- Frame 25 reaches the visible foreground head on **check twelve**, and its raw
  refinement succeeds. Ten independent alternatives remain unverified, so the
  algorithm still refuses selection and never runs the final selected-head fit.
- Frame 49 again spends the checks on upper-window rectangles, leaving 23
  independent hypotheses unverified.

The refusal is explicit in
[head_cold_acquisition.py](../../scripts/aufgabe04/perception/stand_axis/head_cold_acquisition.py)
before `select_verified_head`. A successful raw rectangle check is not an
accepted stand pose. Increasing the check cap alone would not establish which
of several supported rectangles is the intended stand.

The latest early candidate/LiDAR filtering changes are deployed, but this viewer
call supplies neither `candidate_search` nor `proposal_filter`. Its recorded
`fixed` LiDAR mode and calibrated handoff do not inject mission candidate/scan
association into cold head acquisition. Thus the mission-specific filtering and
guided retry do not activate here. A timeout or unresolved alternative also
cannot trigger the guided retry.

## 3. The fitter can fit the visible head

As a separate diagnostic, frames 0, 25 and 49 received the same approximate
manually read foreground rectangle. They retained the complete image, original
edges, measured model and normal raw-border, corner and pose-quality checks.
All three returned usable measured-head fits, taking **12.2–19.8 ms** including
preprocessing, with reprojection RMSE **1.21–1.28 px**. This isolates the locator
from the fitter. It is not automatic acquisition, independent physical angle
ground truth, or live freshness validation.

## Recommended correction order

1. Measure camera/workstation clock offset and actual capture-to-receipt delay;
   reduce genuine image age and receipt-to-processing delay. Keep live evidence
   freshness separate from a diagnostic replay that deliberately omits it.
2. Give the viewer an explicit target-selection/association policy consistent
   with exploration, or improve unconditioned full-image proposal allocation.
   Preserve current-pixel border proof and unresolved-alternative rejection.
   This requires more than widening time or verification limits.
3. Add remaining acquisition budget and deadline stage to the visible status;
   reject already-expired inputs before Canny. This improves diagnosis and avoids
   wasted work, but does not fix the independent background-proposal failure.

This audit changes no production code or workstation configuration.
