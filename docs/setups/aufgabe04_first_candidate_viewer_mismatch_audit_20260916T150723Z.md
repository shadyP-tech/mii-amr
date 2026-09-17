# First-candidate admission versus debug viewer — 2026-09-16

The latest inspected run, `stand_explore_exact2_camera_all5_20260916T150723Z`,
reached its first candidate but never selected a head for the 3D fitter. It also
decoded `QR_003` three times, but rejected all three candidate bindings. The
observer consequently published `unobservable`, with neither an angle-based
recommendation nor the QR-only observation-pose fallback.

The debug recording demonstrates working **current-pixel tracking and fitting**.
It does not demonstrate automatic acquisition or candidate admission. Saved-pixel
replay isolates this difference: the unchanged fitter accepts all ten mission
images when given a diagnostic location hint, while the recorded candidate-crop
acquisition accepts none.

## Sources and scope

- Mission started **17:07:23 CEST**, clean commit
  `ae5f39a6cdfff56055ca5a5396ad97350c072a36`; local and remote checkout agreed.
- First candidate: `survey_candidate_0003`, directory
  `candidates/000_survey_candidate_0003/camera_lidar_attempt_00`.
- Viewer: `recording_20260916_171413_400553048`, **17:14:13.284–17:14:17.370 CEST**.
- Sources were read on mii001. Original images, scans and recordings remained
  there. Only derived numerical diagnostics returned. Replay used the deployed
  code in an Apptainer container with a read-only repository mount, OpenCV 4.5.4
  and NumPy 1.21.5. No production edits, ROS nodes or robot movements occurred.
- All 60 image/metadata files hashed by geometry replay were unchanged afterward.

The earlier shared physical-frame correction is deployed and active. Diagnostics
contain `physical_frame_refinement: true` and measured-frame alias collapsing.
This is not another run of the old selected-border/refinement mismatch.

## What happened at the first pose

The approach completed at **17:12:45.820 CEST**, after 0.246 m of measured travel.
The observer received 64 images, produced 13 synchronized tuples and processed
10 TF-ready images. All ten passed its current freshness and preliminary LiDAR
association checks. Earlier TF retries delayed processing, but did not reject
those ten usable input tuples.

All ten failed automatic head acquisition before a selected head reached the
3D solver:

- **Nine** exhausted the cold head verification budget.
- **One** reached the processing deadline in `cold_texture_topology`.
- Non-timeout frames considered 36–51 proposals and 20–38 border families. Their
  12 verification slots were consumed while independent alternatives remained.
- Individual raw-border checks often succeeded. These were not completed head
  selections: untested competing hypotheses still prevented selection.

The expanded search crop was approximately `(135,76)–(592,533)` in an 800×600
image. The viewer's measured head lies approximately within x=252–357,
y=228–338, so that crop contains the complete head. Its projected center was
about 60 pixels to the right of the observed head. Neither image clipping nor
a neck-validation failure explains this run.

Median acquisition cost was **250.6 ms**; median complete detector cost was
**256.7 ms**. Median source-to-completion age was **326.0 ms**, inside the
observer's 500 ms limit. Median header-to-receipt age was **25.0 ms**. Expensive
acquisition matters, but the recorded ten results were not discarded as stale.

## QR decoding worked; candidate binding failed

Capture frames **4, 7 and 10** decoded `QR_003` with valid decoder-owned corners.
Each probe took approximately 7–8 ms. The other seven frames deferred the
periodic identity search. There was no accepted identity because each decoded
observation returned `camera_bearing_outside_map_cone`.

| Capture | Camera bearing | Projected candidate bearing | Difference |
| --- | ---: | ---: | ---: |
| 4 | 7.912° | 1.863° | 6.049° |
| 7 | 7.917° | 1.860° | 6.057° |
| 10 | 7.917° | 1.847° | 6.070° |

[QR target binding](../../scripts/aufgabe04/real_robot/observer/qr_target_binding.py#L71)
uses camera-led association only after head registration succeeds. Otherwise,
the [map-cone path](../../scripts/aufgabe04/perception/candidate_lidar_association.py#L184)
rejects a QR outside the **3°** cone before examining scan returns. Head
registration, by contrast, permits up to **12°** camera-to-map bearing correction
while retaining a narrow scan search and unique-cluster requirement.

That makes the fallback unnecessarily dependent on head registration: its
purpose is to retain a useful QR observation when angle computation fails, but
it cannot use the camera-led correction available to the head path.

A counterfactual replay through the existing camera-led association function,
using each original scan and recorded completion time, found **one eligible
three-sample cluster on all three frames**, at **0.539, 0.538 and 0.545 m**.
Scan ages were 393, 287 and 328 ms. No fragmentation recovery was needed.
This isolates the narrow map-cone rejection; it is not an end-to-end admission
test or a justification for accepting a neighboring stand. The ordinary path
reproduced the original rejection on all three scans.

The projected/observed bearing mismatch is real. These frames do not establish
whether its origin is previous localization movement, the surveyed candidate
center, or calibration. The stopped observation did not record a motion-epoch
reset; an AMCL jump during these ten frames is not established.

## Why observation ended early

The three decoded-but-unbound frames could not start the existing front-view
recovery because it requires bound QR evidence or a registered head. They were
excluded from advisory accumulation, but did not clear the earlier blank samples.
The remaining seven samples accumulated over **2.523 seconds** and published
`classification: unobservable` at approximately **17:12:56.961 CEST**.

The relevant sequence is
[front-view evidence classification](../../scripts/aufgabe04/real_robot/observer/front_view_recovery.py#L33),
[observer accumulation](../../scripts/aufgabe04/real_robot/observer/node.py#L2754),
and the [seven-frame/two-second advisory gate](../../scripts/aufgabe04/real_robot/observer/inspection_progress.py#L165).
The configured 90 seconds is an upper timeout, not a minimum observation period.
The observer returned a normal advisory artifact; it did not time out.

The next local-inspection route subsequently failed preflight at
**17:13:11.812 CEST** with
`route uncertainty budget exhausted ... remaining_margin=-0.025453 m`.
It published no motion. This is a separate downstream route-admission failure.
The inspected parent bundle lacks a final wrapper exit record, so the exact
terminal process outcome is not asserted here.

## Why the debug viewer looks reliable

All **39/39** recorded viewer frames have fresh accepted current geometry, with
angles **25.57–25.85°**, median **25.70°**. Median model time was **21.2 ms**;
median source-to-completion age was **87.8 ms**.

Every saved frame starts from `tracked_head_search`, with a previous pose hint;
recording begins at source sequence 159. The viewer searches the full image and
has identity decoding disabled. Its purple-overlay contract validates geometry,
not a named candidate's QR/LiDAR admission. Exploration starts from the projected
candidate crop and retains a tracking hint only after head association succeeds.
In this run that state was never reached.

The supplied viewer command confirms this distinction. `--head-hold-sec 0`
disables holding old display observations; it does **not** disable the separate
`MetricPoseTracker`. With the physical profile, that tracker is created with a
250 ms prediction lifetime and a 2 s search-hint lifetime. `--median-window 1`
sets the display ratio/proxy windows, not the model tracker's memory. Each
accepted tracked angle still comes from new image borders. `--calibrated-handoff`
enables calibrated viewer diagnostics; it does not install the mission's named
candidate admission controller. `--no-qr-decode` explicitly removes identity
decoding from this viewer invocation.

Both paths ultimately call `acquire_cold_head_proposal`. The viewer's
`fit_physical_head_in_frame` passes a full image without a candidate projection
on cold startup; exploration's `acquire_viewer_candidate_head` supplies a crop,
projected center/height, candidate-association filter and reserved fit deadline.
Those inputs change the edge/contour hypotheses and their comparison. They
share lower-level code, but do not yet provide identical acquisition behavior.

The two paths share the fitter, but enter it differently:

| Read-only saved-pixel experiment | Accepted geometry |
| --- | ---: |
| Viewer images, recorded tracking hints | 39/39 |
| Viewer images, independent full-image cold acquisition each frame | 6/39 |
| Viewer images, mission crop/projection, no hint | 0/39 |
| Mission images, recorded crop/projection, no hint | 0/10 |
| Mission images, independent full-image cold acquisition | 1/10 |
| Mission images, later viewer location hint, diagnostic only | 10/10 |

These experiments disable QR work and omit live candidate association and
wall-clock deadlines, while retaining cold comparison limits. They isolate
geometry; they are not live admissions. The later viewer hint is deliberately
noncausal and must never become mission evidence. On mission pixels it yields
current fitted angles of 25.46–25.57° and median processing of 11.4 ms.

Removing the crop or raising the time budget alone is insufficient: 32/39
independent full-image viewer cold starts are ambiguous. Likewise, fitting all
70 individually verified mission seeds diagnostically produces 61 accepted
fits spanning **24.12–34.79°**. On capture 4 alone, accepted alternatives span
24.45–34.05°. A convincing overlay or low reprojection error therefore does not
identify which competing physical boundary is correct. No ground-truth angle
measurement is available in this audit.

## Recommended modular correction and acceptance tests

1. **Remove the QR/head-registration dependency.** Give one decoded QR with valid
   current corners an independent candidate-association path. Reuse calibrated
   bearing, bounded correction, current range and unique scan support. Confirm
   continuity with the surveyed target and reject plausible neighboring targets;
   do not simply widen every cone or authorize motion from QR text alone.
2. **Use decoded QR location to guide cold head acquisition.** Currently cold
   acquisition runs before the independent QR probe, and cannot consume its
   location. Reserve the periodic probe before expensive cold comparison, or
   carry a bounded stopped location hint to the next frame. QR should locate a
   search region; model/LiDAR should bound head scale and current outer borders
   should determine angle. Do not restore QR-size agreement, QR-derived angle,
   or selection of the first fit that passes. Keep the shared physical-frame
   refinement and same-image border proof.
3. **Add bounded stopped recovery for unresolved QR association.** A fresh single
   decoded candidate-bounded marker may delay an `unobservable` advisory without
   granting identity or angle authority. Clear the prior blank bucket once and
   use a fixed deadline; repeated detections must not renew it indefinitely.

Required checks: the three shifted QR frames bind only to their supported target;
wrong-neighbor/stale/multiple-symbol cases reject; cold starts and deliberate
track loss recover the same physical frame on these saved images; blank–unbound
QR–blank sequences do not immediately request movement. Validate live timing
and complete admission afterward. Keep route uncertainty rejection separate.

## Reproduction artifacts

Derived numerical evidence and read-only replay programs are in
`results/aufgabe04/debug_audits/stand_explore_exact2_camera_all5_20260916T150723Z/`:

- `run_diagnostics.json`, `collect_run_diagnostics.py`
- `geometry_replay.json`, `replay_geometry.py`
- `qr_scan_binding_replay.json`, `replay_qr_scan_binding.py`

Original mission files are under
`results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260916T150723Z/`
on mii001; the original viewer files remain under
`results/aufgabe04/stand_axis_debug_recordings/recording_20260916_171413_400553048/`.
