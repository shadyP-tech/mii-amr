# Opposite-side arrival and admission audit — 22 September 2026

The opposite-face branch worked. Admission then failed because the stored
candidate geometry disagreed with the current stand direction, while the live
head detector exhausted every frame's processing budget. The final navigation
turn followed its commanded pose reasonably closely; successful navigation did
not establish that the physical stand was centered in the camera.

## Evidence and scope

- Workstation: `mii001`, clean revision
  `b1c42decc9d922ed2da4acd67e72d5177154641d` (new framing policy present).
- Latest run: `stand_explore_exact2_camera_all5_20260922T141529Z`.
- Second candidate: `001_survey_candidate_0001`.
- Opposite view: `camera_lidar_attempt_01` after `inspection_opposite_01`.
- Latest viewer: `recording_20260922_162737_262783118`, 49 recorded frames.
  The operator confirms this was taken at the same pose. The saved run image
  and viewer source image show matching foreground/background alignment.
- Read-only snapshot, provenance, analysis scripts and JSON results:
  `results/implementation_checks/opposite_side_audit_20260922T141529Z/`.
  Snapshot SHA-256:
  `9a4d2313aad3620dd7b156e226651621be7a6fa4452365cd8eab854a96098f17`.

No production changes, remote configuration changes or robot commands were made.
The complete observer event stream covers all 369 processed images. Capture
history retained 256 sensor tuples, including 238 processed images and 18
without detector results; it must not be mistaken for the whole observation.

## What completed, and what failed

The initial backside view accumulated seven accepted bounded backside samples
and published `backside_axis_committed_qr_unresolved` at 16:20:51 CEST. The new
policy therefore preserved enough geometry to trigger the opposite branch.

The run's opposite-route proposals used 0.50, 0.45 and 0.40 m standoffs. The first
two failed dry uncertainty admission without motion. The 0.40 m route passed,
traveled approximately 1.234 m and completed in 55.716 seconds at 16:22:42.
The stopped arrival check passed approximately five seconds later.

The front observation ran from 16:22:47.967 to its 90-second deadline, with no
accepted candidate frame, axis sample or bound QR identity. The parent stopped
the observer; return code 130 is part of that deadline termination, not evidence
of a spontaneous camera crash. It then rejected eight generic route proposals
and began another route attempt, interrupted by `KeyboardInterrupt` at
16:25:38. The saved run does not show all candidate views exhausted.

## Why arrival could pass with the stand visibly off-center

The route planner computes terminal yaw as `atan2(candidate - route endpoint)`
in `candidate_preapproach_compute.py:287`. Arrival admission compares the current
robot-base yaw with that stored candidate bearing in `approach.py:1225`.
It does not measure a current head center or prove image centering.
The receipt explicitly says `camera_centered: false`.

| Measurement | Recorded result |
| --- | --- |
| Planned map endpoint | x=-1.495, y=-0.415 m, yaw=+0.603° |
| Commanded odom endpoint using the execution certificate | x=1.811756, y=-0.023561 m, yaw=+7.036° |
| Arrival odom pose | x=1.816621, y=-0.000326 m, yaw=+5.375° |
| Error in the same odom frame | 2.374 cm; yaw -1.660° |
| Arrival bearing error to stored candidate | -1.290°, within the 3° check |
| Arrival distance to stored candidate | 0.36246 m |
| Predicted head center in saved processed frames | x=400.36–403.76 px; median 402.28 |
| Measured viewer head center in six fitted frames | x=573.73–573.89 px, in an 800 px image |
| Decoded QR bearing in the scan frame | -16.19° to -16.28° |
| Predicted candidate bearing on those four frames | -1.25° to -1.43° |
| Camera/map bearing disagreement | 14.80°–14.94°; certified limit 12° |

Comparing the planned map endpoint directly with the later map pose would mix
localization frames. The execution certificate was used to transform the goal
into odom before measuring navigation error above.

The calibrated optical-forward yaw is approximately -1.047° relative to the
base. Moreover, the full calibrated projection already predicts the target near
the image center. Camera mount yaw alone therefore does not explain the roughly
172-pixel observed discrepancy.

The candidate remains fixed at canonical odom point
`(2.178160374, 0.025493313)`. Arrival reprojects it to map point
`(-1.132293890, -0.367499501)` with the current map-to-odom transform. This keeps
frame arithmetic consistent but does not remeasure the stand. A fresh transform
cannot correct an inaccurate historical candidate-to-robot relative position
when both are transformed together. Initial candidate error, accumulated odom
drift and calibration error cannot be individually identified from this record.
The data does establish a mismatch between stored geometry and current sensors;
it does not establish a missing-TF failure as its cause.

## Why neither geometry nor QR-only admission succeeded

All 369 processed images reported `head_acquisition_deadline_exceeded` and all
369 preliminary map-cone LiDAR associations reported
`no_samples_in_accepted_range`. The original narrow cone looked toward distant
background returns, while the stand returns were to its right.

| Current QR result | Images | Consequence |
| --- | ---: | --- |
| No decoded QR geometry | 287 | No identity evidence |
| Correct `Start` payload but unavailable usable corners | 78 | No bearing/target binding |
| `Start` with corners | 4 | Rejected by the 12° camera/map bearing bound |

The QR-only geometry grace starts only after a fresh, associated QR observation.
That prerequisite never occurred. This was not an erroneous rejection of the
`Start` text, nor an expired fallback grace.

A raw-scan replay of all four corner-bearing frames confirms the recorded
registration rejection. Independently inspecting their narrow camera cones
also finds the nearest returns just outside the original accepted range:
0.436–0.440 m versus an upper bound of approximately 0.43445 m. Thus changing only
the angular limit would leave a range inconsistency. This diagnostic replay does
not modify the certified 12° limit or treat an unregistered cone as admission.

The wider QR search found only one or two in-range beams. The existing geometry
search-hint reconciliation requires three contiguous samples and rejected that
partial support (`search_cluster_insufficient_contiguous_samples`). Its small
identity crop could help decode text, but it did not correct candidate geometry.

Centering advice requires an accepted, currently associated head/crop in
`candidate_centering_receipt.py:40`. None was produced, so the observer never
reached a valid centering advisory. The new scan-boundary veto was not what
blocked a corrective turn here. An arbitrary image turn also would not, by
itself, reconcile the stored candidate position with the physical stand.

## Processing budget and what the viewer adds

The live detector began with images already 269.85 ms old at the median.
Across saved processed tuples, source freshness and the 50 ms publication
reserve left a median 166.89 ms head-work budget (45.58–235.83 ms).
Median detector duration over all 369 images was 171.63 ms; median image age
at detector completion was 447.97 ms. The cooperative deadline stopped work at
rail extraction, corner support, refinement or hypothesis comparison. No live
head measurement completed in time. The final status recorded 803 ingested
scans and 106 scans expiring before TF; TF delays contributed to available
work time but do not explain away the observed bearing discrepancy.

The same-pose viewer independently shows:

- 42/49 frames: `head_proposal_ambiguous`.
- 6/49 frames: a fitted current head, consistently near x=573.8 px.
- 1/49 frame: `head_model_pose_rejected`.
- 0/49 results passed viewer freshness. The six fitted frames were 0.429–0.558 s
  old against the viewer's 0.4 s freshness setting.
- QR decoding was explicitly disabled (`no_qr_decode: true`). Viewer
  `qr_detected: false` is therefore not evidence that this is a backside.

The head is visibly complete and uncut in both sources. The viewer supports a
persistent physical image offset and unstable border selection; it does not
supply an admissible pose to substitute into the robot run. Its options and
freshness thresholds differ from the live observer.

## Recommended next implementation

1. Add a stopped candidate-revalidation stage at opposite-side arrival, before
   repeating full head fitting. Compare fresh scan geometry and calibrated
   image/QR bearing against the stored candidate; distinguish an off-center
   view from a candidate-position conflict. Preserve candidate identity,
   competing-stand checks and independent evidence. When fresh geometry can
   certify a correction, persist its provenance and regenerate affected
   bindings/plans; otherwise explicitly report a target-geometry mismatch.
2. Bound head acquisition using a validated current sensor search region, and
   resolve outer/inset border families inside it. The current geometry search
   remains nearly full-width here and repeatedly spends the source-age budget
   on rail alternatives. QR payload/corner recovery should retain its separate
   budget and must continue to require a current target binding.
3. After candidate consistency is established, assess arrival framing with the
   calibrated camera projection and scan-boundary policy. Reuse the existing
   centering advisory and motion permits, reacquiring evidence after any turn.

Prioritize candidate consistency and bounded acquisition together. A longer
90-second timeout, relaxed freshness, a wider QR bearing limit, or forcing the
head to the image center would not address all observed blockers.

## Reproduction

From the repository root:

```sh
python3 results/implementation_checks/opposite_side_audit_20260922T141529Z/summary.py
PYTHONPATH=. python3 results/implementation_checks/opposite_side_audit_20260922T141529Z/geometry_metrics.py
PYTHONPATH=. python3 results/implementation_checks/opposite_side_audit_20260922T141529Z/binding_replay.py
```

These consume the read-only snapshot and production geometry/association
helpers. They produce `summary.json`, `geometry_metrics.json` and
`binding_replay.json`; no ROS runtime or motion is invoked.
