# Backside camera handoff correction — 10 September 2026

This change addresses the processing starvation and terminal-status retry defect
in [the second-candidate audit](aufgabe04_backside_audit_20260910T140546Z.md).
The audited run and its original evidence remain unchanged.

## Implementation boundaries

| Module | Responsibility |
| --- | --- |
| `real_robot/observer/backside_proposal_reuse.py` | Retain one bounded search hint; fit each new image and recheck registration against its original candidate projection. |
| `perception/stand_axis/model_input_cache.py` | Reuse raw edges and QR quadrilateral detection for exact same-image crops and settings. |
| `real_robot/observer/timeout_policy.py` | Classify observation deadlines using validated accumulated evidence and child cleanup outcome. |
| `real_robot/observer/node.py` | Supply current sensor context and run the existing freshness, QR, LiDAR, consensus and artifact-publication checks. |

After ordinary acquisition has found registered usable backside geometry, the
next image can start with one strict fit on the existing bounded wide ROI.
The hint contains full-image corners only: no angle, pose seed, identity or
motion receipt. A result rejected for latency can locate this search, but
cannot supply an axis sample. Current-image QR decoding and metric fitting
always run; current fitted corners must independently pass the original
projection's registration bound and the downstream LiDAR association.

Hints expire after at most two seconds measured from the source image stamp.
They require advancing image timestamps, unchanged candidate/model/calibration
context, and a stationary pose relative to the original anchor. Any failed fit,
QR detection, marker latch, relevant context change or motion invalidates reuse.
If the nominal ROI extends beyond the wide ROI, ordinary acquisition is used
so the optimization does not omit previously inspected pixels. A failed hinted
fit is reported normally; the next image reacquires instead of repeatedly
processing the same aging image.

The separate input cache is recreated for every rectified image and holds at
most three crop/settings entries. It checks source-image memory, pixel bytes,
ROI, decoder mode, QR observations and preprocessing settings. Strict geometry
and pose fitting are never cached. Full QR acquisition remains enabled for
backside observations, including the wider crop.

The timeout policy now recognizes a trailing `obsolete_detector_result` only
when the status is valid, explicitly unpoisoned and contains earlier TF-ready
candidate processing. This produces the existing typed observation failure,
which records the failed view and continues within the configured inspection
budget. Missing evidence, identity conflicts and child crashes remain fatal.
Stale images remain unusable for motion.

## Existing opposite-face transition

Seven fresh, consistent, candidate-bound backside samples are still required
before writing the backside-axis observation. The existing inspection loop
then selects the opposite face before generic view search. All candidate
keepouts, route checks, frame binding and motion admission remain in effect.
This change introduces no new motion command or freshness-limit override.

## Diagnostics and validation

Observer output adds `metric_model.backside_proposal_reuse`, the
`camera_target_registration.search_hint_used` flag, and per-attempt
`qr_decode.model_inputs` cache timings. The ROI's hint center and the current
registration center are deliberately reported separately.

Regression coverage includes the recorded timeout, bounded two/eight-view
failure handling, stale bootstrap followed by a distinct fresh observer
sample, seven-sample evidence consensus, duplicate-frame rejection, current
QR conflicts, context/motion expiry, exact crop isolation, strict geometry
refitting and the existing opposite-side planning gates.
The combined focused suite passed **228 tests** with Python 3.12.14,
OpenCV 4.13.0 and NumPy 2.5.3; `git diff --check` also passed.

The portable recorded-image regression retains original JPEG frames 000008
and 000010 with CameraInfo, scan, TF and SHA256 provenance in
`tests/aufgabe04/fixtures/recorded_backside_20260910/`. Frame 000008 still
produces **−2.631965°** and passes the existing registered LiDAR association
at **0.758 m** with **8.503256°** bearing correction. The subsequent actual
frame 000010 fails geometry and clears the hint without inheriting the angle.

Five same-host timing trials measured median selection times of **734 ms**
without the input cache, **602 ms** with the cache on initial acquisition,
and **318 ms** with the single-fit hint path. The warmed positive replay uses
repeated image content under an explicit synthetic test clock; it establishes
fit equivalence and workload reduction, not new sensor observations or a
freshness-qualified consensus. The receipt is saved in
[`backside_fix_benchmark.json`](../../results/audits/stand_explore_exact2_camera_all5_20260910T140546Z/derived/backside_fix_benchmark.json).

Offline timing is diagnostic, not hardware qualification. The next real run
must demonstrate seven fresh samples, a committed backside-axis observation,
an admitted opposite-face route and QR decoding at the new stopped view.
Physical angle accuracy and the complete live transition remain unvalidated.
