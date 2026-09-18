# Distance, dimensions and colour for physical head acquisition

Implemented in the shared camera-exploration/debug-viewer acquisition path on
2026-09-18. This improves search and border selection; it does not provide
per-pixel background depth or guarantee that every remaining edge belongs to
the head. The workstation production checkout was not deployed or restarted.

## Implementation

- Preserve correlation between camera depth and projected vertical length.
  The measured 78 mm frame at approximately 0.364 m now permits roughly
  106–177 px rather than 75–245 px. Apply these bounds to **each vertical
  side**, so one short side cannot hide behind an acceptable average height.
  Unknown yaw remains allowed; frontal width is not a minimum width gate.
- Use calibrated candidate location and explicit position uncertainty to
  project a possible head volume and tighter centre bounds. The centre is the
  projected diagonal intersection, avoiding the bias of averaged corners.
  Depth-only inputs retain broad position uncertainty. Viewer nearest mode
  defaults to 2 cm camera X/Y uncertainty, configurable with
  `--head-position-uncertainty-m`; this is an assumption, not a measured
  calibration guarantee. Exploration uses its configured stand uncertainty
  and radius, and activates the position gate only with an independently
  corroborated current LiDAR region.
- Keep rail-selection work bounded. Removing rails outside the candidate
  volume does not refill their quota with progressively more inset/printed
  rails. The existing maximum of 12 strict hypothesis checks remains.
- Permit one additional enclosing-rail refinement inside an at-most-eight-pixel
  corridor when budget allows. It requires coherent current raw support and
  enclosure, preserves existing growth searches, and retains the original
  verified border if the extra search fails. It cannot manufacture corners.
- Share the existing stand HSV palette between exploration and the viewer.
  Colour near an observed rail supplies a soft ranking bonus equivalent to at
  most one pixel of distance. Raw Canny evidence remains intact; blank masks
  do not erase grey borders and full masks do not create edges. This cue is
  used in cold border selection; existing tracked raw fits retain their
  normal verification. The viewer's `--edge-color`, HSV tuning and
  `--no-color-edge-mask` control this cue as well as its preview filtering.
- Preserve original `LaserScan.angle_max` and scanner topology in the viewer.
  Its TurtleBot launcher declares full rotation; the general parser defaults
  to linear. Endpoint clusters merge only after the existing angular and
  physical-adjacency checks pass. New recordings include the raw target scan,
  timestamps and topology; nonfinite ranges are serialized as null.

QR dimensions, decoded identity and previous detections do not supply head
corners. Border detection, reliable yaw and permission to move remain separate.

## Latest-recording replay

Recording: `recording_20260918_153656_468438163`, 257 frames. See the
[original audit](aufgabe04_head_frame_recording_audit_20260918.md) and
[final replay summary with hashes](aufgabe04_head_border_optimization_20260918.json).

The replay used an isolated code overlay on mii001, with its original Python
3.10.12/OpenCV 4.5.4/NumPy 1.21.5 runtime and original images/calibration/model.
It exercised all 17 frames whose recorded scan selection was unique. No
manual corners, QR seed, tracker or live deadline was used for these counts.

| Variant | Reported head border | Reliable yaw |
|---|---:|---:|
| Original recorded pipeline | 0/17 | 0/17 |
| Final dimensions/position/rail changes, colour disabled | 3/17 | 1/17 |
| Same changes, colour enabled | 3/17 | 1/17 |

Final border-positive frames are 47, 176 and 221; only 221 has reliable yaw.
Their overlay corners were visually inspected, but no independently annotated
corner ground truth is available. Twelve frames remain ambiguous and two have
no admissible current border. Frame 191's earlier false diagonal border across
QR texture is rejected by the per-side size check. Earlier experimental
7/17 results are superseded: that selection could suppress valid competing
borders and falsely promote inset borders.

Colour did **not** improve detection counts or selected corners in this
recording. A controlled two-rail test verifies that it can break an equal
geometric tie using observed pixels. The palette is an initial camera-lighting
palette, not a calibration valid under all illumination. Hard HSV masking
would discard real desaturated borders; it is deliberately not used for metric
measurement. Single-run median acquisition times were 126 ms with colour and
113 ms without it; these are sequential offline measurements, not a live
latency guarantee or a randomized performance benchmark.

The other 240 frames stopped at scan-candidate ambiguity. Their recording did
not save raw angular/range geometry, so the new seam handling cannot be
validated retrospectively on those frames. Conditional merged-centroid probes
of frames 0/128/256 detect a border only in 256, without reliable yaw. Those
are hypothetical association diagnostics, not automatic detection successes.
A fresh recording with raw scans is needed to measure the combined live gain.

Detailed replay inputs, scripts, outputs and overlays are retained locally in
`results/aufgabe04/debug_audits/recording_20260918_153656_468438163/`.
Source image, model, metadata and overlay-code hashes were unchanged during
replay. The deployed workstation checkout remained clean at
`e0cbf5ebbbc1a99de3b5ced09c4d2c3f5d96a2d0`.

## Verification

757 tests and 916 subtests passed across 67 affected head/metric/viewer/scan/
physical/raw-edge modules using local Python 3.14/OpenCV 4.13. Additional
workstation evidence above covers the production OpenCV version. Tests cover
495 depth/yaw/tilt/position projections, individual-side bounds, recorded
clutter ambiguity, unchanged raw pixels, colour tie-breaking, blank images,
uncoloured heads, competing heads, failed enclosing-fit fallback, validated
and invalid scan seams, and raw-scan serialization. Existing observer
association, QR independence, freshness and bounded-work checks also pass.
`git diff --check` and launcher shell syntax checks pass.
