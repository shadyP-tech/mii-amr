# Complete current-head selection and QR refresh — 2026-09-15

The front-facing viewer recording contains usable head geometry but the cold
locator previously aborted before verification, then treated finder-sized
rectangles as separate heads when its work budget was experimentally increased.
This correction lets complete current outer borders reach the unchanged measured
3D head fitter. QR observations and neck validation supply no angle authority.

## Implementation

`perception/stand_axis/head_proposal_selection.py` owns the bounded 2D selection
policy. `head_cold_acquisition.py` continues to generate current contour and rail
hints, and delegates ranking and comparison to that module:

1. Each proposed side needs at least 80% current locator support. This rejects
   line pairs that span separate texture boxes with a missing middle interval.
2. Current corner-arm pixels rank viable search endpoints. Nearby alternate
   corridors are retained within the existing strict fitter's 10-pixel maximum
   endpoint shift. These hints never substitute for measured corners.
3. Complete enclosing alternatives are tried first within each border family;
   round-robin family coverage prevents one textured head consuming all work.
4. At most **12 strict raw four-border verifications** run. Independent untested
   families, failed representatives with unresolved alternatives, and untested
   larger borders remain unresolved when that limit is reached.
5. A strictly verified outer frame may contain repeated small rectangular
   texture. This requires at least three disjoint, similarly sized rectangles
   inside its projective coordinates, aligned to the outer sides, distributed
   across three quadrants and spanning both axes. No QR detector output, QR
   dimensions, identity, angle, or neck enters that comparison.
6. Separate heads, a lone nested stand, overlapping/misaligned inner shapes,
   clipped outer borders, and missing borders retain their rejection paths.

The resulting neutral corners still pass `fit_current_measured_head`, including
current raw corner support, outer-border verification, positive-depth physical
pose, planar ambiguity and angular uncertainty. The fixed quality limits and
observer's motion, freshness, association and route gates are unchanged.

The cold locator serves the viewer when there is no named candidate projection.
Mission candidates retain bounded projected acquisition and current LiDAR
association. This change does not turn an unregistered viewer angle into a
candidate identity or opposite-side driving receipt.

`perception/stand_axis/current_image_head_fit.py` handles the companion mission
optimization. The observer creates a one-use holder inside each ROI evaluation.
When full QR decoding adds evidence, the metric pipeline reuses only that same
crop's **undecorated** raw physical-head fit and recomputes QR/side evidence.
Frame identity and a pixel digest, ROI, calibration, profile, proposal/pose hint,
projection and geometry options must match. A new image or changed context
computes fresh geometry; a third call cannot consume the result again. The
nonphysical diagnostic path continues to refit.

Thus newly decoded QR evidence can remove a provisional backside label without
changing the current head's corners or angle. Invalid physical geometry remains
invalid. The final observer freshness check still uses completion time, and
timing diagnostics include the first geometry pass plus the QR refresh. The
new `current_image_geometry_reused` diagnostic distinguishes this operation
from `current_image_geometry_refit`.

## Original-image replay

Source media remained on mii001. The changed modules were loaded **only in
memory**, inside the existing Python 3.10 / OpenCV 4.5.4 container with the remote
repository mounted read-only. Images were rectified with their saved calibration.
No remote source, recordings or robot state changed.

| Saved recording | Replay mode | Usable current 3D fits | Yaw range |
| --- | --- | ---: | --- |
| `recording_20260915_155843_422965485` | Cold acquisition on every image | 46/57 | 24.40°–33.01° |
| Same front sequence | Cold start, then existing current-pixel tracker | **57/57** | **24.78°–25.19°** |
| `recording_20260915_160107_616628722` | Cold acquisition on every image | 22/54 | −21.22° to −7.05° |
| Same backside sequence | Cold start, then existing current-pixel tracker | **50/54** | **−16.61° to −12.93°** |

The front sequence starts automatically with seven strict checks; no manual
corners, preloaded pose or decoded identity seeds it. Median measured processing
time was 89 ms for cold front fitting and 19 ms with tracking; backside tracked
median was 32 ms. These are offline timings, not live freshness certificates.

Cold reacquisition remains imperfect: eight front images retain proposal
ambiguity and three reach the bounded work limit. The four unavailable tracked
backside frames retain three acquisition failures and one planar-angle ambiguity.
Independent cold fits also show larger angle variation than tracked fits; the
existing temporal consistency and uncertainty checks remain necessary. These
recordings establish regression behavior, not external angle ground truth.

## Limits and follow-up evidence

Pixels alone cannot distinguish all physically different scenes with identical
projections, including a large panel containing repeated smaller objects. The
2D texture policy is conditional on the measured head model; robot association
and motion admission remain separate requirements.

This correction does not implement cross-frame registered-crop retention or
resolve AMCL drift. Those mechanisms require separate state and hardware
evidence. The earlier audit documents the first candidate's changing border
selection and stale processing results in
`aufgabe04_first_candidate_and_viewer_audit_20260915.md`.

## Regression validation

**333 distinct tests passed across 29 focused suites**, using OpenCV 4.13.0.
The two invocations reported 273 tests / 523 subtests and 72 tests / 73 subtests;
12 registration tests overlapped. The new coverage includes:

- Complete perspective heads with three/four inset rectangles at 25° and 45°.
- An additional larger/smaller head on either side, including reversed contour
  order; lone nested, misaligned, overlapping, clipped and missing-border cases.
- Thirteen/seventeen independent supported families exhausting exactly twelve
  raw verification calls, with no first-head selection on incomplete coverage.
- Changed QR payload/size preserving geometry; changed frame, pixels, ROI,
  intrinsics, projection, profile, edge settings, proposal or pose invalidating
  one-use reuse; invalid geometry and legacy behavior retaining their gates.
- Recovered QR evidence replacing provisional backside semantics from the raw
  physical result, while all processing time remains accounted for.

All eight changed Python files parsed successfully; `git diff --check` passed.
Hardware execution and deployment were not performed. Original media stayed on
mii001; only compact replay diagnostics returned.
