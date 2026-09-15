# QR-independent physical head acquisition — 2026-09-15

The measured physical head now owns its angle independently of QR dimensions,
QR pose, or joint QR/head reprojection agreement. The debug viewer can acquire
a complete head before any QR or tracked pose exists. Side classification and
identity still require their separate evidence.

## Latest available run diagnostics

Run `stand_explore_exact2_camera_all5_20260915T124514Z` records commit
`82ccff234ae6b60bf12641796cabbfe3580960c5`. Saved JSON diagnostics were read on
`mii001` with the user's approval. No run images or recordings were transferred.
The remote audit is `/tmp/a04_camera_audit_20260915T124514Z/AUDIT.md`, with detailed
aggregation in its adjacent `DETAILS.json`.

The first camera candidate (`survey_candidate_0003`) accumulated seven accepted
angles and committed QR_003 at its first observation. The second candidate
(`survey_candidate_0001`) had different failures:

| Camera attempt | Available evidence |
| --- | --- |
| 00, initial backside view | 410/410 results fresh. Expanded search found 197 complete head-sized proposals. Every proposal failed candidate association: centers were 131.5–138.5 pixels left of prediction and bearings differed by 12.237–12.799°, beyond the unchanged 12° gate. |
| 01 | 403/403 results fresh, no head proposal; no nominal in-range LiDAR returns. |
| 02, third camera observation | 184/194 results obsolete. All 171 expanded proposals exceeded the bearing gate. Six QR quadrilateral detections were stale and unverified. |
| 03 | One accepted angle sample, peak 1/7, then expired. 66/67 expanded proposals exceeded the bearing gate. Neither of two QR quadrilateral detections was verified. |

These records do not establish a successfully decoded identity for the second
candidate. Missing text in stale events alone is inconclusive; explicit marker
verification failures provide stronger evidence. The terminal bundle stops
during pre-run diagnostics for another inspection, without a final exit record,
so it does not establish normal mission completion.

The canonical odometry landmark stayed fixed while its transformed map position
moved 8.27–15.93 cm from the survey position at the later arrivals. This is
coordinate/projection movement, not measured physical ground truth. The camera
change does not repair this separate localization/association problem and does
not widen the association gate.

## Implementation

- `perception/stand_axis/physical_head_pipeline.py` separates head acquisition
  and fitting from QR handling. A current candidate proposal, candidate
  projection, or prior head pose locates current pixels. Without any of these,
  the viewer invokes QR-free cold acquisition. A prior pose cannot resolve an
  ambiguous current fit or supply an old angle.
- `perception/stand_axis/head_cold_acquisition.py` obtains neutral proposals
  from closed contours and paired current rails. Grayscale LSD and raw
  channel-union Hough lines cover luminance and color boundaries. All accepted
  proposals pass the existing raw four-border and corner-arm checks. Nearby
  inset alternatives are grouped; distinct or arbitrarily nested heads remain
  ambiguous. Search/verification work is bounded.
- `model_pipeline.py` fits the physical head before attaching QR marker evidence.
  It no longer computes a QR pose, QR-seeded head, or joint QR/head diagnostic
  fit on this path. QR contributes only identity and marker/side evidence.
  The separate provisional model path retains its legacy behavior.
- `head_outer_border.py` uses fixed bounded outward searches independent of QR
  dimensions. QR-span diagnostics have no admission authority. The obsolete
  QR-specific temporal veto and unused physical QR diagnostic fitter were removed.
- `model_diagnostics.py` records acquisition source, exact current proposals,
  rejection reasons and work budgets. This distinguishes missing image evidence
  from later association or angle-quality rejection.

The neck is not required. Pixel span, raw corner/border evidence, positive depth,
planar ambiguity, reprojection error, uncertainty, exact-image freshness,
stationary temporal evidence and candidate association remain enforced. The
purple dashed drawing uses the existing current-fit viewer admission policy.
It is a single-frame geometry result, not a motion or opposite-side receipt.

## Validation and limits

The focused head, backside, observer, viewer, QR-marker and binding regression
suite passed **457 tests and 753 subtests** in 11.20 seconds. All 18 changed/new
Python files also parsed with Python 3.10 syntax rules; `git diff --check` passed.

Offline tests cover QR absence, conflicting QR sizes/positions, changed QR
profile dimensions, multiple identities, cold head acquisition, equal-luminance
head/neck colors, clipped/open borders, nested background frames, two heads,
freshness expiry, named-candidate search bounds and cache isolation. Synthetic
current-pixel heads through 75° reach the 3D overlay where resolution and pose
quality permit; the high-angle acquisition floor was removed without relaxing
the physical quality gates.

The already committed lossless `back22` crop `(260,260,435,390)` now cold-acquires
and produces a usable angle of approximately **−14.784°**, uncertainty **2.137°**,
and reprojection RMS **0.518 px**, with no QR, neck or prior pose. This is a replay
estimate, not ground truth. The tighter nominal crop `(303,251,439,388)` leaves
about two pixels beside the left border and still fails uncertainty admission;
both outcomes have regression coverage.

The wider saved `front45` crop remains unresolved under the cold-search work
budget. Cold acquisition is not an exhaustive head detector in clutter and has
no stand-identity authority. Mission candidates retain their existing bounded
projection/reacquisition and camera–LiDAR association requirements.

On local Python 3.12/OpenCV 4.13, 15 warm replays of the wider backside crop took
median **45.5 ms**, maximum **66.1 ms** for the model pipeline; front45 took median
**54.6 ms**, maximum **66.2 ms**. These exclude ROS/image delivery and display,
and are not robot CPU or end-to-end timing evidence. Robot OpenCV/runtime and
hardware behavior still require validation. No robot experiment or deployment
was performed by this change.

`obsolete_detector_result` means the result exceeded its age limit before it
could be used or displayed. The viewer checks time since receipt and time since
the image's source timestamp, including transport age. Expired geometry cannot
enter classification/angle evidence even if its overlay would fit the image.
Do not fix that condition by admitting old frames: inspect input age and per-stage
timings. In this run it explains much of attempt 02, but not the fresh attempt 00.
