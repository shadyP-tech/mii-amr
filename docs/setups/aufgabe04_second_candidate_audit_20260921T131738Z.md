# Second candidate backside audit — 21 September 2026

Run: `stand_explore_exact2_camera_all5_20260921T131738Z`, deployed clean revision
`56892e38c9f5aece9e9d65f7f4db8b5ab6c19c85`.

Artifacts below are relative to
`results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260921T131738Z/`.
The parent and second-approach debug bundles are under `results/real_runs/`.
This audit reads recorded data only; no production code or robot state changed.

## Outcome

The first candidate, `survey_candidate_0003`, completed its first view with a
current-head-and-bound-QR recommendation for `QR_003`. Its 3.227° arrival bearing
miss was admitted by the repaired acquisition path. No alignment route was needed.

The second candidate, `survey_candidate_0001`, passed even the ordinary 3°
arrival check: range 0.5462 m, bearing error 0.717°. The observer started, but
all 373 processed frames reported `model_current_head_border_unavailable`.
It admitted zero candidate frames, zero axis samples, and zero QR samples.
The roughly 90-second deadline expired; SIGINT/exit 130 was parent timeout
cleanup, not an unexplained observer crash.

The visible grey backside is substantially left of the stored candidate's
projection. A candidate location/search mismatch blocks the useful head before
backside evidence can accumulate. A subsequent, separate TF admission failure
stops recovery to another view.

## Why the approach looked offset

The planner chose `robot-to-stand`, with no usable LiDAR inspection hint. Its
intended continuous endpoint was 0.50 m short of the projected candidate along
the incoming line. The map has 0.05 m cells. Static-clearance ranking selected
the adjacent cell inside its permitted 0.03536 m quantization envelope.

| Quantity | Recorded / calculated value |
| --- | --- |
| Planning candidate center | (-1.11562, -0.43303) m |
| Continuous requested endpoint | (-0.61735, -0.39140) m |
| Selected grid endpoint | (-0.59500, -0.36500) m |
| Endpoint displacement from continuous request | 34.59 mm |
| Lateral displacement from incoming center line | 24.45 mm |
| Planned endpoint-to-candidate distance | 0.52504 m |
| Largest recorded controller chord deviation | 2.38 mm |
| Last control-cycle distance to certified endpoint | 23.26 mm |

The terminal yaw is explicitly recomputed toward the candidate from the selected
endpoint (`candidate_preapproach_compute.py`). The small sideways endpoint
offset is therefore a deliberate raster/clearance choice. The controller trace
does not show a large departure from its planned line. Its endpoint error must
be compared in the sealed odom execution frame; subtracting fresh AMCL map
coordinates from old planning coordinates would conflate localization changes
with tracking error.

There is a larger discrepancy between the stored target and current sensing:
in 254 saved frames with exact scan TF and at least three nearby returns,
returns at 0.45–0.65 m and 0–25° have a median bearing of 12.35°, whereas the
stored candidate projects to 0.50°. Their mean-point offset from the projected
candidate has median 118.94 mm (range 104.72–132.48 mm), predominantly lateral.
This is a descriptive selection of nearby raw returns, not an authorized target
association or proof of the physical stand center. The image independently
places the visible stand on that left side.

The retained candidate uncertainty is only 20 mm. Two survey epochs supplied
57 and 71 source observations, with their map-space mean centers about 24 mm
apart. This audit establishes a later mismatch with current sensing, but does
not isolate its physical origin between accumulated odometry error, survey
localization/centroid bias, or calibration. Merely reprojection through a fresh
map-to-odom transform preserves the old canonical odom point; it does not
remeasure the stand.

## Why the backside detector did not admit the visible head

The saved rectified image is
`candidates/001_survey_candidate_0001/camera_lidar_attempt_00/perception_debug/latest_frame.png`.
The head occupies approximately x=176–280, y=238–339 pixels, with center near
(228, 289). Across observer results, the expected candidate center is
x=379.43–388.58. The approximately 99-pixel expected head height is plausible;
the lateral prediction is wrong by roughly 150–160 pixels.

The mismatch affects several gates:

- The candidate center bounds and approximately 149-pixel center tolerance
  constrain cold-search hypotheses near the predicted location.
- The LiDAR background region was available for 321 frames and unavailable for
  52 because two clusters were eligible. A region containing the visible head
  does not by itself admit its center or associate it with the candidate.
- Of 373 acquisitions, 360 report `head_proposal_unavailable`; 13 report
  `head_proposal_candidate_association_rejected`.
- There are 118 locally accepted raw-border refinements, but all their centers
  lie at x=411–416, on the central background support visible in the image.
  These are intermediate refinements, not admitted stand-head detections.
- The physical head's calibrated camera-ray bearing is approximately 14.55°
  in the scan frame, beyond the candidate's 12° camera/map registration bound
  when compared with the near-zero expected bearing. Even its 10-pixel rough
  refinement allowance does not bring that ray inside the bound.

The camera-ray bearing helper rotates the ray without translating a finite-depth
point. At this short range, camera/LiDAR parallax is relevant: the measured nearby
LiDAR mean bearing is around 12.35°, rather than the head ray's 14.55°. This
deserves checking in a registration repair; it is not evidence to remove the
association gate or to assign an arbitrary nearby cluster to the candidate.

No admitted current head means no certified backside axis, no opposite-face
receipt, and no camera-centering advisory. The generic
`lidar_target_not_associated` evidence counter is downstream of that unsuccessful
geometry/association pipeline; it does not mean all camera images or scans were
missing. A backside with no QR also cannot use the QR-only discovery fallback.

## Controlled image replay

Using the saved rectified image, the measured stand profile, recorded intrinsics,
and the shared border estimator under local OpenCV 5.0.0:

| Diagnostic prior | Result |
| --- | --- |
| Recorded candidate center/size screen | Head unavailable |
| Same size screen, center changed to visual hypothesis (227, 290) | Current measured head recovered; yaw 2.68° |
| No candidate screen | Background proposal; pose rejected |

The recovered corners are approximately (177.45,238), (280.13,238),
(278.30,339), (175.62,339). No image pixels or edge thresholds were changed.
This isolates the search-location problem; it is not a live observer replay.
The diagnostic omits scan proposal filtering, source-support masking and live
deadlines, and the shifted center is a manually selected hypothesis, not an
authorized correction. It does not prove end-to-end backside receipt readiness.

Reproducible scripts and JSON outputs:
`results/implementation_checks/second_candidate_audit_20260921/`.

## Why recovery then stopped

After timeout, local inspection proposed `inspection_view_01_00` with purpose
`diverse_inspection`. This was not the certified opposite-side branch. Its
dry preflight collected a stable stationary TF window, but odom execution
admission failed:

`preflight final map->odom transform predates its stationary sample window`

The final TF timestamp is 1789997137.1698616; the newest direct stationary-window
sample is 1789997137.2708561. The final value was captured 3.77 ms later but carries
a timestamp 100.995 ms older. Translation is identical and yaw differs by only
2.8e-17 rad. This is consistent with a lagging TF cache snapshot after a newer
direct sample. The ordering check in
`navigation/station_segment/localization_admission.py::_admit_stationary_map_from_odom_window`
rejects it. No recovery driving motion was launched.

## Repair priorities

1. Reconcile the stored candidate with fresh stopped LiDAR and camera evidence
   before using its center as a tight head-search prior or commanding centering.
   Preserve identity/uniqueness checks and record the residual; do not silently
   overwrite candidate geometry. Include finite-depth camera/LiDAR parallax in
   the registration investigation. Increasing capture duration will not resolve
   this persistent geometric mismatch.
2. Make the final execution-certificate TF snapshot coherent with the retained
   stationary direct-TF window, or reacquire it with a bounded wait. Preserve
   the timestamp-ordering and stability requirements.
3. Treat the approximately 24 mm grid-induced lateral endpoint shift separately.
   It is observable and may merit approach-quality refinement, but it does not
   explain the approximately 119 mm current-sensing mismatch or the TF failure.

## Implemented repair and local validation

The implementation following this audit adds a stopped-scan search recovery in
`real_robot/observer/stopped_target_search.py`. It applies only when the original
narrow map cone has no eligible target. A fresh, synchronized scan must contain
exactly one cluster in the existing registration envelope, including single-point
competitors in the ambiguity check. The selected cluster needs at least three
contiguous raw samples. Its mean bearing must stay within the existing 12°
registration limit; its extent and displacement from the original candidate must
stay within twice the configured stand radius plus uncertainty (0.16 m here).

The current scan mean supplies a temporary head-search location, projected with
the recorded camera transform and measured stand height. It does not overwrite
the candidate, provide corners, or authorize motion. Final current-head admission
retains the original candidate bearing, accepted range, physical scale, freshness
and quality checks, and requires overlap with the same scan cluster. Diagnostics
persist both projections, the residual, source indices and the reason when the
hint cannot be used. Nominal searches remain unchanged; both successful saved
first-candidate frames (4 and 7) take that unchanged path.

The proposal filter and final association now account for camera-to-LiDAR
translation at the hint's finite optical depth. Rough proposal screening also
includes the configured depth uncertainty. This corrects close-range parallax
without increasing the 12° bearing allowance. Current image borders and native
marker checks remain independent requirements.

The final map-to-odom observation now services callbacks and reacquires the
buffer sample up to five times if its timestamp precedes the last retained
stationary sample. Persistent lag remains a failure. Frame identity, quaternion,
freshness, stationary ordering and drift admission remain enforced.

Validation on the local Python/OpenCV environment:

- 176 tests and 98 subtests passed across camera processing, current-head
  association, backside/marker handling, bounded head evidence, arrival handoff,
  TF capture and stationary localization admission. Changed Python files also
  parse with the Python 3.10 grammar, and `git diff --check` passes.
- Two original compressed images and their scan/calibration/TF inputs are
  retained under `tests/aufgabe04/fixtures/stopped_target_search_20260921/`, with
  image hashes checked in the replay test. No manually selected center, corners,
  yaw, stored model pose or fabricated decoder result is supplied.
- Both regression frames recover a current head, pass final unique-cluster
  association and complete native marker-absence review. Frame 9 also passes
  single-angle quality and is classified `backside_candidate`; frame 4 retains
  its bounded orientation instead of manufacturing a reliable angle.
- Across 254 saved synchronized scan contexts, 198 permit the search hint;
  20 exceed the existing bearing bound and 36 fail uniqueness. Of the first
  12 eligible images replayed with raw source-support masking and current scan
  proposal filtering, 9 pass head association: 4 with usable single-angle fits
  and 5 with bounded orientation. The other 3 remain ambiguous and rejected.
  The diagnostic script and output are `repair_replay.py` and
  `repair_replay.json` in the implementation-check directory above.
- Negative tests cover stale/unsynchronized sources, range and bearing limits,
  excessive candidate displacement, competing/split/insufficient scan returns,
  mismatched frames or final proof context, nonfinite finite-depth projection,
  and TF buffers that never catch up or return invalid transforms.

This is local replay/component validation with OpenCV 5.0.0, not a new robot run
or a complete mission replay. The saved robot run used OpenCV 4.5.4. Geometry
regression tests deliberately avoid a wall-clock deadline to stay reproducible
on CI; production still enforces its sensor-age and processing deadlines. The
repair has not been deployed to the workstation or validated by a new physical
opposite-side transition. The planner's small grid-induced approach offset and
the unproven physical origin of the stored-target mismatch are unchanged.
