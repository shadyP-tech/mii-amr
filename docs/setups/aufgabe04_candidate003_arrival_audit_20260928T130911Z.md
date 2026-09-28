# Candidate 003 arrival / viewer audit — 20260928T130911Z

## Scope and identifiers

Read-only investigation of workstation run
`stand_explore_exact2_camera_all5_20260928T130911Z` and viewer recording
`recording_20260928_152428_945201347`. The run bundle records clean revision
`0329c686a5be3e4d5ae1a74829c92aa34da6290e`, matching the inspected checkout.

The child leg named `candidate_003` is the **fourth visited candidate**, whose
identity is `survey_candidate_0005`. It is not `survey_candidate_0003` / QR_003;
QR_003 had already been admitted. Recorded progress has three stands admitted
with facing evidence: QR_003, Start and QR_002. The fifth survey candidate is
still `inspection_started`, and survey_candidate_0004 is unvisited.

Local evidence is under
`results/implementation_checks/run_audit_20260928T130911Z/`, including copied
run metadata, candidate controller trace, viewer metadata and three source
frames, the contemporaneous observer debug image, `geometry_audit.json`, and
the reproducible ROS-free `replay_geometry.py` calculation.

## What happened

All times below are UTC; workstation viewer directory names use local time.

| Event | Evidence |
|---|---|
| 13:21:01–13:21:41 | Candidate preapproach executes and reports completed; 1.582 m recorded translation. |
| Stopped arrival | Candidate range 0.528708 m; stored-target bearing error −2.4037°, within the strict 3° limit. `camera_centered=false`, `requires_live_target_association=true`, no validated live target center. |
| Approximately 13:21:48–13:23:17 | Camera observer times out after its 90-second budget. 403 processed/fresh detector results; all 403 rejected for LiDAR target association; zero accepted candidate frames and zero axis consensus. |
| Subsequent recovery | First alternative direction: all four standoffs statically blocked. Another direction at 0.45 m reaches dry admission but fails route uncertainty by −0.016761 m. No `_inspection_001` execution bundle exists and its events show zero motion. |
| 13:24:28–13:24:39 | Viewer records 90 frames. This is later evidence, not synchronized with the original observation. Its framing closely matches the run's own saved debug image. |

The saved top-level terminal log ends after the later dry preflight. It does
not establish a normal final mission termination or its exact process-exit
cause. No exploration process remained when checked. The camera timeout and
subsequent dry rejection above are explicit recorded failures.

## Primary fault: older candidate geometry disagrees with current measurements

Arrival admission compares the stopped robot to the reprojected stored
candidate. It does not assert that the camera has centered a live observed
stand (`real_robot/candidate/approach.py:1282–1313`).

The candidate was last seen during the survey about **519 seconds** before
this camera attempt. Its position uncertainty remained **0.02 m**. The frame
contract retains an authoritative canonical odometry point and projects it
with the latest map-to-odom transform:

| Quantity | x (m) | y (m) |
|---|---:|---:|
| Canonical survey odometry point | 4.666854 | 0.978437 |
| Original survey map point | 1.433706 | −0.248086 |
| Reprojected arrival map target | 1.438764 | −0.489008 |
| Current scan-cluster surface, captured frame 120 | 1.477251 | −0.265398 |

Reprojection displaced the target **0.240975 m**, almost entirely in map y.
The underlying map-to-odom changes are 0.315256 m translation and 0.116530 rad
yaw. The transform arithmetic itself is consistent with the implemented
contract (`navigation/approach/candidate_frame_reprojection.py:231`).

The current cluster surface is **0.226898 m** from the reprojected target but
only **0.046860 m** from the original map target. A LiDAR surface is not the
exact stand center; these distances are diagnostic comparisons, not a new
authorized center estimate. Nevertheless, they demonstrate a substantial
current-measurement disagreement with the retained odometry anchor.

The evidence is consistent with accumulated odometry/localization disagreement
being carried into the later candidate target. It does not independently prove
wheel slip or establish absolute map ground truth. Treating the old odometry
point as timeless with unchanged 2 cm uncertainty is the weak assumption.
Blindly replacing it with the original map point would also be unjustified.

The drive itself was much closer to its commanded goal: the last saved control
sample was 0.032392 m from the route endpoint with about 2.59° heading error.
That sample precedes the completion event; it is not a separate exact final
pose measurement. The stopped arrival check also passed. These small errors
cannot explain the roughly 23° camera correction required for the visible head.

## Why nearest acquisition and centering did not recover

The observer projects its expected target near **x=417–421 px** in an 800-pixel
image. Its candidate center tolerance is about **154 px**. The actual visible
blue head's measured diagonal-intersection center in the viewer is about
**x=108 px**.

`perception/stand_axis/nearest_scan_head.py:82–96` filters projected scan
clusters against that candidate screen **before** choosing the nearest one.
The observer's nearest search therefore reports
`nearest_head_no_visible_scan_candidate`, even though a stand-sized cluster
exists in the camera view.

A pure geometry replay of the run's captured frame 120, using the saved scan,
exact-time transforms and the unchanged accepted range [0.380579, 0.600579] m,
confirms:

- With the recorded candidate screen: zero eligible clusters.
- Removing only that positional screen for diagnosis: one cluster, scan indices
  11–15, about 0.5294 m from the base, projected near x=96 px. No head fitting,
  QR admission or motion is authorized by this diagnostic replay.
- Its scan bearing is **+21.04°**, while the stored target is approximately
  −2.5°. It is outside the observer's 15° expanded identity-search cone.

Reconciliation would also remain blocked after acquisition: its current
maximum bearing discrepancy is 12° and position discrepancy is 0.16 m
(`real_robot/observer/target_reconciliation.py:58–96`). The nearest competing
stored candidate in this replay is about 0.855 m away, which supports further
bounded identity revalidation but is not itself proof of identity.

No associated target reached the centering-advisory path. Independently,
projecting the measured viewer head with the calibrated camera translation and
rotation gives approximately **22.55° left** to center it. This calculation
uses the closely matching later viewer framing and the run's scan range; it is
an approximate diagnostic, not an execution command. Existing centering allows
two steps of at most 6°, with **12° total**, so it could not complete this
adjustment even if acquisition alone were repaired.

## Viewer and detector evidence

The viewer used nearest-head selection with QR decoding disabled. Across its
90 frames: 61 usable measured-head estimates, 18 planar-axis ambiguous, five
high yaw-uncertainty, five ambiguous nearest-scan selections, and one missing
head border. The framing is stable across the inspected beginning/middle/end
source images. This shows the blue border is generally measurable at this
pose; it does not prove candidate identity or an end-to-end QR admission.

The run's 403 detector results comprise 260 missing-head-border, 138 acquisition
deadline, four usable-axis, and one unverified-backside-crop outcomes. The four
usable fits have centers near x=524.5 px, on the radiator region rather than the
blue stand. LiDAR association correctly prevented those background fits from
being admitted. The candidate screen directs processing toward that region.

Only ten synchronized tuples exhausted exact-time TF retry, versus 403
successfully processed detector results. TF availability is not the dominant
failure here. The run's saved image and the later viewer both show the actual
stand well inside the image, far left of its predicted location.

## Recommended correction

1. **Revalidate candidate position epochs before close approach and at arrival.**
   Retain survey provenance, but account for elapsed travel/localization changes
   instead of treating a historic odometry point as an indefinitely accurate
   physical landmark. Detect inconsistent current scan geometry explicitly.
2. **Add bounded position-recovery acquisition with separate identity proof.**
   Search a justified uncertainty region using current camera/scan observations;
   require multiple fresh stationary samples, camera/scan agreement, and
   exclusion of every competing candidate before issuing a validated live
   target-center receipt. The normal narrow gate should remain for consistent
   targets. Do not globally remove association gates or simply inflate limits.
3. **Replan/reorient from the validated live center when the error exceeds the
   small centering budget.** A separately admitted coarse arrival correction
   can address this approximately 23° error; use the existing bounded fine
   centering afterward. Include camera lever-arm calibration and preserve any
   retained stand angle; centering does not require fitting a new stand angle.
4. **Return a structured target-position inconsistency early.** Repeated absence
   of the expected cluster alongside a persistent nearby plausible stand should
   enter bounded recovery instead of spending the full camera timeout looking
   at the wrong region.

Regression acceptance should replay this run's actual captured scan/images:
recover the visible head for diagnosis; validate candidate identity independently;
reject the radiator and competing-candidate cases; recover or safely stop on
the out-of-budget angle; and admit a complete decoded target QR once associated.
No production code, robot state or workstation checkout was changed by this audit.
