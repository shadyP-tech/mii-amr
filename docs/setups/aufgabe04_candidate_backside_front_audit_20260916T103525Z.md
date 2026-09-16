# Candidate backside/front admission audit — 2026-09-16 10:35 UTC run

Run: `stand_explore_exact2_camera_all5_20260916T103525Z`.
Inspected code: clean `af348bd278028ef8fe05d3785f827f7aba49cecd`.

The second candidate's backside **was classified**, and the 3D head model did
produce angles. The opposite-side branch was blocked by angular instability
and an inconsistent consensus/receipt acceptance contract. The third candidate
**decoded QR_002 repeatedly**, but its head fits were mostly ambiguous or just
above the yaw-uncertainty threshold; its only geometrically accepted fit failed
temporal border consistency. A separate recovery-state defect ended that
observation prematurely.

This was a read-only audit, including current-pixel replays in the existing
Python 3.10/OpenCV 4.5.4 Apptainer environment with read-only repository mounts.
Raw images, recordings, scans and run files stayed on mii001. Only derived
diagnostics returned. No ROS commands, robot motion or production edits ran.
Access was through mii001; the bundle manifest reports execution host mii0002.

## Scope and run state

The run began at 10:35:25 UTC / 12:35:25 CEST. Its last recorded parent-terminal
line is at 10:45:59 UTC / 12:45:59 CEST, collecting pre-run diagnostics for the
third candidate's next inspection. The inspected artifacts have no parent
completion/exit marker and no `mission_failure.json`. A process check did not
find the matching exploration parent. These facts do not establish why the
parent stopped or whether it was interrupted; the camera decisions below are
independently established by completed observation artifacts.

The goal ledger contains one confirmed identity out of five:

| Inspection order | Candidate UID | Recorded result |
| --- | --- | --- |
| First | `survey_candidate_0003` | `QR_003` confirmed at the initial camera pose |
| Second | `survey_candidate_0001` | Backside advisory, then fallback arrival rejected; marked `inspection_exhausted` |
| Third | `survey_candidate_0002` | `QR_002` decoded but not admitted; next local route prepared |

Both LiDAR survey legs completed. The second initial arrival was accepted at
0.5074 m range and 0.312° bearing error; the third initial arrival was accepted
at 0.5348 m and −1.766°. Thus their initial camera failures did not originate
in an arrival-admission rejection.

## Second candidate: appearance succeeded, orientation receipt failed

Evidence prefix beneath the run root:
`candidates/001_survey_candidate_0001/camera_lidar_attempt_00`.

- All **30/30 processed images were fresh**. Median source age at receipt was
  28.34 ms and at completion 135.97 ms; maximum completion age was 417.51 ms.
- There were 15 usable individual 3D fits, 16 associated frames and nine accepted
  axis samples across the observation. These counts describe different gates.
- Repeated backside appearance reached seven samples and confidence
  **0.932215**, with state `backside_supported`. Neck validation was unnecessary.
- One seven-angle consensus used **−6.208, −6.192, −10.173, −10.182, −10.788,
  −6.327 and −3.043°**. Its mean was −7.559°, maximum deviation 4.516°, and total
  angular span 7.745°.
- The temporal span policy allowed that window under its 8° limit. However,
  the receipt producer calculated `1 − maximum_deviation / 8° = 0.435444`,
  while the artifact validator requires at least 0.60. The recorded rejection
  was `axis observation axis_confidence must be in [0.60, 1]`. This imposes an
  additional effective maximum deviation of 3.2° at these settings.
- Subsequent fits moved toward −2° to −0.63°, resetting temporal axis evidence.
  The observer emitted an advisory `backside_unresolved` observation after
  eight advisory samples. It did not reach the 90-second camera timeout.

The model fits are unstable despite looking close in the image. Selected rails
shift by approximately 2–3 pixels: left x≈213 versus 216, right x≈319 versus 323,
top y≈233 versus 236. These small changes produce approximately 10° of yaw
variation while stopped.

Unchanged saved-image replay reproduced **15/30 usable fits** with recorded
proposals. Using previous observed borders, or the first observed borders, only
as search seeds produced 13/28 usable fits in either case. Among 15 tracked
searches, five fit, nine failed corridor refinement and one failed corner
evidence. On representative frames 000027 and 000035, top and bottom rails had
approximately 99–100% support, but coherent right-rail selection returned no
accepted rail. The exact distinction between insufficient coherent support and
competing coherent rails was not separately instrumented. This is not explained
by stale delivery, neck validation or simple crop clipping.

Relevant code:

- `scripts/aufgabe04/real_robot/observer/node.py:2321`: confidence conversion.
- `scripts/aufgabe04/artifacts/backside_axis_observation.py:218`: receipt threshold.
- `scripts/aufgabe04/real_robot/observer/head_temporal_consistency.py:162`: temporal policy.
- `scripts/aufgabe04/perception/stand_axis/raw_support.py:314`: coherent rail selection.
- `scripts/aufgabe04/perception/stand_axis/metric_edge_association.py:119`: missing/competing rail handling.

## Second candidate: the fallback also failed before camera observation

The generic fallback drive completed at 10:42:48 UTC. Its arrival-bearing error
was **−3.859°**, beyond the 3° limit. Two correction routes completed, but their
fresh arrival checks still measured **−3.979°** and **−3.128°**. The last arrival
artifact explicitly says the observer did not start. The candidate was marked
exhausted at 10:44:01 UTC because of `candidate_arrival_geometry_rejected`, not
because eight camera viewpoints had been tried; the history contains one
completed local camera view.

Both the follower and arrival gate use a nominal 3° limit, but on different
quantities. The follower targets a fixed planned terminal yaw; arrival checks
the bearing from the new actual robot pose to the freshly projected candidate.
Correction route goals were quantized to (−1.145, 0.085) m. Their planned yaws
were −1.621470 and −1.650160 rad. At the corresponding later arrival poses,
errors to those fixed yaws were approximately −1.526° and −2.434°, whereas the
candidate-bearing errors were −3.979° and −3.128°. These are illustrative
comparisons at the later arrival snapshots, not controller-time TF replays.
They show why satisfying the fixed terminal heading does not guarantee the
candidate-bearing gate. The first correction published no motion; the second
published a small terminal rotation.

Relevant code: `real_robot/candidate/inspection_adapters.py:196` plans bounded
corrections, `real_robot/candidate/approach.py:1126` recomputes arrival geometry,
and `navigation/waypoint_follower/terminal_heading.py:204` checks fixed goal yaw
(all paths under `scripts/aufgabe04/`). Candidate-facing correction needs a
consistent target definition and tolerance reserve for actual position and
fresh projection, rather than repeated nearly identical nominal goals.

## Third candidate: QR succeeded, head orientation did not stabilize

Evidence prefix: `candidates/002_survey_candidate_0002/camera_lidar_attempt_00`.

All **54/54 results were fresh**; 51 frames were associated and 49 provided
candidate-bound QR_002 samples. Identity was latched. There was no QR-size
disagreement rejection.

| Final per-frame geometry outcome | Count |
| --- | ---: |
| Planar head-axis ambiguity | 27 |
| Yaw uncertainty above 3° | 23 |
| Head proposal ambiguity | 2 |
| Usable individual 3D head fit | 1 |
| Head association rejection | 1 |

The single usable fit had yaw **4.984°**, yaw standard-deviation estimate
**2.989°**, reprojection RMSE **0.539 px** and raw support 1.0. It was then
rejected by temporal consistency: normalized corner displacement **0.05839**
exceeded **0.04**. Thus no axis samples accumulated, despite repeated QR reads.

Recorded head borders alternate between roughly 113.5 px and 107.6 px edge
lengths. Bottom corners shift between outer y≈345.5–347.8 and inner
y≈339.6–343.3; one right edge shifts from x≈380.7 to x≈374. All 51 recorded
head fits reported raw support 1.0 and verified outer borders. Strong edge
support therefore does not by itself identify the same physical rail over time.

There is also genuine near-frontal pose ambiguity. One frame's same current
corners support yaw **4.751° at 0.520 px RMSE** and **10.043° at 0.759 px RMSE**.
Their residual gap is only 0.239 px, smaller than the 0.75 px noise floor.
Both overlays can look well aligned. Non-ambiguous uncertainty rejections span
**3.008052–3.435099°**, near the current 3° cutoff in several cases. The covariance
is explicitly a local pixel-noise model requiring hardware calibration, not
validated physical accuracy.

A read-only replay of 12 saved frames decoded QR_002 with a valid quad in all
12. Using the first outer-border fit only to locate a fresh current-pixel fit
stabilized corners around x=263.6–380.7, y=232–348, but produced **0/12 usable
angles**: nine planar-ambiguity and three uncertainty rejections (3.025–3.056°).
The original cold path produced one usable fit out of 12. Border stabilization
alone consequently does not solve the near-frontal observability policy.

### Premature recovery exit

The observer successfully wrote an advisory receipt and returned zero after
approximately 12 seconds. Its 30-second stopped front-recovery period still
had approximately **18 seconds remaining**.

On the final frame, candidate acquisition failed association: eight proposal
filters reported `ambiguous_registered_camera_clusters`. The nominal projected
LiDAR cone still contained one beam (index 0, 0.504 m). No current bound head/QR
qualified this frame for front recovery. `FrontViewRecovery.observe()` returned
false for the current miss even though its existing deadline had not expired.
Previously accumulated advisory evidence then allowed
`_maybe_commit_inspection_progress()` to write an `unobservable` result and set
the observer completed. This is an early advisory-exit defect, not a camera
timeout or lost QR identity.

Code: `scripts/aufgabe04/real_robot/observer/front_view_recovery.py:122–147` and
`scripts/aufgabe04/real_robot/observer/node.py:2552–2659`.

## Assessment of removing ambiguity and uncertainty checks

An accurately aligned overlay supports **current head acquisition**, but it
does not uniquely constrain the 3D orientation of a near-frontal planar head.
The two fits above demonstrate this using the same pixels. Removing both checks
would select one plausible angle without accounting for the alternatives and
would leave temporal instability, receipt consistency and early-exit defects
unresolved.

The current binary thresholds are also not the only possible policy. A more
useful separation would be:

1. Accept and retain a fresh, complete, uniquely associated head observation and
   its QR identity/face classification without claiming a precise orientation.
2. Retain competing current-pixel pose hypotheses and their uncertainty; jointly
   compare complete physical-border hypotheses across the stopped window.
3. Authorize a next pose only when all retained plausible orientations, with
   an appropriate uncertainty allowance, support acceptable clearance, a
   collision-free route and a useful QR view. Otherwise request one purposeful
   viewpoint adjustment.

For illustration, the 5.292° difference above changes an ideal opposite-side
goal by approximately **3.23 cm at 0.35 m standoff**, or **4.62 cm at 0.50 m**,
assuming a fixed exact stand center. This excludes localization, center and
pixel-noise uncertainty, so it is not a route-clearance proof. Some layouts may
tolerate the interval; others may not. The existing `HeadAxisInterval` is only
an observed/supplied hypothesis hull and explicitly does not authorize planning
or motion (`observer/head_temporal_consistency.py:54–69`).

## Recommended correction and validation order

1. Preserve the bounded stopped-recovery period through transient proposal or
   association misses. Such misses must not count as new valid geometry or
   renew the deadline; motion, identity conflict and target changes still
   invalidate the evidence window.
2. Use one explicit orientation-readiness contract for consensus and receipt
   validation; retain backside appearance independently. Do not disguise an
   unstable angle by clamping its confidence to the receipt minimum.
3. Improve complete-border hypothesis selection and temporal correspondence.
   Instrument coherent-rail rejection to distinguish missing evidence from
   competing rails.
4. Add task-dependent bounded-orientation planning instead of discarding all
   ambiguity or requiring a single universally precise angle. Validate the
   uncertainty model against measured orientations before interpreting its
   numerical confidence as physical accuracy.
5. Make candidate-facing correction satisfy the fresh bearing gate with margin.

Regression evidence should include these original sequences: no early advisory
exit during active stopped recovery; separately preserved backside confidence;
no consensus advertised ready when its receipt will fail; stable physical
border selection or explicit retained alternatives; and interval-based route
acceptance only when every retained plausible orientation is supported. Any
hardware validation should separately measure angle error and verify final
candidate bearing. Offline replays here do not establish robot-level success.

## Artifact index

Remote run root:
`/home/amr_gruppe01/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260916T103525Z`.

- `candidate_goal_progress.json`, `candidate_selection.jsonl`,
  `station_segment_runs.csv`.
- Second/third `camera_lidar_attempt_00` capture metadata, frame outcomes and
  inspection observations.
- Second `inspection_view_01_00/arrival/candidate_arrival_admission.json` and
  `alignment_01/arrival/`, `alignment_02/arrival/` arrival artifacts; corresponding
  `route/pipeline_summary.json` and `inspection_view.json`.
- Second/third `inspection_history/revision_001.json`.
- Parent bundle `results/real_runs/<run>/terminal_run.log` and per-leg controller
  traces for `candidate_001_inspection_002` and `_003`.

Derived local replay diagnostics, containing no images:
`/tmp/a04_backside_103525_replay_summary.json`,
`/tmp/a04_backside_103525_rawrail_trace_summary.json`,
`/tmp/a04_audit_latest_third_replay_diagnostics.json`.
