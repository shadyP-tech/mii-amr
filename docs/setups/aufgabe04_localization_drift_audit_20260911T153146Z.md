# Runtime localization drift audit — 2026-09-11 15:31:46 UTC

The latest retrieved run, `stand_explore_exact2_camera_all5_20260911T153146Z`, stopped before camera observation. The first candidate approach and its single permitted replacement both triggered the global-localization consistency monitor. The mission then correctly enforced its configured recovery limit and failed closed.

The deeper issue is the monitor's origin-dependent translation metric: it measures the change in the translation column of `map ← odom`, rather than displacement at the robot or certified route. With the robot about 5.2 m from the odometry origin, a small yaw correction produces a large change in that translation column. The recorded 9.95–18.61 cm values correspond to about 1.37–1.79 cm when the same transforms are evaluated at the last recorded robot position.

## Run outcome and provenance

- Run commit **`a78d94f`**, including the shared planning-frame reader correction. Saved Git status is clean; local and remote HEAD matched at retrieval.
- Parent execution **15:31:46–15:37:18 UTC / 17:31:46–17:37:18 CEST**, exit code **2**.
- Retrieved through `mii001`; saved bundle hostname is `mii0002`.
- **306 original files** verified against the retrieval manifest, with no changed-while-reading files. Snapshot: `2026-09-11T15:38:40.690188+00:00`.
- Both LiDAR survey points were recorded. The first coverage leg needed one localization recovery; the second completed directly. Modelled coverage reached **95.31%**.
- The registry retained **five hypotheses**: two pending camera validation and three provisional. These are not five confirmed stand identities.
- Candidate **`survey_candidate_0003`** was selected. No camera observer, recommendation or camera decision was produced; confirmed QR count remains **0**.
- Recorded command used **0.50 m** candidate approach offset and **`--max-runtime-localization-reseals-per-leg 1`**.

The shared-reader correction was present, but its camera-result commit path was never reached. This run does not test that correction on hardware or supply new evidence about head fitting, QR detection or backside classification.

## Timeline

| UTC | Event | Result |
| --- | --- | --- |
| 15:34:12.966 | First coverage leg stops | Translation-column drift 0.10814 m exceeds 0.10082 m. |
| 15:34:40.151 | Coverage replacement completes | First survey point becomes available. |
| 15:35:46.125 | Second coverage leg completes | LiDAR survey handoff completes. |
| 15:36:42.462 | First candidate approach stops | Translation-column drift 0.18614 m exceeds 0.15000 m. |
| 15:36:55.657 | Replacement dry run passes | Fresh stationary localization, route and authorization evidence admitted. |
| 15:37:16.425 | Candidate replacement stops | Translation-column drift 0.09946 m exceeds 0.07879 m. |
| 15:37:18 | Parent exits | One replacement attempt consumed; recovery budget exhausted. |

The initial candidate attempt spent about **11.02 s** in execution. All its recorded linear velocity commands were zero; it was turning to align with the approach segment. Its approximately **3.65 mm** travel estimate is odometry, not evidence of a commanded translation.

The replacement ran for about **7.88 s**, with approximately **0.210 m** recorded travel. Its last control cycle was **0.0752 m** from the target, with **0.00288 m** route deviation inside the **0.03 m** tracking tube. Its last front clearance was **0.568 m**, above the **0.20 m** stop threshold. These are last-cycle diagnostics, not proof of conditions after that sample; the recorded stop reason is localization consistency, not obstacle clearance or route tracking.

The terminal error is:

```text
candidate runtime localization reseal budget exhausted after 1 replacement attempt(s)
... global localization consistency requires zero and reseal
```

`mission_failure.json` records `failure_phase=candidate_runtime_localization_recovery`, `budget_exhausted`, and `motion_continues_authorized=false`. Recovery explanations appear later in the terminal because of buffering; timestamped child events establish their actual order.

## Why the translation metric is misleading here

For a planar transform, `T(p) = R p + t`. The current translation check uses `||t_live - t_frozen||`, which is the displacement of the odometry origin under the two transforms. At a route point `p`, the displacement is:

```text
|| (R_live - R_frozen) p + (t_live - t_frozen) ||
```

The rotation term can cancel much of the translation-column change near the robot. Changing the odometry origin changes the first quantity even when the physical robot, route and map relationship stay the same. This makes the current stop decision sensitive to the coordinate origin.

| Stop | Reported origin shift | Translation limit | Shift at last recorded robot point | Yaw change |
| --- | --- | --- | --- | --- |
| Coverage 000 | 10.81 cm | 10.08 cm | **1.07 cm** | −1.61° |
| Candidate initial | 18.61 cm | 15.00 cm | **1.79 cm** | +1.95° |
| Candidate replacement | 9.95 cm | 7.88 cm | **1.37 cm** | −1.25° |

The robot's recorded odometry distance from its origin was about **4.05 m**, **5.24 m** and **5.17 m**, respectively. Each yaw change remained within its separate yaw limit. The final translation threshold was derived from the new preflight's position covariance, approximately `2 × 0.03939 m = 0.07879 m`; the initial candidate threshold was capped at 0.15 m. Stable preflight data therefore did not prevent a later origin-sensitive stop.

The robot-point calculations use the last recorded control-cycle odom position and the frozen/live transforms from the stop event. They are not exactly simultaneous observations and are not physical ground truth. They diagnose how the metric responds to the saved geometry; they do not independently authorize continued motion.

### Deterministic route replay

The [offline replay](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T153146Z/drift_metric_replay.py) reproduces all three stop decisions through the unchanged production continuity function. It verifies the saved certificates, uncertainty bindings and map/odom route hashes, then applies the two transforms to each certified route vertex:

| Stop | Maximum displacement at certified route vertices |
| --- | --- |
| Coverage 000 | **1.573 cm** |
| Candidate initial | **2.813 cm** |
| Candidate replacement | **1.655 cm** |

For each straight route segment, the norm of displacement is convex along the segment, so these vertex maxima also bound its centerline. They do not include the robot footprint. Unlike the last robot-point estimate, this calculation does not depend on pairing a controller pose with the stop timestamp.

Rebasing the odometry origin to the frozen certified route start preserves physical route points within **1.12×10⁻¹⁵ m**, but flips all three unchanged production continuity decisions to acceptance. This proves coordinate-origin sensitivity. It is a diagnostic coordinate change, not a reset or a proposed way to bypass admission.

The [origin metric](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/navigation/localization/odom_route_adapter.py:625) has no route anchor. Its [covariance-derived limits](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/navigation/station_segment/localization_admission.py:271) are being compared to this origin displacement, whereas the [route uncertainty heading reference](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/navigation/station_segment/localization_admission.py:508) is the certified route start.

## Localization and transport evidence

All relevant execute and stationary preflights passed. The stopped candidate recovery's five AMCL samples had position spread **0.0055 m** and yaw spread **0.0121 rad**. The replacement execute preflight had position spread **0.0022 m** and yaw spread **0.0046 rad**. Stationary admission succeeded; the problematic drift comparison happened after motion began.

All three stop-event TF samples explicitly passed freshness validation. Their ages were approximately **−0.93 to −0.91 s**, within the recorded **1.1 s** future allowance. These stops were not missing/stale-TF startup failures.

Coverage leg 001 did show burst delivery into its TF buffer: its last eight transform headers spanned **0.658 s**, while recorded buffer-entry times spanned **0.00248 s**. Bounded startup recovered, becoming ready after **3.651 s**. Candidate initial/replacement startup became ready after **0.709 s / 0.964 s**. The tracing measures buffer insertion, not DDS receipt, so it cannot identify the transport cause.

Ownership observations report no external or ambiguous TF owner candidates, and node snapshots list one `/amcl`. TF graphs only identify `default_authority`; no per-publisher trace or live AMCL parameter dump was saved. Duplicate broadcasting and the precise source of AMCL corrections therefore cannot be conclusively diagnosed from this run.

## Recommended correction

Replace the origin-dependent translation admission with displacement at a **fixed, certificate-bound route anchor**, retaining the separate yaw limit and the full route/footprint bound. A suitable anchor is the frozen odom representation of the certified map-route start, matching the existing heading reference. For any route point, displacement is bounded by anchor displacement plus absolute yaw correction times distance from that anchor; the [existing footprint lever-arm contract](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/navigation/execution/route_uncertainty_admission.py:45) must remain part of that accounting.

The same physical transforms and route must produce the same decision after a common coordinate-origin change. Update the metric/version, anchor binding, monitor, uncertainty/margin accounting, certificate evidence, route revisions and recovery validator together so their displacement definitions agree. Preserve the interpretation of existing certificates rather than silently applying a new metric to legacy evidence.

Retain explicit yaw, freshness, localization consistency, route-clearance and bounded-recovery checks. A point-only test at the current robot position is insufficient for certifying the remaining route. Increasing the retry budget would repeat the existing origin-sensitive check.

Regression evidence should cover the three recorded stops, coordinate-origin invariance, pure translation, pure rotation, coupled rotation/translation, real route/footprint displacement beyond the bound and malformed or stale input. Record frozen/live transforms and displacement at the exact odom sample and certified route support points when stopping, so future audits can distinguish coordinate effects from motion-relevant localization changes.

Audit only: no production correction, ROS execution, deployment or robot motion was performed.

## Evidence

- [Run summary and original integrity checks](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T153146Z/audit_run.json)
- [Summary script](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T153146Z/audit_run.py)
- [Drift metric replay results](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T153146Z/drift_metric_replay.json)
- [Mission failure](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T153146Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T153146Z/mission_failure.json)
- [Terminal log](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T153146Z/results/real_runs/stand_explore_exact2_camera_all5_20260911T153146Z/terminal_run.log)
