# Real-run audit: 20260928T121706Z

Audited 2026-09-28. Run `stand_explore_exact2_camera_all5_20260928T121706Z` on `mii001`, revision `4dc00c5d4121820499613968ef2ce571f4f7eda6`; the run bundle records a clean checkout. Local audit code has the same revision. No robot motion, ROS nodes, deployment, or production code changes were performed for this audit.

## Outcome

The mission failed during opposite-face route recovery for the second visited candidate, `survey_candidate_0001`. It never drove to that candidate's opposite side. Only `survey_candidate_0003` / `QR_003` was admitted, with facing geometry. Three candidates remained unvisited. The second candidate's QR identity is unresolved in this run; do not assign it a QR identity based on an older run.

Both LiDAR coverage stops completed. Both initial candidate-approach motions completed. The failure chain is:

| Opposite standoff | Recorded result | Motion |
| --- | --- | --- |
| 0.50 m | Route materialized; dry uncertainty admission rejected, margin **−0.094321 m** | None |
| 0.45 m | Route materialized; dry uncertainty admission rejected, margin **−0.021682 m** | None |
| 0.40 m | Static planning rejected goal cell `(25,24)` as not traversable | None |
| 0.35 m | Planner raised `ValueError`: offset must exceed **0.375355 m** after rasterization | None |

The last error escaped candidate-local recovery and became `mission_failure.json` / `failed_closed`. No opposite-side execute child or arrival camera observation exists. Return-to-Start was not reached.

## What worked in camera processing

The second observer ran for about 3.98 seconds including process overhead. Its final state is `backside_axis_committed_qr_unresolved`: seven retained backside samples, axis **1.479306 rad**, bounded half-width **0.128661 rad (7.37°)**. The receipt contains three stopped-scan reconciliation entries and metric head-position evidence. Projected planning center is **(−1.143912, −0.489246) m**, uncertainty **0.025120 m**. Projected opposite normal is **3.058376 rad**. Thus the previous missing-center/backside-handoff problem did not block this run.

The final image/scan freshness checks passed at approximately **164/144 ms**. Eight images were processed, associated, and supplied geometry; no QR was decoded from the backside. Ten scan witnesses expired awaiting TF, but sufficient evidence was committed. These expirations are a separate performance observation, not this stop's cause.

Both opposite dry children passed their general 16-observation preflight and stationary map←odom stability. Their later route-uncertainty admission failed. There is no evidence here of a map→odom synchronization failure or a camera race causing termination.

## Why the larger routes were rejected

The limiting sample is at the final route segment, near the opposite endpoint. The saved uncertainty artifact uses the uninflated static map plus arena boundary, not camera fit confidence, for this budget:

| Budget term | 0.50 m route | 0.45 m route |
| --- | ---: | ---: |
| Raw centerline clearance | 0.322533 m | 0.372528 m |
| Robot + collision + tracking + drift + braking | 0.190000 m | 0.190000 m |
| Position uncertainty, 2σ | 0.138520 m | 0.123566 m |
| Heading uncertainty contribution | 0.088334 m | 0.080645 m |
| Required clearance | 0.416853 m | 0.394210 m |
| Remaining margin | **−0.094321 m** | **−0.021682 m** |

The endpoints are `(−1.645, −0.465)` and `(−1.595, −0.465)` m. Moving closer to the stand improved static clearance, but the second route still lacked 21.7 mm under its recorded covariance and heading allowance. A passing generic localization-readiness threshold does not imply that this particular route fits its uncertainty budget.

## Recovery defect

`real_robot/candidate/approach.py::_move_certified_opposite_face` calls `bounded_approach_offsets` using only `minimum_active_standoff_m = 0.33`. This yields `0.50, 0.45, 0.40, 0.35, 0.33` m.

`navigation/approach/candidate_preapproach_compute.py::validate_approach_outside_transit_keepout` separately requires strict separation above `0.34 + 0.05 / sqrt(2) = 0.375355` m. Consequently, the fallback generates distances that the planner categorically rejects. `is_approach_feasibility_failure` recognizes typed unreachable errors and two legacy messages, but not this `ValueError`; it is re-raised. The controller catches `CandidateInspectionRouteUnavailableError` to continue bounded view recovery, so that path is bypassed and the whole mission ends.

The generic inspection branch already has `bounded_inspection_standoffs`, which accounts for the base raster floor. The opposite branch does not use it. Reusing that floor prevents this exception, but **does not by itself produce an admissible opposite route**.

## Offline replay: why 0.40 m was blocked

The ROS-free production planner was run locally with the recorded projected candidate snapshot, coverage plan, center estimate, physical clearance, normal, and the matching map bundle. It reproduced both successful static plans, the 0.40 m rejection, and the 0.35 m exception.

The `(25,24)` goal is a `station_keepout` cell at `(−1.545, −0.465)` m. The validated current target occupies cell `(33,24)`. Its keepout radius is `0.34 + 0.025120 = 0.365120` m, rasterized upward to **8 cells / 0.40 m**. The goal is exactly eight cells away and remains blocked. This is the existing conservative target keepout, not a newly detected obstacle.

Additional bounded geometry probes:

- `0.42`, `0.43`, and `0.44` m choose the same endpoint as `0.45` m.
- `0.41` m remains blocked at `(25,24)`.
- `0.375357` m, just above the base raster floor, has only blocked/below-minimum candidates.

These probes establish static planning behavior only. They do not authorize motion or prove that a changed route would pass fresh live admission. Finer distance stepping alone is insufficient for the recorded geometry.

## Recommended correction

1. Share one bounded standoff policy between generic and opposite-face recovery. Include physical minimum, raster floor, and validated-center uncertainty; exclude invalid proposals before calling the planner. Use typed candidate-local infeasibility for exhausted search, while preserving fatal errors for malformed evidence/configuration. Persist the complete rejection chain.
2. When no opposite route passes, return through the existing bounded candidate-view recovery rather than ending the mission with a raw configuration exception. Preserve the retained backside orientation and center; do not replace them with a new angle fit.
3. Address the actual clearance deficit separately: use route-specific stationary localization convergence, or a certified staged route with fresh stationary admission before the final segment. The latter can shorten the heading lever arm, but needs its own clearance and frame-evidence validation. Neither should lower the uncertainty multiplier or ignore keepouts merely to pass.
4. Add a regression using this exact chain: 0.50/0.45 dry uncertainty rejection → 0.40 blocked goal → no invalid 0.35 request → bounded route-unavailable recovery with no motion authorization. Include current-center uncertainty and rasterization; mocked route success alone would miss this defect.

No single command-line offset change is demonstrated to fix this run.

## Evidence

Local evidence root: `results/implementation_checks/run_audit_20260928T121706Z/` (ignored diagnostic artifacts).

- `audit_summary.json`: compact result, budget terms, observer freshness, and replay conclusions.
- `replay_geometry.py`: reproducible ROS-free geometric probe; execute from repository root with `PYTHONPATH=.` and a Python environment containing repository dependencies.
- `results/real_runs/stand_explore_exact2_camera_all5_20260928T121706Z/`: copied command, revision, clean status, manifest, terminal log.
- Nested autonomous-exploration run: `mission_failure.json`, `candidate_goal_progress.json`, `station_segment_runs.csv`, `candidate_selection.jsonl`, `odom_execution/*opposite*uncertainty_budget.json`.
- Second candidate: `camera_lidar_attempt_00/axis_observation.json`, `observer_status.json`, and `inspection_opposite_01/opposite_face_source*/route_diagnostics.json`.

Copied metadata comprises the full latest-run JSON/JSONL/CSV evidence. Raw image captures and PNGs were not needed to establish this planning failure.
