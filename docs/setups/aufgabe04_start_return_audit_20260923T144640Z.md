# Start-return failure audit: 20260923T144640Z

Audited on 2026-09-23 through SSH alias `mii001` (reported hostname `mii0002`). Latest mission at inspection: `stand_explore_exact2_camera_all5_20260923T144640Z`, clean revision `3ad9b00aedfdbde523abd8f863c4b2b01fe817da`. The parent bundle ran **14:46:40–14:59:40 UTC / 16:46:40–16:59:40 CEST**, exiting 2.

**The robot found and stored Start, planned its return, and then rejected that return before motion because the full route could not satisfy its uncertainty budget.** The handoff implementation plans one long geometrically valid route without checking whether the same route can pass uncertainty admission. The final approach to the exact stored pose is the limiting location.

## What completed and what stopped

- Camera discovery completed with all five distinct identities: Start, QR_001, QR_002, QR_003, QR_004. Three records have facing geometry; two, including Start, use admitted QR observation poses. The sixth LiDAR candidate remains an unvisited keepout.
- Start correctly resolves to `survey_candidate_0001`, using the opposite-side QR observation pose. Its stored pose is approximately `(-1.545252, -0.314758, -0.261803)`; after projection into the new map frame the target is `(-1.482981, -0.311892, -0.271957)` in metres/radians. That frame correction is expected; no missing or mismatched Start identity was reported.
- The planned return starts at `(1.098227, 0.218679)`, follows four route vertices, and measures **2.719399 m**. Planning and route sealing succeeded.
- The mission authorization explicitly includes `return_to_start` and records operator confirmation `RUN`. The return was not blocked by a new operator prompt or a legacy authorization scope.
- At **14:59:31 UTC**, the return dry run started. All **16 sensor/localization/ownership readiness observations passed**. The saved preflight JSON consequently says `ok=true`; this covers readiness, not the subsequent odom route admission.
- At **14:59:38.983 UTC**, odom execution admission rejected the route: `route uncertainty budget exhausted: limiting_segment=segment:0002:0048 remaining_margin=-0.223007 m`.
- Only a dry-run return event sequence exists. The ledger reports return distance **0**, duration **0**, and `motion_published=false`. No return motion permit or execute preflight was produced. The mission summary's broader `motion_published=true` describes the earlier exploration legs.
- Camera artifacts remain stored. `start_pose_reached`, `fastapi_request_ready`, and `fastapi_request_sent` are all false.

## Why the route failed

The limiting interval is the **last 4.91 mm before the exact Start endpoint**. Its static-map/arena clearance lower bound is 0.484564 m. The independently calculated admission terms are:

| Clearance budget term | Metres |
| --- | ---: |
| Robot radius | 0.105000 |
| Collision margin | 0.020000 |
| Tracking allowance | 0.030000 |
| Odometry drift allowance | 0.020000 |
| Faster-driving braking/latency reserve | 0.075000 |
| Position uncertainty | 0.138348 |
| Heading uncertainty | 0.319223 |
| **Total required** | **0.707571** |
| **Available** | **0.484564** |
| **Remaining margin** | **−0.223007** |

The stationary five-sample envelope records position sigma **0.069174 m** and yaw sigma **0.058249 rad / 3.337 degrees**, with multiplier 2. The heading term grows with distance from the leg's frozen localization anchor:

`2 × 0.0582487514 × 2.7401734486 = 0.3192233642 m`.

This is the current conservative admission model, not evidence of a measured 32 cm positioning error or a collision. The recorded stationary map-to-odom samples were stable.

The faster policy increases the braking reserve from **0.015 m to 0.075 m**, adding 0.060 m. Restoring the exploration reserve while holding the recorded route and covariance fixed would still leave **−0.163007 m**. Of 548 sampled budget entries, 64 fail with the faster reserve and 52 fail with the old reserve. Reducing speed alone therefore does not solve this run's admission failure. The saved policy correctly contains the requested 0.15 m/s and 0.60 rad/s caps.

The exact endpoint itself has 0.487018 m raw clearance; it fails even before the conservative sampling deduction. Changing smoothing or taking a different path while retaining the same frozen start anchor and exact endpoint cannot eliminate that endpoint deficit under this model.

## Implementation gap and correction

[Start-return orchestration](../../scripts/aufgabe04/real_robot/mission/start_return.py) lines 93–111 seals one entire return and submits it as a single child leg. [Stored-pose planning](../../scripts/aufgabe04/navigation/approach/admitted_pose_route.py) lines 248–255 and 318–320 checks the fixed 0.25 m static inflation and candidate keepouts, including during smoothing. It receives no covariance-aware admission context.

The later [localization admission](../../scripts/aufgabe04/navigation/station_segment/localization_admission.py) lines 495–517 adds the covariance, heading and speed reserves. [Heading computation](../../scripts/aufgabe04/navigation/execution/route_uncertainty_admission.py) lines 471–494 uses the frozen start anchor. The planner and execution admission therefore disagree on feasibility. The child rejects correctly; the handoff lacks a way to produce an executable return after that rejection. Existing candidate selection already performs an uncertainty feasibility check before choosing a candidate route, but the new return path bypasses that integration.

The appropriate correction is to evaluate the return against the same uncertainty model before committing it, then use separately admitted shorter legs when the complete return cannot pass. Intermediate stops must obtain genuinely fresh stationary localization, reproject the remaining target and candidate pool, and produce their own sealed route, dry admission and one-use permit. The current authorization explicitly describes **one** return leg, so its scope and tests must be updated deliberately for a bounded sequence; simply reusing the existing permit or numerically resetting the heading term would be invalid.

A calculation using this run's unchanged covariance and reserves finds a promising stop at the first intermediate vertex, `(-1.145, -0.015)`. The initial part has positive margin; reanchoring the remaining path there gives approximately **+0.0316 m** minimum margin. This is an offline feasibility diagnostic only: the actual second leg must pass new localization and admission after the robot stops. It is not hardware validation, a new certificate, or authorization to move. The exact stored Start endpoint and its final yaw should remain unchanged.

The handoff tests in [test_start_return.py](../../tests/aufgabe04/test_start_return.py) lines 123–148 substitute planner and motion effects. They cover orchestration, identity/provenance and failure reporting, but do not establish that a long return with realistic recorded covariance reaches the execution stage. Add an integration regression using this recorded geometry/covariance: whole-leg rejection, feasible bounded decomposition, fresh per-leg admission, preserved endpoint and no server readiness until verified final arrival.

## Evidence and audit boundaries

Evidence is under `results/implementation_checks/start_return_audit_20260923T144640Z/`. The retrieved 189 text artifacts (24,765,095 uncompressed bytes) all matched their workstation SHA-256 manifest. The local map image hash matches the route's recorded map hash.

`audit_return_uncertainty.py` in that directory reproduces all 548 recorded budget entries to a maximum absolute error of 1.12e-16 m and writes `return_uncertainty_audit.json`. It also records the unchanged-covariance two-leg calculation, including the positive original intermediate-corner margin. Re-run it with `python3 results/implementation_checks/start_return_audit_20260923T144640Z/audit_return_uncertainty.py` from the repository root. A final read-only workstation check confirmed that the same session remained latest, its failure was unchanged, and the checkout remained clean.

Primary files within that evidence root:

- `results/real_runs/stand_explore_exact2_camera_all5_20260923T144640Z/{manifest.txt,git_rev.txt,git_status.txt,terminal_run.log}`
- `results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260923T144640Z/mission_summary.json`
- The same session's `return_to_start/failure.json`, `return_to_start/route/{route.csv,route_diagnostics.json,target_evidence.json}`
- `preflight/stand_explore_exact2_camera_all5_20260923T144640Z_return_to_start_dry.json`
- `odom_execution/stand_explore_exact2_camera_all5_20260923T144640Z_return_to_start_dry_uncertainty_budget.json`
- `run_events/stand_explore_exact2_camera_all5_20260923T144640Z_return_to_start.jsonl`
- `motion_authorization/mission_leg_motion_authorization.json`, `station_segment_runs.csv`

This audit made no production-code changes, workstation writes, robot commands, or HTTP requests. Only local audit evidence and this report were created.
