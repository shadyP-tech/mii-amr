# Current Experiment Findings

Use this as a compact evidence index for `mii-amr` report work. Verify current
CSV schemas before computing new values. File-backed findings reflect files
present in the repo when the skill was updated; items labeled as pasted-run
evidence come from user-provided terminal logs and should be rechecked against
copied CSV/artifact files before final reporting.

## Aufgabe 03 Task Sheet

Source: `docs/tasks/Aufgabe_03.pdf`.

The assignment period is 2026-05-12 to 2026-06-02. The required work is mapping
and SLAM familiarization, navigation in the saved map, custom path planning,
LiDAR obstacle extraction from raw sensor data, active map update during
driving, collision-avoiding navigation on the planned path, temporary-obstacle
validation, and analysis of sensor/mapping failure modes.

## Aufgabe 03 Current Evidence

Map artifacts:

- `maps/aufgabe03/arena_1p898x3p9_auto.yaml`
- `maps/aufgabe03/arena_1p898x3p9_auto.pgm`
- map resolution `0.05 m/cell`, origin `[-2.82, -1.69, 0]`, size `113 x 71`

Offline A* path artifacts:

- `results/aufgabe03/aufgabe03_planned_path.csv`
- `results/aufgabe03/aufgabe03_waypoints.csv`
- `results/aufgabe03/aufgabe03_planned_path.ppm`
- current planned path starts near `(0.005, -0.015) m`, ends near
  `(1.005, 0.285) m`, has 21 dense path cells, 4 simplified waypoints, and
  length about `1.124 m`
- `results/aufgabe03/test_path.csv` and `waypoints_test_path.csv` are alternate
  path artifacts ending near `(-1.495, 0.385) m` with length about `1.666 m`

Mapping coverage:

- `results/aufgabe03/arena_coverage_runs.csv` has 3 rows
- first two attempts failed because `/scan` was unavailable
- `arena_coverage_001` completed after `/scan` was available; final odometry was
  about `x=-0.0098 m`, `y=-0.0248 m`, `yaw=26.27 deg`

Waypoint following:

- old root log `results/aufgabe03_waypoint_follow_runs.csv`: 11 failed setup
  runs, mostly `/amcl_pose` timeouts and stale TF
- current log `results/aufgabe03/aufgabe03_waypoint_follow_runs.csv`: 18 rows,
  with 15 failures and 3 completed slow follower runs
- common failures: `/amcl_pose` startup timeout, stale TF ages from about
  `1.0 s` to `8.1 s`, and waypoint 1 timeouts
- successful rows used slower settings:
  `linear_speed_mps=0.03`, `min_linear_speed_mps=0.01`,
  `linear_gain=0.25`, `max_angular_speed_radps=0.12`, `yaw_gain=0.5`
- completed slow runs reached all 3 waypoints; final poses were approximately
  `(0.898, 0.254, -1.5 deg)`, `(0.921, 0.207, 15.1 deg)`, and
  `(0.926, 0.273, -14.4 deg)`
- completed rows currently lack newer replan diagnostic columns and obstacle
  artifacts, so they support static waypoint navigation, not active obstacle
  replan claims

LiDAR obstacle replanning:

- implementation exists in `scripts/aufgabe03/lidar_obstacle_map.py`,
  `scripts/aufgabe03/follow_planned_waypoints.py`, and
  `scripts/aufgabe03/two_stage_waypoint_run.py`
- `docs/setups/nav2_waypoint.txt` documents dry-run, artifact-only, and active
  temporary-obstacle validation modes
- no recorded active replan artifact files were present in `results/aufgabe03`
  at update time, so repo-local CSV evidence still needs fresh copied artifacts
- recent pasted-run evidence: run `shadow_approach_fallback_pad015_005`
  completed arena-active recovery and late-obstacle navigation with live LiDAR
  replanning. The command used `--heater-wall-side=+x`,
  `--arena-force-short-wall-side axis_positive`,
  `--arena-force-short-wall-type heater`,
  `--no-arena-active-temporary-map-viz`,
  `--no-arena-active-explore-path-viz`,
  `--run-local-map-initial-scan-mode none`,
  `--run-local-map-update-mode full`, `--max-replans 4`, and
  `--follower-min-scan-range-m 0.30`.
- the same pasted logs showed active-explore selecting obstacle-shadow frontier
  candidates, then open-corridor recovery, followed by custom waypoint
  following, live `LiDAR obstacle replan completed` events, existing-map
  corridor repairs, and final route completion
- caveat: the successful run still emitted AMCL localization warnings such as
  `missing_amcl`; interpret it as successful TF/odom-based completion while
  noting that AMCL pose feedback was not fully healthy

## Aufgabe 02 Current Evidence

Primary files are under `results/aufgabe02/`.

Straight 30 cm endpoint model:

- `probabilistic_endpoint_model_summary.csv` reports 30 valid real runs
- endpoint mean: `x=0.19625 m`, `y=0.05187 m`
- endpoint std: `x=0.00234 m`, `y=0.00273 m`
- 95 percent endpoint ellipse axes: major `0.01415 m`, minor `0.01047 m`
- motion error mean: `dx=0.00181 m`, `dy=-0.00185 m`
- motion error std: `x=0.00085 m`, `y=0.00177 m`
- yaw mean/std: `-179.88 deg` / `0.50 deg`
- reported Sim2Real bias: `dx=-0.00605 m`, `dy=-0.00185 m`, magnitude
  `0.00633 m`

Early five-run real tracker repeatability:

- `real_run_statistics.txt` reports 5 runs
- tracker final pose std was about `0.0022 m` in both `x` and `y`
- net tracker displacement was about `0.219 m` to `0.223 m`
- yaw changed by about `-16 deg` to `-18 deg`

Motion primitives:

- `probabilistic_motion_primitives_model_summary.csv` includes F30, F50, CW45,
  CCW45, CW90, CCW90, and CCW180 primitives
- F30: `n=30`, mean local delta `(0.30181, -0.00185) m`, yaw delta
  `0.32 deg`, std about `(0.00085, 0.00177) m`
- F50: `n=15`, mean local delta `(0.50132, -0.00173) m`, yaw delta
  `0.80 deg`, std about `(0.00089, 0.00315) m`
- CW45: `n=15`, yaw delta `-42.99 deg`, yaw std `0.28 deg`
- CCW45: `n=15`, yaw delta `43.22 deg`, yaw std `0.40 deg`
- CW90: `n=15`, yaw delta `-84.98 deg`, yaw std `1.97 deg`
- CCW90: `n=15`, yaw delta `84.57 deg`, yaw std `2.35 deg`
- CCW180: `n=15`, yaw delta `175.11 deg`, yaw std `0.45 deg`
- rotation primitives consistently undershot requested angles; 45 deg turns by
  about `1.8 deg` to `2.0 deg`, 90 deg turns by about `5 deg`, and 180 deg by
  about `4.9 deg`

Primitive path prediction:

- `primitive_path_prediction_summary.csv` uses actions
  `F30,CW90,F30,CCW90,F30`
- predicted endpoint mean about `(0.6121, -0.2856) m`
- endpoint std about `(0.0104, 0.0168) m`
- 95 percent ellipse axes about `0.0895 m` and `0.0361 m`
- final yaw mean/std about `0.60 deg` / `3.09 deg`

Supervisor route prediction:

- `supervisor_route_prediction_summary.csv` predicts a longer fixed-point route
  with 17 actions
- endpoint mean about `(3.9085, 0.1293) m`
- endpoint std about `(0.0210, 0.1002) m`
- 95 percent ellipse axes about `0.4976 m` and `0.0612 m`
- validation run `supervisor_validation_004` had residual magnitude about
  `0.1490 m` and was outside the model's 95 percent endpoint ellipse
- interpret this as a useful model-mismatch finding, not as general robot
  unreliability

## Reporting Cautions

- Treat CSVs and JSON summaries as source of truth; recompute before publishing
  final numbers.
- Treat pasted terminal logs as useful diagnostics until the corresponding
  generated CSVs and run-local artifacts are copied into `results/`.
- Keep camera-tracker pose, odometry pose, AMCL pose, and map-frame TF separate.
- Do not claim active obstacle map update validation until a run logs replan
  diagnostics and artifacts.
- Do not generalize Aufgabe 02 endpoint distributions to Aufgabe 03 navigation
  with obstacles; the dynamics and coordinate frames differ.
