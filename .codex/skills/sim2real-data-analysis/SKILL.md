---
name: sim2real-data-analysis
description: >
  Analyze mii-amr AMR Sim2Real experiment CSVs, pasted CSV output, tracker pose
  CSVs, and analysis scripts after simulation or real TurtleBot runs. Use for
  endpoint statistics, simulation-vs-real comparison, paired run comparison,
  Aufgabe 03 waypoint/replan log interpretation, outlier checks, uncertainty
  estimates, plots, and report-ready interpretation.
---

# Sim2Real Data Analysis

## Purpose

Analyze `mii-amr` simulation and real TurtleBot scripted-drive results and
Aufgabe 03 waypoint/replan logs, then produce concise, report-ready conclusions.

Use when the user provides:

- simulation or real run CSVs
- pasted CSV/table output
- tracker pose CSVs
- analysis scripts
- questions about endpoint deviation, uncertainty, plots, or report wording
- Aufgabe 03 waypoint-following, mapping coverage, AMCL/TF failure, or LiDAR
  obstacle-replan result logs

## Expected Files

Primary:

- `results/aufgabe02/scripted_drive_runs.csv`
- `results/aufgabe02/real_scripted_drive_runs.csv`
- `results/aufgabe02/probabilistic_endpoint_model_summary.csv`
- `results/aufgabe02/probabilistic_motion_primitives_model_summary.csv`
- `results/aufgabe02/primitive_path_prediction_summary.csv`
- `results/aufgabe02/supervisor_route_prediction_summary.csv`
- `results/aufgabe03/aufgabe03_waypoint_follow_runs.csv`
- `results/aufgabe03/arena_coverage_runs.csv`

Fallback:

- `results/aufgabe03_waypoint_follow_runs.csv`
- `results/aufgabe02/latest_tracker_pose.csv`
- `results/aufgabe02/real_start_pose_checks.csv`

For current known findings, read:

```text
.codex/skills/aufgabe03-navigation-mapping/references/current-findings.md
```

## Rules

- Always inspect the CSV schema before assuming column names.
- Treat CSVs as the source of truth.
- Work read-only unless the user explicitly asks to modify files.
- Keep odometry pose, Gazebo pose, and camera-tracker pose distinct.
- Always state units and coordinate frame if known.
- Do not compare poses from different coordinate frames unless the frame
  relationship is clear.
- Do not overstate conclusions from small sample sizes.

## Workflow

1. Identify available CSVs or parse pasted data.
2. Inspect column names and infer final-pose fields.
3. Filter to valid runs where possible.
   - Prefer explicit validity/status columns.
   - If none exist, drop rows missing final pose values and state that
     assumption.
4. Separate simulation data from real data.
5. Detect whether simulation and real runs are paired.
   - If paired, compute per-pair `dx`, `dy`, endpoint error, and yaw error.
   - If unpaired, compare dataset means and state that the comparison is
     unpaired.
6. Compute applicable statistics.
7. Identify potential outliers using transparent rules.
8. Produce report-ready interpretation.

For Aufgabe 03 logs:

1. Inspect schema first; older rows may lack replan diagnostic columns.
2. Count `completed`, `failed`, `timeout`, and `blocked` statuses.
3. Separate setup failures from navigation failures.
4. For completed rows, report `reached_count`, final pose, scan min/p05, AMCL
   covariance, and controller parameters.
5. For replan rows, report obstacle counts, map/path artifacts, old vs new path
   lengths, replan trigger, and whether artifact-only mode was used.
6. Do not claim active map-update validation from completed rows that lack
   `run_local_map_yaml` or `run_local_waypoints_csv`.

## Metrics

Compute where possible:

- number of valid simulation runs
- number of valid real runs
- mean final `x`, `y`, and `yaw`
- standard deviation of `x`, `y`, and `yaw`
- simulation mean vs real mean
- `dx = real_mean_x - sim_mean_x`
- `dy = real_mean_y - sim_mean_y`
- Euclidean endpoint error: `sqrt(dx^2 + dy^2)`
- yaw error normalized to `[-180, 180]` degrees
- endpoint residuals from each dataset mean
- potential outliers
- standard error where useful
- optional covariance over `(x, y)`
- optional 95% confidence interval only when sample size is sufficient

## Yaw Handling

For yaw, use circular handling when values may wrap around. Normalize yaw
differences with:

```text
((real_yaw - sim_yaw + 180) % 360) - 180
```

## Outlier Handling

Use reproducible rules:

- Endpoint residual: distance from that dataset's mean endpoint.
- If enough runs exist, flag likely outliers using `abs(z) > 2` for `x`, `y`,
  `yaw`, or endpoint residual.
- With small `n`, call them potential outliers and avoid strong claims.

## Uncertainty Handling

Prefer simple empirical estimates:

- standard deviation for spread
- standard error: `std / sqrt(n)`
- optional covariance over `(x, y)` for endpoint ellipses
- optional 95% confidence interval: `mean +/- 1.96 * SE` only when justified

## Small Sample Rule

- If `n < 5`, use descriptive statistics only.
- If `5 <= n < 30`, treat confidence intervals as rough.
- Do not claim general Sim2Real performance beyond the collected runs.

## Default Plots

Recommend these when useful:

- scatter plot of final endpoints: simulation vs real
- mean endpoint marker for each dataset
- optional covariance ellipse
- endpoint residual plot or histogram
- paired `dx`/`dy` plot if paired runs exist
- yaw comparison if yaw data is reliable
- Aufgabe 03: table of run status, reached waypoints, scan clearances,
  AMCL variances, and replan artifacts

## Output Contract

Always structure the response as:

1. Data quality check
2. Numerical summary
3. Interpretation
4. What to plot
5. What to write in the report

For Aufgabe 03 navigation logs, replace "simulation vs real" language with:
setup quality, navigation outcome, obstacle/replan evidence, and report wording.

## Units

Use units:

- `x`, `y`, `dx`, `dy`, endpoint error: meters
- `yaw`, yaw error: degrees

## Report Guidance

Be concise and specific. State:

- how many valid simulation and real runs were analyzed
- whether real endpoints are biased relative to simulation
- whether spread is small or large relative to the mean offset
- whether yaw differs meaningfully
- whether outliers affect the conclusion
- what uncertainty remains
- for Aufgabe 03, whether the data supports static navigation, active obstacle
  detection, active replanning, or only failed setup evidence

Separate method, result, and interpretation. Do not claim general Sim2Real
performance beyond the collected runs.
