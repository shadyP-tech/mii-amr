---
name: probabilistic-endpoint-model
description: >
  Build simple empirical probabilistic endpoint models for mii-amr TurtleBot
  repeated-run data. Use for endpoint prediction, covariance, Gaussian models,
  confidence ellipses, sampling, validation, and report wording for Aufgabe 2.
  Prefer 2D Gaussian final-position models over particle filters, SLAM, EKF/UKF,
  full trajectory uncertainty, or Aufgabe 03 map/obstacle uncertainty unless
  explicitly requested.
---

# Probabilistic Endpoint Model

## Purpose

Build a simple, defensible probabilistic model for TurtleBot endpoint prediction
in Aufgabe 2.

For current collected findings and summary values, read:

```text
.codex/skills/aufgabe03-navigation-mapping/references/current-findings.md
```

## Default Model

Use empirical endpoint uncertainty from repeated runs:

```text
p = [x, y]^T
p ~ N(mu, Sigma)

mu = mean(p_i)
Sigma = 1 / (n - 1) * sum_i (p_i - mu)(p_i - mu)^T
```

Treat yaw separately unless the user explicitly asks for a full `(x, y, yaw)`
model. For yaw, use circular mean/std if values may wrap around; otherwise use
simple mean/std and state the limitation.

## Scope Rules

- Keep the model introductory and report-defensible.
- Use collected repeated-run endpoint data as the source of truth.
- Prefer real camera-tracker endpoints for real-robot uncertainty.
- Use simulation endpoints separately when comparing simulation and real data.
- Do not propose particle filters, EKF/UKF, occupancy-grid SLAM, or full
  trajectory uncertainty unless explicitly requested.
- Do not reuse Aufgabe 2 endpoint models as Aufgabe 3 navigation or obstacle
  uncertainty models. For Aufgabe 3, only use this skill for endpoint summaries
  of repeated runs with the same route and coordinate frame.
- If the user says "path uncertainty", default to endpoint uncertainty and state
  that endpoint CSVs alone do not support full time-series trajectory
  uncertainty. Ask only if the user appears to need trajectory-level modeling.

## Expected Data

Inspect CSV schemas before assuming columns.

Primary real-run file:

- `results/aufgabe02/real_scripted_drive_runs.csv`
- preferred real columns: `tracker_final_x`, `tracker_final_y`,
  `tracker_final_yaw_deg`
- optional start-pose columns: `tracker_start_x`, `tracker_start_y`,
  `tracker_start_yaw_deg`

Primary simulation file:

- `results/aufgabe02/scripted_drive_runs.csv`
- typical simulation columns: `odom_final_x`, `odom_final_y`,
  `odom_final_yaw_deg`

Known summary files:

- `results/aufgabe02/probabilistic_endpoint_model_summary.csv`
- `results/aufgabe02/probabilistic_motion_primitives_model_summary.csv`
- `results/aufgabe02/primitive_path_prediction_summary.csv`
- `results/aufgabe02/supervisor_route_prediction_summary.csv`

Use only repeated runs from the same command, start condition, and coordinate
frame. Do not mix tracker, odometry, and Gazebo poses unless the frame
relationship is explicit.

## Required Response Contract

When answering modeling questions, structure the response as:

1. Recommended model
2. Required data
3. Computation steps
4. Validation method
5. Report interpretation
6. Limitations

## Workflow

1. Identify valid repeated runs and state `n`.
2. Extract endpoint positions `(x, y)` in meters.
3. Compute empirical mean endpoint `mu`.
4. Compute unbiased covariance matrix `Sigma` over `(x, y)`.
5. Optionally compute yaw mean/std in degrees, handled separately.
6. Plot a scatter of endpoints, the mean marker, and optional covariance ellipse.
7. Save generated plots to `results/aufgabe02/` unless the user asks otherwise.
8. If useful, sample predicted endpoints from `N(mu, Sigma)`.
9. Validate with held-out or later real runs and report residual distances.

Recommended plot filenames:

- `results/aufgabe02/real_endpoint_gaussian_ellipse.png`
- `results/aufgabe02/sim_vs_real_endpoint_scatter.png`

## Small-Sample Rules

- `n < 2`: do not compute covariance.
- `n == 2`: covariance exists but is not reliable.
- `n < 5`: descriptive only; show points and mean, but avoid strong claims.
- `5 <= n < 30`: covariance and ellipse are rough empirical estimates.
- Always say the model describes the tested command and setup, not all robot
  motion.
- If endpoints are nearly collinear or covariance is singular, use
  `np.linalg.pinv` instead of `np.linalg.inv` for Mahalanobis distance.

## Confidence Ellipse

Use the Gaussian ellipse:

```text
(p - mu)^T Sigma^-1 (p - mu) <= chi2_2(confidence)
```

Common values:

- 68%: `chi2_2 = 2.30`
- 95%: `chi2_2 = 5.99`
- 99%: `chi2_2 = 9.21`

Ellipse axes come from the eigenvectors of `Sigma`.
Axis lengths are `sqrt(eigenvalue * chi2_2(confidence))`.
The plotted ellipse width and height are twice these axis lengths.

## Python Guidance

Keep implementation close to lightweight analysis scripts such as
`scripts/aufgabe02/analyze_probabilistic_endpoint_model.py`.

- Prefer `csv`, `numpy`, and `matplotlib`.
- Use `pandas` only if convenient for the current task.
- Use `np.cov(points, rowvar=False, ddof=1)` for unbiased covariance.
- Use `np.linalg.pinv` for Mahalanobis distance when covariance may be singular.
- Use a fixed RNG seed when sampling predicted endpoints.

## Validation With Real Runs

Use this validation framing:

- Fit the model on an initial set of repeated real runs.
- Collect additional runs with the same start pose and scripted command.
- For each new endpoint, compute Euclidean residual from `mu`.
- Optionally compute Mahalanobis distance and whether it falls inside the 95%
  ellipse.
- Compare observed coverage with the nominal confidence level, but avoid strong
  conclusions with small `n`.

Report useful numbers:

- valid run count
- `mu_x`, `mu_y` in meters
- covariance matrix in `m^2`
- standard deviations in `x` and `y` in meters
- 95% ellipse axes in meters
- yaw mean/std in degrees if used
- residual distances for validation runs

## Report Wording

Use wording like:

```text
The endpoint prediction model was intentionally kept simple for Aufgabe 2. For
repeated executions of the same scripted TurtleBot motion, the final camera
tracker position p = [x, y]^T was modeled as a two-dimensional Gaussian
distribution. The mean endpoint mu represents the expected final position, and
the empirical covariance matrix Sigma represents the observed run-to-run spread.
Yaw was summarized separately because the available data supports a simple
endpoint uncertainty model rather than a full pose-distribution model.

This model assumes that the repeated runs are independent, were started from
approximately the same pose, and were measured in the same coordinate frame. It
does not model the full path, obstacle interactions, map uncertainty, or SLAM
uncertainty. Therefore, the result should be interpreted as a rough empirical
prediction for this specific scripted motion and experimental setup.
```

When interpreting results, state whether the ellipse is tight or broad relative
to the motion distance, whether any endpoint is an outlier, and whether the
sample size limits the strength of the conclusion.
