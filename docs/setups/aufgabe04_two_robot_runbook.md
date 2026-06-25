# Aufgabe 04 Two-Robot Runbook

Draft placeholder. Do not run two robots together until station locks,
namespaced topics, `/cmd_vel` ownership checks, localization checks, and
shared-course conflict tests pass in dry run or simulation.

Before any two-robot physical run, each robot must first have a successful
single-robot station-segment dry run and a bundle-wrapped real run. Use
`scripts/common/run_with_bundle.sh` with per-robot namespace/topic/frame options
that match the wrapped command, and keep each bundle under a distinct
`results/real_runs/<run_id>/` directory.

The run bundle is raw/debug evidence only. It does not replace station locks,
fleet conflict checks, strict ROS preflight, `/cmd_vel` ownership checks,
operator confirmation, or physical stop readiness.
