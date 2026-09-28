# Bounded first sensor delivery in the execution follower

## Recorded trigger

Run `stand_explore_exact2_camera_all5_20260928T124619Z` admitted the refreshed
0.45 m opposite-side route with a positive execution clearance margin. The
execution follower then stopped before motion at approximately 2.025 seconds:
`message_freshness`, sensor `odom`, `has_message=false`. Its isolated TF
executor heartbeat was healthy. Preflight had received fresh odometry, but
that did not establish delivery to the subsequently created follower.

Delayed discovery/first delivery is a hypothesis, not a proven transport root
cause. The recorded run does not show when that follower would have received
its first odometry message had it remained alive.

## Correction

The follower now offers a `cold_sensor_acquisition` phase after the ordinary
two-second startup window. It shares the existing three-second extra TF budget:
the absolute deadline remains five seconds from startup, with no renewal when
sensors arrive or TF acquisition begins. Existing command-line options and
their defaults remain compatible.

Extra sensor acquisition requires explicit evidence that the missing sensor
has never delivered a message, that every other required sensor is fresh, and
that both the follower and TF executors are servicing callbacks. Both scan and
odometry are examined even when the first one fails. Prior received-input loss,
invalid/stale/future input history, prior TF sampling, or continuity failure
cannot become a first-delivery extension. Startup continues publishing zero.

Readiness still requires current sensor timestamps, both required TF edges and
the existing frozen-map continuity check. Inputs and the absolute deadline are
rechecked after potentially blocking lookups. Publisher discovery and QoS
information is diagnostic evidence only; it cannot authorize motion.

The execution permit lifecycle is unchanged. This is a bounded wait inside the
same follower execution attempt, before nonzero motion. It does not create a
new execution attempt, reuse a consumed permit, or classify missing odometry
as a map-to-odom failure eligible for localization resealing.

## Modules and diagnostics

- `initial_sensor_acquisition.py`: first-delivery eligibility, bounded receipt
  counters and publisher diagnostics, separate from TF policy.
- `initial_tf_acquisition.py`: shared absolute budget and acquisition phases.
- `runtime_components/initial_runtime_inputs.py`: stopped orchestration and
  live rechecks.
- `runtime.py`: sensor receipt instrumentation and follower-executor heartbeat.

Existing `initial_tf_acquisition` evidence gains `sensor_acquisition`, including
per-sensor receipt count, first/last monotonic receipt times, current freshness,
failure history and follower-executor health. Phase transitions and terminal
failure capture matched publisher counts, requested/offered QoS, and at most
eight discovered endpoints per sensor. Graph errors are retained as diagnostics.
These graph queries are outside sensor callbacks and are followed by the same
deadline/input rechecks before readiness. The transition event is
`initial_sensor_acquisition_started`; existing readiness/stop events remain.

## Validation

ROS-free focused validation: **135 tests and 199 subtests passed**, covering
initial sensor/TF acquisition, isolated callback service, TF receipt evidence,
map-TF recovery classification, runtime stale-TF handling, follower safety,
certified startup, startup active localization and prestart localization reseal.

The recorded missing-odom failure shape is replayed with a hypothetical first
fresh delivery at 3.25 seconds: startup succeeds after TF and continuity checks.
Regressions also cover absent delivery, delivery at/after five seconds, delayed
TF after sensor arrival without budget renewal, stale/future first messages,
received-input loss, poisoned early freshness history, either executor failing,
blocking TF/graph operations and continuity rejection. No real robot run or
workstation deployment was performed for this correction.
