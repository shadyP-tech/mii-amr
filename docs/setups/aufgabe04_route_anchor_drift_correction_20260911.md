# Fixed route-anchor map/odom continuity correction

Implemented against local base `a78d94f0211ea6ae8e30641d904742a17045162a` for the failure audited in `stand_explore_exact2_camera_all5_20260911T153146Z`.

The execution monitor now measures translation at the original certified route start. Small localization yaw corrections can move the remote odometry origin substantially while barely moving the route. The previous origin-column metric caused three stops in this recording, preventing the camera observer from starting.

## Contract and module boundaries

- `navigation/localization/map_odom_drift_reference.py` defines the immutable reference, its strict evidence format, and its binding to the frozen transform and certified route start. It is independent of ROS and certificate serialization.
- `navigation/localization/odom_route_adapter.py` uses that reference for stationary admission and live continuity. The original origin displacement remains a diagnostic field in new continuity evidence.
- `navigation/station_segment/localization_admission.py` creates one reference and shares it across stationary evidence, uncertainty allocation, the certificate, and the execution context. Route revisions keep the original reference and re-evaluate their clearance with its full lever arms.
- `navigation/localization/map_odom_continuity_evidence.py` reconstructs saved decisions from their transforms, limits and reference. Startup/runtime recovery uses it to reject incomplete or inconsistent new evidence.
- The runtime motion permit preserves the complete new stop decision and checks that the newly admitted certificate and uncertainty budget agree in version, content hash, reference and heading coordinates. This remains separate from the live admission that computes new limits.

New certificates, uncertainty budgets and continuity reports use schema 2. Genuine schema-1 artifacts retain their original odom-origin interpretation and hashes. Schema 2 requires the reference; marked new evidence cannot fall back to the legacy path when damaged. Historical recovery classification alone grants no motion authority.

## Geometry and unchanged limits

Let `a` be the original map-route start transformed into odom using the certificate's frozen transform. The translation metric is:

```text
anchor_drift = || T_live(a) - T_frozen(a) ||
```

For another odom point `p`, its displacement is bounded by:

```text
displacement(p) <= anchor_drift + abs(relative_yaw) * distance(p, a)
```

The existing uncertainty calculation accounts for distance from this same map reference plus robot radius. Explicit larger heading lever-arm settings are also retained in the reported allocation. A replacement route cannot reset the reference to the robot's current position or its new first waypoint to reduce this allowance.

The translation and yaw limits remain bounded by the original covariance allocation and configured hard caps. Equality is accepted; exceeding either limit still requires stopping and re-admission. Freshness, zero-command handoff, collision clearance, tracking and recovery-attempt limits are unchanged.

## Validation

**507 focused offline tests passed across 37 modules**, with no failures, errors or skips. These cover execution certificates, stationary admission, follower handoffs, uncertainty and replacement routes, startup/runtime recovery, motion permits, one-use consumption and autonomous child orchestration. The changed Python files also pass Python 3.10 grammar parsing and `git diff --check`.

The new tracked fixture `tests/aufgabe04/fixtures/map_odom_route_anchor_20260911T153146Z.json` contains the three recorded stops and original source hashes. Regression tests reproduce each legacy stop exactly, then exercise the new metric with unchanged limits:

| Recorded stop | Origin displacement | Fixed-anchor displacement | Original translation limit |
| --- | ---: | ---: | ---: |
| Coverage 000 | 0.108143 m | 0.015734 m | 0.100824 m |
| Candidate 000 | 0.186141 m | 0.018003 m | 0.150000 m |
| Candidate recovery 001 | 0.099456 m | 0.016554 m | 0.078788 m |

All three pass the new continuity comparison. Their independent yaw limits also pass. The saved-run replay verified all **306 original files** against their manifest and confirmed that the tracked fixture matches the original events, routes, certificates and budgets. The maximum route-plus-radius displacements remain within the original translation and heading reserves; all three actual stationary windows also pass with the same final transforms and thresholds.

Additional tests cover common translation/rotation changes of odom coordinates, translation immediately above the limit, yaw about the anchor, route and footprint displacement bounds, fixed-reference replacement routes, invalid references, altered measurements and schema downgrades. Integration tests exercise the actual admission producer and its persisted certificate/budget/context agreement.

The counterfactual replay is not a replacement certificate or authorization to resume the saved run. Recorded transforms are localization estimates, not physical ground truth. No ROS, robot motion or hardware validation was performed, and this change does not establish camera/QR success. The separate candidate-planning readiness comparison predates this correction and retains its existing metric.

Evidence in `results/audits/stand_explore_exact2_camera_all5_20260911T153146Z/`:

- `route_anchor_correction_tests.json` and `.log`: exact module list and consolidated results.
- `route_anchor_correction_replay.py` and `.json`: original-source integrity and saved-transform replay.
- `drift_metric_replay.py` and `.json`: preserved original audit and legacy replay.

An unrelated observer-module inventory assertion in `test_real_robot_orchestration_boundaries` was already stale on the base revision; it is outside the 37-module passing suite above. Its inventory omits observer modules already present before this correction.

No new experiment flags are required. A checkout containing this correction will emit schema-2 anchor evidence on fresh admission. The next hardware run must obtain new preflight and execution certificates through the normal entry point.
