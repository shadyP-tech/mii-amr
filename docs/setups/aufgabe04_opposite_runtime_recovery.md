# Opposite-route recovery after a motion stop

This local correction addresses the October 2 run
`stand_explore_exact2_camera_all5_20261002T084113Z`: a validated localization
yaw-drift stop was followed by a replacement route rejected before motion for
a −3.759 mm uncertainty-adjusted clearance margin. The initial opposite-route
fallback handled startup failures only, so the runtime rejection ended the run.

## Narrow eligibility

The runtime coordinator still raises a terminal error for a rejected replacement.
For the opposite-face routine only, it can attach a typed replan context after
validating all of the following:

- The source motion ended in the exact eligible `FORCE_ZERO_RESEAL` contract.
- Its consumed permit and routine identity are preserved.
- Fresh stationary localization and a same-routine replacement were obtained.
- The replacement failed structured preflight without motion or any permit.
- Its route-uncertainty evidence identifies a finite negative clearance margin.

The context binds original and replacement identities, the stopped source,
the attempt number, source permit, and localization evidence. The opposite
caller rechecks the context and evidence hashes before using it. A matching
error message or failure phase alone cannot enable this path. The attempted
runtime reseal is counted even when its replacement never reaches motion.

## One additional planning epoch

The caller exits the old standoff loop because its start pose predates motion.
It reserves `post_motion_001` with an exclusive directory creation and obtains
another fresh stationary planning frame. It projects the **original** camera
orientation receipt and candidates into that frame, then evaluates the existing
bounded standoff choices. Every route still requires normal target admission,
clearance, uncertainty, dry/live checks and newly issued motion authority.

Both additional startup and runtime reseal budgets are zero. Changing the route
ID does not renew them. After verified no-motion rejections exhaust the fresh
standoffs, the existing checkpoint selector may admit one checkpoint if its
normal geometry/uncertainty conditions hold. Its suffix takes another fresh
frame, retains zero reseal budgets, and cannot request another checkpoint.

Further motion failures remain terminal. Exhausted post-motion routes are
reported as a runtime recovery error, so pre-motion-only localization retries
and local-view fallbacks cannot reuse the stale observation pose. The original
stop and rejected replacement remain distinct in diagnostic events:
`source_motion_published=true`, `rejected_replacement_motion_published=false`.

The recovery correction itself changes no localization, clearance, angle, speed
or physical standoff threshold. A subsequent padding adjustment is described below.
The context permits planning only; it does not authorize movement or reuse the
source permit. It also cannot admit a camera observation before arrival.

## Validation and deployment scope

`test_runtime_route_rejection_context.py` covers typed eligibility, changed
evidence and lineage, source permits, and reseal accounting.
`test_opposite_runtime_retry.py` exercises the real recovery dispatcher and
candidate/axis frame projection, including the audited yaw-drift and negative
margin, fresh alternatives, bounded failures and checkpoint continuation.
Existing startup, runtime, opposite-localization and checkpoint tests also
cover unchanged behavior outside this path.

Local validation on October 2, 2026 passed **115 tests and 116 subtests**:
the combined nine-module recovery suite passed 107 tests and 112 subtests;
the existing autonomous candidate approach tests selected with `-k opposite`
passed another 8 tests and 4 subtests (36 unrelated tests deselected).
The interpreter was `/tmp/a04-id-only-venv/bin/python`; ROS was not sourced.

The implementation and tests are local. No source update was transferred to
the workstation and no real robot run was started. Raw run data remains on the
workstation, as requested. Offline regression outcomes do not establish that
the recorded scene has an admissible alternative route; fresh live preflight
must still establish that.

## Subsequent collision-padding adjustment

The user subsequently selected **10 mm** extra collision padding, replacing
the previous 20 mm. `DEFAULT_COLLISION_MARGIN_M` is now shared by route
admission, dynamic approach geometry, coverage blockage replanning, and the
four mission planning/observation CLI defaults. Explicit CLI overrides remain
available where previously supported. New evidence records the chosen margin;
existing certificates and recorded evidence are not rewritten.

Using the October 2 limiting segment's recorded values, reducing this one term
by 10 mm changes the required clearance from about 376.268 to 366.268 mm and
the remainder from approximately −3.759 to +6.241 mm. The regression uses the
rounded aggregate components, yielding +6.240 mm. This checks budget arithmetic,
not a replay or a guarantee that a fresh complete route will pass.

The 105 mm robot radius, 30 mm tracking allowance, 20 mm odometry drift allowance,
15 mm braking allowance, covariance multiplier and heading contribution remain
unchanged. Static wall inflation, stand transit keepouts and LiDAR guard distances
also remain unchanged. In particular, the coverage omnidirectional hard stop
keeps its separate 20 mm reserve (125 mm from the configured robot center),
instead of implicitly shrinking with route padding. The front stop remains
200 mm. Route admission still rejects zero or negative remaining clearance.

This adjustment is local only; no workstation update or robot motion was made.
The focused planning, admission, child-runner and opposite-recovery suite passed
**153 tests and 71 subtests** after the adjustment. It covers shared defaults,
the rounded audited budget, positive-clearance admission, preserved scan stops,
and collision-free stand approach paths with the new padding. `git diff --check`
also passed.
