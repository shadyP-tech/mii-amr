# Opposite-route audit: 20260928T150641Z

The opposite-side branch triggered twice for the second visited stand,
`001_survey_candidate_0001`. Its drives were withheld because every proposed
route failed either static feasibility or the subsequent route uncertainty
admission. Backside classification was successful. The observer correction from
the preceding task was present, but the robot never reached an opposite-side
camera observation in this run.

## Sources and scope

- Workstation: SSH alias `mii002`.
- Run: `stand_explore_exact2_camera_all5_20260928T150641Z`.
- Workstation and local revision: `696b1efab97a24d747725ed426b397117d847ac3`.
- Local evidence root: `results/implementation_checks/run_audit_20260928T150641Z/`.
- Primary records: `mission/candidate_selection.jsonl`,
  `mission/candidates/001_survey_candidate_0001/inspection_history/revision_006.json`,
  the two `camera_lidar_attempt_*/axis_observation.json` receipts, and
  `mission/odom_execution/*opposite*uncertainty_budget.json`.
- Terminal log: `bundle/terminal_run.log`.
- Offline reproduction: `replay_route_budget.py` and `route_budget_replay.json`.

No production code, workstation checkout, ROS nodes, or robot motion were changed
for this investigation.

## Branch and recovery sequence

The first and second local camera observations each committed a certified
backside angle with seven samples. Their angular uncertainty half-widths were
approximately 6.92 and 3.27 degrees. The planning artifacts also contain validated
metric target centers with 2.53 and 2.89 cm uncertainty respectively.

Each certification invoked the opposite route. Each exhausted its initial
standoff search and then used the existing single stopped localization-refresh
opportunity before exhausting a second search. Across these four planning
epochs there were:

- Seven materialized routes, each passing the standard 16-observation preflight
  and then failing route uncertainty admission.
- Five additional standoff proposals rejected by static goal feasibility.
- Zero motion permits issued or motion published for an opposite route.

The first search proposed 0.50, 0.45 and 0.400661 m standoffs; the second proposed
0.50, 0.45 and 0.404277 m. The shortest proposals were blocked by the rasterized
map/keepout geometry. After the first localization refresh, the 0.50 and 0.45 m
requests snapped to the same actual goal and route, so those nominally different
proposals supplied little geometric diversity.

`candidate/inspection_execution.py` catches the typed
`opposite_route_uncertainty_exhausted` failure and continues the bounded local
view search. Consequently the robot made two short local-view moves (about
0.189 and 0.186 m) rather than driving around the stand.

The third camera attempt ended with a recorded `KeyboardInterrupt` and
`observer_terminal_failure`. The evidence identifies the interruption type, not
who or what generated it. It is separate from the earlier opposite-route vetoes.
No matching exploration process was active when checked.

## Why the drives were rejected

The runtime budget adds robot footprint, tracking, drift, braking, localization
and heading allowances to the static route clearance. Its heading term grows
with distance from the frozen localization anchor:

`heading allowance = 2 × AMCL yaw sigma × (anchor-to-route distance + robot radius)`

This is the robot localization yaw uncertainty, not the camera's backside-angle
uncertainty. `navigation/execution/route_uncertainty_admission.py`, function
`_heading_contribution_for_points`, implements that lever arm. The route extends
about a metre from its anchor toward a boundary-side final section, where the
available clearance is smallest.

| Opposite attempt | Standoff | Minimum remaining margin |
| --- | ---: | ---: |
| First backside, initial frame | 0.50 m | -12.69 cm |
| First backside, initial frame | 0.45 m | -5.21 cm |
| First backside, refreshed frame | 0.50 m | -2.40 cm |
| First backside, refreshed frame | 0.45 m | -2.16 cm |
| Second backside, initial frame | 0.50 m | -9.76 cm |
| Second backside, initial frame | 0.45 m | -1.11 cm |
| Second backside, refreshed frame | 0.45 m | -1.19 cm |

Admission requires a strictly positive margin. For the closest-to-passing route
(`inspection_003_opposite_standoff_001`), the limiting final section had:

| Budget item | Distance |
| --- | ---: |
| Raw centerline clearance lower bound | 37.25 cm |
| Robot radius | 10.50 cm |
| Collision margin | 2.00 cm |
| Tracking bound | 3.00 cm |
| Odometry drift bound | 2.00 cm |
| Braking allowance | 1.50 cm |
| Localization position allowance | 11.08 cm |
| Heading allowance | 8.28 cm |
| Total required clearance | 38.36 cm |

AMCL yaw sigma was 2.22 degrees for that route. All seven stopped
`map->odom` stability checks passed; missing TF or a rejected stationary
transform was not the cause. The covariance-derived clearance allocation still
exceeded the route's available clearance. A standard `Preflight: PASS` therefore
did not imply that the later execution admission had passed.

The first refresh reduced uncertainty substantially. The second refresh reduced
it again, but also shifted the chosen grid goal one 5 cm cell toward the
boundary, reducing clearance. This explains why the second refresh still failed
despite a smaller covariance envelope.

## Tested next correction

Use a bounded, uncertainty-aware checkpoint fallback for an otherwise feasible
opposite route. Select a safe existing route waypoint, certify the prefix,
stop there, admit a fresh localization frame, and reproject/replan the suffix
from the original retained orientation and validated target center. Both legs
must independently pass the unchanged clearance and live motion gates. Failure
at the checkpoint must leave the robot stopped and enter bounded recovery.

The offline replay loads the real map with the recorded arena bounds, verifies
the blocked-geometry hash, and reproduces every recorded rejection to within
1e-9 m. It then tests splits using the recorded covariance for both legs and a
new hypothetical heading anchor for the suffix. It does not assume a smaller
covariance or remove any budget term.

For `inspection_003_opposite_standoff_001`, a checkpoint at the existing waypoint
(-1.545, -0.265) m changes the whole-route -1.11 cm result into:

- Prefix minimum remaining margin: **+4.60 cm**.
- Suffix minimum remaining margin: **+4.77 cm**.

The first-backside refreshed route also has a viable hypothetical split: +8.61 cm
for its prefix and +3.75 cm for its suffix at waypoint 3. Some other proposals
remain inadmissible even when split; checkpoint selection must use the actual
budget, not a fixed index.

These are offline feasibility results, not authorization for motion. The future
checkpoint pose and covariance must be measured and admitted there. Merely
resetting the heading anchor in the current certificate would be invalid.
The fallback should only activate after a verified no-motion uncertainty veto;
ordinary camera legs do not need an extra synchronization step.
