# Station-tour audit: 2026-10-02 15:53 run

Run `station_tour_20261002T135327_234789Z`, **15:53:27–15:54:07
Europe/Berlin**, workstation SSH alias `mii002`, recorded revision
`f7a6f15ad78470594cabb6f5bc265e148b757e55`.

## Recorded outcome

The robot did not move because the planner rejected the first route's
uncertainty budget before issuing a motion permit or launching a controller.
The error was:

> return uncertainty budget exhausted: no meaningful admitted prefix

Here, “return” is the shared stored-pose planner's error wording. The requested
destination was **QR_001**, not Start. The server was functioning and had
already accepted the initial Start arrival.

Offline replay identifies a route-search limitation: the configured 0.25 m
geometric inflation produces a route that fails uncertainty admission, while
a 0.325 m alternative admits a 0.412 m first stage with unchanged uncertainty
and obstacle checks. The implementation used by this run never attempts that wider
route after rejecting its first route and prefixes.

| Time, Berlin | Event |
|---|---|
| 15:53:31.175 | Random-plan POST sent to the server |
| 15:53:31.197 | Initial Start verification requested |
| 15:53:36.152 | Verified Start arrival reported to the server |
| 15:53:36.161 | Start's zero-duration action wait completed |
| 15:53:36.171 | Navigation to server-selected QR_001 requested |
| 15:54:07.443 | Route planning failed; tour stopped |

The server returned mission `M-00002`, state `GO_TO_DEPOT_PICKUP`, and next
target `QR_001 / DEPOT_01`. Its frozen sequence was
`Start → QR_001 → QR_003 → QR_004 → QR_003 → QR_001 → Start`.
QR_002 was reserved for supplemental coverage after that mission.

Start was accepted without driving because its measured position error was
**0.078988844 m**, below the configured **0.08 m** tolerance. Heading error
was **0.024104781 rad**, below **0.15 rad**. Thus “one completed visit” in the
server summary means a stationary Start verification, not a driven leg.

The first destination's failure artifact records `leg_count=0`, `legs=[]`,
`motion_authorized=false`, `motion_published=false`, and `replan_count=0`.
There is no sealed route, child motion permit, consumption receipt, child
execution log or controller trace for this visit. The earlier controller
startup TF fault was not reached in this run.

## Planning inputs

Stopped localization passed, with maximum reported position standard
deviation **0.054609654 m**, yaw standard deviation **0.050440647 rad**, and
position spread **0.005672964 m** across five samples. The final direct
`map ← odom` transform was admitted. Velocity ownership passed with no
publishers and no active Nav2 goal.

The planner loaded three stationary LiDAR scans into its temporary obstacle
map, projected the saved QR_001 observation pose and candidate geometry into
the admitted frame, and entered the shared route-uncertainty selector. This
selector tests the full planned route and bounded shorter prefixes. An
intermediate prefix must move at least 0.20 m and leave at least 0.15 m of
route, while satisfying the route and stopped-endpoint uncertainty envelopes.
Failure of this search alone does not prove that every possible route is
unsafe: the prefix search uses one previously selected geometric route.

## Offline reproduction

Replay of the production planner in the workstation's read-only ROS Humble
container reproduces the exact exception using the recorded source session,
planning frame, covariance and temporary obstacle overlay. No sensor samples
were replaced with new live measurements and no controller was launched.

| Check with recorded live overlay | Raw robot-center clearance | Required clearance | Remaining margin |
|---|---:|---:|---:|
| Initial orientation envelope | 0.368740 m | 0.359812 m | **+0.008928 m** |
| Limiting sample of full route | 0.353338 m | 0.416591 m | **−0.063252 m** |

The initial pose is admissible under the recorded gate, although its reserve
is small. The selected route fails farther along it. In a diagnostic replay
with only the temporary overlay omitted, the full **2.85179 m** route passes,
with minimum remaining margin **+0.055932 m**. Omitting observed obstacles is
an isolation experiment, not a proposed execution policy.

Both cases select the same five-point geometry. With the live overlay,
**all 16 evaluations** (the full route plus 15 shorter prefixes) fail. The
first negative sample is only **0.125–0.130 m** along the initial segment,
before the **0.20 m** minimum displacement for a useful intermediate stage.
Its raw clearance is 0.37288608 m and required clearance 0.37289093 m.
The worst deficit occurs around **0.609–0.614 m** along the route.

At that worst sample, required clearance consists of:

| Budget component | Distance |
|---|---:|
| Robot radius | 0.105000 m |
| Collision reserve | 0.010000 m |
| Tracking bound | 0.030000 m |
| Transform-drift reserve | 0.020000 m |
| Braking/latency reserve | 0.075000 m |
| Position uncertainty, two sigma | 0.109219 m |
| Heading uncertainty projected along the route | 0.067371 m |
| **Total** | **0.416591 m** |

The first failing sample's nearest obstacle is live-layer cell `(33, 23)`,
center `(-1.145, -0.515)` in the planning map; the limiting sample's nearest
cell is `(34, 25)`, center `(-1.095, -0.415)`. These are occupancy-cell
attributions, not semantic identification of the physical object.

There is no direct double inflation: the overlay is merged into the raw map,
the geometric planner inflates one copy, and uncertainty uses the raw map.
However, geometric search and smoothing use a fixed inflation radius before
the uncertainty selector tests the resulting route. The selector does not
search for a wider alternative after this failure. Live LiDAR endpoints can
represent existing walls and stands as well as new objects; the evidence
must not be described as proof that a person or a new obstacle blocked the
robot.

## Confirmed route-search limitation

Further production-planner replays changed only the geometric inflation
radius, retaining the same raw obstacle overlay, localization covariance,
robot size, braking and other reserves, original candidate keepouts and exact
saved destination. The final uncertainty gate was unchanged.

| Geometric inflation | Offline selection result |
|---|---|
| 0.250 m, recorded setting | Full route and all 15 tested prefixes rejected |
| 0.275 / 0.300 m | Same failing geometry |
| **0.325 m** | **0.411567 m stage admitted**, 0.405647 m displacement, minimum margin **+0.005971 m** |
| 0.350 m | 0.283195 m stage admitted, minimum margin +0.005953 m |
| 0.380 / 0.400 m | Exact-start connector cannot satisfy enlarged geometric clearance |
| 0.450 / 0.500 m | No permitted snapped start found |

Thus the recorded stop was not inevitable under the configured uncertainty
requirements. An alternative route permits meaningful progress from the same
recorded starting state. This proves admission of a first stage, not completion
of the physical tour or admission of later stages; subsequent stages still
require fresh localization and all execution checks.

The bounded prefix sampling is also sensitive near the threshold: a direct
0.220 m cut of the original polyline passes with only **0.00003775 m** reserve,
while the production 0.219369 m cut fails by **0.00002039 m**. Adjusting the cut
distance to exploit that microscopic difference is not the recommended remedy.

The corrective direction is bounded alternative-route search using the full
uncertainty requirements: when the initial geometric route has no admitted
stage, try a wider route to the same stored destination and retain the final
uncertainty check. Do not globally replace the inflation setting with an
arbitrarily large value; the larger replay values demonstrate that doing so
can make the current start unplannable. Also persist rejected route geometry
and budget details, which the current exception discards before writing route
artifacts. No production correction was made during this investigation.

Relevant source locations at the audited revision:

- `scripts/aufgabe04/navigation/approach/admitted_pose_route.py:381`: raw overlay merge.
- The same file, lines 429 and 450: geometric route search and smoothing.
- The same file, line 467: subsequent uncertainty-stage selection.
- `scripts/aufgabe04/navigation/approach/admitted_return_uncertainty.py:207`: bounded prefix checks and final rejection.

## Source-pose provenance

The run used saved poses from **October 1 at 12:46**, session
`stand_explore_exact2_camera_all5_20261001T104618Z`, with
`--confirm-odom-continuity` asserted. It did not use today's partial camera
run. The assertion is operator-supplied; these artifacts alone cannot establish
whether the odometry origin and stand layout remained unchanged overnight.

## Evidence

All 14 files from the tour directory were copied without altering the run.
Their original contents and SHA-256 digests are in
`results/implementation_checks/station_tour_failure_20261002T135327Z/workstation_evidence.json`.
The original files remain under the workstation repository's
`results/aufgabe04/real/station_tours/station_tour_20261002T135327_234789Z`.
This audit issued no robot motion or server mutations.

Reproducible diagnostic scripts and results are saved in the same local
evidence directory as `replay_uncertainty.py`, `replay_uncertainty.json`,
`replay_uncertainty_additional.py` and `replay_uncertainty_additional.json`.
They authenticated the source session, reconstructed the same projected
candidate snapshot, and intercepted the planner before route sealing. No
motion permit or executable certificate was created by these replays.

## Implemented correction and verification

The follow-up correction adds a separate `stored_pose_route_alternatives.py`
module. Travelling dynamic tours now try the initial inflation and four
half-cell increments, selecting the first route with an admitted stopped
prefix. Source and frame authentication run before this search. Identical
rejected geometry reuses its uncertainty result only within the same frozen
context; each candidate still receives its own geometry and collision checks.
Successful search evidence is hash-bound into route metadata and the full
route artifact. The child validator requires that evidence, recomputes the
selected route's clearance and preserves the certified exact-start connector.
Exhaustion writes rejected geometry and budget diagnostics beside the absent
route directory, without producing a certificate or motion permit.

The robot profile already uses **0.105 m radius**. The configured planning
floor remains **0.25 m**, derived from the existing 0.20 m LiDAR stop distance
plus 0.02 m scan allowance and 0.03 m tracking allowance. Each new leg starts
at this floor; there is no permanent increase to 0.325 m. The correction
minimizes the additional buffer among five tested choices while retaining the
body radius, candidate keepouts, covariance and braking reserves. It does not
claim that 0.25 m is a newly calibrated physical minimum or a global optimum.

An unmocked replay in the workstation's ROS Humble container used all original
source artifacts and the recorded overlay. The corrected production planner
created a route certificate only in a temporary directory and passed its
route-binding validator. Attempts at 0.250, 0.275 and 0.300 m were rejected;
**0.325 m passed**, admitting the same **0.411567 m** stage with
**+0.005971 m** minimum remaining margin. Planning and binding took 32.7 seconds
during the offline check. No controller was launched, no motion permit was
created, and no server request was sent. This verifies the failed planning
step, not a completed physical tour.

The replay script and complete admission evidence are retained in
`results/implementation_checks/station_tour_correction_20261002/production_replay.py`
and `production_replay.json`. The tracked regression fixture retains the
recorded temporary cells, full candidate geometry, covariance, exact endpoints
and both failing/passing reference routes.

Validation passed **119 targeted tests in ROS Humble / Python 3.10**: the
118-test planner, authorization, obstacle-replanning and server-tour bundle
(315.2 seconds), plus the final mandatory-artifact-stripping regression
(14.9 seconds). The bundle includes the recorded failing and passing geometry
cases. Local runtime tests and `git diff --check` also passed. Logs are retained
beside the replay evidence.
