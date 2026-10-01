# Independent server station tour

`scripts/aufgabe04/real_robot/entrypoints/run_server_station_tour.py` runs a new
tour using the completed camera exploration's stored stand poses. It does not
start exploration, capture images or decode QR codes. Both geometry-validated
facing poses and admitted QR-only observation poses are supported, including
their saved heading. Start this standalone script after the camera runner has
finished its automatic return to Start; it does not launch or control that runner.

Use the original exploration directory and hardware profile on the workstation
inside the same ROS environment. Source artifacts retain their absolute paths
and hashes; copying only the two pose catalogs is insufficient. The source
exploration must have completed its distinct-QR goal. A failed subsequent return
to Start does not invalidate successfully stored discovery artifacts.

## Preview

From the repository root, substitute the completed session, profile and stable
server robot/team ID:

```bash
python3 scripts/aufgabe04/real_robot/entrypoints/run_server_station_tour.py \
  --exploration-session /workspace/mii-amr/results/aufgabe04/real/autonomous_exploration/SESSION_ID \
  --robot-profile /absolute/path/to/robot_profile.json \
  --server-robot-id TEAM_ID \
  --stations 3
```

The site descriptor defaults to `docs/setups/{profile.physical_site_id}.json`.
Use `--physical-site /absolute/path/to/site.json` if it is stored elsewhere.

The default validates the stored evidence and writes an `inputs.json` preview
under a fresh `results/aufgabe04/real/station_tours/station_tour_.../` directory.
It makes no server requests and requires no live ROS or camera process. It does
not claim that a physical route has passed live admission. `--dry-run` is an
explicit spelling of this default.

## Execute

Use the same command with:

```text
--execute --confirm-unloaded --confirm-odom-continuity
```

After local artifact validation and the typed `RUN` confirmation, execution
requests and freezes the server plan before checking Start. By default, the
initial Start visit only verifies a fresh stopped pose against the saved Start
pose reprojected into the current localization frame. Arrival must be within
**0.08 m and 0.15 rad**. This check does not plan a route, capture an
obstacle-scan cohort or dispatch motion. A failed check leaves the randomized
server plan in place but sends no Start scan or arrival report and starts no
onward motion.

Add `--drive-to-start` if the camera runner's automatic return failed or the
robot moved afterward. This explicitly allows an initial approach when needed,
using the usual planning and motion gates. Later final and supplemental returns
to Start continue to navigate normally with either initial policy.

The robot must still use the exploration's continuous odometry frame: no base
restart, odometry reset or replacement of its frame origin. Historical discovery
artifacts do not record an odometry reset identifier, so the runner requires this
operator assertion. Fresh localization alone cannot repair a changed historical
origin. Re-explore after an odometry reset.

Execution asks for one typed `RUN` for this new tour. It does not reuse the
exploration's authorization. Each motion child gets
its own sealed, single-use `stored_pose_tour` permit after a successful dry
admission. All existing live scan, odometry, localization, uncertainty, route,
clearance and velocity-ownership checks remain active.

The sequence is:

1. Request a random plan from
   `POST /api/v1/robots/{robot_id}/plan/randomize` on
   `http://10.42.0.1:8000`, using `qr_count` from the saved numbered QRs and
   `stations` from the command line.
2. Fetch `GET /api/v1/robots/{robot_id}/qr-mappings`, then validate and freeze
   the returned station order and robot-specific QR mapping.
3. Verify the camera runner's stopped arrival at the admitted pose for exact QR
   `Start`; with `--drive-to-start`, approach first if needed.
4. Report verified Start through `POST /api/v1/qr/Start/scan` to receive the
   first `next_target.qr_code_id`. Wait at Start for the server's timed actions
   and earliest-next-scan time before navigating to that target.
5. Follow each validated next target, preserving repeated stand visits. Report
   its QR only after a fresh measured arrival at its stored pose, then honor
   the server's waits before departing. Continue until the server confirms
   `FINISHED` with no next target.
6. By default, visit any saved stands omitted by the server plan in shuffled
   order, then return to Start. These supplemental visits are navigation only;
   they do not send scans to the already-finished server mission. Use
   `--server-plan-only` to omit them.

Console progress announces upcoming server requests, the accepted next target,
waits and failures, so the current stage is visible while the tour runs.

The current server expects exact IDs `Start`, `QR_001`, `QR_002`, and so on, with
4–10 numbered QRs and no gaps. `--stations` requests 3–100 production visits;
the server may also include depot visits and repeated stations. Its built-in
Start mapping is physical `Start` → semantic `START`.

This runner is an **unloaded navigation tour**. It records server actions and
honors their waits, but does not pick up or drop off pucks, process material or
operate a charger. An arrival report identifies a previously observed QR at a
verified saved pose; it is not presented as a fresh camera observation.

Travel uses the admitted-pose policy of up to **0.15 m/s and 0.60 rad/s**, slowing
to **0.055 m/s and 0.18 rad/s** at corners and near the final pose. Travel plans
use the complete stored candidate pool and may use up to four fresh,
independently admitted route stages when uncertainty prevents a single leg.
Temporary obstacles are handled by a tour-local LiDAR occupancy grid and stopped
global replanning. Before each motion leg, three fresh stationary scans are
transformed at their source timestamps into the continuous odometry frame. Their
occupied cells are projected into the fresh map frame, combined with the static
map and all saved stand keepouts, and used by A* and the child clearance checks.
The original stand pose and heading remain the destination of every detour.

While driving, the local waypoint controller monitors the next 0.8 m of the
route corridor. A new obstruction causes a zero-velocity hold; a second distinct
scan confirms the stop. The parent captures fresh stopped scans and localization,
plans a new route, and admits a new single-use permit before moving again. A visit
allows two obstacle replans in addition to its four possible uncertainty stages.
The 0.20 m emergency LiDAR stop and all sensor/ownership checks remain active.
Sensor, localization, ownership and stuck-motion faults terminate the tour.

The occupancy grid uses 0.05 m cells. Cells clear after two valid free-ray
observations or expire after 30 seconds; invalid/infinite rays do not clear
obstacles. Updates and expiry take effect only in subsequent stopped plans.
An executing leg retains its frozen obstacle snapshot. Static map cells and
saved stand locations are never cleared by temporary observations. A blocked
destination, no admissible detour, an unsafe starting clearance or exhausted
replan budget stops the tour without reporting arrival or reversing blindly.

This uses the existing local waypoint controller plus obstacle monitoring and
A* replanning; it does not launch Nav2 or introduce a continuous local trajectory
optimizer. Each new obstacle detour includes a deliberate stopped admission.

## Evidence and failures

Each tour gets a new output directory. `inputs.json` records source identities,
poses and invocation choices, including `initial_start_policy`:
`verify_camera_return` by default or `drive_if_needed` with `--drive-to-start`.
`visits/` contains routes, stopped localization,
permits, temporary obstacle snapshots, stationary scan cohorts, authenticated
terminal outcomes and measured arrivals; `server/` records the frozen plan and an append-only
request/response journal. The normal child-run evidence bundles are also kept.

HTTP requests have bounded timeouts. A rejected or ambiguous arrival report,
changed plan/mapping, missing pose, failed navigation or inconsistent terminal
response stops the tour. Writes are never automatically retried, and existing
tour directories are never resumed or overwritten. The journal records each
`client_event_id` before sending; the server documents that this ID must be
reused when manually reconciling the same scan request. The runner never resets
another robot or clears global server state.

`--server-base-url`, `--http-timeout-sec`, `--tour-id` and `--output-root` can be
overridden. Changing the server ID changes which robot-specific plan is created.

## Module responsibilities

- `tour_scan_capture.py` and `tour_scan_contract.py`: passive ROS capture and pure scan validation.
- `temporary_obstacle_overlay.py`, `temporary_scan_capture.py` and
  `temporary_obstacle_projection.py`: occupancy updates, expiry and immutable map projection.
- `tour_obstacle_monitor.py`: local forward-route checks before velocity publication.
- `tour_navigation_leg.py` and `tour_obstacle_navigation.py`: exact-target planning,
  measured arrival and bounded stopped replanning.
- `tour_replan_binding.py` and `tour_terminal_evidence.py`: new tour authorization,
  single-use execution slots and genuine predecessor outcomes.

These modules do not change camera exploration or the server sequencing policy.
Physical obstacle detours require workstation/robot validation; offline tests do
not establish performance on the real course.
