# Independent server station tour

`scripts/aufgabe04/real_robot/entrypoints/run_server_station_tour.py` runs a new
tour using the completed camera exploration's stored stand poses. It does not
start exploration, capture images or decode QR codes. Both geometry-validated
facing poses and admitted QR-only observation poses are supported, including
their saved heading.

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

The robot must still use the exploration's continuous odometry frame: no base
restart, odometry reset or replacement of its frame origin. Historical discovery
artifacts do not record an odometry reset identifier, so the runner requires this
operator assertion. Fresh localization alone cannot repair a changed historical
origin. Re-explore after an odometry reset.

Execution obtains stopped localization and asks for one typed `RUN` for this new
tour. It does not reuse the exploration's authorization. Each motion child gets
its own sealed, single-use `stored_pose_tour` permit after a successful dry
admission. All existing live scan, odometry, localization, uncertainty, route,
clearance and velocity-ownership checks remain active.

The sequence is:

1. Drive to the admitted pose for exact QR `Start` and verify stopped arrival.
2. Request a random plan from
   `POST /api/v1/robots/{robot_id}/plan/randomize` on
   `http://10.42.0.1:8000`, using `qr_count` from the saved numbered QRs and
   `stations` from the command line.
3. Validate and freeze the returned station order and robot-specific QR mapping.
4. Report Start through `POST /api/v1/qr/Start/scan`. Follow each validated
   `next_target.qr_code_id`, preserving repeated stand visits, and report its QR
   only after a fresh measured arrival at its stored pose.
5. Wait at the stand for the server's timed actions and earliest-next-scan time.
   Continue until the server confirms `FINISHED` with no next target.
6. By default, visit any saved stands omitted by the server plan in shuffled
   order, then return to Start. These supplemental visits are navigation only;
   they do not send scans to the already-finished server mission. Use
   `--server-plan-only` to omit them.

The current server expects exact IDs `Start`, `QR_001`, `QR_002`, and so on, with
4–10 numbered QRs and no gaps. `--stations` requests 3–100 production visits;
the server may also include depot visits and repeated stations. Its built-in
Start mapping is physical `Start` → semantic `START`.

This runner is an **unloaded navigation tour**. It records server actions and
honors their waits, but does not pick up or drop off pucks, process material or
operate a charger. An arrival report identifies a previously observed QR at a
verified saved pose; it is not presented as a fresh camera observation.

Travel uses the admitted-pose policy of up to **0.15 m/s and 0.60 rad/s**, slowing
to **0.055 m/s and 0.18 rad/s** at corners and near the final pose. Each visit
plans around the complete stored candidate pool and may use up to four fresh,
independently admitted route stages when uncertainty prevents a single leg.
Stored geometry accelerates routing; live obstacles still stop the robot.

## Evidence and failures

Each tour gets a new output directory. `inputs.json` records source identities,
poses and invocation choices; `visits/` contains routes, stopped localization,
permits and measured arrivals; `server/` records the frozen plan and an append-only
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
