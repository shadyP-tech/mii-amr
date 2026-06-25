# Aufgabe 04 Real Parkour Checklist

Before any real robot motion:

- Clear the arena and station approach zones.
- Keep an operator beside each robot.
- Keep Ctrl+C and physical stop access ready.
- Verify `/scan`, `/odom`, TF, AMCL, map, namespace, and `/cmd_vel` ownership.
- Verify no active competing controller is publishing velocity commands.
- Start with dry-run and single-robot checks before two-robot operation.

## Single-Segment Bringup Order

1. Start TurtleBot3 bringup.
2. Start saved-map localization or another explicit `map -> base_footprint`
   localization source.
3. Set the initial pose in RViz and verify the LiDAR overlay aligns with the map.
4. Confirm there is no active Nav2 goal before handing off to the custom
   Aufgabe 04 follower.
5. Run the station-segment dry run. Inspect the resolved topics before motion:

```bash
python3 scripts/aufgabe04/navigation/run_single_station_segment.py \
  --dry-run \
  --leg-index 1 \
  --route-csv results/aufgabe04/routes/station_route.csv \
  --diagnostics-json results/aufgabe04/routes/station_route_diagnostics.json
```

6. If the dry run passes, keep a separate stop terminal ready for the resolved
   velocity topic printed by the dry run. For example:

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}'
```

Use `/robot1/cmd_vel` instead when the dry-run output resolves the robot's
velocity topic under a namespace.

7. Run the real segment only after the dry-run gates are clean:

```bash
python3 scripts/aufgabe04/navigation/run_single_station_segment.py \
  --leg-index 1 \
  --route-csv results/aufgabe04/routes/station_route.csv \
  --diagnostics-json results/aufgabe04/routes/station_route_diagnostics.json
```

The command requires typing `RUN` before motion.

## Preflight Gate

The preflight must pass before follower motion starts:

- resolved `/cmd_vel` must match the intended robot, for example `/cmd_vel` vs
  `/robot1/cmd_vel`
- `/scan` and `/odom` must be fresh by header timestamp and local receipt time
- TF must provide fresh `map -> base_footprint` and `odom -> base_footprint`
- AMCL must be fresh when AMCL is the localization source
- real-robot runs require `use_sim_time=false` unless intentionally overridden
- `/cmd_vel` ownership fails closed when another controller might own velocity
- Nav2 publisher count alone is not enough; check active NavigateToPose or
  controller state where possible

Zero-length legs are no-op evidence only. They must be run with `--allow-noop`;
the runner logs `motion_published=false` and does not publish motion.
