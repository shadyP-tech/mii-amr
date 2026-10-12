Aufgabe 03 essential runtime

Run commands from the repository or extracted upload root. Keep the
scripts/aufgabe03 package structure and maps/aufgabe03 map pair together.
Offline planning uses Python's standard library. Robot execution requires the
existing ROS 2 Humble/TurtleBot workspaces: rclpy, geometry_msgs, nav_msgs,
sensor_msgs, tf2_ros, and nav2_msgs for the two-stage runner. Source those
workspaces before robot commands.

Entry points
python3 scripts/aufgabe03/map_path_planner.py --help
python3 scripts/aufgabe03/arena_coverage_drive.py --help
python3 scripts/aufgabe03/two_stage_waypoint_run.py --help
python3 scripts/aufgabe03/follow_planned_waypoints.py --help

Fresh route inputs
Choose start and goal coordinates in maps/aufgabe03/arena_1p898x3p9_auto.yaml's
frame. Replace the four uppercase placeholders with numbers:

python3 scripts/aufgabe03/map_path_planner.py --start X_START Y_START --goal X_GOAL Y_GOAL

This writes a dense path CSV and simplified waypoint CSV under results/aufgabe03/.
Historical routes are archived because some use a different map. Preview image
and replanning-artifact generation are no longer part of this runtime.

Essential workflows
arena_coverage_drive.py runs autonomous shadow-frontier coverage: seed spin,
fresh scan/odometry planning, bounded forward curves, then an independent final
verification spin. It completes only when verification finds no remaining raw
shadow cells or moving frontier. Blocked or exhausted missions report failure.

The two-stage runner performs arena geometry localization, publishes its prior,
waits for stable fresh AMCL/TF, uses Nav2 to reach the staging pose, and then
starts the stop-go follower as a separate process. Existing explicit
--arena-active-recovery-mode active_explore and
--arena-active-recovery-executor cmd_vel options enable localization recovery.
--enable-lidar-map-replan enables initial and live stopped LiDAR replanning.
--output-dir replaces the former --results-csv option.

The follower keeps one stop-go controller. Every nonzero command requires fresh
TF, AMCL and scan data plus obstacle clearance. Initial clear scans may retain a
collision-checked static route. Live obstruction stops the robot, captures fresh
scans with timestamped transforms, updates the run-local map and replans with A*.
Initial mapping has a separate budget from live replan attempts. Rejected plans,
stale data and exhausted budgets terminate stopped; terminal cleanup repeatedly
publishes zero velocity. Required static inputs and ROS setup still apply.

--dry-run checks configuration/input preparation without starting robot motion.
Read each command's --help before using it; obsolete flags are rejected.
Pure-pursuit follower variants, branch/escape repairs, adaptive speed/smoothing,
RViz output, fixed coverage routes, legacy repositioning, verbose CSV histories,
and standalone debug/analysis tools have been removed from the active source.
Shared curve geometry remains for autonomous exploration.

Archive and validation
Original source, retired modules, test suites and validation logs are under the
ignored sanitized/aufgabe03-essential-2026-10-12/ directory in the original repo.
The clean upload excludes tests, results, caches and that archive. Validation is
offline; no ROS hardware mission was performed during this cleanup.
