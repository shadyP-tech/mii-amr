---
name: ros2-terminal-debugging
description: >
  Diagnose pasted terminal output or screenshots from ROS2 Humble, Gazebo,
  Apptainer, colcon, Git, TurtleBot, Nav2, AMCL, SLAM, LiDAR scans, mii-amr
  scripts, camera tracking, bags, or related robotics commands. Use when the
  user says diagnose, asks what an error means, or pastes terminal logs from
  the mii-amr project.
---

# ROS2 Terminal Debugging

## Purpose

Turn terminal output or screenshots into a short, actionable diagnosis for the
`mii-amr` ROS2/Gazebo/TurtleBot project.

If the input is a screenshot, read the visible error lines first. If the
decisive line is unreadable or missing, ask for terminal text only when the
visible context is insufficient. Give only safe checks from the visible context.

## Project Environments

Assume unless contradicted:

- Repo name: `mii-amr`.
- This repo is mainly experiment scripts plus `vision_tracker`; do not assume it
  is a normal ROS package unless `package.xml` is present.
- Important topics: `/cmd_vel`, `/odom`, `/scan`, `/amcl_pose`,
  `/initialpose`, and TF.
- Important generated files:
  - `bags/`
  - `bags/real/`
  - `results/aufgabe02/scripted_drive_runs.csv`
  - `results/aufgabe02/real_scripted_drive_runs.csv`
  - `results/aufgabe02/latest_tracker_pose.csv`
  - `results/aufgabe02/real_start_pose_checks.csv`
  - `results/aufgabe03/aufgabe03_waypoint_follow_runs.csv`
  - `results/aufgabe03/aufgabe03_arena_prior_two_stage_runs.csv`
- Real-run script may expect `/workspace/mii-amr`.

Assume the user may work across these environments:

1. Lab workstation / Apptainer container
   - Main environment for ROS2 Humble, Gazebo, TurtleBot real runs, `/cmd_vel`,
     `/odom`, bags, and `run_real_experiment.sh`.
   - Likely repo path: `/workspace/mii-amr` or
     `~/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr`.
   - Prefer this environment for commands requiring `ros2`, `rclpy`, Gazebo,
     TurtleBot networking, or robot motion.

2. MacBook Air M1 / macOS
   - Main environment for editing code, Git, camera access if the camera is
     plugged into the MacBook, OpenCV tracker work, CSV inspection, plotting,
     and report preparation.
   - Uses conda environment: `mii-amr`.
   - Do not assume ROS2 Humble, Gazebo, or `rclpy` are available natively on
     macOS.

3. UTM Ubuntu on MacBook
   - Useful for Linux practice and some development.
   - Treat ROS2/Gazebo support as environment-dependent because Apple Silicon,
     virtualization, graphics, and package availability can cause issues.
   - Do not assume real robot or camera access works unless the user confirms it.

## Response Format

Always answer with exactly these five sections:

1. What the error means
2. Most likely cause
3. Exact command(s) to run next
4. How to verify it worked
5. What not to do

Keep each section concise. Prefer commands and concrete checks over broad ROS
explanations.

## Diagnostic Rules

- Start from the first real error line, not cleanup warnings.
- Prefer one likely cause. Mention alternatives only if the output is genuinely
  ambiguous.
- Before diagnosing deeply, infer the active environment from the prompt,
  screenshot, path, shell prompt, or command.
- If the active environment is unclear, give commands that identify it before
  giving environment-specific fixes.
- Label commands with `[lab/container]`, `[MacBook conda]`, or `[UTM Ubuntu]`
  when environment matters.
- Check environment, path, sourcing, ROS graph, and file existence before
  suggesting code changes.
- For ROS issues, check `/cmd_vel` and `/odom` before debugging project scripts.
- For Aufgabe 03 navigation issues, check `/scan`, `/amcl_pose`,
  `/initialpose`, `/navigate_to_pose`, TF freshness, and the saved map path
  before changing project code.
- For Apptainer/path issues, verify `pwd` and repo location before changing
  code.
- For colcon errors, first check whether the directory is a ROS workspace and
  whether `package.xml` exists.
- For camera/tracker issues, check camera access, calibration, HSV detection,
  and freshness of `results/aufgabe02/latest_tracker_pose.csv`.
- For start-pose gate timeouts, consider stale pose CSV, tracker not running,
  invalid marker detection, or robot outside tolerance.
- For rerun failures, assume generated files may block reruns because scripts
  avoid overwriting; check existing bag/result output.
- For waypoint failures, classify the first cause as setup (`/amcl_pose`
  missing), freshness (stale TF/scan/AMCL), controller timeout, blocked scan, or
  replan artifact/schema issue.
- For Aufgabe 03 arena-prior failures, distinguish mirrored localization from
  poor scan geometry: with `--heater-wall-side=+x` and forced
  `axis_positive`, `--arena-force-short-wall-type heater` matches the latest
  successful setup; `clean` can mirror the map branch.
- For arena-active RViz issues, remember the temporary map topics are
  `odom`-frame. If they do not render with RViz fixed to `map`, try fixed frame
  `odom` until AMCL provides `map -> odom`.
- For late-obstacle runs, separate startup run-local replanning from
  obstacle-triggered replanning. `--run-local-map-initial-scan-mode full`
  proactively replans before driving; `none` waits for a live scan blockage.
- Never tell Codex or the user to execute physical TurtleBot motion
  automatically.
- Before suggesting `run_real_experiment.sh` or any `/cmd_vel` publishing
  command, state that the robot area must be clear and manual stop must be
  possible.
- Avoid destructive Git/filesystem commands such as `git reset`, `git clean`,
  `rm`, checkout overwrite, or force push unless the user explicitly asks for
  cleanup.
- Use Git commands first for inspection only.

## Conda Rule

If the user is on the MacBook and the error involves `cv2`, `numpy`, `pandas`,
`matplotlib`, camera access, CSV analysis, or `vision_tracker` scripts, check
whether conda env `mii-amr` is active before suggesting package installation or
code changes.

If the error involves `ros2`, `rclpy`, Gazebo, `/cmd_vel`, `/odom`, TurtleBot
networking, or real robot motion, do not assume the MacBook conda environment is
sufficient. Prefer the lab/container or Ubuntu ROS2 environment.

## Useful Command Patterns

Use the smallest relevant subset.

Environment check:

```bash
pwd
uname -a
which python3
python3 --version
which ros2 || true
echo "$CONDA_DEFAULT_ENV"
```

Basic ROS location and sourcing:

```bash
cd /workspace/mii-amr 2>/dev/null || cd ~/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr 2>/dev/null || cd ~/mii-amr
pwd
source /opt/ros/humble/setup.bash
```

Lab/container TurtleBot environment:

```bash
source /opt/ros/humble/setup.bash
source /opt/tb3_src_ws/install/setup.bash
export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-01
```

ROS graph checks:

```bash
ros2 topic list | grep -E '^/(cmd_vel|odom|scan|amcl_pose|initialpose)$'
ros2 topic echo /odom --once
ros2 topic echo /scan --once
ros2 topic echo /amcl_pose --once
ros2 node list
ros2 action list | grep navigate_to_pose
ros2 topic info /initialpose --verbose
ros2 topic info /cmd_vel --verbose
```

MacBook conda project Python:

```bash
conda activate mii-amr
python3 --version
which python3
python3 -c "import cv2; print(cv2.__version__)"
```

MacBook conda camera/tracker checks:

```bash
conda activate mii-amr
cd ~/Documents/stephsWorld/mii-amr 2>/dev/null || cd ~/mii-amr
python3 vision_tracker/list_cameras.py
python3 vision_tracker/calibration.py --verify
python3 vision_tracker/tune_hsv.py
tail -n 2 results/aufgabe02/latest_tracker_pose.csv
```

Start-pose gate checks:

```bash
python3 vision_tracker/main.py
python3 vision_tracker/start_pose_gate.py test_run --timeout 10
```

Aufgabe 03 checks:

```bash
source /opt/ros/humble/setup.bash
source /opt/tb3_src_ws/install/setup.bash
export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-02
ros2 topic echo --once /scan
ros2 topic echo --once /amcl_pose
python3 scripts/aufgabe03/two_stage_waypoint_run.py \
  --waypoints results/aufgabe03/aufgabe03_waypoints.csv \
  --run-id debug_dry_run \
  --dry-run
```

Git inspection:

```bash
git status --short
git branch --show-current
git diff -- <path>
```

Colcon/workspace inspection:

```bash
pwd
find . -maxdepth 2 \( -name package.xml -o -name CMakeLists.txt \)
colcon list
```

## Common Diagnoses

- `ros2: command not found`: ROS is not sourced or the shell is outside the
  Ubuntu/Apptainer ROS environment.
- `ModuleNotFoundError: No module named 'rclpy'`: Python is not running inside
  the ROS environment.
- `/cmd_vel` or `/odom` missing: simulation/robot is not launched, not
  connected, or ROS networking/domain settings are wrong.
- `Timed out waiting for /odom`: topic may exist but no odometry messages are
  arriving.
- `/scan` missing: TurtleBot/LiDAR bringup is incomplete, `LDS_MODEL` may be
  wrong, or ROS networking/domain settings are wrong.
- `/amcl_pose` timeout: Nav2/AMCL is not launched with the saved map, AMCL is
  not receiving scans, or no pose prior has been accepted.
- `AMCL localization warning(s): missing_amcl` during a completed follower run:
  TF/odom may have been sufficient for control, but AMCL pose feedback was not
  healthy; report it as a caveat, not as a failed run by itself.
- no `/initialpose` subscriber: Nav2 localization is not ready for the arena
  prior.
- mirrored arena map after `/initialpose`: forced short-wall side/type likely
  contradicts the physical heater/clean side. For the current setup use
  `--heater-wall-side=+x`, `--arena-force-short-wall-side axis_positive`, and
  `--arena-force-short-wall-type heater`.
- stale TF during waypoint following: map-to-base transform is not updating
  quickly enough; check AMCL, TF, CPU load, and freshness thresholds before
  changing controller logic.
- temporary arena-active map not visible in RViz: the topic is probably being
  published in `odom` while RViz fixed frame is `map` and AMCL has not provided
  a transform yet; switch RViz fixed frame to `odom` for that debug view.
- `Timed out trying to reach waypoint`: check speed/gain/tolerance settings,
  path handoff distance, and scan hard-stop behavior.
- automatic replan before the obstacle is placed: `--enable-lidar-map-replan`
  plus an initial scan mode other than `none` triggers startup run-local A*
  replanning. Use `--run-local-map-initial-scan-mode none` for delayed obstacle
  validation.
- `persistent_scan_blockage_after_existing_map_repair`: fresh replans or
  existing-map repairs were attempted, but the live scan corridor stayed
  blocked after the allowed replan budget.
- missing run-local map artifacts: active replan may not have been enabled, the
  row may be from the old schema, or artifact-only/active replan failed before
  writing outputs.
- robot behavior changes when arena-active RViz publishers are enabled: suspect
  duplicate helper publishers or DDS/CPU timing load before changing controller
  logic; use the no-viz flags for real motion if that was the stable setup.
- Existing bag output: script refuses to overwrite a previous run directory.
- No homography/calibration: camera calibration must be created or verified
  before pose tracking.
- Start-pose timeout: tracker not running, pose CSV stale, markers not detected,
  or robot outside tolerance.
- `/workspace/mii-amr` missing: command/script is being run outside the expected
  container bind path.
