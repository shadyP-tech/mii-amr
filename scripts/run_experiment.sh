#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ] || [[ ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: ./scripts/run_experiment.sh <number_of_runs>"
  echo "Example: ./scripts/run_experiment.sh 15"
  exit 1
fi

RUN_COUNT="$1"
RUN_MODE="${RUN_MODE:-linear-forward}"
RUN_SPEED="${RUN_SPEED:-0.1}"
RUN_DISTANCE="${RUN_DISTANCE:-30cm}"
RUN_NUMBER_WIDTH="${#RUN_COUNT}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [ "$RUN_NUMBER_WIDTH" -lt 2 ]; then
  RUN_NUMBER_WIDTH=2
fi

source_setup_file() {
  local setup_file="$1"

  set +u
  source "$setup_file"
  set -u
}

source_ros_setup() {
  local ros_setup="${ROS_SETUP:-}"
  local ros_distro="${ROS_DISTRO:-humble}"

  if [ "${SKIP_ROS_SETUP:-0}" = "1" ]; then
    echo "Skipping ROS setup because SKIP_ROS_SETUP=1."
    return
  fi

  if [ -n "$ros_setup" ]; then
    if [ ! -f "$ros_setup" ]; then
      echo "ROS_SETUP points to a missing file: $ros_setup"
      exit 1
    fi

    source_setup_file "$ros_setup"
    return
  fi

  ros_setup="/opt/ros/${ros_distro}/setup.bash"
  if [ -f "$ros_setup" ]; then
    source_setup_file "$ros_setup"
    return
  fi

  if command -v ros2 >/dev/null 2>&1; then
    echo "ROS 2 is already available; skipping setup source."
    return
  fi

  echo "Could not find ROS setup file: $ros_setup"
  echo "Set ROS_SETUP=/path/to/setup.bash, set ROS_DISTRO=<distro>, or source ROS before running."
  exit 1
}

cd "$PROJECT_ROOT"
source_ros_setup

mkdir -p bags results

run_experiment() {
  local run_id="$1"
  local bag_pid
  local drive_status=0
  local optional_topic
  local -a bag_topics

  if [ -e "bags/$run_id" ]; then
    echo "Bag output already exists: bags/$run_id"
    echo "Choose different run metadata or remove the existing bag before rerunning."
    exit 1
  fi

  echo "Resetting simulation..."
  ros2 service call /reset_simulation std_srvs/srv/Empty || true

  sleep 2

  echo "Starting bag recording for $run_id..."
  bag_topics=(/cmd_vel /odom)
  for optional_topic in /imu /battery_state; do
    if ros2 topic list | grep -qx "$optional_topic"; then
      bag_topics+=("$optional_topic")
    fi
  done
  echo "Recording topics: ${bag_topics[*]}"
  ros2 bag record -o "bags/$run_id" "${bag_topics[@]}" > "results/${run_id}_bag.log" 2>&1 &
  bag_pid=$!

  sleep 2

  if ! kill -0 "$bag_pid" 2>/dev/null; then
    echo "Bag recording failed for $run_id. See results/${run_id}_bag.log."
    wait "$bag_pid" || true
    return 1
  fi

  echo "Running scripted drive..."
  python3 scripts/scripted_drive.py "$run_id" || drive_status=$?

  sleep 1

  echo "Stopping bag recording..."
  kill -INT "$bag_pid" || true
  wait "$bag_pid" || true

  if [ "$drive_status" -ne 0 ]; then
    echo "Scripted drive failed for $run_id."
    return "$drive_status"
  fi

  echo "Experiment $run_id finished."
}

echo "Running $RUN_COUNT simulation run(s)."
echo "Run ID pattern: simulation_run_${RUN_MODE}_${RUN_SPEED}_${RUN_DISTANCE}_XX"

for ((i = 1; i <= RUN_COUNT; i++)); do
  run_index="$(printf "%0${RUN_NUMBER_WIDTH}d" "$i")"
  run_id="simulation_run_${RUN_MODE}_${RUN_SPEED}_${RUN_DISTANCE}_${run_index}"

  echo "=== Simulation run $i/$RUN_COUNT: $run_id ==="
  run_experiment "$run_id"
done

echo "All $RUN_COUNT simulation run(s) finished."
