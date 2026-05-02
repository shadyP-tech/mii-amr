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
SIM_MODEL_NAME="${SIM_MODEL_NAME:-${TURTLEBOT3_MODEL:-burger}}"
SIM_SET_ENTITY_SERVICE="${SIM_SET_ENTITY_SERVICE:-}"
SIM_START_X="${SIM_START_X:-0.5}"
SIM_START_Y="${SIM_START_Y:-0.05}"
SIM_START_Z="${SIM_START_Z:-0.01}"
SIM_START_YAW_DEG="${SIM_START_YAW_DEG:-180.0}"
SIM_RESET_SETTLE_SEC="${SIM_RESET_SETTLE_SEC:-1}"
RUN_NUMBER_WIDTH="${#RUN_COUNT}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

export RUN_MODE RUN_SPEED RUN_DISTANCE
export SIM_START_X SIM_START_Y SIM_START_YAW_DEG

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

require_ros_graph() {
  local missing=0
  local service_names

  service_names="$(ros2 service list)"

  if ! ros2 topic list | grep -qx "/cmd_vel"; then
    echo "Missing required topic: /cmd_vel"
    missing=1
  fi

  if ! ros2 topic list | grep -qx "/odom"; then
    echo "Missing required topic: /odom"
    missing=1
  fi

  if ! grep -qx "/reset_simulation" <<< "$service_names"; then
    echo "Missing required service: /reset_simulation"
    missing=1
  fi

  if [ -z "$SIM_SET_ENTITY_SERVICE" ]; then
    if grep -qx "/gazebo/set_entity_state" <<< "$service_names"; then
      SIM_SET_ENTITY_SERVICE="/gazebo/set_entity_state"
    elif grep -qx "/set_entity_state" <<< "$service_names"; then
      SIM_SET_ENTITY_SERVICE="/set_entity_state"
    fi
  fi

  if [ -z "$SIM_SET_ENTITY_SERVICE" ] || ! grep -qx "$SIM_SET_ENTITY_SERVICE" <<< "$service_names"; then
    echo "Missing required SetEntityState service."
    echo "Set SIM_SET_ENTITY_SERVICE if your Gazebo launch exposes a different SetEntityState service."
    missing=1
  fi

  if [ "$missing" -ne 0 ]; then
    echo "Start Gazebo/TurtleBot simulation before running experiments."
    return 1
  fi
}

set_simulation_pose() {
  local qz
  local qw
  local request
  local output

  read -r qz qw < <(
    python3 -c 'import math, sys; yaw = math.radians(float(sys.argv[1])); print(f"{math.sin(yaw / 2.0):.12g} {math.cos(yaw / 2.0):.12g}")' "$SIM_START_YAW_DEG"
  )

  request="{state: {name: '${SIM_MODEL_NAME}', pose: {position: {x: ${SIM_START_X}, y: ${SIM_START_Y}, z: ${SIM_START_Z}}, orientation: {x: 0.0, y: 0.0, z: ${qz}, w: ${qw}}}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}, reference_frame: world}}"

  if ! output="$(ros2 service call "$SIM_SET_ENTITY_SERVICE" gazebo_msgs/srv/SetEntityState "$request" 2>&1)"; then
    echo "$output"
    return 1
  fi

  echo "$output"

  if [[ "$output" != *"success=True"* && "$output" != *"success: true"* && "$output" != *"success: True"* ]]; then
    echo "Gazebo did not confirm pose reset for model '$SIM_MODEL_NAME'."
    echo "Check the model name with: ros2 service call $SIM_SET_ENTITY_SERVICE gazebo_msgs/srv/SetEntityState ..."
    return 1
  fi
}

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

  require_ros_graph

  echo "Resetting simulation..."
  ros2 service call /reset_simulation std_srvs/srv/Empty

  echo "Setting simulation model pose..."
  set_simulation_pose

  sleep "$SIM_RESET_SETTLE_SEC"

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
