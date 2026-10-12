#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "Usage: ./scripts/aufgabe02/run_real_experiment.sh [run_id]"
  exit 1
fi

RUN_ID="${1:-}"
RUN_MODE="${REAL_RUN_MODE:-${RUN_MODE:-rotate-in-place}}"
RUN_ANGLE_DEG="${RUN_ANGLE_DEG:--90}"
RUN_ANGULAR_SPEED="${RUN_ANGULAR_SPEED:-0.3}"
RUN_SPEED="${RUN_SPEED:-0.1}"
RUN_DURATION_SEC="${RUN_DURATION_SEC:-3.0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${MII_AMR_PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_DIR"

mkdir -p bags/real results/aufgabe02

if [ -z "$RUN_ID" ]; then
  if [ "$RUN_MODE" = "rotate-in-place" ] || [ "$RUN_MODE" = "rotation" ] || [ "$RUN_MODE" = "rotate" ]; then
    RUN_PREFIX="$(python3 -c 'import sys; angle = float(sys.argv[1]); direction = "cw" if angle < 0 else "ccw"; amount = ("%g" % abs(angle)).replace(".", "p"); print(f"run_real_rot_{direction}{amount}_")' "$RUN_ANGLE_DEG")"
    RUN_ID="$(python3 scripts/common/next_real_run_id.py --results-csv results/aufgabe02/real_rotation_runs.csv --prefix "$RUN_PREFIX")"
  else
    RUN_ID="$(python3 scripts/common/next_real_run_id.py)"
  fi
fi

BAG_DIR="bags/real/$RUN_ID"
BAG_LOG="results/aufgabe02/${RUN_ID}_bag.log"
if [ -e "$BAG_DIR" ] || [ -e "$BAG_LOG" ]; then
  echo "ERROR: bag output or log already exists for $RUN_ID. Choose a new run ID."
  exit 1
fi

echo "Using run ID: $RUN_ID"
echo "Real run mode: $RUN_MODE"
if [ "$RUN_MODE" = "rotate-in-place" ] || [ "$RUN_MODE" = "rotation" ] || [ "$RUN_MODE" = "rotate" ]; then
  echo "Rotation command: angle=${RUN_ANGLE_DEG} deg, angular_speed=${RUN_ANGULAR_SPEED} rad/s"
else
  echo "Linear command: speed=${RUN_SPEED} m/s, duration=${RUN_DURATION_SEC} s"
fi

source_setup() {
  set +u
  source "$1"
  set -u
}

source_setup /opt/ros/humble/setup.bash
source_setup /opt/tb3_src_ws/install/setup.bash

export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-01
export RUN_MODE RUN_ANGLE_DEG RUN_ANGULAR_SPEED RUN_SPEED RUN_DURATION_SEC

echo "Checking robot topics..."
ros2 topic list | grep -E '^/cmd_vel$' >/dev/null
ros2 topic list | grep -E '^/odom$' >/dev/null

echo "Checking camera start pose for $RUN_ID..."
echo "Start vision_tracker/main.py in another terminal before continuing."
python3 vision_tracker/start_pose_gate.py "$RUN_ID"

BAG_PID=""
cleanup() {
  timeout --kill-after=1s 3s ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0}, angular: {z: 0.0}}" >/dev/null 2>&1 || true
  if [ -n "$BAG_PID" ]; then
    if kill -0 "$BAG_PID" >/dev/null 2>&1; then
      echo "Stopping bag recording..."
      kill -INT "$BAG_PID" || true
    fi
    wait "$BAG_PID" || true
    BAG_PID=""
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting bag recording for $RUN_ID..."
ros2 bag record -o "$BAG_DIR" /cmd_vel /odom /imu /battery_state > "$BAG_LOG" 2>&1 &
BAG_PID=$!

sleep 2

if ! kill -0 "$BAG_PID" >/dev/null 2>&1; then
  echo "ERROR: bag recording failed. See $BAG_LOG."
  exit 1
fi

echo "Running real scripted drive..."
python3 scripts/aufgabe02/real_scripted_drive.py "$RUN_ID"

sleep 1

cleanup
trap - EXIT INT TERM

echo "Experiment $RUN_ID finished."
