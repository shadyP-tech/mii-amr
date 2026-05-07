#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "Usage: ./scripts/run_supervisor_validation.sh [run_id]"
  exit 1
fi

RUN_ID="${1:-}"
PROJECT_DIR="${MII_AMR_PROJECT_DIR:-/workspace/mii-amr}"
PREDICTION_FILE="${SUPERVISOR_PREDICTION_FILE:-results/supervisor_route_prediction.json}"
RESULTS_CSV="${SUPERVISOR_VALIDATION_RESULTS_CSV:-results/supervisor_route_validation_runs.csv}"
TRACKER_POSE_FILE="${TRACKER_POSE_FILE:-results/latest_tracker_pose.csv}"
LINEAR_SPEED="${SUPERVISOR_LINEAR_SPEED:-0.10}"
ANGULAR_SPEED="${SUPERVISOR_ANGULAR_SPEED:-0.30}"
FINAL_TRACKER_TIMEOUT_SEC="${FINAL_TRACKER_TIMEOUT_SEC:-90}"

cd "$PROJECT_DIR"

mkdir -p bags/real results

if [ -z "$RUN_ID" ]; then
  RUN_ID="$(python3 scripts/next_real_run_id.py \
    --results-csv "$RESULTS_CSV" \
    --prefix supervisor_validation_)"
fi

BAG_DIR="bags/real/$RUN_ID"
BAG_LOG="results/${RUN_ID}_bag.log"

if [ -e "$BAG_DIR" ]; then
  echo "ERROR: bag output already exists: $BAG_DIR"
  echo "Choose a new run ID or move the existing bag deliberately."
  exit 1
fi

if [ ! -f "$PREDICTION_FILE" ]; then
  echo "ERROR: prediction file not found: $PREDICTION_FILE"
  exit 1
fi

source /opt/ros/humble/setup.bash
source /opt/tb3_src_ws/install/setup.bash

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
export LDS_MODEL="${LDS_MODEL:-LDS-01}"

echo "Supervisor validation run ID: $RUN_ID"
echo "Project directory: $(pwd)"
echo "Prediction file: $PREDICTION_FILE"
echo "Tracker pose file: $TRACKER_POSE_FILE"
echo "Results CSV: $RESULTS_CSV"
echo
echo "Camera host requirement:"
echo "  Start vision_tracker/main.py before the robot reaches the final area."
echo "  If the camera runs on another host, make $TRACKER_POSE_FILE visible here."
echo
echo "Checking robot topics..."
ros2 topic list | grep -E '^/cmd_vel$' >/dev/null
ros2 topic list | grep -E '^/odom$' >/dev/null

echo
echo "Safety requirements:"
echo "  - clear the full supervisor route"
echo "  - keep an operator near the TurtleBot"
echo "  - keep Ctrl+C ready in this terminal"
echo "  - keep a physical stop option available"
echo
echo "Safety stop command for another terminal:"
echo '  ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}"'
echo
read -r -p "Type RUN to start bag recording and publish /cmd_vel: " CONFIRM
if [ "$CONFIRM" != "RUN" ]; then
  echo "Cancelled."
  exit 130
fi

BAG_PID=""

stop_robot() {
  ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0}, angular: {z: 0.0}}" >/dev/null 2>&1 || true
}

stop_bag() {
  if [ -n "$BAG_PID" ] && kill -0 "$BAG_PID" >/dev/null 2>&1; then
    echo "Stopping bag recording..."
    kill -INT "$BAG_PID" || true
    wait "$BAG_PID" || true
  fi
}

cleanup() {
  stop_robot
  stop_bag
}

trap cleanup EXIT INT TERM

echo "Starting bag recording for $RUN_ID..."
ros2 bag record -o "$BAG_DIR" /cmd_vel /odom /imu /battery_state > "$BAG_LOG" 2>&1 &
BAG_PID=$!

sleep 2

echo "Running supervisor validation route..."
python3 scripts/supervisor_route_validation.py \
  --run-id "$RUN_ID" \
  --prediction "$PREDICTION_FILE" \
  --results-csv "$RESULTS_CSV" \
  --tracker-pose-file "$TRACKER_POSE_FILE" \
  --linear-speed "$LINEAR_SPEED" \
  --angular-speed "$ANGULAR_SPEED" \
  --final-tracker-timeout-sec "$FINAL_TRACKER_TIMEOUT_SEC" \
  --yes

stop_robot
stop_bag
trap - EXIT INT TERM

echo "Supervisor validation experiment finished: $RUN_ID"
echo "Bag: $BAG_DIR"
echo "Bag log: $BAG_LOG"
echo "Results CSV: $RESULTS_CSV"
