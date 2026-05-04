#!/usr/bin/env bash

set -e

if [ "$#" -gt 1 ]; then
  echo "Usage: ./scripts/run_real_experiment.sh [run_id]"
  exit 1
fi

RUN_ID="${1:-}"
RUN_MODE="${RUN_MODE:-rotate-in-place}"
RUN_ANGLE_DEG="${RUN_ANGLE_DEG:--90}"
RUN_ANGULAR_SPEED="${RUN_ANGULAR_SPEED:-0.3}"
RUN_SPEED="${RUN_SPEED:-0.1}"
RUN_DURATION_SEC="${RUN_DURATION_SEC:-3.0}"

cd /workspace/mii-amr

mkdir -p bags/real results

if [ -z "$RUN_ID" ]; then
  if [ "$RUN_MODE" = "rotate-in-place" ] || [ "$RUN_MODE" = "rotation" ] || [ "$RUN_MODE" = "rotate" ]; then
    RUN_PREFIX="$(python3 -c 'import sys; angle = float(sys.argv[1]); direction = "cw" if angle < 0 else "ccw"; amount = ("%g" % abs(angle)).replace(".", "p"); print(f"run_real_rot_{direction}{amount}_")' "$RUN_ANGLE_DEG")"
    RUN_ID="$(python3 scripts/next_real_run_id.py --results-csv results/real_rotation_runs.csv --prefix "$RUN_PREFIX")"
  else
    RUN_ID="$(python3 scripts/next_real_run_id.py)"
  fi
fi

echo "Using run ID: $RUN_ID"
echo "Real run mode: $RUN_MODE"
if [ "$RUN_MODE" = "rotate-in-place" ] || [ "$RUN_MODE" = "rotation" ] || [ "$RUN_MODE" = "rotate" ]; then
  echo "Rotation command: angle=${RUN_ANGLE_DEG} deg, angular_speed=${RUN_ANGULAR_SPEED} rad/s"
else
  echo "Linear command: speed=${RUN_SPEED} m/s, duration=${RUN_DURATION_SEC} s"
fi

source /opt/ros/humble/setup.bash
source /opt/tb3_src_ws/install/setup.bash

export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-01
export RUN_MODE RUN_ANGLE_DEG RUN_ANGULAR_SPEED RUN_SPEED RUN_DURATION_SEC

echo "Checking robot topics..."
ros2 topic list | grep /cmd_vel >/dev/null
ros2 topic list | grep /odom >/dev/null

echo "Checking camera start pose for $RUN_ID..."
echo "Start vision_tracker/main.py in another terminal before continuing."
python3 vision_tracker/start_pose_gate.py "$RUN_ID"

echo "Starting bag recording for $RUN_ID..."
ros2 bag record -o "bags/real/$RUN_ID" /cmd_vel /odom /imu /battery_state > "results/${RUN_ID}_bag.log" 2>&1 &
BAG_PID=$!

sleep 2

echo "Running real scripted drive..."
python3 scripts/real_scripted_drive.py "$RUN_ID"

sleep 1

echo "Stopping bag recording..."
kill -INT "$BAG_PID"
wait "$BAG_PID" || true

echo "Experiment $RUN_ID finished."
