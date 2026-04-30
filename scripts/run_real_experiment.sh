#!/usr/bin/env bash

set -e

if [ "$#" -gt 1 ]; then
  echo "Usage: ./scripts/run_real_experiment.sh [run_id]"
  exit 1
fi

RUN_ID="${1:-}"

cd /workspace/mii-amr

mkdir -p bags/real results

if [ -z "$RUN_ID" ]; then
  RUN_ID="$(python3 scripts/next_real_run_id.py)"
fi

echo "Using run ID: $RUN_ID"

source /opt/ros/humble/setup.bash
source /opt/tb3_src_ws/install/setup.bash

export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-01

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
