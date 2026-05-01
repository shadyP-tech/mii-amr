#!/usr/bin/env bash

set -e

if [ -z "$1" ]; then
  echo "Usage: ./scripts/run_experiment.sh run_01"
  exit 1
fi

RUN_ID="$1"

cd /workspace/mii-amr
source /opt/ros/humble/setup.bash

mkdir -p bags results

echo "Resetting simulation..."
ros2 service call /reset_simulation std_srvs/srv/Empty || true

sleep 2

echo "Starting bag recording for $RUN_ID..."
BAG_TOPICS=(/cmd_vel /odom)
for OPTIONAL_TOPIC in /imu /battery_state; do
  if ros2 topic list | grep -qx "$OPTIONAL_TOPIC"; then
    BAG_TOPICS+=("$OPTIONAL_TOPIC")
  fi
done
echo "Recording topics: ${BAG_TOPICS[*]}"
ros2 bag record -o "bags/$RUN_ID" "${BAG_TOPICS[@]}" > "results/${RUN_ID}_bag.log" 2>&1 &
BAG_PID=$!

sleep 2

echo "Running scripted drive..."
python3 scripts/scripted_drive.py "$RUN_ID"

sleep 1

echo "Stopping bag recording..."
kill -INT "$BAG_PID"
wait "$BAG_PID" || true

echo "Experiment $RUN_ID finished."
