#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/mii-amr"

if [ -f /opt/ros/humble/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

if [ -f "${HOME}/turtlebot3_ws/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/turtlebot3_ws/install/setup.bash"
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
export LDS_MODEL="${LDS_MODEL:-LDS-02}"

cd "${REPO_DIR}"
python3 -m scripts.aufgabe04.perception.debug.stand_axis_viewer "$@"
