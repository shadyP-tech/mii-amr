#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="/workspace/mii-amr"

set +u
if [ -f /opt/ros/humble/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

if [ -f /opt/tb3_src_ws/install/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/tb3_src_ws/install/setup.bash
fi

if [ -f "${HOME}/turtlebot3_ws/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/turtlebot3_ws/install/setup.bash"
fi
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
export LDS_MODEL="${LDS_MODEL:-LDS-02}"

cd "${REPO_DIR}"
# This launcher configures the TurtleBot LDS full-rotation scanner. The viewer
# still validates the original angular metadata and endpoint distance before
# joining a seam; an explicit later CLI argument can select linear scans.
python3 -m scripts.aufgabe04.perception.debug.stand_axis_viewer --scan-topology-profile full_rotation "$@"
