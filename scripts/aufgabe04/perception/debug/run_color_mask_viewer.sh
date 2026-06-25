#!/usr/bin/env bash
set -eo pipefail

cd /workspace/mii-amr

source /opt/ros/humble/setup.bash
source /opt/tb3_src_ws/install/setup.bash

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export LDS_MODEL="${LDS_MODEL:-LDS-02}"

python3 -m scripts.aufgabe04.perception.debug.color_mask_viewer "$@"
