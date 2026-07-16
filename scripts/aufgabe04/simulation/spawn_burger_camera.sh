#!/usr/bin/env bash
set -euo pipefail

SOURCE_MODEL_SDF="${BURGER_CAMERA_SDF:-/opt/tb3_src_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger_cam/model.sdf}"
MODEL_SDF="${BURGER_CAMERA_SIM_SDF:-/tmp/aufgabe04_turtlebot3_burger_cam.sdf}"
ENTITY_NAME="${BURGER_CAMERA_ENTITY:-burger}"
SPAWN_X="${BURGER_CAMERA_X:-0.0}"
SPAWN_Y="${BURGER_CAMERA_Y:-0.0}"
SPAWN_Z="${BURGER_CAMERA_Z:-0.01}"
SPAWN_YAW="${BURGER_CAMERA_YAW:-0.0}"

if [[ ! -f "$SOURCE_MODEL_SDF" ]]; then
  echo "Burger camera model not found: $SOURCE_MODEL_SDF" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/prepare_burger_camera_model.py" \
  --source "$SOURCE_MODEL_SDF" \
  --output "$MODEL_SDF" \
  --horizontal-fov-rad "${BURGER_CAMERA_HFOV_RAD:-1.3962634}"

export TURTLEBOT3_MODEL=burger
ros2 run gazebo_ros spawn_entity.py \
  -entity "$ENTITY_NAME" \
  -file "$MODEL_SDF" \
  -x "$SPAWN_X" -y "$SPAWN_Y" -z "$SPAWN_Z" -Y "$SPAWN_YAW"

start_static_tf_once() {
  local node_name="$1"
  local pid_file="/tmp/${node_name}.pid"
  shift
  if [[ -f "$pid_file" ]]; then
    local existing_pid
    existing_pid="$(<"$pid_file")"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      return
    fi
  fi
  nohup ros2 run tf2_ros static_transform_publisher \
    "$@" --ros-args -r "__node:=${node_name}" \
    >"/tmp/${node_name}.log" 2>&1 &
  echo "$!" >"$pid_file"
}

# Gazebo's diff-drive plugin publishes odom -> base_footprint, but the SDF
# fixed joints are not automatically mirrored into ROS TF.  Publish the fixed
# Burger chain so camera_link and base_scan move transitively with the robot.
start_static_tf_once aufgabe04_sim_base_link_tf \
  --x 0.0 --y 0.0 --z 0.010 --roll 0.0 --pitch 0.0 --yaw 0.0 \
  --frame-id base_footprint --child-frame-id base_link
start_static_tf_once aufgabe04_sim_base_scan_tf \
  --x -0.032 --y 0.0 --z 0.171 --roll 0.0 --pitch 0.0 --yaw 0.0 \
  --frame-id base_link --child-frame-id base_scan
start_static_tf_once aufgabe04_sim_camera_link_tf \
  --x 0.076 --y 0.0 --z 0.093 --roll 0.0 --pitch 0.0 --yaw 0.0 \
  --frame-id base_link --child-frame-id camera_link

echo "Spawned $ENTITY_NAME with simulated camera."
echo "Used simulation-only camera SDF: $MODEL_SDF"
echo "Verify camera topics: ros2 topic list | grep -E 'image_raw|camera_info'"
echo "Verify stamped world pose: ros2 topic echo /gazebo_ground_truth --once"
echo "Verify camera TF: ros2 run tf2_ros tf2_echo odom camera_link"
