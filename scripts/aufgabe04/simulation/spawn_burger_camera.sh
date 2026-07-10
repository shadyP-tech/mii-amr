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

echo "Spawned $ENTITY_NAME with simulated camera."
echo "Used simulation-only camera SDF: $MODEL_SDF"
echo "Verify camera topics: ros2 topic list | grep -E 'image_raw|camera_info'"
