#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/aufgabe04/simulation/run_with_debug_bundle.sh RUN_ID [options] -- COMMAND [ARG ...]

Run an explicitly supplied Gazebo simulation command while passively capturing
timestamp-aligned telemetry, camera frames, semantic events, terminal output,
ROS graph snapshots, and an optional ROS bag. It never publishes motion itself
and refuses to start unless /clock is present.

Options:
  --output-root DIR             Default: results/aufgabe04/simulation_debug_runs
  --world PATH                 World description recorded in the manifest
  --expected TEXT              Expected behavior for the GPT debug question
  --observed TEXT              Observed failure/symptom, if already known
  --semantic-log PATH          Existing runner JSONL event log to merge
  --perception-dir DIR         Copy perception images; may be repeated
  --sample-hz HZ               Telemetry sampling rate; default: 5
  --frame-fps FPS              Camera frame sampling rate; default: 1
  --camera-topic TOPIC         Default: /camera/image_raw
  --overview-image-topic TOPIC Optional overhead Gazebo Image topic
  --odom-topic TOPIC           Default: /odom
  --cmd-vel-topic TOPIC        Default: /cmd_vel
  --scan-topic TOPIC           Default: /scan
  --model-states-topic TOPIC   Default: /gazebo/model_states
  --model-name NAME            Default: burger
  --no-camera                  Do not capture camera frames
  --no-model-states            Do not capture Gazebo ground truth
  --no-bag                     Do not record the raw ROS bag
  --help

The command should be a simulation runner such as:
  python3 scripts/aufgabe04/navigation/entrypoints/run_single_station_segment.py ... --allow-sim-time
EOF
}

fail_usage() {
  echo "error: $*" >&2
  usage >&2
  exit 2
}

shell_quote_args() {
  local argument
  for argument in "$@"; do
    printf '%q ' "$argument"
  done
  printf '\n'
}

RUN_ID="${1:-}"
[[ -n "$RUN_ID" && "$RUN_ID" != --* ]] || fail_usage "RUN_ID is required"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail_usage "invalid RUN_ID"
shift

OUTPUT_ROOT="results/aufgabe04/simulation_debug_runs"
WORLD="simulation/gazebo/worlds/aufgabe04_stands.world"
EXPECTED=""
OBSERVED=""
SEMANTIC_LOG=""
PERCEPTION_DIRS=()
SAMPLE_HZ="5"
FRAME_FPS="1"
CAMERA_TOPIC="/camera/image_raw"
OVERVIEW_IMAGE_TOPIC=""
ODOM_TOPIC="/odom"
CMD_VEL_TOPIC="/cmd_vel"
SCAN_TOPIC="/scan"
MODEL_STATES_TOPIC="/gazebo/model_states"
MODEL_NAME="burger"
CAPTURE_CAMERA=1
CAPTURE_MODEL_STATES=1
RECORD_BAG=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; OUTPUT_ROOT="$2"; shift 2 ;;
    --world) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; WORLD="$2"; shift 2 ;;
    --expected) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; EXPECTED="$2"; shift 2 ;;
    --observed) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; OBSERVED="$2"; shift 2 ;;
    --semantic-log) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; SEMANTIC_LOG="$2"; shift 2 ;;
    --perception-dir) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; PERCEPTION_DIRS+=("$2"); shift 2 ;;
    --sample-hz) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; SAMPLE_HZ="$2"; shift 2 ;;
    --frame-fps) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; FRAME_FPS="$2"; shift 2 ;;
    --camera-topic) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; CAMERA_TOPIC="$2"; shift 2 ;;
    --overview-image-topic) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; OVERVIEW_IMAGE_TOPIC="$2"; shift 2 ;;
    --odom-topic) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; ODOM_TOPIC="$2"; shift 2 ;;
    --cmd-vel-topic) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; CMD_VEL_TOPIC="$2"; shift 2 ;;
    --scan-topic) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; SCAN_TOPIC="$2"; shift 2 ;;
    --model-states-topic) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; MODEL_STATES_TOPIC="$2"; shift 2 ;;
    --model-name) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; MODEL_NAME="$2"; shift 2 ;;
    --no-camera) CAPTURE_CAMERA=0; shift ;;
    --no-model-states) CAPTURE_MODEL_STATES=0; shift ;;
    --no-bag) RECORD_BAG=0; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) fail_usage "unknown option: $1" ;;
  esac
done

[[ $# -gt 0 ]] || fail_usage "COMMAND is required after --"
COMMAND_ARGS=("$@")
BUNDLE_DIR="${OUTPUT_ROOT%/}/$RUN_ID"
[[ ! -e "$BUNDLE_DIR" ]] || { echo "error: bundle already exists: $BUNDLE_DIR" >&2; exit 2; }

command -v ros2 >/dev/null 2>&1 || { echo "error: ros2 is not available; source the ROS environment" >&2; exit 2; }
if ! ros2 topic list 2>/dev/null | grep -qx '/clock'; then
  echo "error: /clock is missing; start Gazebo and use this wrapper only for simulation" >&2
  exit 2
fi

mkdir -p "$BUNDLE_DIR"
shell_quote_args "${COMMAND_ARGS[@]}" >"$BUNDLE_DIR/command.txt"
{
  echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-}"
  echo "TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-}"
  echo "GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH:-}"
  echo "camera_topic=$CAMERA_TOPIC"
  echo "overview_image_topic=$OVERVIEW_IMAGE_TOPIC"
  echo "odom_topic=$ODOM_TOPIC"
  echo "cmd_vel_topic=$CMD_VEL_TOPIC"
  echo "scan_topic=$SCAN_TOPIC"
  echo "model_states_topic=$MODEL_STATES_TOPIC"
  echo "model_name=$MODEL_NAME"
} >"$BUNDLE_DIR/environment.txt"
ros2 topic list -t >"$BUNDLE_DIR/ros_topics_before.txt" 2>&1 || true
ros2 node list >"$BUNDLE_DIR/ros_nodes_before.txt" 2>&1 || true

CAPTURE_ARGS=(
  --bundle-dir "$BUNDLE_DIR"
  --sample-hz "$SAMPLE_HZ"
  --frame-fps "$FRAME_FPS"
  --camera-topic "$CAMERA_TOPIC"
  --odom-topic "$ODOM_TOPIC"
  --cmd-vel-topic "$CMD_VEL_TOPIC"
  --scan-topic "$SCAN_TOPIC"
  --model-states-topic "$MODEL_STATES_TOPIC"
  --model-name "$MODEL_NAME"
)
[[ -z "$OVERVIEW_IMAGE_TOPIC" ]] || CAPTURE_ARGS+=(--overview-image-topic "$OVERVIEW_IMAGE_TOPIC")
[[ "$CAPTURE_CAMERA" -eq 1 ]] || CAPTURE_ARGS+=(--no-camera)
[[ "$CAPTURE_MODEL_STATES" -eq 1 ]] || CAPTURE_ARGS+=(--no-model-states)

python3 -m scripts.aufgabe04.simulation.debug_capture_node "${CAPTURE_ARGS[@]}" \
  >"$BUNDLE_DIR/capture.log" 2>&1 &
CAPTURE_PID=$!
BAG_PID=""
BAG_PATH=""

cleanup_background() {
  if [[ -n "${CAPTURE_PID:-}" ]] && kill -0 "$CAPTURE_PID" 2>/dev/null; then
    kill -INT "$CAPTURE_PID" 2>/dev/null || true
    wait "$CAPTURE_PID" 2>/dev/null || true
  fi
  CAPTURE_PID=""
  if [[ -n "${BAG_PID:-}" ]] && kill -0 "$BAG_PID" 2>/dev/null; then
    kill -INT "$BAG_PID" 2>/dev/null || true
    wait "$BAG_PID" 2>/dev/null || true
  fi
  BAG_PID=""
}
trap cleanup_background EXIT

if [[ "$RECORD_BAG" -eq 1 ]]; then
  BAG_PATH="$BUNDLE_DIR/rosbag/run"
  BAG_TOPICS=(/clock "$ODOM_TOPIC" "$CMD_VEL_TOPIC" "$SCAN_TOPIC" /tf /tf_static)
  [[ "$CAPTURE_CAMERA" -eq 1 ]] && BAG_TOPICS+=("$CAMERA_TOPIC")
  [[ -z "$OVERVIEW_IMAGE_TOPIC" ]] || BAG_TOPICS+=("$OVERVIEW_IMAGE_TOPIC")
  [[ "$CAPTURE_MODEL_STATES" -eq 1 ]] && BAG_TOPICS+=("$MODEL_STATES_TOPIC")
  mkdir -p "$BUNDLE_DIR/rosbag"
  ros2 bag record -o "$BAG_PATH" "${BAG_TOPICS[@]}" >"$BUNDLE_DIR/rosbag_record.log" 2>&1 &
  BAG_PID=$!
fi

sleep 0.5
if ! kill -0 "$CAPTURE_PID" 2>/dev/null; then
  echo "error: debug capture exited before the simulation command; inspect $BUNDLE_DIR/capture.log" >&2
  exit 2
fi
if [[ -n "$BAG_PID" ]] && ! kill -0 "$BAG_PID" 2>/dev/null; then
  echo "error: rosbag recording exited before the simulation command; inspect $BUNDLE_DIR/rosbag_record.log" >&2
  exit 2
fi

echo "Simulation debug bundle: $BUNDLE_DIR"
echo "Running: $(shell_quote_args "${COMMAND_ARGS[@]}")"
set +e
"${COMMAND_ARGS[@]}" 2>&1 | tee "$BUNDLE_DIR/terminal.log"
COMMAND_STATUS=${PIPESTATUS[0]}
set -e

cleanup_background
ros2 topic list -t >"$BUNDLE_DIR/ros_topics_after.txt" 2>&1 || true
ros2 node list >"$BUNDLE_DIR/ros_nodes_after.txt" 2>&1 || true
if [[ -n "$BAG_PATH" && -d "$BAG_PATH" ]]; then
  ros2 bag info "$BAG_PATH" >"$BUNDLE_DIR/rosbag_info.txt" 2>&1 || true
fi

BUILD_ARGS=(
  --bundle-dir "$BUNDLE_DIR"
  --run-id "$RUN_ID"
  --telemetry-jsonl "$BUNDLE_DIR/telemetry.jsonl"
  --expected-behavior "$EXPECTED"
  --observed-behavior "$OBSERVED"
  --world "$WORLD"
  --command-exit-code "$COMMAND_STATUS"
)
[[ -z "$SEMANTIC_LOG" ]] || BUILD_ARGS+=(--semantic-jsonl "$SEMANTIC_LOG")
[[ -z "$BAG_PATH" ]] || BUILD_ARGS+=(--bag-path "$BAG_PATH")
for directory in "${PERCEPTION_DIRS[@]}"; do
  BUILD_ARGS+=(--perception-dir "$directory")
done
python3 -m scripts.aufgabe04.simulation.debug_bundle "${BUILD_ARGS[@]}"

echo "Debug summary: $BUNDLE_DIR/summary.md"
exit "$COMMAND_STATUS"
