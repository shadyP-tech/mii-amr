#!/usr/bin/env bash
set -u
set -o pipefail

WRAPPER_VERSION="run_with_bundle_v1"

usage() {
  cat <<'EOF'
Usage:
  scripts/common/run_with_bundle.sh [bundle options] RUN_ID -- COMMAND [ARGS...]

Bundle options:
  --namespace NAME
  --cmd-vel-topic TOPIC
  --scan-topic TOPIC
  --odom-topic TOPIC
  --amcl-topic TOPIC
  --map-frame FRAME
  --odom-frame FRAME
  --base-frame FRAME
  --output-root DIR

The wrapper records diagnostics and runs exactly COMMAND [ARGS...] through tee.
It never publishes motion and does not replace the wrapped command's preflight.
EOF
}

fail_usage() {
  echo "error: $*" >&2
  usage >&2
  exit 2
}

shell_quote_args() {
  local arg
  for arg in "$@"; do
    printf '%q ' "$arg"
  done
  printf '\n'
}

clean_namespace() {
  local ns="$1"
  ns="${ns#/}"
  ns="${ns%/}"
  if [[ -n "$ns" ]]; then
    printf '/%s' "$ns"
  fi
}

resolve_topic() {
  local topic="$1"
  local namespace="$2"
  if [[ -z "$topic" ]]; then
    echo "topic name must not be empty" >&2
    return 1
  fi
  if [[ "$topic" == /* ]]; then
    printf '%s\n' "$topic"
    return 0
  fi
  topic="${topic#/}"
  if [[ -n "$namespace" ]]; then
    printf '%s/%s\n' "$namespace" "$topic"
  else
    printf '/%s\n' "$topic"
  fi
}

validate_run_id() {
  local run_id="$1"
  if [[ -z "$run_id" ]]; then
    echo "run ID must not be empty" >&2
    return 1
  fi
  if [[ "$run_id" == -* || "$run_id" == "." || "$run_id" == *".."* ]]; then
    echo "run ID is not safe: $run_id" >&2
    return 1
  fi
  if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "run ID may contain only letters, numbers, dot, underscore, and dash" >&2
    return 1
  fi
}

log_setup() {
  local message="$1"
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$message" | tee -a "$BUNDLE_SETUP_LOG" >&2
}

run_capture() {
  local output_file="$1"
  shift
  {
    echo "\$ $(shell_quote_args "$@")"
    "$@"
  } >"$output_file" 2>&1 || {
    local status=$?
    echo "command failed with exit code $status" >>"$output_file"
    return 0
  }
}

run_capture_timeout() {
  local output_file="$1"
  local seconds="$2"
  shift 2
  if command -v timeout >/dev/null 2>&1; then
    run_capture "$output_file" timeout "${seconds}s" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    run_capture "$output_file" gtimeout "${seconds}s" "$@"
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$seconds" "$output_file" "$@" <<'PY'
import subprocess
import sys

seconds = float(sys.argv[1])
output_file = sys.argv[2]
command = sys.argv[3:]
with open(output_file, "w", encoding="utf-8") as file:
    file.write("$ " + " ".join(command) + "\n")
    try:
        result = subprocess.run(
            command,
            stdout=file,
            stderr=subprocess.STDOUT,
            timeout=seconds,
            check=False,
        )
        if result.returncode != 0:
            file.write(f"command failed with exit code {result.returncode}\n")
    except subprocess.TimeoutExpired:
        file.write(f"command timed out after {seconds:.1f} seconds\n")
    except FileNotFoundError as exc:
        file.write(f"command not found: {exc}\n")
PY
  else
    run_capture "$output_file" "$@"
  fi
}

write_manifest_start() {
  {
    echo "wrapper_version=$WRAPPER_VERSION"
    echo "run_id=$RUN_ID"
    echo "bundle_dir=$BUNDLE_DIR"
    echo "output_root=$OUTPUT_ROOT"
    echo "start_time_utc=$START_TIME"
    echo "host=$(hostname 2>/dev/null || true)"
    echo "user=${USER:-}"
    echo "pwd=$PWD"
    echo "configured_namespace=$CONFIG_NAMESPACE"
    echo "resolved_namespace=$RESOLVED_NAMESPACE"
    echo "configured_cmd_vel_topic=$CMD_VEL_TOPIC"
    echo "resolved_cmd_vel_topic=$RESOLVED_CMD_VEL_TOPIC"
    echo "configured_scan_topic=$SCAN_TOPIC"
    echo "resolved_scan_topic=$RESOLVED_SCAN_TOPIC"
    echo "configured_odom_topic=$ODOM_TOPIC"
    echo "resolved_odom_topic=$RESOLVED_ODOM_TOPIC"
    echo "configured_amcl_topic=$AMCL_TOPIC"
    echo "resolved_amcl_topic=$RESOLVED_AMCL_TOPIC"
    echo "map_frame=$MAP_FRAME"
    echo "odom_frame=$ODOM_FRAME"
    echo "base_frame=$BASE_FRAME"
    echo "command=$(shell_quote_args "${COMMAND_ARGS[@]}")"
  } >"$MANIFEST_FILE"
}

append_manifest_end() {
  {
    echo "end_time_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "command_exit_code=$1"
  } >>"$MANIFEST_FILE"
}

collect_env() {
  env | sort >"$BUNDLE_DIR/env.txt"
  {
    echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-}"
    echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-}"
    echo "TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-}"
    echo "LDS_MODEL=${LDS_MODEL:-}"
  } >"$BUNDLE_DIR/ros_env.txt"
}

collect_git() {
  run_capture "$BUNDLE_DIR/git_status.txt" git status --short
  {
    echo "\$ git rev-parse --abbrev-ref HEAD"
    git rev-parse --abbrev-ref HEAD 2>&1 || true
    echo "\$ git rev-parse HEAD"
    git rev-parse HEAD 2>&1 || true
  } >"$BUNDLE_DIR/git_rev.txt"
}

collect_ros_pre() {
  run_capture "$BUNDLE_DIR/ros_topics.txt" ros2 topic list
  run_capture "$BUNDLE_DIR/ros_nodes.txt" ros2 node list
  run_capture "$BUNDLE_DIR/ros_actions.txt" ros2 action list
  run_capture "$BUNDLE_DIR/cmd_vel_info.txt" ros2 topic info "$RESOLVED_CMD_VEL_TOPIC" --verbose
  run_capture_timeout "$BUNDLE_DIR/scan_once.txt" 4 ros2 topic echo --once "$RESOLVED_SCAN_TOPIC"
  run_capture_timeout "$BUNDLE_DIR/odom_once.txt" 4 ros2 topic echo --once "$RESOLVED_ODOM_TOPIC"
  run_capture_timeout "$BUNDLE_DIR/amcl_pose_once.txt" 4 ros2 topic echo --once "$RESOLVED_AMCL_TOPIC"
  run_capture_timeout "$BUNDLE_DIR/navigate_to_pose_status_once.txt" 4 ros2 topic echo --once "$NAV2_STATUS_TOPIC"
  if [[ "$NAV2_STATUS_TOPIC" != "$NAMESPACED_NAV2_STATUS_TOPIC" ]]; then
    run_capture_timeout "$BUNDLE_DIR/namespaced_navigate_to_pose_status_once.txt" 4 ros2 topic echo --once "$NAMESPACED_NAV2_STATUS_TOPIC"
  fi
  collect_tf
}

collect_ros_post() {
  run_capture "$BUNDLE_DIR/post_ros_topics.txt" ros2 topic list
  run_capture "$BUNDLE_DIR/post_ros_nodes.txt" ros2 node list
  run_capture "$BUNDLE_DIR/post_ros_actions.txt" ros2 action list
  run_capture "$BUNDLE_DIR/post_cmd_vel_info.txt" ros2 topic info "$RESOLVED_CMD_VEL_TOPIC" --verbose
}

collect_tf() {
  (
    cd "$BUNDLE_DIR" || exit 0
    if ros2 run tf2_tools view_frames >tf_frames.txt 2>&1; then
      exit 0
    fi
    {
      echo "tf2_tools view_frames failed; falling back to tf2_echo captures"
      echo
      echo "map/base check: $MAP_FRAME -> $BASE_FRAME"
    } >tf_frames.txt
  )
  if ! grep -q "fallback" "$BUNDLE_DIR/tf_frames.txt" 2>/dev/null; then
    return 0
  fi
  run_capture_timeout "$BUNDLE_DIR/tf_map_base_once.txt" 4 ros2 run tf2_ros tf2_echo "$MAP_FRAME" "$BASE_FRAME"
  run_capture_timeout "$BUNDLE_DIR/tf_odom_base_once.txt" 4 ros2 run tf2_ros tf2_echo "$ODOM_FRAME" "$BASE_FRAME"
}

CONFIG_NAMESPACE=""
CMD_VEL_TOPIC="cmd_vel"
SCAN_TOPIC="scan"
ODOM_TOPIC="odom"
AMCL_TOPIC="amcl_pose"
MAP_FRAME="map"
ODOM_FRAME="odom"
BASE_FRAME="base_footprint"
OUTPUT_ROOT="${RUN_BUNDLE_ROOT:-results/real_runs}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace)
      [[ $# -ge 2 ]] || fail_usage "--namespace requires a value"
      CONFIG_NAMESPACE="$2"
      shift 2
      ;;
    --cmd-vel-topic)
      [[ $# -ge 2 ]] || fail_usage "--cmd-vel-topic requires a value"
      CMD_VEL_TOPIC="$2"
      shift 2
      ;;
    --scan-topic)
      [[ $# -ge 2 ]] || fail_usage "--scan-topic requires a value"
      SCAN_TOPIC="$2"
      shift 2
      ;;
    --odom-topic)
      [[ $# -ge 2 ]] || fail_usage "--odom-topic requires a value"
      ODOM_TOPIC="$2"
      shift 2
      ;;
    --amcl-topic)
      [[ $# -ge 2 ]] || fail_usage "--amcl-topic requires a value"
      AMCL_TOPIC="$2"
      shift 2
      ;;
    --map-frame)
      [[ $# -ge 2 ]] || fail_usage "--map-frame requires a value"
      MAP_FRAME="$2"
      shift 2
      ;;
    --odom-frame)
      [[ $# -ge 2 ]] || fail_usage "--odom-frame requires a value"
      ODOM_FRAME="$2"
      shift 2
      ;;
    --base-frame)
      [[ $# -ge 2 ]] || fail_usage "--base-frame requires a value"
      BASE_FRAME="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || fail_usage "--output-root requires a value"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      fail_usage "RUN_ID is required before --"
      ;;
    -*)
      fail_usage "unknown option: $1"
      ;;
    *)
      RUN_ID="$1"
      shift
      break
      ;;
  esac
done

[[ "${RUN_ID:-}" != "" ]] || fail_usage "RUN_ID is required"
validate_run_id "$RUN_ID" || exit 2
[[ $# -gt 0 && "$1" == "--" ]] || fail_usage "expected -- before command"
shift
[[ $# -gt 0 ]] || fail_usage "COMMAND is required after --"
COMMAND_ARGS=("$@")

RESOLVED_NAMESPACE="$(clean_namespace "$CONFIG_NAMESPACE")"
RESOLVED_CMD_VEL_TOPIC="$(resolve_topic "$CMD_VEL_TOPIC" "$RESOLVED_NAMESPACE")" || exit 2
RESOLVED_SCAN_TOPIC="$(resolve_topic "$SCAN_TOPIC" "$RESOLVED_NAMESPACE")" || exit 2
RESOLVED_ODOM_TOPIC="$(resolve_topic "$ODOM_TOPIC" "$RESOLVED_NAMESPACE")" || exit 2
RESOLVED_AMCL_TOPIC="$(resolve_topic "$AMCL_TOPIC" "$RESOLVED_NAMESPACE")" || exit 2
NAV2_STATUS_TOPIC="/navigate_to_pose/_action/status"
if [[ -n "$RESOLVED_NAMESPACE" ]]; then
  NAMESPACED_NAV2_STATUS_TOPIC="$RESOLVED_NAMESPACE/navigate_to_pose/_action/status"
else
  NAMESPACED_NAV2_STATUS_TOPIC="$NAV2_STATUS_TOPIC"
fi

BUNDLE_DIR="$OUTPUT_ROOT/$RUN_ID"
BUNDLE_SETUP_LOG="$BUNDLE_DIR/bundle_setup.log"
MANIFEST_FILE="$BUNDLE_DIR/manifest.txt"
TERMINAL_LOG="$BUNDLE_DIR/terminal_run.log"
START_TIME="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

mkdir -p "$BUNDLE_DIR" || exit 1
: >"$BUNDLE_SETUP_LOG"
shell_quote_args "${COMMAND_ARGS[@]}" >"$BUNDLE_DIR/command.txt"
write_manifest_start
collect_env
collect_git

log_setup "collecting pre-run ROS diagnostics"
collect_ros_pre

log_setup "running wrapped command"
set +e
"${COMMAND_ARGS[@]}" 2>&1 | tee "$TERMINAL_LOG"
COMMAND_STATUS=${PIPESTATUS[0]}
set -e

log_setup "collecting post-run ROS diagnostics"
collect_ros_post
append_manifest_end "$COMMAND_STATUS"

ARCHIVE_PATH="${OUTPUT_ROOT%/}/${RUN_ID}_debug_bundle.tar.gz"
{
  echo "tar -czf $ARCHIVE_PATH $BUNDLE_DIR"
} >"$BUNDLE_DIR/archive_hint.txt"

log_setup "bundle complete: $BUNDLE_DIR"
echo "Bundle directory: $BUNDLE_DIR"
echo "Archive command:"
cat "$BUNDLE_DIR/archive_hint.txt"
exit "$COMMAND_STATUS"
