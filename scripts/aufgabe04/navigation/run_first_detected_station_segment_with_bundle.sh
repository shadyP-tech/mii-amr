#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/aufgabe04/navigation/run_first_detected_station_segment_with_bundle.sh RUN_ID [options]

Wrap the first detected-station route runner in scripts/common/run_with_bundle.sh.
The wrapped runner still performs strict preflight and still requires typing RUN
before publishing motion.

Options:
  --leg-index INDEX                  Default: 0
  --namespace NAME                   Default: empty
  --scan-topic TOPIC                 Default: scan
  --odom-topic TOPIC                 Default: odom
  --cmd-vel-topic TOPIC              Default: cmd_vel
  --amcl-topic TOPIC                 Default: amcl_pose
  --map-frame FRAME                  Default: map
  --odom-frame FRAME                 Default: odom
  --base-frame FRAME                 Default: base_footprint
  --localization-source amcl|tf      Default: amcl
  --max-amcl-age-sec SEC             Default: 5.0
  --max-scan-age-sec SEC             Default: 1.0
  --max-odom-age-sec SEC             Default: 1.0
  --max-tf-age-sec SEC               Default: 1.0
  --preflight-observation-window-sec SEC
                                    Default: 8.0
  --initial-sensor-wait-sec SEC      Default: 2.0
  --allow-idle-nav2-publishers       Allow /behavior_server and /velocity_smoother in preflight.
  --no-initialpose-prompt            Do not pause for RViz 2D Pose Estimate before preflight.
  --route-csv PATH                   Default: results/aufgabe04/routes/first_detected_station_route.csv
  --diagnostics-json PATH            Default: results/aufgabe04/routes/first_detected_station_route_diagnostics.json
  --results-csv PATH                 Default: results/aufgabe04/station_segment_runs.csv
  --output-root DIR                  Default: results/real_runs
  --operator-note TEXT
  --help

Safety:
  Clear the arena, keep an operator beside the robot, keep Ctrl+C and physical
  stop ready, and keep a separate zero-Twist terminal ready for the resolved
  cmd_vel topic. The inner runner still uses the typed RUN gate. This wrapper
  records evidence only; it is not a safety gate.
EOF
}

fail_usage() {
  echo "error: $*" >&2
  usage >&2
  exit 2
}

RUN_ID="${1:-}"
[[ -n "$RUN_ID" && "$RUN_ID" != --* ]] || fail_usage "RUN_ID is required"
shift

LEG_INDEX=0
NAMESPACE=""
SCAN_TOPIC="scan"
ODOM_TOPIC="odom"
CMD_VEL_TOPIC="cmd_vel"
AMCL_TOPIC="amcl_pose"
MAP_FRAME="map"
ODOM_FRAME="odom"
BASE_FRAME="base_footprint"
LOCALIZATION_SOURCE="amcl"
MAX_AMCL_AGE_SEC="5.0"
MAX_SCAN_AGE_SEC="1.0"
MAX_ODOM_AGE_SEC="1.0"
MAX_TF_AGE_SEC="1.0"
PREFLIGHT_OBSERVATION_WINDOW_SEC="8.0"
INITIAL_SENSOR_WAIT_SEC="2.0"
ALLOW_IDLE_NAV2_PUBLISHERS=0
PROMPT_FOR_INITIALPOSE=1
ROUTE_CSV="results/aufgabe04/routes/first_detected_station_route.csv"
DIAGNOSTICS_JSON="results/aufgabe04/routes/first_detected_station_route_diagnostics.json"
RESULTS_CSV="results/aufgabe04/station_segment_runs.csv"
OUTPUT_ROOT="results/real_runs"
OPERATOR_NOTE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --leg-index) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; LEG_INDEX="$2"; shift 2 ;;
    --namespace) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; NAMESPACE="$2"; shift 2 ;;
    --scan-topic) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; SCAN_TOPIC="$2"; shift 2 ;;
    --odom-topic) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; ODOM_TOPIC="$2"; shift 2 ;;
    --cmd-vel-topic) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; CMD_VEL_TOPIC="$2"; shift 2 ;;
    --amcl-topic) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; AMCL_TOPIC="$2"; shift 2 ;;
    --map-frame) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; MAP_FRAME="$2"; shift 2 ;;
    --odom-frame) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; ODOM_FRAME="$2"; shift 2 ;;
    --base-frame) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; BASE_FRAME="$2"; shift 2 ;;
    --localization-source) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; LOCALIZATION_SOURCE="$2"; shift 2 ;;
    --max-amcl-age-sec) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; MAX_AMCL_AGE_SEC="$2"; shift 2 ;;
    --max-scan-age-sec) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; MAX_SCAN_AGE_SEC="$2"; shift 2 ;;
    --max-odom-age-sec) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; MAX_ODOM_AGE_SEC="$2"; shift 2 ;;
    --max-tf-age-sec) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; MAX_TF_AGE_SEC="$2"; shift 2 ;;
    --preflight-observation-window-sec) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; PREFLIGHT_OBSERVATION_WINDOW_SEC="$2"; shift 2 ;;
    --initial-sensor-wait-sec) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; INITIAL_SENSOR_WAIT_SEC="$2"; shift 2 ;;
    --allow-idle-nav2-publishers) ALLOW_IDLE_NAV2_PUBLISHERS=1; shift ;;
    --no-initialpose-prompt) PROMPT_FOR_INITIALPOSE=0; shift ;;
    --route-csv) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; ROUTE_CSV="$2"; shift 2 ;;
    --diagnostics-json) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; DIAGNOSTICS_JSON="$2"; shift 2 ;;
    --results-csv) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; RESULTS_CSV="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; OUTPUT_ROOT="$2"; shift 2 ;;
    --operator-note) [[ $# -ge 2 ]] || fail_usage "$1 requires a value"; OPERATOR_NOTE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) fail_usage "unknown option: $1" ;;
  esac
done

[[ "$LOCALIZATION_SOURCE" == "amcl" || "$LOCALIZATION_SOURCE" == "tf" ]] || {
  echo "error: --localization-source must be amcl or tf" >&2
  exit 2
}
[[ -f "$ROUTE_CSV" ]] || { echo "error: missing route CSV: $ROUTE_CSV" >&2; exit 2; }
[[ -f "$DIAGNOSTICS_JSON" ]] || { echo "error: missing diagnostics JSON: $DIAGNOSTICS_JSON" >&2; exit 2; }

BUNDLE_DIR="${OUTPUT_ROOT%/}/$RUN_ID"
PREFLIGHT_JSON="$BUNDLE_DIR/aufgabe04_preflight.json"
SEMANTIC_LOG="$BUNDLE_DIR/aufgabe04_events.jsonl"

echo "About to run a physical TurtleBot station segment."
echo "Safety requirements:"
echo "  - clear the arena and station approach zone"
echo "  - keep an operator beside the robot"
echo "  - keep Ctrl+C and physical stop ready"
echo "  - keep a separate zero-Twist terminal ready for the resolved cmd_vel topic"
echo "  - confirm no active Nav2 goal/controller is running"
echo "  - the inner runner will prompt you to press Enter, then click 2D Pose Estimate during preflight"
echo
echo "Run ID: $RUN_ID"
echo "Bundle dir: $BUNDLE_DIR"
echo "Route CSV: $ROUTE_CSV"
echo "Diagnostics JSON: $DIAGNOSTICS_JSON"
echo

INNER_CMD=(
  python3 scripts/aufgabe04/navigation/run_single_station_segment.py
  --leg-index "$LEG_INDEX"
  --route-csv "$ROUTE_CSV"
  --diagnostics-json "$DIAGNOSTICS_JSON"
  --results-csv "$RESULTS_CSV"
  --run-id "$RUN_ID"
  --namespace "$NAMESPACE"
  --scan-topic "$SCAN_TOPIC"
  --odom-topic "$ODOM_TOPIC"
  --cmd-vel-topic "$CMD_VEL_TOPIC"
  --amcl-topic "$AMCL_TOPIC"
  --map-frame "$MAP_FRAME"
  --odom-frame "$ODOM_FRAME"
  --base-frame "$BASE_FRAME"
  --localization-source "$LOCALIZATION_SOURCE"
  --max-amcl-age-sec "$MAX_AMCL_AGE_SEC"
  --max-scan-age-sec "$MAX_SCAN_AGE_SEC"
  --max-odom-age-sec "$MAX_ODOM_AGE_SEC"
  --max-tf-age-sec "$MAX_TF_AGE_SEC"
  --preflight-observation-window-sec "$PREFLIGHT_OBSERVATION_WINDOW_SEC"
  --initial-sensor-wait-sec "$INITIAL_SENSOR_WAIT_SEC"
  --preflight-json "$PREFLIGHT_JSON"
  --semantic-log "$SEMANTIC_LOG"
)

if [[ -n "$OPERATOR_NOTE" ]]; then
  INNER_CMD+=(--operator-note "$OPERATOR_NOTE")
fi

if [[ "$ALLOW_IDLE_NAV2_PUBLISHERS" -eq 1 ]]; then
  INNER_CMD+=(--allowed-cmd-vel-publisher /behavior_server)
  INNER_CMD+=(--allowed-cmd-vel-publisher /velocity_smoother)
fi

if [[ "$PROMPT_FOR_INITIALPOSE" -eq 1 && "$LOCALIZATION_SOURCE" == "amcl" ]]; then
  INNER_CMD+=(--prompt-for-initialpose)
fi

scripts/common/run_with_bundle.sh \
  --namespace "$NAMESPACE" \
  --cmd-vel-topic "$CMD_VEL_TOPIC" \
  --scan-topic "$SCAN_TOPIC" \
  --odom-topic "$ODOM_TOPIC" \
  --amcl-topic "$AMCL_TOPIC" \
  --map-frame "$MAP_FRAME" \
  --odom-frame "$ODOM_FRAME" \
  --base-frame "$BASE_FRAME" \
  --output-root "$OUTPUT_ROOT" \
  "$RUN_ID" \
  -- \
  "${INNER_CMD[@]}"
