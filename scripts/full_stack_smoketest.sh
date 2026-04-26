#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_PATH=""
ANOMALY_AGENT_PATH="${ANOMALY_AGENT_PATH:-$ROOT_DIR/../AnomalyAgent}"
ANOMALY_AGENT_CONFIG_PATH="${ANOMALY_AGENT_CONFIG_PATH:-$ANOMALY_AGENT_PATH/config/api_config.yaml}"
SEMANTIC_SIMILARITY_MODEL_PATH="${SEMANTIC_SIMILARITY_MODEL_PATH:-}"

CLOUD_HOST="127.0.0.1"
CLOUD_PORT="18001"
RELAY_EDGE_PORT="19000"
RELAY_GRADIO_PORT="17860"
JIGSAW_PORT="18000"

LOG_DIR="${TMPDIR:-/tmp}/anomaly_detection_smoketest"
mkdir -p "$LOG_DIR"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") --image /absolute/path/to/test.jpg [options]

What this script does:
  1. Starts cloud service on the current server
  2. Starts relay service on the current server
  3. Starts edge Jigsaw service on the current server
  4. Runs three checks with the provided image:
     - direct Jigsaw health + detect warm-up
     - direct cloud /api/v1/detect
     - simulated edge -> relay -> cloud deep-analysis path

Important:
  - This validates all three software modules on one server.
  - It does NOT require a RealSense camera.
  - Full run_edge.py is not started because that path depends on RealSense hardware.

Options:
  --image PATH                  Test image path (required)
  --anomaly-agent-path PATH     Path to AnomalyAgent project
  --agent-config-path PATH      Path to AnomalyAgent API yaml config
  --cloud-port PORT             Cloud API port (default: ${CLOUD_PORT})
  --relay-edge-port PORT        Relay receiver port (default: ${RELAY_EDGE_PORT})
  --relay-gradio-port PORT      Relay gradio port (default: ${RELAY_GRADIO_PORT})
  --jigsaw-port PORT            Edge Jigsaw port (default: ${JIGSAW_PORT})
  --log-dir PATH                Log directory (default: ${LOG_DIR})
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE_PATH="$2"
      shift 2
      ;;
    --anomaly-agent-path)
      ANOMALY_AGENT_PATH="$2"
      shift 2
      ;;
    --agent-config-path)
      ANOMALY_AGENT_CONFIG_PATH="$2"
      shift 2
      ;;
    --cloud-port)
      CLOUD_PORT="$2"
      shift 2
      ;;
    --relay-edge-port)
      RELAY_EDGE_PORT="$2"
      shift 2
      ;;
    --relay-gradio-port)
      RELAY_GRADIO_PORT="$2"
      shift 2
      ;;
    --jigsaw-port)
      JIGSAW_PORT="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      mkdir -p "$LOG_DIR"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$IMAGE_PATH" ]]; then
  echo "ERROR: --image is required" >&2
  usage
  exit 1
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "ERROR: image not found: $IMAGE_PATH" >&2
  exit 1
fi

if [[ ! -d "$ANOMALY_AGENT_PATH" ]]; then
  echo "ERROR: AnomalyAgent path not found: $ANOMALY_AGENT_PATH" >&2
  exit 1
fi

cleanup() {
  set +e
  for pid_var in JIGSAW_PID RELAY_PID CLOUD_PID; do
    pid="${!pid_var:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

ensure_port_free() {
  local port="$1"
  local name="$2"
  if ! python3 - <<PY
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int("$port")))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
  then
    echo "[error] ${name} port ${port} is already in use on 127.0.0.1." >&2
    echo "[hint] this usually means an old ${name} process is still running, so the smoketest is talking to a stale service instance." >&2
    echo "[hint] stop the old process or rerun with a different port." >&2
    exit 1
  fi
}

wait_for_http() {
  local url="$1"
  local name="$2"
  local timeout_sec="${3:-120}"
  local start_ts
  start_ts="$(date +%s)"

  echo "[wait] ${name}: ${url}"
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[ok] ${name} is ready"
      return 0
    fi
    if (( $(date +%s) - start_ts >= timeout_sec )); then
      echo "[error] timeout waiting for ${name}" >&2
      return 1
    fi
    sleep 2
  done
}

json_value() {
  local python_expr="$1"
  python3 -c "import json,sys; data=json.load(sys.stdin); print(${python_expr})"
}

IMAGE_B64="$(python3 - <<PY
import base64
from pathlib import Path
print(base64.b64encode(Path(r'''$IMAGE_PATH''').read_bytes()).decode('utf-8'))
PY
)"

CLOUD_HEALTH_URL="http://${CLOUD_HOST}:${CLOUD_PORT}/api/v1/health"
CLOUD_DETECT_URL="http://${CLOUD_HOST}:${CLOUD_PORT}/api/v1/detect"
RELAY_STATUS_URL="http://127.0.0.1:${RELAY_EDGE_PORT}/api/edge/status"
RELAY_FRAME_URL="http://127.0.0.1:${RELAY_EDGE_PORT}/api/edge/frame"
RELAY_RESET_URL="http://127.0.0.1:${RELAY_EDGE_PORT}/api/edge/reset"
JIGSAW_HEALTH_URL="http://127.0.0.1:${JIGSAW_PORT}/health"
JIGSAW_DETECT_URL="http://127.0.0.1:${JIGSAW_PORT}/detect"

echo "[info] logs: $LOG_DIR"
echo "[info] image: $IMAGE_PATH"
echo "[info] anomaly agent path: $ANOMALY_AGENT_PATH"
echo "[info] anomaly agent config: $ANOMALY_AGENT_CONFIG_PATH"

rm -f "$LOG_DIR/cloud.log" "$LOG_DIR/relay.log" "$LOG_DIR/jigsaw.log"

ensure_port_free "$CLOUD_PORT" "cloud"
ensure_port_free "$RELAY_EDGE_PORT" "relay"
ensure_port_free "$JIGSAW_PORT" "jigsaw"

echo "[start] cloud service"
(
  cd "$ROOT_DIR/cloud"
  API_HOST="$CLOUD_HOST" \
  API_PORT="$CLOUD_PORT" \
  ANOMALY_AGENT_PROJECT_PATH="$ANOMALY_AGENT_PATH" \
  ANOMALY_AGENT_CONFIG_PATH="$ANOMALY_AGENT_CONFIG_PATH" \
  SEMANTIC_SIMILARITY_MODEL_PATH="$SEMANTIC_SIMILARITY_MODEL_PATH" \
  python3 start_cloud.py --host "$CLOUD_HOST" --port "$CLOUD_PORT"
) >"$LOG_DIR/cloud.log" 2>&1 &
CLOUD_PID=$!

wait_for_http "$CLOUD_HEALTH_URL" "cloud health"
CLOUD_HEALTH_JSON="$(curl -fsS "$CLOUD_HEALTH_URL")"
echo "[health] cloud => $CLOUD_HEALTH_JSON"

CLOUD_AGENT_AVAILABLE="$(printf '%s' "$CLOUD_HEALTH_JSON" | json_value "str(data.get('agent_available', False)).lower()")"
CLOUD_AGENT_LOADED="$(printf '%s' "$CLOUD_HEALTH_JSON" | json_value "str(data.get('agent_loaded', False)).lower()")"
if [[ "$CLOUD_AGENT_AVAILABLE" != "true" ]]; then
  CLOUD_IMPORT_ERROR="$(printf '%s' "$CLOUD_HEALTH_JSON" | python3 -c 'import json,sys; data=json.load(sys.stdin); print(data.get("import_error", "<missing import_error>"))')"
  echo "[error] cloud service is up, but agent import failed (agent_available=false)." >&2
  echo "[error] import_error: ${CLOUD_IMPORT_ERROR}" >&2
  echo "[hint] check whether ANOMALY_AGENT_PROJECT_PATH / ANOMALY_AGENT_CONFIG_PATH are correct and whether cloud dependencies can import main_memory_vad + api.handlers." >&2
  echo "[hint] tail of cloud log:" >&2
  tail -n 80 "$LOG_DIR/cloud.log" >&2 || true
  exit 1
fi

if [[ "$CLOUD_AGENT_LOADED" != "true" ]]; then
  echo "[warn] cloud agent module is available, but the runtime agent is not loaded yet; direct detect may still fail if startup initialization is incomplete." >&2
fi

echo "[start] relay service"
(
  cd "$ROOT_DIR/relay"
  CLOUD_HOST="$CLOUD_HOST" \
  CLOUD_PORT="$CLOUD_PORT" \
  EDGE_RECEIVER_PORT="$RELAY_EDGE_PORT" \
  python3 start_relay.py --cloud-host "$CLOUD_HOST" --cloud-port "$CLOUD_PORT" --edge-port "$RELAY_EDGE_PORT" --gradio-port "$RELAY_GRADIO_PORT"
) >"$LOG_DIR/relay.log" 2>&1 &
RELAY_PID=$!

wait_for_http "$RELAY_STATUS_URL" "relay receiver status"
echo "[health] relay => $(curl -fsS "$RELAY_STATUS_URL")"

echo "[start] edge jigsaw service"
(
  cd "$ROOT_DIR/edge"
  python3 start_edge.py --start-jigsaw --jigsaw-port "$JIGSAW_PORT"
) >"$LOG_DIR/jigsaw.log" 2>&1 &
JIGSAW_PID=$!

wait_for_http "$JIGSAW_HEALTH_URL" "jigsaw health"
JIGSAW_HEALTH_JSON="$(curl -fsS "$JIGSAW_HEALTH_URL")"
echo "[health] jigsaw => $JIGSAW_HEALTH_JSON"

MODEL_LOADED="$(printf '%s' "$JIGSAW_HEALTH_JSON" | json_value "str(data.get('model_loaded', False)).lower()")"
if [[ "$MODEL_LOADED" != "true" ]]; then
  echo "[error] Jigsaw model is not loaded; see $LOG_DIR/jigsaw.log" >&2
  exit 1
fi

echo "[test-1] warm up jigsaw with repeated single-image detection"
for i in $(seq 1 7); do
  JIGSAW_RESULT="$(curl -fsS -X POST "$JIGSAW_DETECT_URL" -H 'Content-Type: application/json' -d "{\"image\":\"$IMAGE_B64\"}")"
done
echo "[result] jigsaw final detect => $JIGSAW_RESULT"

echo "[test-2] direct cloud detect with repeated frames"
CLOUD_PAYLOAD_FILE="$LOG_DIR/cloud_detect_payload.json"
python3 - <<PY > "$CLOUD_PAYLOAD_FILE"
import json
img = r'''$IMAGE_B64'''
payload = {
    "video_id": "server_smoketest_cloud",
    "scene_type": "general",
    "dataset": "ped2",
    "frames": [{"image_base64": img, "frame_id": i, "timestamp": float(i)} for i in range(5)],
}
print(json.dumps(payload))
PY

CLOUD_RESULT_FILE="$LOG_DIR/cloud_detect_result.json"
curl -fsS -X POST "$CLOUD_DETECT_URL" \
  -H 'Content-Type: application/json' \
  --data @"$CLOUD_PAYLOAD_FILE" > "$CLOUD_RESULT_FILE"
echo "[result] cloud detect => $(cat "$CLOUD_RESULT_FILE")"

echo "[test-3] relay -> cloud deep analysis via simulated edge frames"
curl -fsS -X POST "$RELAY_RESET_URL" >/dev/null
for i in $(seq 1 5); do
  python3 - <<PY > "$LOG_DIR/relay_frame_${i}.json"
import json, time
payload = {
    "frame_id": $i,
    "timestamp": time.time(),
    "color_base64": r'''$IMAGE_B64''',
    "depth_base64": None,
    "jigsaw_score": 0.1,
    "spatial_score": 0.1,
    "temporal_score": 0.1,
    "is_anomalous": True,
}
print(json.dumps(payload))
PY
  curl -fsS -X POST "$RELAY_FRAME_URL" \
    -H 'Content-Type: application/json' \
    --data @"$LOG_DIR/relay_frame_${i}.json" >/dev/null
  sleep 0.3
done

echo "[wait] allow relay aggregator to call cloud"
sleep 5
RELAY_STATUS_JSON="$(curl -fsS "$RELAY_STATUS_URL")"
echo "[result] relay status => $RELAY_STATUS_JSON"

DEEP_ANALYSES="$(printf '%s' "$RELAY_STATUS_JSON" | json_value "data.get('aggregator', {}).get('stats', {}).get('deep_analyses_triggered', 0)")"
if [[ "$DEEP_ANALYSES" == "0" ]]; then
  echo "[error] relay did not trigger cloud deep analysis; see $LOG_DIR/relay.log" >&2
  exit 1
fi

echo
echo "========== SMOKETEST SUMMARY =========="
echo "cloud health : $CLOUD_HEALTH_URL"
echo "relay status : $RELAY_STATUS_URL"
echo "jigsaw health: $JIGSAW_HEALTH_URL"
echo "logs         : $LOG_DIR"
echo "cloud result : $CLOUD_RESULT_FILE"
echo "deep analyses triggered: $DEEP_ANALYSES"
echo "SUCCESS: edge(jigsaw) + relay + cloud software path validated on one server."
