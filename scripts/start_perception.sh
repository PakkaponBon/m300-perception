#!/usr/bin/env bash
# Driver-independent launch: start ONLY a perception node and RViz.
# Start the chosen sensor's ROS 2 PointCloud2 driver separately.
set -eo pipefail
set +m
PERCEPTION_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    printf '%s\n' \
        'Usage: ./scripts/start_perception.sh {ai|obstacle} [node ROS arguments]' \
        'The LiDAR driver must already be running. This script never starts one.' \
        'Set PERCEPTION_INPUT_TOPIC and PERCEPTION_FRAME_ID for your driver.' \
        'AI also needs PERCEPTION_OPENPCDET_ROOT, PERCEPTION_PYTHON and PERCEPTION_CHECKPOINT.' \
        'Optional AI: PERCEPTION_MODEL_CONFIG, PERCEPTION_INTENSITY_FIELD,' \
        'PERCEPTION_DECAY_TIME_SEC, PERCEPTION_SCORE_THRESHOLD.' \
        'Example: PERCEPTION_INPUT_TOPIC=/points PERCEPTION_FRAME_ID=lidar ./scripts/start_perception.sh obstacle' \
        'Topic/frame overrides should use these env vars so RViz stays in sync.' \
        'Other startup settings can be passed with --ros-args -p name:=value.'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
PERCEPTION_MODE="${1:-}"
case "$PERCEPTION_MODE" in ai|obstacle) shift ;; *) usage >&2; exit 2 ;; esac
PERCEPTION_ROS_SETUP="${PERCEPTION_ROS_SETUP:-/opt/ros/humble/setup.bash}"
PERCEPTION_INPUT_TOPIC="${PERCEPTION_INPUT_TOPIC:-/pointcloud}"
PERCEPTION_FRAME_ID="${PERCEPTION_FRAME_ID:-m300_lidar}"
PERCEPTION_PYTHON="${PERCEPTION_PYTHON:-${M300_PYTHON:-/usr/bin/python3}}"
if ! command -v "$PERCEPTION_PYTHON" >/dev/null 2>&1; then
    printf 'Python not executable: %s\n' "$PERCEPTION_PYTHON" >&2; exit 1
fi
if [[ ! -r "$PERCEPTION_ROS_SETUP" ]]; then
    printf 'ROS setup not readable: %s\n' "$PERCEPTION_ROS_SETUP" >&2; exit 1
fi
PERCEPTION_NODE_ARGS=(--ros-args -p "input_topic:=$PERCEPTION_INPUT_TOPIC"
                     -p "expected_frame_id:=$PERCEPTION_FRAME_ID")
case "$PERCEPTION_MODE" in
    ai)
        export PERCEPTION_OPENPCDET_ROOT="${PERCEPTION_OPENPCDET_ROOT:-${M300_OPENPCDET_ROOT:-$PERCEPTION_PROJECT_DIR/../OpenPCDet}}"
        export PERCEPTION_CHECKPOINT="${PERCEPTION_CHECKPOINT:-${M300_CHECKPOINT:-}}"
        export PERCEPTION_MODEL_CONFIG="${PERCEPTION_MODEL_CONFIG:-$PERCEPTION_OPENPCDET_ROOT/tools/cfgs/custom_models/pointpillar_m300_decay03.yaml}"
        for PERCEPTION_REQUIRED in "$PERCEPTION_CHECKPOINT" "$PERCEPTION_MODEL_CONFIG"; do
            if [[ ! -r "$PERCEPTION_REQUIRED" ]]; then
                printf 'AI checkpoint/config not readable: %s\n' "$PERCEPTION_REQUIRED" >&2; exit 1
            fi
        done
        PERCEPTION_SCRIPT="$PERCEPTION_PROJECT_DIR/runtime/live_warehouse_detector_ros2.py"
        PERCEPTION_RVIZ="$PERCEPTION_PROJECT_DIR/rviz/person_detection_topdown.rviz"
        PERCEPTION_INTENSITY_FIELD="${PERCEPTION_INTENSITY_FIELD:-intensity}"
        PERCEPTION_DECAY_TIME_SEC="${PERCEPTION_DECAY_TIME_SEC:-0.3}"
        PERCEPTION_SCORE_THRESHOLD="${PERCEPTION_SCORE_THRESHOLD:-0.3}"
        PERCEPTION_NUMBERS="$("$PERCEPTION_PYTHON" - "$PERCEPTION_PROJECT_DIR" \
            "$PERCEPTION_INPUT_TOPIC" "$PERCEPTION_FRAME_ID" "$PERCEPTION_INTENSITY_FIELD" \
            "$PERCEPTION_DECAY_TIME_SEC" "$PERCEPTION_SCORE_THRESHOLD" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from runtime.lidar_input import InputSettings
settings = InputSettings(sys.argv[2], sys.argv[3], sys.argv[4],
                         float(sys.argv[5]), float(sys.argv[6]))
print(float(settings.decay_time_sec), float(settings.score_threshold))
PY
)"
        read -r PERCEPTION_DECAY_TIME_SEC PERCEPTION_SCORE_THRESHOLD <<< "$PERCEPTION_NUMBERS"
        PERCEPTION_NODE_ARGS+=(-p "intensity_field:=$PERCEPTION_INTENSITY_FIELD"
                              -p "decay_time_sec:=$PERCEPTION_DECAY_TIME_SEC"
                              -p "score_threshold:=$PERCEPTION_SCORE_THRESHOLD")
        ;;
    obstacle)
        PERCEPTION_SCRIPT="$PERCEPTION_PROJECT_DIR/runtime/m300_front_obstacle_publisher.py"
        PERCEPTION_RVIZ="$PERCEPTION_PROJECT_DIR/rviz/front_obstacle_topdown.rviz"
        ;;
esac
source "$PERCEPTION_ROS_SETUP"
if [[ -n "${PERCEPTION_WS_SETUP:-}" ]]; then source "$PERCEPTION_WS_SETUP"; fi
set -u
for PERCEPTION_COMMAND in rviz2 setsid env; do
    command -v "$PERCEPTION_COMMAND" >/dev/null 2>&1 || {
        printf 'Command not found: %s\n' "$PERCEPTION_COMMAND" >&2; exit 1;
    }
done
PERCEPTION_RVIZ_EXECUTABLE="$(command -v rviz2)"
# VS Code's Snap terminal can export GTK/GIO paths into a native ROS app.
# Qt may then dlopen Snap's GTK modules, mixing core20 libpthread with the
# host libc (GLIBC_PRIVATE / __libc_pthread_init). Clean only RViz's child
# environment; preserve ROS libraries, display/auth and the publisher's env.
PERCEPTION_RVIZ_LIBRARY_PATH=""
IFS=':' read -r -a PERCEPTION_LIBRARY_DIRS <<< "${LD_LIBRARY_PATH:-}"
for PERCEPTION_LIBRARY_DIR in "${PERCEPTION_LIBRARY_DIRS[@]}"; do
    case "$PERCEPTION_LIBRARY_DIR" in
        ''|/snap/*|/var/lib/snapd/snap/*|/home/*/snap/*) continue ;;
    esac
    PERCEPTION_RVIZ_LIBRARY_PATH+="${PERCEPTION_RVIZ_LIBRARY_PATH:+:}$PERCEPTION_LIBRARY_DIR"
done
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    printf 'No GUI display available. Run from a desktop terminal for RViz.\n' >&2
    exit 1
fi

PERCEPTION_CHILD_PIDS=()
cleanup() {
    trap '' INT TERM
    if ((${#PERCEPTION_CHILD_PIDS[@]} == 0)); then
        return
    fi
    printf '\nStopping this launcher\047s perception node and RViz...\n'
    # setsid isolates each child; never signal the existing driver or AI node.
    for PERCEPTION_PID in "${PERCEPTION_CHILD_PIDS[@]}"; do
        kill -INT -- "-$PERCEPTION_PID" 2>/dev/null || true
    done
    for PERCEPTION_ATTEMPT in {1..25}; do
        PERCEPTION_STILL_RUNNING=false
        for PERCEPTION_PID in "${PERCEPTION_CHILD_PIDS[@]}"; do
            if kill -0 -- "-$PERCEPTION_PID" 2>/dev/null; then
                PERCEPTION_STILL_RUNNING=true
                break
            fi
        done
        if [[ "$PERCEPTION_STILL_RUNNING" == false ]]; then
            break
        fi
        sleep 0.2
    done
    for PERCEPTION_PID in "${PERCEPTION_CHILD_PIDS[@]}"; do
        kill -TERM -- "-$PERCEPTION_PID" 2>/dev/null || true
    done
    sleep 0.5
    for PERCEPTION_PID in "${PERCEPTION_CHILD_PIDS[@]}"; do
        kill -KILL -- "-$PERCEPTION_PID" 2>/dev/null || true
        wait "$PERCEPTION_PID" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Starting %s perception on %s in frame %s; driver is external.\n' \
    "$PERCEPTION_MODE" "$PERCEPTION_INPUT_TOPIC" "$PERCEPTION_FRAME_ID"
setsid "$PERCEPTION_PYTHON" -u "$PERCEPTION_SCRIPT" "${PERCEPTION_NODE_ARGS[@]}" "$@" &
PERCEPTION_CHILD_PIDS+=("$!")
setsid env \
    -u LD_PRELOAD -u LD_AUDIT \
    -u GTK_PATH -u GTK_EXE_PREFIX -u GTK_DATA_PREFIX -u GTK_MODULES \
    -u GIO_MODULE_DIR -u GIO_EXTRA_MODULES -u GSETTINGS_SCHEMA_DIR \
    -u QT_PLUGIN_PATH -u QT_QPA_PLATFORM_PLUGIN_PATH \
    -u QT_QPA_PLATFORMTHEME -u QT_STYLE_OVERRIDE \
    -u LIBGL_DRIVERS_PATH -u LIBVA_DRIVERS_PATH \
    "LD_LIBRARY_PATH=$PERCEPTION_RVIZ_LIBRARY_PATH" \
    "$PERCEPTION_RVIZ_EXECUTABLE" -d "$PERCEPTION_RVIZ" -f "$PERCEPTION_FRAME_ID" \
    --ros-args -r "/pointcloud:=$PERCEPTION_INPUT_TOPIC" &
PERCEPTION_CHILD_PIDS+=("$!")
printf 'Ctrl+C or closing RViz stops these two processes, not the sensor driver.\n'
if wait -n "${PERCEPTION_CHILD_PIDS[@]}"; then PERCEPTION_EXIT_CODE=0; else PERCEPTION_EXIT_CODE=$?; fi
exit "$PERCEPTION_EXIT_CODE"
