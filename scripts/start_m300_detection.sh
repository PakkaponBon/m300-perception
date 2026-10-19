#!/usr/bin/env bash
# Start the M300 driver, live detector and sensor-centred RViz view together.
# Run only one instance; stop any separately started driver/detector first.
# No networking changes, recording, builds or automatic vehicle stop commands.

set -eo pipefail
set +m

M300_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
M300_TOOL_DIR="$M300_PROJECT_DIR/runtime"
M300_ROS_SETUP="${M300_ROS_SETUP:-/opt/ros/humble/setup.bash}"
M300_WS_SETUP="${M300_WS_SETUP:-$M300_PROJECT_DIR/ros2_ws/install/setup.bash}"
M300_PYTHON="${M300_PYTHON:-python3}"
M300_RVIZ_CONFIG="$M300_PROJECT_DIR/rviz/person_detection_topdown.rviz"
M300_DEADZONE_RADIUS_M="${M300_DEADZONE_RADIUS_M:-1.0}"
M300_WARNING_RADIUS_M="${M300_WARNING_RADIUS_M:-2.0}"

usage() {
    printf '%s\n' \
        'Usage: bash start_m300_detection.sh [--deadzone METRES] [--warning METRES]' \
        'Example: bash start_m300_detection.sh --deadzone 1.0 --warning 2.0' \
        'Zones are display-only, not validated safety distances.' \
        'Ctrl+C or closing RViz stops the processes started by this launcher.'
}

while (($#)); do
    case "$1" in
        --deadzone|--warning)
            if (($# < 2)); then
                printf 'Missing distance after %s\n' "$1" >&2
                exit 2
            fi
            if [[ "$1" == --deadzone ]]; then
                M300_DEADZONE_RADIUS_M="$2"
            else
                M300_WARNING_RADIUS_M="$2"
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for M300_REQUIRED_FILE in "$M300_ROS_SETUP" "$M300_WS_SETUP" \
    "$M300_TOOL_DIR/live_warehouse_detector_ros2.py" "$M300_RVIZ_CONFIG"; do
    if [[ ! -r "$M300_REQUIRED_FILE" ]]; then
        printf 'Required file not readable: %s\n' "$M300_REQUIRED_FILE" >&2
        exit 1
    fi
done
if ! command -v "$M300_PYTHON" >/dev/null 2>&1; then
    printf 'OpenPCDet Python not executable: %s\n' "$M300_PYTHON" >&2
    exit 1
fi
if [[ -z "${M300_CHECKPOINT:-}" || ! -r "$M300_CHECKPOINT" ]]; then
    printf 'Set M300_CHECKPOINT to a readable, trusted checkpoint before launch.\n' >&2
    exit 1
fi
M300_CONFIG_ROOT="${M300_OPENPCDET_ROOT:-$M300_PROJECT_DIR/../OpenPCDet}"
if [[ ! -r "$M300_CONFIG_ROOT/tools/cfgs/custom_models/pointpillar_m300_decay03.yaml" ]]; then
    printf 'Install the OpenPCDet overlay first; see docs/SETUP.md.\n' >&2
    exit 1
fi

# Validate before starting any hardware-facing process.
M300_ZONE_VALUES="$("$M300_PYTHON" - "$M300_DEADZONE_RADIUS_M" "$M300_WARNING_RADIUS_M" <<'PY'
import math
import sys
try:
    red, yellow = map(float, sys.argv[1:])
except ValueError:
    sys.exit('Zone distances must be numbers.')
if not (math.isfinite(red) and math.isfinite(yellow) and 0 < red < yellow):
    sys.exit('Zone distances must satisfy 0 < deadzone < warning, in metres.')
# Force floating-point ROS parameter values, even for --deadzone 1 --warning 2.
print(red, yellow)
PY
)"
read -r M300_DEADZONE_RADIUS_M M300_WARNING_RADIUS_M <<< "$M300_ZONE_VALUES"

# ROS setup scripts can reference unset variables, so enable nounset afterwards.
source "$M300_ROS_SETUP"
source "$M300_WS_SETUP"
set -u

for M300_REQUIRED_COMMAND in ros2 rviz2 setsid env; do
    if ! command -v "$M300_REQUIRED_COMMAND" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$M300_REQUIRED_COMMAND" >&2
        exit 1
    fi
done
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    printf 'No GUI display available for RViz.\n' >&2
    exit 1
fi
# Keep native RViz out of inherited Snap GTK/Qt/library directories.
M300_RVIZ_LIBRARY_PATH=""
IFS=':' read -r -a M300_LIBRARY_DIRS <<< "${LD_LIBRARY_PATH:-}"
for M300_LIBRARY_DIR in "${M300_LIBRARY_DIRS[@]}"; do
    case "$M300_LIBRARY_DIR" in
        ''|/snap/*|/var/lib/snapd/snap/*|/home/*/snap/*) continue ;;
    esac
    M300_RVIZ_LIBRARY_PATH+="${M300_RVIZ_LIBRARY_PATH:+:}$M300_LIBRARY_DIR"
done
M300_DRIVER_SHARE="$(ros2 pkg prefix --share pacecat_m300_driver)"
if [[ ! -r "$M300_DRIVER_SHARE/launch/LDS-M300-E.launch" ]]; then
    printf 'Installed M300 launch file not found in: %s\n' "$M300_DRIVER_SHARE" >&2
    exit 1
fi

M300_CHILD_PIDS=()
cleanup() {
    trap '' INT TERM
    if ((${#M300_CHILD_PIDS[@]} == 0)); then
        return
    fi
    printf '\nStopping this launcher\047s LiDAR, detector and RViz processes...\n'
    # Each child has its own session/process group. Never kill unrelated nodes.
    for M300_PID in "${M300_CHILD_PIDS[@]}"; do
        kill -INT -- "-$M300_PID" 2>/dev/null || true
    done
    # Allow ROS processes a short graceful shutdown, then clean up any survivors.
    for M300_ATTEMPT in {1..25}; do
        M300_STILL_RUNNING=false
        for M300_PID in "${M300_CHILD_PIDS[@]}"; do
            if kill -0 -- "-$M300_PID" 2>/dev/null; then
                M300_STILL_RUNNING=true
                break
            fi
        done
        if [[ "$M300_STILL_RUNNING" == false ]]; then
            break
        fi
        sleep 0.2
    done
    for M300_PID in "${M300_CHILD_PIDS[@]}"; do
        kill -TERM -- "-$M300_PID" 2>/dev/null || true
    done
    sleep 0.5
    for M300_PID in "${M300_CHILD_PIDS[@]}"; do
        kill -KILL -- "-$M300_PID" 2>/dev/null || true
        wait "$M300_PID" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd -- "$M300_TOOL_DIR"
printf 'Starting M300 LiDAR driver...\n'
setsid ros2 launch pacecat_m300_driver LDS-M300-E.launch &
M300_CHILD_PIDS+=("$!")

printf 'Starting live detection (red < %s m; yellow < %s m; otherwise green)...\n' \
    "$M300_DEADZONE_RADIUS_M" "$M300_WARNING_RADIUS_M"
setsid "$M300_PYTHON" "$M300_TOOL_DIR/live_warehouse_detector_ros2.py" \
    --ros-args \
    -p "deadzone_radius_m:=$M300_DEADZONE_RADIUS_M" \
    -p "warning_radius_m:=$M300_WARNING_RADIUS_M" &
M300_CHILD_PIDS+=("$!")

printf 'Opening top-down RViz...\n'
setsid env -u LD_PRELOAD -u LD_AUDIT \
    -u GTK_PATH -u GTK_EXE_PREFIX -u GTK_DATA_PREFIX -u GTK_MODULES \
    -u GIO_MODULE_DIR -u GIO_EXTRA_MODULES -u GSETTINGS_SCHEMA_DIR \
    -u QT_PLUGIN_PATH -u QT_QPA_PLATFORM_PLUGIN_PATH \
    -u QT_QPA_PLATFORMTHEME -u QT_STYLE_OVERRIDE \
    -u LIBGL_DRIVERS_PATH -u LIBVA_DRIVERS_PATH \
    "LD_LIBRARY_PATH=$M300_RVIZ_LIBRARY_PATH" \
    "$(command -v rviz2)" -d "$M300_RVIZ_CONFIG" &
M300_CHILD_PIDS+=("$!")

printf 'Ctrl+C stops all three. If any process exits, the others are stopped too.\n'
if wait -n "${M300_CHILD_PIDS[@]}"; then
    M300_EXIT_CODE=0
else
    M300_EXIT_CODE=$?
fi
printf 'A launched process exited (status %s); shutting down the rest.\n' "$M300_EXIT_CODE"
exit "$M300_EXIT_CODE"
