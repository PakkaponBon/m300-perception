#!/usr/bin/env bash
# Start ONLY the geometric front-obstacle publisher and its RViz view.
# Keep the LiDAR driver running separately. No AI, networking changes or builds.
# Run one instance; stop any separately started obstacle publisher first.

set -eo pipefail
set +m

M300_OBSTACLE_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
M300_OBSTACLE_TOOL_DIR="$M300_OBSTACLE_PROJECT_DIR/runtime"
M300_OBSTACLE_ROS_SETUP="${M300_OBSTACLE_ROS_SETUP:-/opt/ros/humble/setup.bash}"
M300_OBSTACLE_PYTHON="${M300_OBSTACLE_PYTHON:-/usr/bin/python3}"
M300_OBSTACLE_SCRIPT="$M300_OBSTACLE_TOOL_DIR/m300_front_obstacle_publisher.py"
M300_OBSTACLE_RVIZ="$M300_OBSTACLE_PROJECT_DIR/rviz/front_obstacle_topdown.rviz"

if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
    printf '%s\n' \
        'Usage: ./start_m300_obstacle.sh [publisher ROS arguments]' \
        'Starts the obstacle publisher and RViz ONLY; start LiDAR separately.' \
        'Zone defaults come from m300_front_obstacle_publisher.py.' \
        'Example: ./start_m300_obstacle.sh --ros-args -p distance_m:=4.0' \
        'Ctrl+C or closing RViz stops only the two processes started here.'
    exit 0
fi

for M300_OBSTACLE_FILE in "$M300_OBSTACLE_ROS_SETUP" "$M300_OBSTACLE_SCRIPT" "$M300_OBSTACLE_RVIZ"; do
    if [[ ! -r "$M300_OBSTACLE_FILE" ]]; then
        printf 'Required file not readable: %s\n' "$M300_OBSTACLE_FILE" >&2
        exit 1
    fi
done
if [[ ! -x "$M300_OBSTACLE_PYTHON" ]]; then
    printf 'Python not executable: %s\n' "$M300_OBSTACLE_PYTHON" >&2
    exit 1
fi

# Source ROS before nounset: setup scripts may reference unset variables.
source "$M300_OBSTACLE_ROS_SETUP"
set -u
for M300_OBSTACLE_COMMAND in rviz2 setsid env; do
    if ! command -v "$M300_OBSTACLE_COMMAND" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$M300_OBSTACLE_COMMAND" >&2
        exit 1
    fi
done
M300_OBSTACLE_RVIZ_EXECUTABLE="$(command -v rviz2)"
# VS Code's Snap terminal can export GTK/GIO paths into a native ROS app.
# Qt may then dlopen Snap's GTK modules, mixing core20 libpthread with the
# host libc (GLIBC_PRIVATE / __libc_pthread_init). Clean only RViz's child
# environment; preserve ROS libraries, display/auth and the publisher's env.
M300_OBSTACLE_RVIZ_LIBRARY_PATH=""
IFS=':' read -r -a M300_OBSTACLE_LIBRARY_DIRS <<< "${LD_LIBRARY_PATH:-}"
for M300_OBSTACLE_LIBRARY_DIR in "${M300_OBSTACLE_LIBRARY_DIRS[@]}"; do
    case "$M300_OBSTACLE_LIBRARY_DIR" in
        ''|/snap/*|/var/lib/snapd/snap/*|/home/*/snap/*) continue ;;
    esac
    M300_OBSTACLE_RVIZ_LIBRARY_PATH+="${M300_OBSTACLE_RVIZ_LIBRARY_PATH:+:}$M300_OBSTACLE_LIBRARY_DIR"
done
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    printf 'No GUI display available. Run from a desktop terminal for RViz.\n' >&2
    exit 1
fi

M300_OBSTACLE_CHILD_PIDS=()
cleanup() {
    trap '' INT TERM
    if ((${#M300_OBSTACLE_CHILD_PIDS[@]} == 0)); then
        return
    fi
    printf '\nStopping this launcher\047s obstacle publisher and RViz...\n'
    # setsid isolates each child; never signal the existing driver or AI node.
    for M300_OBSTACLE_PID in "${M300_OBSTACLE_CHILD_PIDS[@]}"; do
        kill -INT -- "-$M300_OBSTACLE_PID" 2>/dev/null || true
    done
    for M300_OBSTACLE_ATTEMPT in {1..25}; do
        M300_OBSTACLE_STILL_RUNNING=false
        for M300_OBSTACLE_PID in "${M300_OBSTACLE_CHILD_PIDS[@]}"; do
            if kill -0 -- "-$M300_OBSTACLE_PID" 2>/dev/null; then
                M300_OBSTACLE_STILL_RUNNING=true
                break
            fi
        done
        if [[ "$M300_OBSTACLE_STILL_RUNNING" == false ]]; then
            break
        fi
        sleep 0.2
    done
    for M300_OBSTACLE_PID in "${M300_OBSTACLE_CHILD_PIDS[@]}"; do
        kill -TERM -- "-$M300_OBSTACLE_PID" 2>/dev/null || true
    done
    sleep 0.5
    for M300_OBSTACLE_PID in "${M300_OBSTACLE_CHILD_PIDS[@]}"; do
        kill -KILL -- "-$M300_OBSTACLE_PID" 2>/dev/null || true
        wait "$M300_OBSTACLE_PID" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd -- "$M300_OBSTACLE_TOOL_DIR"
printf 'Starting front-obstacle publisher; LiDAR must already be running...\n'
setsid "$M300_OBSTACLE_PYTHON" -u "$M300_OBSTACLE_SCRIPT" "$@" &
M300_OBSTACLE_CHILD_PIDS+=("$!")

printf 'Opening front-obstacle RViz (isolated from Snap GUI libraries)...\n'
setsid env \
    -u LD_PRELOAD -u LD_AUDIT \
    -u GTK_PATH -u GTK_EXE_PREFIX -u GTK_DATA_PREFIX -u GTK_MODULES \
    -u GIO_MODULE_DIR -u GIO_EXTRA_MODULES -u GSETTINGS_SCHEMA_DIR \
    -u QT_PLUGIN_PATH -u QT_QPA_PLATFORM_PLUGIN_PATH \
    -u QT_QPA_PLATFORMTHEME -u QT_STYLE_OVERRIDE \
    -u LIBGL_DRIVERS_PATH -u LIBVA_DRIVERS_PATH \
    "LD_LIBRARY_PATH=$M300_OBSTACLE_RVIZ_LIBRARY_PATH" \
    "$M300_OBSTACLE_RVIZ_EXECUTABLE" -d "$M300_OBSTACLE_RVIZ" &
M300_OBSTACLE_CHILD_PIDS+=("$!")

printf 'Ctrl+C or closing RViz stops these two processes, not the LiDAR.\n'
if wait -n "${M300_OBSTACLE_CHILD_PIDS[@]}"; then
    M300_OBSTACLE_EXIT_CODE=0
else
    M300_OBSTACLE_EXIT_CODE=$?
fi
printf 'A launched process exited (status %s); stopping the other.\n' "$M300_OBSTACLE_EXIT_CODE"
exit "$M300_OBSTACLE_EXIT_CODE"
