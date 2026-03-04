#!/bin/bash

# ROS2-friendly version of record_bag.sh with configurable camera topic and optional teleop.
# Usage: ./record_bag_ros2.sh <bag_name> [camera_topic] [--no-teleop]
# Env overrides:
#   LAUNCH_CMD  - command to bring up your ROS2 stack (default: ros2 launch realsense2_camera rs_launch.py)
#   TELEOP_CMD  - command for manual control (default: ros2 run teleop_twist_keyboard teleop_twist_keyboard)
#   BAG_DIR     - directory to store bags (default: ../topomaps/bags)

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <bag_name> [camera_topic] [--no-teleop]"
  exit 1
fi

BAG_NAME="$1"
CAMERA_TOPIC="${2:-/camera/image_raw}"
NO_TELEOP=0

if [ "${3:-}" = "--no-teleop" ]; then
  NO_TELEOP=1
fi

LAUNCH_CMD="${LAUNCH_CMD:-ros2 launch realsense2_camera rs_launch.py}"
TELEOP_CMD="${TELEOP_CMD:-ros2 run teleop_twist_keyboard teleop_twist_keyboard}"
BAG_DIR="${BAG_DIR:-../topomaps/bags}"

session_name="record_bag_ros2_$(date +%s)"
tmux new-session -d -s "$session_name"

# Split the window into three panes
tmux selectp -t 0
tmux splitw -v -p 50
tmux selectp -t 0
tmux splitw -h -p 50

# Launch robot stack
tmux select-pane -t 0
tmux send-keys "$LAUNCH_CMD" Enter

# Start teleop unless skipped (override manually via TELEOP_CMD if needed)
tmux select-pane -t 1
if [ "$NO_TELEOP" -eq 0 ]; then
  tmux send-keys "$TELEOP_CMD" Enter
else
  tmux send-keys "echo 'teleop disabled (--no-teleop)'" Enter
fi

# Record bag from chosen camera topic
tmux select-pane -t 2
tmux send-keys "cd $BAG_DIR" Enter
tmux send-keys "ros2 bag record $CAMERA_TOPIC -o $BAG_NAME" Enter

# Attach to the tmux session
tmux -2 attach-session -t "$session_name"
