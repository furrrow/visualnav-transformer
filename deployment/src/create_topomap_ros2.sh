#!/bin/bash

# ROS2-friendly version of create_topomap.sh with configurable image topic and dt.
# Usage: ./create_topomap_ros2.sh <topomap_name> <bag_filename>
# Env overrides:
#   IMAGE_TOPIC   - topic to sample images from (default: /camera/image_raw)
#   DT            - seconds between saved images (default: 1)
#   PLAYBACK_RATE - ros2 bag playback rate multiplier (default: 1.5)
#   LAUNCH_CMD    - bringup command if needed (default: true / no-op)
#   BAG_DIR       - directory containing bags (default: ../topomaps/bags)

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <topomap_name> <bag_filename>"
  exit 1
fi

TOPOMAP_NAME="$1"
BAG_FILE="$2"
IMAGE_TOPIC="${IMAGE_TOPIC:-/camera/image_raw}"
DT="${DT:-1}"
PLAYBACK_RATE="${PLAYBACK_RATE:-1.5}"
LAUNCH_CMD="${LAUNCH_CMD:-true}"
BAG_DIR="${BAG_DIR:-../topomaps/bags}"

session_name="create_topomap_ros2_$(date +%s)"
tmux new-session -d -s "$session_name"

# Split the window into three panes
tmux selectp -t 0
tmux splitw -v -p 50
tmux selectp -t 0
tmux splitw -h -p 50

# Optional bringup (defaults to no-op)
tmux select-pane -t 0
tmux send-keys "$LAUNCH_CMD" Enter

# Run the topomap creator
tmux select-pane -t 1
tmux send-keys "python create_topomap_ros2.py --dt $DT --dir $TOPOMAP_NAME --image-topic $IMAGE_TOPIC" Enter

# Play back the rosbag2
tmux select-pane -t 2
tmux send-keys "mkdir -p $BAG_DIR" Enter
tmux send-keys "cd $BAG_DIR" Enter
tmux send-keys "ros2 bag play -r $PLAYBACK_RATE $BAG_FILE" Enter

# Attach to the tmux session
tmux -2 attach-session -t "$session_name"
