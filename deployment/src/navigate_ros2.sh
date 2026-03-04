#!/bin/bash

# ROS2-friendly version of navigate.sh.
# Usage: ./navigate_ros2.sh --model <model_name> --dir <topomap_dir> [other navigate args...]
# Env overrides:
#   LAUNCH_CMD     - command to bring up your ROS2 stack (default: ros2 launch realsense2_camera rs_launch.py)
#   NAV_CMD        - navigation command (default: python navigate_ros2.py)
#   TELEOP_CMD     - manual control (default: true, i.e., disabled)
#   CONTROLLER_CMD - converts waypoints to velocity commands (default: true, i.e., disabled)

set -e

LAUNCH_CMD="${LAUNCH_CMD:-ros2 launch realsense2_camera rs_launch.py}"
NAV_CMD="${NAV_CMD:-python navigate_ros2.py}"
TELEOP_CMD="${TELEOP_CMD:-true}"
CONTROLLER_CMD="${CONTROLLER_CMD:-true}"

session_name="vint_locobot_ros2_$(date +%s)"
tmux new-session -d -s "$session_name"

# Split into four panes
tmux selectp -t 0
tmux splitw -h -p 50
tmux selectp -t 0
tmux splitw -v -p 50
tmux selectp -t 2
tmux splitw -v -p 50
tmux selectp -t 0

# Launch bringup
tmux select-pane -t 0
tmux send-keys "$LAUNCH_CMD" Enter

# Run navigation
tmux select-pane -t 1
tmux send-keys "$NAV_CMD $*" Enter

# Optional teleop
tmux select-pane -t 2
tmux send-keys "$TELEOP_CMD" Enter

# Optional controller (waypoint -> cmd_vel)
tmux select-pane -t 3
tmux send-keys "$CONTROLLER_CMD" Enter

tmux -2 attach-session -t "$session_name"
