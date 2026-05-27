#!/bin/bash
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages
export PYTHONPATH=/workspace/prune/policies/visualnav-transformer:$PYTHONPATH

# Create a new tmux session
session_name="visualnav_$(date +%s)"
tmux new-session -d -s $session_name

tmux set-option -g mouse on
source /opt/ros/humble/setup.bash

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves


# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 0
# tmux send-keys "export PYTHONPATH=/workspace/prune/policies/flownav" Enter
# tmux send-keys "source /workspace/prune/setup.bash" Enter
# tmux send-keys "source /opt/ros/humble/setup.bash" Enter
tmux send-keys "uv run deployment/src/navigate.py --dir kitchen_5fl"


# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 1
# tmux send-keys "source /workspace/prune/setup.bash" Enter
tmux send-keys "uv run deployment/src/pd_controller.py"


# Attach to the tmux session
tmux -2 attach-session -t $session_name

