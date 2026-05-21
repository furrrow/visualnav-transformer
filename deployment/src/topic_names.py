# topic names for ROS communication

# image obs topics
# FRONT_IMAGE_TOPIC = "/usb_cam_front/image_raw"
# REVERSE_IMAGE_TOPIC = "/usb_cam_reverse/image_raw"
# For example, if your robot is named "turtle1", you can set it to "/turtle1"
ROBOT_NAMESPACE = "/ghost"

# Image observation topics
IMAGE_TOPIC = f"{ROBOT_NAMESPACE}/image_compressed"

# exploration topics
WAYPOINT_TOPIC = f"{ROBOT_NAMESPACE}/waypoint"
REACHED_GOAL_TOPIC = f"{ROBOT_NAMESPACE}/topoplan/reached_goal"
SAMPLED_ACTIONS_TOPIC = f"{ROBOT_NAMESPACE}/sampled_actions"


SUBGOALS_TOPIC = "/subgoals"
GRAPH_NAME_TOPIC = "/graph_name"
REVERSE_MODE_TOPIC = "/reverse_mode"
SAMPLED_OUTPUTS_TOPIC = "/sampled_outputs"
SAMPLED_WAYPOINTS_GRAPH_TOPIC = "/sampled_waypoints_graph"
BACKTRACKING_IMAGE_TOPIC = "/backtracking_image"
FRONTIER_IMAGE_TOPIC = "/frontier_image"
SUBGOALS_SHAPE_TOPIC = "/subgoal_shape"
ANNOTATED_IMAGE_TOPIC = "/annotated_image"
CURRENT_NODE_IMAGE_TOPIC = "/current_node_image"
FLIP_DIRECTION_TOPIC = "/flip_direction"
TURNING_TOPIC = "/turning"
SUBGOAL_GEN_RATE_TOPIC = "/subgoal_gen_rate"
MARKER_TOPIC = "/visualization_marker_array"
VIZ_NAV_IMAGE_TOPIC = "/nav_image"
OVERLAY_TOPIC =  "/ghost/path_overlay"

# visualization topics
CHOSEN_SUBGOAL_TOPIC = "/chosen_subgoal"

# recorded ont the robot
ODOM_TOPIC = "/odom"
BUMPER_TOPIC = "/mobile_base/events/bumper"
JOY_BUMPER_TOPIC = "/joy_bumper"

# move the robot
# VEL_TOPIC = f"{ROBOT_NAMESPACE}/cmd_vel"
VEL_TOPIC = f"/mcu/command/manual_twist"
